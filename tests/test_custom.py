"""Custom Mode: a pair of chains that are both in the allele catalogue is scored exactly as Standard Mode scores the two alleles; a pair with any other chain keeps the legacy full-chain run, labelled.
The two paths are replaced by fakes that record what they were given; no GPU is needed."""
import engine
import numpy as np
import pandas as pd
import pytest

CATALOGUE = {"AAAAAA": "HLA-DRA*01:01", "BBBBBB": "HLA-DRB1*01:01", "CCCCCC": "HLA-DRB1*15:01"}


@pytest.fixture
def custom(bare_engine):
    """An engine whose Standard path (`run`) returns 10, 11, ... and whose legacy full-chain path (`_run`) returns -10, -9, ... in the order of the rows it is given, and which records both calls."""
    bare_engine.chain_names, bare_engine.calls = dict(CATALOGUE), {}

    def finish(df, offset, path, progress):
        out = df.copy()
        out["Logits"] = (np.arange(len(df)) + offset).astype(np.float16)
        out["Score"] = engine.score(out["Logits"])
        out.attrs["engine"] = path
        if progress:
            progress(1, 2), progress(2, 2)
        return out

    def standard(df, head, batch_size=128, progress=None, mode=None, store=None):
        bare_engine.calls["standard"], bare_engine.calls["store"] = df, store
        return finish(df, 10, "fast", progress)

    def legacy(df, model, make_batch, batch_size, progress):
        bare_engine.calls["legacy"] = df
        return finish(df, -10, "exact", progress)

    bare_engine.run, bare_engine._run = standard, legacy
    bare_engine.chain_report = lambda chain: {"tier": "no domain", "nearest": None, "identity": 0.0, "start": None, "end": None, "hit": None}  # the locator finds nothing: such a chain keeps the legacy run
    return bare_engine


def request(*pairs):
    return pd.DataFrame({"MHC": [f"{a}_{b}" for a, b in pairs], "Epitope": ["GELIGILNAAKVPAD"] * len(pairs)})


def test_the_catalogue_map_gives_the_first_allele_of_a_sequence():
    catalogue = pd.DataFrame({"HLA_Name": ["HLA-A", "HLA-B", "HLA-C"], "HLA_Seq": ["AAAA", "BBBB", "AAAA"]})
    assert engine.chain_name_map(catalogue) == {"AAAA": "HLA-A", "BBBB": "HLA-B"}


def test_two_catalogue_chains_go_through_standard_mode_by_allele_name(custom):
    out = custom.run_custom(request(("AAAAAA", "BBBBBB"), ("AAAAAA", "CCCCCC")), "Qualitative")
    assert list(custom.calls["standard"].MHC) == ["HLA-DRA*01:01_HLA-DRB1*01:01", "HLA-DRA*01:01_HLA-DRB1*15:01"] and "legacy" not in custom.calls
    assert list(out["HLA chains"]) == ["catalogue"] * 2 and out.attrs["engine"] == "fast"


def test_a_pair_with_a_chain_outside_the_catalogue_keeps_the_legacy_run_on_the_sequences(custom):
    out = custom.run_custom(request(("AAAAAA", "XXXXXX")), "Qualitative")
    assert list(custom.calls["legacy"].MHC) == ["AAAAAA_XXXXXX"] and "standard" not in custom.calls
    assert list(out["HLA chains"]) == ["full chain"] and out.attrs["engine"] == "exact"


def test_one_known_and_one_unknown_chain_make_a_full_chain_pair(custom):
    assert list(custom.run_custom(request(("XXXXXX", "BBBBBB")), "Qualitative")["HLA chains"]) == ["full chain"]


def test_a_mixed_request_keeps_the_order_and_takes_each_row_from_its_own_path(custom):
    out = custom.run_custom(request(("AAAAAA", "BBBBBB"), ("XXXXXX", "BBBBBB"), ("AAAAAA", "CCCCCC"), ("AAAAAA", "YYYYYY")), "Qualitative")
    assert list(out["HLA chains"]) == ["catalogue", "full chain", "catalogue", "full chain"]
    assert list(out.Logits.astype(float)) == [10, -10, 11, -9]  # the Standard path gave 10 and 11 to its two rows, the legacy path -10 and -9 to its two
    assert list(out.MHC) == ["AAAAAA_BBBBBB", "XXXXXX_BBBBBB", "AAAAAA_CCCCCC", "AAAAAA_YYYYYY"]  # the request's own columns are untouched
    assert out.attrs["engine"] == "fast+exact"
    assert np.allclose(out.Score, engine.score(out.Logits.to_numpy()))


def test_case_and_surrounding_spaces_do_not_hide_a_catalogue_chain(custom):
    out = custom.run_custom(request((" aaaaaa ", "bbbbbb")), "Qualitative")
    assert list(out["HLA chains"]) == ["catalogue"] and list(custom.calls["standard"].MHC) == ["HLA-DRA*01:01_HLA-DRB1*01:01"]


def test_progress_runs_forward_and_ends_at_the_number_of_rows(custom):
    ticks = []
    custom.run_custom(request(("AAAAAA", "BBBBBB"), ("XXXXXX", "BBBBBB"), ("AAAAAA", "CCCCCC")), "Qualitative", progress=lambda done, total: ticks.append((done, total)))
    assert ticks and all(total == 3 for _, total in ticks) and [d for d, _ in ticks] == sorted(d for d, _ in ticks) and ticks[-1][0] == 3


def test_rows_without_an_epitope_are_dropped_as_in_standard_mode(custom):
    frame = request(("AAAAAA", "BBBBBB"), ("AAAAAA", "CCCCCC"))
    frame.loc[0, "Epitope"] = None
    assert len(custom.run_custom(frame, "Qualitative")) == 1


# ---- chains outside the catalogue whose domain is located -------------------------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def located(custom):
    """The locator finds a domain for chains made of X (identity 0.9, window 10-40) and a weak one for chains made of W (identity 0.45, window 5-35); the embedding of a domain is a tensor of ones."""
    import torch

    def report(chain):
        if set(chain) == {"X"}:
            return {"tier": "located", "nearest": "HLA-DRB1*01:01", "identity": 0.9, "start": 10, "end": 40, "hit": None}
        if set(chain) == {"W"}:
            return {"tier": "low confidence", "nearest": "HLA-DRB1*01:01", "identity": 0.45, "start": 5, "end": 35, "hit": None}
        return {"tier": "no domain", "nearest": None, "identity": 0.0, "start": None, "end": None, "hit": None}

    custom.chain_report = report
    custom._domain_embedding = lambda chain, start, end, hit: torch.ones(end - start, 960, dtype=torch.float16)
    return custom


def test_a_located_chain_is_embedded_and_scored_through_the_standard_path_with_its_own_store(located):
    out = located.run_custom(request(("AAAAAA", "X" * 60)), "Qualitative")
    named = located.calls["standard"].MHC.iloc[0]
    alpha, beta = named.split("_")
    assert alpha == "HLA-DRA*01:01" and beta.startswith("custom:") and "legacy" not in located.calls
    store = located.calls["store"]
    assert isinstance(store, engine.OverlayStore) and store.chain(beta).shape == (30, 960)  # the window 10-40 of the chain
    assert list(out["HLA chains"]) == ["located"] and out.attrs["engine"] == "fast"


def test_the_overlay_looks_in_its_own_embeddings_first_and_then_in_the_shared_store():
    class Base:
        def chain(self, name):
            return f"base:{name}"

    overlay = engine.OverlayStore(Base(), {"custom:1": "own"})
    assert overlay.chain("custom:1") == "own" and overlay.chain("HLA-A") == "base:HLA-A"


def test_the_label_of_a_row_is_the_worst_tier_of_its_two_chains(located):
    out = located.run_custom(request(("AAAAAA", "X" * 60), ("X" * 60, "W" * 60), ("W" * 60, "AAAAAA"), ("AAAAAA", "BBBBBB"), ("X" * 60, "Y" * 60)), "Qualitative")
    assert list(out["HLA chains"]) == ["located", "low confidence", "low confidence", "catalogue", "full chain"]


def test_a_chain_with_no_domain_found_sends_its_whole_row_to_the_legacy_run_on_the_sequences(located):
    located.run_custom(request(("X" * 60, "Y" * 60)), "Qualitative")
    assert list(located.calls["legacy"].MHC) == ["X" * 60 + "_" + "Y" * 60] and "standard" not in located.calls


def test_a_window_the_user_gives_replaces_the_located_one_and_rescues_a_chain_with_no_domain(located):
    seen = {}
    located._domain_embedding = lambda chain, start, end, hit: seen.setdefault((chain, start, end, hit), __import__("torch").ones(end - start, 960, dtype=__import__("torch").float16))
    out = located.run_custom(request(("Y" * 80, "AAAAAA"), ("X" * 60, "AAAAAA")), "Qualitative", windows={"y" * 80: (7, 67), "X" * 60: (20, 50)})
    assert list(out["HLA chains"]) == ["manual window", "manual window"]
    assert {(chain[0], start, end) for chain, start, end, _ in seen} == {("Y", 7, 67), ("X", 20, 50)}  # 0-based, half-open; the keys of `windows` are normalised like the chains
    assert all(hit is None for *_, hit in seen)  # a window the user gave is cut as it is: no flanks are added


@pytest.mark.parametrize("window", [(0, 10), (5, 90), (-1, 40), (40, 40)], ids=["shorter than 20", "past the end of the chain", "before the start", "empty"])
def test_a_window_that_is_not_inside_the_chain_or_too_short_is_refused(located, window):
    with pytest.raises(engine.RequestRefused, match="domain window"):
        located.run_custom(request(("Y" * 80, "AAAAAA")), "Qualitative", windows={"Y" * 80: window})


def test_the_same_domain_is_embedded_once_for_all_the_rows_that_use_it(located):
    calls = []
    located._domain_embedding = lambda chain, start, end, hit: calls.append(chain) or __import__("torch").ones(end - start, 960, dtype=__import__("torch").float16)
    located.run_custom(request(("AAAAAA", "X" * 60), ("BBBBBB", "X" * 60), ("CCCCCC", "X" * 60)), "Qualitative")
    assert calls == ["X" * 60]


def test_the_chain_report_of_a_catalogue_chain_is_its_catalogue_window(bare_engine):
    bare_engine.chain_names, bare_engine.domain_windows = dict(CATALOGUE), {"HLA-DRB1*01:01": (33, 109)}
    report = bare_engine.chain_report(" bbbbbb ")
    assert report["tier"] == "catalogue" and report["nearest"] == "HLA-DRB1*01:01" and (report["start"], report["end"]) == (33, 109) and report["identity"] == 1.0
