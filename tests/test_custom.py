"""Custom Mode: a pair of chains that are both in the allele catalogue is scored exactly as Standard Mode scores the two alleles; a pair with any other chain keeps the legacy full-chain run, labelled.
The two paths are replaced by fakes that record what they were given; no GPU is needed."""
import numpy as np
import pandas as pd
import pytest

import engine

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

    def standard(df, head, batch_size=128, progress=None, mode=None):
        bare_engine.calls["standard"] = df
        return finish(df, 10, "fast", progress)

    def legacy(df, model, make_batch, batch_size, progress):
        bare_engine.calls["legacy"] = df
        return finish(df, -10, "exact", progress)

    bare_engine.run, bare_engine._run = standard, legacy
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
