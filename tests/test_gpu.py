"""The real engine of this tree on the GPU (marked `gpu`; skipped without a CUDA GPU, the checkpoints in models/ or matching TensorRT plans). A few minutes of GPU time at most."""
from pathlib import Path

import engine
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def real():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    missing = [f for f in (engine.ESM_CHECKPOINT, *engine.HEAD_CHECKPOINTS.values()) if not (ROOT / "models" / f).exists()]
    if missing or not (ROOT / engine.HLA_DIR).exists():
        pytest.skip(f"model files missing: {missing or engine.HLA_DIR}")
    built = engine.Engine(root=ROOT)
    if built.plans is None:
        pytest.skip(built.plans_note)
    return built


@pytest.fixture(scope="module")
def rows():
    data = pd.read_csv(ROOT / "data" / "test.csv").iloc[:200]
    return pd.DataFrame({"MHC": data.MHC_alpha + "_" + data.MHC_beta, "Epitope": data.Epitope})


def test_fast_and_exact_agree_within_the_noise(real, rows):
    fast, exact = real.run(rows, "Qualitative", mode="fast"), real.run(rows, "Qualitative", mode="exact")
    assert (fast.attrs["engine"], exact.attrs["engine"]) == ("fast", "exact")
    difference = np.abs(fast.Logits.to_numpy(dtype=float) - exact.Logits.to_numpy(dtype=float))
    assert difference.mean() < 0.05
    assert ((fast.Logits > 0) != (exact.Logits > 0)).sum() <= 2  # about 0.1 % of the rows change sign, all near the threshold


def test_a_gpu_out_of_memory_in_the_fast_path_is_served_by_the_exact_path(real, rows, monkeypatch):
    expected = real.run(rows, "Qualitative", mode="exact")

    def out_of_memory(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected)")

    monkeypatch.setattr(real, "_run_fast", out_of_memory)
    result = real.run(rows, "Qualitative")
    assert result.attrs["engine"] == "exact" and (result.Logits.to_numpy() == expected.Logits.to_numpy()).all()


def sequences(rows):
    """The same request with the HLA chains as sequences (Custom Mode's input)."""
    chain = pd.read_csv(ROOT / "data" / "mhc_mapping.csv").set_index("HLA_Name").HLA_Seq
    return pd.DataFrame({"MHC": [chain[a] + "_" + chain[b] for a, b in (m.split("_") for m in rows.MHC)], "Epitope": rows.Epitope})


def test_custom_mode_equals_standard_mode_for_chains_of_the_catalogue(real, rows):
    custom, standard = real.run_custom(sequences(rows), "Qualitative"), real.run(rows, "Qualitative")
    assert (custom["HLA chains"] == "catalogue").all()
    assert (custom.Logits.to_numpy() == standard.Logits.to_numpy()).all()


def mutated(chain, position):
    return chain[:position] + ("A" if chain[position] != "A" else "G") + chain[position + 1:]


def test_a_chain_changed_by_one_residue_is_located_and_scored_close_to_its_allele(real, rows):
    """Not in the catalogue any more, so its domain is located (identity close to 1), cut out of its embedding and scored through the standard path: nearly what Standard Mode gives the allele."""
    frame = sequences(rows.iloc[:60]).reset_index(drop=True)
    standard = real.run(rows.iloc[:60], "Qualitative")
    for k in range(0, 60, 2):
        alpha, beta = frame.MHC[k].split("_")
        frame.loc[k, "MHC"] = mutated(alpha, 60) + "_" + beta
    result = real.run_custom(frame, "Qualitative")
    labels = list(result["HLA chains"])
    assert all(labels[k] == "located" for k in range(0, 60, 2)) and all(labels[k] == "catalogue" for k in range(1, 60, 2)) and result.attrs["engine"] == "fast"
    difference = np.abs(result.Logits.to_numpy(dtype=float) - standard.Logits.to_numpy(dtype=float))
    assert difference.mean() < 0.15 and ((result.Logits > 0) != (standard.Logits > 0)).sum() <= 2


def test_a_domain_pasted_without_its_flanks_gets_them_completed_and_scores_close_to_its_allele(real, rows):
    catalogue = pd.read_csv(ROOT / "data" / "mhc_mapping.csv").set_index("HLA_Name")
    domains = []
    for pair in rows.MHC.iloc[:40]:
        alpha, beta = pair.split("_")
        domains.append("_".join(catalogue.HLA_Seq[a][catalogue.start_idx[a]:catalogue.end_idx[a]] for a in (alpha, beta)))
    result = real.run_custom(pd.DataFrame({"MHC": domains, "Epitope": rows.Epitope.iloc[:40]}), "Qualitative")
    standard = real.run(rows.iloc[:40], "Qualitative")
    assert (result["HLA chains"] == "located").all()
    assert np.abs(result.Logits.to_numpy(dtype=float) - standard.Logits.to_numpy(dtype=float)).mean() < 0.5  # embedded alone the domain would be far off (36.7 % of the signs flip in the prototype)


def test_random_sequences_have_no_domain_and_keep_the_labelled_full_chain_run(real, rows):
    rng = np.random.default_rng(0)
    amino = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    frame = sequences(rows.iloc[:20]).reset_index(drop=True)
    frame.loc[0, "MHC"] = "".join(amino[rng.integers(0, 20, 250)]) + "_" + "".join(amino[rng.integers(0, 20, 250)])
    result = real.run_custom(frame, "Qualitative")
    assert list(result["HLA chains"]) == ["full chain"] + ["catalogue"] * 19 and result.attrs["engine"] == "fast+exact"
    legacy = real._run(frame.iloc[:1], real.custom_models["Qualitative"], real._custom_batch, 128, None)
    assert result.Logits.iloc[0] == legacy.Logits.iloc[0]  # the legacy run, bit for bit


def test_a_window_given_by_the_user_is_used_for_a_chain_with_no_domain_found(real, rows):
    catalogue = pd.read_csv(ROOT / "data" / "mhc_mapping.csv").set_index("HLA_Name")
    alpha, beta = rows.MHC.iloc[0].split("_")
    a, b = catalogue.HLA_Seq[alpha], catalogue.HLA_Seq[beta]
    frame = pd.DataFrame({"MHC": [mutated(a, 60) + "_" + b], "Epitope": [rows.Epitope.iloc[0]]})
    start, end = int(catalogue.start_idx[alpha]), int(catalogue.end_idx[alpha])
    result = real.run_custom(frame, "Qualitative", windows={mutated(a, 60): (start, end)})
    assert list(result["HLA chains"]) == ["manual window"]
    assert abs(float(result.Logits.iloc[0]) - float(real.run(rows.iloc[:1], "Qualitative").Logits.iloc[0])) < 0.3


def test_a_scan_through_the_engine_scores_every_window_against_every_allele(real):
    import scan

    names = set(pd.read_csv(ROOT / "data" / "mhc_mapping.csv").HLA_Name)
    protein = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV"
    pairs = scan.reference_panel(names)
    frame = scan.scan_frame(protein, [15], pairs)
    result = real.run(frame, "Qualitative")
    assert len(result) == (len(protein) - 14) * 27 and np.isfinite(result.Logits.to_numpy(dtype=float)).all() and result.attrs["engine"] == "fast"
    window = scan.scan_frame(protein[9:24], [15], pairs)  # the window that starts at residue 10, scored on its own against the same alleles
    alone = real.run(window, "Qualitative")
    inside = result[result.Start == 10].Logits.to_numpy(dtype=float)
    assert np.abs(inside - alone.Logits.to_numpy(dtype=float)).max() < 0.2  # the fast path is not bitwise across request sizes


def test_a_refused_request_leaves_the_engine_usable(real, rows):
    long = rows.copy()
    long.loc[long.index[0], "Epitope"] = "A" * (engine.MAX_EPITOPE_LENGTH + 1)
    with pytest.raises(engine.RequestRefused):
        real.run(long, "Qualitative")
    assert len(real.run(rows, "Qualitative")) == len(rows)


def test_the_memory_of_a_fast_request_does_not_grow_with_its_size(real, rows):
    """Finding 1 of the review: the embeddings of all distinct epitopes used to be held at once (150,000 distinct 32-mers filled the card)."""
    rng = np.random.default_rng(0)
    amino = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    n = 60000
    big = pd.DataFrame({"MHC": [rows.MHC.iloc[0]] * n, "Epitope": ["".join(r) for r in amino[rng.integers(0, 20, (n, 20))]]})
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    result = real.run(big, "Qualitative", mode="fast")
    assert len(result) == n and np.isfinite(result.Logits.to_numpy(dtype=float)).all()
    assert torch.cuda.max_memory_allocated() - base < 1.6 * 2**30  # one block of epitope embeddings (at most 1 GiB) and the joint batches
