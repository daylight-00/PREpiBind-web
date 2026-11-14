"""The real engine of this tree on the GPU (marked `gpu`; skipped without a CUDA GPU, the checkpoints in models/ or matching TensorRT plans). A few minutes of GPU time at most."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import engine

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
