"""Shared setup. Two tiers: the default tests need no GPU, no checkpoints and no plans (the engine is built bare and its two paths are replaced); tests marked `gpu` run the real engine of this tree and
are skipped where there is no CUDA GPU, no checkpoint or no TensorRT plan. Run:  uv run --extra trt --group test pytest   (add `-m "not gpu"` to skip the GPU tier)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))  # security_config.py


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: runs the real engine of this tree (CUDA GPU, model checkpoints and TensorRT plans); skipped when any of them is missing")


@pytest.fixture
def bare_engine():
    """An Engine without models or a GPU: enough to test how run() routes a request when `_run_fast` and `_run` are replaced by the test."""
    import engine

    bare = engine.Engine.__new__(engine.Engine)
    bare.mode, bare.plans, bare.plans_note = "auto", object(), ""
    bare.models, bare.custom_models = {"Qualitative": object()}, {"Qualitative": object()}
    bare.hla = object()  # a stand-in for the HLA store: run() passes it to the two paths unless the caller gives its own
    return bare


@pytest.fixture
def frame():
    """A request of three rows: a pair of allele names and a 15-mer each."""
    import pandas as pd

    return pd.DataFrame({"MHC": ["HLA-DRA*01:01_HLA-DRB1*15:01"] * 3, "Epitope": ["GELIGILNAAKVPAD", "PKYVKQNTLKLATAA", "LKLREVYTQLCDHRL"]})
