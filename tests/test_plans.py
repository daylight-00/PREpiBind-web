"""Which TensorRT plans are loaded: a manifest that does not fit this code or these checkpoints, a missing or unreadable one, never reaches TensorRT. No GPU is needed (the plan directory, CUDA and TensorRT are replaced)."""
import copy
import json
from types import SimpleNamespace

import pytest
import trt_engines

HEADS = ["Qualitative", "Mass Spectrometry"]


@pytest.fixture
def checkpoints(tmp_path):
    files = []
    for i, name in enumerate(["esm.pth", "head_a.pth", "head_b.pth"]):
        path = tmp_path / name
        path.write_bytes(bytes([i]) * (100 + i))
        files.append(path)
    return files


def manifest_for(files):
    return {"graph_version": trt_engines.GRAPH_VERSION, "max_batch": trt_engines.MAX_BATCH, "ranges": trt_engines.RANGES, "checkpoints": trt_engines._stamp(dict(zip(["esm", *HEADS], files)))}


def roles(files):
    return dict(zip(["esm", *HEADS], files))


def test_a_manifest_built_for_these_checkpoints_fits(checkpoints):
    assert trt_engines._mismatch(manifest_for(checkpoints), roles(checkpoints)) is None


@pytest.mark.parametrize(
    "change, why",
    [
        (lambda m: m.update(graph_version=m["graph_version"] + 1), "graphs"),
        (lambda m: m.update(max_batch=64), "shape ranges"),
        (lambda m: m["ranges"].update(epitope_length=[8, 48]), "shape ranges"),
        (lambda m: m["checkpoints"]["esm"].update(size=1), "checkpoints"),
        (lambda m: m["checkpoints"].update(esm=m["checkpoints"]["Qualitative"]), "checkpoints"),
    ],
    ids=["graph version", "maximum batch", "ranges", "changed file", "swapped roles"],
)
def test_a_manifest_that_does_not_fit_is_named(checkpoints, change, why):
    manifest = copy.deepcopy(manifest_for(checkpoints))
    change(manifest)
    assert why in trt_engines._mismatch(manifest, roles(checkpoints))


def test_swapping_the_two_heads_checkpoints_is_noticed(checkpoints):
    swapped = {"esm": checkpoints[0], HEADS[0]: checkpoints[2], HEADS[1]: checkpoints[1]}
    assert "checkpoints" in trt_engines._mismatch(manifest_for(checkpoints), swapped)


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    """CUDA and TensorRT present, plans expected in tmp_path/plans, and `Plans` replaced by a stub that records that it was built."""
    directory = tmp_path / "plans"
    directory.mkdir()
    monkeypatch.setattr(trt_engines, "trt", SimpleNamespace(__version__="0"))
    monkeypatch.setattr(trt_engines.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trt_engines, "plan_dir", lambda root: directory)
    built = []
    monkeypatch.setattr(trt_engines, "Plans", lambda d, heads: built.append(d) or "plans")
    return directory, built


def test_matching_plans_are_loaded(fake_runtime, checkpoints):
    directory, built = fake_runtime
    (directory / "manifest.json").write_text(json.dumps(manifest_for(checkpoints)))
    assert trt_engines.load("root", checkpoints, HEADS) == ("plans", None) and built == [directory]


def test_no_manifest_means_no_plans_and_says_how_to_build_them(fake_runtime, checkpoints):
    plans, note = trt_engines.load("root", checkpoints, HEADS)
    assert plans is None and "build_engines.py" in note


@pytest.mark.parametrize("text", ['{"graph_version": 1, "max_ba', "", "not json at all"], ids=["truncated", "empty", "garbage"])
def test_an_unreadable_manifest_means_no_plans_not_a_crash(fake_runtime, checkpoints, text):
    directory, built = fake_runtime
    (directory / "manifest.json").write_text(text)
    plans, note = trt_engines.load("root", checkpoints, HEADS)
    assert plans is None and "cannot be read" in note and not built


def test_a_manifest_of_other_checkpoints_means_no_plans(fake_runtime, checkpoints):
    directory, built = fake_runtime
    (directory / "manifest.json").write_text(json.dumps(manifest_for(checkpoints)))
    checkpoints[1].write_bytes(b"retrained")
    plans, note = trt_engines.load("root", checkpoints, HEADS)
    assert plans is None and "other checkpoints" in note and not built


def test_plans_that_the_runtime_refuses_mean_no_plans_not_a_crash(fake_runtime, checkpoints, monkeypatch):
    directory, _ = fake_runtime
    (directory / "manifest.json").write_text(json.dumps(manifest_for(checkpoints)))

    def refuse(d, heads):
        raise RuntimeError("esm.plan was not accepted by this TensorRT runtime")

    monkeypatch.setattr(trt_engines, "Plans", refuse)
    plans, note = trt_engines.load("root", checkpoints, HEADS)
    assert plans is None and "cannot be loaded" in note


def test_without_tensorrt_there_are_no_plans(monkeypatch, checkpoints):
    monkeypatch.setattr(trt_engines, "trt", None)
    plans, note = trt_engines.load("root", checkpoints, HEADS)
    assert plans is None and "TensorRT is not installed" in note


def test_the_plan_directory_name_carries_tensorrt_gpu_and_graph_version(monkeypatch):
    """So a rebuild for another graph, TensorRT or GPU never overwrites the plans an older release loads."""
    monkeypatch.setattr(trt_engines, "trt", SimpleNamespace(__version__="11.4"))
    monkeypatch.setattr(trt_engines.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(trt_engines.torch.cuda, "get_device_capability", lambda index: (8, 9))
    monkeypatch.setattr(trt_engines.torch.cuda, "get_device_name", lambda index: "NVIDIA GeForce RTX 4070 Ti")
    assert trt_engines.plan_dir("root").name == f"trt11.4-nvidia_geforce_rtx_4070_ti-sm89-g{trt_engines.GRAPH_VERSION}"
