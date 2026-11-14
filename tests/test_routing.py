"""How Engine.run chooses between the fast and the exact path and what it does when the fast path fails. The two paths are replaced by the test; no GPU is needed."""
import pytest
import torch

import engine
import trt_engines


class Spy:
    """Stands in for a path: records its calls and returns, or raises, what the test says."""

    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def wire(bare, fast, exact):
    bare._run_fast, bare._run = fast, exact
    return bare


def test_a_mode_that_is_not_auto_fast_or_exact_is_an_error_not_a_silent_fast_run(bare_engine, frame):
    fast = Spy("fast")
    with pytest.raises(ValueError, match="mode must be"):
        wire(bare_engine, fast, Spy("exact")).run(frame, "Qualitative", mode="exat")
    assert fast.calls == 0


def test_auto_serves_the_fast_path_when_it_can(bare_engine, frame):
    fast, exact = Spy("fast"), Spy("exact")
    assert wire(bare_engine, fast, exact).run(frame, "Qualitative") == "fast"
    assert (fast.calls, exact.calls) == (1, 0)


def test_exact_mode_never_tries_the_fast_path(bare_engine, frame):
    fast, exact = Spy("fast"), Spy("exact")
    assert wire(bare_engine, fast, exact).run(frame, "Qualitative", mode="exact") == "exact"
    assert (fast.calls, exact.calls) == (0, 1)


def test_auto_without_plans_serves_the_exact_path(bare_engine, frame):
    bare_engine.plans, bare_engine.plans_note = None, "no plans"
    fast, exact = Spy("fast"), Spy("exact")
    assert wire(bare_engine, fast, exact).run(frame, "Qualitative") == "exact"
    assert fast.calls == 0


def test_auto_serves_the_exact_path_when_the_fast_path_declines_the_request(bare_engine, frame):
    fast, exact = Spy(error=engine.FastUnavailable("epitope lengths 8-40 are outside the fast engine's 8-32")), Spy("exact")
    assert wire(bare_engine, fast, exact).run(frame, "Qualitative") == "exact"
    assert bare_engine.plans is not None  # declining one request does not disable the plans


def test_auto_falls_back_on_gpu_out_of_memory_for_that_request_only(bare_engine, frame, capsys):
    fast, exact = Spy(error=torch.cuda.OutOfMemoryError("CUDA out of memory")), Spy("exact")
    assert wire(bare_engine, fast, exact).run(frame, "Qualitative") == "exact"
    assert "ran out of GPU memory" in capsys.readouterr().out
    assert bare_engine.plans is not None
    wire(bare_engine, Spy("fast"), exact)
    assert bare_engine.run(frame, "Qualitative") == "fast"  # the next request tries the fast path again


def test_a_tensorrt_failure_while_running_switches_to_the_exact_path_for_good(bare_engine, frame, capsys):
    fast, exact = Spy(error=trt_engines.TrtError("TensorRT execution failed")), Spy("exact")
    wire(bare_engine, fast, exact)
    assert bare_engine.run(frame, "Qualitative") == "exact"
    assert bare_engine.plans is None and "TensorRT failed while running" in capsys.readouterr().out
    assert bare_engine.run(frame, "Qualitative") == "exact"
    assert fast.calls == 1  # not tried again


@pytest.mark.parametrize("error", [engine.FastUnavailable("no plans"), torch.cuda.OutOfMemoryError("CUDA out of memory"), trt_engines.TrtError("boom")], ids=["unavailable", "out-of-memory", "tensorrt"])
def test_fast_mode_never_falls_back(bare_engine, frame, error):
    exact = Spy("exact")
    with pytest.raises(type(error)):
        wire(bare_engine, Spy(error=error), exact).run(frame, "Qualitative", mode="fast")
    assert exact.calls == 0
