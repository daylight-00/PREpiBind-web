# Tests

`tests/` has two tiers. The default tier needs no GPU, no checkpoints and no plans (the hard limits, how the engine routes a request and falls back, the staleness checks of the TensorRT plans, the security headers and robots.txt; about two seconds). The tier marked `gpu` runs the real engine of the tree (fast against exact, an injected out-of-memory, a refused request, the memory of a large request) and skips itself without a CUDA GPU, the checkpoints or matching plans. pytest is in the `test` dependency group, which a plain `uv sync` does not install:

```bash
uv run --python 3.14 --extra trt --group test pytest               # both tiers
uv run --python 3.14 --extra trt --group test pytest -m "not gpu"  # no GPU needed
```
