"""Build the TensorRT plans of the fast engine for this TensorRT version and GPU (about 4.5 minutes for the four heads, once; see trt_engines.py).

    python code/build_engines.py [--root .]

Needs the extra (`uv sync --extra trt`: tensorrt, onnx, onnxscript) on Python 3.14: NVIDIA publishes no TensorRT wheel for Python 3.15 yet.
Run it again after a TensorRT, driver or GPU change or new checkpoints; until then the server keeps serving the exact path.
"""
import argparse
import sys
from pathlib import Path

import trt_engines
from engine import ESM_CHECKPOINT, HEAD_CHECKPOINTS, Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="the repository root, where models/ and data/ are (default: the current directory)")
    root = Path(parser.parse_args().root).resolve()
    if trt_engines.trt is None:
        sys.exit("TensorRT is not installed: `uv sync --extra trt` on Python 3.14 (NVIDIA publishes no TensorRT wheel for Python 3.15 yet)")
    engine = Engine(root=root, mode="exact")  # the eager modules only: nothing to load yet
    files = [root / "models" / ESM_CHECKPOINT, *(root / "models" / file for file in HEAD_CHECKPOINTS.values())]
    trt_engines.build(engine.embedder, {name: model.model for name, model in engine.models.items()}, files, root)


if __name__ == "__main__":
    main()
