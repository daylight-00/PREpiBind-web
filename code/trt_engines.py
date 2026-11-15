"""TensorRT plans of the fast engine: build (export and compile), load, run.

The model is cut where the cache path of `Engine._run_fast` cuts it, and the cuts get dynamic shapes:
    esm    epitope tokens (B, S) -> ESMC embeddings (B, S, 960); one engine, shared by the four heads (identical weights)
    hla    (B, L, 960) and its mask -> the head's two-block HLA encoder; a function of the HLA pair alone
    joint  HLA encoding, epitope embeddings and their masks -> logit (B, 1): epitope encoder, joint block, pooling, output layer
All fp16 (TensorRT 11 networks are strongly typed: the dtypes of the exported graph decide the precision). Plans are files under models/engines/<tensorrt>-<gpu>/
with a manifest of the checkpoints they were built from, because a plan is valid for one TensorRT version on one GPU model and one set of weights; they are built
by `python code/build_engines.py` (needs the `trt` extra: onnx, onnxscript) and loaded at start-up when they match.
All execution contexts share ONE scratch buffer (a context otherwise reserves activation memory for the largest shape of its profile: 3 GB per head at batch 512),
which is only safe while one engine runs at a time: the caller serialises (Engine holds a lock).
"""
import json
import re
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

try:
    import tensorrt as trt
except ImportError:  # the extra is not installed, or NVIDIA publishes no wheel for this Python (none for 3.15 as of TensorRT 11.4)
    trt = None

MAX_BATCH = 128
TOKENS = (10, 34)    # epitope length 8-32 plus [CLS] and [SEP]
HLA_LEN = (64, 200)  # a pair of catalogue chains is at most 156
EPI_LEN = (8, 32)
OPT = {"tokens": 17, "hla": 156, "epi": 15}  # the shapes the kernels are tuned for: 15-mers and the longest catalogue pair


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


GRAPH_VERSION = 1  # bump when the exported graphs (the wrappers below) change: plans of another version are neither loaded nor overwritten
RANGES = {"tokens": list(TOKENS), "hla_length": list(HLA_LEN), "epitope_length": list(EPI_LEN)}


def plan_dir(root):
    """Plans live per TensorRT version, GPU model and graph version, so that building for a newer graph or TensorRT never overwrites what an older release loads."""
    index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return Path(root) / "models" / "engines" / f"trt{trt.__version__}-{slug(torch.cuda.get_device_name(index))}-sm{major}{minor}-g{GRAPH_VERSION}"


def _stamp(roles):
    """What each plan was built from, by role ({'esm': path, '<head>': path}): a swapped checkpoint or head mapping is noticed, not only a changed file."""
    return {role: {"file": Path(f).name, "size": Path(f).stat().st_size, "mtime_ns": Path(f).stat().st_mtime_ns} for role, f in roles.items()}


def _mismatch(manifest, roles):
    """Why a manifest does not fit this code and these checkpoints, or None."""
    if manifest.get("graph_version") != GRAPH_VERSION:
        return "were built from other graphs"
    if manifest.get("max_batch") != MAX_BATCH or manifest.get("ranges") != RANGES:
        return "were built for other shape ranges"
    if manifest.get("checkpoints") != _stamp(roles):
        return "were built from other checkpoints"
    return None


def load(root, files, heads):
    """(Plans, None) when plans for this TensorRT, GPU and graph version exist and were built from `files` (the ESMC checkpoint, then one per head in the order of `heads`); else (None, the reason)."""
    if trt is None:
        return None, "TensorRT is not installed (extra 'trt'; NVIDIA publishes no TensorRT wheel for Python 3.15 yet)"
    if not torch.cuda.is_available():
        return None, "no GPU"
    directory = plan_dir(root)
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except FileNotFoundError:
        return None, f"no plans for this TensorRT, GPU and graph version: run `python code/build_engines.py` (expected in {directory})"
    except (OSError, ValueError) as error:
        return None, f"the manifest of the plans in {directory} cannot be read ({error}): rebuild them with `python code/build_engines.py`"
    why = _mismatch(manifest, dict(zip(["esm", *heads], files)))
    if why:
        return None, f"the plans in {directory} {why}: rebuild them with `python code/build_engines.py`"
    try:
        return Plans(directory, heads), None
    except Exception as error:  # noqa: BLE001 - a plan from another driver or a damaged file, whatever the runtime raises: serve the exact path, say why
        return None, f"the plans in {directory} cannot be loaded ({type(error).__name__}: {error}): rebuild them with `python code/build_engines.py`"


class TrtError(RuntimeError):
    """TensorRT refused or failed an execution (an input outside the plan's profile or of another type, a driver or runtime that no longer matches the plans, a context on another device)."""


_DTYPE = {"HALF": torch.float16, "FLOAT": torch.float32, "INT32": torch.int32, "INT64": torch.int64, "BOOL": torch.bool}


class _Plan:
    """One deserialized engine and its execution context; inputs are bound by address (no copy) and the output is allocated per call."""

    def __init__(self, runtime, path):
        self.engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"{Path(path).name} was not accepted by this TensorRT runtime")
        self.context = self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output = next(n for n in names if self.engine.get_tensor_mode(n) != trt.TensorIOMode.INPUT)
        self.dtypes = {n: _DTYPE[self.engine.get_tensor_dtype(n).name] for n in names}
        self.scratch = self.engine.get_device_memory_size_for_profile_v2(0)

    def __call__(self, *tensors):
        if len(tensors) != len(self.inputs):
            raise TrtError(f"{len(tensors)} tensors for {len(self.inputs)} inputs")
        for name, tensor in zip(self.inputs, tensors):
            if tensor.dtype != self.dtypes[name]:
                raise TrtError(f"input {name} is {tensor.dtype}, the plan takes {self.dtypes[name]}")
            if not (self.context.set_input_shape(name, tuple(tensor.shape)) and self.context.set_tensor_address(name, tensor.data_ptr())):
                raise TrtError(f"input {name} of shape {tuple(tensor.shape)} is outside the plan's profile")
        out = torch.empty(tuple(self.context.get_tensor_shape(self.output)), dtype=self.dtypes[self.output], device="cuda")
        if not self.context.set_tensor_address(self.output, out.data_ptr()) or not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise TrtError("TensorRT execution failed")
        return out


class Plans:
    """The engines of one plan directory on one shared scratch buffer. Inputs must be contiguous and in the profile's ranges."""

    def __init__(self, directory, heads):
        directory = Path(directory)
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self._esm = _Plan(self._runtime, directory / "esm.plan")
        self._hla = {head: _Plan(self._runtime, directory / f"hla_{slug(head)}.plan") for head in heads}
        self._joint = {head: _Plan(self._runtime, directory / f"joint_{slug(head)}.plan") for head in heads}
        plans = [self._esm, *self._hla.values(), *self._joint.values()]
        size = max(plan.scratch for plan in plans)
        self._scratch = torch.empty(size, dtype=torch.uint8, device="cuda")
        for plan in plans:
            plan.context.set_device_memory(self._scratch.data_ptr(), size)

    def esm(self, tokens):
        return self._esm(tokens)

    def hla(self, head, x, mask):
        return self._hla[head](x, mask)

    def joint(self, head, hla, epitope, mask_hla, mask_epitope):
        return self._joint[head](hla, epitope, mask_hla, mask_epitope)


# ---- build -------------------------------------------------------------------------------------------------------------------------------------------------

class _Esm(nn.Module):
    """EsmcModel.forward for tokens without padding (every token real, no attention mask) minus its data-dependent test, which cannot be traced: the same arithmetic."""

    def __init__(self, embedder):
        super().__init__()
        self.encoder = embedder.encoder

    def forward(self, tokens):
        x = self.encoder.embed(tokens)
        return self.encoder.transformer(x, sequence_id=None, layers_to_collect=[], output_attentions=False)[0]


class _Hla(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x, mask):
        return self.head.hla_self_attn(x, padding_mask=mask)


class _Joint(nn.Module):
    """plm_cat_mean_inf.forward from the HLA encoding on."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, hla, epitope, mask_hla, mask_epitope):
        epitope = self.head.epi_self_attn(epitope, padding_mask=None)
        mask = torch.cat((mask_hla, mask_epitope), dim=-1)
        x = self.head.self_attn(torch.cat((hla, epitope), dim=1), padding_mask=mask)
        keep = (~mask).unsqueeze(-1)
        x = (x * keep).sum(dim=1) / keep.sum(dim=1)
        return self.head.output_layer(x)


def _compile(onnx_path, profile, plan_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):  # the file parser: the exporter keeps the weights in a side file
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    shapes = builder.create_optimization_profile()
    for name, (low, opt, high) in profile.items():
        shapes.set_shape(name, low, opt, high)
    config.add_optimization_profile(shapes)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"TensorRT could not build {Path(plan_path).name}")
    Path(plan_path).write_bytes(bytes(blob))


def build(embedder, heads, files, root, log=print):
    """Export the three kinds of graph from the eager modules (`embedder`: the Embedder; `heads`: {name: plm_cat_mean_inf}, fp16 on the GPU) and compile them.
    `files` are the checkpoints the modules were loaded from (the ESMC checkpoint, then one per head in the order of `heads`); the manifest written last records them."""
    from torch.export import Dim

    directory = plan_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").unlink(missing_ok=True)  # a directory without a manifest is not loaded
    device = next(embedder.parameters()).device
    batch, tokens = Dim("batch", min=1, max=MAX_BATCH), Dim("seq", min=TOKENS[0], max=TOKENS[1])
    hla_len, epi_len = Dim("hla_len", min=HLA_LEN[0], max=HLA_LEN[1]), Dim("epi_len", min=EPI_LEN[0], max=EPI_LEN[1])
    with torch.inference_mode():  # the rotary cache is built lazily: fill it for the widest input before tracing
        embedder(torch.randint(4, 24, (2, TOKENS[1]), device=device))

    def export_and_compile(module, example, names, outputs, dynamic, profile, plan, tmp):
        started = time.time()
        onnx_path = Path(tmp) / f"{Path(plan).stem}.onnx"
        torch.onnx.export(module.eval(), example, str(onnx_path), input_names=names, output_names=outputs, opset_version=20, dynamo=True, dynamic_shapes=dynamic)
        _compile(onnx_path, profile, directory / plan)
        log(f"{plan}: {time.time() - started:.0f} s, {(directory / plan).stat().st_size / 2**20:.0f} MiB")

    started = time.time()
    x = torch.randn(4, OPT["hla"], 960, device=device, dtype=torch.float16)
    mask_hla = torch.zeros(4, OPT["hla"], dtype=torch.bool, device=device)
    epitope = torch.randn(4, OPT["epi"], 960, device=device, dtype=torch.float16)
    mask_epitope = torch.zeros(4, OPT["epi"], dtype=torch.bool, device=device)
    lo_hla, hi_hla = HLA_LEN
    lo_epi, hi_epi = EPI_LEN
    with tempfile.TemporaryDirectory(prefix="onnx-", dir=directory) as tmp:
        export_and_compile(_Esm(embedder), (torch.randint(4, 24, (4, OPT["tokens"]), device=device),), ["tokens"], ["embeddings"], ({0: batch, 1: tokens},),
                           {"tokens": ((1, TOKENS[0]), (MAX_BATCH, OPT["tokens"]), (MAX_BATCH, TOKENS[1]))}, "esm.plan", tmp)
        for name, head in heads.items():
            stem = slug(name)
            export_and_compile(_Hla(head), (x, mask_hla), ["x", "mask"], ["encoding"], ({0: batch, 1: hla_len}, {0: batch, 1: hla_len}),
                               {"x": ((1, lo_hla, 960), (MAX_BATCH, OPT["hla"], 960), (MAX_BATCH, hi_hla, 960)), "mask": ((1, lo_hla), (MAX_BATCH, OPT["hla"]), (MAX_BATCH, hi_hla))}, f"hla_{stem}.plan", tmp)
            export_and_compile(_Joint(head), (x, epitope, mask_hla, mask_epitope), ["hla", "epitope", "mask_hla", "mask_epitope"], ["logit"],
                               ({0: batch, 1: hla_len}, {0: batch, 1: epi_len}, {0: batch, 1: hla_len}, {0: batch, 1: epi_len}),
                               {"hla": ((1, lo_hla, 960), (MAX_BATCH, OPT["hla"], 960), (MAX_BATCH, hi_hla, 960)), "epitope": ((1, lo_epi, 960), (MAX_BATCH, OPT["epi"], 960), (MAX_BATCH, hi_epi, 960)),
                                "mask_hla": ((1, lo_hla), (MAX_BATCH, OPT["hla"]), (MAX_BATCH, hi_hla)), "mask_epitope": ((1, lo_epi), (MAX_BATCH, OPT["epi"]), (MAX_BATCH, hi_epi))}, f"joint_{stem}.plan", tmp)
    index = torch.cuda.current_device()
    manifest = {"graph_version": GRAPH_VERSION, "tensorrt": trt.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(index), "compute_capability": list(torch.cuda.get_device_capability(index)),
                "max_batch": MAX_BATCH, "ranges": RANGES, "checkpoints": _stamp(dict(zip(["esm", *heads], files))),
                "built": time.strftime("%Y-%m-%d %H:%M:%S"), "build_seconds": round(time.time() - started),
                "plans": {p.name: p.stat().st_size for p in sorted(directory.glob("*.plan"))}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"plans in {directory} ({manifest['build_seconds']} s)")
    return directory
