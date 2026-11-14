"""Inference engine of the img1 web server: the four PREpiBind heads behind one ESM C 300M, as plain functions of a DataFrame.

It replaces what the baseline spread over app.py (model loading), inference.py (DataLoader, Streamlit progress bar, sigmoid) and encoder.py / dataprovider.py (HLA lookup, tokenising):
no Streamlit import, no DataLoader workers, no configuration file executed as code, no hard-coded GPU index (select the GPU with CUDA_VISIBLE_DEVICES). The arithmetic is the baseline's:
the same batches (consecutive `batch_size` rows), the same collate functions, the same fp16 modules; the one change in structure is a single ESM C shared by the four heads instead of
four copies of the same weights.

Two ways to run a request. The EXACT path is the baseline's arithmetic in eager mode (bitwise equal to the baseline code in the same environment). The FAST path
(`_run_fast`) computes every distinct epitope through ESM C once and every distinct HLA pair through the head's HLA encoder once, runs the joint blocks per batch of
rows of one epitope length, and executes the three kinds of graph as TensorRT fp16 plans (trt_engines.py): most rows differ from the exact path in the last bits (mean
0.01-0.02 logits), about 0.1 % of the rows change sign (almost all within 0.15 logits of the threshold), 4.5x faster on bulk uploads and in less GPU memory.
`PREPIBIND_ENGINE` picks: `auto` (default) = fast when plans for this TensorRT and GPU are loaded and the request is in the plans' ranges, else exact; `fast` = fast or
an error; `exact` = never load the plans. Custom mode is always exact.
"""
import os
import threading
from pathlib import Path

import numpy as np
import torch
from esm.tokenization import EsmSequenceTokenizer
from esm.utils import encoding

import trt_engines
from collate import pad_and_mask_collate_fn_inf, pad_and_mask_collate_fn_inf_custom_hla
from embedder import load_embedder
from model import UnifiedModel, UnifiedModel_custom_hla, plm_cat_mean_inf

ESM_CHECKPOINT = "esmc_300m_2024_12_v0_fp16.pth"
HLA_DIR = "data/emb_hla_esmc_small_0601_fp16"
HEAD_CHECKPOINTS = {
    "Qualitative": "prepi_esmc_small_e5_s128_f4_fp16.pth",
    "Mass Spectrometry": "prepi_esmc_small_ms_e5_s100_f0_fp16.pth",
    "IC50 (<500nM)": "prepi_esmc_small_ic50_500_e5_s128_f4_fp16.pth",
    "IC50 (<1000nM)": "prepi_esmc_small_ic50_1000_e5_s128_f1_fp16.pth",
}
HEAD_ARGS = {"hla_dim": 960, "epi_dim": 960, "head_div": 64}

_tokenizer = EsmSequenceTokenizer()


def tokenize(sequence):
    return encoding.tokenize_sequence(sequence, _tokenizer, add_special_tokens=True)


def score(logits):
    """Probability of a logit. The baseline took the fp16 logit as a Python float, so the sigmoid is evaluated in float64."""
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


class HlaStore:
    """Per-chain HLA embeddings (float16, (length, 960)), read from `<directory>/<chain>.npy` on first use and then kept."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.names = {path.stem for path in self.directory.glob("*.npy")}
        self._chains = {}

    def chain(self, name):
        embedding = self._chains.get(name)
        if embedding is None:
            if name not in self.names:
                raise KeyError(f"no HLA embedding for chain {name!r}")
            embedding = self._chains[name] = torch.from_numpy(np.load(self.directory / f"{name}.npy")).to(torch.float16)
        return embedding

    def preload(self):
        for name in sorted(self.names):
            self.chain(name)


class FastUnavailable(Exception):
    """The fast path cannot take this request; the message says why."""


class Engine:
    def __init__(self, root=".", device=None, mode=None):
        root = Path(root)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.hla = HlaStore(root / HLA_DIR)
        self.embedder = load_embedder(root / "models" / ESM_CHECKPOINT, self.device)
        heads = {name: self._load_head(root / "models" / file) for name, file in HEAD_CHECKPOINTS.items()}
        self.models = {name: UnifiedModel(self.embedder, head).eval() for name, head in heads.items()}
        self.custom_models = {name: UnifiedModel_custom_hla(self.embedder, head).eval() for name, head in heads.items()}
        self.mode = mode or os.environ.get("PREPIBIND_ENGINE", "auto")
        if self.mode not in ("auto", "fast", "exact"):
            raise ValueError(f"PREPIBIND_ENGINE must be auto, fast or exact, not {self.mode!r}")
        self.plans, self.plans_note = None, "PREPIBIND_ENGINE=exact"
        if self.mode != "exact":
            files = [root / "models" / ESM_CHECKPOINT, *(root / "models" / file for file in HEAD_CHECKPOINTS.values())]
            self.plans, self.plans_note = trt_engines.load(root, files, list(HEAD_CHECKPOINTS)) if self.device.type == "cuda" else (None, "no GPU")
            if self.plans is None and self.mode == "fast":
                raise RuntimeError(f"PREPIBIND_ENGINE=fast, but {self.plans_note}")
        print(f"engine: mode {self.mode}, fast path {'ready' if self.plans else 'not available (' + self.plans_note + ')'}", flush=True)  # the service log says which path serves
        self._fast_lock = threading.Lock()  # the plans share one scratch buffer: one fast request at a time

    def _load_head(self, path):
        head = plm_cat_mean_inf(**HEAD_ARGS)
        head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"])
        return head.to(self.device, dtype=torch.float16).eval()

    def run(self, df, head, batch_size=128, progress=None, mode=None):
        """Predict every row of `df` with the named head. Columns: MHC = '<alpha allele>_<beta allele>' (names in the HLA store) and Epitope.
        Returns a copy with Logits (float16) and Score (float64) added; rows lacking MHC or Epitope are dropped, as in the baseline.
        `progress(batches_done, batches_total)` is called after every batch. `mode` overrides the engine's (auto, fast, exact) for this call;
        `result.attrs['engine']` says which path ran ('fast' or 'exact')."""
        mode = mode or self.mode
        if mode != "exact":
            try:
                if self.plans is None:
                    raise FastUnavailable(self.plans_note)
                return self._run_fast(df, head, batch_size, progress)
            except FastUnavailable:
                if mode == "fast":
                    raise
        return self._run(df, self.models[head], self._standard_batch, batch_size, progress)

    def _run_fast(self, df, head, batch_size, progress):
        """Distinct epitopes through ESM C, distinct HLA pairs through the head's HLA encoder, then the joint blocks, in batches of rows of one epitope length (no padding
        of the epitope side); the logits go back to the rows' order. Raises FastUnavailable for what the plans do not cover."""
        out = df.dropna(subset=["MHC", "Epitope"]).copy()
        mhc, epitope = out["MHC"].tolist(), out["Epitope"].tolist()
        if not mhc:
            raise FastUnavailable("no rows")
        low, high = trt_engines.EPI_LEN
        if not low <= min(map(len, epitope)) <= max(map(len, epitope)) <= high:
            raise FastUnavailable(f"epitope lengths {min(map(len, epitope))}-{max(map(len, epitope))} are outside the fast engine's {low}-{high}")
        distinct = {}  # length -> the distinct epitopes of that length; an epitope's position in its list is its row in the embedding buffer
        for e in dict.fromkeys(epitope):
            distinct.setdefault(len(e), []).append(e)
        position = {e: i for es in distinct.values() for i, e in enumerate(es)}
        tokens = {e: torch.as_tensor(tokenize(e), dtype=torch.long) for es in distinct.values() for e in es}
        if any(len(t) != len(e) + 2 for e, t in tokens.items()):
            raise FastUnavailable("an epitope contains characters the tokenizer does not take one by one")
        pairs = list(dict.fromkeys(mhc))
        pair_embeddings = [torch.cat([self.hla.chain(a), self.hla.chain(b)]) for a, b in (pair.split("_") for pair in pairs)]
        pair_width = max(max(map(len, pair_embeddings)), trt_engines.HLA_LEN[0])
        if pair_width > trt_engines.HLA_LEN[1]:
            raise FastUnavailable(f"an HLA pair of {pair_width} residues is longer than the fast engine's {trt_engines.HLA_LEN[1]}")
        step, dev, plans = min(batch_size, trt_engines.MAX_BATCH), self.device, self.plans
        rows = {}  # epitope length -> the rows (in the request's order) with that length
        for i, e in enumerate(epitope):
            rows.setdefault(len(e), []).append(i)
        total = sum(-(-len(es) // step) for es in distinct.values()) + -(-len(pairs) // step) + sum(-(-len(r) // step) for r in rows.values())
        done = 0
        with self._fast_lock, torch.inference_mode():
            embeddings = {}
            for length, es in distinct.items():
                buffer = torch.empty(len(es), length, 960, dtype=torch.float16, device=dev)
                for i in range(0, len(es), step):
                    chunk = torch.stack([tokens[e] for e in es[i:i + step]]).to(dev)
                    buffer[i:i + len(chunk)] = plans.esm(chunk)[:, 1:-1, :]
                    done += 1
                    if progress:
                        progress(done, total)
                embeddings[length] = buffer
            hla = torch.zeros(len(pairs), pair_width, 960, dtype=torch.float16, device=dev)
            hla_mask = torch.ones(len(pairs), pair_width, dtype=torch.bool, device=dev)
            for i in range(0, len(pairs), step):
                chunk = pair_embeddings[i:i + step]
                width = max(max(map(len, chunk)), trt_engines.HLA_LEN[0])
                x = torch.zeros(len(chunk), width, 960, dtype=torch.float16)
                mask = torch.ones(len(chunk), width, dtype=torch.bool)
                for j, embedding in enumerate(chunk):
                    x[j, :len(embedding)] = embedding
                    mask[j, :len(embedding)] = False
                mask = mask.to(dev)
                hla[i:i + len(chunk), :width] = plans.hla(head, x.to(dev), mask)
                hla_mask[i:i + len(chunk), :width] = mask
                done += 1
                if progress:
                    progress(done, total)
            pair_index = {pair: i for i, pair in enumerate(pairs)}
            pair_of_row = torch.tensor([pair_index[m] for m in mhc], device=dev)
            epitope_of_row = torch.tensor([position[e] for e in epitope], device=dev)
            logits = torch.empty(len(mhc), dtype=torch.float16, device=dev)
            for length, members in rows.items():
                members = torch.tensor(members, device=dev)
                for i in range(0, len(members), step):
                    r = members[i:i + step]
                    p = pair_of_row[r]
                    logit = plans.joint(head, hla[p], embeddings[length][epitope_of_row[r]], hla_mask[p], torch.zeros(len(r), length, dtype=torch.bool, device=dev))
                    logits[r] = logit[:, 0]
                    done += 1
                    if progress:
                        progress(done, total)
            logits = logits.cpu().numpy()
        out["Logits"] = logits
        out["Score"] = score(logits)
        out.attrs["engine"] = "fast"
        return out

    def run_custom(self, df, head, batch_size=128, progress=None):
        """The same for HLA chains given as sequences: MHC = '<alpha sequence>_<beta sequence>'. The chains are embedded on the fly with the same ESM C."""
        return self._run(df, self.custom_models[head], self._custom_batch, batch_size, progress)

    def _run(self, df, model, make_batch, batch_size, progress):
        out = df.dropna(subset=["MHC", "Epitope"]).copy()
        mhc, epitope = out["MHC"].tolist(), out["Epitope"].tolist()
        n_batches = -(-len(out) // batch_size)
        logits = []
        with torch.inference_mode():
            for i in range(n_batches):
                rows = slice(i * batch_size, (i + 1) * batch_size)
                batch = make_batch(mhc[rows], epitope[rows])
                logits.append(model(*(tensor.to(self.device) for tensor in batch)).cpu())
                if progress:
                    progress(i + 1, n_batches)
        out["Logits"] = torch.cat(logits).numpy()[:, 0]
        out["Score"] = score(out["Logits"].to_numpy())
        out.attrs["engine"] = "exact"
        return out

    def _standard_batch(self, mhc, epitope):
        chains = (name.split("_") for name in mhc)
        return pad_and_mask_collate_fn_inf([(torch.cat([self.hla.chain(a), self.hla.chain(b)], dim=0), tokenize(e)) for (a, b), e in zip(chains, epitope)])

    def _custom_batch(self, mhc, epitope):
        chains = (name.split("_") for name in mhc)
        return pad_and_mask_collate_fn_inf_custom_hla([(tokenize(a), tokenize(b), tokenize(e)) for (a, b), e in zip(chains, epitope)])
