"""Inference engine of the img1 web server: the four PREpiBind heads behind one ESMC 300M, as plain functions of a DataFrame.

It replaces what the baseline spread over app.py (model loading), inference.py (DataLoader, Streamlit progress bar, sigmoid) and encoder.py / dataprovider.py (HLA lookup, tokenising):
no Streamlit import, no DataLoader workers, no configuration file executed as code, no hard-coded GPU index (select the GPU with CUDA_VISIBLE_DEVICES). The arithmetic is the baseline's:
the same batches (consecutive `batch_size` rows), the same collate functions, the same fp16 modules; the one change in structure is a single ESMC shared by the four heads instead of
four copies of the same weights.

Two ways to run a request. The EXACT path is the baseline's arithmetic in eager mode (bitwise equal to the baseline code in the same environment). The FAST path
(`_run_fast`) computes every distinct epitope through ESMC once and every distinct HLA pair through the head's HLA encoder once, runs the joint blocks per batch of
rows of one epitope length, and executes the three kinds of graph as TensorRT fp16 plans (trt_engines.py): most rows differ from the exact path in the last bits (mean
0.01-0.02 logits), about 0.1 % of the rows change sign (almost all within 0.15 logits of the threshold), 4.5x faster on bulk uploads and in less GPU memory.
`PREPIBIND_ENGINE` picks: `auto` (default) = fast when plans for this TensorRT and GPU are loaded and the request is in the plans' ranges, else exact; `fast` = fast or
an error; `exact` = never load the plans. Custom mode is always exact.
"""
import contextlib
import functools
import hashlib
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from esm.tokenization import EsmSequenceTokenizer
from esm.utils import encoding

import trt_engines
from collate import pad_and_mask_collate_fn_inf, pad_and_mask_collate_fn_inf_custom_hla
from embedder import load_embedder
from locator import Locator, tier
from model import UnifiedModel, UnifiedModel_custom_hla, plm_cat_mean_inf

ESM_CHECKPOINT = "esmc_300m_2024_12_v0_fp16.pth"
HLA_DIR = "data/emb_hla_esmc_small_0601_fp16"
CATALOGUE = "data/mhc_mapping.csv"  # allele name, full chain sequence and the domain window [start_idx, end_idx) the stored embedding was cut to
# The four task checkpoints released with the paper (float16 copies: huggingface.co/daylight-00/prepibind-demo). Each is the run with the lowest validation loss among the 15 runs (3 seeds x 5 folds) the paper reports for the task, chosen without
# any test-set quantity; the seed and the fold are in the names. One run, not the paper's mean over the 15. Before 2026-11-15 the server loaded other runs of the same tasks (prepi_esmc_small_*_e5_s128_f4 and others, never published).
HEAD_CHECKPOINTS = {
    "Qualitative": "prepibind_qualitative_s100_f0_fp16.pt",
    "Mass Spectrometry": "prepibind_ms_s128_f3_fp16.pt",
    "IC50 (<500nM)": "prepibind_ic50_500_s128_f2_fp16.pt",
    "IC50 (<1000nM)": "prepibind_ic50_1000_s42_f1_fp16.pt",
}
HEAD_ARGS = {"hla_dim": 960, "epi_dim": 960, "head_div": 64}
EMBEDDING_BUDGET = 1 << 30  # bytes of epitope embeddings the fast path holds at once (about 37,000 distinct 15-mers, 17,000 distinct 32-mers)
MAX_PAIRS = 2048  # distinct HLA pairs the fast path encodes at once (the pair encodings take 0.6 GB); a request with more runs on the exact path
MAX_EPITOPE_LENGTH = 20  # residues, refused above (261115_img1_web_caps_scan_custom): the heads were trained on 15-mers only and with random flanks the AUC falls from 0.93 at 15 to 0.87 at 20 and 0.82 at 25 (a shuffled epitope: 0.80) while the share of non-binders scored above 0.5 rises from 15 % to 27 % and 41 % (the released Qualitative head, 261115_img1_web_snu_conditions/swap_eval/length)
MAX_CHAIN_LENGTH = 300  # residues of an HLA chain given as a sequence (Custom Mode), refused above: the allele catalogue's longest chain has 285, 99 % have at most 266

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


def chain_name_map(catalogue):
    """Chain sequence -> the first allele of the catalogue (columns HLA_Name, HLA_Seq) with that sequence."""
    names = {}
    for name, sequence in zip(catalogue.HLA_Name, catalogue.HLA_Seq):
        names.setdefault(sequence, name)
    return names


class OverlayStore:
    """The HLA store plus the domain embeddings made for one request (chains that are not in the catalogue): `chain(name)` looks in the request's own first. Nothing is added to the shared store."""

    def __init__(self, base, extra):
        self.base, self.extra = base, extra

    def chain(self, name):
        return self.extra[name] if name in self.extra else self.base.chain(name)


MIN_WINDOW = 20  # residues: a domain window shorter than this is not a domain (the catalogue's are 58-82)


class FastUnavailable(Exception):
    """The fast path cannot take this request; the message says why."""


class RequestRefused(ValueError):
    """The request is outside the limits the service accepts (MAX_EPITOPE_LENGTH, MAX_CHAIN_LENGTH); the message says which and is meant to be shown to the user as it is."""


def check_request(epitopes, chains=()):
    """Refuse an epitope longer than MAX_EPITOPE_LENGTH or an HLA chain longer than MAX_CHAIN_LENGTH. The limits are read when called, so a test or a benchmark can raise them."""
    epitopes, chains = list(epitopes), list(chains)
    too_long = [len(e) for e in epitopes if len(e) > MAX_EPITOPE_LENGTH]
    if too_long:
        raise RequestRefused(f"{len(too_long)} epitope(s) are longer than the maximum of {MAX_EPITOPE_LENGTH} residues (the longest has {max(too_long)}): the model was trained on 15-mers and its scores beyond {MAX_EPITOPE_LENGTH} residues carry almost no information")
    too_long = [len(c) for c in chains if len(c) > MAX_CHAIN_LENGTH]
    if too_long:
        raise RequestRefused(f"{len(too_long)} HLA chain(s) are longer than the maximum of {MAX_CHAIN_LENGTH} residues (the longest has {max(too_long)}); the longest chain of the allele catalogue has 285")


_engines, _engines_lock = {}, threading.Lock()


def shared(root="."):
    """The process's Engine for `root`. Streamlit runs app.py as the main script and again as the module `app` that the pages import, so a cache inside app.py
    builds two engines: two copies of the weights and plans on the GPU and of the HLA store in memory."""
    key = str(Path(root).resolve())
    with _engines_lock:
        if key not in _engines:
            _engines[key] = Engine(root=root)
        return _engines[key]


class Engine:
    def __init__(self, root=".", device=None, mode=None):
        root = Path(root)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.hla = HlaStore(root / HLA_DIR)
        catalogue = pd.read_csv(root / CATALOGUE, usecols=["HLA_Name", "HLA_Seq", "start_idx", "end_idx"])
        self.chain_names = chain_name_map(catalogue)  # Custom Mode scores a chain of the catalogue as Standard Mode scores the allele
        self.domain_windows = dict(zip(catalogue.HLA_Name, zip(catalogue.start_idx.astype(int), catalogue.end_idx.astype(int))))
        self._catalogue, self._locator, self._locator_lock = catalogue, None, threading.Lock()  # the locator (a 4-mer index of the catalogue) is built when a chain outside the catalogue first needs it
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
            with self._device_scope():
                self.plans, self.plans_note = trt_engines.load(root, files, list(HEAD_CHECKPOINTS)) if self.device.type == "cuda" else (None, "no GPU")
            if self.plans is None and self.mode == "fast":
                raise RuntimeError(f"PREPIBIND_ENGINE=fast, but {self.plans_note}")
        print(f"engine: mode {self.mode}, fast path {'ready' if self.plans else 'not available (' + self.plans_note + ')'}", flush=True)  # the service log says which path serves
        self._fast_lock = threading.Lock()  # the plans share one scratch buffer: one fast request at a time

    def _load_head(self, path):
        head = plm_cat_mean_inf(**HEAD_ARGS)
        head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"])
        return head.to(self.device, dtype=torch.float16).eval()

    def run(self, df, head, batch_size=128, progress=None, mode=None, store=None):
        """Predict every row of `df` with the named head. Columns: MHC = '<alpha allele>_<beta allele>' (names in the HLA store) and Epitope.
        Returns a copy with Logits (float16) and Score (float64) added; rows lacking MHC or Epitope are dropped, as in the baseline.
        `progress(batches_done, batches_total)` is called after every batch. `mode` overrides the engine's (auto, fast, exact) for this call;
        `result.attrs['engine']` says which path ran ('fast' or 'exact'). In `auto` the exact path takes over, and the log says so for the cases that are not obvious, when the plans do not
        cover the request (epitope length, HLA pair length, more than MAX_PAIRS distinct pairs), when the GPU runs out of memory, when TensorRT produces a non-finite logit,
        and (for the rest of the process) when TensorRT fails while running. `store` replaces the engine's HLA store for this call (Custom Mode's OverlayStore)."""
        mode = mode or self.mode
        store = store or self.hla
        if mode not in ("auto", "fast", "exact"):
            raise ValueError(f"mode must be auto, fast or exact, not {mode!r}")
        check_request(df["Epitope"].dropna())
        out_of_memory = False
        if mode != "exact":
            try:
                if self.plans is None:
                    raise FastUnavailable(self.plans_note)
                return self._run_fast(df, head, batch_size, progress, store)
            except FastUnavailable:
                if mode == "fast":
                    raise
            except torch.cuda.OutOfMemoryError:
                if mode == "fast":
                    raise
                out_of_memory = True
            except trt_engines.TrtError as error:
                if mode == "fast":
                    raise
                self.plans, self.plans_note = None, f"TensorRT failed while running ({error}); serving the exact path until the service restarts"
                print(f"engine: {self.plans_note}", flush=True)
        if out_of_memory:
            torch.cuda.empty_cache()  # the tensors of the failed request are released now that its exception is gone
            print("engine: the fast path ran out of GPU memory; this request runs on the exact path", flush=True)
        return self._run(df, self.models[head], functools.partial(self._standard_batch, store=store), batch_size, progress)

    def _device_scope(self):
        """The engine's GPU as the current CUDA device: TensorRT contexts, the scratch buffer and the outputs are created on the current device."""
        return torch.cuda.device(self.device) if self.device.type == "cuda" else contextlib.nullcontext()

    def _run_fast(self, df, head, batch_size, progress, store):
        """Distinct epitopes through ESMC, distinct HLA pairs through the head's HLA encoder, then the joint blocks, in batches of rows of one epitope length (no padding
        of the epitope side); the logits go back to the rows' order. The memory of a request is bounded whatever its size: the epitope embeddings are held in blocks of at most
        EMBEDDING_BUDGET bytes (rows are ordered by their epitope), at most MAX_PAIRS distinct pairs are encoded at once and their embeddings are built chunk by chunk.
        Raises FastUnavailable for what the plans do not cover."""
        out = df.dropna(subset=["MHC", "Epitope"]).copy()
        mhc, epitope = out["MHC"].tolist(), out["Epitope"].tolist()
        if not mhc:
            raise FastUnavailable("no rows")
        low, high = trt_engines.EPI_LEN
        if not low <= min(map(len, epitope)) <= max(map(len, epitope)) <= high:
            raise FastUnavailable(f"epitope lengths {min(map(len, epitope))}-{max(map(len, epitope))} are outside the fast engine's {low}-{high}")
        pairs = list(dict.fromkeys(mhc))
        if len(pairs) > MAX_PAIRS:
            raise FastUnavailable(f"{len(pairs)} distinct HLA pairs are more than the fast engine takes at once ({MAX_PAIRS})")
        chains = [pair.split("_") for pair in pairs]
        pair_width = max(max(len(store.chain(a)) + len(store.chain(b)) for a, b in chains), trt_engines.HLA_LEN[0])
        if pair_width > trt_engines.HLA_LEN[1]:
            raise FastUnavailable(f"an HLA pair of {pair_width} residues is longer than the fast engine's {trt_engines.HLA_LEN[1]}")
        distinct = {}  # length -> the distinct epitopes of that length; an epitope's position in its list decides its block and its row in the block
        for e in dict.fromkeys(epitope):
            distinct.setdefault(len(e), []).append(e)
        position = {e: i for es in distinct.values() for i, e in enumerate(es)}
        tokens = {e: torch.as_tensor(tokenize(e), dtype=torch.long) for es in distinct.values() for e in es}
        if any(len(t) != len(e) + 2 for e, t in tokens.items()):
            raise FastUnavailable("an epitope contains characters the tokenizer does not take one by one")
        step, dev, plans = min(batch_size, trt_engines.MAX_BATCH), self.device, self.plans
        rows = {}  # epitope length -> the rows (in the request's order) with that length
        for i, e in enumerate(epitope):
            rows.setdefault(len(e), []).append(i)
        plan = {}  # length -> (rows ordered by the position of their epitope, those positions, epitopes per block, the first row of every block)
        total = -(-len(pairs) // step)
        for length, es in distinct.items():
            members = np.asarray(rows[length], dtype=np.int64)
            positions = np.fromiter((position[epitope[i]] for i in members), dtype=np.int64, count=len(members))
            order = np.argsort(positions, kind="stable")
            members, positions = members[order], positions[order]
            size = max(1, EMBEDDING_BUDGET // (length * 960 * 2 * step)) * step  # a multiple of the batch, so the blocks cut the epitopes at batch borders
            edges = np.searchsorted(positions, np.arange(0, len(es) + size, size))
            plan[length] = (members, positions, size, edges)
            total += -(-len(es) // step) + sum(-(-int(edges[b + 1] - edges[b]) // step) for b in range(len(edges) - 1))
        done = 0
        with self._device_scope(), self._fast_lock, torch.inference_mode():
            hla = torch.zeros(len(pairs), pair_width, 960, dtype=torch.float16, device=dev)
            hla_mask = torch.ones(len(pairs), pair_width, dtype=torch.bool, device=dev)
            for i in range(0, len(pairs), step):
                chunk = [torch.cat([store.chain(a), store.chain(b)]) for a, b in chains[i:i + step]]
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
            logits = torch.empty(len(mhc), dtype=torch.float16, device=dev)
            for length, (members, positions, size, edges) in plan.items():
                es = distinct[length]
                for b in range(len(edges) - 1):
                    start = b * size
                    block = torch.empty(min(size, len(es) - start), length, 960, dtype=torch.float16, device=dev)
                    for i in range(0, len(block), step):
                        chunk = torch.stack([tokens[e] for e in es[start + i:start + i + step]]).to(dev)
                        block[i:i + len(chunk)] = plans.esm(chunk)[:, 1:-1, :]
                        done += 1
                        if progress:
                            progress(done, total)
                    lo, hi = int(edges[b]), int(edges[b + 1])
                    in_block = torch.from_numpy(members[lo:hi]).to(dev)
                    row_in_block = torch.from_numpy(positions[lo:hi] - start).to(dev)
                    for i in range(0, hi - lo, step):
                        r = in_block[i:i + step]
                        p = pair_of_row[r]
                        logit = plans.joint(head, hla[p], block[row_in_block[i:i + step]], hla_mask[p], torch.zeros(len(r), length, dtype=torch.bool, device=dev))
                        logits[r] = logit[:, 0]
                        done += 1
                        if progress:
                            progress(done, total)
                    del block
            logits = logits.cpu().numpy()
            if torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev) > 1 << 30:
                torch.cuda.empty_cache()  # the GPU is shared with the portal: do not keep the cache of a large request
        if not np.isfinite(logits).all():
            raise FastUnavailable("TensorRT produced a non-finite logit")
        out["Logits"] = logits
        out["Score"] = score(logits)
        out.attrs["engine"] = "fast"
        return out

    @property
    def locator(self):
        with self._locator_lock:
            if self._locator is None:
                self._locator = Locator.from_catalogue(self._catalogue)
            return self._locator

    def chain_report(self, chain):
        """What Custom Mode makes of one HLA chain: its tier ('catalogue', 'located', 'low confidence' or 'no domain'), the nearest allele of the catalogue, the identity with it over its domain and the window
        of the chain's domain (positions in the chain, half-open; None for 'no domain', also when the located window is shorter than MIN_WINDOW). `hit` is the locator's own result."""
        chain = chain.strip().upper()
        name = self.chain_names.get(chain)
        if name:
            start, end = self.domain_windows[name]
            return {"tier": "catalogue", "nearest": name, "identity": 1.0, "start": start, "end": end, "hit": None}
        hit = self.locator.locate(chain)
        if hit is None:
            return {"tier": "no domain", "nearest": None, "identity": 0.0, "start": None, "end": None, "hit": None}
        found = tier(hit["identity"]) if hit["end"] - hit["start"] >= MIN_WINDOW else "no domain"
        window = (None, None) if found == "no domain" else (hit["start"], hit["end"])
        return {"tier": found, "nearest": hit["nearest"], "identity": hit["identity"], "start": window[0], "end": window[1], "hit": hit}

    def _plan(self, chain, windows):
        """What one normalised chain gets: ('catalogue', name), ('manual window', start, end, None), ('located' or 'low confidence', start, end, hit) or ('full chain',)."""
        if chain in self.chain_names:
            return ("catalogue", self.chain_names[chain])
        if chain in windows:
            start, end = windows[chain]
            if not 0 <= start < end <= len(chain) or end - start < MIN_WINDOW:
                raise RequestRefused(f"the domain window {start + 1}-{end} is not inside the chain of {len(chain)} residues or is shorter than {MIN_WINDOW} residues")
            return ("manual window", start, end, None)
        report = self.chain_report(chain)
        return ("full chain",) if report["tier"] == "no domain" else (report["tier"], report["start"], report["end"], report["hit"])

    def _domain_embedding(self, chain, start, end, hit):
        """The embedding of the domain [start, end) of `chain` as the model was trained on it: the whole chain through ESMC, with the nearest chain's flanks added where a located chain has less of them
        (`hit`; a window the user gave is cut as it is), then the domain's positions. Float16 on the CPU, like the stored embeddings."""
        grafted, offset = self.locator.complete_flanks(chain, hit) if hit else (chain, 0)
        tokens = torch.as_tensor(tokenize(grafted), dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embedding = self.embedder(tokens)[0]
        return embedding[1 + offset + start:1 + offset + end].to("cpu", torch.float16)

    def run_custom(self, df, head, batch_size=128, progress=None, windows=None):
        """The same for HLA chains given as sequences: MHC = '<alpha sequence>_<beta sequence>'. The model's HLA input is the domain of a chain (58-82 residues) cut from the embedding of the whole chain, so:
        a chain of the allele catalogue (the same sequence) is scored as Standard Mode scores the allele; any other chain has its domain located by alignment to the catalogue and is embedded on the fly with
        that window cut out (with the nearest chain's flanks added where the chain has fewer); `windows` ({chain: (start, end)}, 0-based, half-open) replaces the located window of a chain. A chain with no
        domain found (identity with the nearest catalogue chain below 0.4) and no window given keeps the legacy run over its full length, which is not validated. Returns a copy with Logits, Score and
        `HLA chains`: catalogue, located, low confidence (identity 0.4 to 0.5), manual window or full chain, the worst of the two chains' tiers; `result.attrs['engine']` names the path or paths that ran."""
        out = df.dropna(subset=["MHC", "Epitope"]).copy()
        pairs = [[chain.strip().upper() for chain in pair.split("_")] for pair in out["MHC"]]
        check_request(out["Epitope"], (chain for pair in {tuple(p) for p in pairs} for chain in pair))
        windows = {chain.strip().upper(): window for chain, window in (windows or {}).items()}
        plans = {chain: self._plan(chain, windows) for chain in dict.fromkeys(chain for pair in pairs for chain in pair)}
        worst = {"full chain": 4, "low confidence": 3, "manual window": 2, "located": 1, "catalogue": 0}
        out["HLA chains"] = [max((plans[chain][0] for chain in pair), key=worst.__getitem__) for pair in pairs]
        full = (out["HLA chains"] == "full chain").to_numpy()
        names, extra = {}, {}  # chain -> its name in the store; the embeddings made for this request
        for chain, plan in plans.items():
            if plan[0] == "catalogue":
                names[chain] = plan[1]
            elif plan[0] != "full chain":
                _, start, end, hit = plan
                names[chain] = f"custom:{hashlib.sha1(f'{chain}:{start}:{end}'.encode()).hexdigest()[:12]}"
                if names[chain] not in extra:
                    extra[names[chain]] = self._domain_embedding(chain, start, end, hit)
        logits, paths, offset = np.empty(len(out), dtype=np.float16), [], 0
        for selected, sliced in ((~full, True), (full, False)):
            count = int(selected.sum())
            if not count:
                continue
            tick = None if progress is None else (lambda done, total, o=offset, c=count: progress(o + int(c * done / total), len(out)))
            part = out[selected]
            if sliced:
                named = ["_".join(names[chain] for chain in pair) for pair, keep in zip(pairs, selected) if keep]
                result = self.run(part.assign(MHC=named), head, batch_size, tick, store=OverlayStore(self.hla, extra) if extra else None)
            else:
                result = self._run(part, self.custom_models[head], self._custom_batch, batch_size, tick)
            logits[selected] = result["Logits"].to_numpy()
            paths.append(result.attrs["engine"])
            offset += count
        out["Logits"], out["Score"] = logits, score(logits)
        out.attrs["engine"] = "+".join(dict.fromkeys(paths))
        return out

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

    def _standard_batch(self, mhc, epitope, store=None):
        store = store or self.hla
        chains = (name.split("_") for name in mhc)
        return pad_and_mask_collate_fn_inf([(torch.cat([store.chain(a), store.chain(b)], dim=0), tokenize(e)) for (a, b), e in zip(chains, epitope)])

    def _custom_batch(self, mhc, epitope):
        chains = (name.split("_") for name in mhc)
        return pad_and_mask_collate_fn_inf_custom_hla([(tokenize(a), tokenize(b), tokenize(e)) for (a, b), e in zip(chains, epitope)])
