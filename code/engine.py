"""Inference engine of the img1 web server: the four PREpiBind heads behind one ESM C 300M, as plain functions of a DataFrame.

It replaces what the baseline spread over app.py (model loading), inference.py (DataLoader, Streamlit progress bar, sigmoid) and encoder.py / dataprovider.py (HLA lookup, tokenising):
no Streamlit import, no DataLoader workers, no configuration file executed as code, no hard-coded GPU index (select the GPU with CUDA_VISIBLE_DEVICES). The arithmetic is the baseline's:
the same batches (consecutive `batch_size` rows), the same collate functions, the same fp16 modules; the one change in structure is a single ESM C shared by the four heads instead of
four copies of the same weights. Eager execution only.
"""
from pathlib import Path

import numpy as np
import torch
from esm.tokenization import EsmSequenceTokenizer
from esm.utils import encoding

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


class Engine:
    def __init__(self, root=".", device=None):
        root = Path(root)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.hla = HlaStore(root / HLA_DIR)
        self.embedder = load_embedder(root / "models" / ESM_CHECKPOINT, self.device)
        heads = {name: self._load_head(root / "models" / file) for name, file in HEAD_CHECKPOINTS.items()}
        self.models = {name: UnifiedModel(self.embedder, head).eval() for name, head in heads.items()}
        self.custom_models = {name: UnifiedModel_custom_hla(self.embedder, head).eval() for name, head in heads.items()}

    def _load_head(self, path):
        head = plm_cat_mean_inf(**HEAD_ARGS)
        head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"])
        return head.to(self.device, dtype=torch.float16).eval()

    def run(self, df, head, batch_size=128, progress=None):
        """Predict every row of `df` with the named head. Columns: MHC = '<alpha allele>_<beta allele>' (names in the HLA store) and Epitope.
        Returns a copy with Logits (float16) and Score (float64) added; rows lacking MHC or Epitope are dropped, as in the baseline.
        `progress(batches_done, batches_total)` is called after every batch."""
        return self._run(df, self.models[head], self._standard_batch, batch_size, progress)

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
        return out

    def _standard_batch(self, mhc, epitope):
        chains = (name.split("_") for name in mhc)
        return pad_and_mask_collate_fn_inf([(torch.cat([self.hla.chain(a), self.hla.chain(b)], dim=0), tokenize(e)) for (a, b), e in zip(chains, epitope)])

    def _custom_batch(self, mhc, epitope):
        chains = (name.split("_") for name in mhc)
        return pad_and_mask_collate_fn_inf_custom_hla([(tokenize(a), tokenize(b), tokenize(e)) for (a, b), e in zip(chains, epitope)])
