"""Inference: one ESMC, three heads, and no Streamlit anywhere in this file.

Two things this module exists to get right.

**One shared ESMC.** The img1 server built a separate ESMC-300M per head — four copies of a 300M
model to serve four tails of 55.8M parameters each. Here the language model is loaded once and the
heads are swapped over it. Measured budget for the whole set, fp16: ~870 MiB of heads plus ~600 MiB
of ESMC (`img2-server-scope.md` §2).

**The %Rank path is the grid's path.** A percentile is only meaningful against a background computed
the same way, and `scan.py` records two measurements that make this sharp:

  * batching shifts a logit by ~1.9 % of its spread, and the offset only cancels if query and
    background share `embed_windows`. So peptides are embedded through that function, not through a
    DataLoader of our own.
  * batch *size* moves a %Rank by 0.32 percentile points between 64 and 128 — six times the entire
    fp16 effect. `EMBED_BATCH` is therefore pinned to the value the panel grids were built with and
    is not a tuning knob.

Peptides are also scored in batches of a single length, never padded. That is what
`scan.score_windows` does and it is why the epitope padding mask cannot bite here.

Nothing in this file imports `streamlit`: progress is a callback. The scoring backend is meant to be
replaceable — a TensorRT or fp8 engine substitutes for `_forward` — but note that any such change
moves a served logit and therefore every %Rank, so the grids must be rebuilt through the same path
before it is served. That needs a decision, not just a benchmark.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts
import chains
import rank as rankmod
import scan

# Pinned to the panel grid build (`grid_b_*_meta.json`: emb_batch 64). Not a tuning knob.
EMBED_BATCH = 64
SCORE_BATCH = 256

# The validated / reportable range. Input is accepted to 25 and scored, but nothing above this is
# pooled into a performance figure anywhere in the project.
VALIDATED_MAX = 21

# The ESMC weights shipped with the server are fp16 and there is no fp32 copy; the panel grids were
# scored in fp32 (`grid_precision`). The resulting deviation is the smallest term the project has
# measured on this path — fp16 embeddings differ by at most 8.79e-03 and move a %Rank by well under
# the 0.32 pp that batch size does (DISCLOSURES item 19). The head runs in fp32, which is free.
ESM_DTYPE = "float16"
HEAD_DTYPE = "float32"


class Engine:
    """One ESMC and the requested heads, resident on one device."""

    def __init__(self, heads=None, device=None):
        import torch
        from esm.models.esmc import ESMC
        from esm.tokenization import get_esmc_model_tokenizers

        from model import plm_cat_mean_inf

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.client = ESMC(d_model=960, n_heads=15, n_layers=30,
                           tokenizer=get_esmc_model_tokenizers(), use_flash_attn=True)
        self.client.load_state_dict(
            torch.load(artifacts.ESM_CHECKPOINT, map_location="cpu", weights_only=False))
        self.client.to(self.device, dtype=getattr(torch, ESM_DTYPE)).eval()

        self.heads = {}
        for head in (heads or artifacts.HEADS):
            m = plm_cat_mean_inf(hla_dim=960, epi_dim=960, head_div=64)
            state = torch.load(artifacts.checkpoint(head), map_location="cpu",
                               weights_only=False)["model_state_dict"]
            m.load_state_dict({k: v.float() for k, v in state.items()})
            self.heads[head] = m.to(self.device, dtype=getattr(torch, HEAD_DTYPE)).eval()

    def add_head(self, name, checkpoint):
        """Load one more head over the shared ESMC — used for the img1 legacy checkpoints.

        The language model is the expensive resident; a head is 55.8 M parameters, so the legacy
        page costs a tail rather than a second model.
        """
        import torch

        from model import plm_cat_mean_inf

        if name not in self.heads:
            m = plm_cat_mean_inf(hla_dim=960, epi_dim=960, head_div=64)
            state = torch.load(checkpoint, map_location="cpu",
                               weights_only=False)["model_state_dict"]
            m.load_state_dict({k: v.float() for k, v in state.items()})
            self.heads[name] = m.to(self.device, dtype=getattr(torch, HEAD_DTYPE)).eval()
        return self.heads[name]

    # ------------------------------------------------------------------ embedding
    def embed(self, peptides, progress=None):
        """{peptide: (L, 960)} through the same function the %Rank background was built with."""
        return scan.embed_windows(peptides, self.client, batch_size=EMBED_BATCH, progress=progress)

    # ------------------------------------------------------------------ scoring
    def _forward(self, head, hla_emb, epi_batch):
        """One batch of equal-length peptides against one molecule. The replaceable part."""
        torch = self.torch
        xh = torch.tensor(hla_emb, dtype=torch.float32, device=self.device).unsqueeze(0)
        xe = torch.tensor(np.stack(epi_batch), dtype=torch.float32, device=self.device)
        n = xe.shape[0]
        with torch.no_grad():
            out = self.heads[head](
                xh.expand(n, -1, -1), xe,
                torch.zeros(n, xh.shape[1], dtype=torch.bool, device=self.device),
                torch.zeros(n, xe.shape[1], dtype=torch.bool, device=self.device))
        return out.float().cpu().numpy().ravel()

    def score(self, df, head, epi_emb, progress=None, legacy=False):
        """Logits for a frame of (molecule, Epitope) rows. Batched within one molecule and length.

        Never mixes lengths in a batch: `score_windows` does the same, and it is what keeps the
        epitope side unpadded.
        """
        logits = np.full(len(df), np.nan, dtype=np.float32)
        done, total = 0, len(df)
        for molecule, per_mol in df.groupby("molecule", sort=False):
            alpha, beta = per_mol.iloc[0].MHC_alpha, per_mol.iloc[0].MHC_beta
            hla = chains.molecule_embedding(alpha, beta, legacy=legacy)
            for _, per_len in per_mol.groupby(per_mol.Epitope.str.len(), sort=True):
                pos = df.index.get_indexer(per_len.index)
                peps = per_len.Epitope.tolist()
                for i in range(0, len(peps), SCORE_BATCH):
                    sl = slice(i, i + SCORE_BATCH)
                    logits[pos[sl]] = self._forward(head, hla, [epi_emb[p] for p in peps[sl]])
                    done += len(peps[sl])
                    if progress:
                        progress(done, total)
        return logits

    # ------------------------------------------------------------------ the whole call
    def predict(self, df, heads=None, progress=None):
        """(MHC_alpha, MHC_beta, Epitope) -> the same frame with logit, %Rank and band per head.

        Embedding is done once and reused across heads: it is the whole cost of a prediction, while
        a head forward over the same embeddings is negligible. Showing the heads together is the
        point of reporting a percentile — a raw logit of 2.0 is 3.30 % on MS and 0.32 % on
        IC50 <500 nM, so the three columns are only commensurable once they are ranks.

        `<head>_rank_pct` is NaN for a molecule outside the 306-molecule panel: it has no
        background, and a bare logit is not comparable across alleles, lengths or heads. Building
        that background is `img2-server-output-contract.md` §3 and does not happen here.
        """
        heads = [heads] if isinstance(heads, str) else list(heads or self.heads)
        out = df.copy().reset_index(drop=True)
        out["molecule"] = [chains.molecule(a, b) for a, b in zip(out.MHC_alpha, out.MHC_beta)]
        out["length"] = out.Epitope.str.len()
        # 12-21 is the only range a pooled figure on this server covers; 22-25 is scored and
        # published per length but never pooled (`img2-ms-arm-and-length-range.md` §2). Carrying it
        # as a column means the caveat survives into a downloaded CSV.
        out["length_support"] = np.where(out.length <= VALIDATED_MAX, "validated", "limited")

        epi_emb = self.embed(out.Epitope.tolist(), progress=progress)
        bands = artifacts.bands()

        for head in heads:
            logit = self.score(out, head, epi_emb)
            pct = np.full(len(out), np.nan)
            for (molecule, length), grp in out.groupby(["molecule", "length"], sort=False):
                r = rankmod.rank(head, molecule, int(length), logit[out.index.get_indexer(grp.index)])
                if r is not None:
                    pct[out.index.get_indexer(grp.index)] = r
            out[f"{head}_logit"] = logit
            out[f"{head}_rank_pct"] = pct
            out[f"{head}_band"] = [rankmod.band(v, bands) for v in pct]

        out["on_panel"] = ~np.isnan(out[f"{heads[0]}_rank_pct"])
        return out


def load(heads=None, device=None):
    return Engine(heads=heads, device=device)
