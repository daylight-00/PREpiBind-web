"""Building a %Rank background for a molecule that does not have one, on first request.

The panel is 306 molecules per head and the two chain lists compose about 1.36 million, so off-panel
is the *common* case. Precomputing all of it is 3.4 GPU-years for one head, which is why
`img2-server-output-contract.md` §3 chose to build on first request and cache: the user waits once.

This implements the same on-disk contract as `260905/5_rank/build_grid.py`, and departs from it in
exactly one way, for a reason that only applies to a server:

  * the canonical builder makes one grid per head, streaming the background embeddings and
    discarding them, because rebuilding for another head is cheaper than storing 7 GB. A server
    wants all three heads for the molecule the user just asked about, so the embedding pass — which
    is the whole cost — is shared across the three heads instead, and the head forwards ride along.

Everything else is held to the canonical contract, because a grid built differently is not a
background the query can be ranked against:

  * the background is embedded through the engine, which pins `scan.embed_windows` at the batch size
    the panel grids used;
  * percentiles are taken WITHIN a length, stored as 1,000 descending quantile breakpoints;
  * human molecules are scored against the human background and H2 against the mouse one.

A grid belongs to one head **on one host**. These are built on the serving host, whereas the panel
grids were built on abc's H100; the measured difference is 0.05 percentile points on average and
0.20 at most, and a query embedded here and ranked against the panel comes out uniform on the
background, which is the check `code/test_lineage.py` makes.
"""
import json
import os
import threading
import time

import numpy as np

import artifacts
import chains

# One build at a time. A burst of distinct molecules would otherwise pin the serving GPU, which is
# the failure `img2-server-output-contract.md` §3 asks for a queue and a per-session cap against.
# The cap is the interface's job; keeping the card serial is this module's.
GPU = threading.Lock()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "grids_cache")
N_BREAKS = 1000
SCORE_BATCH = 256
EMBED_CHUNK = 4096          # background peptides embedded at a time, within one length


def _background(molecule):
    """Mouse molecules are ranked against the mouse proteome, everything else against human."""
    name = "mouse" if molecule.startswith("H2") else "human"
    return os.path.join(ROOT, "data", f"background_{name}.txt")


def _safe(molecule):
    return molecule.replace("*", "_").replace(":", "_").replace("/", "_")


def path(molecule, head):
    return os.path.join(CACHE, f"grid_{head}_{_safe(molecule)}.npz")


def cached(molecule, head):
    return os.path.exists(path(molecule, head))


def _quantile_breaks(values, n_breaks=N_BREAKS):
    """`n_breaks` descending breakpoints — the layout `scan.PercentileGrid.rank` searches."""
    q = np.linspace(0.0, 1.0, n_breaks)
    return np.quantile(np.asarray(values, dtype=np.float64), q)[::-1].astype(np.float32)


def build(engine, alpha, beta, heads=None, progress=None, hla=None, molecule=None):
    """Score the whole background for one molecule and write a grid per head.

    Returns the metadata written beside the grids. Blocking and expensive by design: the serving GPU
    does one of these at a time, which is what the per-session cap in the interface is protecting.

    Measured on the serving 4070 Ti, in fp32: the embedding pass over the 100,000-peptide background
    is 73 s and is shared, and each head then costs 77 s, so one head is ~150 s and all three ~306 s.
    `img2-rank-panel-scope.md`'s 77.6 s per molecule is the *marginal* cost inside a batch build,
    where hundreds of molecules amortise one embedding pass; a lazy build has nothing to amortise it
    against, so it costs about twice that for a single head.
    """
    import torch

    heads = list(heads or artifacts.HEADS)
    # `hla` / `molecule` are the custom-HLA path: chains the catalogue does not have, embedded by
    # the caller. Everything else about the build is identical, which is the point -- a custom
    # molecule's background has to come off the same 100k peptides as the panel's.
    molecule = molecule or chains.molecule(alpha, beta)
    peptides = [p.strip() for p in open(_background(molecule)) if p.strip()]
    lengths = np.array([len(p) for p in peptides])

    hla = chains.molecule_embedding(alpha, beta) if hla is None else hla
    xh = torch.tensor(hla, dtype=torch.float32, device=engine.device).unsqueeze(0)
    mh = torch.zeros(1, xh.shape[1], dtype=torch.bool, device=engine.device)

    logits = {h: np.empty(len(peptides), dtype=np.float32) for h in heads}
    order = np.argsort(lengths, kind="stable")
    started, done = time.time(), 0

    # One pass over the background: embed a same-length chunk, score it with every head, drop it.
    for length in sorted(set(lengths.tolist())):
        idx_len = order[lengths[order] == length]
        for c in range(0, len(idx_len), EMBED_CHUNK):
            idx = idx_len[c:c + EMBED_CHUNK]
            chunk = [peptides[i] for i in idx]
            emb = engine.embed(chunk)
            arr = np.stack([emb[p] for p in chunk])
            for b in range(0, len(chunk), SCORE_BATCH):
                sl = slice(b, b + SCORE_BATCH)
                xe = torch.tensor(arr[sl], dtype=torch.float32, device=engine.device)
                n = xe.shape[0]
                me = torch.zeros(n, xe.shape[1], dtype=torch.bool, device=engine.device)
                for head in heads:
                    with torch.no_grad():
                        out = engine.heads[head](xh.expand(n, -1, -1), xe, mh.expand(n, -1), me)
                    logits[head][idx[sl]] = out.float().cpu().numpy().ravel()
            done += len(chunk)
            if progress:
                progress(done, len(peptides))

    os.makedirs(CACHE, exist_ok=True)
    meta = {
        "molecule": molecule, "alpha": alpha, "beta": beta,
        "background": os.path.basename(_background(molecule)), "n_background": len(peptides),
        "lengths": sorted(set(lengths.tolist())), "n_breaks": N_BREAKS,
        "heads": heads, "host": os.uname().nodename,
        "chain_source": ({c: chains.source(c) for c in (alpha, beta)}
                         if alpha and beta else "custom sequences supplied by the user"),
        "seconds": round(time.time() - started, 1),
        "built_by": "PREpiBind-web/code/grids.py, the on-disk contract of 260905/5_rank/build_grid.py",
        "caveat": ("scores from one head on one host; a query ranked against this grid must be "
                   "embedded and scored the same way"),
    }
    for head in heads:
        grid = {f"{molecule}|{L}": _quantile_breaks(logits[head][lengths == L])
                for L in sorted(set(lengths.tolist()))}
        np.savez_compressed(path(molecule, head), **grid)
        meta["checkpoint_sha256"] = artifacts.manifest(head)["served_sha256"]
        with open(path(molecule, head).replace(".npz", "_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
    return meta
