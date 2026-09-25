"""Which sequence — and which embedding — the server uses for an MHC-II chain.

`IMG/docs/decisions/pair-source-chains-are-the-chain-table.md`, extended to the webserver on
2026-09-25, splits the 7,282 offered chains in two:

  * the **134 the chain table defines** → the table wins, and the embedding comes from the training
    store `emb_hla_chain_table_0329.h5` (full length, sliced by the table's own indices). These are
    the chains the 306-molecule %Rank panel is composed from, and its grids were built from exactly
    that store.
  * **the other ~7,148** → `mhc_mapping.csv` stays the source, with its pre-sliced per-chain
    embedding. They have no table entry and no grid; under `img2-server-output-contract.md` §3 a
    grid is built for them on first request, from whatever sequence the server used — self-consistent
    by construction.

The invariant is per MOLECULE, not global: the sequence that produced a query embedding and the
sequence behind that molecule's %Rank grid must be the same one. Mixing them puts a score from one
lineage onto a background built from another.

Six of the 134 differ between the two sources and every one of them resolves to the table here:
`H2-IAdA` / `H2-IAdB` (the confirmed α/β swap, fixed upstream 2026-09-09 and stale only in
`mhc_mapping.csv`), `H2-IAg7A`, and `HLA-DQB1*03:01` / `*05:03` / `*06:01` (an 8-residue cytoplasmic
tail, outside every window). The remaining 128 agree over the sliced region to fp16 rounding.
"""
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TABLE_CSV = os.path.join(ROOT, "data", "chain_table.csv")
MAPPING_CSV = os.path.join(ROOT, "data", "mhc_mapping.csv")
TABLE_H5 = os.path.join(ROOT, "data", "emb_hla_chain_table_0329.h5")
MAPPING_EMB = os.path.join(ROOT, "data", "emb_hla_esmc_small_0601_fp16")

_table = None
_mapping = None
_h5 = None


def _load_table():
    """{chain: (sequence, start, end)} for the 134-ish chains the training table defines.

    `HLA_Seq` is `<sequence>|<start>|<end>`; the indices cut the binding-domain window out of the
    full chain, and the same cut is applied to the stored full-length embedding.
    """
    global _table
    if _table is None:
        df = pd.read_csv(TABLE_CSV)
        out = {}
        for name, raw in zip(df.HLA_Name, df.HLA_Seq):
            parts = str(raw).split("|")
            seq = parts[0]
            a, b = (int(parts[1]), int(parts[2])) if len(parts) == 3 else (None, None)
            out[name] = (seq, a, b)
        _table = out
    return _table


def _load_mapping():
    global _mapping
    if _mapping is None:
        df = pd.read_csv(MAPPING_CSV).dropna(subset=["HLA_Name", "HLA_Seq"])
        _mapping = {
            r.HLA_Name: (r.HLA_Seq, int(r.start_idx), int(r.end_idx))
            for r in df.itertuples()
        }
    return _mapping


def _store():
    global _h5
    if _h5 is None:
        import h5py

        _h5 = h5py.File(TABLE_H5, "r")
    return _h5


def source(name):
    """'table' for the chains the training table defines, 'mapping' for the rest."""
    return "table" if name in _load_table() else "mapping"


def known(name):
    return name in _load_table() or name in _load_mapping()


def sequence(name, legacy=False):
    """The binding-domain window the server scores, from whichever source owns this chain.

    `legacy=True` forces `mhc_mapping.csv` even for the 134 the table defines. That is only correct
    for the img1 checkpoints, which were fitted before the 2026-09-09 H2 chain fix: for *them* the
    stale copy is the self-consistent one, and using the corrected table would neither reproduce the
    img1-era server nor match what those weights were trained on.
    """
    if not legacy and name in _load_table():
        seq, a, b = _load_table()[name]
        return seq[a:b] if a is not None else seq
    seq, a, b = _load_mapping()[name]
    return seq[a:b]


def embedding(name, legacy=False):
    """(L, 960) float32 for one chain, sliced as in training. See `sequence` for `legacy`."""
    if not legacy and name in _load_table():
        _, a, b = _load_table()[name]
        emb = np.squeeze(_store()[name][()])
        return (emb[a:b] if a is not None else emb).astype(np.float32)
    return np.load(os.path.join(MAPPING_EMB, f"{name}.npy")).astype(np.float32)


def molecule(alpha, beta):
    """The molecule key, `<beta>_<alpha>` — the dataset key order the %Rank grids are keyed by.

    The order is a naming convention only. The head has no positional encoding over the
    concatenated chains and ends in a masked mean, so it is permutation invariant: measured
    |Δlogit| 3e-8 between the two concatenation orders on the served MS head.
    """
    return f"{beta}_{alpha}"


def molecule_embedding(alpha, beta, legacy=False):
    return np.concatenate([embedding(beta, legacy), embedding(alpha, legacy)], axis=0)


def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def chain_lists():
    """{'alpha': [...], 'beta': [...]} over the chains the server offers, naturally sorted.

    The catalogue is `mhc_mapping.csv` — the 7,282 human HLA-DP/DQ/DR and mouse H2 chains. The chain
    table is a *sequence source* for the 134 it defines, not a second catalogue: it also carries
    BoLA, SLA and Mamu chains that this server does not offer and has no background for.

    A class-II beta chain is named by the locus gene, `…DPB1`, `…DQB1`, `…DRB1/3/4/5`, or a mouse
    `…B` suffix. Matching a bare capital `B` anywhere in the name would classify `BoLA-DRA` as a beta
    chain, which is how this was first written and why it is now anchored to the gene.
    """
    names = set(_load_mapping())
    is_beta = re.compile(r"(D[PQR]B\d|-I[AE]\w*B$|B\d\*)").search
    beta = sorted((n for n in names if is_beta(n)), key=_natural_key)
    alpha = sorted((n for n in names if not is_beta(n)), key=_natural_key)
    return {"alpha": alpha, "beta": beta}
