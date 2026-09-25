"""Locating the binding-domain window in a chain sequence the catalogue does not have.

Every stored HLA embedding — and therefore every %Rank background — was made by embedding the
**full-length** chain through ESM C and slicing the binding domain out of the *embedding*. Embedding
the window on its own is a different object: cosine 0.34 against the stored vectors for
`HLA-DRA*01:01`.

So a custom chain has to be handled the same way, and that needs the window's coordinates in a
sequence nobody has annotated. They are transferable: MHC-II chains are conserved, and
`mhc_mapping.csv` carries `(start_idx, end_idx)` for 7,282 of them. This aligns the custom sequence
to the catalogue chain it most resembles and carries that chain's window across the alignment.

Measured cost of the alternatives, on three panel molecules x 150 labelled peptides, against the
catalogue path (`IMG/docs/notes/2026-09-25-img1s-users-need-not-trim-holds-in-auc-and-fails-in-rank`):

| what the user's sequence becomes | ROC-AUC | mean \\|Δ%Rank\\| | band calls changed |
|---|---|---|---|
| full-length, sliced to the window | −0.002 | 0.45 pp | 2 % |
| full-length, left whole | −0.017 | 3.01 pp | **20 %** |
| the window, embedded on its own | −0.111 | 14.11 pp | 39 % |

The AUC column is why img1 concluded users need not trim, and it is not wrong; the band column is
why img2 cannot repeat that conclusion.
"""
import functools
import re

import pandas as pd

import chains

# Enough of a scan to find the right neighbourhood without aligning against all 7,282 chains: the
# window transfers cleanly only from a chain of the same locus anyway.
CANDIDATES_PER_LOCUS = 40
MIN_IDENTITY = 0.60


def _locus(name):
    m = re.match(r"(HLA-D[PQR][AB]\d?|H2-I[AE])", name)
    return m.group(1) if m else name.split("*")[0]


@functools.lru_cache(maxsize=1)
def _catalogue():
    """[(name, sequence, start, end)] — the annotated chains a window can be carried from."""
    df = pd.read_csv(chains.MAPPING_CSV).dropna(subset=["HLA_Name", "HLA_Seq"])
    return [(r.HLA_Name, r.HLA_Seq, int(r.start_idx), int(r.end_idx)) for r in df.itertuples()]


@functools.lru_cache(maxsize=1)
def _aligner():
    from Bio import Align

    a = Align.PairwiseAligner(scoring="blastp")
    a.mode = "global"
    return a


def _identity(a, b):
    """Cheap pre-filter: fraction of positions equal when the two are laid end to end."""
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / max(len(a), len(b))


def infer_window(sequence, kind=None):
    """(start, end, reference_chain, identity) for a custom full-length chain, or None.

    `kind` is 'alpha' or 'beta' — the interface asks for the two chains separately, so it knows.
    Passing it matters: without it an SLA DR *alpha* chain was matched against `HLA-DPB1*112:01`, a
    DP *beta*, and the window came out [33:101] against a true [26:106].

    Returns None when nothing in the catalogue is close enough to carry a window from. The caller
    must then refuse rather than guess: a misplaced window is not a smaller error than no window.
    """
    sequence = "".join(sequence.split()).upper()
    pool = _catalogue()
    if kind in ("alpha", "beta"):
        allowed = set(chains.chain_lists()[kind])
        pool = [c for c in pool if c[0] in allowed] or pool

    # Rank by the cheap measure, then align only the plausible few.
    ranked = sorted(pool, key=lambda c: -_identity(sequence, c[1]))[:CANDIDATES_PER_LOCUS]
    aligner = _aligner()
    best = None
    for name, ref, start, end in ranked:
        aln = aligner.align(ref, sequence)[0]
        ref_idx, qry_idx = aln.aligned
        span = None
        for (r0, r1), (q0, q1) in zip(ref_idx, qry_idx):
            # Carry the reference window across each aligned block it touches.
            if r1 <= start or r0 >= end:
                continue
            lo = q0 + max(start - r0, 0)
            hi = q1 - max(r1 - end, 0)
            span = (lo, hi) if span is None else (min(span[0], lo), max(span[1], hi))
        if span is None:
            continue
        matched = sum(min(r1, end) - max(r0, start)
                      for (r0, r1), _ in zip(ref_idx, qry_idx) if r1 > start and r0 < end)
        identity = matched / (end - start)
        if best is None or identity > best[3]:
            best = (span[0], span[1], name, identity)

    if best is None or best[3] < MIN_IDENTITY:
        return None
    return best
