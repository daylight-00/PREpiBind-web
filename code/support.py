"""Training-support facts for a molecule. Facts, never a confidence.

`img2-server-output-contract` §5 fixes both what this shows and what it must not imply. Track D
found that nearest-training distance is associated with held-out per-molecule performance, but the
association is **cross-locus** — absent within HLA-DR — and the mapping from distance to expected
performance was **not** validated. So the server reports how well supported a molecule is and stops
there: no reliability tier, no traffic light, no expected-AUC band, and the word reliability does
not appear.

The molecule vector is the one Track D fixed and nothing else: the mean over residue positions of
the concatenated sliced (beta, alpha) embedding the head pools, L2-normalised. The percentile is
taken against a frozen reference population — the nearest-training distance of every one of the 306
%Rank panel molecules against the same training set — so it cannot drift with the query.

Raw cosine distances are deliberately not exposed: on the LOMO panel they span 0.00001 to 0.00557,
which is a percentile, not a magnitude, to any reader.
"""
import os

import numpy as np

import chains

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPORT = os.path.join(ROOT, "data", "support")
DISCLAIMER = "Describes similarity to training molecules; not a confidence estimate."

# `img2-validated-length-range`: trained 12-25, reportable 12-21, 22-25 accepted but annotated.
VALIDATED = (12, 21)
TRAINED = (12, 25)

_CACHE = {}


def available(head="ms"):
    return os.path.isfile(os.path.join(SUPPORT, f"{head}_support.npz"))


def _load(head="ms"):
    if head not in _CACHE:
        z = np.load(os.path.join(SUPPORT, f"{head}_support.npz"), allow_pickle=True)
        _CACHE[head] = {
            "molecules": list(z["train_molecules"]),
            "vectors": z["train_vectors"].astype(np.float32),
            "counts": {m: int(c) for m, c in zip(z["train_molecules"], z["train_counts"])},
            "reference": z["reference_distances"].astype(np.float64),
        }
    return _CACHE[head]


def _ordinal(p):
    n = int(round(p))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def length_support(n):
    """One of 'validated', 'limited-validation', 'outside' — never a quality judgement."""
    if VALIDATED[0] <= n <= VALIDATED[1]:
        return "validated"
    if TRAINED[0] <= n <= TRAINED[1]:
        return "limited-validation"
    return "outside"


def _vector(alpha, beta):
    v = np.asarray(chains.molecule_embedding(alpha, beta), dtype=np.float64).mean(0)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n else None


def describe(alpha, beta, head="ms"):
    """{seen, ligands, nearest, percentile} for one molecule, or None if the artifact is absent.

    `percentile` is the share of the reference population at least as close to training as this
    molecule — smaller means better supported. It is a **support** percentile.
    """
    if not available(head):
        return None
    d = _load(head)
    molecule = chains.molecule(alpha, beta)
    seen = molecule in d["counts"]
    try:
        v = _vector(alpha, beta)
    except Exception:
        v = None
    if v is None:
        return {"molecule": molecule, "seen": seen,
                "ligands": d["counts"].get(molecule, 0), "nearest": None, "percentile": None}
    sims = d["vectors"] @ v
    i = int(np.argmax(sims))
    dist = max(0.0, float(1.0 - sims[i]))
    ref = d["reference"]
    pct = float(np.searchsorted(ref, dist, side="right") * 100.0 / len(ref))
    # A training molecule is at distance zero and therefore tied with every other training
    # molecule, so its percentile is an artefact of how many of them there are, not a statement
    # about support. The contract says the ligand count is the informative field there.
    return {"molecule": molecule, "seen": seen, "ligands": d["counts"].get(molecule, 0),
            "nearest": None if seen else d["molecules"][i],
            "percentile": None if seen else round(pct, 1)}


def rows(alpha, beta, lengths=(), head="ms"):
    """The block as label/value pairs, in the order the contract lists them."""
    s = describe(alpha, beta, head)
    if s is None:
        return []
    out = [("Seen in training", "Yes" if s["seen"] else "No")]
    if s["seen"]:
        out.append(("Training ligands for this molecule", f"{s['ligands']:,}"))
    if s["nearest"] is not None:
        out.append(("Nearest training molecule", s["nearest"]))
        out.append(("Relative training distance",
                    f"{_ordinal(s['percentile'])} percentile among catalogue molecules"))
    if lengths:
        bands = sorted({length_support(int(n)) for n in lengths})
        out.append(("Length support", ", ".join(bands)))
    return out
