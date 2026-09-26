"""Is an uploaded evaluation set actually independent of what the served heads were fitted on?

This exists because the server's own built-in test sets were not. `test_ms.csv` overlaps 80.7% of
its POSITIVES against ~0% of its negatives — an asymmetry that inflates an AUC rather than merely
leaking, and one that no amount of reading the file would have revealed. img2's heads are a single
fit on the whole development pool, so an img1-era split is not held out from them.

Scope, and the page must say so: this audits an upload against **PREpiBind img2's** training
lineage and nothing else. Calling it a benchmark auditor would claim knowledge of other predictors'
training data, which we do not have.

Nothing here is a verdict. Overlap is reported as strata and the metric is recomputed on each, so
the reader sees how much of a number the overlap was carrying. A set with no overlap is not
thereby a good benchmark, and a set with overlap is not thereby worthless — it is the *gap between
the strata* that carries the information.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINEAGE = os.path.join(ROOT, "data", "lineage")

VALIDATED_MAX = 21
VALIDATED_MIN = 12

_cache = {}


def available():
    return os.path.isfile(os.path.join(LINEAGE, "lineage.json"))


def _load(head, name):
    """One lineage set, for ONE head.

    Pooling the three heads answers a different question. Measured: `test_ms.csv`'s negatives show
    36.6 % peptide+MHC overlap against the pooled lineage and **0.0 %** against the MS head's own,
    because the IC50 arms contain them. An MS benchmark must be audited against the MS head.
    """
    key = f"{head}__{name}"
    if key not in _cache:
        with open(os.path.join(LINEAGE, f"{key}.txt")) as f:
            _cache[key] = frozenset(f.read().split("\n"))
    return _cache[key]


def manifest():
    if "manifest" not in _cache:
        with open(os.path.join(LINEAGE, "lineage.json")) as f:
            _cache["manifest"] = json.load(f)
    return _cache["manifest"]


def _nmers(peptide, k=9):
    return {peptide[i:i + k] for i in range(len(peptide) - k + 1)}


def annotate(df, head="ms"):
    """Add one independence column per overlap kind. `df` needs Epitope and molecule.

    `molecule` is the `<beta>_<alpha>` key; `chains.molecule()` produces it.
    """
    peptides = _load(head, "peptides")
    pmhc = _load(head, "pmhc")
    nmers = _load(head, "nmers9")
    pos, neg = _load(head, "pmhc_pos"), _load(head, "pmhc_neg")
    seen_mol = set(manifest()["molecules"].get(head, {}))

    out = df.copy()
    out["seen_peptide"] = out.Epitope.isin(peptides)
    key = out.Epitope + "|" + out.molecule
    out["seen_pmhc"] = key.isin(pmhc)
    # which side of the training labels a row was seen on. A test positive that was a training
    # POSITIVE is a memory test; one that was a training DECOY is a contradiction, and the two
    # must not be summed into one "overlap" number.
    out["seen_as_positive"] = key.isin(pos)
    out["seen_as_negative"] = key.isin(neg)
    # a shared 9-mer is the weakest kind of overlap and the one nobody checks; at MHC-II lengths
    # a 15-mer contributes seven of them, so this fires often and is reported, never used to reject
    out["shares_9mer"] = [bool(_nmers(p) & nmers) for p in out.Epitope]
    out["seen_molecule"] = out.molecule.isin(seen_mol)
    n = out.Epitope.str.len()
    out["length_support"] = np.where((n >= VALIDATED_MIN) & (n <= VALIDATED_MAX),
                                     "validated", "outside")
    return out


#: progressively stricter views of the same upload. Each drops one more kind of contact with the
#: training lineage; the last two are not strata of the first but orthogonal cuts, so they are
#: labelled rather than ordered.
STRATA = [
    ("Full set", lambda d: d.index),
    ("No exact peptide+MHC", lambda d: d.index[~d.seen_pmhc]),
    ("No exact peptide", lambda d: d.index[~d.seen_peptide]),
    ("No shared 9-mer", lambda d: d.index[~d.shares_9mer]),
    ("Unseen molecules only", lambda d: d.index[~d.seen_molecule]),
    ("Validated length only", lambda d: d.index[d.length_support == "validated"]),
]


def composition(annotated):
    """The overlap table, as shares of the upload."""
    n = len(annotated)
    if not n:
        return {}
    return {
        "rows": n,
        "Exact peptide seen in training": float(annotated.seen_peptide.mean()),
        "Exact peptide + MHC seen": float(annotated.seen_pmhc.mean()),
        "Shares a 9-mer with training": float(annotated.shares_9mer.mean()),
        "Molecule seen in training": float(annotated.seen_molecule.mean()),
        "Outside the validated length range": float(
            (annotated.length_support == "outside").mean()),
    }


def asymmetry(annotated, target="Target"):
    """Overlap split by label — the defect that made this page necessary.

    A set whose positives overlap and whose negatives do not will report an AUC that is partly a
    memory test. Reporting one pooled overlap rate hides exactly that.
    """
    if target not in annotated:
        return None
    rows = []
    for kind in ("seen_peptide", "seen_pmhc", "shares_9mer"):
        pos = annotated[annotated[target] == 1][kind]
        neg = annotated[annotated[target] == 0][kind]
        rows.append({
            "overlap": kind,
            "positives": float(pos.mean()) if len(pos) else float("nan"),
            "negatives": float(neg.mean()) if len(neg) else float("nan"),
            "gap": (float(pos.mean()) - float(neg.mean())) if len(pos) and len(neg)
                   else float("nan"),
        })
    return pd.DataFrame(rows)


def by_stratum(annotated, score, target="Target", min_rows=30):
    """ROC-AUC and AP on each stratum, with the count that produced them.

    A stratum with one class present has no AUC; it is reported as such rather than dropped, so a
    reader can see that the cleanest view of their set is also the emptiest.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for name, sel in STRATA:
        idx = sel(annotated)
        sub = annotated.loc[idx]
        y = sub[target].values
        s = np.asarray(score)[annotated.index.get_indexer(idx)]
        ok = ~np.isnan(s)
        y, s = y[ok], s[ok]
        rec = {"stratum": name, "rows": len(y),
               "positives": int((y == 1).sum()), "negatives": int((y == 0).sum())}
        if len(y) >= min_rows and 0 < rec["positives"] < len(y):
            rec["roc_auc"] = float(roc_auc_score(y, s))
            rec["average_precision"] = float(average_precision_score(y, s))
        else:
            rec["roc_auc"] = float("nan")
            rec["average_precision"] = float("nan")
            rec["note"] = ("fewer than %d usable rows" % min_rows if len(y) < min_rows
                           else "only one class present")
        rows.append(rec)
    return pd.DataFrame(rows)
