"""Benchmarking a labelled set: discrimination, reported the way img2 aggregates it.

The img1 page reported F1, Precision and MCC at sigmoid ≥ 0.5. **None of those three survive the
reboot**, and not because they are unfashionable: img2 has no probability threshold to compute them
at. Its binary call is a **%Rank cut** — strong ≤ 1 %, weak ≤ 5 %, read from `data/BANDS.json`
(`img2-server-output-contract.md` §2) — so 0.5 on a sigmoid is not an operating point this server
has, and a metric quoted at it describes a decision rule nobody runs.

What replaces them:

  * **ROC-AUC and PR-AUC**, which need no threshold at all, and
  * **sensitivity and the positive-call rate at the two bands**, which are the operating points the
    server actually uses. They are computed on the uploaded set and are that set's numbers — the
    figures in `BANDS.json` were measured on the development pool against 1:1 matched decoys and are
    not reproduced here.

The other change is the aggregation unit. `IMG/docs/decisions/aggregation-rule.md` §3 makes the
allele the statistical unit, and the reason is visible in the frozen results: for the IC50 heads the
*pooled* memorisation null is 0.678 while the molecule-wise null is 0.5000, so a pooled 0.81 is not
the evidence it looks like (`260905/7_final/REPORTABLE.md`, Appendix). This page therefore reports
the molecule-wise distribution beside the pooled figure, and the pooled figure is never alone.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score,
                             roc_curve)

import artifacts
import chains
import independence
from app import get_bands, get_engine, write_st_end

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 12, 25          # accepted
VALIDATED_LEN = (12, 21)           # img2-validated-length-range.md
MAX_PEPTIDES = 2000                # same interactive limit as the prediction page
# A molecule enters the molecule-wise table only with at least this many of each class, which is the
# rule the frozen allele-wise table was built with (REPORTABLE.md, "Allele-wise").
MIN_PER_CLASS = 5

# Measured once, 2026-09-25, on `abc`, against the pools the served heads were fit on:
# `260905/1_data/ms_cap40k_decoy1.csv` (MS) and `260905/1_data/ic50_{500,1000}_pos.csv`. Static on
# purpose — the page must not recompute it per session, and the answer does not change.
BUILTIN_OVERLAP = [
    ("`test_ms.csv`", "33,490", "MS", "34.7 %", "31.7 %", "80.7 %"),
    ("`test.csv`", "48,352", "MS", "24.1 %", "21.8 %", "37.4 %"),
    ("`test_ic50_500.csv`", "14,150", "IC50 < 500 nM", "100.0 %", "99.8 %", "99.5 %"),
    ("`test_ic50_1000.csv`", "14,150", "IC50 < 1000 nM", "100.0 %", "99.7 %", "99.5 %"),
]

st.title("Evaluation")
st.caption(
    "Upload a labelled set and the server scores it with one head and reports how well that head "
    "separates the two classes. It reports discrimination only: this page does not compare "
    "PREpiBind with any other tool, and no such comparison can be drawn from a set chosen here."
)

bands = get_bands()
cut = bands["bands"]

# ---------------------------------------------------------------- the img1 test sets
# The four CSVs in `data/` are img1-era splits. The img2 heads are a single fit on the WHOLE
# development pool (`app.py` docstring), so an img1 test split is not held out from them — it is
# training data. Stated before an upload box, because this is the failure the page would otherwise
# invite silently.
st.error(
    "**The four test sets shipped in `data/` are img1-era and are not offered here.** The img2 "
    "heads are one fit on the entire development pool, not a cross-validation fold, so those "
    "splits are inside the training pool rather than held out from it. Measured overlap:"
)
st.markdown(
    "| img1 test set | rows | training pool | peptide-level | (peptide, molecule) | positives only |\n"
    "|---|---|---|---|---|---|\n"
    + "".join(f"| {a} | {b} | {c} | {d} | {e} | {f} |\n" for a, b, c, d, e, f in BUILTIN_OVERLAP)
)
st.markdown(
    "Peptide-level is the share of rows whose peptide appears anywhere in that pool; the third "
    "column pairs the peptide with its molecule; the last restricts to the labelled positives, "
    "which is where the memorisation would act. **None of these four sets can measure img2 "
    "generalisation** — for the two IC50 sets essentially every row was trained on. The numbers "
    "that do carry a generalisation claim are the held-out study and held-out molecule panels:"
)
st.page_link("pages/4_about.py", label="About — the held-out panels and their intervals")
st.caption(
    "Overlap measured 2026-09-25 against `260905/1_data/ms_cap40k_decoy1.csv` and "
    "`260905/1_data/ic50_{500,1000}_pos.csv` on `abc`. Static figure, not recomputed per session."
)

st.markdown("---")

# ---------------------------------------------------------------- head and data
st.subheader("Head")
head = st.selectbox("Head", list(artifacts.HEADS), format_func=lambda h: artifacts.HEADS[h],
                    index=list(artifacts.HEADS).index(artifacts.HEADLINE),
                    label_visibility="collapsed")

st.subheader("Labelled set")
st.caption(
    f"Columns `MHC_alpha`, `MHC_beta`, `Epitope`, `Target`, where `Target` is 1 for a binder and 0 "
    f"for a non-binder. Up to {MAX_PEPTIDES:,} rows, {MIN_LEN}–{MAX_LEN} residues. Only "
    f"{VALIDATED_LEN[0]}–{VALIDATED_LEN[1]} enters the reported figures: nothing above "
    f"{VALIDATED_LEN[1]} is pooled anywhere in this project."
)
csv_file = st.file_uploader("CSV", type="csv", label_visibility="collapsed")

REQUIRED = ["MHC_alpha", "MHC_beta", "Epitope", "Target"]


def build_input():
    """The labelled frame to score, or (None, message)."""
    if csv_file is None:
        return None, "Upload a CSV with a Target column."
    df = pd.read_csv(csv_file)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        return None, f"CSV is missing {', '.join(sorted(missing))}."
    return df[REQUIRED].copy(), None


def validate(df):
    """Errors that stop a run, and warnings that do not. Same checks as the prediction page, plus
    the label column, which only this page has."""
    errors, warnings = [], []
    bad_aa = df[~df.Epitope.apply(lambda p: set(p) <= VALID_AA)]
    if not bad_aa.empty:
        errors.append(f"{len(bad_aa)} peptides contain non-standard residues "
                      f"(e.g. `{bad_aa.Epitope.iloc[0]}`).")
    length = df.Epitope.str.len()
    out_of_range = df[(length < MIN_LEN) | (length > MAX_LEN)]
    if not out_of_range.empty:
        errors.append(f"{len(out_of_range)} peptides fall outside {MIN_LEN}–{MAX_LEN} residues.")
    unknown = sorted({c for c in pd.concat([df.MHC_alpha, df.MHC_beta]).unique()
                      if not chains.known(c)})
    if unknown:
        errors.append(f"{len(unknown)} unrecognised chains: {', '.join(unknown[:5])}.")
    if len(df) > MAX_PEPTIDES:
        errors.append(f"{len(df):,} rows exceeds the {MAX_PEPTIDES:,}-row interactive limit.")

    labels = set(pd.to_numeric(df.Target, errors="coerce").dropna().unique())
    if not labels <= {0, 1}:
        errors.append("`Target` must be 1 or 0; other values were found.")
    elif len(labels) < 2:
        errors.append("`Target` holds only one class. Discrimination needs both.")

    extended = int(((length > VALIDATED_LEN[1]) & (length <= MAX_LEN)).sum())
    if extended:
        warnings.append(f"{extended} peptides are longer than {VALIDATED_LEN[1]} residues. They are "
                        "scored and appear in the download, but they are excluded from every figure "
                        "below: no pooled performance number in this project covers them.")
    return errors, warnings


# ---------------------------------------------------------------- run
if st.button("Run evaluation", type="primary"):
    df, message = build_input()
    if df is None:
        st.error(message)
    else:
        errors, warnings = validate(df)
        for w in warnings:
            st.warning(w)
        if errors:
            for e in errors:
                st.error(e)
        else:
            df["Target"] = pd.to_numeric(df.Target).astype(int)
            bar = st.progress(0.0, text="Embedding peptides…")
            engine = get_engine()
            result = engine.predict(
                df, heads=head,
                progress=lambda done, total: bar.progress(min(done / total, 1.0)))
            bar.empty()
            st.session_state["eval_result"] = (head, result)

# ---------------------------------------------------------------- metrics
def molecule_wise(df, score, target):
    """One ROC-AUC per molecule, over the molecules that carry both classes.

    The allele is the statistical unit (`aggregation-rule.md` §3). A molecule with a handful of one
    class gives an AUC that is mostly noise, so it is left out rather than averaged in.
    """
    rows = []
    for molecule, grp in df.groupby("molecule", sort=True):
        y = target[grp.index]
        if int((y == 1).sum()) < MIN_PER_CLASS or int((y == 0).sum()) < MIN_PER_CLASS:
            continue
        rows.append({"Molecule": molecule, "n": len(grp), "positives": int((y == 1).sum()),
                     "ROC-AUC": roc_auc_score(y, score[grp.index])})
    return pd.DataFrame(rows)


def band_operating_points(pct, target):
    """Sensitivity and positive-call rate at the strong and weak cuts, on rows that have a %Rank."""
    rows = []
    for label, key in (("Strong", "strong_rank_pct_max"), ("Weak", "weak_rank_pct_max")):
        called = pct <= cut[key]
        rows.append({
            "Band": f"{label} ≤ {cut[key]:g} %",
            "Sensitivity": f"{called[target == 1].mean():.1%}",
            "Positive-call rate": f"{called.mean():.1%}",
            "Called": f"{int(called.sum()):,} of {len(pct):,}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- results
stored = st.session_state.get("eval_result")
if stored is not None:
    head, result = stored
    label = artifacts.HEADS[head]

    # 22-25 is trained and scored but never pooled (`img2-ms-arm-and-length-range.md` §2). The
    # engine already carries that judgement as a column, so the page reads it rather than
    # re-deriving the range.
    # reset_index: the metric helpers index the numpy score arrays by row label.
    scored = result[result.length_support == "validated"].reset_index(drop=True)
    target = scored.Target.to_numpy()
    # The logit, not the %Rank. ROC-AUC and PR-AUC are invariant to any monotone transform of the
    # score, and within a (molecule, length) stratum the grid makes %Rank a monotone function of the
    # logit — so the two agree there, and the logit has the advantage of being defined for every
    # row, including molecules with no background.
    logit = scored[f"{head}_logit"].to_numpy()

    st.markdown("## Results")
    st.caption(f"{label} head · {len(scored):,} of {len(result):,} rows in "
               f"{VALIDATED_LEN[0]}–{VALIDATED_LEN[1]} residues · "
               f"{int((target == 1).sum()):,} positives, {int((target == 0).sum()):,} negatives · "
               f"{scored.molecule.nunique()} molecules.")

    if len(set(target)) < 2:
        st.error("Both classes are needed within the reported length range, and only one is left.")
        write_st_end()
        st.stop()

    # ------------------------------------------------------------ pooled
    st.subheader("Pooled")
    pooled_roc = roc_auc_score(target, logit)
    pooled_pr = average_precision_score(target, logit)
    col_a, col_b = st.columns(2)
    col_a.metric("ROC-AUC", f"{pooled_roc:.4f}")
    col_b.metric("PR-AUC (average precision)", f"{pooled_pr:.4f}")
    st.caption(
        "Pooling ranks every molecule on one score axis, so between-molecule separation inflates "
        "this figure — it is not the molecule-wise number and must not be quoted as one. PR-AUC "
        "additionally moves with the positive rate of the uploaded set, so it is comparable only "
        "against a set of the same balance."
    )

    # ------------------------------------------------------------ independence
    # The reason this section exists: the server's own built-in MS test set overlaps 80.7% of its
    # POSITIVES against 0.0% of its negatives. A pooled ROC-AUC cannot show that, and reading the
    # file will not either.
    st.subheader("Independence from training")
    if not independence.available():
        st.info("The training-lineage index is not installed, so independence cannot be checked. "
                "Every number on this page should then be read as an upper bound.")
    else:
        audit = independence.annotate(scored, head)
        comp = independence.composition(audit)
        st.caption(
            f"Checked against what the **{label}** head was fitted on — "
            f"{independence.manifest()['heads'][head]['rows']:,} rows, "
            f"{independence.manifest()['heads'][head]['molecules']} molecules. "
            "This is PREpiBind's lineage only; it says nothing about whether your set is "
            "independent of any other predictor."
        )
        cols = st.columns(5)
        cols[0].metric("Exact peptide seen", f"{comp['Exact peptide seen in training']:.1%}")
        cols[1].metric("Peptide + MHC seen", f"{comp['Exact peptide + MHC seen']:.1%}")
        cols[2].metric("9-mer, same molecule",
                       f"{comp['Shares a 9-mer on the same molecule']:.1%}")
        cols[3].metric("9-mer, any molecule", f"{comp['Shares a 9-mer with training']:.1%}")
        cols[4].metric("Molecule seen", f"{comp['Molecule seen in training']:.1%}")
        st.caption(
            "A 9-mer seen against *any* molecule is sequence familiarity. A 9-mer seen against "
            "**the same** molecule is a narrower proxy for it — nine residues is the length of an "
            "MHC-II core, but the core's position is unknown here and the shared stretch may be "
            "flank. It is the more specific measure, not a mechanistic one."
        )

        asym = independence.asymmetry(audit)
        if asym is not None:
            worst = float(asym.gap.abs().max())
            show = asym.rename(columns={"overlap": "Overlap", "positives": "Positives",
                                        "negatives": "Negatives", "gap": "Gap"})
            st.dataframe(show.style.format({"Positives": "{:.1%}", "Negatives": "{:.1%}",
                                            "Gap": "{:+.1%}"}),
                         hide_index=True, width="stretch")
            contact = independence.contact_table(audit)
            if contact is not None:
                tbl, summ = contact
                if summ["overlapping_rows"]:
                    st.markdown("**Where the exact peptide + MHC hits came from**")
                    st.dataframe(tbl, hide_index=True, width="stretch")
                    conflict = summ["conflicting"] / summ["overlapping_rows"]
                    if conflict >= 0.10:
                        st.warning(
                            f"{conflict:.0%} of the overlapping rows carry the **opposite** label "
                            "in training. Those rows do not measure memory, they measure which of "
                            "two disagreeing sources the model follows — a different problem, and "
                            "one a single overlap rate would have hidden."
                        )
            if worst >= 0.10:
                st.warning(
                    f"Overlap differs by label by up to {worst:.0%}. Where that overlap is "
                    "concordant, a metric on this set is partly a memory test: the model can "
                    "score the overlapping class from having seen it. Read the strata below, "
                    "not the pooled figure."
                )
            else:
                st.caption("Overlap is balanced across labels. That does not make the set "
                           "independent — it means the overlap is not, on its own, tilting the "
                           "metric toward one class.")

        strata = independence.by_stratum(
            audit, logit, rank_pct=scored[f"{head}_rank_pct"].to_numpy())
        st.dataframe(
            strata.rename(columns={
                "stratum": "View", "rows": "Rows", "positives": "Pos", "negatives": "Neg",
                "molecules": "Mol", "molecule_mean_auc": "Molecule-wise AUC (mean)",
                "molecule_median_auc": "median", "pooled_auc_rank": "Pooled AUC (−%Rank)",
                "rank_coverage": "%Rank cov.", "pooled_auc_logit": "Pooled AUC (logit)",
                "pooled_ap_logit": "Pooled PR-AUC (logit)"}),
            hide_index=True, width="stretch")
        paired = independence.paired_molecule_delta(audit, logit)
        st.markdown("**Same molecules, before and after**")
        st.dataframe(
            paired.rename(columns={
                "comparison": "Comparison", "molecules_paired": "Molecules paired",
                "mean_delta": "Mean ΔAUC", "median_delta": "median",
                "ci_low": "CI low", "ci_high": "CI high",
                "molecules_lost": "Molecules dropped"}),
            hide_index=True, width="stretch")
        st.caption(
            "This is the comparison the table above cannot make. Removing overlapping rows also "
            "removes whole molecules from the eligible set, and a mean over per-molecule AUCs "
            "moves when that set changes even if no molecule got better — three molecules at "
            "0.9 / 0.8 / 0.5 average 0.733, and losing the 0.5 for want of rows lifts the average "
            "to 0.85 with nothing improved. Pairing fixes the set first, so the delta is what "
            "happened to the **same** molecules; the interval is a bootstrap over molecules. "
            "**Molecules dropped** is information too: a view that is clean because it is empty "
            "has told you something about the upload, not about the model."
        )

        st.caption(
            "**Read the molecule-wise column.** Dropping overlapping rows also changes which "
            "molecules, which lengths and which class balance are left, so a *pooled* figure that "
            "moves between views has moved for two reasons at once — and the composition one says "
            "nothing about independence. An average over per-molecule AUCs does not move when the "
            "molecule mixture *within* a molecule does — but it is still not comparable across "
            "strata, which is what the paired table above is for. The pooled "
            "−%Rank column is comparable across molecules by construction but is defined only "
            "where a background exists, so its coverage is printed; the pooled logit column is the "
            "one the rest of this page reports, kept here as a diagnostic.  \n"
            "The last two views are orthogonal cuts, not a continuation of the series. **The "
            "information is in the gap between views, not in any single one**, and a view that "
            "loses most of the rows loses most of the precision of its estimate."
        )
        st.download_button("Download the independence audit (CSV)",
                           audit.to_csv(index=False).encode(),
                           file_name="independence_audit.csv", mime="text/csv")

    # ------------------------------------------------------------ molecule-wise
    st.subheader("Molecule-wise")
    per_mol = molecule_wise(scored, logit, target)
    if per_mol.empty:
        st.info(f"No molecule in this set carries at least {MIN_PER_CLASS} of each class, so there "
                "is no molecule-wise figure. The pooled number above is then the only one, and it "
                "is the weaker of the two.")
    else:
        a = per_mol["ROC-AUC"]
        cols = st.columns(3)
        cols[0].metric("Molecules", f"{len(per_mol)}")
        cols[1].metric("Mean ROC-AUC", f"{a.mean():.4f}")
        cols[2].metric("Median", f"{a.median():.4f}")
        st.caption(f"IQR {a.quantile(0.25):.4f}–{a.quantile(0.75):.4f}, "
                   f"min {a.min():.4f}, max {a.max():.4f}. One value per molecule with at least "
                   f"{MIN_PER_CLASS} positives and {MIN_PER_CLASS} negatives; "
                   f"{scored.molecule.nunique() - len(per_mol)} of "
                   f"{scored.molecule.nunique()} molecules did not qualify.")
        st.dataframe(per_mol.style.format({"ROC-AUC": "{:.4f}"}),
                     width="stretch", hide_index=True)

    # ------------------------------------------------------------ the bands
    st.subheader("At the server's operating points")
    pct = scored[f"{head}_rank_pct"].to_numpy()
    has_rank = ~np.isnan(pct)
    if not has_rank.any():
        st.warning("No row in this set is on a molecule with a %Rank background, so the server "
                   "makes no binary call on any of them and there is no operating point to report. "
                   "The AUCs above are unaffected: they are computed from the logit.")
    else:
        st.dataframe(band_operating_points(pct[has_rank], target[has_rank]),
                     width="stretch", hide_index=True)
        st.caption(
            f"**Measured on the set you uploaded**, not the figures in `BANDS.json` — those were "
            f"measured on the development pool against 1:1 matched decoys and are shown on the "
            f"prediction page. Sensitivity is over this set's positives; the positive-call rate is "
            f"over all its rows, so it moves with the set's class balance and is not a specificity."
        )
        if (~has_rank).any():
            st.info(f"{int((~has_rank).sum()):,} of {len(scored):,} rows sit on a molecule with no "
                    "%Rank background, so they carry no band and are excluded from this table only. "
                    "Build their background on the prediction page if the operating point matters.")

    # ------------------------------------------------------------ curves
    fpr, tpr, _ = roc_curve(target, logit)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC-AUC = {pooled_roc:.3f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(dash="dash", width=1)))
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                      height=380, margin=dict(t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    precision, recall, _ = precision_recall_curve(target, logit)
    prevalence = float((target == 1).mean())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                             name=f"PR-AUC = {pooled_pr:.3f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[prevalence, prevalence], mode="lines",
                            name=f"Chance = {prevalence:.3f}", line=dict(dash="dash", width=1)))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                      height=380, margin=dict(t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Both curves are pooled over molecules and carry the same inflation as the pooled "
               "figures above.")

    st.download_button("Download scored rows (CSV)", result.to_csv(index=False).encode(),
                       "prepibind_evaluation.csv", "text/csv")

write_st_end()
