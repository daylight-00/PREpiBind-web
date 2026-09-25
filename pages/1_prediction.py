"""Peptide x MHC-II prediction: %Rank per head, with the strong / weak bands.

The number this page reports is a **percentile against a fixed background**, not a probability. The
three heads are shown together because that is what a percentile buys: the same raw logit of 2.0 at
length 15 is 3.30 % on MS and 0.32 % on IC50 <500 nM, so the columns only become comparable once
they are ranks.
"""
import numpy as np
import pandas as pd
import streamlit as st

import artifacts
import chains
import rank as rankmod
from app import get_bands, get_chain_lists, get_engine, write_st_end

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 12, 25          # accepted
VALIDATED_LEN = (12, 21)           # img2-validated-length-range.md
MAX_MOLECULES = 10
MAX_PEPTIDES = 2000
# A background build is minutes of exclusive GPU on the serving card: 73 s for the shared
# embedding pass over the 100,000-peptide background, then 77 s per head in fp32. The cap keeps
# one user working through a chain list from pinning the card (output-contract §3).
BUILD_MINUTES = 5
BUILD_CAP_PER_SESSION = 3

st.title("Prediction")

lists = get_chain_lists()
bands = get_bands()

# ---------------------------------------------------------------- molecule selection
st.subheader("MHC class II molecule")

loci = ["HLA-DR", "HLA-DP", "HLA-DQ", "H2"]
locus = st.radio("Locus", loci, horizontal=True, label_visibility="collapsed")


def _of_locus(names):
    return [n for n in names if n.startswith(locus)]


alpha_options = _of_locus(lists["alpha"])
beta_options = _of_locus(lists["beta"])

col_a, col_b = st.columns([1, 2])
alpha = col_a.selectbox("α chain", alpha_options,
                        index=alpha_options.index("HLA-DRA*01:01")
                        if "HLA-DRA*01:01" in alpha_options else 0)
default_beta = [b for b in ("HLA-DRB1*15:01",) if b in beta_options]
betas = col_b.multiselect(f"β chain (up to {MAX_MOLECULES})", beta_options,
                          default=default_beta, max_selections=MAX_MOLECULES)

# Which of the chosen molecules have a precomputed background. Off-panel is the common case: the
# two chain lists compose ~1.36 million molecules and the panel is 306 per head, so this is stated
# before a run rather than discovered in the results.
selected = [chains.molecule(alpha, b) for b in betas]
missing = [(b, m) for b, m in zip(betas, selected)
           if not rankmod.has_background(artifacts.HEADLINE, m)]
if missing:
    st.warning(
        f"**{len(missing)} of {len(selected)} selected molecules have no %Rank background yet.** "
        "They can be scored, but the percentile is what makes a score mean anything across "
        "molecules, lengths and heads, so the server leaves it blank rather than showing a bare "
        "logit in its place.\n\n"
        + ", ".join(f"`{m}`" for _, m in missing[:6]) + (" …" if len(missing) > 6 else "")
    )
    built_here = st.session_state.setdefault("built", 0)
    # A per-session cap, because the build is minutes of exclusive GPU and one user working
    # through a chain list would otherwise pin the card for everyone.
    if built_here >= BUILD_CAP_PER_SESSION:
        st.error(f"This session has already built {built_here} backgrounds, which is the cap.")
    elif st.button(f"Build {len(missing)} background(s) — about "
                   f"{len(missing) * BUILD_MINUTES:.0f} min of exclusive GPU", key="build_bg"):
        import grids

        bar = st.progress(0.0)
        for i, (beta_i, mol) in enumerate(missing):
            if st.session_state["built"] >= BUILD_CAP_PER_SESSION:
                st.error("Per-session cap reached; the remaining molecules were not built.")
                break
            bar.progress(i / len(missing), text=f"{mol} — {i + 1} of {len(missing)}")
            with grids.GPU:
                grids.build(get_engine(), alpha, beta_i,
                            progress=lambda d, t, i=i: bar.progress(
                                (i + d / t) / len(missing), text=f"{mol} — {i + 1} of {len(missing)}"))
            for head in artifacts.HEADS:
                rankmod.invalidate(head, mol)
            st.session_state["built"] += 1
        bar.empty()
        st.rerun()

# ---------------------------------------------------------------- peptides
st.subheader("Peptides")
st.caption(
    f"One per line, {MIN_LEN}–{MAX_LEN} residues, standard amino acids. "
    f"Validated range is {VALIDATED_LEN[0]}–{VALIDATED_LEN[1]}; "
    f"{VALIDATED_LEN[1] + 1}–{MAX_LEN} is scored but carries limited validation support."
)

tab_text, tab_csv = st.tabs(["Paste", "Upload CSV"])
with tab_text:
    raw = st.text_area("Peptides", height=160, label_visibility="collapsed",
                       placeholder="GELIGILNAAKVPAD\nPKYVKQNTLKLATAA")
with tab_csv:
    st.caption("Columns `MHC_alpha`, `MHC_beta`, `Epitope`. Overrides the selection above.")
    csv_file = st.file_uploader("CSV", type="csv", label_visibility="collapsed")


def build_input():
    """The frame to score, or (None, message). CSV wins when both are given."""
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        missing = {"MHC_alpha", "MHC_beta", "Epitope"} - set(df.columns)
        if missing:
            return None, f"CSV is missing {', '.join(sorted(missing))}."
        return df[["MHC_alpha", "MHC_beta", "Epitope"]].copy(), None

    peptides = [p.strip().upper() for p in raw.splitlines() if p.strip()]
    if not peptides:
        return None, "Enter at least one peptide."
    if not betas:
        return None, "Select at least one β chain."
    return pd.DataFrame(
        [{"MHC_alpha": alpha, "MHC_beta": b, "Epitope": p} for b in betas for p in peptides]
    ), None


def validate(df):
    """Errors that stop a run, and warnings that do not."""
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

    extended = int(((length > VALIDATED_LEN[1]) & (length <= MAX_LEN)).sum())
    if extended:
        warnings.append(f"{extended} peptides are longer than {VALIDATED_LEN[1]} residues. They are "
                        "scored, but the training pool holds almost no negatives at those lengths, "
                        "so their support is weak. No pooled performance figure covers them.")
    return errors, warnings


# ---------------------------------------------------------------- run
if st.button("Run prediction", type="primary"):
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
            bar = st.progress(0.0, text="Embedding peptides…")
            engine = get_engine()
            result = engine.predict(
                df, progress=lambda done, total: bar.progress(min(done / total, 1.0)))
            bar.empty()
            st.session_state["result"] = result

# ---------------------------------------------------------------- results
result = st.session_state.get("result")
if result is not None:
    st.markdown("## Results")

    res = rankmod.resolution(artifacts.HEADLINE)
    table = pd.DataFrame({
        "Peptide": result.Epitope,
        "α": result.MHC_alpha,
        "β": result.MHC_beta,
        "Length": result.length,
    })
    for head, label in artifacts.HEADS.items():
        table[f"{label} %Rank"] = [rankmod.format_pct(v, res) for v in result[f"{head}_rank_pct"]]
        table[f"{label} call"] = result[f"{head}_band"].fillna("")

    headline_pct = result[f"{artifacts.HEADLINE}_rank_pct"]
    st.dataframe(table.sort_values(
        f"{artifacts.HEADS[artifacts.HEADLINE]} %Rank",
        key=lambda c: headline_pct.reindex(c.index)), width="stretch", hide_index=True)

    cut = bands["bands"]
    st.caption(
        f"**%Rank** is the percentage of a fixed 100,000-peptide background, drawn from the "
        f"reviewed proteome and stratified by (molecule, length), that scores at least as high. "
        f"Lower is a stronger binder, and `<{res:g}` means the grid cannot resolve further. "
        f"**Strong** is ≤ {cut['strong_rank_pct_max']:g} %, **Weak** ≤ {cut['weak_rank_pct_max']:g} %, "
        "matching NetMHCIIpan-4.3's installed defaults so the two are directly comparable."
    )
    with st.expander("What the bands were measured to do"):
        ev = bands["evidence"]
        st.markdown(
            f"On the development pool ({ev['n_rows']:,} rows, lengths 12–21):\n\n"
            f"| band | ligands recovered | negatives passed |\n|---|---|---|\n"
            f"| Strong ≤ {cut['strong_rank_pct_max']:g} % | {ev['strong']['sensitivity']:.1%} | "
            f"{ev['strong']['negative_rate']:.1%} |\n"
            f"| Weak ≤ {cut['weak_rank_pct_max']:g} % | {ev['weak']['sensitivity']:.1%} | "
            f"{ev['weak']['negative_rate']:.1%} |\n\n"
            f"**The negatives here are matched decoys, 1:1 with the positives — not a natural "
            f"proteome.** The passed-negative rate is therefore not a specificity, and must not be "
            f"quoted as one. {bands['not_a_calibration']}"
        )

    if not result.on_panel.all():
        n = int((~result.on_panel).sum())
        st.info(f"{n} of {len(result)} rows are on a molecule with no background, so their %Rank "
                "is blank. The raw logit is in the download.")

    st.download_button("Download results (CSV)", result.to_csv(index=False).encode(),
                       "prepibind_prediction.csv", "text/csv")

write_st_end()
