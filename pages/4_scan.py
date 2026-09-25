"""Antigen scanning: one protein x a panel of MHC-II molecules -> a binding landscape.

Every window of every length is embedded as its **own peptide** and scored. That is not an
implementation preference: a window sliced out of a whole-protein embedding disagrees with the real
thing at Spearman 0.42 and 2/20 top-20 overlap (`code/scan.py` §1), so there is no shortcut and the
cost is linear in antigen length and in panel size. The page says what a scan will cost before it
runs it.

What a candidate region is, is **decided, not chosen here**:
`IMG/docs/decisions/img2-scan-candidate-rule.md` fixed the criterion — coverage admissibility first,
then lift at the matched operating point — before the deciding measurement existed, and on
2026-09-25 the `position` rule (`scan.collapse_positions`) became the first and only rule to clear
it, displacing `span`. §7 of that decision says the server may choose only among the rules it admits
and may neither introduce a fifth rule nor relax the criterion. So this page offers one rule, and
the one knob it exposes — the coverage budget — is capped at the admissibility bound.

The three things the scan was measured to do are reported as **three separate results** and are
never merged into one accuracy number. The localisation result carries its own caveat in the same
block, because that caveat is the part a reader would otherwise walk away without.
"""
import numpy as np
import pandas as pd
import streamlit as st

import artifacts
import chains
import rank as rankmod
import scan
from app import get_bands, get_chain_lists, get_engine, write_st_end

# An interactive budget, not a model limit: the scan embeds every window of every length, so its
# cost is (antigen length) x (lengths) peptides through ESMC and that many rows x molecules through
# the head. These are the figures the scan was prototyped against (IMG/pipeline/scan/app_scan.py).
MAX_AA = 1200
MAX_MOLECULES = 10

VALIDATED_LENGTHS = (12, 21)       # scan.DEFAULT_LENGTHS, img2-validated-length-range.md
EXTENDED_MAX = 25                  # scan.EXTENDED_LENGTHS: trained, limited validation support

# The `position` rule's budget is a fraction of the antigen's residues. 0.18 is the canonical
# operating point the decision selected on (§3.2: achieved coverage 0.1730, lift 1.839), so it is
# the default here — a page that quotes a lift should run at the point that lift was measured at.
# 0.20 is §3.1's admissibility bound: above it the selected region is no longer a candidate region,
# whatever its lift, so the slider stops there rather than letting the UI relax the criterion.
BUDGET_DEFAULT = 0.18
BUDGET_MAX = 0.20

st.title("Antigen scanning")
st.caption(
    "Every window of every length is embedded as its own peptide and scored against each selected "
    "molecule. Candidate regions are then selected **by position** under a coverage budget: each "
    "residue takes the best score of any window covering it, and the best residues within the "
    "budget are kept. This is the rule `img2-scan-candidate-rule.md` adopted; the retired `span`, "
    "`best` and `peak` rules are not offered."
)

with st.expander("What the scan was measured to do — three separate results"):
    st.markdown(
        "Canonical validation, 3,707 locked eluted ligands over 476 protein–allele pairs, "
        "lengths 12–21, MS head. **These are three results. They do not combine into one accuracy "
        "number, and quoting them as one would misstate all three.**\n\n"
        "**1. Window discrimination** — mean AUC **0.6479** over the 476 pairs, 88.4 % of them "
        "above 0.5. This is where the scan carries real signal: it orders windows within a protein.\n\n"
        "**2. Candidate-region localisation** — at the adopted 0.18 budget, mean coverage "
        "**0.1730** of the protein and recovery lift **1.839** over the position-matched null.\n\n"
        "> **The caveat travels with that number.** At the 264 of 3,707 ligands sitting at "
        "> positions the model never saw in training, `position` has lift **0.955** — no lift at "
        "> all, slightly below its own null — where the retired `span` rule keeps 1.513. The rule "
        "> buys its tight coverage by concentrating on regions resembling those already seen. "
        "> Treat a candidate region on a novel antigen accordingly.\n\n"
        "**3. Strict no-training-epitope control** — mean AUC **0.6416** over the 90 evaluable "
        "pairs, i.e. discrimination survives when every training epitope is removed.\n\n"
        "**Comparison.** At matched coverage img2 localises **worse** than NetMHCIIpan-4.3 — "
        "NetMHCIIpan's lift is 3.152 against img2's 1.953 in the exploratory sweep that measured "
        "both, and this rule does not reach 2.0 anywhere in the same coverage range. "
        "This page makes no comparative localisation claim in img2's favour. The ordering "
        "does invert against peptide-level benchmarks, which is task dependence, not a "
        "contradiction.\n\n"
        # one block, deliberately: the lift figures and the unseen-position caveat are never
        # rendered apart, because a reader who sees only the first half has been misled
        "Lift over the position-matched null across the budget, same validation:\n\n"
        "| budget | achieved coverage | recovery | lift |\n|---|---:|---:|---:|\n"
        "| 0.20 | 0.1913 | 0.3458 | 1.777 |\n"
        "| **0.18** | **0.1730** | 0.3221 | **1.839** |\n"
        "| 0.15 | 0.1444 | 0.2811 | 1.964 |\n"
        "| 0.10 | 0.0941 | 0.2012 | 2.145 |\n"
        "| 0.05 | 0.0311 | 0.0839 | 2.436 |\n\n"
        "A tighter budget lifts more over chance and recovers less. The unseen-position caveat "
        "above applies to every row of this table."
    )

lists = get_chain_lists()
bands = get_bands()

# ---------------------------------------------------------------- antigen
st.subheader("Antigen")

tab_paste, tab_upload = st.tabs(["Paste", "Upload FASTA"])
with tab_paste:
    raw = st.text_area("Antigen", height=140, label_visibility="collapsed",
                       placeholder=">P14136 GFAP_HUMAN\nMERRRITSAARRSYVSSGEMVVGGLAPGRRLGPGTRLSLARM…")
with tab_upload:
    fasta = st.file_uploader("FASTA", type=["fa", "fasta", "txt"], label_visibility="collapsed")

if fasta is not None:
    raw = fasta.getvalue().decode("utf-8", "replace")


def read_fasta(text):
    """(name, sequence) from FASTA or a bare sequence. One antigen per scan — the cost is linear."""
    text = (text or "").strip()
    if not text.startswith(">"):
        return "antigen", "".join(text.split()).upper()
    header, *rest = text.splitlines()
    body = []
    for line in rest:
        if line.startswith(">"):
            break
        body.append(line.strip())
    name = header[1:].strip().split(" ")[0] if header[1:].strip() else "antigen"
    return name, "".join(body).upper()


name, seq = read_fasta(raw)

# ---------------------------------------------------------------- molecules
st.subheader("MHC class II molecules")

loci = ["HLA-DR", "HLA-DP", "HLA-DQ", "H2"]
locus = st.radio("Locus", loci, horizontal=True, label_visibility="collapsed")
alpha_options = [n for n in lists["alpha"] if n.startswith(locus)]
beta_options = [n for n in lists["beta"] if n.startswith(locus)]

col_a, col_b = st.columns([1, 2])
alpha = col_a.selectbox("α chain", alpha_options,
                        index=alpha_options.index("HLA-DRA*01:01")
                        if "HLA-DRA*01:01" in alpha_options else 0)
default_beta = [b for b in ("HLA-DRB1*15:01",) if b in beta_options]
betas = col_b.multiselect(f"β chain (up to {MAX_MOLECULES})", beta_options,
                          default=default_beta, max_selections=MAX_MOLECULES)

head = st.selectbox("Head", list(artifacts.HEADS), format_func=lambda h: artifacts.HEADS[h],
                    index=list(artifacts.HEADS).index(artifacts.HEADLINE))
if head != artifacts.HEADLINE:
    st.caption(f"The scan validation above was run on the "
               f"{artifacts.HEADS[artifacts.HEADLINE].lower()} head. None of its three results "
               f"covers {artifacts.HEADS[head]}.")

molecules = [chains.molecule(alpha, b) for b in betas]
# `has_background`, not `on_panel`: a molecule whose grid was built on demand from the prediction
# page ranks exactly like a panel one, and the pre-run warning must not claim otherwise.
on_panel = [m for m in molecules if rankmod.has_background(head, m)]
off_panel = [m for m in molecules if m not in on_panel]
if off_panel:
    st.warning(
        f"**{len(off_panel)} of {len(molecules)} selected molecules have no %Rank background.** "
        "They are scanned, but their candidate regions are selected and ordered by the raw logit, "
        "which makes the result **relative to this antigen only** — not an absolute call and not a "
        "statement about how this antigen compares to any other, nor about how these molecules "
        "compare to each other. Off-panel and on-panel molecules are therefore reported separately "
        "below.\n\n"
        + ", ".join(f"`{m}`" for m in off_panel[:6]) + (" …" if len(off_panel) > 6 else "")
    )

# ---------------------------------------------------------------- settings
st.subheader("Scan settings")

col_len, col_bud = st.columns([2, 1])
lo, hi = col_len.select_slider("Window lengths", value=VALIDATED_LENGTHS,
                               options=list(range(VALIDATED_LENGTHS[0], EXTENDED_MAX + 1)))
budget = col_bud.slider("Coverage budget", 0.05, BUDGET_MAX, BUDGET_DEFAULT, 0.01,
                        help="Fraction of the antigen's residues kept as candidate. Capped at the "
                             "0.20 admissibility bound.")
lengths = tuple(range(lo, hi + 1))
if hi > VALIDATED_LENGTHS[1]:
    st.caption(f"{VALIDATED_LENGTHS[1] + 1}–{EXTENDED_MAX} is inside the trained range but carries "
               "limited validation support: the pool holds almost no negatives at those lengths, "
               "and no pooled figure above covers them.")

n_windows = scan.window_count(len(seq), lengths) if seq else 0
cost = (f"**Cost:** {n_windows:,} windows × {len(molecules)} molecules = "
        f"{n_windows * len(molecules):,} scored rows. Those {n_windows:,} windows are embedded once "
        "and reused across molecules. "
        if n_windows and molecules else "")
st.caption(
    cost + "Embedding is nearly the whole cost, and it cannot be avoided by embedding the protein "
    "once and slicing it (measured Spearman 0.42 against real per-window embeddings); scoring then "
    f"grows linearly with the number of molecules. Limits: {MAX_AA:,} aa and {MAX_MOLECULES} "
    "molecules — an interactive budget, not a model limit."
)


def validate():
    if not seq:
        return "Paste or upload an antigen sequence."
    if len(seq) > MAX_AA:
        return f"{len(seq):,} aa exceeds the {MAX_AA:,} aa interactive limit."
    if len(seq) < hi:
        return f"{len(seq)} aa is shorter than the longest window ({hi})."
    if not betas:
        return "Select at least one β chain."
    if not chains.known(alpha) or not all(chains.known(b) for b in betas):
        return "One of the selected chains is not recognised."
    return None


# ---------------------------------------------------------------- run
if st.button("Scan antigen", type="primary"):
    message = validate()
    if message:
        st.error(message)
    else:
        windows = scan.generate_windows(seq, lengths)
        if windows.empty:
            st.error("No scoreable window: every window touches a non-standard residue.")
        else:
            engine = get_engine()
            bar = st.progress(0.0, text=f"Embedding {len(windows):,} windows…")
            epi_emb = engine.embed(windows.peptide.tolist(),
                                   progress=lambda d, t: bar.progress(min(d / t, 1.0)))

            # One row per (window, molecule). `Epitope` and `molecule` are the names engine.score
            # reads; `allele` is the name scan.py groups by, and it holds the molecule key
            # `<beta>_<alpha>` that the %Rank grids are keyed by.
            rows = pd.concat([windows.assign(MHC_alpha=alpha, MHC_beta=b,
                                             allele=chains.molecule(alpha, b))
                              for b in betas], ignore_index=True)
            rows["Epitope"] = rows.peptide
            rows["molecule"] = rows.allele

            # NOT scan.score_windows: that takes the six-argument training head
            # (x_hla_s, x_hla_p, x_epi_s, x_epi_p, mask_hla, mask_epi) and the served head is
            # code/model.py's four-argument plm_cat_mean_inf. engine.score applies the same batching
            # discipline — grouped by molecule, then by length, never padding the epitope side — to
            # the served head, and shares the batch size the %Rank grids were built at.
            bar.progress(0.0, text=f"Scoring {len(rows):,} rows…")
            rows["logit"] = engine.score(rows, head, epi_emb,
                                         progress=lambda d, t: bar.progress(min(d / t, 1.0)))
            bar.empty()
            # scan.candidate_table reads a `prob` column. It is computed here and shown nowhere:
            # the server reports %Rank, not a sigmoid probability (img2-server-output-contract §1).
            rows["prob"] = 1.0 / (1.0 + np.exp(-rows.logit.values))

            # %Rank per (molecule, length) against this head's grid, exactly as engine.predict does
            # it. rank.rank returns None for a molecule with no background and those rows keep NaN,
            # which is what scan.ranking_column reads to fall back to the raw logit.
            pct = np.full(len(rows), np.nan)
            for (molecule, length), grp in rows.groupby(["molecule", "length"], sort=False):
                pos = rows.index.get_indexer(grp.index)
                r = rankmod.rank(head, molecule, int(length), rows.logit.values[pos])
                if r is not None:
                    pct[pos] = r
            rows["rank_pct"] = pct

            # Which molecules actually carry a %Rank, rather than which were expected to before the
            # run: the split below decides whether a result is reported as absolute or as relative
            # to this antigen, and that label has to match the numbers it sits above.
            ranked = {m for m, g in rows.groupby("allele") if g.rank_pct.notna().any()}
            st.session_state["scan"] = {
                "name": name, "seq_len": len(seq), "head": head, "budget": budget,
                "lengths": lengths, "scored": rows,
                "on_panel": [m for m in molecules if m in ranked],
                "off_panel": [m for m in molecules if m not in ranked],
                "dropped": scan.window_count(len(seq), lengths) - len(windows),
            }

# ---------------------------------------------------------------- results
state = st.session_state.get("scan")
if state is not None:
    scored, seq_len, head = state["scored"], state["seq_len"], state["head"]
    res = rankmod.resolution(head)

    st.markdown(f"## {state['name']} — {seq_len:,} aa")
    if state["dropped"]:
        st.caption(f"{state['dropped']:,} windows were dropped for containing a non-standard "
                   "residue; no training peptide contained one, so they are not scoreable.")

    def regions(frame):
        """Candidate regions and the per-residue landscape for one set of molecules.

        Called once per panel group. `scan.ranking_column` reads the whole frame, so a frame mixing
        on-panel and off-panel molecules would order off-panel clusters by an all-NaN %Rank column.
        """
        # `score_col` is left at the vendored default (the raw logit). Which of logit and -%Rank
        # scores positions better is measured only outside 0.12-0.22 coverage, and the adopted
        # budget sits inside that gap, so the default stands rather than the page picking a side.
        f = scan.collapse_positions(frame, budget=state["budget"])
        # `core_support` counts, per residue, how many CANDIDATE windows cover it; left to itself it
        # falls back to a top-fraction of windows, which would be a second selection rule on screen.
        # The candidate set is the one `position` selected, and nothing else.
        f["is_candidate"] = f.cluster.notna()
        return f, scan.candidate_table(f), scan.core_support(f, seq_len)

    groups = [(state["on_panel"], True), (state["off_panel"], False)]
    for members, panel in groups:
        present = [m for m in members if m in set(scored.allele)]
        if not present:
            continue
        collapsed, cand, support = regions(scored[scored.allele.isin(present)].reset_index(drop=True))

        st.markdown(f"### {len(cand)} candidate region{'' if len(cand) == 1 else 's'}"
                    + ("" if panel else " — relative to this antigen only"))
        if not panel:
            st.warning(
                "These molecules have no %Rank background, so their regions were selected and "
                "ordered by the raw logit. That ranks positions **within this antigen for this "
                "molecule** and nothing more: it is not an absolute call, it is not comparable "
                "across molecules, lengths or heads, and it must not be read as one."
            )

        if cand.empty:
            # A legitimate answer, not a failure: the budget selected no run of 9 residues — one
            # class-II core — so there is nothing concentrated enough to call a region.
            st.info(f"No run of {scan.CORE} residues — one binding core — survived the "
                    f"{state['budget']:.2f} budget on {seq_len:,} aa. Nothing here is concentrated "
                    "enough to be called a region at that exposure.")

        table = pd.DataFrame({
            "Molecule": cand.allele,
            "Region": [f"{a}–{b}" for a, b in zip(cand.span_start, cand.span_end)],
            "Best window": [f"{a}–{b}" for a, b in zip(cand.best_start, cand.best_end)],
            "Peptide": cand.best_peptide,
            "Windows": cand.n_windows,
        })
        if panel:
            table["%Rank"] = [rankmod.format_pct(v, res) for v in cand.rank_pct]
            table["Call"] = [rankmod.band(v, bands) or "" for v in cand.rank_pct]
        else:
            table["Logit"] = cand.logit.round(3)
        st.dataframe(table, width="stretch", hide_index=True)

        cut = bands["bands"]
        st.caption(
            f"**Region** is the run of residues the `position` rule selected at budget "
            f"{state['budget']:.2f}; **best window** is the highest-scoring window sharing a 9-mer "
            "core with it, and is the sequence to act on. "
            + (f"**%Rank** is that window's percentile against the fixed background for its "
               f"(molecule, length); lower is stronger, `<{res:g}` is below the grid's resolution. "
               f"**Strong** ≤ {cut['strong_rank_pct_max']:g} %, **Weak** ≤ "
               f"{cut['weak_rank_pct_max']:g} %." if panel else "")
        )

        # `core_support` is already long, and it is charted long: pivoting it would put the molecule
        # key in a column NAME, and a `:` in a Vega field name is read as a type shorthand.
        col = "best_rank_pct" if panel else "best_logit"
        st.markdown("**Best %Rank covering each residue**" if panel
                    else "**Best logit covering each residue**")
        st.line_chart(support, x="position", y=col, color="allele", height=220)
        st.caption("Lower is a stronger binder." if panel else "Higher is a stronger binder.")

        st.markdown("**Core support**")
        st.line_chart(support, x="position", y="core_support", color="allele", height=220)
        st.caption(
            "How many selected windows cover each residue. A single lucky window and a genuine "
            "binding core are indistinguishable under a best-score track and very different here — "
            "this is what scanning every length at every offset buys, and a 15-mer-only scan cannot "
            "compute it."
        )

        # `prob` is dropped: the server reports %Rank, not a sigmoid probability
        # (img2-server-output-contract.md §1). It exists in the frame only because
        # scan.candidate_table reads it.
        tag = "panel" if panel else "offpanel"
        c1, c2 = st.columns(2)
        c1.download_button(f"Windows, {tag} (CSV)",
                           collapsed.drop(columns=["prob", "Epitope", "molecule"],
                                          errors="ignore").to_csv(index=False).encode(),
                           f"{state['name']}_{head}_windows_{tag}.csv", "text/csv")
        c2.download_button(f"Regions, {tag} (CSV)",
                           cand.drop(columns=["prob"], errors="ignore").to_csv(index=False).encode(),
                           f"{state['name']}_{head}_regions_{tag}.csv", "text/csv")

    st.caption(
        "The region CSV also carries `peak_start` / `peak_end`, the retired `peak` rule's "
        "half-maximum sub-region. The compute path emits it; it is not what this page selects on, "
        "and it is not shown."
    )

write_st_end()
