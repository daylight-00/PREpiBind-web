"""Custom HLA — scoring a molecule the catalogue does not have, from its chain sequences.

The img1 page accepted *"full-length or domain-trimmed"* sequences and embedded whatever arrived,
whole. Neither input reproduces a catalogue molecule, because a stored HLA embedding is the
**full-length chain embedded and then sliced**, not the window embedded on its own. Measured against
the catalogue path over three molecules x 150 labelled peptides:

| what the sequence becomes | ROC-AUC | mean \\|Δ%Rank\\| | band calls changed |
|---|---|---|---|
| full-length, sliced to the window *(this page)* | −0.002 | 0.45 pp | 2 % |
| full-length, left whole *(img1)* | −0.017 | 3.01 pp | **20 %** |
| the window, embedded alone *(what img1 also invited)* | −0.111 | 14.11 pp | 39 % |

img1 measured the first column and concluded users need not trim, which was right for a server
reporting a sigmoid — an AUC only reads order within a set. A percentile reads position against a
background built the catalogue way, and that is where the shift lands.

So this page asks for **full-length** chains and locates the binding domain itself, by aligning to
the catalogue chain the sequence most resembles. On the 22 annotated non-human chains the catalogue
does not carry — BoLA, SLA, Mamu, Patr — that recovers the window exactly for 20 and within one
residue for all 22, and a one-residue error costs 0.15 pp and changed none of 150 band calls.
"""
import numpy as np
import pandas as pd
import streamlit as st

import artifacts
import chains
import domain
import grids
import rank as rankmod
from app import get_bands, get_engine, write_st_end

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")
MIN_LEN, MAX_LEN = 12, 25
MAX_PEPTIDES = 500
BUILD_CAP_PER_SESSION = 2

st.title("Custom HLA")
st.caption(
    "For alleles absent from the 7,282-chain catalogue, including non-human primate (Mamu, Patr), "
    "porcine (SLA) and bovine (BoLA) molecules. Paste the **full-length** α and β chains — not the "
    "binding domain: the server locates that itself, and pasting a pre-trimmed window is the input "
    "that degrades a prediction most."
)

col_a, col_b = st.columns(2)
alpha_seq = col_a.text_area("α chain, full length", height=120, key="alpha_seq")
beta_seq = col_b.text_area("β chain, full length", height=120, key="beta_seq")

mode = st.radio(
    "Chain representation",
    ["Locate the binding domain by alignment (recommended)", "Use the sequence whole (img1 behaviour)"],
    help="The second is kept so the two can be compared directly; it is what the img1 server did, "
         "and it changes about one band call in five.",
)
aligned_mode = mode.startswith("Locate")


def clean(s):
    return "".join(s.split()).upper()


def resolve(seq, kind, label):
    """(embedding-ready sequence slice or None, a line of provenance to show)."""
    seq = clean(seq)
    if not seq:
        return None, f"{label}: empty."
    if not set(seq) <= VALID_AA:
        bad = sorted(set(seq) - VALID_AA)
        return None, f"{label}: non-standard residues {', '.join(bad)}."
    if not aligned_mode:
        return (0, len(seq)), f"{label}: {len(seq)} aa, used whole (img1 behaviour)."
    found = domain.infer_window(seq, kind=kind)
    if found is None:
        return None, (f"{label}: no catalogue chain is close enough to locate the binding domain. "
                      "Refused rather than guessed — a misplaced window is not a smaller error.")
    start, end, ref, identity = found
    return (start, end), (f"{label}: {len(seq)} aa → domain `[{start}:{end}]` "
                          f"({end - start} aa), carried from `{ref}` at {identity:.0%} identity.")


raw = st.text_area("Peptides, one per line", height=130,
                   placeholder="GELIGILNAAKVPAD\nPKYVKQNTLKLATAA")
st.caption(f"{MIN_LEN}–{MAX_LEN} residues. 12–21 is the validated range; 22–25 is scored with "
           f"limited validation support. Up to {MAX_PEPTIDES} peptides here — this mode embeds the "
           "two chains on every run, so it is slower than the catalogue path.")

if st.button("Run prediction", type="primary"):
    a_win, a_msg = resolve(alpha_seq, "alpha", "α chain")
    b_win, b_msg = resolve(beta_seq, "beta", "β chain")
    peptides = [p.strip().upper() for p in raw.splitlines() if p.strip()]
    bad = [p for p in peptides if not set(p) <= VALID_AA or not MIN_LEN <= len(p) <= MAX_LEN]

    for msg, win in ((a_msg, a_win), (b_msg, b_win)):
        (st.info if win else st.error)(msg)
    if not peptides:
        st.error("Enter at least one peptide.")
    elif bad:
        st.error(f"{len(bad)} peptides are invalid or outside {MIN_LEN}–{MAX_LEN} residues "
                 f"(e.g. `{bad[0]}`).")
    elif len(peptides) > MAX_PEPTIDES:
        st.error(f"{len(peptides)} peptides exceeds the {MAX_PEPTIDES} limit for this mode.")
    elif a_win and b_win:
        with st.spinner("Embedding the two chains and the peptides…"):
            engine = get_engine()
            a_s, b_s = clean(alpha_seq), clean(beta_seq)
            chain_emb = engine.embed([a_s, b_s])
            hla = np.concatenate([chain_emb[b_s][b_win[0]:b_win[1]],
                                  chain_emb[a_s][a_win[0]:a_win[1]]], axis=0)
            key = chains.custom_key(a_s, b_s, aligned_mode)
            df = pd.DataFrame({"MHC_alpha": "custom α", "MHC_beta": "custom β",
                               "Epitope": peptides})
            df["molecule"] = key
            df["length"] = df.Epitope.str.len()
            logits = {h: engine.score(df, h, engine.embed(peptides), hla_emb={key: hla})
                      for h in artifacts.HEADS}
        st.session_state["custom"] = dict(key=key, hla=hla, df=df, logits=logits,
                                          provenance=[a_msg, b_msg])

state = st.session_state.get("custom")
if state:
    st.markdown("## Results")
    bands = get_bands()
    key, df = state["key"], state["df"]
    table = pd.DataFrame({"Peptide": df.Epitope, "Length": df.length})
    ranked = False
    for head, label in artifacts.HEADS.items():
        logit = state["logits"][head]
        pct = np.full(len(df), np.nan)
        for length, grp in df.groupby("length", sort=False):
            r = rankmod.rank(head, key, int(length), logit[df.index.get_indexer(grp.index)])
            if r is not None:
                pct[df.index.get_indexer(grp.index)] = r
        ranked = ranked or not np.isnan(pct).all()
        table[f"{label} logit"] = np.round(logit, 4)
        if not np.isnan(pct).all():
            res = rankmod.resolution(head)
            table[f"{label} %Rank"] = [rankmod.format_pct(v, res) for v in pct]
            table[f"{label} call"] = [rankmod.band(v, bands) or "" for v in pct]
    st.dataframe(table, width="stretch", hide_index=True)

    if not ranked:
        st.warning(
            "**No %Rank.** This molecule has no background, so only the raw score is shown — and a "
            "raw score is not comparable between molecules, lengths or heads. Building a background "
            "scores the same 100,000-peptide set the panel used against these two chains."
        )
        built = st.session_state.setdefault("custom_built", 0)
        if built >= BUILD_CAP_PER_SESSION:
            st.error(f"This session has already built {built} backgrounds, which is the cap.")
        elif st.button("Build a %Rank background (about 5 min of exclusive GPU)"):
            bar = st.progress(0.0, text="Scoring the background…")
            with grids.GPU:
                grids.build(get_engine(), None, None, hla=state["hla"], molecule=key,
                            progress=lambda d, t: bar.progress(min(d / t, 1.0)))
            for head in artifacts.HEADS:
                rankmod.invalidate(head, key)
            st.session_state["custom_built"] = built + 1
            bar.empty()
            st.rerun()

    for line in state["provenance"]:
        st.caption(line)
    st.download_button("Download (CSV)", table.to_csv(index=False).encode(),
                       "prepibind_custom.csv", "text/csv")

write_st_end()
