"""img1 legacy — the four checkpoints the previous server ran, kept so its numbers can be reproduced.

This page exists for one purpose: someone has a result from the img1-era server and needs to get the
same number again. Everything on it is chosen to serve that and nothing else.

It is therefore deliberately *unlike* the rest of the server:

  * it reports a **sigmoid score**, because that is what the img1 server reported. There is no %Rank
    here — no background grid was ever built for these checkpoints, and inventing one would produce
    a number the old server never emitted.
  * it resolves every chain through `mhc_mapping.csv`, **not** the corrected chain table. These
    weights predate the 2026-09-09 H2 chain fix, so the stale sequences are the ones they were
    fitted on. Correcting them here would break reproduction and would not make the model right.
  * it does not filter on the img2 validated length range. The img1 models were trained on 15-mers.

None of these is a recommendation. Each served checkpoint is **one cross-validation fold** — trained
on four fifths of its data, picked per task, at three different seeds, none of them the deployment
seed — which is the single thing about the img1 server a reviewer was most likely to reject on, and
the reason img2 replaced them.

The img2 Qualitative head is **not** offered here or anywhere: hwjang discarded it on 2026-09-25.
The qualitative task itself is retired — at variable length its pool is 94.1 % mass-spectrometry
positives (once lengths 13 and 15 are set aside) against a single assay's negatives, so a
multi-length "qualitative" model is an MS model with assay-mismatch noise.
"""
import numpy as np
import pandas as pd
import streamlit as st

import chains
from app import get_chain_lists, get_engine, write_st_end

# The four checkpoints `config_demo.py` loaded, with the fold and seed each one actually is.
LEGACY = {
    "Qualitative": ("prepi_esmc_small_e5_s128_f4_fp16.pth", 128, 4),
    "Mass spectrometry": ("prepi_esmc_small_ms_e5_s100_f0_fp16.pth", 100, 0),
    "IC50 < 500 nM": ("prepi_esmc_small_ic50_500_e5_s128_f4_fp16.pth", 128, 4),
    "IC50 < 1000 nM": ("prepi_esmc_small_ic50_1000_e5_s128_f1_fp16.pth", 128, 1),
}
MAX_PEPTIDES = 2000
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

st.title("img1 legacy")
st.error(
    "**These are img1 checkpoints and they are not the validated model.** Each is a single "
    "cross-validation fold, trained on four fifths of its data and selected per task, at three "
    "different seeds. They are kept so that a number from the previous server can be reproduced. "
    "For anything else use **Prediction**, which serves the img2 final artifacts and reports %Rank."
)

st.markdown(
    """
| task | checkpoint | seed | fold |
|---|---|---:|---:|
"""
    + "\n".join(f"| {task} | `{f}` | {s} | {k} |" for task, (f, s, k) in LEGACY.items())
)
st.caption(
    "Chains are resolved through `mhc_mapping.csv`, the source these checkpoints were fitted "
    "against, rather than the corrected chain table the rest of the server uses. For `H2-IAd`, "
    "`H2-IAg7` and three DQB1 chains those two disagree, and here the stale one is the correct "
    "choice — it is what these weights saw.\n\n"
    "**Reproduction is close, not exact.** Measured against the img1 code path on the same input, "
    "logits agree to within 0.002 — 0.06 % of their spread — and the displayed score to four "
    "decimal places. The residue is the serving path: this server embeds in batches of 64 through "
    "the function the img2 %Rank background was built with and runs the head in fp32, where the "
    "img1 server used batches of 128 and an fp16 head. Neither is more correct; they are different "
    "arithmetic."
)

lists = get_chain_lists()
task = st.selectbox("Task", list(LEGACY), index=0)

col_a, col_b = st.columns(2)
alpha = col_a.selectbox("α chain", lists["alpha"],
                        index=lists["alpha"].index("HLA-DRA*01:01")
                        if "HLA-DRA*01:01" in lists["alpha"] else 0)
beta = col_b.selectbox("β chain", lists["beta"],
                       index=lists["beta"].index("HLA-DRB1*15:01")
                       if "HLA-DRB1*15:01" in lists["beta"] else 0)

raw = st.text_area("Peptides, one per line", height=140,
                   placeholder="GELIGILNAAKVPAD\nPKYVKQNTLKLATAA")
st.caption("The img1 models were trained on 15-mers. Other lengths are scored; nothing validates "
           "them, and the img2 pages are where variable length was actually fitted and tested.")

if st.button("Run (img1 legacy)", type="primary"):
    peptides = [p.strip().upper() for p in raw.splitlines() if p.strip()]
    bad = [p for p in peptides if not set(p) <= VALID_AA]
    if not peptides:
        st.error("Enter at least one peptide.")
    elif bad:
        st.error(f"{len(bad)} peptides contain non-standard residues (e.g. `{bad[0]}`).")
    elif len(peptides) > MAX_PEPTIDES:
        st.error(f"{len(peptides):,} peptides exceeds the {MAX_PEPTIDES:,} limit.")
    else:
        checkpoint, seed, fold = LEGACY[task]
        with st.spinner(f"Scoring with the img1 {task} fold {fold}, seed {seed}…"):
            engine = get_engine()
            engine.add_head(f"legacy:{task}", f"models/{checkpoint}")
            df = pd.DataFrame({"MHC_alpha": alpha, "MHC_beta": beta, "Epitope": peptides})
            df["molecule"] = chains.molecule(alpha, beta)
            epi_emb = engine.embed(peptides)
            # legacy=True: mhc_mapping sequences, the ones these weights were fitted on.
            logit = engine.score(df, f"legacy:{task}", epi_emb, legacy=True)
        out = pd.DataFrame({
            "Peptide": peptides,
            "Length": [len(p) for p in peptides],
            "Score": 1.0 / (1.0 + np.exp(-logit)),
            "Logits": logit,
        }).sort_values("Score", ascending=False)
        st.session_state["legacy_result"] = (out, task, seed, fold, alpha, beta)

result = st.session_state.get("legacy_result")
if result is not None:
    out, task, seed, fold, alpha, beta = result
    st.markdown(f"### {task} — img1, seed {seed}, fold {fold}")
    st.dataframe(out.style.format({"Score": "{:.5f}", "Logits": "{:.5f}"}),
                 width="stretch", hide_index=True)
    st.caption(
        "`Score` is the sigmoid of `Logits`, the output the img1 server reported. It is **not** "
        "calibrated and **not** comparable between molecules, lengths or tasks — on the one head "
        "where an operating point was ever measured, a score of 0.5 ran at sensitivity 0.839 "
        "against specificity 0.324, which calls almost everything a binder. The img2 pages report "
        "a percentile instead, which is what makes a number comparable."
    )
    st.download_button("Download (CSV)", out.to_csv(index=False).encode(),
                       f"prepibind_img1_legacy_{task.replace(' ', '_')}.csv", "text/csv")

write_st_end()
