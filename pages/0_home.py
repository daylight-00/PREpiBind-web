"""Home — what the server is, and what it does not claim.

The positioning block is not modesty. `260905/7_final/REPORTABLE.md` records that img2 is behind
NetMHCIIpan-4.3 on both held-out panels with intervals excluding zero, and states that no accuracy
claim over either comparator is supported by the campaign. A landing page that implied otherwise
would be the first thing a reviewer checked and the first thing they found wrong.
"""
import streamlit as st

import artifacts
from app import get_bands, write_st_end

st.title("PREpiBind")
st.markdown(
    "**P**rotein **R**epresentation-integrated **Epi**tope–MHC Class II **Bind**ing Prediction")

st.markdown(
    """
### Overview

PREpiBind predicts peptide binding to human (HLA-DP/DQ/DR) and mouse (H2-IA/IE) MHC class II
molecules. Each head is a transformer over ESM C 300M embeddings of the peptide and of the α and β
chain sequences: the MHC branch is self-attended, the peptide tokens are joined, and the
concatenation is collapsed by a masked mean (55.8 M parameters per head).

Three heads are served, each trained on a different experimental definition of binding:

| head | trained on | weight |
|---|---|---|
| **MS (eluted ligand)** | naturally presented ligands from immunopeptidomics, against `(molecule, length)`-matched decoys at 1:1 | headline |
| **IC50 < 500 nM** | IC50 measurements binarised at 500 nM | appendix |
| **IC50 < 1000 nM** | IC50 measurements binarised at 1000 nM | appendix |

Training data derive from an IEDB export of 2026-08-23. The MS head was fitted over lengths 12–25 on
366,924 rows (183,462 positives) spanning 127 molecules.
"""
)

bands = get_bands()["bands"]
st.markdown(
    f"""
### What the server reports

**A %Rank, per head** — the percentile of the query score within a length-matched background for
that molecule. Lower is stronger. A band call follows from it: **strong** at %Rank ≤
{bands['strong_rank_pct_max']:g}, **weak** at ≤ {bands['weak_rank_pct_max']:g}.

There is no probability output and no probability threshold. Raw scores are not comparable across
molecules, across lengths, or across heads, and the server does not ask you to compare them — the
same raw score of 2.0 at length 15 is the 3.30rd percentile on MS and the 0.32nd on IC50 < 500 nM.

The bands match NetMHCIIpan-4.3's installed defaults (`-rankS 1.0`, `-rankW 5.0`) so that output
from the two tools is read at the same operating point. **That is a comparability choice, not a
calibration**: no published criterion exists behind any class-II rank threshold, in any tool.
"""
)

col_start, col_scope = st.columns(2)
with col_start:
    st.markdown(
        """
### Quick start

1. Open **Prediction**.
2. Enter peptides (12–25 aa) and select an α and a β chain, or upload a CSV.
3. Run. All three heads are reported together, as %Rank with a band call.
4. Download the results as CSV.

Peptides of 12–21 residues are inside the validated range. 22–25 is accepted and scored but marked
*limited validation support*; no pooled performance figure covers it.
"""
    )
with col_scope:
    st.markdown(
        """
### Scope

| | |
|---|---|
| length, trained | 12–25 aa |
| length, validated | **12–21 aa** |
| length, accepted | 12–25 aa |
| molecules with a background | **306 per head** |
| of those, pairs in the training data | 159 |
| molecules the chain lists compose | ~1.36 million |

A molecule outside the panel has no precomputed background; building one costs about 78 s on the
serving GPU.
"""
    )

st.markdown("### Honest positioning")
st.warning(
    """
**PREpiBind does not outperform the established class-II predictors it was measured against.**
On the MS head, over held-out studies and held-out molecules, it is behind NetMHCIIpan-4.3 by
0.0305 and 0.0491 ROC-AUC, with 95 % intervals that exclude zero. Against MixMHC2pred-2.0 it is at
parity over held-out studies and behind over held-out molecules. No accuracy claim over either tool
is made, and none is supported by this work.
"""
)
st.markdown(
    """
- The window-level antigen scan carries real signal but is the **weaker localiser**. Its three
  results are reported separately on the Instructions page and are never combined into one accuracy
  number.
- **Binding is not presentation and not immunogenicity.** Nothing here models antigen processing,
  HLA-DM editing, or T-cell receptor recognition.
- Research use only. Not for clinical or diagnostic decision-making.
"""
)

st.page_link("pages/3_instructions.py", label="Instructions — input format, output semantics, limitations")
st.page_link("pages/4_about.py", label="About — the full performance record, authors and licensing")

with st.expander("Served artifacts"):
    integrity = {h: artifacts.manifest(h) for h in artifacts.HEADS}
    st.markdown(
        "\n".join(
            [f"| head | file | sha256 |", "|---|---|---|"]
            + [f"| {artifacts.HEADS[h]} | `{m['served_file']}` | `{m['served_sha256'][:16]}…` |"
               for h, m in integrity.items()]
        )
    )
    st.caption("Each is one fit on the whole development pool at seed 42 for a pre-declared epoch "
               "count, frozen and hashed. The server re-hashes them against these manifests at "
               "startup.")

write_st_end()
