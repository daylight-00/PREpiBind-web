"""About — the full performance record, stated including the parts that do not flatter it.

Every figure here is recomputed in `260905/7_final/freeze_results.py` from the per-cell prediction
tables rather than transcribed, and the freeze refuses to run if its three gates fail. That is why
this page can carry exact numbers; it is also why the numbers include a head-to-head the project
loses.
"""
import streamlit as st

import artifacts
from app import write_st_end

st.title("About PREpiBind")
st.markdown(
    "**P**rotein **R**epresentation-integrated **Epi**tope–MHC Class II **Bind**ing Prediction")

st.markdown(
    """
### Development team

For inquiries regarding the web server, please contact the developers.

- **David Hyunyoo Jang** *(Primary developer)*
  [hwjang00@snu.ac.kr](mailto:hwjang00@snu.ac.kr) | [GitHub](https://github.com/daylight-00)
- **Dongwoo Kim** *(Co-developer)* [dingoh@snu.ac.kr](mailto:dingoh@snu.ac.kr)
- **Juyong Lee** *(Corresponding author, PI)* [nicole23@snu.ac.kr](mailto:nicole23@snu.ac.kr)

[Lab of Computational Drug Discovery](https://sites.google.com/view/lcbc) |
[College of Pharmacy, Seoul National University](https://snupharm.snu.ac.kr/en/)
"""
)

st.markdown("---")
st.header("What this server is")
st.markdown(
    """
PREpiBind predicts peptide binding to human (HLA-DP/DQ/DR) and mouse (H2-IA/IE) MHC class II
molecules over peptide lengths 12–25. Each head is a transformer over ESM C 300M embeddings: the α
and β chain representations are self-attended, concatenated with the peptide tokens, and collapsed
by a masked mean — 55,820,161 parameters per head, served in float16.

Three heads are served, each trained on a different experimental definition of binding, from an IEDB
export of 2026-08-23:

| head | training rows | epochs | seed |
|---|---:|---:|---:|
| MS (eluted ligand) — headline | 366,924 | 12 | 42 |
| IC50 < 500 nM | 57,733 | 13 | 42 |
| IC50 < 1000 nM | 57,721 | 15 | 42 |

The MS arm holds 183,462 positives against `(molecule, length)`-matched decoys at 1:1, over 127
molecules. Each head is one fit on the whole development pool at a pre-declared epoch count — not a
cross-validation fold — so the checkpoint that was evaluated is the checkpoint that is served.

The MS arm caps any single study at 40,000 rows. That is a composition decision, not a performance
selection: immunopeptidomics carries cell-line, HLA-typing and sample-preparation signatures per
study, and a model that learns one study's signature looks excellent in cross-validation and fails
on a new laboratory's data. An uncapped arm was later trained as a falsifier on six matched held-out
studies and came out **ahead by 0.0020 ROC-AUC** (0.9112 against 0.9092) — far below the ~0.009 the
design could have resolved. That is a failed falsification, not a demonstration of equivalence, and
the cap stands.

**The server reports %Rank, not a probability.** The binary call is a %Rank cut: strong ≤ 1 %,
weak ≤ 5 %, matching NetMHCIIpan-4.3's installed defaults so the two tools are read at the same
operating point. This is a comparability choice, not a calibration.
"""
)

st.markdown("---")
st.header("Performance")
st.caption(
    "Every figure is restricted to the validated length range 12–21. Confidence intervals are "
    "bootstrap over the held-out unit — a study, a molecule or a fold — 4,000 resamples, never over "
    "rows."
)

st.subheader("MS head — held-out panels")
st.markdown(
    """
| panel | units | rows | ROC-AUC | SD | median | 95 % CI |
|---|---:|---:|---:|---:|---:|---|
| cross-validation, 5 epitope-grouped folds | 5 | 355,316 | 0.9571 | 0.0012 | 0.9566 | 0.9563–0.9581 |
| **leave-one-study-out, 18 studies** | 18 | 340,870 | **0.8841** | 0.0487 | 0.8890 | 0.8604–0.9051 |
| **leave-one-molecule-out, 37 molecules** | 37 | 327,494 | **0.9059** | 0.0450 | 0.9096 | 0.8909–0.9198 |

The cross-validation interval is the weakest of the three: five folds share one pool and are not
independent replicates. It is reported for completeness and carries no generalisation claim.

Per molecule rather than per row, the same panels give mean ROC-AUC 0.9460 (90 molecules,
cross-validation), 0.8817 (101, leave-one-study-out) and 0.9059 (37, leave-one-molecule-out), with
individual molecules ranging from 0.51 to 1.00.
"""
)

st.subheader("Head to head — the same held-out units")
st.markdown(
    """
| panel | compared | PREpiBind | NetMHCIIpan-4.3 | Δ | 95 % CI | units won | MixMHC2pred-2.0 | Δ | 95 % CI |
|---|---|---:|---:|---:|---|---|---:|---:|---|
| leave-one-study-out | 17 / 18 | 0.8853 | 0.9158 | **−0.0305** | [−0.0536, −0.0038] | 3 / 17 | 0.8843 | +0.0010 | [−0.0244, +0.0284] |
| leave-one-molecule-out | 36 / 37 | 0.9079 | 0.9569 | **−0.0491** | [−0.0610, −0.0373] | 3 / 36 | 0.9323 | −0.0245 | [−0.0410, −0.0063] |
"""
)
st.warning(
    "**PREpiBind is behind NetMHCIIpan-4.3 on both held-out panels, with intervals that exclude "
    "zero.** It wins 3 of 17 held-out studies and 3 of 36 held-out molecules. Against "
    "MixMHC2pred-2.0 it is at parity over held-out studies (the interval spans zero) and behind over "
    "held-out molecules. No claim of accuracy superiority over either tool is made, and none is "
    "supported by this work."
)
st.markdown(
    """
One unit per panel is dropped from the comparison because neither external tool scored it, so the
PREpiBind column above is computed on the compared subset. PREpiBind's own figures over the full
panels are 0.8841 and 0.9059; the subset figures exist only inside this comparison.

One asymmetry runs the other way, stated because it does not rescue the result: PREpiBind's
predictions come from models that never saw the held-out unit, whereas both external tools were
trained on IEDB and almost certainly saw these studies and these molecules.
"""
)

st.subheader("IC50 heads")
st.markdown(
    """
**The primary figure is molecule-wise.** Their pooled memorisation null is 0.678 while the
molecule-wise null is 0.5000, so only the molecule-wise number sits on the same aggregation unit as
its own null.

| head | molecules | mean ROC-AUC | median | IQR |
|---|---:|---:|---:|---|
| IC50 < 500 nM | 42 | **0.7585** | 0.7728 | 0.7198–0.7981 |
| IC50 < 1000 nM | 43 | **0.7590** | 0.7711 | 0.7351–0.8019 |

The corresponding pooled cross-validation values, 0.8103 and 0.8036, are retained for continuity but
are **never quoted alone**: pooling ranks every molecule on one score axis, so between-molecule
separation inflates them, and against a pooled null of 0.678 they are not the evidence they look
like.

Both IC50 heads have cross-validation only. Neither has a held-out study or held-out molecule panel,
so neither carries a generalisation claim. 81.5 % of their training rows are exactly 15 aa.
"""
)

st.subheader("Antigen scan")
st.markdown(
    """
Three separate results, never combined into one accuracy number. Rule `position`, 3,707 ligands over
476 protein × molecule pairs.

| result | figure |
|---|---|
| window discrimination | mean AUC **0.6479** over 476 pairs (median 0.6586; 88.4 % above 0.5) |
| candidate-region localisation | coverage 0.1730, recovery 0.3221, **lift 1.839** over the position-matched null |
| strict no-training-epitope control | mean AUC **0.6416** over the 90 evaluable pairs |

**The localisation lift carries a caveat that travels with it:** at the 264 ligands sitting at
positions the model never saw in training, `position` has **lift 0.955 — no lift at all**. The rule
buys its tight coverage by concentrating on regions resembling those already seen.

At matched coverage PREpiBind localises worse than NetMHCIIpan-4.3, and no comparative localisation
claim is made. Only 90 of 476 pairs are evaluable under the strict control because no protein in the
validation set is free of training epitopes — a training epitope covers a median 66.89 % of a
candidate protein's residues. Because immunopeptidomics is not exhaustive, every scan AUC above is a
lower bound.

Tool ordering differs between the scan and peptide-level ranking. That is task dependence, not a
contradiction, and it is one more reason the scan numbers are kept separate from the peptide panels.
"""
)

st.subheader("What this work does not claim")
st.markdown(
    """
- No accuracy superiority over NetMHCIIpan-4.3 or MixMHC2pred-2.0.
- No temporal validation. The held-out panels are study- and molecule-generalisation tests.
- No seed stability. One seed, 42; SD columns are across units, not seeds.
- No claim at peptide lengths 22–25. Per-length results are published; they are never pooled.
- No generalisation claim for either IC50 head.
- No claim that the training arm or the validated length range was prespecified. Both were fixed
  after results existed.
- No comparative localisation claim from the antigen scan.
"""
)

st.markdown("---")
st.header("System information")
st.markdown(
    f"""
- **Model version:** img2 v1.0 served heads — {', '.join(artifacts.HEADS.values())} — float16
- **Result freeze:** 2026-09-25
- **Status:** img2 reboot, in development
- **Platform:** Python / Streamlit
- **Access:** freely available, no login required
- **User privacy:** input data and prediction results are processed in memory and are never written
  to disk. No user data is retained.

A page labelled **img1 legacy** exposes the earlier Qualitative head. It is an img1 cross-validation
fold, not an img2 artifact, and it is not on the main prediction path. It is retained so that
results produced with the previous server can be reproduced.
"""
)

st.markdown("---")
st.header("Resources and licensing")
col_res, col_lic = st.columns(2)
with col_res:
    st.markdown(
        """
##### Academic and technical resources
- PREpiBind methodology: *manuscript in preparation*
- Training and evaluation code: [GitHub](https://github.com/daylight-00/PREpiBind)
- Web server code: [GitHub](https://github.com/daylight-00/PREpiBind-web)
- MHC-II allele datasets: [Zenodo](https://zenodo.org/communities/prepibind-mhc-alleles)
- PREpiBind checkpoints: [HuggingFace](https://huggingface.co/daylight00/prepibind-esmc-300m)
- ESM C 300M checkpoints (float16): [HuggingFace](https://huggingface.co/daylight00/esmc-300m-2024-12)
"""
    )
with col_lic:
    st.markdown(
        """
##### Licensing
- PREpiBind source code, web server and datasets: **MIT License**
- PREpiBind checkpoints: **MIT License**
- ESM C 300M checkpoints and generated embeddings:
  **[Cambrian Open License](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement)**
  (EvolutionaryScale)

##### Research use only
This server is for research purposes. Not for clinical use, and not for diagnostic
decision-making.
"""
    )

st.markdown("---")
st.header("Citation")
st.markdown("**TBD.** The manuscript describing this release is not yet written. A citation will be "
            "added here on submission.")

with st.expander("Abstract"):
    st.markdown(
        """
Prediction of peptide–MHC class II binding is limited by molecular polymorphism, by variable peptide
length, and by the fact that each available training assay measures a different thing. PREpiBind is
a transformer model over ESM C 300M representations of the peptide and of both MHC chains, trained
over peptide lengths 12–25. This release serves three independently trained heads — mass
spectrometry eluted ligands, IC50 < 500 nM and IC50 < 1000 nM — rather than one merged "binding"
model, because at variable length a pooled qualitative dataset is dominated by eluted ligands on the
positive side and by a single assay on the negative side, and ceases to be the composite its name
implies.

The server reports a %Rank per head against a length-stratified proteome-derived background, with
strong and weak bands at 1 % and 5 % to match the installed defaults of NetMHCIIpan-4.3. It does not
report a probability and does not apply a probability threshold.

The eluted-ligand head was evaluated on two held-out designs over the validated range 12–21: 18
held-out studies (ROC-AUC 0.8841) and 37 held-out molecules (0.9059), with intervals bootstrapped
over held-out units rather than rows. Against the same units, PREpiBind is behind NetMHCIIpan-4.3 by
0.0305 and 0.0491 ROC-AUC with intervals excluding zero, at parity with MixMHC2pred-2.0 over
held-out studies, and behind it over held-out molecules. The IC50 heads reach molecule-wise mean
ROC-AUC 0.7585 and 0.7590 under cross-validation only. Applied to whole antigens the model
discriminates windows (mean AUC 0.6479 over 476 protein–molecule pairs, 0.6416 under a strict
no-training-epitope control) but is the weaker localiser: its candidate-region lift of 1.839 falls
to 0.955 — no lift — at ligand positions the model never saw in training.

PREpiBind is therefore offered as an openly documented, freely available class-II predictor with an
explicit statement of where it stands relative to existing tools, not as a claim to improved accuracy
over them.
"""
    )

write_st_end()
