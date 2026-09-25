"""Instructions — input format, what the output means, and the limitations that qualify it.

Written from `260905/7_final/{REPORTABLE.md,DISCLOSURES.md,BANDS.json}` and the img2 decisions.
Where a number appears here it is quoted with the set it was measured on, because most of the ways
to misread this server's output come from dropping that context — above all the band evidence, whose
negatives are matched decoys rather than a proteome.
"""
import streamlit as st

from app import get_bands, write_st_end

bands = get_bands()
cut = bands["bands"]
ev = bands["evidence"]

st.title("Instructions")

# ---------------------------------------------------------------------------- input
st.header("1. Input")

st.subheader("Peptides")
st.markdown(
    """
Standard amino acids `ACDEFGHIKLMNPQRSTVWY`, plus `X` for an ambiguous residue.

Three different length statements are involved and they must not be collapsed into one:

| range | value | meaning |
|---|---|---|
| **trained** | 12–25 aa | what the served checkpoints were fitted on |
| **validated / reportable** | **12–21 aa** | the only range any pooled performance figure covers |
| **server input** | 12–25 aa | accepted and scored, not blocked |

The validated range came from an operational support criterion applied to the development pool only
— positives ≥ 2,000, molecules ≥ 10, references ≥ 5 per length. **That criterion returns 12–22.**
The range was tightened by one length to 12–21 on grounds outside the data: MixMHC2pred-2.0 returns
NA outside 12–21, so a claim at length 22 is a claim no competitor can be measured against, and
every head-to-head figure in this project is already computed on 12–21. Length 22 is genuinely
supported (2,420 positives, 65 molecules, 48 studies) and is excluded from the headline for
comparability, not for weakness.

The validated range was fixed **after** the holdout had been opened. It is not prespecified, and we
say so rather than let a reviewer find it.
"""
)

st.subheader("Molecules")
st.markdown(
    """
α and β chains are selected independently. Two sources define the chain sequences:

| chains | source | %Rank background |
|---|---|---|
| **134** | the project chain table, which is the single definition of a chain everywhere in this project | precomputed, from the same embedding store the head was trained with |
| **~7,148** | `mhc_mapping.csv`, derived from IPD-IMGT/HLA | none precomputed |

Where the two disagree the chain table wins for the 134 chains it defines. The invariant this
protects is **per molecule, not global**: the sequence that produces a query embedding and the
sequence behind that molecule's %Rank background must be the same sequence. Mixing them puts a
score from one lineage onto a background built from another.

Mouse H2 is restricted to observed I-A / I-E pairs. A free cross product of H2 chains would compose
molecules that do not exist.
"""
)

st.subheader("The %Rank panel")
st.markdown(
    """
A %Rank needs a background grid, and a grid belongs to one molecule and one head.

| family | α chains | β chains | molecules |
|---|---:|---:|---:|
| DR | 1 | 55 | 55 |
| DP | 4 | 15 | 60 |
| DQ | 13 | 14 | 182 |
| H2 (observed pairs only) | — | — | 9 |
| **per head** | | | **306** |

Three heads × 306 molecules = 918 grids. Of the 306, **159 are pairs the training data actually
contains**; the rest are compositions the interface permits from chains it does.

The two chain lists compose about **1,364,793** molecules, so an off-panel request is the common
case rather than an edge case. Precomputing all of it is not possible: at a measured 77.6 s per
molecule, one head alone is roughly 3.4 GPU-years.
"""
)
st.info(
    "**Off-panel molecules are currently scored without a %Rank.** The interface marks them and "
    "leaves the percentile blank rather than showing a raw score that is not comparable between "
    "molecules, lengths or heads. Building a background on first request and caching it is the "
    "agreed design and is not yet implemented."
)

st.subheader("CSV upload")
st.code("MHC_alpha,MHC_beta,Epitope\n"
        "HLA-DQA1*05:01,HLA-DQB1*02:01,PKYVKQNTLKLATAA\n"
        "HLA-DRA*01:01,HLA-DRB1*15:01,GELIGILNAAKVPAD", language="text")
st.markdown(
    "Headers must match exactly. WHO HLA nomenclature; unrecognised names are rejected. UTF-8, "
    "10 MB maximum, and 2,000 rows per interactive run."
)

# ---------------------------------------------------------------------------- output
st.header("2. Output")

st.markdown(
    """
| column | description |
|---|---|
| `MHC_alpha`, `MHC_beta` | the molecule as submitted |
| `Epitope` | the peptide as submitted |
| `<head>_rank_pct` | **%Rank** — percentile of the score within the background for this `(molecule, length)` under that head. Lower is stronger. |
| `<head>_band` | `Strong`, `Weak`, or empty |
| `<head>_logit` | the raw score, for reference only — not comparable across molecules, lengths or heads |
| `length_support` | `validated` (12–21) or `limited` (22–25) |

**There is no probability column and no probability threshold.** The binary call is a %Rank cut.
This is deliberate: no binary operating point for the MS head has ever been measured, and the one
sigmoid operating point this project holds — sensitivity 0.839 against specificity 0.324 at
score ≥ 0.5, "calls almost everything a binder" — belongs to a different, demoted head at ROC-AUC
0.69 and says nothing reliable about the head served here.
"""
)

st.subheader("How %Rank is computed")
st.markdown(
    """
- **Background**: 100,000 peptides drawn uniformly over positions of the reviewed-SwissProt proteome
  — human for HLA, mouse for H2 — spanning 12–25 aa, about 7,000 per length. Same source and sampler
  as the decoy generator, so the background is the distribution the training negatives came from.
- **Stratification**: percentiles are taken within `(molecule, length)`, never pooled across
  lengths. This matches the construction the model was trained and validated under, and keeps a
  12-mer's rank from being dominated by the model's overall length preference.
- **Storage**: 1,000 quantile breakpoints per stratum, so the finest %Rank the grid can express is
  0.1 %. A displayed `<0.1` means "better than every breakpoint", not "zero".
"""
)

st.subheader("The bands, and the evidence behind them")
st.markdown(
    f"""
**Strong: %Rank ≤ {cut['strong_rank_pct_max']:g}. Weak: %Rank ≤ {cut['weak_rank_pct_max']:g}.**

Chosen to match NetMHCIIpan-4.3's installed defaults (`-rankS 1.0`, `-rankW 5.0`). **This is a
comparability choice and not a calibration.** No published criterion exists behind any class-II rank
threshold, in any tool or any version; adopting 1 % / 5 % inherits a number nobody derived. What it
buys is that output held next to NetMHCIIpan's is read at the same operating point.

A separate ligand-recovery calibration on the development pool ({ev['set']}; {ev['n_rows']:,} rows,
{ev['n_pos']:,} positives, {ev['n_neg']:,} negatives) agrees with the shipped bands to within 0.2
percentile points:

| band | %Rank | ligand recovery | negatives passed |
|---|---:|---:|---:|
| strong | ≤ {cut['strong_rank_pct_max']:g} % | **{ev['strong']['sensitivity']:.4f}** | **{ev['strong']['negative_rate']:.4f}** |
| weak | ≤ {cut['weak_rank_pct_max']:g} % | **{ev['weak']['sensitivity']:.4f}** | **{ev['weak']['negative_rate']:.4f}** |

| target recovery | %Rank required |
|---:|---:|
| 50 % | {ev['rank_required_for_recovery']['0.50']} |
| 80 % | {ev['rank_required_for_recovery']['0.80']} |
| 90 % | {ev['rank_required_for_recovery']['0.90']} |
| 95 % | {ev['rank_required_for_recovery']['0.95']} |
"""
)
st.warning(
    f"**The negatives in that table are {ev['negatives_are']}** The passed-negative rate is "
    "therefore **not a specificity** and must not be read as one. On a real proteome, where true "
    "binders are rare, the false-positive burden at a fixed %Rank is a different quantity that this "
    "table does not measure."
)
st.caption(bands["not_a_calibration"])

st.subheader("The three heads are not on one scale")
st.markdown(
    """
The same raw score of **2.0** at length 15, averaged over the 297 human panel molecules:

| head | %Rank | sd | max |
|---|---:|---:|---:|
| MS | **3.30 %** | 0.74 | 5.70 |
| IC50 < 500 nM | **0.32 %** | 0.83 | 4.60 |
| IC50 < 1000 nM | 2.29 % | 3.57 | 24.30 |

A tenfold difference in rarity behind the same displayed raw number — against a cross-*molecule*
spread of only 2.0x, which is what justified building %Rank in the first place. Compare heads on
%Rank, never on score. The length trend also runs in opposite directions between heads.
"""
)

# ---------------------------------------------------------------------------- heads
st.header("3. Choosing a head")
st.markdown(
    """
| head | the question it answers |
|---|---|
| **MS (eluted ligand)** | is this peptide plausibly presented on this molecule by a cell? Trained on immunopeptidomics positives against matched decoys. The only head with held-out study and held-out molecule validation, and the default. |
| **IC50 < 500 nM** | would this peptide bind at ≤ 500 nM in a competition assay? |
| **IC50 < 1000 nM** | the same, at ≤ 1000 nM. |

The two IC50 heads have **cross-validation only** — no held-out study and no held-out molecule panel
— so neither carries a generalisation claim of any kind. 81.5 % of their training rows are exactly
15 aa, so the variable-length extension has little purchase on that assay.

A fourth head, **Qualitative**, is not on the main path. It is reachable only from the page labelled
**img1 legacy**, and it is an img1 cross-validation fold, not an img2 artifact. At variable length
its training pool stops being what its name claims: over 12–25, 74.8 % of its positives are
mass-spectrometry rows (94.1 % once lengths 13 and 15 are set aside) and 90.0 % of its negatives come
from a single high-throughput assay. A multi-length "qualitative" model is an MS model with
assay-mismatch noise, which is why the MS head was trained directly instead.
"""
)

# ---------------------------------------------------------------------------- scan
st.header("4. Antigen scan")
st.markdown(
    """
The scan slides a window along a protein, embeds **every window as its own peptide**, scores it with
the MS head, and marks candidate regions. A window cannot be sliced out of a whole-protein embedding
— measured Spearman 0.42 against per-window embedding — so there is no shortcut and the cost is
linear in panel size.

The candidate rule is **`position`**: a run of residues selected per position under a coverage
budget. It was adopted under a criterion fixed *before* the deciding measurement existed — coverage
admissibility first (mean coverage ≤ 0.20), then recovery lift over a position-matched null, then a
required margin over the incumbent rule.

**Three separate results. They answer different questions, they do not agree with each other, and
they are never merged into one accuracy number.** Measured on 3,707 ligands over 476
protein × molecule pairs (301 proteins), lengths 12–21.

**1. Window discrimination — the scan carries real signal.**

| pairs | mean AUC | median | fraction > 0.5 |
|---:|---:|---:|---:|
| 476 | **0.6479** | 0.6586 | 0.884 |

Identical under every candidate rule: the rule changes how a region is drawn, not how a window is
scored.

**2. Candidate-region localisation — PREpiBind is the weaker localiser.**

| coverage | recovery | null | lift |
|---:|---:|---:|---:|
| 0.1730 | 0.3221 | 0.1752 | **1.839** |
"""
)
st.warning(
    "**The caveat that travels with that lift, always:** at the 264 ligands sitting at positions the "
    "model never saw in training, `position` has **lift 0.955 — no lift at all** (recovery 0.2235 "
    "against a null of 0.2339). The rule buys its tight coverage by concentrating on regions that "
    "resemble ones already seen. Read the 1.839 with that attached, or do not read it.\n\n"
    "At matched coverage PREpiBind localises worse than NetMHCIIpan-4.3. No comparative "
    "localisation claim is made."
)
st.markdown(
    """
**3. Strict no-training-epitope control.**

| evaluable pairs | mean AUC | median | fraction > 0.5 |
|---:|---:|---:|---:|
| 90 | **0.6416** | 0.6888 | 0.833 |

Only 90 of the 476 pairs are evaluable, because **no protein in the validation set is free of
training epitopes** — a training epitope covers a median 66.89 % of a candidate protein's residues,
and 0 of 476 pairs have zero training coverage.

A documented failure, carried rather than dropped. GFAP is the antigen with the least training
contamination in this set:

| protein | molecule | ligands | AUC | coverage |
|---|---|---:|---:|---:|
| GFAP_HUMAN | `HLA-DRB1*15:01 / HLA-DRA*01:01` | 11 | 0.5513 | 0.2500 |
| GFAP_HUMAN | `HLA-DRB5*01:01 / HLA-DRA*01:01` | 4 | 0.4448 | 0.2500 |

One of the two is below chance.

Finally: immunopeptidomics is not exhaustive. A window that was never eluted may still be a binder,
so the scan's negative class is contaminated with true positives and **every scan AUC above is a
lower bound.**
"""
)

# ---------------------------------------------------------------------------- per length
st.header("5. Per-length performance — MS head, cross-validation")
st.markdown(
    """
Published across the whole trained range. 22–25 is shown and is pooled into nothing.

| len | n | positives | ROC-AUC | pooled into primary |
|---:|---:|---:|---:|---|
| 12 | 16,756 | 8,378 | 0.9348 | yes |
| 13 | 35,706 | 17,853 | 0.9534 | yes |
| 14 | 56,644 | 28,322 | 0.9566 | yes |
| 15 | 70,492 | 35,246 | 0.9575 | yes |
| 16 | 64,684 | 32,342 | 0.9575 | yes |
| 17 | 46,934 | 23,467 | 0.9588 | yes |
| 18 | 28,856 | 14,428 | 0.9612 | yes |
| 19 | 17,396 | 8,698 | 0.9615 | yes |
| 20 | 10,922 | 5,461 | 0.9593 | yes |
| 21 | 6,926 | 3,463 | 0.9518 | yes |
| 22 | 4,840 | 2,420 | 0.9501 | **no** |
| 23 | 2,780 | 1,390 | 0.9434 | **no** |
| 24 | 2,098 | 1,049 | 0.9453 | **no** |
| 25 | 1,890 | 945 | 0.9189 | **no** |

Cross-validation folds share one pool and are not independent replicates. The held-out study and
held-out molecule results on the About page are the ones that carry a generalisation claim.
"""
)

# ---------------------------------------------------------------------------- limitations
st.header("6. Limitations")
st.markdown(
    """
Constraints on what the output means, not boilerplate.

1. **PREpiBind is behind NetMHCIIpan-4.3** on both held-out panels of the MS head, with intervals
   excluding zero (−0.0305 over held-out studies, −0.0491 over held-out molecules). It is at parity
   with MixMHC2pred-2.0 over held-out studies and behind over held-out molecules. If your question
   is which tool is most accurate, this is not the answer.
2. **The negative rate quoted for the bands is not a specificity.** The negatives are matched
   decoys, 1:1 with positives, not a natural proteome.
3. **147 of the 306 panel molecules have no training support at all.** They are compositions the
   interface permits from chains the data contains, not pairs the data contains; the MS head was
   trained on 127 molecules. A %Rank for such a molecule is computed correctly against its own
   background, which does not make the underlying score informative.
4. **Lengths 22–25 carry no validated claim.** Per-length numbers are published above; nothing
   pools them.
5. **The IC50 heads have cross-validation only.** No generalisation claim. Their headline figure is
   molecule-wise (0.7585 and 0.7590), not the pooled 0.81 — that sits against a memorisation null of
   0.678 and is never quoted alone.
6. **A %Rank grid belongs to one head and one batch size.** Batch size alone moves a %Rank by 0.32
   percentile points; the serving host contributes 0.05 pp on average and 0.20 pp at most. Near a
   band boundary a call can flip for reasons that are numerical, not biological.
7. **A peptide's score depends slightly on what you submitted it with.** ESM C embeds in batches, so
   the same peptide in a submission of 4 and a submission of 400 gets marginally different
   embeddings. Measured here: the logit moves by 0.004 on average (0.038 at most, against a spread
   of 2.2), the %Rank by **0.053 percentile points on average and 0.5 at most**, and over 300
   peptides **no band call changed**. That is the same order as the fp16 and serving-host terms.
   Re-submitting the identical set is bit-identical.
8. **One seed.** Every served head is a single fit at seed 42. There is no seed-level variance
   component, and SD columns anywhere in this documentation are across held-out units, not seeds.
9. **Neither the training arm nor the validated range was prespecified.** Both were fixed after
   results existed. The sensitivity sweeps behind them are published, which makes the derivation
   auditable — it does not make it prespecified.
10. **Binding is not presentation and not immunogenicity.** Nothing here models antigen processing,
    HLA-DM-mediated editing, or T-cell receptor recognition.
11. **No temporal validation claim.** The held-out panels are study- and molecule-generalisation
    tests; neither is a temporal set.
12. **Research use only.** Not for clinical or diagnostic decision-making.
"""
)

# ---------------------------------------------------------------------------- troubleshooting
st.header("7. Troubleshooting")
st.markdown(
    """
| symptom | cause | action |
|---|---|---|
| "peptides contain non-standard residues" | a peptide has a character outside the 20 standard codes | use `ACDEFGHIKLMNPQRSTVWY` |
| "peptides fall outside 12–25 residues" | shorter than 12 or longer than 25 | the model was not fitted outside 12–25; trim or split |
| `limited` in `length_support` | the peptide is 22–25 aa | expected. The score is real; the validation behind it is not |
| %Rank column is blank | the molecule is off the 306-molecule panel and has no background | the raw logit is in the CSV, but it is not comparable between molecules, lengths or heads |
| "unrecognised chains" | the name is in neither the chain table nor `mhc_mapping.csv` | check it against IPD-IMGT/HLA nomenclature |
| CSV not loading | missing or misspelled headers | they must be exactly `MHC_alpha`, `MHC_beta`, `Epitope` |

Session state is independent per browser tab. Input data and results are held in memory and are not
written to disk.
"""
)

st.header("8. Citation")
st.markdown("**TBD.** The img2 manuscript is not yet written; a citation will be added on submission.")

write_st_end()
