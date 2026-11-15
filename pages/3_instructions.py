import streamlit as st
from app import write_st_end

st.set_page_config(
    page_title="Instructions | PREpiBind",
    page_icon=":dna:",
    layout="centered",
)
st.title("Instructions")

st.markdown("""

### Input Data Format

#### Manual Input

**Peptide Sequences:**
- **Length**: at most 20 amino acids (longer peptides are refused); the model was trained exclusively on 15-mer peptides, and the farther a peptide is from 15 residues the less reliable its score (see Limitations)
- **Alphabet**: Standard 20 amino acids (ACDEFGHIKLMNPQRSTVWY) plus 'X' for ambiguous residues
- **Validation**: Sequences are validated in real time; inputs outside the accepted alphabet are rejected

**MHC Class II Allele Selection:**
- Alleles are derived from the IPD-IMGT/HLA database (human) and curated entries for murine H2 alleles
- **Alpha chains**: 838 alleles — HLA-DPA1, HLA-DQA1, HLA-DQA2, HLA-DRA, and murine H2-IA variants
    - Default: HLA-DRA*01:01
- **Beta chains**: 6,444 alleles — HLA-DPB1, HLA-DQB1, HLA-DQB2, HLA-DRB1, HLA-DRB3/4/5/6/9, and murine H2-IA/IE variants
    - Default: HLA-DRB1*15:01
- **Nomenclature**: WHO HLA nomenclature (e.g., HLA-DRB1*01:01)
- **Coverage note**: Pre-computed ESM C embeddings are available for all 7,282 alleles listed above. The underlying models were trained on 148 unique HLA dimers (Qualitative dataset; 138 MS, 67 IC50: see Prediction Models) from IEDB; predictions for alleles outside the training set rely on PLM-based zero-shot generalization.

#### CSV Upload

**Required format:**
```
MHC_alpha,MHC_beta,Epitope
HLA-DQA1*05:01,HLA-DQB1*02:01,PKYVKQNTLKLATAA
HLA-DRA*01:01,HLA-DRB1*15:01,GELIGILNAAKVPAD
```

**Requirements:**
- Column headers must match exactly: `MHC_alpha`, `MHC_beta`, `Epitope`
- Allele names must exist in the server database; unrecognized alleles are rejected with an error message
- File encoding: UTF-8
- Maximum file size: 10 MB
- Maximum entries via web form: 20; larger inputs require CSV upload

---

### Prediction Models

PREpiBind provides four models trained on distinct dataset types derived from the Immune Epitope Database (IEDB; export of 21 April 2025). The Qualitative, MS and IC50 datasets hold 161,223, 111,444 and 47,154 entries of 15-mer peptides with human HLA-DP/DQ/DR and murine H2 molecules, split 70:30 into a training set and an independent test set. The training sets span 148 unique HLA class II dimers (27 alpha chains, 88 beta chains) in the Qualitative dataset, 138 (27 alpha, 80 beta) in MS and 67 (22 alpha, 61 beta) in IC50; all the data together span 151 dimers (27 alpha chains, 89 beta chains). All models share the same architecture: ESM C 300M embeddings processed by separate encoders for MHC and peptide inputs (two transformer layers each), concatenated and passed through one self-attention interaction block, mean pooling and a two-layer prediction head.

Each model is one trained run, the checkpoint released with the paper for that task: the run with the lowest validation loss among the 15 runs (3 seeds x 5 folds) the paper reports for it, chosen without any test-set quantity. The paper's numbers are means over the 15 runs, so a single released run does not reproduce them exactly.

#### Available Models

**Qualitative Binding**
- **Training data**: Binary binding labels (positive/negative) from IEDB qualitative assays
- **Output**: Binding probability (0–1); score ≥ 0.5 indicates predicted binding

**IC50 < 500 nM**
- **Training data**: IC50 measurements binarized at the 500 nM threshold
- **Output**: Probability of binding at IC50 ≤ 500 nM (strong binder)

**IC50 < 1000 nM**
- **Training data**: IC50 measurements binarized at the 1000 nM threshold
- **Output**: Probability of binding at IC50 ≤ 1000 nM (moderate binder)

**Mass Spectrometry (MS) Elution**
- **Training data**: Naturally presented ligands identified by HLA ligandomics (MS-derived positives from IEDB; negatives: experimentally measured non-binders of the Qualitative dataset)
- **Output**: Probability of natural MHC-II presentation

#### Technical Details
- **Architecture**: ESM C 300M protein language model with task-specific transformer layers
- **MHC representation**: Pre-computed embeddings of domain-trimmed alpha and beta chain sequences (Standard Mode), or on-the-fly embedding computation (Custom Mode)
- **Inference**: GPU-accelerated (CUDA); mixed-precision (FP16)

#### Prediction Modes

**Standard Mode:**
- Uses pre-computed ESM C 300M embeddings for all 7,282 IPD-IMGT/HLA-catalogued alleles (human HLA-DP/DQ/DR and murine H2)
- Training covered 148 unique HLA dimers (Qualitative dataset); predictions for the remaining alleles rely on PLM-based zero-shot generalization
- Recommended for routine screening with human or murine alleles

**Custom HLA Mode:**
- Accepts full-length or domain-trimmed MHC α and β chain sequences as direct input (up to 300 residues per chain; the longest chain of the allele catalogue has 285)
- A chain whose sequence is in the allele catalogue is scored exactly as Standard Mode scores that allele: a pair of two such chains gives the same result as selecting the two alleles, at the same speed. The result table says so in the column `HLA chains` (`catalogue`)
- Any other chain: the model reads the domain of an HLA chain (58-82 residues) cut from the embedding of the whole chain, so the domain is located by alignment to the allele catalogue. The nearest allele, the identity with it and the window are listed in a table under the input, where Start and End (residue numbers from 1) can be corrected; the chain is embedded on-the-fly and the domain cut out. A domain pasted without its flanks gets the nearest chain's flanks added. The column `HLA chains` says `located` (identity 0.5 or more with the nearest catalogue chain), `low confidence` (0.4 to 0.5: check the window) or `manual window`. Measured with every catalogue chain treated as unknown and its own entry or its whole gene hidden from the search, full chains differ from Standard Mode in 0.2 to 0.3 % of the signs and domains pasted alone in 0.4 to 3.1 % (against 7 % for the earlier full-chain run)
- A chain with no domain found (identity below 0.4: not an MHC class II chain, or too far from the catalogue) is embedded over its full length unless you enter Start and End; that is not what the model was trained on, so the result is not validated and is marked `full chain`
- Required for alleles absent from the server database, for example MHC alleles of other species (non-human primates, pigs, cattle); the paper did not evaluate these, so their results are not validated

---

### Running a Prediction

1. **Input data**: Enter peptide–MHC pairs manually (up to 20) or upload a CSV file
2. **Select model**: Choose the model appropriate for your experimental context (Qualitative, IC50 <500 nM, IC50 <1000 nM, or MS)
3. **Configure output** (optional): Set result filtering (All / Top N) and KDE visualization
4. **Run**: Click **Run Prediction**; progress is reported as a percentage
5. **Download**: Results are available as a CSV file

---

### Protein Scan

The **Scan** page scores every window of a protein against a panel of HLA class II alleles, to find the regions that bind.

1. **Protein**: paste a sequence or FASTA text, or upload a FASTA file. Digits, spaces, gaps and a trailing stop are ignored; only amino acid codes (and X) are accepted. With several FASTA records you choose one.
2. **Alleles**: the 27-allele reference panel (16 HLA-DR, 6 HLA-DQ, 5 HLA-DP), the 11 DRB1 alleles of that panel, or your own list, one pair per line (`HLA-DQA1*05:01, HLA-DQB1*02:01`; for HLA-DR a beta chain alone is enough, its alpha chain is HLA-DRA*01:01). Names must be in the allele catalogue.
3. **Window lengths**: 15 residues by default, the length the model was trained on; 12 to 20 are optional and less reliable (see Limitations).
4. **Limits**: a protein of at most 5,000 residues, at most 100 alleles and at most 500,000 predictions (windows × alleles × window lengths); that is about 30 seconds of computation. A protein of 500 residues against the 27-allele panel makes 13,122 predictions and takes about a second. If the fast engine is unavailable the limit is 100,000 predictions.
5. **Results**: the 20 highest-scoring windows, the windows bound by the most alleles, a heat map of the score along the protein for each allele, and a CSV of every prediction (`Start`, `End`, `Peptide`, `MHC_alpha`, `MHC_beta`, `Score`, `Logits`). Scores are model outputs, not measured affinities.

---

### Output Format

#### Columns

| Column | Description |
|--------|-------------|
| `MHC_alpha` | Alpha chain allele identifier |
| `MHC_beta` | Beta chain allele identifier |
| `Epitope` | Input peptide sequence |
| `Score` | Sigmoid-transformed binding probability (0–1) |
| `Logits` | Pre-sigmoid model output (linear scale) |
| `HLA chains` | Custom Mode only, the worst of the two chains: `catalogue` (scored as Standard Mode scores the allele), `located` (domain found by alignment), `low confidence` (identity 0.4 to 0.5), `manual window` (Start and End given by you) or `full chain` (no domain found, not validated); see Prediction Modes |

Results are sorted by descending `Score`. A score ≥ 0.5 is the default positive binding threshold, consistent with a sigmoid output; users should calibrate thresholds for their specific experimental context.

#### Score Distribution Plot
A kernel density estimate (KDE) of prediction scores is displayed when ≥ 2 unique score values are present. A dashed vertical line marks the 0.5 threshold.

---

### Model Evaluation

The **Evaluation** page allows performance assessment using labeled test data.

#### Input format:
```
MHC_alpha,MHC_beta,Epitope,Target
HLA-DQA1*05:01,HLA-DQB1*02:01,PKYVKQNTLKLAT,1
HLA-DRA*01:01,HLA-DRB1*15:01,GELIGILNAAKVPAD,0
```
- `Target = 1`: experimentally confirmed binder
- `Target = 0`: confirmed non-binder

#### Metrics reported:
- **ROC AUC**: Discrimination ability across all thresholds
- **PR AUC**: Precision-recall performance (informative under class imbalance)
- **F1 Score**: Harmonic mean of precision and recall at threshold 0.5
- **Precision**: Positive predictive value at threshold 0.5
- **MCC (Matthews Correlation Coefficient)**: Balanced measure accounting for all four confusion matrix cells

Built-in test sets (the independent test sets of the paper: 48,352 Qualitative, 33,490 MS and 14,150 IC50 rows) are available for each model and can be downloaded from the Evaluation page.

---

### Usage Notes

#### Input recommendations
- Submit 15-mer peptides for best performance; this is the peptide length used for training
- Use standard WHO HLA nomenclature; allele names not matching the database will be flagged
- Remove duplicate entries before batch submission

#### Interpreting scores
- Scores should be used for rank-based prioritization, not as absolute affinity values
- The 0.5 threshold is a default; optimal operating thresholds depend on the dataset and intended application
- Binding prediction does not account for antigen processing, peptide editing, or TCR recognition

#### Computational notes
- Batch size is fixed at 128 for GPU processing
- Processing time scales approximately linearly with the number of input entries
- For large datasets (>10,000 entries), consider splitting into multiple jobs

---

### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Invalid characters found" | Peptide contains non-standard amino acid codes | Use only ACDEFGHIKLMNPQRSTVWY or X |
| "Epitopes longer than 20 residues are not supported" | A peptide is longer than the model can score meaningfully | Cut the sequence into peptides of at most 20 residues (15 is best) |
| "HLA chains longer than 300 residues are not supported" | A chain given in Custom Mode is longer than any chain of the allele catalogue | Trim the chain (the signal peptide and the cytoplasmic tail are not needed) |
| "Invalid MHC alleles found" | Allele not in server database | Verify nomenclature against IPD-IMGT/HLA; use Custom Mode for novel alleles |
| CSV not loading | Missing or misspelled column headers | Headers must be exactly `MHC_alpha`, `MHC_beta`, `Epitope` |
| KDE plot not displayed | Fewer than 2 unique score values | No action needed; results are still available |
| Slow prediction | Large input or Custom Mode (on-the-fly embedding) | Standard Mode is faster; reduce batch size if memory-limited |

Session state is independent per browser tab; results are not persisted across sessions.

---

### Limitations

- The model was trained exclusively on 15-mer peptides. Peptides of up to 20 residues are accepted, but their accuracy is not validated and falls with the distance from 15: with random flanks added to the 15-mers of the built-in Qualitative test set the ROC AUC is 0.93 at 15 residues and 0.87 at 20 (0.82 at 25, little above what a shuffled epitope gives, 0.80), and the share of non-binders scored above 0.5 rises from 15 % at 15 residues to 27 % at 20. Longer peptides are refused for that reason; to score a longer sequence, use the Scan page, which scores windows of at most 20 residues.
- Training comprised 148 unique HLA class II dimers (27 alpha × 88 beta chains; Qualitative dataset, fewer for MS and IC50, see Prediction Models) from IEDB. The 7,282 alleles available in Standard Mode exceed the training coverage; predictions for alleles absent from the training set rely on PLM-based zero-shot generalization, and performance may degrade for alleles distant from those seen during training.
- Non-human primate (Mamu), porcine (SLA), and bovine (BoLA) MHC alleles are not available in Standard Mode; use Custom HLA Mode with explicit sequence input for these species.
- Murine H2 alleles have the lowest accuracy of the four serotypes in the paper's Qualitative benchmark (ROC AUC 0.81 to 0.87), and when all H2 data are withheld from training the pooled ROC AUC falls to 0.68 (ESMC 300M); see the preprint for the quantitative results.
- Binding affinity prediction does not model antigen processing efficiency, HLA-DM-mediated peptide editing, or T cell receptor recognition.
- Predictions should not be used for clinical or diagnostic decision-making.

---

### Citation

If you use PREpiBind, please cite:

> Jang DH, Kim D, Hwang U, Park B, Choi Y, Lee J. PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction. *bioRxiv* (2026). [doi:10.64898/2026.09.15.751749](https://doi.org/10.64898/2026.09.15.751749)

""")

write_st_end()
