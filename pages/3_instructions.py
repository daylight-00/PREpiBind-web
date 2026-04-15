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
- **Length**: 9–25 amino acids; the model was trained exclusively on 15-mer peptides, and performance on other lengths is not guaranteed
- **Alphabet**: Standard 20 amino acids (ACDEFGHIKLMNPQRSTVWY) plus 'X' for ambiguous residues
- **Validation**: Sequences are validated in real time; inputs outside the accepted alphabet are rejected

**MHC Class II Allele Selection:**
- Alleles are derived from the IPD-IMGT/HLA database (human) and curated entries for murine H2 alleles
- **Alpha chains**: 838 alleles — HLA-DPA1, HLA-DQA1, HLA-DQA2, HLA-DRA, and murine H2-IA variants
    - Default: HLA-DRA*01:01
- **Beta chains**: 6,444 alleles — HLA-DPB1, HLA-DQB1, HLA-DQB2, HLA-DRB1, HLA-DRB3/4/5/6/9, and murine H2-IA/IE variants
    - Default: HLA-DRB1*15:01
- **Nomenclature**: WHO HLA nomenclature (e.g., HLA-DRB1*01:01)
- **Coverage note**: Pre-computed ESM C embeddings are available for all 7,282 alleles listed above. The underlying models were trained on 151 unique HLA dimers (27 alpha × 89 beta allele combinations) from IEDB; predictions for alleles outside the training set rely on PLM-based zero-shot generalization.

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

PREpiBind provides four models trained on distinct dataset types derived from the Immune Epitope Database (IEDB; April 2025). All models were trained on 15-mer peptides spanning 151 unique HLA class II dimers (27 alpha chains, 89 beta chains; human HLA-DP/DQ/DR and murine H2). All models share the same architecture: ESM C 300M embeddings processed by a two-block transformer encoder, with separate encoders for MHC and peptide inputs, followed by a joint interaction layer and mean pooling.

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
- **Training data**: Naturally presented ligands identified by HLA ligandomics (MS-derived positives from IEDB; negatives from the quantitative dataset)
- **Output**: Probability of natural MHC-II presentation

#### Technical Details
- **Architecture**: ESM C 300M protein language model with task-specific transformer layers
- **MHC representation**: Pre-computed embeddings of domain-trimmed alpha and beta chain sequences (Standard Mode), or on-the-fly embedding computation (Custom Mode)
- **Inference**: GPU-accelerated (CUDA); mixed-precision (FP16)

#### Prediction Modes

**Standard Mode:**
- Uses pre-computed ESM C 300M embeddings for all 7,282 IPD-IMGT/HLA-catalogued alleles (human HLA-DP/DQ/DR and murine H2)
- Training covered 151 unique HLA dimers; predictions for the remaining alleles rely on PLM-based zero-shot generalization
- Recommended for routine screening with human or murine alleles

**Custom HLA Mode:**
- Accepts full-length or domain-trimmed MHC α and β chain sequences as direct input
- Embeddings are computed on-the-fly; processing time is longer than Standard Mode
- Required for alleles absent from the server database, including non-human primate (*Macaca mulatta*, Mamu), porcine (SLA), and bovine (BoLA) MHC alleles evaluated in the original study

---

### Running a Prediction

1. **Input data**: Enter peptide–MHC pairs manually (up to 20) or upload a CSV file
2. **Select model**: Choose the model appropriate for your experimental context (Qualitative, IC50 <500 nM, IC50 <1000 nM, or MS)
3. **Configure output** (optional): Set result filtering (All / Top N) and KDE visualization
4. **Run**: Click **Run Prediction**; progress is reported as a percentage
5. **Download**: Results are available as a CSV file

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

Built-in test sets (from the original publication) are available for each model and can be downloaded from the Evaluation page.

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
| "Invalid MHC alleles found" | Allele not in server database | Verify nomenclature against IPD-IMGT/HLA; use Custom Mode for novel alleles |
| CSV not loading | Missing or misspelled column headers | Headers must be exactly `MHC_alpha`, `MHC_beta`, `Epitope` |
| KDE plot not displayed | Fewer than 2 unique score values | No action needed; results are still available |
| Slow prediction | Large input or Custom Mode (on-the-fly embedding) | Standard Mode is faster; reduce batch size if memory-limited |

Session state is independent per browser tab; results are not persisted across sessions.

---

### Limitations

- The model was trained exclusively on 15-mer peptides. Predictions for peptides of other lengths are provided but their accuracy is not validated.
- Training comprised 151 unique HLA class II dimers (27 alpha × 89 beta allele combinations) from IEDB. The 7,282 alleles available in Standard Mode exceed the training coverage; predictions for alleles absent from the training set rely on PLM-based zero-shot generalization, and performance may degrade for alleles distant from those seen during training.
- Non-human primate (Mamu), porcine (SLA), and bovine (BoLA) MHC alleles are not available in Standard Mode; use Custom HLA Mode with explicit sequence input for these species.
- Cross-species predictions (murine H2 alleles) show reduced accuracy relative to human HLA alleles in leave-one-molecule-out validation; see the original publication for quantitative results.
- Binding affinity prediction does not model antigen processing efficiency, HLA-DM-mediated peptide editing, or T cell receptor recognition.
- Predictions should not be used for clinical or diagnostic decision-making.

---

### Citation

If you use PREpiBind, please cite:

> Jang DH, Kim D, Park B, Hwang U, Choi Y, Lee J. PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction. *TBD* (2025).

""")

write_st_end()
