# Input, modes and limits

## Standard and Custom HLA Mode

**Standard Mode** uses pre-computed embeddings stored in `data/emb_hla_esmc_small_0601_fp16/`.
**Custom HLA Mode** takes the HLA chains as sequences. A pair of chains that are both in the allele catalogue (`data/mhc_mapping.csv`, the same sequence) is scored exactly as Standard Mode scores the two alleles (the stored embeddings and, where it can, the fast path; the result table says `catalogue` in the column `HLA chains`). Any other chain (including non-human alleles such as Mamu, SLA, and BoLA) has its domain located by alignment to the catalogue (`code/locator.py`: a 4-mer index picks the nearest chains, a BLOSUM62 alignment with free end gaps chooses among them and carries the nearest chain's window across; Biopython), is embedded on-the-fly with ESMC and has the domain cut out, as the stored embeddings are; a domain pasted without its flanks gets the nearest chain's flanks added. The identity with the nearest chain over its window decides: 0.5 or more `located`, 0.4 to 0.5 `low confidence`, and Start and End typed on the page give a `manual window`. Below 0.4 no domain is found and the chain is embedded over its full length, as in earlier versions, marked `full chain` and not validated (its HLA input would be 254-266 positions instead of the 58-82 the model was trained on). Measured on the 6,400-row custom sample with every chain treated as unknown and its own entry or its whole gene hidden from the search: full chains differ from Standard Mode in 0.23 to 0.31 % of the signs (AUC 0.9182 against 0.9186; the earlier full-chain run: 7.4 %, 0.9104), domains pasted alone in 0.4 to 3.1 %; locating a chain takes 3 ms and embedding it 6 ms (the released Qualitative head).

## Input format and limits

**Prediction (CSV upload):**

```csv
MHC_alpha,MHC_beta,Epitope
HLA-DRA*01:01,HLA-DRB1*15:01,GELIGILNAAKVPAD
HLA-DQA1*05:01,HLA-DQB1*02:01,PKYVKQNTLKLATAA
```

**Evaluation (CSV upload with binary labels):**

```csv
MHC_alpha,MHC_beta,Epitope,Target
HLA-DRA*01:01,HLA-DRB1*15:01,GELIGILNAAKVPAD,1
HLA-DQA1*05:01,HLA-DQB1*02:01,PKYVKQNTLKLATAA,0
```

Column headers must match exactly. Allele names must follow WHO HLA nomenclature and be present in `data/mhc_mapping.csv`. The model was trained on 15-mer peptides; other lengths up to 20 residues are accepted but not validated.

**Limits (hard, in `code/engine.py`):** an epitope of more than `MAX_EPITOPE_LENGTH` = 20 residues and, in Custom Mode, an HLA chain of more than `MAX_CHAIN_LENGTH` = 300 residues are refused by the pages with a message and by `Engine.run` / `Engine.run_custom` with a `RequestRefused` (a `ValueError`). Why 20: every built-in data set is 15-mers only, so the range cannot be read from the data; it was measured by keeping each test 15-mer as the core and adding random flanks (the released heads): the Qualitative AUC is 0.93 at 15 residues, 0.87 at 20 and 0.82 at 25 (a shuffled epitope gives 0.80), the share of non-binders scored above 0.5 rises from 15 % at 15 residues to 27 % at 20 and 41 % at 25, and each of the four heads steps down between 20 and 21 residues (AUC -0.031 Qualitative, -0.014 Mass Spectrometry, -0.025 IC50 <500nM, -0.039 IC50 <1000nM). The chain limit is the allele catalogue's longest chain (285) rounded up. A test or a benchmark that needs longer input sets the two constants itself.

**Protein scan limits (`code/scan.py`):** at most 5,000 residues, 100 alleles and 500,000 predictions (windows x alleles x window lengths) per scan, 100,000 predictions while the fast engine is unavailable. They come from the measured throughput: the fast path scores 16,000 to 17,000 predictions per second from 13,000 to 1,200,000 predictions, with the GPU memory staying under 350 MiB over the engine's own; the exact path about 2,000 per second. 500,000 predictions are about 30 s, which is 5,000 residues against 100 alleles.

## Allele coverage

The server provides pre-computed ESMC embeddings for **7,282 alleles** (838 alpha, 6,444 beta chains) from IPD-IMGT/HLA. However, the underlying models were trained on **148 unique HLA dimers** (27 alpha × 88 beta chains; Qualitative dataset, 138 for MS and 67 for IC50; 151 across all the data) from IEDB. Predictions for alleles outside the training set rely on PLM-based zero-shot generalization.

MHC alleles of other species (non-human primates, pigs, cattle) are not available in Standard Mode; use Custom HLA Mode with explicit sequence input for them. The paper did not evaluate these, so their results are not validated.
