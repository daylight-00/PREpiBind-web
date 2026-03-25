# PREpiBind Web Server

**P**rotein **R**epresentation-integrated **Epi**tope–MHC Class II **Bind**ing Prediction

![banner](banner.png)

> **PREpiBind Webserver**
> Jang DH, Kim D, Park B, Hwang U, Choi Y, Lee J.
> *TBD* (2025)

A Streamlit-based web server for predicting peptide binding to human and mouse MHC class II molecules using ESMC 300M protein language model embeddings. The server implements four task-specific models trained on qualitative, IC50, and mass spectrometry ligandomics data from IEDB, and provides pre-computed embeddings for 7,282 alleles (838 alpha, 6,444 beta chains) from IPD-IMGT/HLA.

**Live server:** [https://lcdd.snu.ac.kr/prepibind](https://lcdd.snu.kr/prepibind)

---

## Repository Contents

```text
PREpiBind-web/
├── app.py                        # Entry point: model loading, session state, navigation
├── security_config.py            # HTTP security headers via Tornado monkey-patching
├── config_demo.py                # Inference config — Standard Mode
├── config_demo_custom_hla.py     # Inference config — Custom HLA Mode
├── robots.txt
├── .streamlit/config.toml        # Streamlit server settings
│
├── pages/
│   ├── 0_home.py                 # Home / overview
│   ├── 1_prediction.py           # Standard prediction interface
│   ├── 2_evaluation.py           # Model benchmarking with labeled data
│   ├── 3_instructions.py         # Full user guide
│   ├── 4_about.py                # Authors, citation, licensing
│   └── 5_custom.py               # Custom HLA sequence mode
│
├── code/
│   ├── model.py                  # Neural network architecture (plm_cat_mean_inf)
│   ├── inference.py              # Inference pipeline and DataLoader logic
│   ├── encoder.py                # Dataset classes for standard/custom HLA modes
│   ├── dataprovider.py           # Data loading and MHC sequence mapping
│   └── collate.py                # Batch collation with padding and masking
│
├── data/
│   ├── mhc_mapping.csv           # 7,282 allele name → domain-trimmed sequence
│   ├── dataset_demo.csv          # Demo input for smoke-testing
│   ├── test.csv                  # Built-in test set — Qualitative model
│   ├── test_ms.csv               # Built-in test set — MS model
│   ├── test_ic50_500.csv         # Built-in test set — IC50 <500 nM model
│   └── test_ic50_1000.csv        # Built-in test set — IC50 <1000 nM model
│
├── models/                       # Model checkpoints (download required; see below)
└── outputs/                      # Prediction output directory (runtime)
```

---

## Requirements

- Python ≥ 3.10
- CUDA-enabled GPU (tested on CUDA 12.8)

| Package | Version tested |
| ------- | -------------- |
| streamlit | 1.55.0 |
| torch | 2.7.1+cu128 |
| esm | 3.2.0 |
| flash-attn | 2.8.0.post2 |
| numpy | 2.3.0 |
| pandas | 2.3.0 |
| plotly | 6.1.2 |
| scikit-learn | 1.7.0 |
| h5py | 3.14.0 |

Install ESM (EvolutionaryScale):

```bash
pip install esm
```

Install Flash Attention (requires CUDA toolkit and NVCC):

```bash
pip install flash-attn --no-build-isolation
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/daylight-00/PREpiBind-web.git
cd PREpiBind-web
```

### 2. Download model checkpoints

Download from HuggingFace and place in `models/`:

```bash
# PREpiBind task-specific checkpoints
huggingface-cli download daylight00/prepibind-esmc-300m --local-dir models/

# ESMC 300M base model (FP16)
huggingface-cli download daylight00/esmc-300m-2024-12 --local-dir models/
```

Expected files after download:

```text
models/
├── esmc_300m_2024_12_v0_fp16.pth
├── prepi_esmc_small_e5_s128_f4_fp16.pth        # Qualitative
├── prepi_esmc_small_ms_e5_s100_f0_fp16.pth     # Mass Spectrometry
├── prepi_esmc_small_ic50_500_e5_s128_f4_fp16.pth    # IC50 <500 nM
└── prepi_esmc_small_ic50_1000_e5_s128_f1_fp16.pth   # IC50 <1000 nM
```

### 3. Download pre-computed HLA embeddings

Pre-computed ESMC 300M embeddings for all 7,282 IPD-IMGT/HLA alleles are required for Standard Mode:

```bash
huggingface-cli download daylight00/emb_hla_esmc_small_0601_fp16 \
    --repo-type dataset \
    --local-dir data/emb_hla_esmc_small_0601_fp16
```

The embedding directory should resolve to `data/emb_hla_esmc_small_0601_fp16/`.

### 4. Configure GPU

The server defaults to GPU device 1 (`CUDA_VISIBLE_DEVICES=1` in `app.py` line 18). Change this to match your system before running.

---

## Running the Server

```bash
streamlit run app.py \
    --server.port 8501 \
    --server.baseUrlPath /prepibind
```

The server will be available at `http://localhost:8501/prepibind`.

For production deployment behind a reverse proxy, set `--server.baseUrlPath` to match your proxy path prefix. Once TLS is configured, uncomment the HSTS header and `secure` cookie flag in `security_config.py`.

---

## Model Architecture

The prediction model (`code/model.py`) takes two inputs — a pre-computed HLA embedding (concatenated alpha + beta chain from ESMC 300M) and a tokenized peptide sequence — and processes them through:

1. Separate two-block self-attention encoders for HLA and epitope representations
2. A single joint interaction block (concatenation + self-attention)
3. Masked mean pooling → two-layer MLP → scalar logit

The output score is sigmoid-transformed; scores ≥ 0.5 indicate predicted binding.

**Standard Mode** uses pre-computed embeddings stored in `data/emb_hla_esmc_small_0601_fp16/`.
**Custom HLA Mode** computes ESMC embeddings on-the-fly for arbitrary MHC sequences not in the database (including non-human alleles such as Mamu, SLA, and BoLA).

---

## Input Format

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

Column headers must match exactly. Allele names must follow WHO HLA nomenclature and be present in `data/mhc_mapping.csv`. The model was trained on 15-mer peptides; predictions for other lengths are accepted but not validated.

---

## Notes on Allele Coverage

The server provides pre-computed ESMC embeddings for **7,282 alleles** (838 alpha, 6,444 beta chains) from IPD-IMGT/HLA. However, the underlying models were trained on **151 unique HLA dimers** (27 alpha × 89 beta allele combinations) from IEDB. Predictions for alleles outside the training set rely on PLM-based zero-shot generalization.

Non-human primate (Mamu), porcine (SLA), and bovine (BoLA) MHC alleles evaluated in the original study are not available in Standard Mode; use Custom HLA Mode with explicit sequence input for these species.

---

## Citation

If you use PREpiBind or this web server, please cite:

```bibtex
@article{Jang2025PREpiBind,
  author  = {Jang, David Hyunyoo and Kim, Dongwoo and Park, Byungho
             and Hwang, Untaek and Choi, Yoonjoo and Lee, Juyong},
  title   = {{PREpiBind}: Protein Representation-integrated
             Epitope--{MHC} Class {II} Binding Prediction},
  journal = {TBD},
  year    = {2025},
}
```

The ESMC 300M model is developed by EvolutionaryScale and is subject to the [Cambrian Open License Agreement](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement).

---

## License

The PREpiBind web server source code is released under the [MIT License](LICENSE).
ESMC 300M model weights are subject to the Cambrian Open License (EvolutionaryScale).
