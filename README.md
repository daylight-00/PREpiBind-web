# PREpiBind Web Server

**P**rotein **R**epresentation-integrated **Epi**tope–MHC Class II **Bind**ing Prediction

![banner](banner.png)

> **PREpiBind Webserver**
> Jang DH, Kim D, Choi Y, Lee J.
> *TBD* (2026)

A Streamlit-based web server for predicting peptide binding to human and mouse MHC class II molecules using ESM C 300M protein language model embeddings. The server implements four task-specific models trained on qualitative, IC50, and mass spectrometry ligandomics data from IEDB, and provides pre-computed embeddings for 7,282 alleles (838 alpha, 6,444 beta chains) from IPD-IMGT/HLA.

**Live server:** [https://lcdd.snu.ac.kr/prepibind](https://lcdd.snu.ac.kr/prepibind)

---

## Repository Contents

```text
PREpiBind-web/
├── app.py                        # The Streamlit script: model loading, session state, navigation
├── serve.py                      # ASGI entry point (st.App): app.py with the security headers
├── security_config.py            # The security-headers middleware and the robots.txt route
├── pyproject.toml, uv.lock       # Environment (uv)
├── tests/                        # pytest: a GPU-free tier and a tier marked gpu (see Tests)
├── robots.txt                    # What /robots.txt answers
├── .streamlit/config.toml        # Streamlit server settings
│
├── pages/
│   ├── 0_home.py                 # Home / overview
│   ├── 1_prediction.py           # Standard prediction interface
│   ├── 2_evaluation.py           # Model benchmarking with labeled data
│   ├── 3_instructions.py         # Full user guide
│   ├── 4_about.py                # Authors, citation, licensing
│   ├── 5_custom.py               # Custom HLA sequence mode
│   └── 6_scan.py                 # Protein scan: every window of a protein against a panel of alleles
│
├── code/
│   ├── model.py                  # Neural network architecture (plm_cat_mean_inf)
│   ├── embedder.py               # ESM C 300M epitope/HLA embedder (loads the released checkpoint)
│   ├── engine.py                 # Inference engine: four heads on one ESM C, HLA store, batching, scores
│   ├── scan.py                   # Protein scan: FASTA, windows, allele panels, limits (plain Python)
│   ├── trt_engines.py            # Fast engine: TensorRT fp16 plans (build, load, run)
│   ├── build_engines.py          # Builds the plans for this TensorRT version and GPU
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

- Python ≥ 3.14
- NVIDIA GPU with a current driver. PyTorch brings its own CUDA 13 runtime; no CUDA toolkit or nvcc is needed
- [uv](https://docs.astral.sh/uv/)

The environment is declared in `pyproject.toml` and locked in `uv.lock`:

| Package | Version locked |
| ------- | -------------- |
| python | 3.15.2 |
| streamlit | 1.67.0 |
| torch | 2.15.0 |
| esm | 3.4.3 |
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| plotly | 7.1.0 |
| scikit-learn | 1.9.1 |

Attention runs through PyTorch's fused scaled-dot-product attention; flash-attn is not used.

---

## Setup

### 1. Clone the repository and create the environment

```bash
git clone https://github.com/daylight-00/PREpiBind-web.git
cd PREpiBind-web
uv sync
```

### 2. Download model checkpoints

Download from HuggingFace and place in `models/`:

```bash
# PREpiBind task-specific checkpoints
huggingface-cli download daylight-00/prepibind-esmc-300m --local-dir models/

# ESM C 300M base model (FP16)
huggingface-cli download daylight-00/esmc-300m-2024-12 --local-dir models/
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

Pre-computed ESM C 300M embeddings for all 7,282 IPD-IMGT/HLA alleles are required for Standard Mode:

```bash
huggingface-cli download daylight-00/emb_hla_esmc_small_0601_fp16 \
    --repo-type dataset \
    --local-dir data/emb_hla_esmc_small_0601_fp16
```

The embedding directory should resolve to `data/emb_hla_esmc_small_0601_fp16/`.

### 4. Configure GPU

The server uses the first GPU that CUDA exposes. To use another one, set `CUDA_VISIBLE_DEVICES` when starting it (for example `CUDA_VISIBLE_DEVICES=1`).

---

## Running the Server

```bash
uv run streamlit run serve.py \
    --server.port 8501 \
    --server.baseUrlPath /prepibind
```

The server will be available at `http://localhost:8501/prepibind`.

For production deployment behind a reverse proxy, set `--server.baseUrlPath` to match your proxy path prefix.

`serve.py` is the ASGI entry point: it hands `app.py` to Streamlit's `st.App` together with the middleware of `security_config.py`, which sets `Content-Security-Policy`, `X-Frame-Options: SAMEORIGIN`, `X-Content-Type-Options: nosniff`, `X-XSS-Protection: 1; mode=block` and `Referrer-Policy` on every response and answers `/robots.txt` and `<base path>/robots.txt` with the `robots.txt` of this directory (`Disallow: /`). These are what the SNU information-security office's scan (wscan) asks of a service on the university domain. `streamlit run app.py` still works but sends none of them.

Not in the app: the `Server` header (uvicorn sends `Server: uvicorn`, without a version; nginx removes it on the public route), HSTS (TLS ends in nginx) and cookie flags. The app sets no cookie: the XSRF token is the only one Streamlit has, its frontend reads it from JavaScript, so it cannot be HttpOnly (measured in Chromium: with an HttpOnly token every upload fails with 403) and without a login it protects nothing, so `.streamlit/config.toml` turns it off.

---

## Fast engine (TensorRT)

The engine has two paths. The **exact** path is the baseline's arithmetic in eager mode (bitwise equal to the baseline code in the same environment). The **fast** path computes every distinct epitope through ESM C once and every distinct HLA pair through the head's HLA encoder once, runs the joint blocks in batches of rows of one epitope length, and executes the three parts as TensorRT fp16 plans. On the four built-in test sets it is 3.4 to 6.0 times faster (48,352 rows in 6.4 s against 24.4 s), and the plans of all four heads take 1.8 GB of GPU memory (2.9 GB in total with the exact path's weights, which stay resident).

It is not bit-identical. Against the exact path 0.08 to 0.16 % of the rows change sign on the built-in sets, almost all within 0.15 logits of the decision threshold, and the AUC moves by at most 0.00004. Mixed epitope lengths differ by design: the exact path pads a batch to its longest epitope and the epitope encoder sees the padding, so the logit of a short epitope depends on the other epitopes of its batch; the fast path never pads the epitope side. Custom Mode, epitopes shorter than 8 or longer than 32 residues, HLA pairs longer than 200 residues and requests with more than 2,048 distinct HLA pairs always run on the exact path.

```bash
uv sync --python 3.14 --extra trt   # Python 3.14: NVIDIA publishes no TensorRT wheel for Python 3.15 yet (a plain `uv sync` picks 3.15 where it is installed)
python code/build_engines.py   # about 3 minutes, once per TensorRT, driver or GPU change or new checkpoints; writes models/engines/<tensorrt>-<gpu>-<graph version>/
```

`PREPIBIND_ENGINE` selects the path: `auto` (default) serves the fast path when matching plans are loaded and the request is in their ranges, otherwise the exact path; `fast` fails instead of falling back; `exact` never loads the plans. The service log says which path is active (the pages do not show it), and the result of `Engine.run` names its path in `attrs["engine"]`. The plans share one scratch buffer, so fast requests run one at a time. The GPU memory of a request does not grow with its size (epitope embeddings are held in blocks of at most 1 GiB, at most 2,048 distinct pairs at once); if the GPU still runs out of memory, TensorRT produces a non-finite logit, or TensorRT fails while running (after a driver change, say), the exact path takes over and the log says so (after a TensorRT failure for the rest of the process: rebuild the plans and restart). A plan directory belongs to one TensorRT version, GPU model and graph version, so a newer release never overwrites the plans an older one loads. Starting the server with `uv run` syncs the environment without extras and removes TensorRT: use `uv run --extra trt ...` or the environment's own `streamlit`.

---

## Tests

`tests/` has two tiers. The default tier needs no GPU, no checkpoints and no plans (the hard limits, how the engine routes a request and falls back, the staleness checks of the TensorRT plans, the security headers and robots.txt; about two seconds). The tier marked `gpu` runs the real engine of the tree (fast against exact, an injected out-of-memory, a refused request, the memory of a large request) and skips itself without a CUDA GPU, the checkpoints or matching plans. pytest is in the `test` dependency group, which a plain `uv sync` does not install:

```bash
uv run --python 3.14 --extra trt --group test pytest               # both tiers
uv run --python 3.14 --extra trt --group test pytest -m "not gpu"  # no GPU needed
```

---

## Model Architecture

The prediction model (`code/model.py`) takes two inputs — a pre-computed HLA embedding (concatenated alpha + beta chain from ESM C 300M) and a tokenized peptide sequence — and processes them through:

1. Separate two-block self-attention encoders for HLA and epitope representations
2. A single joint interaction block (concatenation + self-attention)
3. Masked mean pooling → two-layer MLP → scalar logit

The output score is sigmoid-transformed; scores ≥ 0.5 indicate predicted binding.

**Standard Mode** uses pre-computed embeddings stored in `data/emb_hla_esmc_small_0601_fp16/`.
**Custom HLA Mode** takes the HLA chains as sequences. A pair of chains that are both in the allele catalogue (`data/mhc_mapping.csv`, the same sequence) is scored exactly as Standard Mode scores the two alleles (the stored embeddings and, where it can, the fast path; the result table says `catalogue` in the column `HLA chains`). Any other chain (including non-human alleles such as Mamu, SLA, and BoLA) has its domain located by alignment to the catalogue (`code/locator.py`: a 4-mer index picks the nearest chains, a BLOSUM62 alignment with free end gaps chooses among them and carries the nearest chain's window across; Biopython), is embedded on-the-fly with ESM C and has the domain cut out, as the stored embeddings are; a domain pasted without its flanks gets the nearest chain's flanks added. The identity with the nearest chain over its window decides: 0.5 or more `located`, 0.4 to 0.5 `low confidence`, and Start and End typed on the page give a `manual window`. Below 0.4 no domain is found and the chain is embedded over its full length, as in earlier versions, marked `full chain` and not validated (its HLA input would be 254-266 positions instead of the 58-82 the model was trained on). Measured on the 6,400-row custom sample with every chain treated as unknown and its own entry or its whole gene hidden from the search: full chains differ from Standard Mode in 0.34 to 0.39 % of the signs (AUC 0.9245 against 0.9250; the earlier full-chain run: 8 %, 0.9153), domains pasted alone in 0.5 to 2.7 %; locating a chain takes 3 ms and embedding it 6 ms (`261115_img1_web_caps_scan_custom/custom_step2_check.json`).

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

Column headers must match exactly. Allele names must follow WHO HLA nomenclature and be present in `data/mhc_mapping.csv`. The model was trained on 15-mer peptides; other lengths up to 20 residues are accepted but not validated.

**Limits (hard, in `code/engine.py`):** an epitope of more than `MAX_EPITOPE_LENGTH` = 20 residues and, in Custom Mode, an HLA chain of more than `MAX_CHAIN_LENGTH` = 300 residues are refused by the pages with a message and by `Engine.run` / `Engine.run_custom` with a `RequestRefused` (a `ValueError`). Why 20: every built-in data set is 15-mers only, so the range cannot be read from the data; it was measured by keeping each test 15-mer as the core and adding random flanks (`261115_img1_web_caps_scan_custom/length_auc*.json`): the Qualitative AUC is 0.93 at 15 residues, 0.87 at 20 and 0.80 at 25 (a shuffled epitope gives 0.79), the share of non-binders scored above 0.5 rises from 15 % at 15 residues to 29 % at 20 and 49 % at 25, and each of the three heads measured (Qualitative, Mass Spectrometry, IC50 <500nM) steps down between 20 and 21 residues (AUC -0.036, -0.020 and -0.031). The chain limit is the allele catalogue's longest chain (285) rounded up. A test or a benchmark that needs longer input sets the two constants itself.

**Protein scan limits (`code/scan.py`):** at most 5,000 residues, 100 alleles and 500,000 predictions (windows x alleles x window lengths) per scan, 100,000 predictions while the fast engine is unavailable. They come from the measured throughput (`261115_img1_web_caps_scan_custom/scan_throughput.json`): the fast path scores 16,000 to 17,000 predictions per second from 13,000 to 1,200,000 predictions, with the GPU memory staying under 350 MiB over the engine's own; the exact path about 2,000 per second. 500,000 predictions are about 30 s, which is 5,000 residues against 100 alleles.

---

## Notes on Allele Coverage

The server provides pre-computed ESM C embeddings for **7,282 alleles** (838 alpha, 6,444 beta chains) from IPD-IMGT/HLA. However, the underlying models were trained on **151 unique HLA dimers** (27 alpha × 89 beta allele combinations) from IEDB. Predictions for alleles outside the training set rely on PLM-based zero-shot generalization.

Non-human primate (Mamu), porcine (SLA), and bovine (BoLA) MHC alleles evaluated in the original study are not available in Standard Mode; use Custom HLA Mode with explicit sequence input for these species.

---

## Citation

If you use PREpiBind or this web server, please cite:

```bibtex
@article{Jang2026PREpiBind,
  author  = {Jang, David Hyunyoo and Kim, Dongwoo and Park, Byungho
             and Hwang, Untaek and Choi, Yoonjoo and Lee, Juyong},
  title   = {{PREpiBind}: Protein Representation-integrated
             Epitope--{MHC} Class {II} Binding Prediction},
  journal = {TBD},
  year    = {2026},
}
```

The ESM C 300M model is developed by EvolutionaryScale and is subject to the [Cambrian Open License Agreement](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement).

---

## License

The PREpiBind web server source code is released under the [MIT License](LICENSE).
ESM C 300M model weights are subject to the Cambrian Open License (EvolutionaryScale).
