# Setup

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

## Install

### 1. Clone the repository and create the environment

```bash
git clone https://github.com/daylight-00/PREpiBind-web.git
cd PREpiBind-web
uv sync
```

### 2. Model checkpoints

Download the ESMC 300M base model (FP16) and the four task checkpoints released with the paper (float16 copies) from HuggingFace and place them in `models/`:

```bash
huggingface-cli download daylight-00/esmc-300m-2024-12 --local-dir models/
huggingface-cli download daylight-00/prepibind-demo --local-dir models/
```

Each task checkpoint is one run: the one with the lowest validation loss among the 15 runs (3 seeds x 5 folds) the paper reports for that task, chosen without any test-set quantity; the seed and the fold are in the file name. The paper's numbers are means over the 15 runs, so a single checkpoint does not reproduce them exactly. [`daylight-00/prepibind`](https://huggingface.co/daylight-00/prepibind) holds the float32 originals. The server expects these files in `models/`:

```text
models/
├── esmc_300m_2024_12_v0_fp16.pth
├── prepibind_qualitative_s100_f0_fp16.pt       # Qualitative
├── prepibind_ms_s128_f3_fp16.pt                # Mass Spectrometry
├── prepibind_ic50_500_s128_f2_fp16.pt          # IC50 <500 nM
└── prepibind_ic50_1000_s42_f1_fp16.pt          # IC50 <1000 nM
```

### 3. Download pre-computed HLA embeddings

Pre-computed ESMC 300M embeddings for all 7,282 IPD-IMGT/HLA alleles are required for Standard Mode:

```bash
huggingface-cli download daylight-00/emb_hla_esmc_small_0601_fp16 \
    --repo-type dataset \
    --local-dir data/emb_hla_esmc_small_0601_fp16
```

The embedding directory should resolve to `data/emb_hla_esmc_small_0601_fp16/`.

### 4. Configure GPU

The server uses the first GPU that CUDA exposes. To use another one, set `CUDA_VISIBLE_DEVICES` when starting it (for example `CUDA_VISIBLE_DEVICES=1`).

## Running the server

```bash
uv run streamlit run serve.py \
    --server.port 8501 \
    --server.baseUrlPath /prepibind
```

The server will be available at `http://localhost:8501/prepibind`.

For production deployment behind a reverse proxy, set `--server.baseUrlPath` to match your proxy path prefix.

What `serve.py` adds (security headers, robots.txt, the XSRF setting) is in [security.md](security.md).
