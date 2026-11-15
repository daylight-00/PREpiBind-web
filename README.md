# PREpiBind Web Server

**P**rotein **R**epresentation-integrated **Epi**tope–MHC Class II **Bind**ing Prediction

![banner](banner.png)

A Streamlit-based web server for predicting peptide binding to human and mouse MHC class II molecules using ESMC 300M protein language model embeddings. It runs the four task models of PREpiBind (qualitative binding, mass spectrometry ligandomics, thresholded IC50) and provides pre-computed embeddings for 7,282 alleles (838 alpha, 6,444 beta chains) from IPD-IMGT/HLA.

**Live server:** [https://lcdd.snu.ac.kr/prepibind](https://lcdd.snu.ac.kr/prepibind)

- **Prediction** and **Evaluation** of peptide–MHC class II pairs (CSV upload or table input)
- **Custom HLA Mode**: HLA chains given as sequences, including non-human alleles
- **Protein scan**: every window of a protein against a panel of alleles
- A TensorRT fast engine (optional) next to the exact PyTorch path

## Quick start

```bash
git clone https://github.com/daylight-00/PREpiBind-web.git && cd PREpiBind-web
uv sync
huggingface-cli download daylight-00/esmc-300m-2024-12 --local-dir models/
huggingface-cli download daylight-00/prepibind-demo --local-dir models/
huggingface-cli download daylight-00/emb_hla_esmc_small_0601_fp16 --repo-type dataset --local-dir data/emb_hla_esmc_small_0601_fp16
uv run streamlit run serve.py --server.port 8501 --server.baseUrlPath /prepibind
```

Then open `http://localhost:8501/prepibind`. Requirements (Python ≥ 3.14, an NVIDIA GPU, uv) and every step are in [docs/setup.md](docs/setup.md).

## Documentation

See [docs/](docs/README.md): [setup](docs/setup.md), [input, modes and limits](docs/usage.md), [model and fast engine](docs/engine.md), [security headers](docs/security.md), [tests](docs/testing.md).

## Repository layout

```text
PREpiBind-web/
├── app.py                        # The Streamlit script: model loading, session state, navigation
├── serve.py                      # ASGI entry point (st.App): app.py with the security headers
├── security_config.py            # The security-headers middleware and the robots.txt route
├── pyproject.toml, uv.lock       # Environment (uv)
├── tests/                        # pytest: a GPU-free tier and a tier marked gpu
├── docs/                         # Setup, usage, engine, security, tests
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
│   ├── embedder.py               # ESMC 300M epitope/HLA embedder (loads the released checkpoint)
│   ├── engine.py                 # Inference engine: four heads on one ESMC, HLA store, batching, scores
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
└── models/                       # Model checkpoints and TensorRT plans (download / build; see docs/setup.md)
```

## Citation

If you use PREpiBind or this web server, please cite the PREpiBind paper:

```bibtex
@article{jang2026prepibind,
  title   = {PREpiBind: Protein Representation-integrated Epitope-MHC Class II Binding Prediction},
  author  = {Jang, David Hyunyoo and Kim, Dongwoo and Hwang, Untaek and Park, Byungho and Choi, Yoonjoo and Lee, Juyong},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.15.751749},
  url     = {https://www.biorxiv.org/content/10.64898/2026.09.15.751749v1}
}
```

The ESMC 300M weights are released under the MIT License by Chan Zuckerberg Biohub ([`biohub/esmc-300m-2024-12`](https://huggingface.co/biohub/esmc-300m-2024-12)).

## License

The PREpiBind web server source code is released under the [MIT License](LICENSE).
The ESMC 300M weights are released under the MIT License (Chan Zuckerberg Biohub).

The data are not MIT: the tables under `data/` derive from IEDB (CC BY 4.0), IPD-IMGT/HLA (CC BY-NoDerivs) and UniProt (CC BY 4.0). Terms per file are in [data/LICENSE.md](data/LICENSE.md); attribution is in the [third-party notices](https://github.com/daylight-00/PREpiBind/blob/main/THIRD_PARTY_NOTICES.md) of the PREpiBind repository. Third-party models, weights and tools carry their own terms.
