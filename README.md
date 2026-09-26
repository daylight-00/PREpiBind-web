# PREpiBind Web Server

**P**rotein **R**epresentation-integrated **Epi**tope–MHC Class II **Bind**ing Prediction

![banner](banner.png)

> **PREpiBind Webserver**
> Jang DH, Kim D, Choi Y, Lee J.
> *TBD* (2026)

A Streamlit web server for predicting peptide binding to human (HLA-DP/DQ/DR) and mouse (H2-IA/IE)
MHC class II molecules, using ESM C 300M protein language model embeddings.

This branch is **img2**, a reboot of the earlier server rather than an update of it. Three things
differ from the version a returning user remembers, and each is a decision rather than a preference:

| | img1 | **img2** |
|---|---|---|
| served weights | one cross-validation fold per task, three different seeds | one fit on the whole development pool at seed 42, frozen and hashed |
| output | sigmoid score, binder at ≥ 0.5 | **%Rank** against a fixed background, per head; the binary call is a %Rank cut |
| peptide length | trained on 15-mers | trained 12–25, validated 12–21, accepted 12–25 |
| heads | Qualitative, MS, IC50 ×2 | MS (headline), IC50 ×2. Qualitative is retired to a clearly labelled legacy page |

**PREpiBind does not outperform the established class-II predictors it was measured against.** On
the MS head, over held-out studies and held-out molecules, it is behind NetMHCIIpan-4.3 by 0.0305
and 0.0491 ROC-AUC with 95 % intervals excluding zero; it is at parity with MixMHC2pred-2.0 over
held-out studies and behind over held-out molecules. The server says so on its own front page.

---

## Repository contents

```text
PREpiBind-web/
├── app.py                        # entry point: navigation, cached engine, shared footer
├── security_config.py            # HTTP security headers — SEE THE WARNING BELOW
├── robots.txt
├── .streamlit/config.toml
│
├── pages/
│   ├── 0_home.py                 # what the server is, and what it does not claim
│   ├── 1_prediction.py           # peptide x molecule -> %Rank per head
│   ├── 2_evaluation.py           # benchmarking a labelled set
│   ├── 3_instructions.py         # input, output semantics, limitations
│   ├── 4_about.py                # the full performance record
│   ├── 4_scan.py                 # antigen scanning
│   ├── 5_legacy.py               # the img1 checkpoints, for reproducing old numbers
│   └── 6_custom.py               # custom HLA chains
│
├── code/
│   ├── artifacts.py              # served checkpoints, their manifests, the %Rank bands
│   ├── chains.py                 # which sequence and embedding a chain resolves to
│   ├── rank.py                   # %Rank and the strong/weak bands
│   ├── engine.py                 # one shared ESM C, the heads, scoring
│   ├── grids.py                  # building a %Rank background on demand
│   ├── domain.py                 # locating the binding domain in an uncatalogued chain
│   ├── independence.py           # is an uploaded set independent of the training lineage?
│   ├── model.py                  # the architecture
│   ├── scan.py                   # VENDORED, see below
│   ├── selftest.py               # startup checks — run this first
│   └── VENDORED.json
│
├── data/                         # symlinks into run directories; none of it is in the repo
└── models/                       # checkpoints; none of it is in the repo
```

### `code/scan.py` is vendored

It is copied from the private research tree, where it is the module every canonical scan number was
produced by, so that the published server and the measurement path cannot drift apart. Its hash is
recorded in `code/VENDORED.json` and `artifacts.check_vendored()` fails loudly if the copy changes.
Do not edit it here; edit it upstream and re-vendor.

---

## Requirements

- Python ≥ 3.10, a CUDA GPU (developed on an RTX 4070 Ti, 12 GB)

| package | version tested |
| --- | --- |
| streamlit | 1.55.0 |
| torch | 2.7.1+cu128 |
| esm | 3.2.0 |
| flash-attn | 2.8.0.post2 |
| numpy | 2.3.0 |
| pandas | 2.3.0 |
| plotly | 6.1.2 |
| scikit-learn | 1.7.0 |
| h5py | 3.14.0 |
| biopython | 1.85 |

---

## Artifacts the server needs

None of these are in the repository. `data/` and `models/` hold symlinks to them.

| path | what it is |
| --- | --- |
| `models/esmc_300m_2024_12_v0_fp16.pth` | ESM C 300M, float16 |
| `models/prepi_esmc_small_img2_{ms,ic50_500,ic50_1000}_v1.0_fp16.pth` | the three served heads, each with a `_manifest.json` beside it |
| `data/grids/grid_b_{head}_{human,h2}.npz` | the %Rank backgrounds: 306 molecules per head |
| `data/BANDS.json` | the strong/weak cut **and the evidence that qualifies it** |
| `data/chain_table.csv` | the training chain table; defines 134 chains and their domain windows |
| `data/emb_hla_chain_table_0329.h5` | the embedding store those 134 chains were trained with |
| `data/mhc_mapping.csv` | the 7,282-chain catalogue (in the repo) |
| `data/lineage/` | per-head training-lineage index for the independence audit |
| `data/emb_hla_esmc_small_0601_fp16/` | per-chain embeddings for the catalogue |
| `data/background_{human,mouse}.txt` | the 100,000-peptide proteome background |

The img2 artifacts are not published yet. The img1 checkpoints and catalogue embeddings are on
HuggingFace:

```bash
huggingface-cli download daylight-00/esmc-300m-2024-12 --local-dir models/
huggingface-cli download daylight-00/emb_hla_esmc_small_0601_fp16 \
    --repo-type dataset --local-dir data/emb_hla_esmc_small_0601_fp16
```

### Two rules about the artifacts, both load-bearing

- **A %Rank grid belongs to one head, one host and one batch size.** It is a set of scores, not a
  property of the model. `code/engine.py` pins the embedding batch size to the value the grids were
  built at; it is not a tuning knob.
- **The chain table wins for the 134 chains it defines; `mhc_mapping.csv` is the source for the
  rest.** Six chains disagree between the two, and the invariant being protected is per molecule,
  not global: the sequence that produced a query embedding and the sequence behind that molecule's
  %Rank background must be the same one.

---

## Running

```bash
PREPIBIND_GPU=0 streamlit run app.py --server.port 8501
python code/selftest.py          # 19 checks: artifact hashes, the chain split, the panel, the bands
```

`PREPIBIND_GPU` selects the CUDA device; the img1 server hardcoded device 1 in `app.py`.

Behind a reverse proxy, add `--server.baseUrlPath` to match the proxy prefix.

### Security headers

`security_config.py` monkey-patches Tornado to set `X-Content-Type-Options`, `X-Frame-Options` and
`Referrer-Policy`. **Measured on streamlit 1.55.0 / tornado 6.5.1, it sets none of them** — a server
running under it answers every path with no security header and with `Server: TornadoServer/6.5.1`
still set. These headers belong at the reverse proxy in any case, which is where TLS terminates and
where HSTS has to live. The module is left in place so that removing it is a deliberate change made
together with the nginx block that replaces it.

---

## What the server reports

A **%Rank** per head: the percentage of a fixed 100,000-peptide background — drawn from the
reviewed proteome and stratified by `(molecule, length)` — that scores at least as high. Lower is a
stronger binder, and `<0.1` means "better than every breakpoint", not zero.

**Strong ≤ 1 %, weak ≤ 5 %**, matching NetMHCIIpan-4.3's installed defaults so the two tools are
read at the same operating point. This is a comparability choice, not a calibration. The bands are
read from `data/BANDS.json` rather than hardcoded, because that file carries the limits with the
numbers — in particular that its negatives are matched decoys, 1:1 with positives, **not a natural
proteome**, so the negative rate is not a specificity.

There is no probability output and no probability threshold. The three heads are not on one scale: a
raw logit of 2.0 at length 15 is the 3.30th percentile on MS and the 0.32nd on IC50 < 500 nM, which
is why the server reports ranks and not scores.

### Molecules without a background

The panel is 306 molecules per head; the two chain lists compose about 1.36 million, so an off-panel
request is the common case. Precomputing all of it is roughly 3.4 GPU-years for one head. The server
therefore builds a background on first request and caches it — about 150 s for one head on the
serving card, or 306 s for all three, of which 73 s is the shared embedding pass. Until one exists
the %Rank is left blank rather than showing a bare logit in its place.

---

## Input format

```csv
MHC_alpha,MHC_beta,Epitope
HLA-DRA*01:01,HLA-DRB1*15:01,GELIGILNAAKVPAD
HLA-DQA1*05:01,HLA-DQB1*02:01,PKYVKQNTLKLATAA
```

Add a `Target` column of 1/0 for the Evaluation page. Headers must match exactly and allele names
must be in `data/mhc_mapping.csv`.

### Independence from training

The Evaluation page audits an uploaded set against **what the served head was fitted on** — exact
peptide, exact peptide+MHC, any shared 9-mer, molecule seen/unseen, length support — and recomputes
the metric on each progressively cleaner view. Overlap is reported **split by label**, because that
is the failure mode that actually distorts a number.

The lineage index is per head. Pooling the three answers a different question: `test_ms.csv`'s
negatives show 36.6 % peptide+MHC overlap against a pooled index and **0.0 %** against the MS head's
own, since the IC50 arms contain them.

**The molecule-wise column is the one to read.** Dropping overlapping rows also changes which
molecules, lengths and class balance remain, so a pooled figure that moves between views has moved
for two reasons at once. An average over per-molecule AUCs does not move when the molecule mixture
does. A pooled −%Rank column is given beside it (comparable across molecules by construction,
defined only where a background exists) and the pooled logit is kept as a labelled diagnostic.

Exact peptide+MHC hits are broken out as a 2×2 against the **training** label, because a test
positive that was a training positive is a memory test while one that was a training *decoy* is a
contradiction, and a single overlap rate hides the difference. 9-mer overlap is reported twice:
against any molecule (sequence familiarity) and against the **same** molecule, which is far closer
to binding-context leakage since an MHC-II core is nine residues.

This checks PREpiBind's lineage only. It says nothing about whether a set is independent of
NetMHCIIpan or MixMHC2pred, whose training data we do not have.

**The built-in test sets in `data/` cannot measure img2 generalisation.** Both IC50 sets are 100 %
inside their training pool at peptide level, and `test_ms.csv` is 80.7 % of its *positives* against
~0 % of its negatives — the asymmetry that inflates an AUC rather than merely leaking. img2's heads
are one fit on the whole development pool, so an img1 split is not held out from them. They remain
valid input for the legacy page, whose checkpoints they genuinely were held out from.

---

## Custom HLA mode

Paste **full-length** α and β chains. The server locates the binding domain itself by aligning to
the catalogue chain the sequence most resembles, because a stored HLA embedding is the full-length
chain embedded and *then* sliced — embedding a pre-trimmed window on its own is a different object.
Measured against the catalogue path:

| what the sequence becomes | ROC-AUC | mean \|Δ%Rank\| | band calls changed |
| --- | --- | --- | --- |
| full-length, sliced to the window *(this server)* | −0.002 | 0.45 pp | 2 % |
| full-length, left whole *(img1)* | −0.017 | 3.01 pp | 20 % |
| the window, embedded alone *(img1 also invited this)* | −0.111 | 14.11 pp | 39 % |

The img1 analysis measured the AUC column and concluded users need not trim, which was right for a
server reporting a sigmoid — an AUC reads order within a set. A percentile reads position against a
background, and that is where the shift lands.

---

## Provenance note

The served manifests record `"loader": "PREpiBind-web/code/inference.py:62"`. That file was the img1
inference stack and is not in this branch; `code/engine.py` is its successor and loads the same way,
`torch.load(...)["model_state_dict"]`. The manifests are frozen artifacts and are not edited.

---

## Citation

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

The img2 manuscript is not yet written; a citation will be added on submission.

---

## License

MIT ([LICENSE](LICENSE)). ESM C 300M weights and embeddings derived from them are subject to the
[Cambrian Open License](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement)
(EvolutionaryScale).

Research use only. Not for clinical or diagnostic decision-making.
