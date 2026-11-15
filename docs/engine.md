# Model and engine

## Model architecture

The prediction model (`code/model.py`) takes two inputs — a pre-computed HLA embedding (concatenated alpha + beta chain from ESMC 300M) and a tokenized peptide sequence — and processes them through:

1. Separate two-block self-attention encoders for HLA and epitope representations
2. A single joint interaction block (concatenation + self-attention)
3. Masked mean pooling → two-layer MLP → scalar logit

The output score is sigmoid-transformed; scores ≥ 0.5 indicate predicted binding.

## Fast engine (TensorRT)

The engine has two paths. The **exact** path is the baseline's arithmetic in eager mode (bitwise equal to the original implementation's arithmetic in the same environment). The **fast** path computes every distinct epitope through ESMC once and every distinct HLA pair through the head's HLA encoder once, runs the joint blocks in batches of rows of one epitope length, and executes the three parts as TensorRT fp16 plans. On the four built-in test sets it is 3.4 to 6.0 times faster (48,352 rows in 6.4 s against 24.4 s), and the plans of all four heads take 1.8 GB of GPU memory (2.9 GB in total with the exact path's weights, which stay resident).

It is not bit-identical. Against the exact path 0.05 to 0.15 % of the rows change sign on the built-in sets, almost all within 0.15 logits of the decision threshold, and the AUC moves by at most 0.00005. Mixed epitope lengths differ by design: the exact path pads a batch to its longest epitope and the epitope encoder sees the padding, so the logit of a short epitope depends on the other epitopes of its batch; the fast path never pads the epitope side. Custom Mode, epitopes shorter than 8 or longer than 32 residues, HLA pairs longer than 200 residues and requests with more than 2,048 distinct HLA pairs always run on the exact path.

```bash
uv sync --python 3.14 --extra trt   # Python 3.14: NVIDIA publishes no TensorRT wheel for Python 3.15 yet (a plain `uv sync` picks 3.15 where it is installed)
python code/build_engines.py   # about 3 minutes, once per TensorRT, driver or GPU change or new checkpoints; writes models/engines/<tensorrt>-<gpu>-<graph version>/
```

`PREPIBIND_ENGINE` selects the path: `auto` (default) serves the fast path when matching plans are loaded and the request is in their ranges, otherwise the exact path; `fast` fails instead of falling back; `exact` never loads the plans. The service log says which path is active (the pages do not show it), and the result of `Engine.run` names its path in `attrs["engine"]`. The plans share one scratch buffer, so fast requests run one at a time. The GPU memory of a request does not grow with its size (epitope embeddings are held in blocks of at most 1 GiB, at most 2,048 distinct pairs at once); if the GPU still runs out of memory, TensorRT produces a non-finite logit, or TensorRT fails while running (after a driver change, say), the exact path takes over and the log says so (after a TensorRT failure for the rest of the process: rebuild the plans and restart). A plan directory belongs to one TensorRT version, GPU model and graph version, so a newer release never overwrites the plans an older one loads. Starting the server with `uv run` syncs the environment without extras and removes TensorRT: use `uv run --extra trt ...` or the environment's own `streamlit`.
