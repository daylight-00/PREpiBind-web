"""The served img2 artifacts: three heads, their frozen manifests, and the %Rank bands.

Every path here is a symlink into a run directory — the repository carries none of the binaries.
`verify()` re-hashes each served checkpoint against the manifest frozen beside it, so a swapped or
truncated file is caught at startup instead of inside a prediction.

The bands are READ from `data/BANDS.json` and never written here. `IMG/pipeline/freeze/README.md`
says why: the file carries the limits that qualify the numbers — in particular that its negatives
are matched decoys, not a natural proteome, so the negative rate is not a specificity. A band copied
out as a bare `1.0` loses that, and the loss is invisible.

  IMG/docs/decisions/img2-server-output-contract.md  §1 (%Rank per head), §2 (the bands)
  IMG/docs/decisions/img2-served-artifacts.md        (which checkpoints)
"""
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ordered: MS leads and the IC50 heads are appendix weight (img2-server-scope.md §1).
# ql is deliberately absent from the main path (img2-server-output-contract.md §4); the legacy
# page loads the img1 checkpoints itself and must label them as img1 on screen.
HEADS = {
    "ms": "Mass spectrometry (eluted ligand)",
    "ic50_500": "IC50 < 500 nM",
    "ic50_1000": "IC50 < 1000 nM",
}
HEADLINE = "ms"

ESM_CHECKPOINT = os.path.join(ROOT, "models", "esmc_300m_2024_12_v0_fp16.pth")


def _stem(head):
    return f"prepi_esmc_small_img2_{head}_v1.0_fp16"


def checkpoint(head):
    return os.path.join(ROOT, "models", _stem(head) + ".pth")


def manifest(head):
    with open(os.path.join(ROOT, "models", _stem(head) + "_manifest.json")) as fh:
        return json.load(fh)


def grid(head, family):
    """`family` is 'human' or 'h2' — the two were scored against different backgrounds."""
    return os.path.join(ROOT, "data", "grids", f"grid_b_{head}_{family}.npz")


def bands():
    """The strong/weak cut, WITH the evidence and limits recorded alongside it."""
    with open(os.path.join(ROOT, "data", "BANDS.json")) as fh:
        return json.load(fh)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify():
    """{head: (ok, sha256)} — each served checkpoint against its own manifest."""
    return {
        head: ((got := sha256(checkpoint(head))) == manifest(head)["served_sha256"], got)
        for head in HEADS
    }


def check_vendored():
    """{path: (ok, sha256)} for every file copied in from IMG. See code/VENDORED.json."""
    with open(os.path.join(ROOT, "code", "VENDORED.json")) as fh:
        want = json.load(fh)
    return {
        path: ((got := sha256(os.path.join(ROOT, path))) == spec["sha256"], got)
        for path, spec in want.items()
    }
