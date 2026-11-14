"""A protein scan: every window of a protein against a panel of HLA class II alleles.
Everything here is plain Python and pandas (no GPU, no model): reading a protein or a FASTA file, the windows, the allele panel, and the hard limits of a scan. The page (pages/6_scan.py) does the rest through Engine.run."""
import re

import numpy as np
import pandas as pd

AMINO = "ACDEFGHIKLMNPQRSTVWYX"
DEFAULT_WINDOW = 15  # what the model was trained on
WINDOW_LENGTHS = tuple(range(12, 21))  # optional lengths: the engine refuses epitopes above 20 residues (engine.MAX_EPITOPE_LENGTH) and the scores fall off with the distance from 15
# The limits of a scan, from the measured throughput (261115_img1_web_caps_scan_custom/scan_throughput.json): the fast path scores 16,000-17,000 predictions per second from 13,000 to 1,200,000 predictions
# (GPU memory stays under 350 MiB over the engine's own), the exact path, which serves when the plans are unavailable, about 2,000. A scan may take about 30 s on the fast path: 500,000 predictions, which is
# 5,000 residues (the longest proteins of interest short of a coronavirus polyprotein) against 100 alleles; and about 50 s when only the exact path is available: 100,000 predictions.
MAX_PROTEIN_LENGTH = 5000  # residues
MAX_ALLELES = 100  # alleles (alpha and beta chain pairs) in the panel
MAX_SCAN_ROWS = 500_000  # predictions of one scan (windows x alleles) with the fast path
MAX_SCAN_ROWS_EXACT = 100_000  # the same without it

DR_ALPHA = "HLA-DRA*01:01"  # the alpha chain of HLA-DR is practically monomorphic: a DR beta chain alone names the dimer
# The 27-allele HLA class II reference panel (Greenbaum et al., Immunogenetics 2011): 16 DR (alpha DRA*01:01 with these beta chains), 6 DQ and 5 DP pairs.
REFERENCE_DR = ["DRB1*01:01", "DRB1*03:01", "DRB1*04:01", "DRB1*04:05", "DRB1*07:01", "DRB1*08:02", "DRB1*09:01", "DRB1*11:01", "DRB1*12:01", "DRB1*13:02", "DRB1*15:01", "DRB3*01:01", "DRB3*02:02", "DRB3*03:01", "DRB4*01:01", "DRB5*01:01"]
REFERENCE_DQ = [("DQA1*01:01", "DQB1*05:01"), ("DQA1*01:02", "DQB1*06:02"), ("DQA1*03:01", "DQB1*03:02"), ("DQA1*04:01", "DQB1*04:02"), ("DQA1*05:01", "DQB1*02:01"), ("DQA1*05:01", "DQB1*03:01")]
REFERENCE_DP = [("DPA1*01:03", "DPB1*02:01"), ("DPA1*01:03", "DPB1*04:01"), ("DPA1*02:01", "DPB1*01:01"), ("DPA1*02:01", "DPB1*05:01"), ("DPA1*03:01", "DPB1*04:02")]


class ScanRefused(ValueError):
    """The input is not a protein, or the scan is outside the limits; the message says which and is meant to be shown to the user as it is."""


def clean_protein(text):
    """Letters of a protein from pasted text: whitespace, digits, gaps and a trailing stop are dropped, case is ignored; anything else that is not an amino acid code (or X) is refused."""
    protein = re.sub(r"[\s\d\-]", "", text).upper().rstrip("*")
    bad = sorted(set(protein) - set(AMINO))
    if bad:
        raise ScanRefused(f"The sequence contains characters that are not amino acid codes: {' '.join(bad[:10])}")
    return protein


def read_proteins(text):
    """[(name, protein)] from FASTA (one record per '>' line) or from a plain sequence."""
    text = text.strip()
    if not text:
        return []
    if not text.startswith(">"):
        return [("sequence", clean_protein(text))]
    records, name, parts = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(parts)))
            name, parts = line[1:].strip() or "protein", []
        elif line:
            parts.append(line)
    records.append((name, "".join(parts)))
    found = [(name, clean_protein(sequence)) for name, sequence in records if sequence.strip()]
    if not found:
        raise ScanRefused("The FASTA text has a header but no sequence.")
    return found


def count_windows(protein_length, lengths):
    return sum(max(0, protein_length - w + 1) for w in set(lengths))


def check_scan(protein_length, lengths, n_alleles, fast=True):
    """Refuse what a scan does not take: an empty request, a protein, panel or number of predictions over the limits (`fast`: the engine has its TensorRT plans). Returns the number of windows."""
    if not lengths:
        raise ScanRefused("Choose at least one window length.")
    if n_alleles < 1:
        raise ScanRefused("Choose at least one allele.")
    if protein_length > MAX_PROTEIN_LENGTH:
        raise ScanRefused(f"The protein has {protein_length:,} residues; a scan takes at most {MAX_PROTEIN_LENGTH:,}.")
    if n_alleles > MAX_ALLELES:
        raise ScanRefused(f"The panel has {n_alleles} alleles; a scan takes at most {MAX_ALLELES}.")
    n = count_windows(protein_length, lengths)
    if n == 0:
        raise ScanRefused(f"The protein ({protein_length} residues) is shorter than the shortest window ({min(lengths)}).")
    limit = MAX_SCAN_ROWS if fast else MAX_SCAN_ROWS_EXACT
    if n * n_alleles > limit:
        reason = "" if fast else " while the fast engine is not available"
        raise ScanRefused(f"This scan would make {n * n_alleles:,} predictions ({n:,} windows x {n_alleles} alleles); a scan may make at most {limit:,}{reason}. Use a shorter protein, fewer window lengths or fewer alleles.")
    return n


def reference_panel(names):
    """The 27-allele reference panel, restricted to the alleles `names` (the catalogue's) knows: [(alpha, beta)]."""
    pairs = [(DR_ALPHA, f"HLA-{beta}") for beta in REFERENCE_DR] + [(f"HLA-{a}", f"HLA-{b}") for a, b in REFERENCE_DQ + REFERENCE_DP]
    return [(a, b) for a, b in pairs if a in names and b in names]


def drb1_panel(names):
    """The 11 DRB1 alleles of the reference panel with DRA*01:01."""
    return [(DR_ALPHA, f"HLA-{beta}") for beta in REFERENCE_DR if beta.startswith("DRB1") and DR_ALPHA in names and f"HLA-{beta}" in names]


def parse_panel(text, alpha_names, beta_names):
    """One allele pair per line, 'alpha, beta' (also separated by / _ ; or spaces), or a single DR beta chain (HLA-DRB1*15:01), which takes HLA-DRA*01:01 as its alpha chain. Names must be in the catalogue."""
    pairs, unknown = [], []
    for line in text.splitlines():
        parts = [p for p in re.split(r"[,;/_\s]+", line.strip()) if p]
        if not parts:
            continue
        if len(parts) == 1 and parts[0].startswith("HLA-DRB"):
            parts = [DR_ALPHA, parts[0]]
        if len(parts) != 2:
            raise ScanRefused(f"'{line.strip()}' is not an allele pair: write 'alpha chain, beta chain' (for HLA-DR a beta chain alone is enough).")
        alpha, beta = parts
        unknown += [name for name, known in ((alpha, alpha_names), (beta, beta_names)) if name not in known]
        pairs.append((alpha, beta))
    if unknown:
        raise ScanRefused(f"Not in the allele catalogue: {', '.join(dict.fromkeys(unknown))}")
    return list(dict.fromkeys(pairs))


def short_label(alpha, beta):
    """The allele pair as a short label: the beta chain alone for HLA-DR, both chains otherwise."""
    alpha, beta = alpha.removeprefix("HLA-"), beta.removeprefix("HLA-")
    return beta if alpha == DR_ALPHA.removeprefix("HLA-") else f"{alpha}/{beta}"


def scan_frame(protein, lengths, pairs):
    """The rows of a scan as an Engine.run request: MHC = 'alpha_beta', Epitope = the window, plus Start (1-based) and Length. Windows run fastest through the engine when the rows of one window are together."""
    starts, peptides = [], []
    for w in sorted(set(lengths)):
        for i in range(len(protein) - w + 1):
            starts.append(i + 1)
            peptides.append(protein[i:i + w])
    k = len(pairs)
    return pd.DataFrame({
        "MHC": [f"{a}_{b}" for a, b in pairs] * len(peptides),
        "Epitope": np.repeat(np.array(peptides, dtype=object), k),
        "Start": np.repeat(np.array(starts, dtype=np.int32), k),
        "Length": np.repeat(np.array([len(p) for p in peptides], dtype=np.int16), k),
    })
