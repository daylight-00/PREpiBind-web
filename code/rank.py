"""%Rank against the per-head background grid, and the strong / weak bands.

`img2-server-output-contract.md` §1: the server reports **%Rank, per head**. The reason is not
cosmetic. A raw logit of 2.0 at length 15 is 3.30 % on MS and 0.32 % on IC50 <500 nM — a tenfold
difference in rarity behind the same displayed number, on heads the interface shows side by side.
The cross-*allele* spread that first justified building %Rank at all is only 2.0x.

A grid belongs to one head, so there are three, and they are not interchangeable. Each is keyed
`<molecule>|<length>` over the 306-molecule panel and holds 1,000 descending background
breakpoints per stratum.

`PercentileGrid` itself is `code/scan.py`, vendored from IMG so the page and the paper cannot drift.
"""
import numpy as np

import artifacts
import scan

_grids = {}


def load(head):
    """The merged human + mouse grid for one head. Cache this; it is ~16 MB per head."""
    if head not in _grids:
        merged = {}
        for family in ("human", "h2"):
            with np.load(artifacts.grid(head, family), allow_pickle=True) as z:
                merged.update({k: z[k] for k in z.files})
        _grids[head] = scan.PercentileGrid(path=None, _g=merged)
    return _grids[head]


def molecules(head):
    """The panel: every molecule this head has a background for."""
    return sorted({k.rsplit("|", 1)[0] for k in load(head)._g})


def on_panel(head, molecule):
    return any(k.startswith(molecule + "|") for k in load(head)._g)


def rank(head, molecule, length, logits):
    """Percent of the background scoring at least as high. Lower is a stronger binder.

    None when the molecule is off-panel — it has no background, and under
    `img2-server-output-contract.md` §3 one is built for it on first request rather than the score
    being reported bare.
    """
    return load(head).rank(molecule, int(length), logits)


def resolution(head):
    """The finest %Rank a grid can express: 0.1 % for the 1,000-breakpoint grids."""
    return 100.0 / load(head).n_breaks


def band(pct, bands):
    """'Strong' / 'Weak' / None, from the cut in BANDS.json. Never from a hardcoded number."""
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return None
    cut = bands["bands"]
    if pct <= cut["strong_rank_pct_max"]:
        return "Strong"
    if pct <= cut["weak_rank_pct_max"]:
        return "Weak"
    return None


def format_pct(pct, res):
    """`<0.1` rather than `0.00`. A returned 0.0 means 'better than every breakpoint'.

    A percentile of zero is not a thing, and printing one claims a resolution the grid does not have.
    """
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return ""
    return f"<{res:g}" if pct < res else f"{pct:g}"
