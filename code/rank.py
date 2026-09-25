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
_built = {}


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
    """In the precomputed 306-molecule panel. Distinct from `has_background`."""
    return any(k.startswith(molecule + "|") for k in load(head)._g)


def has_background(head, molecule):
    """Rankable at all — on the panel, or with a grid already built on demand."""
    return on_panel(head, molecule) or _cached(head, molecule) is not None


def _cached(head, molecule):
    """A grid built on demand for an off-panel molecule, if one exists. See `grids.py`."""
    import grids

    key = (head, molecule)
    if key not in _built:
        if not grids.cached(molecule, head):
            _built[key] = None
        else:
            with np.load(grids.path(molecule, head), allow_pickle=True) as z:
                _built[key] = scan.PercentileGrid(path=None, _g={k: z[k] for k in z.files})
    return _built[key]


def invalidate(head, molecule):
    """Forget a cached grid — call after building one so the next query sees it."""
    _built.pop((head, molecule), None)


def rank(head, molecule, length, logits):
    """Percent of the background scoring at least as high. Lower is a stronger binder.

    Falls back to a grid built on demand for this molecule. None when neither exists: the molecule
    has no background, and under `img2-server-output-contract.md` §3 a bare logit is not offered in
    its place, because it is not comparable across alleles, lengths or heads.
    """
    out = load(head).rank(molecule, int(length), logits)
    if out is not None:
        return out
    grid = _cached(head, molecule)
    return grid.rank(molecule, int(length), logits) if grid is not None else None


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
