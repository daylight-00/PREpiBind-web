"""Antigen scanning: one protein x a panel of MHC-II alleles -> a binding landscape.

Design and the measurements behind it:
IMG/docs/notes/2026-09-05-antigen-scanning-design.md. Three of them are load-bearing and are
enforced here rather than left to the caller:

  1. A window CANNOT be sliced out of a whole-protein embedding. Measured: per-window cosine 0.79,
     mean |delta logit| 1.75 against a logit spread of 1.59, Spearman 0.42, top-20 overlap 2/20.
     Every window is embedded as its own peptide. `embed_windows` has no protein-level shortcut and
     is not going to grow one.

  2. Batched embedding is 38x faster and preserves order -- mean |delta logit| 0.030, i.e. 1.9% of
     the logit spread, Spearman 0.9996, top-20 overlap 20/20. So batching is used, and the %Rank
     background MUST be built through this same function so the offset cancels.

  3. Batches are grouped BY LENGTH and always at the SAME batch size as the %Rank background.
     Padding is not the reason -- measured, ESMC builds its own mask when `sequence_id` is omitted,
     so a padded mixed-length batch costs nothing (2026-09-07-batch-size-not-padding.md). BATCH SIZE
     is the reason: 64 vs 128 moves a logit by 1.2% of its SD and a %Rank by 0.32 percentile points,
     six times the entire fp16 effect. Grouping is free and keeps the batch shapes reproducible.

No stride: every start at every length. A stride would destroy the core-support signal, which is the
one thing multi-length scanning gets that a 15-mer scan cannot.

%Rank is an interface here, not an implementation. `PercentileGrid` returns None until the grid
built per IMG/docs/decisions/img2-server-scope.md exists; every downstream function falls back to
the raw logit and says so, so the scan is usable now and gains comparability later without a rewrite.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CORE = 9                      # MHC-II binding core; also the K of the no-shared-9mer analysis
DEFAULT_LENGTHS = tuple(range(12, 22))      # the validated range
EXTENDED_LENGTHS = tuple(range(12, 26))     # trained range, limited validation support

# The server's %Rank bands -- IMG/docs/decisions/img2-server-output-contract.md §2. Matching
# NetMHCIIpan-4.3's installed defaults (-rankS 1.0, -rankW 5.0) for comparability; on our own
# development pool they also sit within 0.2 points of the 50% / 80% ligand-recovery points.
# Keep these and the server in step: a scan that draws a different line than the prediction page
# is the kind of inconsistency a reviewer finds and a user never reports.
STRONG_RANK_PCT = 1.0
WEAK_RANK_PCT = 5.0
STD_AA = set('ACDEFGHIKLMNPQRSTVWY')


# --------------------------------------------------------------------------- windows
def generate_windows(seq: str, lengths=DEFAULT_LENGTHS) -> pd.DataFrame:
    """Every window at every length. Returns start (1-based), end, length, peptide.

    Non-standard residues (B J O U X Z and anything else) are not scoreable -- ESMC would accept
    them but no training peptide contained one -- so windows touching them are dropped and counted.
    """
    seq = ''.join(seq.split()).upper()
    rows = []
    for L in lengths:
        for i in range(len(seq) - L + 1):
            pep = seq[i:i + L]
            if set(pep) <= STD_AA:
                rows.append((i + 1, i + L, L, pep))
    df = pd.DataFrame(rows, columns=['start', 'end', 'length', 'peptide'])
    return df.sort_values(['start', 'length']).reset_index(drop=True)


def window_count(n: int, lengths=DEFAULT_LENGTHS) -> int:
    return sum(max(0, n - L + 1) for L in lengths)


# --------------------------------------------------------------------------- embedding
def embed_windows(peptides, client, batch_size: int = 64, progress=None) -> dict:
    """Embed each peptide as an isolated peptide, batched, grouped by length (no padding).

    `client` is an ESMC model. Returns {peptide: (L, 960) float32}. Duplicates are embedded once.
    """
    import torch
    uniq = sorted(set(peptides))
    tok = client.tokenizer
    pad = tok.pad_token_id
    dev = next(client.parameters()).device
    out = {}
    by_len = {}
    for p in uniq:
        by_len.setdefault(len(p), []).append(p)
    done = 0
    for L, peps in sorted(by_len.items()):
        for i in range(0, len(peps), batch_size):
            chunk = peps[i:i + batch_size]
            ids = [torch.tensor(tok.encode(p)) for p in chunk]
            width = max(len(x) for x in ids)          # equal by construction; kept for safety
            b = torch.full((len(ids), width), pad, dtype=torch.long)
            for j, t in enumerate(ids):
                b[j, :len(t)] = t
            m = (b != pad).to(dev)
            with torch.no_grad():
                o = client(sequence_tokens=b.to(dev), sequence_id=m)
            e = o.embeddings.detach().float().cpu().numpy()
            for j, p in enumerate(chunk):
                out[p] = e[j, 1:1 + len(p), :]        # strip BOS/EOS
            done += len(chunk)
            if progress:
                progress(done, len(uniq))
    return out


# --------------------------------------------------------------------------- alleles
def load_hla_embeddings(allele_names, hla_h5: str, mhc_table: str) -> dict:
    """{allele -> (L, 960)} for '<beta>_<alpha>' or single-chain names, sliced as in training."""
    import h5py
    tbl = pd.read_csv(mhc_table)
    seqmap = dict(zip(tbl.iloc[:, 0], tbl.iloc[:, 1]))
    out = {}
    with h5py.File(hla_h5, 'r') as f:
        for name in allele_names:
            parts = name.split('_')
            chunks = []
            for chain in parts:
                if chain not in seqmap:
                    raise KeyError(f'{chain} not in {mhc_table}')
                s = str(seqmap[chain])
                if '|' in s:
                    _, a, b = s.split('|')
                    a, b = int(a), int(b)
                else:
                    a = b = None
                e = np.squeeze(f[chain][()])
                chunks.append(e[a:b] if a is not None else e)
            out[name] = np.concatenate(chunks, axis=0).astype(np.float32)
    return out


# --------------------------------------------------------------------------- scoring
def score_windows(windows: pd.DataFrame, hla_emb: dict, epi_emb: dict, head,
                  batch_size: int = 256, progress=None) -> pd.DataFrame:
    """Cartesian product of windows x alleles. Returns the long table with a `logit` column."""
    import torch
    dev = next(head.parameters()).device
    frames = []
    total = len(windows) * len(hla_emb)
    done = 0
    for allele, h in hla_emb.items():
        xh = torch.tensor(h, dtype=torch.float32, device=dev).unsqueeze(0)
        mh = torch.zeros(1, xh.shape[1], dtype=torch.bool, device=dev)
        logits = np.empty(len(windows), dtype=np.float32)
        # batch WITHIN a length: a batch straddling two lengths could not be stacked, and padding
        # the epitope side is the exact failure this project already fixed in the model.
        for L, grp in windows.groupby('length', sort=True):
            idx_all = grp.index.to_numpy()
            for i in range(0, len(idx_all), batch_size):
                idx = idx_all[i:i + batch_size]
                arr = np.stack([epi_emb[p] for p in windows.loc[idx, 'peptide']])
                xe = torch.tensor(arr, dtype=torch.float32, device=dev)
                n = len(idx)
                with torch.no_grad():
                    o = head(xh.expand(n, -1, -1), False, xe, False,
                             mh.expand(n, -1),
                             torch.zeros(n, xe.shape[1], dtype=torch.bool, device=dev))
                logits[windows.index.get_indexer(idx)] = o.cpu().numpy().ravel()
                done += n
                if progress:
                    progress(done, total)
        f = windows.copy()
        f['allele'] = allele
        f['logit'] = logits
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out['prob'] = 1.0 / (1.0 + np.exp(-out.logit.values))
    return out


# --------------------------------------------------------------------------- %Rank
@dataclass
class PercentileGrid:
    """(allele, length) -> sorted background logits, for %Rank.

    Built per IMG/docs/decisions/img2-server-scope.md: 100,000 proteome-derived peptides spanning
    12-25 (~7,000 per length), scored per allele, stored as quantile breakpoints. **The background
    must be embedded through embed_windows()** so it shares the batched path with the query.

    Until the grid exists this returns None and the caller falls back to the raw logit.
    """
    path: str | None = None
    _g: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.path and os.path.exists(self.path):
            with np.load(self.path, allow_pickle=True) as z:
                self._g = {k: z[k] for k in z.files}

    @property
    def available(self) -> bool:
        return bool(self._g)

    @property
    def n_breaks(self) -> int:
        """Breakpoints per stratum; 100/n_breaks is the finest %Rank this grid can express."""
        return len(next(iter(self._g.values()))) if self._g else 0

    def rank(self, allele: str, length: int, logit: np.ndarray):
        """Percent of the background scoring at least as high. Lower = stronger binder.

        Resolution is 100/len(b) -- 0.1% for the standard 1,000-breakpoint grid. A returned 0.0
        therefore means "better than every breakpoint", i.e. **below 0.1%**, not "exactly zero".
        Display it as `<0.1` rather than `0.00`; a percentile of zero is not a thing.
        """
        b = self._g.get(f'{allele}|{length}')
        if b is None:
            return None
        return (np.searchsorted(-b, -np.asarray(logit), side='left') / len(b)) * 100.0


def add_percentile(df: pd.DataFrame, grid: PercentileGrid) -> pd.DataFrame:
    df = df.copy()
    if not grid.available:
        df['rank_pct'] = np.nan
        df['score_basis'] = 'logit (no percentile grid; not comparable across alleles)'
        return df
    r = np.full(len(df), np.nan)
    for (a, L), g in df.groupby(['allele', 'length']):
        v = grid.rank(a, int(L), g.logit.values)
        if v is not None:
            r[df.index.get_indexer(g.index)] = v
    df['rank_pct'] = r
    df['score_basis'] = 'percent rank vs proteome background, per (allele, length)'
    return df


def ranking_column(df: pd.DataFrame) -> tuple[str, bool]:
    """Which column orders candidates, and whether smaller is better."""
    if 'rank_pct' in df and df.rank_pct.notna().any():
        return 'rank_pct', True
    return 'logit', False


# --------------------------------------------------------------------------- consolidation
def select_strong(df: pd.DataFrame, rank_pct_max: float = STRONG_RANK_PCT,
                  top_frac: float = 0.02) -> pd.Series:
    """Which windows count as candidates, per allele.

    With a percentile grid: `rank_pct <= rank_pct_max`, defaulting to STRONG_RANK_PCT.

    The default was 2.0 until 2026-09-25, justified in this docstring as "the conventional class-II
    weak-binder line and what NetMHCIIpan and MixMHC2pred users already read". Both halves were
    false: NetMHCIIpan-4.3 ships 1% strong / 5% weak on `%Rank_EL` (`-rankS 1.0`, `-rankW 5.0`,
    verified in the installed man page), and MixMHC2pred defines no threshold at all. 2.0 was this
    module's own invention and matched no published tool.

    It is now STRONG_RANK_PCT, so a caller taking the default gets the same line the server draws.
    The tighter band is also the one consistent with this module's documented deficiency, which is
    over-coverage: img2's old 2% line flagged 63% of a protein against NetMHCIIpan's 18%.

    Without a grid: the top `top_frac` of that allele's windows by logit, which is a *relative*
    statement and must be labelled as such -- a raw logit has no absolute meaning.
    """
    col, asc = ranking_column(df)
    if col == 'rank_pct':
        return df.rank_pct <= rank_pct_max
    keep = pd.Series(False, index=df.index)
    for _, g in df.groupby('allele', sort=False):
        k = max(1, int(round(len(g) * top_frac)))
        keep.loc[g.sort_values(col, ascending=asc).head(k).index] = True
    return keep


def consolidate(df: pd.DataFrame, min_overlap: int = CORE, rank_pct_max: float = STRONG_RANK_PCT,
                top_frac: float = 0.02) -> pd.DataFrame:
    """Group overlapping CANDIDATE windows of one allele into binding events.

    Selection comes first, and that is not a detail. Scanning every start at every length means
    consecutive windows always overlap by >= 12 - 1 residues, so clustering the *whole* scan by
    overlap merges the entire antigen into one cluster -- measured: 2,115 windows -> 1 cluster
    spanning 1-227. Only windows that pass `select_strong` are clustered; the rest carry cluster
    NaN and are still returned, because the full table is part of the output.

    Two candidates join the same cluster when they overlap by >= min_overlap residues, the class-II
    core length: MHC-II binding is set by a 9-mer core, so windows sharing one are the same event.
    Single linkage over a position-sorted sweep, so a contiguous run of support is one cluster.
    """
    col, asc = ranking_column(df)
    df = df.copy()
    df['is_candidate'] = select_strong(df, rank_pct_max, top_frac)
    df['cluster'] = pd.NA
    df['is_best'] = False
    for allele, g in df[df.is_candidate].groupby('allele', sort=False):
        g = g.sort_values(['start', 'end'])
        cid, reach, ids = -1, None, []
        for st, en in zip(g.start.values, g.end.values):
            if reach is None or (min(reach, en) - st + 1) < min_overlap:
                cid += 1
                reach = en
            else:
                reach = max(reach, en)
            ids.append(f'{allele}#{cid}')
        df.loc[g.index, 'cluster'] = ids
        gg = df.loc[g.index]
        df.loc[gg.sort_values(col, ascending=asc).groupby('cluster').head(1).index, 'is_best'] = True
    return df


def collapse_positions(df: pd.DataFrame, budget: float = 0.15, min_run: int = CORE,
                       score_col: str = 'logit') -> pd.DataFrame:
    """Candidates by POSITION instead of by window: score each residue, keep the best `budget` of them.

    `consolidate` selects windows and then merges anything sharing a core, which rebuilds wide spans
    out of a few survivors: measured, it still flags **26% of a protein at `%Rank <= 0.1`** and cannot
    be operated at NetMHCIIpan-4.3's 13% coverage at all. Scoring positions removes that floor,
    because the budget is spent on residues rather than on windows -- one strong 9-mer core emits
    ~100 passing windows across ten lengths and their offsets, and all of them collapse onto the same
    residues.

    Measured on the 3,707 locked ligands, lift over the per-ligand null
    (notes/2026-09-18-collapsing-the-length-dimension-nearly-doubles-the-scan-lift.md):

        coverage 0.135   this rule 1.988   consolidate() unreachable   NetMHCIIpan-4.3 3.152
        coverage 0.233   this rule 1.756   consolidate() 1.565 at 0.262
        coverage 0.029   this rule 2.520   consolidate() unreachable

    Greedy non-maximum suppression and a fixed top-k window budget were both measured and are both
    worse, so the gain is the position collapse itself, not suppression and not budgeting.

    `budget` is a fraction of the protein's residues. The default 0.15 lands at ~0.135 coverage,
    which is NetMHCIIpan's own operating point -- chosen so the two are compared at equal exposure
    rather than at equal nominal threshold.

    Ranking by the raw logit beats ranking by `-rank_pct` below 0.12 coverage and loses above 0.22;
    the crossover is the grid's 0.1-percentile quantisation, which leaves only 21 distinct values
    under `%Rank` 2.0. `score_col` exposes the choice.
    """
    df = df.copy()
    # object dtype up front: assigning cluster ids into a float64 column is a pandas FutureWarning
    # and a future error
    df['cluster'] = pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    df['is_best'] = False
    df['run_start'] = np.nan
    df['run_end'] = np.nan
    col, asc = ranking_column(df)
    for allele, g in df.groupby('allele', sort=False):
        n = int(g.end.max())
        best = np.full(n + 2, -np.inf)
        v = g[score_col].values * (-1.0 if score_col == 'rank_pct' else 1.0)
        for s, e, x in zip(g.start.values, g.end.values, v):
            np.maximum.at(best, np.arange(s, e + 1), x)
        pos = np.arange(1, n + 1)
        val = best[1:n + 1]
        k = max(1, int(round(budget * n)))
        keep = np.zeros(n + 2, dtype=bool)
        keep[pos[np.argsort(-val, kind='stable')[:k]]] = True
        runs, i = [], 1
        while i <= n:
            if keep[i]:
                j = i
                while j + 1 <= n and keep[j + 1]:
                    j += 1
                if j - i + 1 >= min_run:
                    runs.append((i, j))
                i = j + 1
            else:
                i += 1
        for cid, (rs, re_) in enumerate(runs):
            # a window belongs to the run it shares a core with; a window may reach outside the run,
            # which is what `span_*` reports, while `run_*` is what was actually selected
            m = (g.start <= re_ - CORE + 1) & (g.end >= rs + CORE - 1)
            idx = g.index[m.values]
            if not len(idx):
                continue
            df.loc[idx, 'cluster'] = f'{allele}#{cid}'
            df.loc[idx, 'run_start'] = rs
            df.loc[idx, 'run_end'] = re_
        gg = df.loc[g.index]
        gg = gg[gg.cluster.notna()]
        if len(gg):
            df.loc[gg.sort_values(col, ascending=asc).groupby('cluster').head(1).index, 'is_best'] = True
    return df


def peak_region(starts, ends, frac: float = 0.5):
    """The half-maximum span of a cluster's core support -- a candidate narrow enough to act on.

    Measured against NetMHCIIpan-4.3 on 3,707 locked ligands
    (notes/2026-09-07-scan-vs-external-tools.md): at the same nominal 2% line our cluster spans cover
    **62.7%** of a protein and NetMHCIIpan's cover **18.2%**, and we get half its lift over chance
    for it. The grid is not miscalibrated -- ~2% of a protein's WINDOWS do pass -- but one strong
    9-mer core emits up to ~100 passing windows across ten lengths and their offsets, so the UNION of
    what passes is wide even when the evidence is concentrated.

    Core support is the fix and it is free: count, per residue, how many candidate windows cover it,
    then keep the contiguous run at or above `frac` of that cluster's peak. Half-maximum is the
    ordinary convention for the width of a peak and introduces no tuned constant. A region defined
    this way narrows with the evidence rather than with the threshold.
    """
    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    lo, hi = int(starts.min()), int(ends.max())
    cov = np.zeros(hi - lo + 2, dtype=np.int32)
    for s, e in zip(starts, ends):
        cov[s - lo:e - lo + 1] += 1
    peak = cov.max()
    if peak <= 0:
        return lo, hi, 0
    keep = cov >= max(1, int(np.ceil(frac * peak)))
    idx = np.flatnonzero(keep)
    # the run containing the first maximum, so a bimodal cluster reports its dominant lobe
    top = int(np.argmax(cov))
    a = top
    while a - 1 >= 0 and keep[a - 1]:
        a -= 1
    b = top
    while b + 1 < len(keep) and keep[b + 1]:
        b += 1
    return a + lo, b + lo, int(peak)


def candidate_table(df: pd.DataFrame, peak_frac: float = 0.5) -> pd.DataFrame:
    """One row per binding event: its best window, its span, and how much support it has.

    **The best window is the candidate; the span is context.** Measured on 3,707 locked eluted
    ligands (notes/2026-09-07-scan-validated-on-locked-ligands.md): a cluster SPAN is the union of
    every window that passed the 2% line, and because one strong 9-mer core produces up to ~100
    passing windows across ten lengths and their offsets, those unions cover **62% of a protein** --
    at which point "the ligand is inside a candidate" is nearly free (1.36x its null). The best
    window covers 36% and lifts better (1.57x overall, 2.04x on proteins >= 451 aa). `best_start`
    and `best_end` are therefore the columns to act on and to draw; `span_*` says how far the
    supporting evidence reaches.
    """
    col, asc = ranking_column(df)
    if 'cluster' not in df:
        df = consolidate(df)
    rows = []
    for cl, g in df[df.cluster.notna()].groupby('cluster', sort=False):
        b = g.sort_values(col, ascending=asc).iloc[0]
        ps, pe, pk = peak_region(g.start.values, g.end.values, peak_frac)
        rows.append({'cluster': cl, 'allele': b.allele,
                     'peak_start': int(ps), 'peak_end': int(pe), 'peak_support': pk,
                     # collapse_positions() selects a RUN of residues; the window union around it
                     # is context, so the selected region is reported when there is one
                     'span_start': int(g.run_start.iloc[0]) if 'run_start' in g and pd.notna(g.run_start.iloc[0])
                                   else int(g.start.min()),
                     'span_end': int(g.run_end.iloc[0]) if 'run_end' in g and pd.notna(g.run_end.iloc[0])
                                 else int(g.end.max()),
                     'best_start': int(b.start), 'best_end': int(b.end),
                     'best_length': int(b.length), 'best_peptide': b.peptide,
                     'logit': float(b.logit), 'prob': float(b.prob),
                     'rank_pct': float(b.rank_pct) if 'rank_pct' in b and pd.notna(b.rank_pct) else np.nan,
                     'n_windows': len(g)})
    cols = ['cluster', 'allele', 'peak_start', 'peak_end', 'peak_support',
            'span_start', 'span_end', 'best_start', 'best_end',
            'best_length', 'best_peptide', 'logit', 'prob', 'rank_pct', 'n_windows']
    if not rows:
        # An antigen with no window under the threshold is a legitimate answer, not an error --
        # that is the whole point of an ABSOLUTE %Rank line. Return the empty table with its
        # columns so every caller can keep indexing it.
        return pd.DataFrame(columns=cols)
    t = pd.DataFrame(rows)
    return t.sort_values(col if col in t else 'logit', ascending=asc).reset_index(drop=True)


# --------------------------------------------------------------------------- core support
def core_support(df: pd.DataFrame, seq_len: int, top_frac: float = 0.02) -> pd.DataFrame:
    """Per residue, how many covering windows are strong binders -- not just the best one.

    A single lucky window and a genuine binding core are indistinguishable under "best score at this
    position". They are very different under "how many of the ten lengths and their offsets agree",
    and that agreement is exactly what scanning every length at every offset produces. A 15-mer-only
    scanner cannot compute this column at all.

    `top_frac` selects the strong set: the best `top_frac` of this allele's windows by whichever
    column is available (percent rank if the grid is loaded, raw logit otherwise).
    """
    col, asc = ranking_column(df)
    out = []
    for allele, g in df.groupby('allele', sort=False):
        k = max(1, int(round(len(g) * top_frac)))
        strong = g[g.is_candidate] if 'is_candidate' in g else g.sort_values(col, ascending=asc).head(k)
        cov = np.zeros(seq_len + 1, dtype=np.int32)
        for s, e in zip(strong.start.values, strong.end.values):
            cov[s:e + 1] += 1
        best = np.full(seq_len + 1, np.nan)
        for s, e, v in zip(g.start.values, g.end.values, g[col].values):
            seg = best[s:e + 1]
            better = np.isnan(seg) | ((v < seg) if asc else (v > seg))
            seg[better] = v
            best[s:e + 1] = seg
        out.append(pd.DataFrame({'allele': allele, 'position': np.arange(1, seq_len + 1),
                                 'core_support': cov[1:], f'best_{col}': best[1:]}))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- report
def scan_report(seq: str, windows: pd.DataFrame, scored: pd.DataFrame) -> dict:
    col, asc = ranking_column(scored)
    return {'antigen_length': len(''.join(seq.split())),
            'lengths_scanned': sorted(windows.length.unique().tolist()),
            'windows': int(len(windows)),
            'alleles': sorted(scored.allele.unique().tolist()),
            'rows': int(len(scored)),
            'ranking_column': col,
            'lower_is_better': asc,
            'score_basis': scored.score_basis.iloc[0] if 'score_basis' in scored else 'logit'}
