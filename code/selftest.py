"""Startup checks for the served artifacts. `python code/selftest.py` from the repository root.

Each check is one thing that has silently broken before, or that a decision says must hold:
integrity of the served weights, the vendored compute path, the chain-table split, the panel size,
and that the bands come from the file rather than from prose.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts
import chains
import rank

FAIL = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(name)


def main():
    for head, (ok, got) in artifacts.verify().items():
        check(f"served checkpoint {head}", ok, got[:12])

    for path, (ok, got) in artifacts.check_vendored().items():
        check(f"vendored {path}", ok, got[:12])

    b = artifacts.bands()
    check("bands from BANDS.json",
          b["bands"]["strong_rank_pct_max"] == 1.0 and b["bands"]["weak_rank_pct_max"] == 5.0,
          f"strong<={b['bands']['strong_rank_pct_max']} weak<={b['bands']['weak_rank_pct_max']}")
    check("bands carry their limits", "negatives_are" in b["evidence"])

    # The six chains where the two sources disagree must all resolve to the table.
    disputed = ["H2-IAdA", "H2-IAdB", "H2-IAg7A",
                "HLA-DQB1*03:01", "HLA-DQB1*05:03", "HLA-DQB1*06:01"]
    check("disputed chains resolve to the table",
          all(chains.source(c) == "table" for c in disputed))
    check("H2-IAd is no longer swapped",
          len(chains.sequence("H2-IAdA")) == 81 and len(chains.sequence("H2-IAdB")) == 75,
          f"A={len(chains.sequence('H2-IAdA'))} B={len(chains.sequence('H2-IAdB'))}")

    lists = chains.chain_lists()
    check("offered chains are the 7,282 catalogue",
          (len(lists["alpha"]), len(lists["beta"])) == (838, 6444),
          f"{len(lists['alpha'])} alpha + {len(lists['beta'])} beta")
    # Every panel molecule must be reachable from the two selectboxes, or part of the panel is
    # precomputed and unusable.
    alpha, beta = set(lists["alpha"]), set(lists["beta"])
    panel = rank.molecules(artifacts.HEADLINE)
    reachable = [m for m in panel
                 if m.split("_")[0] in beta and m.split("_")[1] in alpha]
    check("panel is composable from those lists", len(reachable) == len(panel),
          f"{len(reachable)}/{len(panel)}")

    for head in artifacts.HEADS:
        mols = rank.molecules(head)
        check(f"panel {head}", len(mols) == 306, f"{len(mols)} molecules")

    # A molecule on the panel ranks. One with no background at all returns None rather than a bare
    # logit — picked here as a molecule that is neither on the panel nor in the on-demand cache,
    # since a built grid legitimately makes an off-panel molecule rankable.
    mol = "HLA-DRB1*15:01_HLA-DRA*01:01"
    r = rank.rank("ms", mol, 15, [2.0])
    check("panel molecule ranks", r is not None and 0 <= r[0] <= 100, f"logit 2.0 -> {r[0]:.2f}%")

    lists = chains.chain_lists()
    ungrounded = next(
        (m for a in lists["alpha"][:40] for b in lists["beta"][:40]
         if not rank.has_background("ms", (m := chains.molecule(a, b)))), None)
    check("no background returns None, not a bare logit",
          ungrounded is not None and rank.rank("ms", ungrounded, 15, [2.0]) is None,
          str(ungrounded))

    # The reason %Rank exists: the same logit is a different rarity on each head.
    per_head = {h: rank.rank(h, mol, 15, [2.0]) for h in artifacts.HEADS}
    spread = {h: round(float(v[0]), 2) for h, v in per_head.items() if v is not None}
    check("heads are not on one scale", len(set(spread.values())) == len(spread), str(spread))

    # Domain inference for a chain the catalogue does not have, against the chain table's own
    # annotation. Getting this wrong is worth 20% of band calls, so it is checked, not assumed.
    import csv as _csv

    import domain

    annotated = {}
    for row in _csv.DictReader(open(os.path.join(artifacts.ROOT, "data", "chain_table.csv"))):
        parts = row["HLA_Seq"].split("|")
        if len(parts) == 3:
            annotated[row["HLA_Name"]] = (parts[0], int(parts[1]), int(parts[2]))
    catalogued = {n for n, *_ in domain._catalogue()}
    novel = sorted(n for n in annotated if n not in catalogued)
    off = []
    for name in novel:
        seq, lo, hi = annotated[name]
        kind = "alpha" if re.search(r"D[PQR]A|-I[AE]\w*A$", name) else "beta"
        got = domain.infer_window(seq, kind=kind)
        off.append(None if got is None else max(abs(got[0] - lo), abs(got[1] - hi)))
    exact = sum(1 for d in off if d == 0)
    check("domain inference on non-catalogue chains",
          all(d is not None and d <= 1 for d in off),
          f"{exact}/{len(novel)} exact, all within {max(d for d in off if d is not None)} residue")

    check("band from the file", rank.band(0.5, b) == "Strong" and rank.band(3.0, b) == "Weak"
          and rank.band(9.0, b) is None)
    check("sub-resolution prints as <0.1", rank.format_pct(0.0, rank.resolution("ms")) == "<0.1")

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED: {', '.join(FAIL)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
