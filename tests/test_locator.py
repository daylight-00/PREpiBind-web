"""The locator that finds the domain window of an MHC chain that is not in the allele catalogue, tested on the catalogue itself (7,282 chains with their true windows). No GPU is needed."""
from pathlib import Path

import locator as loc
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def catalogue():
    return pd.read_csv(ROOT / "data" / "mhc_mapping.csv")


@pytest.fixture(scope="module")
def finder(catalogue):
    return loc.Locator.from_catalogue(catalogue)


@pytest.fixture(scope="module")
def sample(catalogue):
    return np.random.default_rng(0).choice(len(catalogue), 120, replace=False)


def test_the_tiers_follow_the_identity():
    assert [loc.tier(x) for x in (1.0, 0.5, 0.499, 0.4, 0.399, 0.0)] == ["located", "located", "low confidence", "low confidence", "no domain", "no domain"]


def test_a_chain_of_the_catalogue_is_found_at_its_own_window_with_identity_one(finder, catalogue, sample):
    for i in sample[:30]:
        hit = finder.locate(catalogue.HLA_Seq[i])
        assert hit["identity"] == 1.0 and (hit["start"], hit["end"]) == (catalogue.start_idx[i], catalogue.end_idx[i])


def test_with_its_own_entry_removed_the_window_is_found_within_two_residues(finder, catalogue, sample):
    """Alleles of one gene differ by a few residues, so the nearest other chain is a close relative (the prototype: 99.8 % of all 7,282 chains)."""
    close = 0
    for i in sample:
        hit = finder.locate(catalogue.HLA_Seq[i], excluded=(i,))
        close += max(abs(hit["start"] - catalogue.start_idx[i]), abs(hit["end"] - catalogue.end_idx[i])) <= 2 and hit["identity"] >= loc.CONFIDENT
    assert close / len(sample) >= 0.95


def test_a_point_mutation_does_not_move_the_window(finder, catalogue, sample):
    i = sample[0]
    chain = catalogue.HLA_Seq[i]
    k = (catalogue.start_idx[i] + catalogue.end_idx[i]) // 2
    mutated = chain[:k] + ("A" if chain[k] != "A" else "G") + chain[k + 1:]
    hit = finder.locate(mutated)
    assert (hit["start"], hit["end"]) == (catalogue.start_idx[i], catalogue.end_idx[i]) and 0.9 < hit["identity"] < 1.0


def test_a_chain_with_a_signal_peptide_and_a_tail_added_gets_the_window_shifted_with_it(finder, catalogue, sample):
    i = sample[1]
    chain = "M" + "ACDEFGHIKL" * 3 + catalogue.HLA_Seq[i] + "WYWYWY"
    hit = finder.locate(chain, excluded=(i,))
    shift = 31
    assert abs(hit["start"] - (catalogue.start_idx[i] + shift)) <= 2 and abs(hit["end"] - (catalogue.end_idx[i] + shift)) <= 2


def test_a_shuffled_chain_is_not_a_domain(finder, catalogue, sample):
    rng = np.random.default_rng(1)
    hits = [finder.locate("".join(rng.permutation(list(catalogue.HLA_Seq[i])))) for i in sample[:30]]
    identities = [hit["identity"] for hit in hits if hit is not None]
    assert identities and max(identities) < loc.LOW  # the prototype: shuffled and random sequences never exceed 0.33


def test_a_sequence_with_no_4_mer_in_common_with_the_catalogue_is_not_located(finder):
    assert finder.locate("W" * 12) is None and finder.locate("MK") is None and finder.locate("") is None


def test_a_domain_without_its_flanks_gets_the_nearest_chains_flanks_added(finder, catalogue, sample):
    i = sample[2]
    domain = catalogue.HLA_Seq[i][catalogue.start_idx[i]:catalogue.end_idx[i]]
    hit = finder.locate(domain, excluded=(i,))
    grafted, offset = finder.complete_flanks(domain, hit)
    s0, e0 = hit["reference_window"]
    reference = finder.sequences[hit["reference"]]
    assert grafted[offset:offset + len(domain)] == domain and offset == max(0, s0 - hit["start"])
    assert grafted.startswith(reference[:offset]) and grafted.endswith(reference[e0 + max(0, len(domain) - hit["end"]):])
    assert len(grafted) > len(domain)


def test_completing_the_flanks_never_changes_the_chain_itself_and_adds_only_what_the_chain_lacks(finder, catalogue, sample):
    for i in sample[3:23]:
        chain = catalogue.HLA_Seq[i]
        hit = finder.locate(chain, excluded=(i,))
        grafted, offset = finder.complete_flanks(chain, hit)
        s0, e0 = hit["reference_window"]
        assert grafted[offset:offset + len(chain)] == chain
        assert offset == max(0, s0 - hit["start"])  # nothing is added on the left when the chain has at least as much there as the reference
        assert len(grafted) - offset - len(chain) == (len(finder.sequences[hit["reference"]]) - e0 - (len(chain) - hit["end"]) if len(chain) - hit["end"] < len(finder.sequences[hit["reference"]]) - e0 else 0)


def test_a_domain_alone_takes_its_flanks_from_the_longest_of_the_chains_that_share_its_domain():
    """A domain pasted without flanks is identical to the domain of many alleles, some of them only fragments; the nearest chain's flanks are wanted, so of equal alignments the longest chain wins."""
    rng = np.random.default_rng(0)
    amino = list("ACDEFGHIKLMNPQRSTVWY")
    rand = lambda n: "".join(rng.choice(amino, n))
    domain, left, right = rand(80), rand(30), rand(120)
    full, medium, fragment = left + domain + right, rand(20) + domain + rand(40), rand(5) + domain + rand(5)
    finder = loc.Locator(["fragment", "medium", "full"], [fragment, medium, full], [5, 20, 30], [85, 100, 110])
    hit = finder.locate(domain)
    assert hit["nearest"] == "full" and hit["identity"] == 1.0
    assert finder.complete_flanks(domain, hit) == (full, 30)
