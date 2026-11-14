"""The protein scan's plain-Python part: reading a protein or FASTA, the windows, the allele panels, the hard limits. No GPU is needed."""
from pathlib import Path

import pandas as pd
import pytest

import engine
import scan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def names():
    catalogue = pd.read_csv(ROOT / "data" / "mhc_mapping.csv")
    return set(catalogue.HLA_Name)


# ---- reading a protein -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_a_plain_sequence_loses_whitespace_digits_gaps_and_a_stop_and_case():
    assert scan.read_proteins("  1 mkta yiak\n 11 qrq-isf *\n") == [("sequence", "MKTAYIAKQRQISF")]


def test_fasta_gives_one_record_per_header_with_its_lines_joined():
    text = ">first one\nMKTAY\nIAKQR\n>second\nGGGG\n"
    assert scan.read_proteins(text) == [("first one", "MKTAYIAKQR"), ("second", "GGGG")]


def test_x_is_allowed_other_non_amino_acid_letters_are_refused_by_name():
    assert scan.read_proteins("MKXA") == [("sequence", "MKXA")]
    with pytest.raises(scan.ScanRefused, match="B.*J|J.*B|B"):
        scan.read_proteins("MKBJA")


def test_nothing_gives_no_protein_and_a_header_alone_is_refused():
    assert scan.read_proteins("   ") == []
    with pytest.raises(scan.ScanRefused, match="no sequence"):
        scan.read_proteins(">only a header\n")


# ---- windows --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_number_of_windows_of_each_length():
    assert scan.count_windows(20, [15]) == 6
    assert scan.count_windows(20, [15, 16]) == 6 + 5
    assert scan.count_windows(10, [15]) == 0
    assert scan.count_windows(20, [15, 15]) == 6  # a length counted once


def test_the_frame_has_every_window_against_every_pair_with_one_based_starts():
    frame = scan.scan_frame("ACDEFGHIKLMNPQRSTV", [15, 16], [("A1", "B1"), ("A2", "B2")])
    assert len(frame) == (4 + 3) * 2
    first = frame.iloc[:2]
    assert list(first.MHC) == ["A1_B1", "A2_B2"] and list(first.Epitope) == ["ACDEFGHIKLMNPQR"] * 2 and list(first.Start) == [1, 1] and list(first.Length) == [15, 15]
    last = frame.iloc[-2:]
    assert list(last.Start) == [3, 3] and list(last.Length) == [16, 16] and list(last.Epitope) == ["DEFGHIKLMNPQRSTV"] * 2


def test_every_window_is_short_enough_for_the_engine():
    assert max(scan.WINDOW_LENGTHS) <= engine.MAX_EPITOPE_LENGTH and scan.DEFAULT_WINDOW in scan.WINDOW_LENGTHS


# ---- the limits -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def small_limits(monkeypatch):
    """Limits small enough to test each boundary: 100 residues, 10 alleles, 500 predictions (200 without the fast engine)."""
    for name, value in (("MAX_PROTEIN_LENGTH", 100), ("MAX_ALLELES", 10), ("MAX_SCAN_ROWS", 500), ("MAX_SCAN_ROWS_EXACT", 200)):
        monkeypatch.setattr(scan, name, value)


def test_a_scan_at_each_limit_is_accepted(small_limits):
    assert scan.check_scan(100, [15], 1) == 86  # the longest protein
    assert scan.check_scan(30, [15], 10) == 16  # the largest panel
    assert scan.check_scan(100, [15], 5) == 86  # 430 predictions, under the 500
    assert scan.check_scan(41, [15], 10, fast=True) == 27  # 270 predictions
    assert scan.check_scan(100, [15, 16], 1) == 86 + 85  # a second window length counts


@pytest.mark.parametrize(
    "protein_length, lengths, alleles, message",
    [
        (101, [15], 1, "at most 100"),
        (50, [15], 11, "at most 10"),
        (100, [15], 6, "predictions"),  # 516 > 500
        (100, [15, 16], 3, "predictions"),  # a second window length doubles the windows
        (10, [15], 5, "shorter"),
        (50, [], 5, "window length"),
        (50, [15], 0, "allele"),
    ],
    ids=["protein too long", "panel too large", "too many predictions", "two window lengths", "protein shorter than a window", "no window length", "no allele"],
)
def test_a_scan_over_a_limit_or_empty_is_refused_with_the_reason(small_limits, protein_length, lengths, alleles, message):
    with pytest.raises(scan.ScanRefused, match=message):
        scan.check_scan(protein_length, lengths, alleles)


def test_without_the_fast_engine_the_row_limit_is_smaller_and_says_why(small_limits):
    scan.check_scan(100, [15], 3, fast=True)  # 258 predictions: fine with the fast engine
    with pytest.raises(scan.ScanRefused, match="not available"):
        scan.check_scan(100, [15], 3, fast=False)  # over the 200 without it


def test_the_real_limits_are_one_budget_that_the_measurements_support():
    """5,000 residues x 100 alleles is about 500,000 predictions: the 30 s the fast path measured (16,000 to 17,000 per second); the exact path (2,000 per second) gets a fifth of it, about 50 s."""
    assert (scan.MAX_PROTEIN_LENGTH - 14) * scan.MAX_ALLELES <= scan.MAX_SCAN_ROWS == 500_000
    assert scan.MAX_SCAN_ROWS_EXACT == 100_000 < scan.MAX_SCAN_ROWS


def test_the_refusal_is_a_value_error():
    assert issubclass(scan.ScanRefused, ValueError)


# ---- the allele panels ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_reference_panel_has_its_27_alleles_in_the_catalogue(names):
    panel = scan.reference_panel(names)
    assert len(panel) == 27 and len(set(panel)) == 27 and all(a in names and b in names for a, b in panel)
    assert sum(a == scan.DR_ALPHA for a, _ in panel) == 16


def test_the_drb1_panel_has_the_eleven_drb1_alleles(names):
    panel = scan.drb1_panel(names)
    assert len(panel) == 11 and all(b.startswith("HLA-DRB1*") and a == scan.DR_ALPHA for a, b in panel)


def test_a_panel_takes_what_the_catalogue_knows_and_names_the_rest():
    pairs = scan.reference_panel({scan.DR_ALPHA, "HLA-DRB1*01:01"})
    assert pairs == [(scan.DR_ALPHA, "HLA-DRB1*01:01")]


def test_a_typed_panel_accepts_pairs_with_any_separator_and_a_dr_beta_alone(names):
    alpha = {n for n in names if "A1" in n or "RA" in n}
    beta = {n for n in names if "B" in n}
    text = "HLA-DRB1*15:01\nHLA-DQA1*05:01, HLA-DQB1*02:01\nHLA-DQA1*05:01/HLA-DQB1*03:01\n\nHLA-DRB1*15:01\n"
    assert scan.parse_panel(text, alpha, beta) == [(scan.DR_ALPHA, "HLA-DRB1*15:01"), ("HLA-DQA1*05:01", "HLA-DQB1*02:01"), ("HLA-DQA1*05:01", "HLA-DQB1*03:01")]  # the repeated line is dropped


def test_a_typed_panel_names_the_alleles_the_catalogue_does_not_know_and_the_lines_it_cannot_read(names):
    alpha = {n for n in names if "A1" in n or "RA" in n}
    beta = {n for n in names if "B" in n}
    with pytest.raises(scan.ScanRefused, match=r"HLA-DRB1\*99:99"):
        scan.parse_panel("HLA-DRB1*99:99", alpha, beta)
    with pytest.raises(scan.ScanRefused, match="not an allele pair"):
        scan.parse_panel("HLA-DQA1*05:01", alpha, beta)


def test_a_dr_pair_is_labelled_by_its_beta_chain_and_the_others_by_both():
    assert scan.short_label(scan.DR_ALPHA, "HLA-DRB1*15:01") == "DRB1*15:01"
    assert scan.short_label("HLA-DQA1*05:01", "HLA-DQB1*02:01") == "DQA1*05:01/DQB1*02:01"
