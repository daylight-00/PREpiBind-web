"""Locating the peptide-binding domain of an MHC class II chain from its sequence alone.

The model's HLA input is the domain of a chain (58-82 residues) cut out of the ESMC embedding of the FULL chain; the allele catalogue stores each chain's window [start_idx, end_idx). For a chain that is not in the
catalogue the window is found by alignment to the catalogue: a 4-mer index picks the nearest few chains, a BLOSUM62 alignment with free end gaps (fragments and full-length chains both occur) chooses among them, and the
nearest chain's window is carried across the alignment to the query. The identity of the query with the reference over the reference's window is the confidence. Measured on the catalogue itself (measured on a prototype): within 2 residues of the true window for 99.8 % of the chains with their own entry removed and 99.5 % with their whole gene removed; by identity 0.5 and up 98.5-99.9 %,
0.4-0.5 95.8 %, 0.3-0.4 78.9 %; shuffled and random sequences never exceed 0.33.
"""
import collections
import threading

import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices

K = 4  # length of the k-mers of the index
CANDIDATES = 12  # references aligned per query
CONFIDENT = 0.5  # from this identity the located window is used as it is
LOW = 0.4  # from this identity the located window is used with a warning; below it no window is located


def tier(identity):
    """'located', 'low confidence' or 'no domain' for an identity over the reference window."""
    return "located" if identity >= CONFIDENT else "low confidence" if identity >= LOW else "no domain"


class Locator:
    def __init__(self, names, sequences, starts, ends):
        self.names, self.sequences = list(names), list(sequences)
        self.starts, self.ends = np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)
        self.lengths = np.array([len(s) for s in self.sequences], dtype=np.int64)
        index = collections.defaultdict(list)
        for i, sequence in enumerate(self.sequences):
            for kmer in {sequence[j:j + K] for j in range(len(sequence) - K + 1)}:
                index[kmer].append(i)
        self.index = {kmer: np.array(ids, dtype=np.int32) for kmer, ids in index.items()}
        self.aligner = Align.PairwiseAligner(mode="global")
        self.aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        self.aligner.open_gap_score, self.aligner.extend_gap_score = -11, -1
        self.aligner.end_insertion_score = self.aligner.end_deletion_score = 0  # free end gaps
        self.lock = threading.Lock()

    @classmethod
    def from_catalogue(cls, catalogue):
        """From a DataFrame with the columns HLA_Name, HLA_Seq, start_idx and end_idx (data/mhc_mapping.csv)."""
        return cls(catalogue.HLA_Name, catalogue.HLA_Seq, catalogue.start_idx, catalogue.end_idx)

    def candidates(self, chain, excluded=()):
        """The catalogue chains that share the most 4-mers with `chain` (at most CANDIDATES), best first; of equally good ones the longest first, because a domain pasted without its flanks is identical to the domain
        of hundreds of alleles, some of them only fragments, and the nearest chain's flanks are what is wanted."""
        hits = [self.index[chain[j:j + K]] for j in range(len(chain) - K + 1) if chain[j:j + K] in self.index]
        if not hits:
            return []
        count = np.bincount(np.concatenate(hits), minlength=len(self.names)).astype(np.float64)
        count[list(excluded)] = -1
        order = np.lexsort((-self.lengths, -count))[:CANDIDATES]
        return [int(i) for i in order if count[i] > 0]

    @staticmethod
    def _to_query(reference_blocks, query_blocks, position, side):
        """A position of the reference as a position of the query; one that falls in a gap snaps to the nearest aligned residue."""
        for (t0, t1), (q0, _) in zip(reference_blocks, query_blocks):
            if t0 <= position < t1:
                return q0 + (position - t0)
        if side == "start":
            later = [q0 for (t0, _), (q0, _) in zip(reference_blocks, query_blocks) if t0 > position]
            return later[0] if later else query_blocks[-1][1]
        earlier = [q1 for (_, t1), (_, q1) in zip(reference_blocks, query_blocks) if t1 <= position]
        return earlier[-1] if earlier else query_blocks[0][0]

    def locate(self, chain, excluded=()):
        """The domain window of `chain` and what it was found from, or None when the catalogue has nothing alike:
        {'nearest': allele name, 'start', 'end' (positions in `chain`, half-open), 'identity' (over the reference's window), 'reference': index, 'reference_window': (start, end)}."""
        with self.lock:
            candidates = self.candidates(chain, excluded)
            if not candidates:
                return None
            best = max(candidates, key=lambda i: (self.aligner.score(self.sequences[i], chain), self.lengths[i]))  # equal alignments: the longest chain, which has the most flank to give
            alignment = self.aligner.align(self.sequences[best], chain)[0]
        reference_blocks, query_blocks = alignment.aligned
        s0, e0 = int(self.starts[best]), int(self.ends[best])
        start = int(self._to_query(reference_blocks, query_blocks, s0, "start"))
        end = int(self._to_query(reference_blocks, query_blocks, e0 - 1, "end")) + 1
        reference, same = self.sequences[best], 0
        for (t0, t1), (q0, _) in zip(reference_blocks, query_blocks):
            lo, hi = max(t0, s0), min(t1, e0)
            if hi > lo:
                same += sum(a == b for a, b in zip(reference[lo:hi], chain[q0 + (lo - t0):q0 + (hi - t0)]))
        return {"nearest": self.names[best], "start": start, "end": max(end, start), "identity": same / (e0 - s0), "reference": best, "reference_window": (s0, e0)}

    def complete_flanks(self, chain, hit):
        """The chain with the nearest chain's flanks added where the chain has less of them (a pasted domain without its flanks): the embedding of a domain without context is not the embedding the model
        was trained on (36.7 % of the signs flip against 2.9 to 5.8 % with the flanks grafted; the prototype's domain_only_graft.py). Returns (the chain with flanks, the offset of the chain's own first residue in it)."""
        s0, e0 = hit["reference_window"]
        reference = self.sequences[hit["reference"]]
        left = reference[:max(0, s0 - hit["start"])]
        right = reference[e0 + max(0, len(chain) - hit["end"]):] if len(chain) - hit["end"] < len(reference) - e0 else ""
        return left + chain + right, len(left)
