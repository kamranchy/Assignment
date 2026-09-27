"""Sequence I/O, reference FoTeRs, and profile-HMM construction.

The nine full-length reference FoTeRs come from the authors' public
repository (github.com/RahnamaLab/FoTeR_Project). They are PUBLISHED DATA,
not something this pipeline infers.
"""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import pyhmmer
from Bio import SeqIO
from Bio.Seq import Seq
from pyhmmer import easel, plan7

DNA = easel.Alphabet.dna()
AA = easel.Alphabet.amino()

# REL endonuclease core motif of FoTeR/MoTeR/SLACS/CRE-type elements
# (paper: C[X2]C[X6]H[X3]C[X9]RHD/N ...). A permissive regex; the RHN / RHD
# residue distinguishes the RHNA-type and RHDK-type FoTeR subgroups.
REL_CORE = re.compile(r"C.{1,3}C.{5,8}H.{2,4}C.{8,11}[RK]H[DN]")


def read_fasta(path: str | Path) -> dict[str, str]:
    path = str(path)
    handle = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    with handle:
        return {r.id: str(r.seq).upper() for r in SeqIO.parse(handle, "fasta")}


def revcomp(s: str) -> str:
    return str(Seq(s).reverse_complement())


def longest_orf_protein(seq: str, both_strands: bool = False) -> tuple[str, int, int, str]:
    """Longest Met-initiated ORF. Returns (protein, nt_start0, nt_end0, strand)."""
    best = ("", 0, 0, "+")
    strands = [("+", seq)] + ([("-", revcomp(seq))] if both_strands else [])
    for strand, s in strands:
        for fr in range(3):
            sub = s[fr: len(s) - ((len(s) - fr) % 3)]
            prot = str(Seq(sub).translate())
            for m in re.finditer(r"M[^*]*", prot):
                if len(m.group()) > len(best[0]):
                    a = fr + 3 * m.start()
                    b = a + 3 * len(m.group())
                    if strand == "-":
                        a, b = len(s) - b, len(s) - a
                    best = (m.group(), a, b, strand)
    return best


@dataclass
class Reference:
    name: str
    seq: str
    orf_prot: str

    @property
    def length(self) -> int:
        return len(self.seq)


def load_references(path: str | Path) -> dict[str, Reference]:
    refs = {}
    for name, seq in read_fasta(path).items():
        prot, *_ = longest_orf_protein(seq)
        refs[name] = Reference(name, seq, prot)
    return refs


def _build(name: str, seq: str, alphabet) -> plan7.HMM:
    ds = easel.TextSequence(name=name.encode(), sequence=seq).digitize(alphabet)
    hmm, _, _ = plan7.Builder(alphabet).build(ds, plan7.Background(alphabet))
    return hmm


def build_nt_hmms(refs: dict[str, Reference], exclude: set[str] | None = None) -> list[plan7.HMM]:
    """One single-sequence nucleotide profile per reference element.

    `exclude` implements the leave-own-reference-out rule: when a benchmark
    genome is processed, its own strain's reference is NOT used, so that
    homology features are never computed against a sequence derived from
    that same genome (prevents leakage).
    """
    exclude = exclude or set()
    return [_build(r.name, r.seq, DNA) for r in refs.values() if r.name not in exclude]


def build_prot_hmms(refs: dict[str, Reference], exclude: set[str] | None = None) -> list[plan7.HMM]:
    exclude = exclude or set()
    return [_build("FoTeR_RT_" + r.name, r.orf_prot, AA) for r in refs.values() if r.name not in exclude]


def load_hmm_file(path: str | Path | None) -> list[plan7.HMM]:
    """Load e.g. Pfam PF00078 (RVT_1). Returns [] if not available."""
    if not path or not Path(path).exists():
        return []
    with plan7.HMMFile(str(path)) as fh:
        return list(fh)
