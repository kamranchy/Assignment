"""Stage A: high-recall candidate generation.

Deliberately permissive. Nothing here decides whether a locus is a FoTeR;
that is the job of the learned classifier (Stage C). Two evidence streams:

1. nucleotide alignment (minimap2/mappy) with the reference FoTeRs,
   chained collinearly per reference so that tandem copies at a chromosome
   end are NOT merged into one element (a new copy restarts near the
   reference 5' end);
2. protein profile-HMM search of six-frame ORFs, with FoTeR RT proteins and
   (optionally) the generic Pfam RVT_1 model. Protein-only loci supply
   diverged copies and, importantly, *other* RT-containing elements that act
   as hard negatives.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pyhmmer
from Bio.Seq import Seq
from pyhmmer import easel

from .reference import AA, DNA, revcomp

NT_COLS = ["scaffold", "start", "end", "strand", "ref", "hmm_from", "hmm_to", "score", "ident"]


def nt_search(genome: dict[str, str], refs, exclude=None, chunk: int = 20000, step: int = 10000,
              preset: str = "map-ont") -> pd.DataFrame:
    """Nucleotide homology search with minimap2 (mappy).

    Each reference FoTeR gets its own index so that EVERY reference reports
    its own alignment at a locus (-> n_refs_support). The genome is streamed
    as overlapping 20-kb chunks (step 10 kb) so that all copies, including
    tandem copies, are reported rather than a single best mapping. The
    'map-ont' preset tolerates ~15-20% divergence, which covers the
    between-strain divergence of FoTeRs; more diverged copies are caught by
    the protein stream. (An nhmmer implementation was tried first; with
    single-sequence ~5-kb DNA profiles it was orders of magnitude slower
    in our tests.)
    """
    import mappy as mp
    import os, tempfile
    exclude = exclude or set()
    rows = []
    for name, r in refs.items():
        if name in exclude:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as fh:
            fh.write(f">{name}\n{r.seq}\n"); tmp = fh.name
        al = mp.Aligner(tmp, preset=preset, best_n=50)
        os.unlink(tmp)
        for sc, s in genome.items():
            for st in range(0, max(1, len(s) - chunk + step), step):
                for h in al.map(s[st: st + chunk]):
                    rows.append((sc, st + h.q_st + 1, st + h.q_en, "+" if h.strand == 1 else "-", name,
                                 h.r_st + 1, h.r_en, float(h.mlen), h.mlen / max(1, h.blen)))
    df = pd.DataFrame(rows, columns=NT_COLS)
    return df.drop_duplicates(["scaffold", "start", "end", "strand", "ref"]).reset_index(drop=True)


def chain_nt_hits(hits: pd.DataFrame, ref_len: dict[str, int], max_gap: int = 1500, back_tol: int = 200) -> pd.DataFrame:
    """Collinear chaining of hits from the same reference/strand/scaffold.

    Hits are ordered along the element direction (5'->3'). A hit extends the
    current chain only if it continues forward in reference coordinates
    (hmm_from > previous hmm_to - back_tol) and the genomic gap is small.
    A jump back to low reference coordinates starts a new chain => tandem
    copies stay separate.
    """
    out = []
    if hits.empty:
        return pd.DataFrame(columns=["scaffold", "start", "end", "strand", "ref", "ref_from", "ref_to", "nt_score", "nt_ident", "n_segments"])
    for (sc, st, ref), g in hits.groupby(["scaffold", "strand", "ref"]):
        g = g.sort_values("start", ascending=(st == "+"))
        chain = None
        for r in g.itertuples(index=False):
            if chain is not None:
                gap = (r.start - chain["end"]) if st == "+" else (chain["start"] - r.end)
                if gap <= max_gap and r.hmm_from > chain["ref_to"] - back_tol:
                    chain["start"] = min(chain["start"], r.start)
                    chain["end"] = max(chain["end"], r.end)
                    chain["ref_to"] = max(chain["ref_to"], r.hmm_to)
                    chain["nt_score"] += r.score
                    chain["nt_ident"] = max(chain["nt_ident"], r.ident)
                    chain["n_segments"] += 1
                    continue
                out.append(chain)
            chain = dict(scaffold=sc, start=r.start, end=r.end, strand=st, ref=ref, ref_from=r.hmm_from,
                         ref_to=r.hmm_to, nt_score=r.score, nt_ident=r.ident, n_segments=1)
        out.append(chain)
    df = pd.DataFrame(out)
    df["ref_len"] = df["ref"].map(ref_len)
    return df


def _overlap(a0, a1, b0, b1) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def collapse_chains(chains: pd.DataFrame, min_overlap: int = 50) -> pd.DataFrame:
    """Single-linkage clustering of chains (all references) on the same
    strand that overlap by >= min_overlap bp -> one locus per cluster.

    Tandem copies at a chromosome end are separated by telomeric repeats and
    do not overlap, so they stay separate; fragments of ONE copy produced by
    VNTR length differences (local alignments that restart a few hundred bp
    back) overlap and are merged. The locus spans the union of all member
    alignments; the representative (best_ref, ref_from/to) is the highest-
    scoring chain; end completeness is the best over all members because the
    5' region (VNTRs) is the most variable part of FoTeRs.
    """
    if chains.empty:
        return chains.assign(n_refs_support=[], min_miss5=[], min_miss3=[])
    out = []
    for (sc, st), g in chains.groupby(["scaffold", "strand"]):
        g = g.sort_values("start")
        clusters, cur, cur_end = [], [], -1
        for idx, r in g.iterrows():
            if cur and r.start <= cur_end - min_overlap:
                cur.append(idx); cur_end = max(cur_end, r.end)
            else:
                if cur:
                    clusters.append(cur)
                cur, cur_end = [idx], r.end
        clusters.append(cur)
        for ids in clusters:
            sub = chains.loc[ids]
            rec = sub.loc[sub.nt_score.idxmax()].to_dict()
            rec["start"], rec["end"] = int(sub.start.min()), int(sub.end.max())
            rec["n_refs_support"] = sub.ref.nunique()
            rec["min_miss5"] = float((sub.ref_from - 1).min())
            rec["min_miss3"] = float((sub.ref_len - sub.ref_to).min())
            rec["nt_score"] = float(sub.groupby("ref").nt_score.sum().max())
            out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- protein side
def six_frame_orfs(genome: dict[str, str], min_aa: int = 150):
    """Stop-to-stop ORFs (not requiring Met: degraded copies lose starts)."""
    orfs = []
    for sc, seq in genome.items():
        L = len(seq)
        for strand, s in (("+", seq), ("-", revcomp(seq))):
            for fr in range(3):
                sub = s[fr: L - ((L - fr) % 3)]
                prot = str(Seq(sub).translate())
                for m in re.finditer(r"[^*]{%d,}" % min_aa, prot):
                    a = fr + 3 * m.start()
                    b = fr + 3 * m.end()
                    if strand == "+":
                        g0, g1 = a + 1, b
                    else:
                        g0, g1 = L - b + 1, L - a
                    orfs.append((f"{sc}|{strand}|{g0}|{g1}", sc, g0, g1, strand, m.group().replace("X", "")))
    return orfs


def prot_search(orfs, hmms, label: str, max_evalue: float = 1e-3, cpus: int = 0) -> pd.DataFrame:
    cols = ["scaffold", "start", "end", "strand", "score", "query", "label"]
    if not hmms or not orfs:
        return pd.DataFrame(columns=cols)
    block = easel.DigitalSequenceBlock(
        AA, [easel.TextSequence(name=o[0].encode(), sequence=o[5]).digitize(AA) for o in orfs]
    )
    rows = []
    for hmm, hits in zip(hmms, pyhmmer.hmmsearch(hmms, block, cpus=cpus, E=max_evalue)):
        q = hmm.name.decode() if isinstance(hmm.name, bytes) else hmm.name
        for h in hits:
            if h.evalue > max_evalue:
                continue
            nm = h.name.decode() if isinstance(h.name, bytes) else h.name
            sc, strand, g0, g1 = nm.split("|")
            rows.append((sc, int(g0), int(g1), strand, h.score, q, label))
    return pd.DataFrame(rows, columns=cols)


def protein_only_loci(prot_hits: pd.DataFrame, nt_loci: pd.DataFrame) -> pd.DataFrame:
    """ORFs with FoTeR-RT or generic-RT hits that do not overlap any nt locus."""
    if prot_hits.empty:
        return pd.DataFrame()
    orf = prot_hits.groupby(["scaffold", "start", "end", "strand"], as_index=False).size()
    keep = []
    for r in orf.itertuples(index=False):
        sub = nt_loci[(nt_loci.scaffold == r.scaffold)] if not nt_loci.empty else nt_loci
        if not sub.empty and any(_overlap(r.start, r.end, a, b) > 0 for a, b in zip(sub.start, sub.end)):
            continue
        keep.append(dict(scaffold=r.scaffold, start=r.start, end=r.end, strand=r.strand, ref="none",
                         ref_from=np.nan, ref_to=np.nan, ref_len=np.nan, nt_score=0.0,
                         nt_ident=0.0, n_segments=0, n_refs_support=0,
                         min_miss5=np.nan, min_miss3=np.nan))
    return pd.DataFrame(keep)
