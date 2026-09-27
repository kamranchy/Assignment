"""Stage B: biologically interpretable features for every candidate locus.

Feature groups (used for ablations in the report):
  HOMOLOGY  - similarity to FoTeR references (nucleotide + FoTeR RT protein)
  SPECIFIC  - FoTeR-RT vs generic-RT discrimination, REL motif, ORF integrity
  CONTEXT   - position relative to scaffold end, orientation, end arrays
  TELOMERE  - TTAGGG/CCCTAA evidence in flanks and at the scaffold end
  COMPOS    - GC and RIP indices (RIP-mutated copies are expected)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import telomere as tel
from .candidates import (chain_nt_hits, collapse_chains, nt_search, prot_search,
                         protein_only_loci, six_frame_orfs)
from .reference import (REL_CORE, build_prot_hmms, load_hmm_file,
                        longest_orf_protein, revcomp)

FEATURE_GROUPS = {
    "HOMOLOGY": ["nt_score", "nt_score_per_kb", "nt_identity", "n_refs_support", "ref_cov_frac",
                 "ref_from_frac", "ref_to_frac", "prot_foter_score"],
    "SPECIFIC": ["prot_generic_rt_score", "foter_minus_generic", "rel_motif", "longest_orf_aa", "orf_frac_of_ref"],
    "CONTEXT": ["log10_dist_to_end", "orient_consistent", "n_cands_to_end", "log10_max_gap_to_end",
                "log10_scaffold_len"],
    "TELOMERE": ["tel5_units", "tel3_units", "tel5_run", "tel3_run", "tel5_adjacent", "tel3_adjacent",
                 "end_tel_units", "end_complete"],
    "COMPOS": ["gc", "rip_substrate", "rip_product"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
# features for the full-length vs truncated sub-model
STATUS_FEATURES = ["ref_from_frac", "ref_to_frac", "min_miss5", "min_miss3", "len_ratio", "longest_orf_aa",
                   "orf_frac_of_ref", "tel5_units", "tel3_units", "tel5_adjacent", "tel3_adjacent", "rel_motif"]


def _dinuc(s, d):
    return s.count(d)


def composition(s: str) -> dict:
    n = max(1, len(s))
    gc = (s.count("G") + s.count("C")) / n
    ca_tg = _dinuc(s, "CA") + _dinuc(s, "TG")
    ac_gt = _dinuc(s, "AC") + _dinuc(s, "GT")
    ta, at = _dinuc(s, "TA"), _dinuc(s, "AT")
    return dict(gc=gc, rip_substrate=ca_tg / max(1, ac_gt), rip_product=ta / max(1, at))


def build_candidates(genome, refs, exclude=None, rt_hmm_path=None, cpus=0, min_orf_aa=150, log=print):
    exclude = set(exclude or [])
    ref_len = {r.name: r.length for r in refs.values()}
    log(f"  minimap2 with {len(refs) - len(exclude & set(refs))} reference FoTeRs (excluded: {sorted(exclude) or 'none'})")
    hits = nt_search(genome, refs, exclude)
    loci = collapse_chains(chain_nt_hits(hits, ref_len))
    log(f"  {len(hits)} nt HSPs -> {len(loci)} nucleotide loci")
    orfs = six_frame_orfs(genome, min_orf_aa)
    log(f"  {len(orfs)} six-frame ORFs >= {min_orf_aa} aa")
    fo = prot_search(orfs, build_prot_hmms(refs, exclude), "foter", cpus=cpus)
    gen = prot_search(orfs, load_hmm_file(rt_hmm_path), "generic", cpus=cpus)
    prot = pd.concat([fo, gen], ignore_index=True)
    ponly = protein_only_loci(prot, loci)
    log(f"  protein hits: FoTeR-RT {len(fo)}, generic RT {len(gen)}; protein-only loci {len(ponly)}")
    cands = pd.concat([loci, ponly], ignore_index=True)
    cands["source"] = np.where(cands["nt_score"] > 0, "nucleotide", "protein_only")
    return cands, prot


def _best_prot(prot, sc, a, b, strand, label):
    if prot.empty:
        return 0.0
    p = prot[(prot.label == label) & (prot.scaffold == sc) & (prot.strand == strand) & (prot.end >= a) & (prot.start <= b)]
    return float(p.score.max()) if len(p) else 0.0


def compute_features(genome, cands, prot, refs, genome_name="genome"):
    if cands.empty:
        return pd.DataFrame(columns=["genome", "cand_id"] + ALL_FEATURES)
    ref_orf_len = {r.name: len(r.orf_prot) for r in refs.values()}
    med_orf = float(np.median(list(ref_orf_len.values())))
    ends = {sc: tel.end_status(s) for sc, s in genome.items()}
    rows = []
    for i, c in cands.reset_index(drop=True).iterrows():
        s = genome[c.scaffold]
        L = len(s)
        a, b = int(c.start), int(c.end)
        seg = s[a - 1: b]
        d_left, d_right = a - 1, L - b
        near = "left" if d_left <= d_right else "right"
        es = ends[c.scaffold]
        end_units = es["left_units"] if near == "left" else es["right_units"]
        # element 5' end should face the chromosome terminus
        orient_ok = (c.strand == "+" and near == "left") or (c.strand == "-" and near == "right")
        ref_len = c.ref_len if pd.notna(c.ref_len) else np.nan
        el = seg if c.strand == "+" else revcomp(seg)
        orf, *_ = longest_orf_protein(el)
        pf = _best_prot(prot, c.scaffold, a, b, c.strand, "foter")
        pg = _best_prot(prot, c.scaffold, a, b, c.strand, "generic")
        rec = dict(
            genome=genome_name, cand_id=f"{genome_name}:{c.scaffold}:{a}-{b}:{c.strand}",
            scaffold=c.scaffold, start=a, end=b, strand=c.strand, length=b - a + 1, scaffold_len=L,
            best_ref=c.ref, source=c.source, ref_from=c.ref_from, ref_to=c.ref_to, ref_len=ref_len,
            nearest_end=near, dist_to_end=min(d_left, d_right),
            nt_score=c.nt_score, nt_score_per_kb=1000 * c.nt_score / max(1, b - a + 1),
            nt_identity=float(c.nt_ident), n_refs_support=c.n_refs_support,
            ref_cov_frac=(c.ref_to - c.ref_from + 1) / ref_len if pd.notna(ref_len) else 0.0,
            ref_from_frac=(c.ref_from - 1) / ref_len if pd.notna(ref_len) else np.nan,
            ref_to_frac=c.ref_to / ref_len if pd.notna(ref_len) else np.nan,
            min_miss5=c.min_miss5, min_miss3=c.min_miss3, len_ratio=(b - a + 1) / ref_len if pd.notna(ref_len) else np.nan,
            prot_foter_score=pf, prot_generic_rt_score=pg, foter_minus_generic=pf - pg,
            rel_motif=int(bool(REL_CORE.search(orf))), longest_orf_aa=len(orf),
            orf_frac_of_ref=len(orf) / med_orf,
            log10_dist_to_end=np.log10(1 + min(d_left, d_right)), orient_consistent=int(orient_ok),
            log10_scaffold_len=np.log10(L), end_tel_units=end_units, end_complete=int(end_units >= tel.MIN_TERMINAL_UNITS),
        )
        rec.update({k: (int(v) if isinstance(v, bool) else v) for k, v in tel.flank_evidence(s, a, b, c.strand).items()})
        rec.update(composition(seg))
        rows.append(rec)
    df = pd.DataFrame(rows)
    # tandem-array context: how many candidates lie between this one and the
    # nearest end, and the largest gap along that path (an uninterrupted array
    # of FoTeRs running into the telomere has only small gaps).
    df["n_cands_to_end"] = 0
    df["log10_max_gap_to_end"] = 0.0
    for sc, g in df.groupby("scaffold"):
        g = g.sort_values("start")
        L = g.scaffold_len.iloc[0]
        starts, ends_ = g.start.to_numpy(), g.end.to_numpy()
        for k, idx in enumerate(g.index):
            if g.loc[idx, "nearest_end"] == "left":
                path = [starts[0] - 1] + [starts[j + 1] - ends_[j] for j in range(k)]
                n_between = k
            else:
                n = len(g)
                path = [L - ends_[-1]] + [starts[j] - ends_[j - 1] for j in range(n - 1, k, -1)]
                n_between = n - 1 - k
            df.loc[idx, "n_cands_to_end"] = n_between
            df.loc[idx, "log10_max_gap_to_end"] = np.log10(1 + max(0, max(path)))
    return df


def extract(genome, refs, genome_name, exclude=None, rt_hmm_path=None, cpus=0, log=print):
    cands, prot = build_candidates(genome, refs, exclude, rt_hmm_path, cpus, log=log)
    return compute_features(genome, cands, prot, refs, genome_name)
