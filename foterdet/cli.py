"""foterdet command-line interface.

  python -m foterdet.cli benchmark --genomes config/benchmark_genomes.tsv --truth data/ground_truth_published.tsv ...
  python -m foterdet.cli predict   --genome new.fna --name GCA_xxx --model results/benchmark/model/foter_model.joblib
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import figures as fg
from .features import extract
from .model import OUTPUT_COLUMNS, Bundle, annotate, candidate_level_comparison, fit_bundle, label_candidates
from .reference import load_references, read_fasta

DEF_REFS = str(Path(__file__).resolve().parent.parent / "data/reference/FoTeR_full_length_refs.fasta")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, file=sys.stderr, flush=True)


def _med_orf(refs):
    return float(np.median([len(r.orf_prot) for r in refs.values()]))


def final_table(ann: pd.DataFrame) -> pd.DataFrame:
    t = ann[OUTPUT_COLUMNS].rename(columns=dict(genome="Assembly", scaffold="Scaffold/Chromosome", pred_start="Start",
                                                pred_end="End", strand="Strand", pred_length="Length",
                                                status="Full/Truncated", dist_to_end="Distance to Assembly End",
                                                telomeric_repeat_evidence="Telomeric Repeat Evidence",
                                                rt_evidence="RT Evidence", confidence="Confidence", call="Call"))
    return t.sort_values(["Assembly", "Scaffold/Chromosome", "Start"])


def cmd_benchmark(a):
    out = Path(a.outdir); (out / "model").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(exist_ok=True)
    refs = load_references(a.refs)
    gl = pd.read_csv(a.genomes, sep="\t", comment="#")
    truth = pd.read_csv(a.truth, sep="\t", comment="#")
    genomes, feats = {}, []
    for r in gl.itertuples(index=False):
        log(f"[{r.name}] reading {r.fasta}")
        genomes[r.name] = read_fasta(r.fasta)
        cache = out / f"features_{r.name}.tsv"
        if cache.exists() and not a.no_cache:
            f = pd.read_csv(cache, sep="\t")
        else:
            # leave-own-reference-out: this strain's published consensus is never used on its own genome
            own = getattr(r, 'own_ref', r.name)
            f = extract(genomes[r.name], refs, r.name, exclude={own}, rt_hmm_path=a.rt_hmm, cpus=a.cpus, log=log)
            f.to_csv(cache, sep="\t", index=False)
        feats.append(f)
    df = pd.concat(feats, ignore_index=True)
    y, gt_full = label_candidates(df, truth)
    log(f"{len(df)} candidates, {int(y.sum())} overlap published FoTeRs")

    # 1) candidate-level LOSO comparison: ML vs baselines vs ablations
    oof = candidate_level_comparison(df, y, log=log)
    oof.to_csv(out / "loso_candidate_oof.tsv", sep="\t", index=False)
    cm = ev.candidate_metrics(oof); cm.to_csv(out / "candidate_level_metrics.tsv", sep="\t", index=False)

    # 2) element-level LOSO: for each strain, a model frozen on the other 8, applied unmodified
    anns = []
    for g in gl.name:
        tr = df.genome != g
        b = fit_bundle(df[tr].reset_index(drop=True), y[tr].reset_index(drop=True), gt_full[tr].reset_index(drop=True))
        anns.append(annotate(df[df.genome == g].reset_index(drop=True), genomes[g], b, _med_orf(refs)))
        log(f"  [{g}] held-out threshold={b.threshold:.3f}")
    ann = pd.concat(anns, ignore_index=True)
    ann.to_csv(out / "benchmark_all_candidates_annotated.tsv", sep="\t", index=False)
    pos = ann[ann.call.isin(ev.POSITIVE_CALLS)]
    final_table(pos).to_csv(out / "PREDICTIONS_9_published_genomes.tsv", sep="\t", index=False)
    scaf = {g: set(v) for g, v in genomes.items()}
    per, tp, fp, fn, summ = ev.evaluate(ann, truth, scaf)
    per.to_csv(out / "COMPARISON_per_strain_metrics.tsv", sep="\t", index=False)
    tp.to_csv(out / "COMPARISON_matched_elements.tsv", sep="\t", index=False)
    fp.to_csv(out / "COMPARISON_false_positives.tsv", sep="\t", index=False)
    fn.to_csv(out / "COMPARISON_false_negatives.tsv", sep="\t", index=False)
    json.dump(summ, open(out / "COMPARISON_summary.json", "w"), indent=2)

    # 3) final frozen model on all nine strains
    b = fit_bundle(df, y, gt_full); mp = out / "model/foter_model.joblib"; b.save(mp)
    sha = hashlib.sha256(open(mp, "rb").read()).hexdigest()
    json.dump(dict(threshold=b.threshold, uncertain_low=b.uncertain_low, features=b.features,
                   n_candidates=len(df), n_positive=int(y.sum()), sha256=sha,
                   trained_on=list(gl.name)), open(out / "model/model_card.json", "w"), indent=2)
    log(f"frozen model sha256={sha[:12]}…")

    # figures
    fg.pr_curves(oof, out / "figures/fig1_pr_curves_loso.png")
    fg.bars(per[per.genome != "ALL (micro)"], "genome", ["precision", "recall", "F1"], out / "figures/fig2_per_strain.png",
            "Element-level performance per held-out strain")
    fg.bars(cm, "method", ["precision", "recall", "F1"], out / "figures/fig3_methods_ablation.png",
            "Candidate-level LOSO: ML vs baselines vs ablations")
    fg.boundary_hist(tp, out / "figures/fig4_boundary_error.png")
    fg.feature_distributions(df, y, out / "figures/fig5_feature_distributions.png")
    if len(pos):
        top = pos.sort_values("confidence", ascending=False).iloc[0]
        fg.locus_plot(genomes[top.genome], ann[ann.genome == top.genome], top.scaffold, top.nearest_end,
                      out / "figures/fig6_example_locus.png")
    log("benchmark summary: " + json.dumps(summ))


def run_predict(fasta, name, bundle, refs, rt_hmm, cpus):
    g = read_fasta(fasta)
    f = extract(g, refs, name, exclude=None, rt_hmm_path=rt_hmm, cpus=cpus, log=log)
    return annotate(f, g, bundle, _med_orf(refs)), g


def cmd_predict(a):
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    refs = load_references(a.refs); b = Bundle.load(a.model)
    jobs = pd.read_csv(a.genomes, sep="\t", comment="#") if a.genomes else pd.DataFrame(dict(name=[a.name], fasta=[a.genome]))
    alls, tabs = [], []
    for r in jobs.itertuples(index=False):
        log(f"[{r.name}] predicting with frozen model")
        ann, g = run_predict(r.fasta, r.name, b, refs, a.rt_hmm, a.cpus)
        ann.to_csv(out / f"{r.name}_all_candidates.tsv", sep="\t", index=False)
        alls.append(ann)
        keep = ann[ann.call.isin(ev.POSITIVE_CALLS + ("uncertain",))]
        tabs.append(final_table(keep))
        if len(keep) and a.plots:
            (out / "figures").mkdir(exist_ok=True)
            top = keep.sort_values("confidence", ascending=False).iloc[0]
            fg.locus_plot(g, ann, top.scaffold, top.nearest_end, out / f"figures/{r.name}_top_locus.png")
    tab = pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
    tab.to_csv(out / "PREDICTIONS_new_genomes.tsv", sep="\t", index=False)
    if alls:
        fg.confidence_hist(pd.concat(alls), out / "confidence_distribution.png")
        s = pd.concat(alls).groupby(["genome", "call"]).size().unstack(fill_value=0)
        s.to_csv(out / "summary_calls_per_genome.tsv", sep="\t")
        log("\n" + s.to_string())


def main(argv=None):
    p = argparse.ArgumentParser(prog="foterdet")
    sp = p.add_subparsers(dest="cmd", required=True)
    for n in ("benchmark", "predict"):
        s = sp.add_parser(n)
        s.add_argument("--refs", default=DEF_REFS)
        s.add_argument("--rt-hmm", default=None, help="Pfam PF00078 RVT_1 HMM (generic RT); strongly recommended")
        s.add_argument("--cpus", type=int, default=0)
        s.add_argument("--outdir", required=True)
    b = sp.choices["benchmark"]
    b.add_argument("--genomes", required=True, help="TSV: name<TAB>fasta")
    b.add_argument("--truth", required=True, help="TSV of published FoTeRs: genome scaffold start end strand status")
    b.add_argument("--no-cache", action="store_true")
    q = sp.choices["predict"]
    q.add_argument("--model", required=True)
    q.add_argument("--genome"); q.add_argument("--name"); q.add_argument("--genomes", help="TSV name<TAB>fasta")
    q.add_argument("--plots", action="store_true")
    a = p.parse_args(argv)
    {"benchmark": cmd_benchmark, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
