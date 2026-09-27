"""Turn pipeline outputs into markdown tables to paste into report/REPORT.md.
Usage: python scripts/results_to_markdown.py results > results/RESULTS_SUMMARY.md
Everything printed here is AUTOMATED output; nothing is edited by hand."""
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
b, n = root / "benchmark", root / "new_genomes"


def md(df, nd=3):
    return df.round(nd).to_markdown(index=False) if len(df) else "_none_"


print("# Automated results summary\n")
if b.exists():
    print("## Part 3 — element-level comparison with the published FoTeRs (leave-one-strain-out)\n")
    print(md(pd.read_csv(b / "COMPARISON_per_strain_metrics.tsv", sep="\t")), "\n")
    print("```json\n" + json.dumps(json.load(open(b / "COMPARISON_summary.json")), indent=2) + "\n```\n")
    print("## Candidate-level: ML vs baselines vs feature-group ablations\n")
    print(md(pd.read_csv(b / "candidate_level_metrics.tsv", sep="\t")), "\n")
    for kind in ("false_positives", "false_negatives"):
        f = b / f"COMPARISON_{kind}.tsv"
        d = pd.read_csv(f, sep="\t") if f.stat().st_size > 1 else pd.DataFrame()
        print(f"## {kind.replace('_', ' ').title()} by automated reason\n")
        print(md(d.groupby("reason").size().rename("n").reset_index()) if len(d) else "_none_", "\n")
    card = json.load(open(b / "model/model_card.json"))
    print(f"Frozen model sha256: `{card['sha256']}`; threshold {card['threshold']:.3f}\n")
if n.exists():
    print("## Part 4 — calls per new genome (frozen model)\n")
    print(pd.read_csv(n / "summary_calls_per_genome.tsv", sep="\t").to_markdown(index=False), "\n")
