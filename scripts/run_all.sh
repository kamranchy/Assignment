#!/usr/bin/env bash
# Full reproduction: Part 3 (benchmark on 9 published genomes) then Part 4 (frozen model on 8 new genomes).
set -euo pipefail
CPUS=${CPUS:-8}
TRUTH=${TRUTH:-data/ground_truth_published.tsv}
[[ -s data/hmm/PF00078.hmm ]] || scripts/get_pfam_rt.sh
scripts/download_genomes.sh config/benchmark_genomes.tsv genomes/benchmark
scripts/download_genomes.sh config/new_genomes.tsv genomes/new

python -m foterdet.cli benchmark --genomes config/benchmark_genomes.tsv --truth "$TRUTH" \
    --rt-hmm data/hmm/PF00078.hmm --cpus "$CPUS" --outdir results/benchmark

# ---- model is frozen here; its sha256 is recorded in results/benchmark/model/model_card.json
python -m foterdet.cli predict --genomes config/new_genomes.tsv \
    --model results/benchmark/model/foter_model.joblib --rt-hmm data/hmm/PF00078.hmm \
    --cpus "$CPUS" --plots --outdir results/new_genomes

python scripts/results_to_markdown.py results > results/RESULTS_SUMMARY.md
echo "done -> results/RESULTS_SUMMARY.md"
