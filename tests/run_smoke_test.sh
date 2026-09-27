#!/usr/bin/env bash
# End-to-end test on SIMULATED genomes (built from the real reference FoTeRs).
# Proves the code runs; its metrics are NOT assignment results.
set -euo pipefail
cd "$(dirname "$0")/.."
python tests/make_synthetic.py tests/synthetic
python -m foterdet.cli benchmark --genomes tests/synthetic/genomes.tsv --truth tests/synthetic/truth.tsv \
    --outdir tests/synthetic_run --no-cache
python -m foterdet.cli predict --genome tests/synthetic/SYN_Fo47.fna --name SYN_TEST \
    --model tests/synthetic_run/model/foter_model.joblib --outdir tests/predict_run --plots
echo "SMOKE TEST OK"
