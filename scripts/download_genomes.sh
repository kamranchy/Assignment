#!/usr/bin/env bash
# Usage: scripts/download_genomes.sh config/new_genomes.tsv genomes/new
# Requires NCBI datasets CLI (conda install -c conda-forge ncbi-datasets-cli).
set -euo pipefail
tsv=$1; outdir=$2; mkdir -p "$outdir"
grep -v '^#' "$tsv" | tail -n +2 | while IFS=$'\t' read -r name acc rest; do
  [[ "$acc" == FILL* ]] && { echo "!! $name: accession not filled in $tsv" >&2; continue; }
  [[ -s "$outdir/$name.fna" ]] && { echo "== $name exists"; continue; }
  echo "== downloading $name ($acc)"
  tmp=$(mktemp -d)
  datasets download genome accession "$acc" --include genome --filename "$tmp/pkg.zip"
  unzip -q "$tmp/pkg.zip" -d "$tmp"
  cat "$tmp"/ncbi_dataset/data/"$acc"/*.fna > "$outdir/$name.fna"
  rm -rf "$tmp"
done
