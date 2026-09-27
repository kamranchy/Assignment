# foterdet — automated detection of FoTeR retrotransposons in *Fusarium oxysporum*

Given a genome FASTA, `foterdet` reports every predicted FoTeR (F. oxysporum
Telomeric Retrotransposon) with location, strand, boundaries, full-length /
truncated / unresolved status, distance to the assembly end, telomeric-repeat
evidence, reverse-transcriptase evidence and a calibrated confidence — with no
manual inspection.

Pipeline: **candidate generation** (minimap2 against reference FoTeRs + profile-HMM
search of six-frame ORFs with FoTeR-RT and generic RT models) → **biological
features** (homology, FoTeR-vs-generic-RT specificity, REL motif, ORF integrity,
chromosome-end context, orientation, TTAGGG/CCCTAA evidence, RIP indices) →
**gradient-boosted classifier** validated leave-one-strain-out, isotonic-calibrated,
plus a learned full-length/truncated sub-model → **structural annotation**
(boundary refinement to telomeric junctions, "unresolved" when the assembly end
is incomplete). See `report/REPORT.md` for the rationale.

## 1. Install

```bash
conda env create -f environment.yml && conda activate foterdet
# or: pip install -r requirements.txt   (+ NCBI datasets CLI for downloads)
```

## 2. Predict FoTeRs in ONE new genome (the common use case)

```bash
scripts/get_pfam_rt.sh                              # once: Pfam PF00078 (generic RT)
python -m foterdet.cli predict \
    --genome my_strain.fna --name my_strain \
    --model results/benchmark/model/foter_model.joblib \
    --rt-hmm data/hmm/PF00078.hmm --cpus 8 --plots --outdir results/my_strain
```

Outputs in `--outdir`:

| file | content |
|---|---|
| `PREDICTIONS_new_genomes.tsv` | final table (FoTeR, FoTeR-telomere-unresolved and *uncertain* calls) |
| `<name>_all_candidates.tsv` | every candidate with all features, confidence and call (audit trail) |
| `summary_calls_per_genome.tsv`, `confidence_distribution.png`, `figures/` | summaries |

Columns of the final table: Assembly, Scaffold/Chromosome, Start, End, Strand,
Length, Full/Truncated, Distance to Assembly End, Telomeric Repeat Evidence,
RT Evidence, Confidence, Call. `Call` is one of
`FoTeR`, `FoTeR (telomere unresolved: assembly end incomplete)`, `uncertain`;
the all-candidates file additionally contains `non-FoTeR` and
`FoTeR-like sequence, non-telomeric (rejected by context)`.

## 3. Reproduce the whole assignment

1. Fill the `accession` column of `config/benchmark_genomes.tsv` from Table S1A of
   Salimi & Rahnama (2025) — use the exact assembly versions.
2. Transcribe Table S5 into `data/ground_truth_published.tsv` using
   `data/ground_truth_TEMPLATE.tsv` (delete the EXAMPLE row). Scaffold names must
   match the downloaded assemblies.
3. Run `CPUS=8 scripts/run_all.sh`. This
   * downloads the 9 + 8 assemblies,
   * **Part 3:** runs the leave-one-strain-out benchmark and writes
     `results/benchmark/PREDICTIONS_9_published_genomes.tsv`,
     `COMPARISON_*.tsv/json`, `candidate_level_metrics.tsv`, `figures/`,
     then trains and **freezes** the final model (sha256 in `model/model_card.json`);
   * **Part 4:** applies the frozen model, unmodified, to the 8 new assemblies →
     `results/new_genomes/PREDICTIONS_new_genomes.tsv`;
   * writes `results/RESULTS_SUMMARY.md` (tables for the report).

## 4. Smoke test (no downloads, ~2 min)

```bash
tests/run_smoke_test.sh
```
Builds simulated genomes from the real reference FoTeRs (with decoy RT elements,
internal FoTeR fragments, incomplete ends) and runs benchmark + predict.
**Simulated metrics only prove the code runs; they are not results.**

## 5. Layout

```
foterdet/reference.py   reference FoTeRs, ORFs, REL motif, HMM building
foterdet/candidates.py  stage A: minimap2 + six-frame ORF HMM search, chaining, locus merging
foterdet/telomere.py    TTAGGG/CCCTAA runs, end completeness, junction boundary refinement
foterdet/features.py    stage B: feature groups
foterdet/model.py       stage C/D: LOSO training, calibration, baselines, annotation
foterdet/evaluate.py    element-level comparison with published FoTeRs, FP/FN reasons
foterdet/figures.py     figures
foterdet/cli.py         `benchmark` and `predict` commands
data/reference/         9 published full-length FoTeRs (github.com/RahnamaLab/FoTeR_Project)
config/                 genome lists; scripts/ download + run helpers; tests/ smoke test
```

## 6. Compute resources

Designed for a laptop/workstation. Per ~50-Mb genome: minimap2 stage < 1 min,
six-frame ORF + HMM search a few minutes on one core (scales with `--cpus`),
feature extraction < 1 min; model training on the nine strains takes minutes
(no GPU). Memory < 4 GB. Record your actual CPU model, core count, RAM and
wall-clock times in the report (Section 9).

## 7. Provenance

* Reference sequences: published by the original authors (public GitHub).
* Ground truth: Table S5 of the paper, transcribed by the user.
* Everything else (candidate generation, features, model, annotation) is this
  pipeline's own methodology. No manual edits are made to predictions before
  scoring; FP/FN "reasons" are generated automatically from features.
