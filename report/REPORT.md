# Automated detection of FoTeR retrotransposons with machine learning

**Author:** Proloy · **Code:** `foterdet/` (this repository)

Provenance tags used throughout:
**[PAPER]** information from Salimi & Rahnama, *Mobile DNA* 2025 (PMID 41287068);
**[METHOD]** my computational methodology; **[AUTO]** automated model output;
**[POST-HOC]** anything I examined manually *after* predictions were fixed.

> Sections marked ⟦FILL⟧ are completed from `results/RESULTS_SUMMARY.md`
> after running `scripts/run_all.sh`. No numbers in this report are edited by hand.

---

## 1. The original identification problem (Part 1) [PAPER]

**How candidates were found.** The MoTeR1 (*Magnaporthe oryzae*) RT ORF protein
was used as a tBLASTn query against each of nine assemblies; 5 kb of flank was
extracted on each side of each hit, candidates were aligned (Clustal Omega), and
the alignments were **manually inspected** to set element boundaries and confirm
telomere association. A per-strain consensus was then used to find all copies.

**What makes a FoTeR a FoTeR.** Non-LTR retrotransposon with one large ORF
(~1,150–1,184 aa) encoding an RT domain (RVT_1/PF00078) followed by a C-terminal
restriction-enzyme-like endonuclease (REL) domain with a
C‑X2‑C‑X6‑H‑X3‑C‑X9‑RH(D/N) core; N-terminal zinc-finger motifs; VNTRs concentrated
in the 5′ region; element length 4,291–7,285 bp. RT and REL phylogenies place
FoTeRs in a clade with MoTeR1, Cnl1, SLACS, CRE1/2 and CZAR, with two subgroups
(RHNA-type: Race3, MAFF727510, GL1381, GL1315, GL1080, BRIP62122; RHDK-type: II5,
Fo5176, Fo47). Many copies are decayed (indels, RIP C→T at CpA).

**Telomere association.** Elements are found only at chromosome ends — none at
internal loci — with the 5′ end oriented toward the terminus. The 5′ junction
carries a short (often single) telomeric motif; the 3′ end is followed by 3–7
TAACCC units. Because many assembly ends did not reach the telomere, the authors
aligned PacBio reads (minimap2) and inspected them in IGV; reads containing 10–30
consecutive TTAGGG units defined a complete end.

**Full-length vs truncated.** Full-length = telomeric repeats at both 5′ and 3′
ends + a single RT-encoding ORF between them; truncated = incomplete structure.
In truncated copies the non-truncated side always retains telomeric repeats.
Copy numbers range from 4 (Race3) to 55 (GL1080); truncated copies outnumber
full-length ones in every strain.

**Manual steps.** Boundary definition, confirmation of telomere association,
full vs truncated classification, PacBio-based end extension and ambiguity
resolution in IGV.

**Features useful for automation (my reading).** (i) similarity to FoTeR — not
just to any RT; (ii) the RT+REL architecture and REL motif; (iii) ORF integrity;
(iv) distance to a scaffold end and 5′-toward-terminus orientation; (v) telomeric
motifs at the junctions and a terminal tract; (vi) tandem arrays of copies running
into the telomere; (vii) RIP signature (explains decayed copies rather than
excluding them).

## 2. Problem formulation [METHOD]

I treat detection as **locus classification plus structural annotation**, not as
sequence-window classification:

1. *Where could a FoTeR be?* — a high-recall candidate generator.
2. *Is this candidate a FoTeR?* — a learned classifier over biologically
   interpretable evidence, which must separate FoTeRs from other RT elements,
   unrelated TEs, internal sequence and weak partial hits.
3. *What does it look like?* — boundaries, orientation, completeness, telomere
   relationship, with an explicit **unresolved** state when the assembly end is
   incomplete (absence of TTAGGG ≠ absence of FoTeR).

**Why not end-to-end deep learning.** The labelled data are nine strain-level
consensus elements and a few hundred copies that are near-identical within a
strain (within-strain ORF identity up to 100% [PAPER]). The effective number of
independent examples is ≈ 9. A CNN/transformer on raw windows would learn
strain-specific k-mers; with a random window split, fragments of the same element
fall in train and test, inflating accuracy (I note that a naive window-level
approach reaching ~96% accuracy would be exactly this artefact). The appropriate
unit of generalisation is the **strain**, so the model is small, regularised and
feature-based, and every estimate is leave-one-strain-out.

## 3. Pipeline (Part 2) [METHOD]

**Reference data.** The nine full-length FoTeRs published by the authors
(github.com/RahnamaLab/FoTeR_Project). Their longest ORFs are 1,137–1,184 aa, all
on the + strand, and the REL core motif recovers RHN in the RHNA-type and RHD in
the RHDK-type strains, matching the paper (motif not recovered by my regex in
GL1080) [AUTO].

**Stage A — candidates.** (a) minimap2 (`map-ont` preset, tolerant to ~15–20%
divergence), one index per reference, genome streamed as 20-kb chunks (10-kb step)
so all copies are reported; collinear chaining per reference; chains from all
references merged by single-linkage overlap (≥50 bp), which keeps tandem copies
separate (they are separated by telomeric repeats) but merges fragments of one copy
split by VNTR-length differences. (b) Six-frame stop-to-stop ORFs ≥150 aa searched
with profile HMMs of the FoTeR RT proteins and Pfam RVT_1 (pyhmmer). ORFs with RT
evidence that do not overlap a nucleotide locus become protein-only candidates —
they add diverged copies and, crucially, other RT-containing elements as hard
negatives. (nhmmer was tried first and abandoned: with single-sequence ~5-kb DNA
profiles it was orders of magnitude slower.)

**Stage B — features** (`foterdet/features.py`), five groups:
HOMOLOGY (aligned bases, identity, number of references supporting, reference
coverage, FoTeR-RT HMM score); SPECIFIC (generic RT score, FoTeR-minus-generic
score, REL motif, longest ORF); CONTEXT (log distance to nearest end, 5′-toward-end
orientation, candidates between it and the end, largest gap along that path,
scaffold length); TELOMERE (TTAGGG/CCCTAA units and longest run in 300-bp 5′/3′
flanks, motif directly at the junction, terminal tract units, end complete = ≥10
units); COMPOS (GC, RIP substrate and product indices).

**Stage C — models.** HistGradientBoosting (depth 3, L2, balanced classes).
Labels: a candidate is positive if it overlaps a published FoTeR by ≥50% of the
shorter interval. A second small model predicts full-length vs truncated from
end-coverage, length ratio, ORF and junction features; a rule-based version is
reported alongside for comparison.

**Stage D — annotation.** Boundaries are the union of reference alignments,
extended over small unaligned ends and refined outward to the adjacent telomeric
motif within a junction window (because the 5′ ~150–250 bp are strain-specific and
covered by no other reference — observed in simulation). Status is
`full-length`, `truncated (5′/3′)`, or `unresolved` when an incompletely covered
end runs into the scaffold edge. Calls: `FoTeR`, `FoTeR (telomere unresolved)`
when the end has no telomeric evidence but the assembly end is incomplete,
`uncertain`, `non-FoTeR`, and `FoTeR-like sequence, non-telomeric` (high
sequence-only score but rejected by context — reported, not hidden).

## 4. Training, validation and leakage control [METHOD]

* **Leave-own-reference-out features:** each benchmark genome's features are
  computed with the other eight references only; a strain's own consensus is never
  aligned to its own genome.
* **Leave-one-strain-out (LOSO):** nine folds; all copies of a strain are in the
  same fold. Thresholds are chosen by an *inner* LOSO on the eight training strains
  (nested), then applied to the held-out strain.
* **Element-level benchmark:** for each held-out strain a complete model
  (classifier + calibration + threshold + status model) is fitted on the other
  eight and applied unmodified, via the same code path used for new genomes.
* **Harder split (optional):** leave-one-subgroup-out (RHNA vs RHDK) — to report
  if time allows, as an upper bound on divergence-driven failure.
* **Freezing:** the final model is fitted on all nine strains, saved with its
  sha256, and never retrained after seeing Part 4 genomes. The Part 4 genomes
  (including GCA_050916195.1) were not used for any design decision.
* **Baselines:** B1 homology score only (BLAST-like); B2 homology + "within 30 kb
  of an end" rule; ablations removing each feature group.

## 5. Results — published genomes (Part 3) [AUTO]

⟦FILL from RESULTS_SUMMARY.md⟧

* Table 5.1 — per-strain published / recovered / missed / additional, P, R, F1.
* Boundary agreement: median |Δstart|, |Δend|, fraction within 50 bp (Fig. 4).
* Full vs truncated accuracy (learned vs rule).
* Telomere association: fraction of TPs called `FoTeR` vs `telomere unresolved`.
* Table 5.2 — ML vs baselines vs ablations (Fig. 1 PR curves, Fig. 3).
* False positives and false negatives by automated reason
  (`COMPARISON_false_*.tsv`). Expected FN classes: elements on PacBio-extended ends
  absent from the public assembly; short/heavily decayed truncated copies below
  minimap2 sensitivity with no RT ORF. Expected FP classes: fragments of published
  copies, subtelomeric non-FoTeR RT elements, genuinely unlisted copies.
* [POST-HOC] ⟦list any FPs you inspected afterwards and what you concluded; these
  inspections do not change the metrics⟧.

Pipeline validation on simulated genomes (NOT a result): with simulated
chromosome ends built from the real references, decoy RT elements and internal
fragments, the full model separated all simulated FoTeRs, the BLAST-like baseline
had F1 ≈ 0.93 at candidate level, and the learned status model outperformed the
rule (≈0.92 vs 0.79). This shows the code works and that context features are what
remove homology-only false positives — it says nothing about real performance.

## 6. Results — new genomes (Part 4) [AUTO]

⟦FILL: `results/new_genomes/PREDICTIONS_new_genomes.tsv` (full table in
supplementary), `summary_calls_per_genome.tsv`, per-genome top-locus figures⟧.

## 7. Interpretation (Part 5)

Answers marked ⟦confirm⟧ state my expectation and must be checked against §5–6.

1. **Reproducing the manual FoTeRs.** ⟦FILL recall/precision⟧. Recall is bounded
   by the public assemblies: ends that were completed with PacBio in the paper
   cannot be recovered and are reported separately.
2. **Most informative characteristics.** ⟦confirm with ablations⟧ Expected:
   chromosome-end context and orientation (removes internal and generic RT hits),
   FoTeR-specific RT score relative to generic RT, reference coverage; telomeric
   junction motifs mainly matter for boundaries and completeness.
3. **FoTeRs in the eight new genomes.** ⟦FILL⟧
4. **Divergent predictions.** Flag predictions whose best reference identity is
   low, whose REL motif type is new, or whose length/5′ structure differs from all
   nine references; `FoTeR-like, non-telomeric` calls would contradict the paper's
   "only at chromosome ends" finding and deserve follow-up. ⟦FILL⟧
5. **Hardest FoTeRs.** Short 5′-truncated copies (little sequence, no ORF),
   RIP-/indel-decayed copies, copies on incomplete assembly ends, and copies in
   collapsed/mis-assembled subtelomeric repeats. ⟦confirm with FN reasons⟧
6. **FP/FN sources.** See §5 automated reasons.
7. **Generalisation to a new *F. oxysporum* genome.** LOSO is the direct estimate.
   Risk rises with divergence from both subgroups and with assembly quality
   (fragmented ends → more `unresolved`). The model never needs the new strain's
   own consensus, which is exactly the LOSO setting.
8. **Scaling to thousands of genomes.** Replace per-chunk mapping with a single
   minimap2/MMseqs2 run per genome; restrict HMM search to ORFs; iterative
   self-training (add confident new copies to the reference panel, retrain with
   strain-grouped CV); positive-unlabelled learning since unlisted copies are not
   true negatives; protein language-model embeddings of the RT/REL region when
   the number of independent FoTeR families grows; long-read-aware end
   completeness; Snakemake/Nextflow with per-genome QC (BUSCO, telomere-to-telomere
   end counts).

## 8. Limitations
Ground truth depends on manual transcription of Table S5 and on scaffold naming;
only nine independent families; per-strain 5′ regions are unique so 5′ boundaries
rely on telomeric-junction refinement; "truncated vs decayed full-length" is a
judgement call in the paper itself; no RT activity is inferred.

## 9. Computational resources
⟦FILL: CPU, cores, RAM, OS, wall-clock per genome and for training⟧. No GPU.

## 10. Conclusions
⟦FILL after results⟧
