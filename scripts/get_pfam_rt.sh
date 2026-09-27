#!/usr/bin/env bash
# Generic reverse-transcriptase profile (Pfam PF00078, RVT_1) used as the
# "any RT" contrast feature (FoTeR-RT score minus generic-RT score).
set -euo pipefail
mkdir -p data/hmm
wget -q -O data/hmm/PF00078.hmm.gz "https://www.ebi.ac.uk/interpro/wwwapi//entry/pfam/PF00078?annotation=hmm"
gunzip -f data/hmm/PF00078.hmm.gz
grep -m1 '^NAME' data/hmm/PF00078.hmm
