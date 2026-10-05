#!/usr/bin/env bash
# Reproduce every result in the README from the raw CHB-MIT EDF files.
#
#   bash run_all.sh /path/to/chb-mit          # folder containing chb01/ ... chb24/
#
# Already have data/processed_v3 from the ST-GNN repo? Link it and the preprocessing step is skipped:
#   ln -s ~/seizure_stgnn_v3/data data
set -euo pipefail

RAW=${1:?usage: bash run_all.sh /path/to/chb-mit}
P=data/processed_v3

ev() {  # seizure-level metrics (all seizures, then lead seizures only) + figures
  python -m src.evaluate --results_dir "results/$1"
  python -m src.evaluate --results_dir "results/$1" --lead_gap_h 4 --processed_dir "$P" \
    --out_dir "results/$1/lead4h"
  python -m src.figures  --results_dir "results/$1"
}

python -m pytest -q tests
[ -d "$P" ] || python -m src.preprocess --raw_dir "$RAW" --out_dir "$P" --workers 4

for PR in lopo chrono; do
  for A in eegnet spectro; do
    python -m src.train   --processed_dir "$P" --arch $A --protocol $PR --out_dir results/${A}_$PR
    python -m src.context --processed_dir "$P" --source results/${A}_$PR --out_dir results/context_${A}_$PR
    ev ${A}_$PR
    ev context_${A}_$PR
  done
done
echo "Done. Results in results/*/results.md"
