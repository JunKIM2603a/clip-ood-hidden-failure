#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BACKBONE="ViT-B/32"

for f in \
  results/tables/reproduction_ViT-B-32_mcm.csv \
  results/tables/reproduction_ViT-B-32_neglabel.csv
do
  if [[ ! -f "${f}" ]]; then
    echo "[FAIL] missing Stage-1A aggregate result: ${f}" >&2
    exit 2
  fi
done

echo "[H1] predefined semantic groups: MCM"
python scripts/analysis/run_h1_predefined.py \
  --method mcm \
  --backbone "${BACKBONE}" &
PID_MCM=$!

echo "[H1] predefined semantic groups: NegLabel"
python scripts/analysis/run_h1_predefined.py \
  --method neglabel \
  --backbone "${BACKBONE}" &
PID_NEG=$!

wait "${PID_MCM}"
wait "${PID_NEG}"

echo
echo "Stage-1B H1 predefined analysis completed."
echo "Inspect:"
echo "  results/h1_predefined/ViT-B-32/mcm/aggregate_vs_worst.csv"
echo "  results/h1_predefined/ViT-B-32/neglabel/aggregate_vs_worst.csv"
