#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BACKBONE="ViT-B/32"

for f in \
  configs/subgroups/text_clustering_manifest.json \
  configs/subgroups/mappings/text_clusters_inaturalist_minilm.csv \
  configs/subgroups/mappings/text_clusters_sun_minilm.csv
do
  git ls-files --error-unmatch "${f}" >/dev/null 2>&1 || {
    echo "[FAIL] text-cluster artifact not committed: ${f}" >&2
    exit 2
  }
done

echo "[H1 text clusters] MCM"
python scripts/analysis/run_h1_text_clusters.py \
  --method mcm \
  --backbone "${BACKBONE}" &
PID_MCM=$!

echo "[H1 text clusters] NegLabel"
python scripts/analysis/run_h1_text_clusters.py \
  --method neglabel \
  --backbone "${BACKBONE}" &
PID_NEG=$!

wait "${PID_MCM}"
wait "${PID_NEG}"

echo
echo "Stage-1C text-cluster robustness analysis completed."
echo "Inspect:"
echo "  results/h1_text_clusters/ViT-B-32/mcm/aggregate_vs_worst.csv"
echo "  results/h1_text_clusters/ViT-B-32/neglabel/aggregate_vs_worst.csv"
