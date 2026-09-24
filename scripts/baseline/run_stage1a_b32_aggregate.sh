#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BACKBONE="ViT-B/32"
NUM_WORKERS="${NUM_WORKERS:-8}"

echo "[gate] environment and score formulas"
python scripts/env/verify_environment.py
pytest -q tests/test_baseline_scores.py tests/test_h1_metrics.py

echo "[gate] frozen data/subgroup definitions"
python scripts/data/dataset_lock.py check
for f in \
  configs/subgroups/text_clustering_manifest.json \
  configs/subgroups/mappings/text_clusters_inaturalist_minilm.csv \
  configs/subgroups/mappings/text_clusters_sun_minilm.csv
do
  if [[ ! -f "${f}" ]]; then
    echo "[FAIL] missing frozen subgroup artifact: ${f}" >&2
    exit 2
  fi
  git ls-files --error-unmatch "${f}" >/dev/null 2>&1 || {
    echo "[FAIL] subgroup artifact is not committed: ${f}" >&2
    exit 2
  }
done

echo "[1/4] GPU0: ImageNet B/32 features"
CUDA_VISIBLE_DEVICES=0 python scripts/baseline/prepare_features.py \
  --backbone "${BACKBONE}" \
  --gpu 0 \
  --num-workers "${NUM_WORKERS}" \
  --datasets imagenet &
PID_ID=$!

echo "[2/4] GPU1: backbone-specific NegLabel mining"
CUDA_VISIBLE_DEVICES=1 python scripts/baseline/mine_neglabel.py \
  --backbone "${BACKBONE}" \
  --gpu 0 &
PID_MINE=$!
wait "${PID_MINE}"

echo "[3/4] GPU1: OOD B/32 features"
CUDA_VISIBLE_DEVICES=1 python scripts/baseline/prepare_features.py \
  --backbone "${BACKBONE}" \
  --gpu 0 \
  --num-workers "${NUM_WORKERS}" \
  --datasets inaturalist sun places dtd &
PID_OOD=$!

wait "${PID_ID}"
wait "${PID_OOD}"

echo "[4/4] B/32 aggregate MCM + NegLabel in parallel"
CUDA_VISIBLE_DEVICES=0 python scripts/baseline/run_reproduction.py \
  --method mcm \
  --backbone "${BACKBONE}" \
  --gpu 0 \
  --num-workers "${NUM_WORKERS}" &
PID_MCM=$!

CUDA_VISIBLE_DEVICES=1 python scripts/baseline/run_reproduction.py \
  --method neglabel \
  --backbone "${BACKBONE}" \
  --gpu 0 \
  --num-workers "${NUM_WORKERS}" &
PID_NEG=$!

wait "${PID_MCM}"
wait "${PID_NEG}"

echo
echo "Stage-1A B/32 aggregate sanity run completed."
echo "Inspect BEFORE opening subgroup results:"
echo "  cat results/tables/reproduction_ViT-B-32_mcm.csv"
echo "  cat results/tables/reproduction_ViT-B-32_neglabel.csv"
echo
echo "Reference means (sanity anchors, not post-hoc tuning targets):"
echo "  MCM B/32 follow-up reproductions: AUROC ~89.82-89.96, FPR95 ~45.75-49.96"
echo "  NegLabel B/32: AUROC  93.67, FPR95  27.92"
