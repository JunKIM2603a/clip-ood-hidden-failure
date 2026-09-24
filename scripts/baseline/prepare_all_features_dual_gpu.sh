#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BACKBONE="${BACKBONE:-ViT-B/16}"
NUM_WORKERS="${NUM_WORKERS:-8}"

echo "[preflight] verifying datasets"
python scripts/data/setup_datasets.py --datasets all --verify-only

echo "[preflight] running frozen metric / baseline tests"
pytest -q \
  tests/test_semantic_mapping.py \
  tests/test_baseline_scores.py \
  tests/test_h1_metrics.py

echo "[features] GPU 0 -> ImageNet-1K validation (50k)"
CUDA_VISIBLE_DEVICES=0 python scripts/baseline/prepare_features.py \
  --backbone "${BACKBONE}" \
  --gpu 0 \
  --num-workers "${NUM_WORKERS}" \
  --datasets imagenet &
PID_ID=$!

echo "[features] GPU 1 -> iNaturalist + SUN + Places + DTD"
CUDA_VISIBLE_DEVICES=1 python scripts/baseline/prepare_features.py \
  --backbone "${BACKBONE}" \
  --gpu 0 \
  --num-workers "${NUM_WORKERS}" \
  --datasets inaturalist sun places dtd &
PID_OOD=$!

wait "${PID_ID}"
wait "${PID_OOD}"

echo "[features] all shared CLIP feature caches are ready."
echo "[next] bash scripts/baseline/run_dual_gpu.sh"
