#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BACKBONE="${BACKBONE:-ViT-B/16}"
NUM_WORKERS="${NUM_WORKERS:-8}"

echo "Preparing OOD feature caches while ImageNet gated access is pending."
echo "GPU 0: iNaturalist + SUN"
echo "GPU 1: Places + DTD"

CUDA_VISIBLE_DEVICES=0 python scripts/baseline/prepare_features.py \
  --backbone "${BACKBONE}" \
  --gpu 0 \
  --num-workers "${NUM_WORKERS}" \
  --datasets inaturalist sun &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/baseline/prepare_features.py \
  --backbone "${BACKBONE}" \
  --gpu 0 \
  --num-workers "${NUM_WORKERS}" \
  --datasets places dtd &
PID1=$!

wait "${PID0}"
wait "${PID1}"

echo "OOD feature caches completed."
echo "After ImageNet access is granted, run:"
echo "python scripts/baseline/prepare_features.py --backbone ${BACKBONE} --gpu 0 --datasets imagenet"
