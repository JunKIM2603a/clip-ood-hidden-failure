#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BACKBONE="${BACKBONE:-ViT-B/16}"
NUM_WORKERS="${NUM_WORKERS:-8}"

echo "[1/2] Preparing shared CLIP image-feature cache on GPU 0"
CUDA_VISIBLE_DEVICES=0 python scripts/baseline/prepare_features.py   --backbone "${BACKBONE}"   --gpu 0   --num-workers "${NUM_WORKERS}"

echo "[2/2] Running MCM and NegLabel in parallel"
CUDA_VISIBLE_DEVICES=0 python scripts/baseline/run_reproduction.py   --method mcm   --backbone "${BACKBONE}"   --gpu 0   --num-workers "${NUM_WORKERS}" &
PID_MCM=$!

CUDA_VISIBLE_DEVICES=1 python scripts/baseline/run_reproduction.py   --method neglabel   --backbone "${BACKBONE}"   --gpu 0   --num-workers "${NUM_WORKERS}" &
PID_NEGLABEL=$!

wait "${PID_MCM}"
wait "${PID_NEGLABEL}"

echo "Reproduction runs completed."
