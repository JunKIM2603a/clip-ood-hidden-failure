#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BACKBONE="${BACKBONE:-ViT-B/32}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
LOG_DIR="$ROOT/results/logs/h2_prompt"
mkdir -p "$LOG_DIR"

if [[ "$BACKBONE" != "ViT-B/32" ]]; then
  echo "ERROR: primary H2 is frozen to ViT-B/32." >&2
  echo "Use a separate robustness script/label for other backbones." >&2
  exit 2
fi

echo "H2 primary prompt-sensitivity run"
echo "Backbone: $BACKBONE"
echo "Prompts: frozen 8-template family from configs/pilot.yaml"
echo "GPU 0: MCM"
echo "GPU 1: NegLabel"

PYTHONUNBUFFERED=1 python scripts/analysis/run_h2_prompt_scores.py \
  --method mcm \
  --backbone "$BACKBONE" \
  --gpu 0 \
  --datasets imagenet inaturalist sun \
  > "$LOG_DIR/mcm_scores.log" 2>&1 &
PID_MCM=$!

PYTHONUNBUFFERED=1 python scripts/analysis/run_h2_prompt_scores.py \
  --method neglabel \
  --backbone "$BACKBONE" \
  --gpu 1 \
  --datasets imagenet inaturalist sun \
  > "$LOG_DIR/neglabel_scores.log" 2>&1 &
PID_NEGLABEL=$!

set +e
wait "$PID_MCM"
STATUS_MCM=$?
wait "$PID_NEGLABEL"
STATUS_NEGLABEL=$?
set -e

if [[ "$STATUS_MCM" -ne 0 || "$STATUS_NEGLABEL" -ne 0 ]]; then
  echo "H2 prompt scoring failed." >&2
  echo "MCM status: $STATUS_MCM -> $LOG_DIR/mcm_scores.log" >&2
  echo "NegLabel status: $STATUS_NEGLABEL -> $LOG_DIR/neglabel_scores.log" >&2
  exit 1
fi

echo "Prompt scoring complete. Running frozen H2 statistical analysis."
PYTHONUNBUFFERED=1 python scripts/analysis/run_h2_prompt_analysis.py \
  --backbone "$BACKBONE" \
  --bootstrap "$BOOTSTRAP" \
  | tee "$LOG_DIR/analysis.log"

echo
echo "Primary H2 outputs:"
echo "  results/h2_prompt/ViT-B-32/"
echo "Decision file:"
echo "  results/h2_prompt/ViT-B-32/h2_decision.json"
echo
echo "H3 is intentionally not run by this script."
