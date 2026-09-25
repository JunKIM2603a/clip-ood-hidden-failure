#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BACKBONE="${BACKBONE:-ViT-B/32}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
LOG_DIR="$ROOT/results/logs/alternative_controls"
mkdir -p "$LOG_DIR"

if [[ "$BACKBONE" != "ViT-B/32" ]]; then
  echo "ERROR: primary alternative controls are frozen to ViT-B/32." >&2
  exit 2
fi

echo "Stage 3: alternative-explanation controls"
echo "GPU 0 -> iNaturalist proximity"
echo "GPU 1 -> SUN proximity"

PYTHONUNBUFFERED=1 python scripts/analysis/prepare_alternative_control_proximity.py \
  --source inaturalist \
  --backbone "$BACKBONE" \
  --gpu 0 \
  > "$LOG_DIR/inaturalist_proximity.log" 2>&1 &
PID_INAT=$!

PYTHONUNBUFFERED=1 python scripts/analysis/prepare_alternative_control_proximity.py \
  --source sun \
  --backbone "$BACKBONE" \
  --gpu 1 \
  > "$LOG_DIR/sun_proximity.log" 2>&1 &
PID_SUN=$!

set +e
wait "$PID_INAT"
STATUS_INAT=$?
wait "$PID_SUN"
STATUS_SUN=$?
set -e

if [[ "$STATUS_INAT" -ne 0 || "$STATUS_SUN" -ne 0 ]]; then
  echo "Proximity preparation failed." >&2
  echo "iNaturalist: $STATUS_INAT -> $LOG_DIR/inaturalist_proximity.log" >&2
  echo "SUN: $STATUS_SUN -> $LOG_DIR/sun_proximity.log" >&2
  exit 1
fi

echo "Similarity proxies ready. Running fixed-group residual analysis."
PYTHONUNBUFFERED=1 python scripts/analysis/run_alternative_controls.py \
  --backbone "$BACKBONE" \
  --bootstrap "$BOOTSTRAP" \
  | tee "$LOG_DIR/analysis.log"

echo
echo "Primary outputs:"
echo "  results/alternative_controls/ViT-B-32/fixed_h1_worst_groups.csv"
echo "  results/alternative_controls/ViT-B-32/similarity_risk_summary.csv"
echo "  results/alternative_controls/ViT-B-32/README.md"
echo
echo "This stage does not run H3."
