#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BACKBONE="ViT-B/16"
BOOTSTRAP="${BOOTSTRAP:-5000}"
LOG_DIR="$ROOT/results/logs/b16_h1_replication"
mkdir -p "$LOG_DIR"

missing=0
for method in mcm neglabel; do
  for dataset in imagenet inaturalist sun; do
    path="$ROOT/results/raw/reproduction/ViT-B-16/$method/$dataset.csv"
    if [[ ! -f "$path" ]]; then
      echo "Missing: $path" >&2
      missing=1
    fi
  done
done

if [[ "$missing" -ne 0 ]]; then
  cat >&2 <<'EOF'
ViT-B/16 raw baseline scores are required before H1 replication.

Generate them with the already validated baseline reproduction code, preferably
in parallel on the two GPUs:

  python scripts/baseline/run_reproduction.py     --method mcm --backbone ViT-B/16 --gpu 0 --ood inaturalist sun

  python scripts/baseline/run_reproduction.py     --method neglabel --backbone ViT-B/16 --gpu 1 --ood inaturalist sun

Then rerun this launcher.
EOF
  exit 2
fi

echo "ViT-B/16 H1 replication"
echo "Frozen subgroup definitions and ID95 rules are reused unchanged."
echo "Bootstrap resamples: $BOOTSTRAP"

PYTHONUNBUFFERED=1 python scripts/analysis/run_h1_predefined.py \
  --method mcm \
  --backbone "$BACKBONE" \
  --bootstrap "$BOOTSTRAP" \
  > "$LOG_DIR/mcm_predefined.log" 2>&1 &
PID_A=$!

PYTHONUNBUFFERED=1 python scripts/analysis/run_h1_predefined.py \
  --method neglabel \
  --backbone "$BACKBONE" \
  --bootstrap "$BOOTSTRAP" \
  > "$LOG_DIR/neglabel_predefined.log" 2>&1 &
PID_B=$!

set +e
wait "$PID_A"; STATUS_A=$?
wait "$PID_B"; STATUS_B=$?
set -e
if [[ "$STATUS_A" -ne 0 || "$STATUS_B" -ne 0 ]]; then
  echo "Predefined B/16 H1 analysis failed." >&2
  echo "MCM status=$STATUS_A log=$LOG_DIR/mcm_predefined.log" >&2
  echo "NegLabel status=$STATUS_B log=$LOG_DIR/neglabel_predefined.log" >&2
  exit 1
fi

PYTHONUNBUFFERED=1 python scripts/analysis/run_h1_text_clusters.py \
  --method mcm \
  --backbone "$BACKBONE" \
  --bootstrap "$BOOTSTRAP" \
  > "$LOG_DIR/mcm_minilm.log" 2>&1 &
PID_C=$!

PYTHONUNBUFFERED=1 python scripts/analysis/run_h1_text_clusters.py \
  --method neglabel \
  --backbone "$BACKBONE" \
  --bootstrap "$BOOTSTRAP" \
  > "$LOG_DIR/neglabel_minilm.log" 2>&1 &
PID_D=$!

set +e
wait "$PID_C"; STATUS_C=$?
wait "$PID_D"; STATUS_D=$?
set -e
if [[ "$STATUS_C" -ne 0 || "$STATUS_D" -ne 0 ]]; then
  echo "MiniLM B/16 H1 analysis failed." >&2
  echo "MCM status=$STATUS_C log=$LOG_DIR/mcm_minilm.log" >&2
  echo "NegLabel status=$STATUS_D log=$LOG_DIR/neglabel_minilm.log" >&2
  exit 1
fi

echo "Standard H1 B/16 analyses complete."
echo "Running selection-aware 5000-bootstrap replication analysis."

PYTHONUNBUFFERED=1 python scripts/analysis/run_b16_h1_selection_aware.py \
  --backbone "$BACKBONE" \
  --bootstrap "$BOOTSTRAP" \
  | tee "$LOG_DIR/selection_aware.log"

echo
echo "Primary outputs:"
echo "  results/h1_replication/ViT-B-16/replication_interpretation.json"
echo "  results/h1_replication/ViT-B-16/selection_aware_summary.csv"
