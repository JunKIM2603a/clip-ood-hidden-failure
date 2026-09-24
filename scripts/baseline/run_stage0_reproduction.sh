#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BACKBONE="${BACKBONE:-ViT-B/16}"

echo "[gate] environment"
python scripts/env/verify_environment.py

echo "[gate] dataset lock must be frozen"
python - <<'PY'
import json
from pathlib import Path
p = Path("configs/datasets/pilot_data_lock.json")
d = json.loads(p.read_text())
if d.get("status") != "frozen":
    raise SystemExit("[FAIL] pilot_data_lock.json is not frozen")
print("[OK] dataset lock frozen")
PY

echo "[gate] text-cluster mappings must exist"
for f in \
  configs/subgroups/text_clustering_manifest.json \
  configs/subgroups/mappings/text_clusters_inaturalist_minilm.csv \
  configs/subgroups/mappings/text_clusters_sun_minilm.csv \
  configs/subgroups/mappings/text_clusters_places_minilm.csv
do
  if [[ ! -f "${f}" ]]; then
    echo "[FAIL] missing ${f}" >&2
    exit 2
  fi
  git ls-files --error-unmatch "${f}" >/dev/null 2>&1 || {
    echo "[FAIL] ${f} is not tracked by Git" >&2
    exit 2
  }
done

echo "[gate] frozen definitions must have no uncommitted changes"
if ! git diff --quiet -- \
  configs/datasets/pilot_data_lock.json \
  configs/subgroups/text_clustering.yaml \
  configs/subgroups/text_clustering_manifest.json \
  configs/subgroups/mappings/text_clusters_inaturalist_minilm.csv \
  configs/subgroups/mappings/text_clusters_sun_minilm.csv \
  configs/subgroups/mappings/text_clusters_places_minilm.csv
then
  echo "[FAIL] frozen data/subgroup definitions have unstaged changes" >&2
  exit 2
fi
if ! git diff --cached --quiet -- \
  configs/datasets/pilot_data_lock.json \
  configs/subgroups/text_clustering.yaml \
  configs/subgroups/text_clustering_manifest.json \
  configs/subgroups/mappings/text_clusters_inaturalist_minilm.csv \
  configs/subgroups/mappings/text_clusters_sun_minilm.csv \
  configs/subgroups/mappings/text_clusters_places_minilm.csv
then
  echo "[FAIL] frozen data/subgroup definitions are staged but not committed" >&2
  echo "Commit them before running detector reproduction." >&2
  exit 2
fi

echo "[stage0] prepare shared image features with two GPUs"
BACKBONE="${BACKBONE}" bash scripts/baseline/prepare_all_features_dual_gpu.sh

echo "[stage0] run MCM and NegLabel in parallel"
BACKBONE="${BACKBONE}" bash scripts/baseline/run_dual_gpu.sh

echo
echo "Stage-0 aggregate reproduction completed."
echo "Inspect:"
echo "  results/tables/reproduction_ViT-B-16_mcm.csv"
echo "  results/tables/reproduction_ViT-B-16_neglabel.csv"
