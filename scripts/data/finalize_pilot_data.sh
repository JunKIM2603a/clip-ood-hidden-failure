#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

echo "[1/4] Verify all frozen pilot datasets"
python scripts/data/setup_datasets.py --datasets all --verify-only

echo "[2/4] Verify strict ImageNet-1K validation structure"
python scripts/data/verify_imagenet_manifest.py

echo "[3/4] Recheck semantic mapping feasibility without detector scores"
python scripts/data/reconstruct_mos_labels.py

echo "[4/4] Freeze dataset provenance fingerprint"
python scripts/data/dataset_lock.py freeze

echo
echo "Pilot dataset preflight completed."
echo "Review the tracked lock before any detector subgroup results:"
echo "  git diff -- configs/datasets/pilot_data_lock.json"
echo "Then commit it:"
echo "  git add configs/datasets/pilot_data_lock.json"
echo "  git commit -m \"Freeze pilot dataset fingerprint\""
echo "  git push"
