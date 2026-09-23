#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "configs" / "datasets" / "pilot_data_lock.json"
DATASETS = ["imagenet", "inaturalist", "sun", "places", "dtd"]

def data_root() -> Path:
    return Path(os.environ.get("CLIP_OOD_DATA_ROOT", REPO_ROOT / "data")).expanduser().resolve()

def reduced(state: dict) -> dict:
    keys = [
        "dataset", "images", "expected_images", "classes", "expected_classes",
        "leaf_concepts", "expected_leaf_concepts", "source_kind",
        "source_archive_sha256", "manifest_sha256", "hf_repo", "hf_revision",
        "official_val_archive_md5", "official_devkit_archive_md5",
    ]
    return {k: state[k] for k in keys if k in state}

def states() -> dict:
    root = data_root() / "state"
    out, missing = {}, []
    for name in DATASETS:
        path = root / f"{name}.json"
        if not path.exists():
            missing.append(str(path))
        else:
            out[name] = reduced(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        raise SystemExit("Missing dataset state file(s):\n- " + "\n- ".join(missing))
    return out

def freeze() -> None:
    payload = {
        "status": "frozen",
        "schema_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "datasets": states(),
    }
    LOCK_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {LOCK_PATH}")
    print("Review it, then commit the small lock file to Git.")

def check() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "frozen":
        raise SystemExit("Dataset lock is not frozen yet. Run: python scripts/data/dataset_lock.py freeze")
    current = states()
    failed = False
    for name in DATASETS:
        if lock["datasets"].get(name) == current.get(name):
            print(f"[OK]   {name}")
        else:
            failed = True
            print(f"[FAIL] {name}")
            print(" expected:", json.dumps(lock["datasets"].get(name), sort_keys=True))
            print(" actual:  ", json.dumps(current.get(name), sort_keys=True))
    if failed:
        raise SystemExit(2)
    print("All datasets match the committed pilot fingerprint.")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "check"])
    args = parser.parse_args()
    freeze() if args.mode == "freeze" else check()

if __name__ == "__main__":
    main()
