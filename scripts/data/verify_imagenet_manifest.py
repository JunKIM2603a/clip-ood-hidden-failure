#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get("CLIP_OOD_DATA_ROOT", ROOT / "data")
).expanduser().resolve()

MANIFEST = DATA_ROOT / "manifests" / "imagenet_val.csv"
LABELS = DATA_ROOT / "images_largescale" / "imagenet_1k" / "labels.json"
STATE = DATA_ROOT / "state" / "imagenet.json"


def fail(message: str) -> None:
    raise SystemExit("[FAIL] " + message)


def main() -> int:
    if not MANIFEST.exists():
        fail("missing ImageNet manifest: {}".format(MANIFEST))
    if not LABELS.exists():
        fail("missing ImageNet labels.json: {}".format(LABELS))
    if not STATE.exists():
        fail("missing ImageNet state file: {}".format(STATE))

    with MANIFEST.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 50000:
        fail("manifest rows {} != 50000".format(len(rows)))

    labels = [int(row["label"]) for row in rows]
    counts = Counter(labels)
    expected_labels = set(range(1000))
    actual_labels = set(counts)

    if actual_labels != expected_labels:
        missing = sorted(expected_labels - actual_labels)
        extra = sorted(actual_labels - expected_labels)
        fail(
            "label set mismatch; missing={} extra={}".format(
                missing[:20], extra[:20]
            )
        )

    bad_counts = {
        label: count for label, count in counts.items() if count != 50
    }
    if bad_counts:
        fail(
            "ImageNet validation should contain 50 images/class; examples: {}".format(
                list(sorted(bad_counts.items()))[:20]
            )
        )

    relative_paths = [row["relative_path"] for row in rows]
    if len(relative_paths) != len(set(relative_paths)):
        fail("manifest contains duplicate image paths")

    missing_files = []
    for rel in relative_paths:
        if not (DATA_ROOT / rel).is_file():
            missing_files.append(rel)
            if len(missing_files) >= 20:
                break
    if missing_files:
        fail("missing image files: {}".format(missing_files))

    names = json.loads(LABELS.read_text(encoding="utf-8"))
    if len(names) != 1000:
        fail("labels.json entries {} != 1000".format(len(names)))

    state = json.loads(STATE.read_text(encoding="utf-8"))
    if int(state.get("images", -1)) != 50000:
        fail("state images != 50000")
    if int(state.get("classes", -1)) != 1000:
        fail("state classes != 1000")

    print("ImageNet manifest verification PASSED")
    print("  rows: 50000")
    print("  labels: 0..999")
    print("  images/class: 50")
    print("  labels.json: 1000 entries")
    print("  source_kind:", state.get("source_kind"))
    if state.get("hf_revision"):
        print("  hf_revision:", state["hf_revision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
