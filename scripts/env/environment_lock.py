#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "configs" / "environment" / "pilot_environment_lock.json"
OPENOOD = ROOT / "third_party" / "OpenOOD-VLM"
MCM = ROOT / "third_party" / "MCM"
NEGLABEL = ROOT / "third_party" / "NegLabel"

PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "PyYAML",
    "json5",
    "tqdm",
    "Pillow",
    "opencv-python-headless",
    "scikit-image",
    "imageio",
    "Shapely",
    "imgaug",
    "faiss-cpu",
    "libmr",
    "diffdist",
    "ftfy",
    "regex",
    "ipdb",
    "gdown",
    "huggingface-hub",
    "datasets",
    "pytest",
    "clip",
    "open-clip-torch",
    "sentence-transformers",
    "transformers",
    "tokenizers",
    "openood-vlm",
]


def package_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "MISSING"


def direct_git_commit(name: str) -> str | None:
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        return None
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data.get("vcs_info", {}).get("commit_id")


def git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def snapshot() -> dict:
    import torch

    packages = {name: package_version(name) for name in PACKAGES}
    canonical = json.dumps(packages, sort_keys=True).encode("utf-8")
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch_cuda_runtime": torch.version.cuda,
        "packages": packages,
        "packages_sha256": hashlib.sha256(canonical).hexdigest(),
        "openood_commit": git_head(OPENOOD),
        "mcm_commit": git_head(MCM),
        "neglabel_commit": git_head(NEGLABEL),
        "openai_clip_commit": direct_git_commit("clip"),
    }


def freeze() -> None:
    payload = {
        "status": "frozen",
        "schema_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "environment": snapshot(),
    }
    LOCK.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {LOCK}")
    print("Review and commit this small file to Git.")


def check() -> None:
    expected = json.loads(LOCK.read_text(encoding="utf-8"))
    if expected.get("status") != "frozen":
        raise SystemExit(
            "Environment lock is not frozen. First run: "
            "python scripts/env/environment_lock.py freeze"
        )
    actual = snapshot()
    wanted = expected["environment"]
    if wanted == actual:
        print("Environment matches the committed pilot lock.")
        return

    print("Environment mismatch.")
    keys = sorted(set(wanted) | set(actual))
    for key in keys:
        if wanted.get(key) != actual.get(key):
            print(f"\n[{key}]")
            print(" expected:", json.dumps(wanted.get(key), sort_keys=True))
            print(" actual:  ", json.dumps(actual.get(key), sort_keys=True))
    raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "check"])
    args = parser.parse_args()
    freeze() if args.mode == "freeze" else check()


if __name__ == "__main__":
    main()
