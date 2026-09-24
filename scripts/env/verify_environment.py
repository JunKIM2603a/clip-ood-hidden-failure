#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ROOT / "configs" / "environment" / "versions.env"


def parse_versions() -> dict[str, str]:
    out = {}
    for line in VERSIONS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "MISSING"


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(2)


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not fail when CUDA is unavailable; useful for dataset-only PCs.",
    )
    args = parser.parse_args()

    expected = parse_versions()
    print("=== CLIP-OOD environment smoke test ===")
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())

    if not sys.version.startswith(expected["PYTHON_VERSION"] + "."):
        fail(
            f"Python {sys.version.split()[0]} != expected "
            f"{expected['PYTHON_VERSION']}.x"
        )

    import numpy as np
    import torch
    import torchvision

    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("torch CUDA runtime:", torch.version.cuda)
    print("numpy:", np.__version__)

    if not torch.__version__.startswith(expected["TORCH_VERSION"]):
        fail(f"torch {torch.__version__} != {expected['TORCH_VERSION']}")
    if not torchvision.__version__.startswith(expected["TORCHVISION_VERSION"]):
        fail(
            f"torchvision {torchvision.__version__} != "
            f"{expected['TORCHVISION_VERSION']}"
        )
    if np.__version__ != expected["NUMPY_VERSION"]:
        fail(f"numpy {np.__version__} != {expected['NUMPY_VERSION']}")

    expected_cuda = expected["TORCH_CUDA"].removeprefix("cu")
    expected_cuda = expected_cuda[:-1] + "." + expected_cuda[-1]
    if torch.version.cuda and not torch.version.cuda.startswith(expected_cuda):
        fail(
            f"PyTorch CUDA runtime {torch.version.cuda} does not match "
            f"{expected['TORCH_CUDA']}"
        )

    cuda_ok = torch.cuda.is_available()
    print("CUDA available:", cuda_ok)
    if not cuda_ok and not args.allow_cpu:
        fail("CUDA is unavailable. Check NVIDIA driver / container GPU access.")

    if cuda_ok:
        count = torch.cuda.device_count()
        print("CUDA device count:", count)
        if count < 2:
            print("[WARN] fewer than 2 visible GPUs; pilot can still run serially.")
        for idx in range(count):
            props = torch.cuda.get_device_properties(idx)
            print(
                f"gpu[{idx}]: {props.name}; "
                f"VRAM={props.total_memory / 1024**3:.1f} GiB; "
                f"compute={props.major}.{props.minor}"
            )
            with torch.cuda.device(idx):
                a = torch.randn((512, 512), device=f"cuda:{idx}", dtype=torch.float16)
                b = torch.randn((512, 512), device=f"cuda:{idx}", dtype=torch.float16)
                c = a @ b
                torch.cuda.synchronize(idx)
                if not torch.isfinite(c).all():
                    fail(f"GPU {idx} matmul produced non-finite values")
        print("[OK] CUDA matmul smoke test")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            print("nvidia-smi:")
            for line in result.stdout.strip().splitlines():
                print(" ", line)
    except FileNotFoundError:
        print("[WARN] nvidia-smi not found in PATH")

    critical = [
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
        "gdown",
        "huggingface-hub",
        "datasets",
        "open-clip-torch",
        "sentence-transformers",
    ]
    print("\ncritical packages:")
    missing = []
    for package in critical:
        v = version(package)
        print(f"  {package:<24} {v}")
        if v == "MISSING":
            missing.append(package)
    if missing:
        fail("missing packages: " + ", ".join(missing))

    # Import the modules that OpenOOD eagerly imports. This catches most
    # dependency problems before a long CLIP inference run starts.
    for module in [
        "cv2",
        "imgaug",
        "faiss",
        "libmr",
        "clip",
        "open_clip",
        "openood.preprocessors",
        "openood.postprocessors",
        "openood.networks",
    ]:
        importlib.import_module(module)
        print(f"[OK] import {module}")

    from openood.networks.clip import clip as bundled_clip

    models = bundled_clip.available_models()
    for required in ["ViT-B/32", "ViT-B/16"]:
        if required not in models:
            fail(f"Bundled CLIP does not expose {required}")
    print("[OK] bundled CLIP backbones:", "ViT-B/32, ViT-B/16")

    openood_dir = ROOT / "third_party" / "OpenOOD-VLM"
    if not (openood_dir / ".git").exists():
        fail(f"OpenOOD checkout missing: {openood_dir}")
    head = git_head(openood_dir)
    if head != expected["OPENOOD_COMMIT"]:
        fail(f"OpenOOD HEAD {head} != pinned {expected['OPENOOD_COMMIT']}")
    print("[OK] OpenOOD commit:", head)

    import openood
    openood_path = Path(openood.__file__).resolve()
    if openood_dir.resolve() not in openood_path.parents:
        fail(
            "openood import is not coming from the pinned third_party checkout: "
            + str(openood_path)
        )
    print("[OK] editable OpenOOD path:", openood_path)

    print("\nEnvironment verification PASSED.")


if __name__ == "__main__":
    main()
