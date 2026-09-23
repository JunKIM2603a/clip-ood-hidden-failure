#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from baselines.common import extract_image_features, load_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="ViT-B/16", choices=["ViT-B/16", "ViT-B/32"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["imagenet", "inaturalist", "sun", "places", "dtd"],
        choices=["imagenet", "inaturalist", "sun", "places", "dtd"],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pilot feature extraction")
    device = torch.device(f"cuda:{args.gpu}")
    model, preprocess = load_clip(args.backbone, device)

    for dataset in args.datasets:
        result = extract_image_features(
            model=model,
            preprocess=preprocess,
            backbone=args.backbone,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force=args.force,
        )
        print(
            f"[feature-cache] {dataset}: "
            f"{result['features'].shape} -> {result['cache_path']}"
        )


if __name__ == "__main__":
    main()
