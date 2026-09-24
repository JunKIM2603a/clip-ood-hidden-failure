#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from baselines.common import load_clip, verify_reference_repos
from baselines.neglabel_mining import (
    compare_with_official_b16,
    load_mined_negative_prompts,
    mine_negative_prompts,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        required=True,
        choices=["ViT-B/16", "ViT-B/32"],
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--compare-official-b16", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NegLabel mining")

    verify_reference_repos()
    device = torch.device(f"cuda:{args.gpu}")
    model, _ = load_clip(args.backbone, device)

    metadata = mine_negative_prompts(
        model,
        args.backbone,
        device,
        quantile=0.95,
        select_fraction=0.15,
        embedding_batch_size=args.batch_size,
    )
    print(json.dumps(metadata, indent=2))

    if args.compare_official_b16:
        if args.backbone != "ViT-B/16":
            raise RuntimeError("--compare-official-b16 requires ViT-B/16")
        prompts, _ = load_mined_negative_prompts(args.backbone)
        comparison = compare_with_official_b16(prompts)
        print("\nB/16 dynamic mining vs official provided selected list:")
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
