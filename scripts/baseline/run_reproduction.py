#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from baselines.common import (
    DATA_ROOT,
    backbone_slug,
    extract_image_features,
    load_clip,
    ood_metrics,
    verify_reference_repos,
    write_metrics_csv,
    write_score_csv,
)
from baselines.mcm import mcm_scores, prepare_mcm_text
from baselines.neglabel import neglabel_scores, prepare_neglabel_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["mcm", "neglabel"])
    parser.add_argument("--backbone", default="ViT-B/16", choices=["ViT-B/16", "ViT-B/32"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--score-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--ood",
        nargs="+",
        default=["inaturalist", "sun", "places", "dtd"],
        choices=["inaturalist", "sun", "places", "dtd"],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    verify_reference_repos()
    device = torch.device(f"cuda:{args.gpu}")
    model, preprocess = load_clip(args.backbone, device)

    datasets = ["imagenet"] + list(args.ood)
    cache = {}
    for dataset in datasets:
        cache[dataset] = extract_image_features(
            model=model,
            preprocess=preprocess,
            backbone=args.backbone,
            dataset=dataset,
            device=device,
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
            force=False,
        )

    method_meta = {
        "method": args.method,
        "backbone": args.backbone,
        "image_encoder": "OpenAI CLIP",
    }

    if args.method == "mcm":
        text = prepare_mcm_text(model, device)
        batch_size = args.score_batch_size or 2048

        def score_fn(features):
            return mcm_scores(
                features,
                text,
                device=device,
                temperature=1.0,
                batch_size=batch_size,
            )

        method_meta.update({
            "prompt": "a photo of a {label}",
            "temperature": 1.0,
            "reference": "deeplearning-wisc/MCM",
        })
    else:
        pos_text, neg_text, neg_meta = prepare_neglabel_text(
            model,
            device,
            backbone=args.backbone,
        )
        batch_size = args.score_batch_size or 512

        def score_fn(features):
            return neglabel_scores(
                features,
                pos_text,
                neg_text,
                device=device,
                ngroup=100,
                temperature=1.0,
                logit_scale=100.0,
                batch_size=batch_size,
            )

        method_meta.update({
            "reference": "XueJiang16/NegLabel",
            **neg_meta,
        })

    scores = {}
    raw_root = (
        ROOT
        / "results"
        / "raw"
        / "reproduction"
        / backbone_slug(args.backbone)
        / args.method
    )

    for dataset in datasets:
        print(f"[score] {args.method} / {args.backbone} / {dataset}")
        scores[dataset] = score_fn(cache[dataset]["features"])
        write_score_csv(
            raw_root / f"{dataset}.csv",
            dataset,
            cache[dataset]["paths"],
            scores[dataset],
            cache[dataset]["labels"],
        )

    rows = []
    for dataset in args.ood:
        m = ood_metrics(scores["imagenet"], scores[dataset])
        row = {
            "method": args.method,
            "backbone": args.backbone,
            "dataset": dataset,
            **m,
        }
        rows.append(row)
        print(
            f"{dataset:<12} "
            f"FPR95={100*m['fpr95']:.2f} "
            f"AUROC={100*m['auroc']:.2f} "
            f"AUPR_IN={100*m['aupr_in']:.2f} "
            f"AUPR_OUT={100*m['aupr_out']:.2f}"
        )

    mean = {
        key: float(np.mean([r[key] for r in rows]))
        for key in ["fpr95", "auroc", "aupr_in", "aupr_out"]
    }
    rows.append({
        "method": args.method,
        "backbone": args.backbone,
        "dataset": "traditional_four_mean",
        **mean,
    })
    print(
        f"{'MEAN':<12} "
        f"FPR95={100*mean['fpr95']:.2f} "
        f"AUROC={100*mean['auroc']:.2f}"
    )

    table = (
        ROOT
        / "results"
        / "tables"
        / f"reproduction_{backbone_slug(args.backbone)}_{args.method}.csv"
    )
    write_metrics_csv(table, rows)

    meta_path = raw_root / "run_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(method_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[table] {table}")
    print(f"[metadata] {meta_path}")


if __name__ == "__main__":
    main()
