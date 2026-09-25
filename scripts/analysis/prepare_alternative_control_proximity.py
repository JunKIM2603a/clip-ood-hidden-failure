#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.alternative_controls import nearest_cosine_similarity
from baselines.common import backbone_slug, feature_cache_path
from baselines.mcm import official_imagenet_class_names

CFG = ROOT / "configs" / "alternative_controls.yaml"


def load_cfg() -> dict:
    with CFG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_root() -> Path:
    return Path(
        os.environ.get("CLIP_OOD_DATA_ROOT", ROOT / "data")
    ).expanduser().resolve()


def load_features(backbone: str, dataset: str) -> dict[str, np.ndarray]:
    path = feature_cache_path(backbone, dataset)
    if not path.exists():
        raise RuntimeError(
            f"Missing feature cache: {path}\n"
            f"Run feature extraction for {dataset} first."
        )
    data = np.load(path, allow_pickle=False)
    return {
        "features": data["features"],
        "paths": data["paths"],
        "labels": data["labels"],
    }


def semantic_mapping(source: str) -> pd.DataFrame:
    path = (
        data_root()
        / "semantic_labels"
        / f"{source}_image_semantic_labels.csv"
    )
    if not path.exists():
        raise RuntimeError(f"Missing semantic mapping: {path}")
    frame = pd.read_csv(path)
    required = {"relative_path", "leaf_concept"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    if frame["relative_path"].duplicated().any():
        raise RuntimeError(f"Duplicate relative_path in {path}")
    return frame[["relative_path", "leaf_concept"]].copy()


def semantic_similarity(
    leaf_concepts: list[str],
    cfg: dict,
) -> dict[str, float]:
    sem_cfg = cfg["difficulty_proxies"]["semantic_id_proximity"]
    model = SentenceTransformer(
        sem_cfg["encoder"],
        revision=sem_cfg["revision"],
        device="cpu",
    )

    imagenet_names = official_imagenet_class_names()
    id_emb = model.encode(
        imagenet_names,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    leaf_emb = model.encode(
        leaf_concepts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    similarities = leaf_emb @ id_emb.T
    maximum = similarities.max(axis=1)
    return {
        leaf: float(value)
        for leaf, value in zip(leaf_concepts, maximum)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        choices=["inaturalist", "sun"],
    )
    parser.add_argument(
        "--backbone",
        default="ViT-B/32",
        choices=["ViT-B/32"],
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--visual-batch-size",
        type=int,
        default=256,
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for visual proximity computation")

    cfg = load_cfg()
    frozen_backbone = cfg["scope"]["backbone"]
    if args.backbone != frozen_backbone:
        raise RuntimeError(
            f"Alternative-control primary backbone is frozen to {frozen_backbone}"
        )

    id_cache = load_features(args.backbone, "imagenet")
    ood_cache = load_features(args.backbone, args.source)

    print(
        f"[visual] {args.source}: nearest ImageNet-val CLIP image similarity"
    )
    device = torch.device(f"cuda:{args.gpu}")
    visual = nearest_cosine_similarity(
        ood_cache["features"],
        id_cache["features"],
        device,
        batch_size=args.visual_batch_size,
    )

    feature_frame = pd.DataFrame({
        "relative_path": ood_cache["paths"].astype(str),
        "visual_id_similarity": visual,
    })
    semantic = semantic_mapping(args.source)
    merged = feature_frame.merge(
        semantic,
        on="relative_path",
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(feature_frame):
        raise RuntimeError("Feature/semantic merge changed sample count")
    if merged["leaf_concept"].isna().any():
        missing = int(merged["leaf_concept"].isna().sum())
        raise RuntimeError(
            f"{args.source}: {missing} samples lack leaf_concept; "
            "primary control requires complete semantic proximity."
        )

    leaves = sorted(merged["leaf_concept"].astype(str).unique())
    print(
        f"[semantic] {args.source}: {len(leaves)} OOD concepts "
        "against 1000 ImageNet class names"
    )
    leaf_similarity = semantic_similarity(leaves, cfg)
    merged["semantic_id_similarity"] = (
        merged["leaf_concept"].astype(str).map(leaf_similarity)
    )
    if merged["semantic_id_similarity"].isna().any():
        raise RuntimeError("Semantic similarity mapping failed")
    if not np.isfinite(
        merged[
            ["visual_id_similarity", "semantic_id_similarity"]
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Non-finite similarity proxy")

    out_dir = (
        ROOT
        / "results"
        / "raw"
        / "alternative_controls"
        / backbone_slug(args.backbone)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.source}_proximity.csv.gz"
    merged.to_csv(
        out_path,
        index=False,
        compression="gzip",
    )

    metadata = {
        "source": args.source,
        "backbone": args.backbone,
        "n_samples": int(len(merged)),
        "n_leaf_concepts": int(len(leaves)),
        "visual_proxy": cfg["difficulty_proxies"]["visual_id_proximity"],
        "semantic_proxy": cfg["difficulty_proxies"]["semantic_id_proximity"],
        "output": str(out_path.relative_to(ROOT)),
    }
    meta_path = out_dir / f"{args.source}_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[output] {out_path}")
    print(
        "[summary] visual min/mean/max = "
        f"{visual.min():.4f}/{visual.mean():.4f}/{visual.max():.4f}"
    )
    sem_values = merged["semantic_id_similarity"].to_numpy(dtype=float)
    print(
        "[summary] semantic min/mean/max = "
        f"{sem_values.min():.4f}/{sem_values.mean():.4f}/{sem_values.max():.4f}"
    )


if __name__ == "__main__":
    main()
