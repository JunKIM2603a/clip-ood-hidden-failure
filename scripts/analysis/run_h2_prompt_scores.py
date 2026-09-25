#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from baselines.common import (
    backbone_slug,
    encode_texts,
    feature_cache_path,
    load_clip,
    verify_reference_repos,
)
from baselines.mcm import mcm_scores, official_imagenet_class_names
from baselines.neglabel import (
    neglabel_prompt_scores,
    official_positive_class_names,
    prepare_neglabel_text,
)

PILOT = ROOT / "configs" / "pilot.yaml"


def load_pilot() -> dict:
    with PILOT.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_feature_cache(
    backbone: str,
    dataset: str,
) -> dict[str, np.ndarray]:
    path = feature_cache_path(backbone, dataset)
    if not path.exists():
        raise RuntimeError(
            f"Missing feature cache: {path}\n"
            f"Run: python scripts/baseline/prepare_features.py "
            f"--backbone {backbone} --datasets {dataset}"
        )
    data = np.load(path, allow_pickle=False)
    required = {"features", "labels", "paths"}
    missing = required - set(data.files)
    if missing:
        raise RuntimeError(
            f"{path} missing arrays: {sorted(missing)}"
        )
    return {
        "features": data["features"],
        "labels": data["labels"],
        "paths": data["paths"],
    }


def render_template(template: str, label: str) -> str:
    try:
        return template.format(label=label)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"Invalid H2 prompt template: {template!r}"
        ) from exc


def encode_prompt_family(
    model,
    class_names: list[str],
    templates: list[str],
    device: torch.device,
) -> list[torch.Tensor]:
    encoded = []
    for idx, template in enumerate(templates):
        prompts = [
            render_template(template, name)
            for name in class_names
        ]
        print(f"[text] template {idx:02d}: {template}")
        encoded.append(
            encode_texts(model, prompts, device)
        )
    return encoded


def mcm_prompt_matrix(
    image_features: np.ndarray,
    text_by_prompt: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    cols = [
        mcm_scores(
            image_features,
            text,
            device=device,
            temperature=1.0,
            batch_size=batch_size,
        )
        for text in text_by_prompt
    ]
    return np.column_stack(cols).astype(
        np.float32,
        copy=False,
    )


def write_prompt_scores(
    path: Path,
    dataset: str,
    paths: np.ndarray,
    labels: np.ndarray,
    score_matrix: np.ndarray,
) -> None:
    if score_matrix.ndim != 2:
        raise ValueError(
            "score_matrix must be [num_samples, num_prompts]"
        )
    if not (
        len(paths) == len(labels) == score_matrix.shape[0]
    ):
        raise ValueError("Prompt score output length mismatch")
    if not np.isfinite(score_matrix).all():
        raise ValueError("Non-finite prompt scores")

    columns = {
        "dataset": np.repeat(dataset, len(paths)),
        "relative_path": paths.astype(str),
        "id_label": labels.astype(np.int64),
    }
    for idx in range(score_matrix.shape[1]):
        columns[f"prompt_{idx:02d}"] = score_matrix[:, idx]

    frame = pd.DataFrame(columns)
    frame["mu"] = score_matrix.mean(
        axis=1,
        dtype=np.float64,
    )
    frame["sigma"] = score_matrix.std(
        axis=1,
        ddof=0,
        dtype=np.float64,
    )
    frame["q10"] = np.quantile(
        score_matrix,
        0.10,
        axis=1,
    )
    frame["q25"] = np.quantile(
        score_matrix,
        0.25,
        axis=1,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression="gzip",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["mcm", "neglabel"],
    )
    parser.add_argument(
        "--backbone",
        default="ViT-B/32",
        choices=["ViT-B/16", "ViT-B/32"],
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["imagenet", "inaturalist", "sun"],
        choices=["imagenet", "inaturalist", "sun"],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    verify_reference_repos()

    cfg = load_pilot()
    prompt_cfg = cfg["prompts"]
    templates = list(prompt_cfg["templates"])
    declared = int(prompt_cfg["num_templates"])
    if declared != 8 or len(templates) != declared:
        raise RuntimeError(
            "Primary H2 is frozen to the committed 8-template "
            "pilot family. "
            f"Found declared={declared}, actual={len(templates)}."
        )

    device = torch.device(f"cuda:{args.gpu}")
    model, _ = load_clip(args.backbone, device)

    metadata = {
        "stage": "H2_prompt_sensitivity",
        "backbone": args.backbone,
        "method": args.method,
        "prompt_count": len(templates),
        "templates": templates,
        "prompt_family_source": "configs/pilot.yaml",
        "std_ddof": 0,
        "lower_quantiles": [0.10, 0.25],
        "score_orientation": "larger_is_more_ID_like",
        "datasets": list(args.datasets),
    }

    if args.method == "mcm":
        names = official_imagenet_class_names()
        text_by_prompt = encode_prompt_family(
            model,
            names,
            templates,
            device,
        )
        score_batch_size = args.score_batch_size or 2048

        def score_fn(
            features: np.ndarray,
        ) -> np.ndarray:
            return mcm_prompt_matrix(
                features,
                text_by_prompt,
                device,
                score_batch_size,
            )

        metadata["prompt_variation"] = (
            "ID_class_prompt_template"
        )
        metadata["method_specific_frozen_components"] = []
    else:
        names = official_positive_class_names()
        text_by_prompt = encode_prompt_family(
            model,
            names,
            templates,
            device,
        )
        _, negative_text, neg_meta = prepare_neglabel_text(
            model,
            device,
            backbone=args.backbone,
        )
        score_batch_size = args.score_batch_size or 512

        def score_fn(
            features: np.ndarray,
        ) -> np.ndarray:
            return neglabel_prompt_scores(
                features,
                text_by_prompt,
                negative_text,
                device=device,
                ngroup=int(neg_meta["ngroup"]),
                temperature=float(neg_meta["temperature"]),
                logit_scale=float(neg_meta["logit_scale"]),
                batch_size=score_batch_size,
            )

        metadata.update({
            "prompt_variation":
                "positive_ID_class_prompt_template_only",
            "neglabel_negative_policy": (
                "freeze H1 negative-label prompt embeddings, "
                "identities, deterministic grouping, and mining "
                "result across H2 templates"
            ),
            "neglabel_negative_prompt_file_sha256":
                neg_meta["negative_prompt_file_sha256"],
            "neglabel_negative_prompt_count": int(
                neg_meta["negative_prompt_count"]
            ),
            "neglabel_negative_source":
                neg_meta["negative_source"],
        })

    out_root = (
        ROOT
        / "results"
        / "raw"
        / "h2_prompt"
        / backbone_slug(args.backbone)
        / args.method
    )
    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dataset in args.datasets:
        cache = load_feature_cache(
            args.backbone,
            dataset,
        )
        print(
            f"\n=== H2 prompt scores: {args.method} / "
            f"{args.backbone} / {dataset} ==="
        )
        score_matrix = score_fn(cache["features"])
        out_path = out_root / f"{dataset}.csv.gz"
        write_prompt_scores(
            out_path,
            dataset,
            cache["paths"],
            cache["labels"],
            score_matrix,
        )
        print(f"[prompt-scores] {out_path}")

    meta_path = out_root / "score_metadata.json"
    meta_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"[metadata] {meta_path}")


if __name__ == "__main__":
    main()
