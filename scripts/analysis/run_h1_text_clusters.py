#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.h1_predefined import (
    auc_contributions,
    bootstrap_group_metric_cis,
    build_fixed_id_reference,
    conditional_metrics,
    fpr95_contributions,
    paired_source_bootstrap_gap_ci,
    realized_id_tpr,
)
from baselines.common import ood_metrics

PILOT = ROOT / "configs" / "pilot.yaml"
CLUSTER_CFG = ROOT / "configs" / "subgroups" / "text_clustering.yaml"
MAPPING_DIR = ROOT / "configs" / "subgroups" / "mappings"


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def backbone_slug(backbone):
    return backbone.replace("/", "-")


def score_path(method, backbone, dataset):
    return (
        ROOT
        / "results"
        / "raw"
        / "reproduction"
        / backbone_slug(backbone)
        / method
        / (dataset + ".csv")
    )


def data_root():
    return Path(
        os.environ.get("CLIP_OOD_DATA_ROOT", ROOT / "data")
    ).expanduser().resolve()


def semantic_path(dataset):
    return data_root() / "semantic_labels" / (
        dataset + "_image_semantic_labels.csv"
    )


def cluster_path(dataset):
    return MAPPING_DIR / (
        "text_clusters_{}_minilm.csv".format(dataset)
    )


def read_scores(path):
    if not path.exists():
        raise RuntimeError("Missing score file: {}".format(path))
    frame = pd.read_csv(path)
    required = {"relative_path", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            "{} missing columns: {}".format(path, sorted(missing))
        )
    if frame["relative_path"].duplicated().any():
        raise RuntimeError("Duplicate score paths: {}".format(path))
    return frame


def primary_sources():
    cfg = load_yaml(PILOT)
    return [x["name"] for x in cfg["data"]["primary_ood_sources"]]


def rules():
    return load_yaml(CLUSTER_CFG)["eligibility"]


def load_leaf_clusters(dataset):
    path = cluster_path(dataset)
    if not path.exists():
        raise RuntimeError(
            "Missing frozen text-cluster mapping: {}".format(path)
        )
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {
        row["leaf_concept"].strip(): row["text_cluster"].strip()
        for row in rows
    }
    if not out:
        raise RuntimeError("Empty text-cluster mapping: {}".format(path))
    return out


def prepare_join(source, score_frame):
    semantic = pd.read_csv(semantic_path(source))
    required = {"relative_path", "leaf_concept"}
    missing = required - set(semantic.columns)
    if missing:
        raise RuntimeError(
            "{} semantic mapping missing {}".format(
                source, sorted(missing)
            )
        )

    leaf_to_cluster = load_leaf_clusters(source)
    semantic = semantic[["relative_path", "leaf_concept"]].copy()
    semantic["text_cluster"] = semantic["leaf_concept"].map(
        leaf_to_cluster
    )
    merged = score_frame.merge(
        semantic,
        on="relative_path",
        how="left",
        validate="one_to_one",
    )
    return merged


def plot_table(table, aggregate, source, out_dir):
    eligible = table[table["eligible_primary"]].copy()

    ordered = eligible.sort_values("fpr95")
    fig, ax = plt.subplots(
        figsize=(8, max(4, 0.38 * len(ordered)))
    )
    ax.barh(ordered["group"], 100 * ordered["fpr95"])
    ax.axvline(
        100 * aggregate["fpr95_fixed_id95"],
        linestyle="--",
    )
    ax.set_xlabel("FPR95 (%)")
    ax.set_ylabel("MiniLM text cluster")
    ax.set_title("{}: text-cluster FPR95".format(source))
    fig.tight_layout()
    fig.savefig(
        out_dir / (source + "_text_cluster_fpr95.png"),
        dpi=160,
    )
    plt.close(fig)

    ordered = eligible.sort_values("auroc")
    fig, ax = plt.subplots(
        figsize=(8, max(4, 0.38 * len(ordered)))
    )
    ax.barh(ordered["group"], 100 * ordered["auroc"])
    ax.axvline(
        100 * aggregate["auroc_fixed_id"],
        linestyle="--",
    )
    ax.set_xlabel("AUROC (%)")
    ax.set_ylabel("MiniLM text cluster")
    ax.set_title("{}: text-cluster AUROC".format(source))
    fig.tight_layout()
    fig.savefig(
        out_dir / (source + "_text_cluster_auroc.png"),
        dpi=160,
    )
    plt.close(fig)


def evaluate_source(
    source,
    id_scores,
    reference,
    score_frame,
    bootstrap,
    confidence,
    seed,
    out_dir,
):
    merged = prepare_join(source, score_frame)
    cfg = rules()

    mapped = merged["text_cluster"].notna()
    coverage = float(mapped.mean())
    if coverage < float(cfg["minimum_mapping_coverage"]):
        raise RuntimeError(
            "{} text-cluster coverage {:.4f} < frozen {:.4f}".format(
                source,
                coverage,
                float(cfg["minimum_mapping_coverage"]),
            )
        )

    source_scores = merged["score"].to_numpy(dtype=float)
    fixed = conditional_metrics(reference, source_scores)
    official = ood_metrics(id_scores, source_scores)
    source_auc = auc_contributions(reference, source_scores)
    source_fpr = fpr95_contributions(reference, source_scores)

    rows = []
    groups = sorted(
        x for x in merged["text_cluster"].dropna().unique()
    )
    for idx, group in enumerate(groups):
        mask = (merged["text_cluster"] == group).to_numpy()
        group_scores = merged.loc[mask, "score"].to_numpy(dtype=float)
        n_images = int(mask.sum())
        n_leafs = int(
            merged.loc[mask, "leaf_concept"]
            .dropna()
            .astype(str)
            .nunique()
        )
        eligible = (
            n_images >= int(cfg["minimum_images_per_group"])
            and n_leafs
            >= int(cfg["minimum_leaf_concepts_per_group"])
        )
        row = {
            "source": source,
            "group": group,
            "n_images": n_images,
            "n_leaf_concepts": n_leafs,
            "eligible_primary": eligible,
        }

        if eligible:
            m = conditional_metrics(reference, group_scores)
            cis = bootstrap_group_metric_cis(
                reference,
                group_scores,
                n_resamples=bootstrap,
                confidence=confidence,
                seed=seed + idx * 1000,
            )
            fpr_gap_ci = paired_source_bootstrap_gap_ci(
                source_fpr,
                mask,
                gap_direction="group_minus_aggregate",
                n_resamples=bootstrap,
                confidence=confidence,
                seed=seed + idx * 1000 + 101,
            )
            auc_gap_ci = paired_source_bootstrap_gap_ci(
                source_auc,
                mask,
                gap_direction="aggregate_minus_group",
                n_resamples=bootstrap,
                confidence=confidence,
                seed=seed + idx * 1000 + 202,
            )
            row.update({
                "fpr95": m["fpr95"],
                "fpr95_ci_low": cis["fpr95"][0],
                "fpr95_ci_high": cis["fpr95"][1],
                "auroc": m["auroc"],
                "auroc_ci_low": cis["auroc"][0],
                "auroc_ci_high": cis["auroc"][1],
                "fpr95_gap_vs_source": m["fpr95"] - fixed["fpr95"],
                "fpr95_gap_ci_low": fpr_gap_ci[0],
                "fpr95_gap_ci_high": fpr_gap_ci[1],
                "auroc_gap_vs_source": fixed["auroc"] - m["auroc"],
                "auroc_gap_ci_low": auc_gap_ci[0],
                "auroc_gap_ci_high": auc_gap_ci[1],
            })
        else:
            for key in [
                "fpr95", "fpr95_ci_low", "fpr95_ci_high",
                "auroc", "auroc_ci_low", "auroc_ci_high",
                "fpr95_gap_vs_source", "fpr95_gap_ci_low",
                "fpr95_gap_ci_high", "auroc_gap_vs_source",
                "auroc_gap_ci_low", "auroc_gap_ci_high",
            ]:
                row[key] = np.nan
        rows.append(row)

    table = pd.DataFrame(rows)
    eligible_table = table[table["eligible_primary"]].copy()
    minimum = int(cfg["minimum_eligible_groups_per_source"])
    if len(eligible_table) < minimum:
        raise RuntimeError(
            "{} text clustering has {} eligible groups < {}".format(
                source, len(eligible_table), minimum
            )
        )

    worst_fpr = eligible_table.loc[
        eligible_table["fpr95"].idxmax()
    ]
    worst_auc = eligible_table.loc[
        eligible_table["auroc"].idxmin()
    ]

    aggregate = {
        "source": source,
        "n_ood": len(merged),
        "mapping_coverage": coverage,
        "id95_threshold": reference.threshold_95,
        "realized_id_tpr": realized_id_tpr(
            id_scores, reference.threshold_95
        ),
        "fpr95_fixed_id95": fixed["fpr95"],
        "auroc_fixed_id": fixed["auroc"],
        "fpr95_official_pairwise": official["fpr95"],
        "auroc_official_pairwise": official["auroc"],
        "eligible_group_count": int(len(eligible_table)),
        "worst_fpr95_group": str(worst_fpr["group"]),
        "worst_fpr95": float(worst_fpr["fpr95"]),
        "worst_fpr95_gap": float(
            worst_fpr["fpr95_gap_vs_source"]
        ),
        "worst_auroc_group": str(worst_auc["group"]),
        "worst_auroc": float(worst_auc["auroc"]),
        "worst_auroc_gap": float(
            worst_auc["auroc_gap_vs_source"]
        ),
    }

    table.to_csv(
        out_dir / (source + "_text_clusters.csv"),
        index=False,
    )
    plot_table(table, aggregate, source, out_dir)
    return aggregate


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
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sources = primary_sources()
    id_frame = read_scores(
        score_path(args.method, args.backbone, "imagenet")
    )
    id_scores = id_frame["score"].to_numpy(dtype=float)
    reference = build_fixed_id_reference(id_scores)

    out_dir = (
        ROOT
        / "results"
        / "h1_text_clusters"
        / backbone_slug(args.backbone)
        / args.method
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregates = []
    for source in sources:
        frame = read_scores(
            score_path(args.method, args.backbone, source)
        )
        aggregate = evaluate_source(
            source,
            id_scores,
            reference,
            frame,
            args.bootstrap,
            args.confidence,
            args.seed,
            out_dir,
        )
        aggregates.append(aggregate)
        print(
            "{}: aggregate FPR95={:.2f}% worst={:.2f}% ({}) | "
            "aggregate AUROC={:.2f}% worst={:.2f}% ({})".format(
                source,
                100 * aggregate["fpr95_fixed_id95"],
                100 * aggregate["worst_fpr95"],
                aggregate["worst_fpr95_group"],
                100 * aggregate["auroc_fixed_id"],
                100 * aggregate["worst_auroc"],
                aggregate["worst_auroc_group"],
            )
        )

    summary = pd.DataFrame(aggregates)
    summary.to_csv(
        out_dir / "aggregate_vs_worst.csv",
        index=False,
    )
    metadata = {
        "method": args.method,
        "backbone": args.backbone,
        "grouping": "MiniLM KMeans text clusters",
        "bootstrap_resamples": args.bootstrap,
        "confidence": args.confidence,
        "seed": args.seed,
        "id95_threshold": reference.threshold_95,
        "id_reference_size": int(len(id_scores)),
        "score_orientation": "larger_is_more_ID_like",
    }
    (out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
