#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
SUBGROUP = ROOT / "configs" / "subgroups" / "predefined.yaml"


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


def semantic_path(dataset):
    data_root = Path(
        os.environ.get("CLIP_OOD_DATA_ROOT", ROOT / "data")
    ).expanduser().resolve()
    return data_root / "semantic_labels" / (dataset + "_image_semantic_labels.csv")


def read_scores(path):
    if not path.exists():
        raise RuntimeError(
            "Missing score file: {}\nRun baseline reproduction first.".format(path)
        )
    frame = pd.read_csv(path)
    missing = {"relative_path", "score"} - set(frame.columns)
    if missing:
        raise RuntimeError("{} missing columns: {}".format(path, sorted(missing)))
    if frame["relative_path"].duplicated().any():
        raise RuntimeError("Duplicate score paths in {}".format(path))
    if not np.isfinite(frame["score"].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite scores in {}".format(path))
    return frame


def primary_sources():
    cfg = load_yaml(PILOT)
    return [x["name"] for x in cfg["data"]["primary_ood_sources"]]


def thresholds():
    return load_yaml(SUBGROUP)["primary_eligibility"]


def plot_source(frame, source, aggregate, out_dir):
    ordered = frame.sort_values("fpr95", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(ordered))))
    ax.barh(ordered["group"], 100.0 * ordered["fpr95"])
    ax.axvline(100.0 * aggregate["fpr95_fixed_id95"], linestyle="--")
    ax.set_xlabel("FPR95 (%)")
    ax.set_ylabel("Predefined semantic group")
    ax.set_title("{}: subgroup FPR95".format(source))
    fig.tight_layout()
    fig.savefig(out_dir / (source + "_subgroup_fpr95.png"), dpi=160)
    plt.close(fig)

    ordered = frame.sort_values("auroc", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(ordered))))
    ax.barh(ordered["group"], 100.0 * ordered["auroc"])
    ax.axvline(100.0 * aggregate["auroc_fixed_id"], linestyle="--")
    ax.set_xlabel("AUROC (%)")
    ax.set_ylabel("Predefined semantic group")
    ax.set_title("{}: subgroup AUROC".format(source))
    fig.tight_layout()
    fig.savefig(out_dir / (source + "_subgroup_auroc.png"), dpi=160)
    plt.close(fig)


def evaluate_source(
    source,
    id_scores,
    id_reference,
    ood_frame,
    n_bootstrap,
    confidence,
    seed,
    out_dir,
):
    mapping_path = semantic_path(source)
    if not mapping_path.exists():
        raise RuntimeError(
            "Missing semantic mapping: {}\n"
            "Run reconstruct_mos_labels.py first.".format(mapping_path)
        )
    semantic = pd.read_csv(mapping_path)
    missing = {"relative_path", "leaf_concept", "primary_groups"} - set(
        semantic.columns
    )
    if missing:
        raise RuntimeError(
            "{} missing columns: {}".format(mapping_path, sorted(missing))
        )
    if semantic["relative_path"].duplicated().any():
        raise RuntimeError("Duplicate mapping paths in {}".format(mapping_path))

    merged = ood_frame.merge(
        semantic[["relative_path", "leaf_concept", "primary_groups"]],
        on="relative_path",
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(ood_frame):
        raise RuntimeError("Score/mapping join changed sample count")

    rules = thresholds()
    coverage = float(merged["primary_groups"].notna().mean())
    if coverage < float(rules["minimum_mapping_coverage"]):
        raise RuntimeError(
            "{} mapping coverage {:.4f} < frozen {:.4f}".format(
                source, coverage, float(rules["minimum_mapping_coverage"])
            )
        )

    source_scores = merged["score"].to_numpy(dtype=float)
    fixed = conditional_metrics(id_reference, source_scores)
    official = ood_metrics(id_scores, source_scores)
    aggregate = {
        "source": source,
        "n_ood": len(merged),
        "mapping_coverage": coverage,
        "id95_threshold": id_reference.threshold_95,
        "realized_id_tpr": realized_id_tpr(
            id_scores, id_reference.threshold_95
        ),
        "auroc_fixed_id": fixed["auroc"],
        "fpr95_fixed_id95": fixed["fpr95"],
        "auroc_official_pairwise": official["auroc"],
        "fpr95_official_pairwise": official["fpr95"],
        "aupr_in": official["aupr_in"],
        "aupr_out": official["aupr_out"],
    }

    source_auc = auc_contributions(id_reference, source_scores)
    source_fpr = fpr95_contributions(id_reference, source_scores)

    exploded = merged.copy()
    exploded["primary_groups"] = exploded["primary_groups"].fillna("")
    exploded["group"] = exploded["primary_groups"].str.split("|")
    exploded = exploded.explode("group")
    exploded = exploded[exploded["group"].astype(str).str.len() > 0].copy()

    min_images = int(rules["minimum_images_per_group"])
    min_leafs = int(rules["minimum_leaf_concepts_per_group"])
    rows = []

    for group_idx, group in enumerate(sorted(exploded["group"].unique())):
        group_rows = exploded[exploded["group"] == group]
        unique_paths = group_rows["relative_path"].drop_duplicates()
        mask = merged["relative_path"].isin(unique_paths).to_numpy()
        group_scores = merged.loc[mask, "score"].to_numpy(dtype=float)
        n_images = int(mask.sum())
        n_leafs = int(
            merged.loc[mask, "leaf_concept"].dropna().astype(str).nunique()
        )
        eligible = n_images >= min_images and n_leafs >= min_leafs
        row = {
            "source": source,
            "group": group,
            "n_images": n_images,
            "n_leaf_concepts": n_leafs,
            "eligible_primary": eligible,
        }

        if eligible:
            m = conditional_metrics(id_reference, group_scores)
            cis = bootstrap_group_metric_cis(
                id_reference,
                group_scores,
                n_resamples=n_bootstrap,
                confidence=confidence,
                seed=seed + 1000 * group_idx,
            )
            fpr_gap_ci = paired_source_bootstrap_gap_ci(
                source_fpr,
                mask,
                gap_direction="group_minus_aggregate",
                n_resamples=n_bootstrap,
                confidence=confidence,
                seed=seed + 1000 * group_idx + 101,
            )
            auc_gap_ci = paired_source_bootstrap_gap_ci(
                source_auc,
                mask,
                gap_direction="aggregate_minus_group",
                n_resamples=n_bootstrap,
                confidence=confidence,
                seed=seed + 1000 * group_idx + 202,
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
            for name in [
                "fpr95",
                "fpr95_ci_low",
                "fpr95_ci_high",
                "auroc",
                "auroc_ci_low",
                "auroc_ci_high",
                "fpr95_gap_vs_source",
                "fpr95_gap_ci_low",
                "fpr95_gap_ci_high",
                "auroc_gap_vs_source",
                "auroc_gap_ci_low",
                "auroc_gap_ci_high",
            ]:
                row[name] = np.nan
        rows.append(row)

    table = pd.DataFrame(rows)
    eligible = table[table["eligible_primary"]].copy()
    minimum_groups = int(rules["minimum_eligible_groups_per_source"])
    if len(eligible) < minimum_groups:
        raise RuntimeError(
            "{} has {} eligible groups; frozen minimum is {}".format(
                source, len(eligible), minimum_groups
            )
        )

    worst_fpr = eligible.loc[eligible["fpr95"].idxmax()]
    worst_auc = eligible.loc[eligible["auroc"].idxmin()]
    aggregate.update({
        "eligible_group_count": int(len(eligible)),
        "worst_fpr95_group": str(worst_fpr["group"]),
        "worst_fpr95": float(worst_fpr["fpr95"]),
        "worst_fpr95_gap": float(worst_fpr["fpr95_gap_vs_source"]),
        "worst_auroc_group": str(worst_auc["group"]),
        "worst_auroc": float(worst_auc["auroc"]),
        "worst_auroc_gap": float(worst_auc["auroc_gap_vs_source"]),
    })

    table.to_csv(out_dir / (source + "_subgroups.csv"), index=False)
    plot_source(eligible, source, aggregate, out_dir)
    return aggregate


def write_markdown(method, backbone, summary, out_dir):
    lines = [
        "# H1 predefined semantic subgroup analysis",
        "",
        "- method: {}".format(method),
        "- backbone: {}".format(backbone),
        "- one fixed ImageNet ID 95%-TPR threshold is shared by all subgroups",
        "- bootstrap resamples OOD samples while the full ID reference stays fixed",
        "",
        "| Source | Aggregate FPR95 | Worst FPR95 | Worst group | Aggregate AUROC | Worst AUROC | Worst group |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {0} | {1:.2f}% | {2:.2f}% | {3} | {4:.2f}% | {5:.2f}% | {6} |".format(
                row["source"],
                100 * row["fpr95_fixed_id95"],
                100 * row["worst_fpr95"],
                row["worst_fpr95_group"],
                100 * row["auroc_fixed_id"],
                100 * row["worst_auroc"],
                row["worst_auroc_group"],
            )
        )
    lines += [
        "",
        "GO / CONDITIONAL GO / KILL is intentionally not assigned automatically.",
        "Apply the pre-registered decision rule after checking effect size,",
        "bootstrap intervals, recurrence, and the second grouping definition.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["mcm", "neglabel"])
    parser.add_argument(
        "--backbone",
        default="ViT-B/32",
        choices=["ViT-B/16", "ViT-B/32"],
    )
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sources = args.sources or primary_sources()
    non_primary = sorted(set(sources) - set(primary_sources()))
    if non_primary:
        raise RuntimeError(
            "Not primary after pre-score mapping gate: "
            + ", ".join(non_primary)
        )

    id_frame = read_scores(score_path(args.method, args.backbone, "imagenet"))
    id_scores = id_frame["score"].to_numpy(dtype=float)
    reference = build_fixed_id_reference(id_scores)

    out_dir = (
        ROOT
        / "results"
        / "h1_predefined"
        / backbone_slug(args.backbone)
        / args.method
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregates = []
    for source in sources:
        print(
            "\n=== H1 predefined: {} / {} / {} ===".format(
                args.method, args.backbone, source
            )
        )
        ood_frame = read_scores(score_path(args.method, args.backbone, source))
        aggregate = evaluate_source(
            source,
            id_scores,
            reference,
            ood_frame,
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
    summary.to_csv(out_dir / "aggregate_vs_worst.csv", index=False)
    metadata = {
        "method": args.method,
        "backbone": args.backbone,
        "primary_sources": sources,
        "bootstrap_resamples": args.bootstrap,
        "confidence": args.confidence,
        "seed": args.seed,
        "id95_threshold": reference.threshold_95,
        "realized_id_tpr": realized_id_tpr(
            id_scores, reference.threshold_95
        ),
        "id_reference_size": int(len(id_scores)),
        "score_orientation": "larger_is_more_ID_like",
        "bootstrap_reference_policy": "full_ID_fixed_OOD_sample_bootstrap",
    }
    (out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.method, args.backbone, summary, out_dir)
    print("\nH1 outputs: {}".format(out_dir))


if __name__ == "__main__":
    main()
