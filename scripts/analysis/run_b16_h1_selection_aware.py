#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.h1_predefined import (
    auc_contributions,
    build_fixed_id_reference,
    fpr95_contributions,
)
from analysis.selection_aware import selection_aware_worst_gap
from baselines.common import backbone_slug

CFG = ROOT / "configs" / "b16_h1_replication.yaml"
MAPPING_DIR = ROOT / "configs" / "subgroups" / "mappings"


def load_cfg() -> dict:
    with CFG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_root() -> Path:
    return Path(
        os.environ.get("CLIP_OOD_DATA_ROOT", ROOT / "data")
    ).expanduser().resolve()


def score_path(method: str, backbone: str, dataset: str) -> Path:
    return (
        ROOT
        / "results"
        / "raw"
        / "reproduction"
        / backbone_slug(backbone)
        / method
        / f"{dataset}.csv"
    )


def read_scores(method: str, backbone: str, dataset: str) -> pd.DataFrame:
    path = score_path(method, backbone, dataset)
    if not path.exists():
        raise RuntimeError(f"Missing score file: {path}")
    frame = pd.read_csv(path)
    required = {"relative_path", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns {sorted(missing)}")
    if frame["relative_path"].duplicated().any():
        raise RuntimeError(f"Duplicate paths in {path}")
    if not np.isfinite(frame["score"].to_numpy(dtype=float)).all():
        raise RuntimeError(f"Non-finite scores in {path}")
    return frame


def semantic_path(source: str) -> Path:
    return (
        data_root()
        / "semantic_labels"
        / f"{source}_image_semantic_labels.csv"
    )


def load_leaf_clusters(source: str) -> dict[str, str]:
    path = MAPPING_DIR / f"text_clusters_{source}_minilm.csv"
    if not path.exists():
        raise RuntimeError(f"Missing frozen text-cluster mapping: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    mapping = {
        row["leaf_concept"].strip(): row["text_cluster"].strip()
        for row in rows
    }
    if not mapping:
        raise RuntimeError(f"Empty text-cluster mapping: {path}")
    return mapping


def subgroup_result_path(
    grouping: str,
    method: str,
    backbone: str,
    source: str,
) -> Path:
    if grouping == "predefined":
        return (
            ROOT
            / "results"
            / "h1_predefined"
            / backbone_slug(backbone)
            / method
            / f"{source}_subgroups.csv"
        )
    if grouping == "minilm_kmeans":
        return (
            ROOT
            / "results"
            / "h1_text_clusters"
            / backbone_slug(backbone)
            / method
            / f"{source}_text_clusters.csv"
        )
    raise ValueError(grouping)


def h1_aggregate_path(
    grouping: str,
    method: str,
    backbone: str,
) -> Path:
    root_name = (
        "h1_predefined"
        if grouping == "predefined"
        else "h1_text_clusters"
    )
    return (
        ROOT
        / "results"
        / root_name
        / backbone_slug(backbone)
        / method
        / "aggregate_vs_worst.csv"
    )


def eligible_group_names(
    grouping: str,
    method: str,
    backbone: str,
    source: str,
) -> list[str]:
    path = subgroup_result_path(grouping, method, backbone, source)
    if not path.exists():
        raise RuntimeError(
            f"Missing H1 subgroup result: {path}\n"
            "Run the standard H1 analysis for ViT-B/16 first."
        )
    table = pd.read_csv(path)
    required = {"group", "eligible_primary"}
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError(f"{path} missing {sorted(missing)}")
    eligible = table.loc[
        table["eligible_primary"].astype(bool),
        "group",
    ].astype(str).tolist()
    if len(eligible) < 3:
        raise RuntimeError(
            f"{method}/{source}/{grouping} has fewer than 3 eligible groups"
        )
    return eligible


def build_group_masks(
    grouping: str,
    source: str,
    score_frame: pd.DataFrame,
    eligible_names: list[str],
) -> dict[str, np.ndarray]:
    semantic = pd.read_csv(semantic_path(source))
    if grouping == "predefined":
        required = {"relative_path", "primary_groups"}
        missing = required - set(semantic.columns)
        if missing:
            raise RuntimeError(
                f"{source} semantic mapping missing {sorted(missing)}"
            )
        merged = score_frame[["relative_path"]].merge(
            semantic[["relative_path", "primary_groups"]],
            on="relative_path",
            how="left",
            validate="one_to_one",
        )
        group_strings = merged["primary_groups"].fillna("").astype(str)
        masks = {}
        for group in eligible_names:
            masks[group] = group_strings.apply(
                lambda text: group in text.split("|")
            ).to_numpy(dtype=bool)
        return masks

    if grouping == "minilm_kmeans":
        required = {"relative_path", "leaf_concept"}
        missing = required - set(semantic.columns)
        if missing:
            raise RuntimeError(
                f"{source} semantic mapping missing {sorted(missing)}"
            )
        mapping = load_leaf_clusters(source)
        sem = semantic[["relative_path", "leaf_concept"]].copy()
        sem["text_cluster"] = sem["leaf_concept"].map(mapping)
        merged = score_frame[["relative_path"]].merge(
            sem,
            on="relative_path",
            how="left",
            validate="one_to_one",
        )
        masks = {
            group: (merged["text_cluster"].astype(str) == group).to_numpy()
            for group in eligible_names
        }
        return masks

    raise ValueError(grouping)


def validate_against_standard_h1(
    *,
    grouping: str,
    method: str,
    backbone: str,
    source: str,
    fpr_result,
    auc_result,
) -> None:
    path = h1_aggregate_path(grouping, method, backbone)
    if not path.exists():
        raise RuntimeError(f"Missing standard H1 aggregate result: {path}")
    table = pd.read_csv(path)
    row = table.loc[table["source"] == source]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected one standard H1 row for {method}/{source}/{grouping}"
        )
    item = row.iloc[0]

    checks = [
        (
            "aggregate FPR95",
            fpr_result.point_aggregate,
            float(item["fpr95_fixed_id95"]),
        ),
        (
            "worst FPR95",
            fpr_result.point_worst,
            float(item["worst_fpr95"]),
        ),
        (
            "FPR95 gap",
            fpr_result.point_gap,
            float(item["worst_fpr95_gap"]),
        ),
        (
            "aggregate AUROC",
            auc_result.point_aggregate,
            float(item["auroc_fixed_id"]),
        ),
        (
            "worst AUROC",
            auc_result.point_worst,
            float(item["worst_auroc"]),
        ),
        (
            "AUROC gap",
            auc_result.point_gap,
            float(item["worst_auroc_gap"]),
        ),
    ]
    for name, actual, expected in checks:
        if not np.isclose(actual, expected, atol=5e-6, rtol=1e-4):
            raise RuntimeError(
                f"{method}/{source}/{grouping} {name} mismatch: "
                f"{actual} vs {expected}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        default="ViT-B/16",
        choices=["ViT-B/16"],
    )
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_cfg()
    methods = list(cfg["scope"]["methods"])
    sources = list(cfg["scope"]["primary_sources"])
    groupings = list(cfg["scope"]["subgroup_definitions"])
    n_bootstrap = (
        int(args.bootstrap)
        if args.bootstrap is not None
        else int(cfg["uncertainty"]["bootstrap_resamples"])
    )
    confidence = (
        float(args.confidence)
        if args.confidence is not None
        else float(cfg["uncertainty"]["confidence"])
    )
    seed = (
        int(args.seed)
        if args.seed is not None
        else int(cfg["uncertainty"]["seed"])
    )

    out_dir = (
        ROOT
        / "results"
        / "h1_replication"
        / backbone_slug(args.backbone)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    selection_rows = []

    for method in methods:
        id_frame = read_scores(method, args.backbone, "imagenet")
        reference = build_fixed_id_reference(
            id_frame["score"].to_numpy(dtype=float)
        )

        for source in sources:
            ood_frame = read_scores(method, args.backbone, source)
            ood_scores = ood_frame["score"].to_numpy(dtype=float)
            fpr_values = fpr95_contributions(reference, ood_scores)
            auc_values = auc_contributions(reference, ood_scores)

            for grouping in groupings:
                names = eligible_group_names(
                    grouping,
                    method,
                    args.backbone,
                    source,
                )
                masks = build_group_masks(
                    grouping,
                    source,
                    ood_frame,
                    names,
                )

                print(
                    f"\n=== selection-aware H1 replication: "
                    f"{method}/{source}/{grouping} ==="
                )
                fpr_result = selection_aware_worst_gap(
                    fpr_values,
                    masks,
                    gap_direction="group_minus_aggregate",
                    n_resamples=n_bootstrap,
                    confidence=confidence,
                    seed=seed,
                )
                auc_result = selection_aware_worst_gap(
                    auc_values,
                    masks,
                    gap_direction="aggregate_minus_group",
                    n_resamples=n_bootstrap,
                    confidence=confidence,
                    seed=seed + 500000,
                )
                validate_against_standard_h1(
                    grouping=grouping,
                    method=method,
                    backbone=args.backbone,
                    source=source,
                    fpr_result=fpr_result,
                    auc_result=auc_result,
                )

                for metric, result in [
                    ("fpr95", fpr_result),
                    ("auroc", auc_result),
                ]:
                    support = bool(result.gap_ci_low > 0.0)
                    summary_rows.append({
                        "method": method,
                        "backbone": args.backbone,
                        "source": source,
                        "grouping": grouping,
                        "metric": metric,
                        "point_aggregate": result.point_aggregate,
                        "point_worst": result.point_worst,
                        "point_gap": result.point_gap,
                        "point_worst_group": result.point_worst_group,
                        "selection_aware_gap_median": result.gap_median,
                        "selection_aware_ci_low": result.gap_ci_low,
                        "selection_aware_ci_high": result.gap_ci_high,
                        "empirical_p_gap_le_zero": result.empirical_p_gap_le_zero,
                        "condition_supported": support,
                    })
                    for group, count in result.selected_counts.items():
                        selection_rows.append({
                            "method": method,
                            "backbone": args.backbone,
                            "source": source,
                            "grouping": grouping,
                            "metric": metric,
                            "group": group,
                            "selected_count": int(count),
                            "selected_fraction": count / n_bootstrap,
                        })

                print(
                    "FPR95 gap median={:.2f}pp CI=[{:.2f},{:.2f}]pp "
                    "support={}".format(
                        100.0 * fpr_result.gap_median,
                        100.0 * fpr_result.gap_ci_low,
                        100.0 * fpr_result.gap_ci_high,
                        fpr_result.gap_ci_low > 0.0,
                    )
                )

    summary = pd.DataFrame(summary_rows)
    selections = pd.DataFrame(selection_rows)
    summary.to_csv(
        out_dir / "selection_aware_summary.csv",
        index=False,
    )
    selections.to_csv(
        out_dir / "worst_group_selection_counts.csv",
        index=False,
    )

    fpr = summary[summary["metric"] == "fpr95"].copy()
    minilm = fpr[fpr["grouping"] == "minilm_kmeans"].copy()
    minilm_supported = minilm[minilm["condition_supported"]].copy()
    n_minilm_supported = int(len(minilm_supported))
    represented_methods = set(minilm_supported["method"].astype(str))
    both_methods = represented_methods == {"mcm", "neglabel"}

    pre_inat = fpr[
        (fpr["grouping"] == "predefined")
        & (fpr["source"] == "inaturalist")
    ].copy()
    predefined_inat_both = bool(
        len(pre_inat) == 2
        and pre_inat["condition_supported"].astype(bool).all()
    )

    strong = (
        n_minilm_supported
        >= int(
            cfg[
                "overall_replication_interpretation"
            ]["strong"]["requirements"][
                "min_supported_minilm_conditions"
            ]
        )
        and both_methods
        and predefined_inat_both
    )
    any_pre_inat = bool(
        pre_inat["condition_supported"].astype(bool).any()
    )
    partial = (
        not strong
        and (
            n_minilm_supported >= 2
            or any_pre_inat
        )
    )
    if strong:
        interpretation = "STRONG_REPLICATION"
    elif partial:
        interpretation = "PARTIAL_REPLICATION"
    else:
        interpretation = "LIMITED_REPLICATION"

    decision = {
        "stage": "ViT-B16_H1_replication",
        "freeze_date": "2026-09-25",
        "backbone": args.backbone,
        "bootstrap_resamples": n_bootstrap,
        "confidence": confidence,
        "seed": seed,
        "primary_endpoint": "selection_aware_fpr95_gap",
        "condition_support_rule": "bootstrap_95CI_lower_gt_0",
        "minilm_supported_conditions": n_minilm_supported,
        "minilm_total_conditions": 4,
        "minilm_both_methods_represented": both_methods,
        "predefined_inaturalist_both_methods_supported":
            predefined_inat_both,
        "replication_interpretation": interpretation,
        "affects_project_gate": False,
        "project_gate_remains": "CONDITIONAL_GO_after_H2_FAIL",
    }
    (out_dir / "replication_interpretation.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"\nViT-B/16 replication interpretation: {interpretation}"
    )
    print(
        f"MiniLM supported: {n_minilm_supported}/4 | "
        f"predefined iNaturalist both methods: {predefined_inat_both}"
    )
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
