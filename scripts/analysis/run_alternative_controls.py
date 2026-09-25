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

from analysis.alternative_controls import oof_similarity_risk
from analysis.h1_predefined import (
    build_fixed_id_reference,
    paired_source_bootstrap_gap_ci,
)
from baselines.common import backbone_slug

CONTROL_CFG = ROOT / "configs" / "alternative_controls.yaml"
PREDEFINED_CFG = ROOT / "configs" / "subgroups" / "predefined.yaml"
TEXT_CFG = ROOT / "configs" / "subgroups" / "text_clustering.yaml"
MAPPING_DIR = ROOT / "configs" / "subgroups" / "mappings"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_root() -> Path:
    return Path(
        os.environ.get("CLIP_OOD_DATA_ROOT", ROOT / "data")
    ).expanduser().resolve()


def baseline_path(method: str, backbone: str, dataset: str) -> Path:
    return (
        ROOT
        / "results"
        / "raw"
        / "reproduction"
        / backbone_slug(backbone)
        / method
        / f"{dataset}.csv"
    )


def proximity_path(backbone: str, source: str) -> Path:
    return (
        ROOT
        / "results"
        / "raw"
        / "alternative_controls"
        / backbone_slug(backbone)
        / f"{source}_proximity.csv.gz"
    )


def read_baseline(method: str, backbone: str, dataset: str) -> pd.DataFrame:
    path = baseline_path(method, backbone, dataset)
    if not path.exists():
        raise RuntimeError(f"Missing baseline score file: {path}")
    frame = pd.read_csv(path)
    required = {"relative_path", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing {sorted(missing)}")
    if frame["relative_path"].duplicated().any():
        raise RuntimeError(f"Duplicate baseline path in {path}")
    return frame


def read_proximity(backbone: str, source: str) -> pd.DataFrame:
    path = proximity_path(backbone, source)
    if not path.exists():
        raise RuntimeError(
            f"Missing control proximity file: {path}\n"
            "Run prepare_alternative_control_proximity.py first."
        )
    frame = pd.read_csv(path)
    required = {
        "relative_path",
        "leaf_concept",
        "visual_id_similarity",
        "semantic_id_similarity",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing {sorted(missing)}")
    if frame["relative_path"].duplicated().any():
        raise RuntimeError(f"Duplicate proximity path in {path}")
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
        raise RuntimeError(f"Missing text-cluster mapping: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    mapping = {
        row["leaf_concept"].strip(): row["text_cluster"].strip()
        for row in rows
    }
    if not mapping:
        raise RuntimeError(f"Empty text-cluster mapping: {path}")
    return mapping


def h1_summary_path(
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


def h1_fixed_worst(
    grouping: str,
    method: str,
    backbone: str,
    source: str,
) -> dict:
    path = h1_summary_path(grouping, method, backbone)
    if not path.exists():
        raise RuntimeError(f"Missing completed H1 summary: {path}")
    frame = pd.read_csv(path)
    row = frame.loc[frame["source"] == source]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected one H1 row for {method}/{source}/{grouping}"
        )
    item = row.iloc[0]
    return {
        "group": str(item["worst_fpr95_group"]),
        "stored_aggregate_fpr95": float(item["fpr95_fixed_id95"]),
        "stored_worst_fpr95": float(item["worst_fpr95"]),
        "stored_gap": float(item["worst_fpr95_gap"]),
    }


def group_masks_predefined(
    source: str,
    frame: pd.DataFrame,
) -> list[dict]:
    semantic = pd.read_csv(semantic_path(source))
    required = {"relative_path", "leaf_concept", "primary_groups"}
    missing = required - set(semantic.columns)
    if missing:
        raise RuntimeError(
            f"{source} semantic mapping missing {sorted(missing)}"
        )

    merged = frame[["relative_path"]].merge(
        semantic[["relative_path", "leaf_concept", "primary_groups"]],
        on="relative_path",
        how="left",
        validate="one_to_one",
    )
    rules = load_yaml(PREDEFINED_CFG)["primary_eligibility"]
    coverage = float(merged["primary_groups"].notna().mean())
    if coverage < float(rules["minimum_mapping_coverage"]):
        raise RuntimeError(
            f"{source} predefined coverage {coverage:.4f} below frozen minimum"
        )

    exploded = merged.copy()
    exploded["primary_groups"] = exploded["primary_groups"].fillna("")
    exploded["group"] = exploded["primary_groups"].str.split("|")
    exploded = exploded.explode("group")
    exploded = exploded[
        exploded["group"].astype(str).str.len() > 0
    ].copy()

    records = []
    for group in sorted(exploded["group"].unique()):
        paths = (
            exploded.loc[
                exploded["group"] == group,
                "relative_path",
            ]
            .drop_duplicates()
        )
        mask = frame["relative_path"].isin(paths).to_numpy()
        n_leafs = int(
            merged.loc[mask, "leaf_concept"]
            .dropna()
            .astype(str)
            .nunique()
        )
        eligible = (
            int(mask.sum()) >= int(rules["minimum_images_per_group"])
            and n_leafs
            >= int(rules["minimum_leaf_concepts_per_group"])
        )
        records.append({
            "group": str(group),
            "mask": mask,
            "n_images": int(mask.sum()),
            "n_leaf_concepts": n_leafs,
            "eligible": eligible,
        })
    return records


def group_masks_text(
    source: str,
    frame: pd.DataFrame,
) -> list[dict]:
    semantic = pd.read_csv(semantic_path(source))
    required = {"relative_path", "leaf_concept"}
    missing = required - set(semantic.columns)
    if missing:
        raise RuntimeError(
            f"{source} semantic mapping missing {sorted(missing)}"
        )
    mapping = load_leaf_clusters(source)
    semantic = semantic[["relative_path", "leaf_concept"]].copy()
    semantic["text_cluster"] = semantic["leaf_concept"].map(mapping)

    merged = frame[["relative_path"]].merge(
        semantic,
        on="relative_path",
        how="left",
        validate="one_to_one",
    )
    rules = load_yaml(TEXT_CFG)["eligibility"]
    coverage = float(merged["text_cluster"].notna().mean())
    if coverage < float(rules["minimum_mapping_coverage"]):
        raise RuntimeError(
            f"{source} text-cluster coverage {coverage:.4f} below frozen minimum"
        )

    records = []
    for group in sorted(merged["text_cluster"].dropna().unique()):
        mask = (merged["text_cluster"] == group).to_numpy()
        n_leafs = int(
            merged.loc[mask, "leaf_concept"]
            .dropna()
            .astype(str)
            .nunique()
        )
        eligible = (
            int(mask.sum()) >= int(rules["minimum_images_per_group"])
            and n_leafs
            >= int(rules["minimum_leaf_concepts_per_group"])
        )
        records.append({
            "group": str(group),
            "mask": mask,
            "n_images": int(mask.sum()),
            "n_leaf_concepts": n_leafs,
            "eligible": eligible,
        })
    return records


def group_rows(
    *,
    method: str,
    source: str,
    grouping: str,
    frame: pd.DataFrame,
    error: np.ndarray,
    residual: np.ndarray,
    expected_risk: np.ndarray,
    n_bootstrap: int,
    confidence: float,
    seed: int,
    backbone: str,
) -> list[dict]:
    fixed = h1_fixed_worst(
        grouping,
        method,
        backbone,
        source,
    )
    records = (
        group_masks_predefined(source, frame)
        if grouping == "predefined"
        else group_masks_text(source, frame)
    )

    rows = []
    for idx, rec in enumerate(records):
        if not rec["eligible"]:
            continue
        mask = rec["mask"]
        unadjusted_gap = float(error[mask].mean() - error.mean())
        adjusted_gap = float(residual[mask].mean() - residual.mean())

        unadj_ci = paired_source_bootstrap_gap_ci(
            error.astype(np.float64),
            mask,
            gap_direction="group_minus_aggregate",
            n_resamples=n_bootstrap,
            confidence=confidence,
            seed=seed + idx * 1000 + 11,
        )
        adj_ci = paired_source_bootstrap_gap_ci(
            residual,
            mask,
            gap_direction="group_minus_aggregate",
            n_resamples=n_bootstrap,
            confidence=confidence,
            seed=seed + idx * 1000 + 29,
        )
        attenuation = (
            1.0 - adjusted_gap / unadjusted_gap
            if abs(unadjusted_gap) > 1e-12
            else np.nan
        )
        is_fixed = rec["group"] == fixed["group"]

        rows.append({
            "method": method,
            "backbone": backbone,
            "source": source,
            "grouping": grouping,
            "group": rec["group"],
            "n_images": rec["n_images"],
            "n_leaf_concepts": rec["n_leaf_concepts"],
            "is_h1_fixed_worst_fpr95_group": is_fixed,
            "source_fpr95": float(error.mean()),
            "group_fpr95": float(error[mask].mean()),
            "unadjusted_gap": unadjusted_gap,
            "unadjusted_gap_ci_low": unadj_ci[0],
            "unadjusted_gap_ci_high": unadj_ci[1],
            "source_expected_risk": float(expected_risk.mean()),
            "group_expected_risk": float(expected_risk[mask].mean()),
            "source_mean_residual": float(residual.mean()),
            "group_mean_residual": float(residual[mask].mean()),
            "adjusted_residual_gap": adjusted_gap,
            "adjusted_gap_ci_low": adj_ci[0],
            "adjusted_gap_ci_high": adj_ci[1],
            "attenuation_fraction": attenuation,
            "residual_failure_persists": bool(adj_ci[0] > 0.0),
            "h1_stored_gap": fixed["stored_gap"] if is_fixed else np.nan,
        })

    fixed_rows = [r for r in rows if r["is_h1_fixed_worst_fpr95_group"]]
    if len(fixed_rows) != 1:
        raise RuntimeError(
            f"Could not locate exactly one fixed H1 worst group for "
            f"{method}/{source}/{grouping}: {fixed['group']}"
        )
    current = fixed_rows[0]
    if not np.isclose(
        current["unadjusted_gap"],
        fixed["stored_gap"],
        atol=5e-6,
        rtol=1e-4,
    ):
        raise RuntimeError(
            "Alternative-control reconstruction changed the stored H1 gap: "
            f"{current['unadjusted_gap']} vs {fixed['stored_gap']}"
        )
    return rows


def write_readme(
    fixed: pd.DataFrame,
    condition: pd.DataFrame,
    out_dir: Path,
) -> None:
    lines = [
        "# Alternative-explanation controls",
        "",
        "This stage follows H1 PASS and H2 FAIL/CONDITIONAL GO.",
        "",
        "It does not redefine H1 and has no new global PASS/FAIL gate.",
        "The fixed H1 worst-FPR95 group is carried forward unchanged.",
        "",
        "Difficulty model: 5-fold OOF logistic regression using only:",
        "",
        "- nearest ImageNet-val CLIP image similarity;",
        "- independent MiniLM leaf-concept-to-ImageNet-class similarity;",
        "- frozen degree-2 polynomial terms of those two proxies.",
        "",
        "Detector score, prompt mean/std, and subgroup labels are forbidden predictors.",
        "",
        "| Method | Source | Grouping | Fixed H1 worst | Raw gap | Adjusted residual gap | 95% CI | Persists? |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for _, row in fixed.iterrows():
        lines.append(
            "| {method} | {source} | {grouping} | {group} | "
            "{raw:.2f}pp | {adj:.2f}pp | [{lo:.2f},{hi:.2f}]pp | {persist} |".format(
                method=row["method"],
                source=row["source"],
                grouping=row["grouping"],
                group=row["group"],
                raw=100.0 * row["unadjusted_gap"],
                adj=100.0 * row["adjusted_residual_gap"],
                lo=100.0 * row["adjusted_gap_ci_low"],
                hi=100.0 * row["adjusted_gap_ci_high"],
                persist="yes" if row["residual_failure_persists"] else "no",
            )
        )

    lines += [
        "",
        "Interpretation:",
        "",
        "- persists=yes: the frozen similarity proxies do not fully explain the fixed H1 worst-group excess;",
        "- persists=no: the fixed H1 excess is no longer distinguishable from zero after adjustment, supporting the competing explanation as a plausible account;",
        "- neither result is a causal attribution.",
        "",
        "Similarity-only OOF risk-model diagnostics:",
        "",
        "| Method | Source | Failure rate | OOF AUROC | OOF log-loss |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in condition.iterrows():
        lines.append(
            "| {method} | {source} | {fpr:.2f}% | {auc:.4f} | {ll:.4f} |".format(
                method=row["method"],
                source=row["source"],
                fpr=100.0 * row["failure_rate"],
                auc=row["similarity_risk_auc"],
                ll=row["similarity_risk_logloss"],
            )
        )

    lines += [
        "",
        "After this control, the next pre-planned robustness stage is ViT-B/16 H1 replication.",
        "",
    ]
    (out_dir / "README.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        default="ViT-B/32",
        choices=["ViT-B/32"],
    )
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    cfg = load_yaml(CONTROL_CFG)
    methods = list(cfg["scope"]["methods"])
    sources = list(cfg["scope"]["sources"])
    n_bootstrap = (
        int(args.bootstrap)
        if args.bootstrap is not None
        else int(cfg["group_adjustment"]["bootstrap"]["resamples"])
    )
    seed = int(cfg["risk_model"]["seed"])
    folds = int(
        str(cfg["risk_model"]["validation"])
        .replace("stratified_", "")
        .replace("fold_oof", "")
    )
    if folds != 5:
        raise RuntimeError("Frozen control protocol requires 5 OOF folds")

    out_dir = (
        ROOT
        / "results"
        / "alternative_controls"
        / backbone_slug(args.backbone)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    condition_rows = []
    all_group_rows = []

    for method in methods:
        id_frame = read_baseline(method, args.backbone, "imagenet")
        reference = build_fixed_id_reference(
            id_frame["score"].to_numpy(dtype=float)
        )

        for source in sources:
            print(
                f"\n=== alternative controls: {method} / {source} ==="
            )
            baseline = read_baseline(
                method,
                args.backbone,
                source,
            )[["relative_path", "score"]].rename(
                columns={"score": "baseline_score"}
            )
            proximity = read_proximity(args.backbone, source)
            frame = baseline.merge(
                proximity,
                on="relative_path",
                how="inner",
                validate="one_to_one",
            )
            if len(frame) != len(baseline):
                raise RuntimeError(
                    f"{method}/{source}: baseline/proximity sample mismatch"
                )

            error = (
                frame["baseline_score"].to_numpy(dtype=float)
                >= reference.threshold_95
            ).astype(np.int32)
            visual = frame["visual_id_similarity"].to_numpy(dtype=float)
            semantic = frame["semantic_id_similarity"].to_numpy(dtype=float)

            risk = oof_similarity_risk(
                error,
                visual,
                semantic,
                folds=folds,
                seed=seed,
            )
            residual = error.astype(np.float64) - risk.probabilities

            condition_rows.append({
                "method": method,
                "backbone": args.backbone,
                "source": source,
                "n_ood": int(len(frame)),
                "n_errors": int(error.sum()),
                "failure_rate": float(error.mean()),
                "visual_similarity_mean": float(visual.mean()),
                "semantic_similarity_mean": float(semantic.mean()),
                "similarity_risk_auc": risk.auc,
                "similarity_risk_logloss": risk.logloss,
                "mean_oof_predicted_risk": float(risk.probabilities.mean()),
                "mean_residual": float(residual.mean()),
            })

            for grouping in ["predefined", "minilm_kmeans"]:
                rows = group_rows(
                    method=method,
                    source=source,
                    grouping=grouping,
                    frame=frame,
                    error=error,
                    residual=residual,
                    expected_risk=risk.probabilities,
                    n_bootstrap=n_bootstrap,
                    confidence=args.confidence,
                    seed=seed,
                    backbone=args.backbone,
                )
                all_group_rows.extend(rows)

    condition = pd.DataFrame(condition_rows)
    groups = pd.DataFrame(all_group_rows)
    fixed = groups[
        groups["is_h1_fixed_worst_fpr95_group"]
    ].copy()

    condition.to_csv(
        out_dir / "similarity_risk_summary.csv",
        index=False,
    )
    groups.to_csv(
        out_dir / "all_group_adjusted_gaps.csv",
        index=False,
    )
    fixed.to_csv(
        out_dir / "fixed_h1_worst_groups.csv",
        index=False,
    )

    metadata = {
        "stage": "alternative_explanation_controls",
        "freeze_date": "2026-09-25",
        "backbone": args.backbone,
        "methods": methods,
        "sources": sources,
        "bootstrap_resamples": n_bootstrap,
        "confidence": args.confidence,
        "seed": seed,
        "global_binary_pass_fail": False,
        "fixed_group_policy": cfg["group_adjustment"]["fixed_group_policy"],
        "next_stage_after_review": "ViT-B/16_H1_replication",
    }
    (out_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_readme(fixed, condition, out_dir)

    print("\nFixed H1 worst-group residual gaps:")
    for _, row in fixed.iterrows():
        print(
            "{}/{}/{} {}: raw={:.2f}pp adjusted={:.2f}pp "
            "CI=[{:.2f},{:.2f}]pp persists={}".format(
                row["method"],
                row["source"],
                row["grouping"],
                row["group"],
                100.0 * row["unadjusted_gap"],
                100.0 * row["adjusted_residual_gap"],
                100.0 * row["adjusted_gap_ci_low"],
                100.0 * row["adjusted_gap_ci_high"],
                bool(row["residual_failure_persists"]),
            )
        )

    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
