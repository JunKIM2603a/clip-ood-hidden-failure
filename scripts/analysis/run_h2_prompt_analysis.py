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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analysis.h1_predefined import build_fixed_id_reference
from analysis.h2_prompt_sensitivity import (
    compare_predictive_models,
    fit_association,
    wilson_interval,
)
from baselines.common import backbone_slug

PILOT = ROOT / "configs" / "pilot.yaml"
PREDEFINED_CFG = ROOT / "configs" / "subgroups" / "predefined.yaml"
TEXT_CLUSTER_CFG = ROOT / "configs" / "subgroups" / "text_clustering.yaml"
MAPPING_DIR = ROOT / "configs" / "subgroups" / "mappings"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_root() -> Path:
    return Path(
        os.environ.get("CLIP_OOD_DATA_ROOT", ROOT / "data")
    ).expanduser().resolve()


def primary_sources() -> list[str]:
    cfg = load_yaml(PILOT)
    return [
        item["name"]
        for item in cfg["data"]["primary_ood_sources"]
    ]


def baseline_score_path(method: str, backbone: str, dataset: str) -> Path:
    return (
        ROOT / "results" / "raw" / "reproduction"
        / backbone_slug(backbone) / method / f"{dataset}.csv"
    )


def prompt_score_path(method: str, backbone: str, dataset: str) -> Path:
    return (
        ROOT / "results" / "raw" / "h2_prompt"
        / backbone_slug(backbone) / method / f"{dataset}.csv.gz"
    )


def read_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Missing H1 baseline score file: {path}")
    frame = pd.read_csv(path)
    required = {"relative_path", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns {sorted(missing)}")
    if frame["relative_path"].duplicated().any():
        raise RuntimeError(f"Duplicate baseline relative_path in {path}")
    if not np.isfinite(frame["score"].to_numpy(dtype=float)).all():
        raise RuntimeError(f"Non-finite baseline scores in {path}")
    return frame


def read_prompt_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Missing H2 prompt score file: {path}")
    frame = pd.read_csv(path)
    required = {"relative_path", "mu", "sigma"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns {sorted(missing)}")
    prompt_cols = sorted(
        col for col in frame.columns if col.startswith("prompt_")
    )
    if len(prompt_cols) != 8:
        raise RuntimeError(
            "Primary H2 requires the frozen 8 prompt columns; "
            f"found {len(prompt_cols)} in {path}"
        )
    if frame["relative_path"].duplicated().any():
        raise RuntimeError(f"Duplicate H2 relative_path in {path}")
    numeric = frame[prompt_cols + ["mu", "sigma"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"Non-finite H2 prompt values in {path}")
    return frame


def merge_condition(
    method: str,
    backbone: str,
    source: str,
) -> tuple[pd.DataFrame, float, float]:
    id_frame = read_baseline(
        baseline_score_path(method, backbone, "imagenet")
    )
    id_scores = id_frame["score"].to_numpy(dtype=float)
    reference = build_fixed_id_reference(id_scores)

    baseline = read_baseline(
        baseline_score_path(method, backbone, source)
    )[["relative_path", "score"]].rename(
        columns={"score": "baseline_score"}
    )
    prompt = read_prompt_scores(
        prompt_score_path(method, backbone, source)
    )

    merged = baseline.merge(
        prompt,
        on="relative_path",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(baseline):
        raise RuntimeError(
            f"{method}/{source}: baseline/H2 sample mismatch "
            f"{len(baseline)} != {len(merged)}"
        )

    merged["error"] = (
        merged["baseline_score"] >= reference.threshold_95
    ).astype(np.int32)
    return (
        merged,
        float(reference.threshold_95),
        float(merged["error"].mean()),
    )


def sigma_quartiles(
    frame: pd.DataFrame,
    *,
    method: str,
    source: str,
    confidence: float,
) -> pd.DataFrame:
    bins = pd.qcut(
        frame["sigma"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="raise",
    )
    rows = []
    for quartile in ["Q1", "Q2", "Q3", "Q4"]:
        subset = frame.loc[bins == quartile]
        n = int(len(subset))
        failures = int(subset["error"].sum())
        low, high = wilson_interval(
            failures,
            n,
            confidence=confidence,
        )
        rows.append({
            "method": method,
            "source": source,
            "sigma_quartile": quartile,
            "n": n,
            "failures": failures,
            "failure_rate": failures / n,
            "failure_ci_low": low,
            "failure_ci_high": high,
            "sigma_mean": float(subset["sigma"].mean()),
            "sigma_min": float(subset["sigma"].min()),
            "sigma_max": float(subset["sigma"].max()),
        })
    return pd.DataFrame(rows)


def plot_quartiles(
    table: pd.DataFrame,
    *,
    method: str,
    source: str,
    out_dir: Path,
) -> None:
    ordered = (
        table.set_index("sigma_quartile")
        .loc[["Q1", "Q2", "Q3", "Q4"]]
        .reset_index()
    )
    y = 100.0 * ordered["failure_rate"].to_numpy()
    low = 100.0 * (
        ordered["failure_rate"] - ordered["failure_ci_low"]
    ).to_numpy()
    high = 100.0 * (
        ordered["failure_ci_high"] - ordered["failure_rate"]
    ).to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(
        ordered["sigma_quartile"],
        y,
        yerr=np.vstack([low, high]),
        capsize=4,
    )
    ax.set_xlabel("Prompt-score dispersion quartile")
    ax.set_ylabel("H1 baseline OOD failure rate (%)")
    ax.set_title(f"{method.upper()} / {source}: sigma vs failure")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{method}_{source}_sigma_quartiles.png",
        dpi=180,
    )
    plt.close(fig)


def semantic_path(source: str) -> Path:
    return (
        data_root()
        / "semantic_labels"
        / f"{source}_image_semantic_labels.csv"
    )


def subgroup_row(
    *,
    source: str,
    grouping: str,
    group: str,
    subset: pd.DataFrame,
    n_leafs: int,
    eligible: bool,
) -> dict:
    sigma = subset["sigma"].to_numpy(dtype=float)
    return {
        "source": source,
        "grouping": grouping,
        "group": group,
        "n_images": int(len(subset)),
        "n_leaf_concepts": int(n_leafs),
        "eligible_primary": bool(eligible),
        "fpr95": float(subset["error"].mean()),
        "sigma_mean": float(np.mean(sigma)),
        "sigma_q75": float(np.quantile(sigma, 0.75)),
        "sigma_q90": float(np.quantile(sigma, 0.90)),
    }


def predefined_subgroups(
    source: str,
    condition: pd.DataFrame,
) -> pd.DataFrame:
    semantic = pd.read_csv(semantic_path(source))
    required = {"relative_path", "leaf_concept", "primary_groups"}
    missing = required - set(semantic.columns)
    if missing:
        raise RuntimeError(
            f"{source} predefined semantic mapping missing {sorted(missing)}"
        )

    merged = condition.merge(
        semantic[["relative_path", "leaf_concept", "primary_groups"]],
        on="relative_path",
        how="left",
        validate="one_to_one",
    )
    cfg = load_yaml(PREDEFINED_CFG)["primary_eligibility"]
    coverage = float(merged["primary_groups"].notna().mean())
    if coverage < float(cfg["minimum_mapping_coverage"]):
        raise RuntimeError(
            f"{source} predefined mapping coverage {coverage:.4f} "
            "below frozen minimum"
        )

    exploded = merged.copy()
    exploded["primary_groups"] = exploded["primary_groups"].fillna("")
    exploded["group"] = exploded["primary_groups"].str.split("|")
    exploded = exploded.explode("group")
    exploded = exploded[
        exploded["group"].astype(str).str.len() > 0
    ].copy()

    rows = []
    for group in sorted(exploded["group"].unique()):
        group_rows = exploded[exploded["group"] == group]
        unique_paths = group_rows["relative_path"].drop_duplicates()
        mask = merged["relative_path"].isin(unique_paths)
        subset = merged.loc[mask]
        n_images = int(len(subset))
        n_leafs = int(
            subset["leaf_concept"].dropna().astype(str).nunique()
        )
        eligible = (
            n_images >= int(cfg["minimum_images_per_group"])
            and n_leafs
            >= int(cfg["minimum_leaf_concepts_per_group"])
        )
        rows.append(
            subgroup_row(
                source=source,
                grouping="predefined",
                group=str(group),
                subset=subset,
                n_leafs=n_leafs,
                eligible=eligible,
            )
        )
    return pd.DataFrame(rows)


def load_leaf_clusters(source: str) -> dict[str, str]:
    path = MAPPING_DIR / f"text_clusters_{source}_minilm.csv"
    if not path.exists():
        raise RuntimeError(f"Missing frozen MiniLM cluster mapping: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    mapping = {
        row["leaf_concept"].strip(): row["text_cluster"].strip()
        for row in rows
    }
    if not mapping:
        raise RuntimeError(f"Empty frozen MiniLM mapping: {path}")
    return mapping


def text_cluster_subgroups(
    source: str,
    condition: pd.DataFrame,
) -> pd.DataFrame:
    semantic = pd.read_csv(semantic_path(source))
    required = {"relative_path", "leaf_concept"}
    missing = required - set(semantic.columns)
    if missing:
        raise RuntimeError(
            f"{source} semantic mapping missing {sorted(missing)}"
        )

    leaf_to_cluster = load_leaf_clusters(source)
    semantic = semantic[["relative_path", "leaf_concept"]].copy()
    semantic["text_cluster"] = semantic["leaf_concept"].map(
        leaf_to_cluster
    )
    merged = condition.merge(
        semantic,
        on="relative_path",
        how="left",
        validate="one_to_one",
    )
    cfg = load_yaml(TEXT_CLUSTER_CFG)["eligibility"]
    coverage = float(merged["text_cluster"].notna().mean())
    if coverage < float(cfg["minimum_mapping_coverage"]):
        raise RuntimeError(
            f"{source} text-cluster coverage {coverage:.4f} "
            "below frozen minimum"
        )

    rows = []
    for group in sorted(merged["text_cluster"].dropna().unique()):
        subset = merged.loc[merged["text_cluster"] == group]
        n_images = int(len(subset))
        n_leafs = int(
            subset["leaf_concept"].dropna().astype(str).nunique()
        )
        eligible = (
            n_images >= int(cfg["minimum_images_per_group"])
            and n_leafs
            >= int(cfg["minimum_leaf_concepts_per_group"])
        )
        rows.append(
            subgroup_row(
                source=source,
                grouping="minilm_kmeans",
                group=str(group),
                subset=subset,
                n_leafs=n_leafs,
                eligible=eligible,
            )
        )
    return pd.DataFrame(rows)


def subgroup_correlations(
    table: pd.DataFrame,
    *,
    method: str,
) -> pd.DataFrame:
    rows = []
    for (source, grouping), part in table.groupby(
        ["source", "grouping"]
    ):
        eligible = part[part["eligible_primary"]].copy()
        row = {
            "method": method,
            "source": source,
            "grouping": grouping,
            "n_eligible_groups": int(len(eligible)),
        }
        for predictor in ["sigma_mean", "sigma_q75", "sigma_q90"]:
            if (
                len(eligible) >= 3
                and eligible[predictor].nunique() > 1
                and eligible["fpr95"].nunique() > 1
            ):
                result = spearmanr(
                    eligible[predictor],
                    eligible["fpr95"],
                )
                rho = float(result.statistic)
                pvalue = float(result.pvalue)
            else:
                rho = np.nan
                pvalue = np.nan
            row[f"{predictor}_spearman_rho"] = rho
            row[f"{predictor}_spearman_p"] = pvalue
        rows.append(row)
    return pd.DataFrame(rows)


def write_readme(
    summary: pd.DataFrame,
    decision: dict,
    out_dir: Path,
) -> None:
    lines = [
        "# H2 prompt-sensitivity analysis",
        "",
        "Primary H2 uses the 8 meaning-preserving templates already frozen in configs/pilot.yaml.",
        "",
        "Failure is not redefined from the prompt ensemble. It is the original H1 baseline",
        "false positive under the original method-specific ImageNet ID95 threshold.",
        "",
        "Primary comparison:",
        "",
        "- M0: error ~ z(mu)",
        "- M1: error ~ z(mu) + z(sigma)",
        "- association support: bootstrap 95% CI for beta_sigma is strictly above 0",
        "- predictive support: 5-fold OOF log-loss improvement M0-M1 has bootstrap 95% CI strictly above 0",
        "- condition support requires both criteria",
        "",
        "| Method | Source | Errors | beta_sigma | OR / 1SD sigma | delta OOF log-loss | delta OOF AUROC | Supported |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in summary.iterrows():
        if not bool(row["eligible"]):
            lines.append(
                f"| {row['method']} | {row['source']} | "
                f"{int(row['n_errors'])} | NA | NA | NA | NA | no (ineligible) |"
            )
            continue
        lines.append(
            f"| {row['method']} | {row['source']} | "
            f"{int(row['n_errors'])} | "
            f"{row['beta_sigma']:.4f} | "
            f"{row['odds_ratio_sigma']:.4f} | "
            f"{row['delta_logloss']:.6f} | "
            f"{row['delta_auc']:.6f} | "
            f"{'yes' if bool(row['condition_supported']) else 'no'} |"
        )

    lines += [
        "",
        f"**H2 decision: {decision['h2_decision']}**",
        "",
        f"- eligible conditions: {decision['eligible_conditions']}",
        f"- supported conditions: {decision['supported_conditions']}",
        f"- required supported conditions: {decision['minimum_supported_conditions']}",
        f"- project gate after H2: {decision['project_gate_after_h2']}",
        "",
        "H3 is intentionally not evaluated by this stage.",
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
        choices=["ViT-B/16", "ViT-B/32"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["mcm", "neglabel"],
        choices=["mcm", "neglabel"],
    )
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(PILOT)
    h2_cfg = cfg["prompt_sensitivity"]
    primary_backbone = h2_cfg.get("primary_backbone", "ViT-B/32")
    if args.backbone != primary_backbone:
        raise RuntimeError(
            "Primary H2 analysis is frozen to "
            f"{primary_backbone}; got {args.backbone}. "
            "Run other backbones only as separately labeled robustness."
        )

    frozen_methods = {"mcm", "neglabel"}
    if set(args.methods) != frozen_methods or len(args.methods) != 2:
        raise RuntimeError(
            "Primary H2 decision requires both frozen methods: "
            "mcm and neglabel. Run partial conditions only with a "
            "separate diagnostic script, not this decision script."
        )

    sources = args.sources or primary_sources()
    frozen_sources = set(primary_sources())
    if set(sources) != frozen_sources or len(sources) != len(frozen_sources):
        raise RuntimeError(
            "Primary H2 decision requires all frozen OOD sources: "
            + ", ".join(primary_sources())
            + ". Run partial conditions only as diagnostics."
        )

    n_bootstrap = (
        int(args.bootstrap)
        if args.bootstrap is not None
        else int(h2_cfg["bootstrap_resamples"])
    )
    min_errors = int(h2_cfg["minimum_errors_per_condition"])
    min_correct = int(h2_cfg["minimum_correct_per_condition"])
    min_supported = int(h2_cfg["minimum_supported_conditions"])
    folds = int(h2_cfg["cv_folds"])

    out_dir = (
        ROOT
        / "results"
        / "h2_prompt"
        / backbone_slug(args.backbone)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    condition_rows = []
    quartile_tables = []
    subgroup_tables = []
    correlation_tables = []

    for method in args.methods:
        for source in sources:
            print(
                f"\n=== H2: {method} / {args.backbone} / {source} ==="
            )
            frame, threshold, baseline_fpr = merge_condition(
                method,
                args.backbone,
                source,
            )
            error = frame["error"].to_numpy(dtype=np.int32)
            mu = frame["mu"].to_numpy(dtype=float)
            sigma = frame["sigma"].to_numpy(dtype=float)
            n_errors = int(error.sum())
            n_correct = int(len(error) - n_errors)
            eligible = (
                n_errors >= min_errors
                and n_correct >= min_correct
            )

            row = {
                "method": method,
                "backbone": args.backbone,
                "source": source,
                "n_ood": int(len(frame)),
                "n_errors": n_errors,
                "n_correct": n_correct,
                "eligible": eligible,
                "h1_id95_threshold": threshold,
                "h1_baseline_fpr95": baseline_fpr,
                "sigma_mean_correct": float(
                    frame.loc[frame["error"] == 0, "sigma"].mean()
                ),
                "sigma_mean_error": (
                    float(
                        frame.loc[frame["error"] == 1, "sigma"].mean()
                    )
                    if n_errors > 0
                    else np.nan
                ),
            }

            if eligible:
                association = fit_association(
                    error,
                    mu,
                    sigma,
                    n_bootstrap=n_bootstrap,
                    confidence=args.confidence,
                    seed=args.seed,
                )
                predictive = compare_predictive_models(
                    error,
                    mu,
                    sigma,
                    folds=folds,
                    n_bootstrap=n_bootstrap,
                    confidence=args.confidence,
                    seed=args.seed,
                )
                sigma_auc = float(roc_auc_score(error, sigma))
                association_supported = (
                    association.beta_sigma_ci_low > 0.0
                )
                predictive_supported = (
                    predictive.delta_logloss_ci_low > 0.0
                )
                condition_supported = (
                    association_supported and predictive_supported
                )

                row.update({
                    "beta_mu": association.beta_mu,
                    "beta_sigma": association.beta_sigma,
                    "odds_ratio_sigma": association.odds_ratio_sigma,
                    "beta_sigma_ci_low": association.beta_sigma_ci_low,
                    "beta_sigma_ci_high": association.beta_sigma_ci_high,
                    "logloss_m0": predictive.logloss_m0,
                    "logloss_m1": predictive.logloss_m1,
                    "delta_logloss": predictive.delta_logloss,
                    "delta_logloss_ci_low": predictive.delta_logloss_ci_low,
                    "delta_logloss_ci_high": predictive.delta_logloss_ci_high,
                    "auc_m0": predictive.auc_m0,
                    "auc_m1": predictive.auc_m1,
                    "delta_auc": predictive.delta_auc,
                    "delta_auc_ci_low": predictive.delta_auc_ci_low,
                    "delta_auc_ci_high": predictive.delta_auc_ci_high,
                    "sigma_error_auc": sigma_auc,
                    "association_supported": association_supported,
                    "predictive_supported": predictive_supported,
                    "condition_supported": condition_supported,
                })
            else:
                for name in [
                    "beta_mu",
                    "beta_sigma",
                    "odds_ratio_sigma",
                    "beta_sigma_ci_low",
                    "beta_sigma_ci_high",
                    "logloss_m0",
                    "logloss_m1",
                    "delta_logloss",
                    "delta_logloss_ci_low",
                    "delta_logloss_ci_high",
                    "auc_m0",
                    "auc_m1",
                    "delta_auc",
                    "delta_auc_ci_low",
                    "delta_auc_ci_high",
                    "sigma_error_auc",
                ]:
                    row[name] = np.nan
                row.update({
                    "association_supported": False,
                    "predictive_supported": False,
                    "condition_supported": False,
                })

            condition_rows.append(row)

            quartiles = sigma_quartiles(
                frame,
                method=method,
                source=source,
                confidence=args.confidence,
            )
            quartile_tables.append(quartiles)
            plot_quartiles(
                quartiles,
                method=method,
                source=source,
                out_dir=out_dir,
            )

            predefined = predefined_subgroups(source, frame)
            predefined.insert(0, "method", method)
            text_clusters = text_cluster_subgroups(source, frame)
            text_clusters.insert(0, "method", method)
            combined = pd.concat(
                [predefined, text_clusters],
                ignore_index=True,
            )
            subgroup_tables.append(combined)
            correlation_tables.append(
                subgroup_correlations(combined, method=method)
            )

            if eligible:
                print(
                    "errors={}/{} | beta_sigma={:.4f} "
                    "CI=[{:.4f},{:.4f}] | "
                    "delta_logloss={:.6f} "
                    "CI=[{:.6f},{:.6f}] | support={}".format(
                        n_errors,
                        len(frame),
                        row["beta_sigma"],
                        row["beta_sigma_ci_low"],
                        row["beta_sigma_ci_high"],
                        row["delta_logloss"],
                        row["delta_logloss_ci_low"],
                        row["delta_logloss_ci_high"],
                        row["condition_supported"],
                    )
                )
            else:
                print(
                    f"ineligible: errors={n_errors}, correct={n_correct}"
                )

    summary = pd.DataFrame(condition_rows)
    quartile_table = pd.concat(quartile_tables, ignore_index=True)
    subgroup_table = pd.concat(subgroup_tables, ignore_index=True)
    correlation_table = pd.concat(correlation_tables, ignore_index=True)

    summary.to_csv(out_dir / "condition_summary.csv", index=False)
    quartile_table.to_csv(out_dir / "sigma_quartiles.csv", index=False)
    subgroup_table.to_csv(
        out_dir / "subgroup_sigma_vs_fpr95.csv",
        index=False,
    )
    correlation_table.to_csv(
        out_dir / "subgroup_correlations.csv",
        index=False,
    )

    eligible_count = int(summary["eligible"].sum())
    supported_count = int(
        summary.loc[
            summary["eligible"],
            "condition_supported",
        ].sum()
    )
    expected_conditions = len(args.methods) * len(sources)

    if eligible_count < expected_conditions:
        h2_decision = "INSUFFICIENT_ELIGIBLE_CONDITIONS"
        project_gate = "UNRESOLVED"
    elif supported_count >= min_supported:
        h2_decision = "PASS"
        project_gate = "GO"
    else:
        h2_decision = "FAIL"
        project_gate = "CONDITIONAL_GO"

    decision = {
        "stage": "H2_prompt_sensitivity",
        "date_frozen": "2026-09-25",
        "backbone": args.backbone,
        "primary_sources": sources,
        "methods": args.methods,
        "prompt_count": int(cfg["prompts"]["num_templates"]),
        "error_definition": (
            "original_H1_baseline_single_prompt_score_at_"
            "original_method_specific_ImageNet_ID95"
        ),
        "association_model": "error ~ z(mu) + z(sigma)",
        "predictive_comparison": (
            "5fold_OOF_logloss_M0(error~z(mu))_vs_"
            "M1(error~z(mu)+z(sigma))"
        ),
        "condition_support_rule": {
            "association": "bootstrap_95CI_lower(beta_sigma) > 0",
            "prediction": (
                "paired_bootstrap_95CI_lower("
                "OOF_logloss_M0_minus_M1) > 0"
            ),
            "requires_both": True,
        },
        "minimum_errors_per_condition": min_errors,
        "minimum_correct_per_condition": min_correct,
        "minimum_supported_conditions": min_supported,
        "eligible_conditions": eligible_count,
        "supported_conditions": supported_count,
        "expected_conditions": expected_conditions,
        "bootstrap_resamples": n_bootstrap,
        "confidence": args.confidence,
        "cv_folds": folds,
        "seed": args.seed,
        "h2_decision": h2_decision,
        "project_gate_after_h2": project_gate,
        "h3_evaluated": False,
    }
    (out_dir / "h2_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_readme(summary, decision, out_dir)

    print(
        f"\nH2 decision: {h2_decision} | "
        f"supported {supported_count}/{eligible_count} eligible conditions"
    )
    print(f"H2 outputs: {out_dir}")
    print(
        "H3 has not been evaluated. Do not start H3 until "
        "this H2 decision is reviewed."
    )


if __name__ == "__main__":
    main()
