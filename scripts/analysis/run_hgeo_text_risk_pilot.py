#!/usr/bin/env python3
"""Exploratory SUN text-risk pilot. Never train or re-score an image.

Run from an existing project checkout with its local baseline score CSVs.
Existing H1/H2 files, thresholds, decisions and negative labels are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SOURCE = "sun"
METHODS = ("mcm", "neglabel")
PROMPT = "a photo of a {}"
N_BOOT = 2000
SEED = 42
MIN_IMAGES = 20
MIN_CONCEPTS = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_frame(path: Path, required: set[str], key: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing local input: {path}")
    frame = pd.read_csv(path)
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"{path}: empty table or missing columns {sorted(required)}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{path}: missing values in required columns")
    if frame[key].duplicated().any():
        raise ValueError(f"{path}: duplicate {key}")
    if not frame[key].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError(f"{path}: invalid {key}")
    if "score" in required:
        frame["score"] = pd.to_numeric(frame["score"], errors="raise")
        if not np.isfinite(frame["score"].to_numpy(float)).all():
            raise ValueError(f"{path}: non-finite score")
    return frame


def collect_inputs(backbone: str) -> tuple[dict, list[str], dict]:
    data = Path(os.environ.get("CLIP_OOD_DATA_ROOT", ROOT / "data")).expanduser().resolve()
    slug = backbone.replace("/", "-")
    paths = {
        "concept_names": ROOT / "configs/subgroups/mappings/sun_mos50_hierarchy.csv",
        "semantic_mapping": data / "semantic_labels/sun_image_semantic_labels.csv",
    }
    for method in METHODS:
        for dataset in ("imagenet", SOURCE):
            paths[f"{method}_{dataset}"] = (
                ROOT / "results/raw/reproduction" / slug / method / f"{dataset}.csv"
            )
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required local files are missing:\n" + "\n".join(missing)
            + "\nUse the existing experiment checkout, not a fresh clone. "
            "Set CLIP_OOD_DATA_ROOT if data is stored elsewhere. "
            "This pilot does not regenerate baseline scores."
        )
    names = read_frame(paths["concept_names"], {"leaf_concept"}, "leaf_concept")
    semantic = read_frame(
        paths["semantic_mapping"], {"relative_path", "leaf_concept"}, "relative_path"
    )[["relative_path", "leaf_concept"]]
    leaves = sorted(names["leaf_concept"].tolist())
    frames = {"semantic": semantic}
    for method in METHODS:
        for dataset in ("imagenet", SOURCE):
            key = f"{method}_{dataset}"
            frames[key] = read_frame(paths[key], {"relative_path", "score"}, "relative_path")
    for dataset in ("imagenet", SOURCE):
        if set(frames[f"mcm_{dataset}"].relative_path) != set(frames[f"neglabel_{dataset}"].relative_path):
            raise ValueError(f"Methods have different {dataset} sample sets")
    joined = frames["mcm_sun"][["relative_path"]].merge(
        semantic, on="relative_path", how="left", validate="one_to_one"
    )
    if joined.leaf_concept.isna().any() or set(joined.leaf_concept) != set(leaves):
        raise ValueError("Incomplete SUN mapping or concept set differs from frozen name table")
    provenance = {k: {"path": str(p), "sha256": sha256(p)} for k, p in paths.items()}
    return frames, leaves, provenance


def concept_errors(scores: pd.DataFrame, semantic: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    if scores.relative_path.duplicated().any() or semantic.relative_path.duplicated().any():
        raise ValueError("Duplicate sample path")
    merged = scores[["relative_path", "score"]].merge(
        semantic[["relative_path", "leaf_concept"]],
        on="relative_path", how="left", validate="one_to_one",
    )
    if merged.leaf_concept.isna().any() or merged.empty:
        raise ValueError("Missing leaf concept after score/mapping join")
    if not np.isfinite(merged.score.to_numpy(float)).all():
        raise ValueError("Non-finite OOD score")
    merged["failure"] = (merged.score >= threshold).astype(int)
    result = merged.groupby("leaf_concept", sort=True)["failure"].agg(n="size", errors="sum").reset_index()
    result["fpr95"] = result.errors / result.n
    result["eligible"] = result.n >= MIN_IMAGES
    return result


def cosine_geometry(concepts: np.ndarray, positives: np.ndarray) -> dict[str, np.ndarray]:
    if concepts.ndim != 2 or positives.ndim != 2 or concepts.shape[1] != positives.shape[1]:
        raise ValueError("Feature matrices must be 2-D with equal embedding dimensions")
    if not len(concepts) or not len(positives):
        raise ValueError("Feature matrices must be non-empty")
    for arr in (concepts, positives):
        if not np.isfinite(arr).all() or not np.allclose(np.linalg.norm(arr, axis=1), 1, atol=1e-4):
            raise ValueError("Expected finite L2-normalized embeddings")
    sim = concepts @ positives.T
    maximum = sim.max(axis=1)
    return {"max_id_cosine": maximum, "peak": maximum - sim.mean(axis=1),
            "id_q95": np.quantile(sim, .95, axis=1)}


def rank_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlations over the final axis; constants return NaN."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim not in (1, 2) or x.shape[-1] < 2:
        raise ValueError("Expected matching vectors or row matrices with at least two concepts")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Correlation inputs must be finite")
    rx, ry = rankdata(x, axis=-1), rankdata(y, axis=-1)
    rx = rx - rx.mean(axis=-1, keepdims=True)
    ry = ry - ry.mean(axis=-1, keepdims=True)
    numerator = (rx * ry).sum(axis=-1)
    denominator = np.sqrt((rx * rx).sum(axis=-1) * (ry * ry).sum(axis=-1))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def summarize_ranks(table: pd.DataFrame, predictors: list[str], n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    if n_boot < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    df = table.loc[table.eligible].sort_values("leaf_concept").reset_index(drop=True)
    if len(df) < MIN_CONCEPTS:
        raise ValueError(f"Need at least {MIN_CONCEPTS} eligible concepts, found {len(df)}")
    y = df.fpr95.to_numpy(float)
    baseline = df.max_id_cosine.to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(df), size=(n_boot, len(df)))
    baseline_rho = float(rank_corr(baseline, y))
    baseline_boot = rank_corr(baseline[idx], y[idx])
    rows = []
    for predictor in predictors:
        x = df[predictor].to_numpy(float)
        rho = float(rank_corr(x, y))
        boot = rank_corr(x[idx], y[idx])
        valid = np.isfinite(boot)
        usable = np.isfinite(rho) and valid.sum() >= .95 * n_boot
        low, high = np.quantile(boot[valid], [.025, .975]) if usable else (np.nan, np.nan)
        paired = np.isfinite(boot) & np.isfinite(baseline_boot)
        delta_ok = usable and np.isfinite(baseline_rho) and paired.sum() >= .95 * n_boot
        dlow, dhigh = np.quantile((boot - baseline_boot)[paired], [.025, .975]) if delta_ok else (np.nan, np.nan)
        status = "UNDEFINED_OR_UNSTABLE"
        if usable:
            status = "PROMISING_DESCRIPTIVE_SIGNAL" if rho >= .3 and low > 0 else "NO_CLEAR_POSITIVE_SIGNAL"
        rows.append({"predictor": predictor, "role": "main_exploratory" if predictor == "text_proxy_score" else "comparison",
                     "n_concepts": len(df), "rho": rho, "ci_low": low, "ci_high": high,
                     "delta_rho_vs_max_id": rho - baseline_rho, "delta_ci_low": dlow, "delta_ci_high": dhigh,
                     "valid_bootstrap": int(valid.sum()), "status": status})
    return pd.DataFrame(rows)


def compute_text_risks(backbone: str, leaves: list[str], gpu: int) -> tuple[dict, dict]:
    # Import heavyweight dependencies only after local input validation.
    import torch
    from baselines.common import load_clip, encode_texts, verify_reference_repos
    from baselines.mcm import prepare_mcm_text, mcm_scores
    from baselines.neglabel import prepare_neglabel_text, neglabel_scores

    if not torch.cuda.is_available() or not 0 <= gpu < torch.cuda.device_count():
        raise RuntimeError("Requested CUDA GPU is unavailable. --check-inputs needs no GPU.")
    verify_reference_repos()
    device = torch.device(f"cuda:{gpu}")
    model, _ = load_clip(backbone, device)
    # Same one fixed template for both methods; no prompt search/ensemble.
    concepts = encode_texts(model, [PROMPT.format(c) for c in leaves], device, batch_size=128)
    features = concepts.cpu().numpy().astype(np.float32)
    mcm_pos = prepare_mcm_text(model, device)
    neg_pos, neg_text, neg_meta = prepare_neglabel_text(model, device, backbone=backbone)
    outputs = {}
    for method, pos in (("mcm", mcm_pos), ("neglabel", neg_pos)):
        geometry = cosine_geometry(features, pos.cpu().numpy())
        outputs[method] = pd.DataFrame({"leaf_concept": leaves, **geometry})
    outputs["mcm"]["text_proxy_score"] = mcm_scores(features, mcm_pos, device, temperature=1.0)
    outputs["neglabel"]["text_proxy_score"] = neglabel_scores(
        features, neg_pos, neg_text, device, ngroup=int(neg_meta["ngroup"]),
        temperature=float(neg_meta["temperature"]), logit_scale=float(neg_meta["logit_scale"]),
    )
    # Respect the same unused-tail removal as the actual NegLabel scorer.
    n_used = len(neg_text) - len(neg_text) % int(neg_meta["ngroup"])
    coverage = (concepts @ neg_text[:n_used].T).max(dim=1).values.cpu().numpy()
    outputs["neglabel"]["negative_coverage"] = coverage
    outputs["neglabel"]["negative_coverage_risk"] = -coverage
    return outputs, neg_meta


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["ViT-B/32", "ViT-B/16"], default="ViT-B/32")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-name", default="sun_pilot_v1")
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    if not args.run_name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in args.run_name):
        raise ValueError("run-name must contain only letters, digits, underscores or hyphens")
    frames, leaves, provenance = collect_inputs(args.backbone)
    print(f"[inputs OK] {args.backbone}; SUN {len(leaves)} concepts; both detector score sets available", flush=True)
    if args.check_inputs:
        print("Local CSV checks only. CLIP weights, reference repos and CUDA have not been checked.")
        return
    out = ROOT / "results/hgeo_text_risk" / args.backbone.replace("/", "-") / args.run_name
    if out.exists():
        raise FileExistsError(f"Will not overwrite {out}. Use a new --run-name for an explicitly recorded rerun.")
    from analysis.h1_predefined import build_fixed_id_reference, realized_id_tpr
    risks, neg_meta = compute_text_risks(args.backbone, leaves, args.gpu)
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", "unknown"
    all_tables, all_stats, method_info = [], [], {}
    for method in METHODS:
        id_values = frames[f"{method}_imagenet"].score.to_numpy(float)
        reference = build_fixed_id_reference(id_values)
        errors = concept_errors(frames[f"{method}_sun"], frames["semantic"], reference.threshold_95)
        table = risks[method].merge(errors, on="leaf_concept", how="left", validate="one_to_one")
        if table.n.isna().any():
            raise ValueError("A named concept is missing its observed FPR")
        predictors = ["text_proxy_score", "max_id_cosine", "peak"] if method == "mcm" else [
            "text_proxy_score", "max_id_cosine", "negative_coverage_risk", "id_q95"]
        stats = summarize_ranks(table, predictors)
        table.insert(0, "method", method)
        stats.insert(0, "method", method)
        main_row = stats.loc[stats.predictor == "text_proxy_score"].iloc[0].to_dict()
        method_info[method] = {"id95_threshold": reference.threshold_95,
                               "realized_id_tpr": realized_id_tpr(id_values, reference.threshold_95),
                               "n_ood": int(table.n.sum()), "aggregate_fpr95": float(table.errors.sum() / table.n.sum()),
                               "eligible_concepts": int(table.eligible.sum()), "main_result": main_row}
        all_tables.append(table)
        all_stats.append(stats)
    metadata = {"stage": "H_geo_text_risk_pilot", "study_status": "EXPLORATORY_ONLY",
                "hypothesis": "Concept-name risk ranks are associated with observed concept image FPR95.",
                "h1_h2_decisions_changed": False, "causal_mining_claim_tested": False,
                "new_images_encoded": 0, "training_steps": 0, "source": SOURCE,
                "backbone": args.backbone, "prompt": PROMPT, "seed": SEED,
                "bootstrap_resamples": N_BOOT, "minimum_images_per_concept": MIN_IMAGES,
                "bootstrap_scope": "paired concept bootstrap; conditional on observed images and fixed ID reference; descriptive, no multiple-testing correction",
                "git_commit": commit, "git_status_before_output": dirty,
                "utc_created": datetime.now(timezone.utc).isoformat(), "inputs": provenance,
                "runner_sha256": sha256(Path(__file__)), "negative_metadata": neg_meta, "methods": method_info,
                "limitations": ["Previously inspected benchmark; not held-out confirmation.",
                                "Text proxy scores are ranking indices, not calibrated image FPR predictions.",
                                "Secondary predictors cannot replace a failed main predictor post hoc.",
                                "No test of novelty, causation, semantic-group selection or H2 rescue."]}
    out.mkdir(parents=True, exist_ok=False)
    pd.concat(all_tables, ignore_index=True).to_csv(out / "concept_results.csv", index=False)
    summary = pd.concat(all_stats, ignore_index=True)
    summary.to_csv(out / "correlation_summary.csv", index=False)
    (out / "pilot_summary.json").write_text(json.dumps(json_safe(metadata), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(summary[["method", "predictor", "rho", "ci_low", "ci_high", "delta_rho_vs_max_id", "status"]].to_string(index=False))
    print(f"[output] {out}\nExploratory only. Existing H1/H2 decisions are unchanged.")


if __name__ == "__main__":
    main()
