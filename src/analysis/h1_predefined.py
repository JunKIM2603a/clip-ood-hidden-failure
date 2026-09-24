from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedIDReference:
    threshold_95: float
    sorted_id_scores: np.ndarray


def build_fixed_id_reference(id_scores: np.ndarray) -> FixedIDReference:
    """Freeze the ID reference used by every OOD subgroup.

    Scores are oriented so larger means more ID-like. The threshold is the
    5th percentile order statistic using NumPy's "higher" rule. With unique
    scores this accepts exactly 95% of the ID reference; ties can only make
    the realized ID TPR slightly larger.
    """
    scores = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("ID scores must be finite and non-empty")
    threshold = float(np.quantile(scores, 0.05, method="higher"))
    return FixedIDReference(
        threshold_95=threshold,
        sorted_id_scores=np.sort(scores),
    )


def realized_id_tpr(id_scores: np.ndarray, threshold: float) -> float:
    scores = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    return float(np.mean(scores >= threshold))


def auc_contributions(
    reference: FixedIDReference,
    ood_scores: np.ndarray,
) -> np.ndarray:
    """Per-OOD-sample AUROC contributions against one fixed ID reference.

    AUROC with ID as the positive class is:
      P(score_ID > score_OOD) + 0.5 P(tie)

    Holding the full ID validation set fixed lets each OOD sample contribute
    one scalar. Subgroup AUROC is simply the mean contribution of its samples.
    """
    ood = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    sorted_id = reference.sorted_id_scores
    n_id = sorted_id.size
    left = np.searchsorted(sorted_id, ood, side="left")
    right = np.searchsorted(sorted_id, ood, side="right")
    greater = n_id - right
    equal = right - left
    return (greater + 0.5 * equal) / n_id


def fpr95_contributions(
    reference: FixedIDReference,
    ood_scores: np.ndarray,
) -> np.ndarray:
    """Per-OOD-sample false-positive indicators at the fixed ID95 threshold."""
    ood = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    return (ood >= reference.threshold_95).astype(np.float64)


def conditional_metrics(
    reference: FixedIDReference,
    ood_scores: np.ndarray,
) -> dict[str, float]:
    auc_c = auc_contributions(reference, ood_scores)
    fpr_c = fpr95_contributions(reference, ood_scores)
    return {
        "auroc": float(np.mean(auc_c)),
        "fpr95": float(np.mean(fpr_c)),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    chunk_size: int = 100,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty array")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    done = 0
    while done < n_resamples:
        take = min(chunk_size, n_resamples - done)
        idx = rng.integers(0, values.size, size=(take, values.size))
        means[done : done + take] = values[idx].mean(axis=1)
        done += take

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def bootstrap_group_metric_cis(
    reference: FixedIDReference,
    group_scores: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    auc_c = auc_contributions(reference, group_scores)
    fpr_c = fpr95_contributions(reference, group_scores)
    return {
        "auroc": bootstrap_mean_ci(
            auc_c,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        ),
        "fpr95": bootstrap_mean_ci(
            fpr_c,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed + 1,
        ),
    }


def paired_source_bootstrap_gap_ci(
    source_values: np.ndarray,
    group_mask: np.ndarray,
    *,
    gap_direction: str,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    chunk_size: int = 50,
) -> tuple[float, float]:
    """Bootstrap aggregate-to-group gap while preserving source dependence.

    A source-level sample bootstrap is used. Group membership travels with each
    sampled OOD image. This keeps the subgroup nested inside the same aggregate
    OOD source instead of bootstrapping aggregate and subgroup independently.

    gap_direction:
      "group_minus_aggregate" for FPR95
      "aggregate_minus_group" for AUROC
    """
    values = np.asarray(source_values, dtype=np.float64).reshape(-1)
    mask = np.asarray(group_mask, dtype=bool).reshape(-1)
    if values.shape != mask.shape:
        raise ValueError("source_values and group_mask must have equal length")
    if not np.any(mask):
        raise ValueError("group_mask selects no samples")
    if gap_direction not in {"group_minus_aggregate", "aggregate_minus_group"}:
        raise ValueError(gap_direction)

    rng = np.random.default_rng(seed)
    gaps = np.empty(n_resamples, dtype=np.float64)
    n = values.size
    done = 0
    while done < n_resamples:
        take = min(chunk_size, n_resamples - done)
        idx = rng.integers(0, n, size=(take, n))
        sampled_values = values[idx]
        sampled_mask = mask[idx]
        agg = sampled_values.mean(axis=1)

        numerator = (sampled_values * sampled_mask).sum(axis=1)
        denominator = sampled_mask.sum(axis=1)
        if np.any(denominator == 0):
            raise RuntimeError(
                "Bootstrap produced an empty group; primary groups should be "
                "large enough that this is effectively impossible."
            )
        group = numerator / denominator

        if gap_direction == "group_minus_aggregate":
            gaps[done : done + take] = group - agg
        else:
            gaps[done : done + take] = agg - group
        done += take

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(gaps, [alpha, 1.0 - alpha])
    return float(low), float(high)
