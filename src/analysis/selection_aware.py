from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectionAwareResult:
    point_aggregate: float
    point_worst: float
    point_gap: float
    point_worst_group: str
    gap_median: float
    gap_ci_low: float
    gap_ci_high: float
    empirical_p_gap_le_zero: float
    selected_counts: dict[str, int]


def selection_aware_worst_gap(
    values: np.ndarray,
    group_masks: dict[str, np.ndarray],
    *,
    gap_direction: str,
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
    chunk_size: int = 25,
) -> SelectionAwareResult:
    """Bootstrap a worst-group gap with group selection inside each resample.

    Values are per-OOD-sample metric contributions under a fixed ID reference.
    For FPR95 they are false-positive indicators. For AUROC they are
    per-OOD-sample AUROC contributions against the fixed full ID set.

    gap_direction:
      group_minus_aggregate: larger group mean is worse (FPR95)
      aggregate_minus_group: smaller group mean is worse (AUROC)
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("values must be finite and non-empty")
    if gap_direction not in {
        "group_minus_aggregate",
        "aggregate_minus_group",
    }:
        raise ValueError(gap_direction)
    if not group_masks:
        raise ValueError("group_masks must not be empty")

    names = list(group_masks)
    masks = []
    for name in names:
        mask = np.asarray(group_masks[name], dtype=bool).reshape(-1)
        if mask.shape != x.shape:
            raise ValueError(f"Mask shape mismatch for group {name}")
        if not np.any(mask):
            raise ValueError(f"Group {name} is empty")
        masks.append(mask)

    aggregate = float(x.mean())
    group_means = np.asarray(
        [float(x[m].mean()) for m in masks],
        dtype=np.float64,
    )

    if gap_direction == "group_minus_aggregate":
        point_idx = int(np.argmax(group_means))
        point_worst = float(group_means[point_idx])
        point_gap = point_worst - aggregate
    else:
        point_idx = int(np.argmin(group_means))
        point_worst = float(group_means[point_idx])
        point_gap = aggregate - point_worst

    rng = np.random.default_rng(seed)
    gaps = np.empty(n_resamples, dtype=np.float64)
    selected = np.zeros(len(names), dtype=np.int64)
    n = len(x)
    done = 0

    while done < n_resamples:
        take = min(chunk_size, n_resamples - done)
        idx = rng.integers(0, n, size=(take, n))
        sampled_values = x[idx]
        aggregate_boot = sampled_values.mean(axis=1)

        group_boot = np.empty(
            (take, len(names)),
            dtype=np.float64,
        )
        for j, mask in enumerate(masks):
            sampled_mask = mask[idx]
            counts = sampled_mask.sum(axis=1)
            if np.any(counts == 0):
                raise RuntimeError(
                    f"Bootstrap produced empty eligible group {names[j]}"
                )
            group_boot[:, j] = (
                (sampled_values * sampled_mask).sum(axis=1)
                / counts
            )

        if gap_direction == "group_minus_aggregate":
            chosen = np.argmax(group_boot, axis=1)
            worst = group_boot[
                np.arange(take),
                chosen,
            ]
            gaps[done : done + take] = worst - aggregate_boot
        else:
            chosen = np.argmin(group_boot, axis=1)
            worst = group_boot[
                np.arange(take),
                chosen,
            ]
            gaps[done : done + take] = aggregate_boot - worst

        selected += np.bincount(
            chosen,
            minlength=len(names),
        )
        done += take

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(
        gaps,
        [alpha, 1.0 - alpha],
    )

    return SelectionAwareResult(
        point_aggregate=aggregate,
        point_worst=point_worst,
        point_gap=point_gap,
        point_worst_group=names[point_idx],
        gap_median=float(np.median(gaps)),
        gap_ci_low=float(low),
        gap_ci_high=float(high),
        empirical_p_gap_le_zero=float(np.mean(gaps <= 0.0)),
        selected_counts={
            name: int(selected[i])
            for i, name in enumerate(names)
        },
    )
