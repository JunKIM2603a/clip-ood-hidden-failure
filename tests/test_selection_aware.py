from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.selection_aware import selection_aware_worst_gap


def test_selection_aware_fpr_finds_persistent_bad_group():
    rng = np.random.default_rng(123)
    n = 2400
    values = rng.binomial(1, 0.20, size=n).astype(np.float64)

    group_a = np.zeros(n, dtype=bool)
    group_a[:800] = True
    values[:800] = rng.binomial(1, 0.48, size=800)

    group_b = np.zeros(n, dtype=bool)
    group_b[800:1600] = True

    group_c = np.zeros(n, dtype=bool)
    group_c[1600:] = True

    result = selection_aware_worst_gap(
        values,
        {
            "bad_group": group_a,
            "other_1": group_b,
            "other_2": group_c,
        },
        gap_direction="group_minus_aggregate",
        n_resamples=400,
        seed=42,
    )

    assert result.point_worst_group == "bad_group"
    assert result.point_gap > 0.10
    assert result.gap_ci_low > 0.0
    assert result.selected_counts["bad_group"] > 300


def test_selection_aware_auroc_uses_low_group_as_worst():
    rng = np.random.default_rng(456)
    n = 1800
    values = np.clip(
        rng.normal(0.90, 0.03, size=n),
        0.0,
        1.0,
    )

    low = np.zeros(n, dtype=bool)
    low[:600] = True
    values[:600] = np.clip(
        rng.normal(0.72, 0.03, size=600),
        0.0,
        1.0,
    )
    mid = np.zeros(n, dtype=bool)
    mid[600:1200] = True
    high = np.zeros(n, dtype=bool)
    high[1200:] = True

    result = selection_aware_worst_gap(
        values,
        {
            "low_auc": low,
            "mid": mid,
            "high": high,
        },
        gap_direction="aggregate_minus_group",
        n_resamples=300,
        seed=7,
    )

    assert result.point_worst_group == "low_auc"
    assert result.point_gap > 0.05
    assert result.gap_ci_low > 0.0
