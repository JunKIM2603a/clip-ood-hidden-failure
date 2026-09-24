from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.h1_predefined import (
    auc_contributions,
    bootstrap_mean_ci,
    build_fixed_id_reference,
    conditional_metrics,
    realized_id_tpr,
)


def test_fixed_id95_threshold_is_shared_order_statistic():
    id_scores = np.arange(100, dtype=np.float64)
    ref = build_fixed_id_reference(id_scores)
    assert ref.threshold_95 == 5.0
    assert realized_id_tpr(id_scores, ref.threshold_95) == 0.95


def test_auc_contributions_match_sklearn_pairwise_auroc():
    id_scores = np.asarray([0.91, 0.83, 0.72, 0.61, 0.55], dtype=np.float64)
    ood_scores = np.asarray([0.70, 0.52, 0.25, 0.83], dtype=np.float64)
    ref = build_fixed_id_reference(id_scores)
    expected = roc_auc_score(
        np.r_[np.ones(len(id_scores)), np.zeros(len(ood_scores))],
        np.r_[id_scores, ood_scores],
    )
    actual = auc_contributions(ref, ood_scores).mean()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_subgroup_fpr_uses_same_id_threshold():
    id_scores = np.arange(100, dtype=np.float64)
    ref = build_fixed_id_reference(id_scores)

    source = np.asarray([0, 1, 4, 5, 6, 90], dtype=np.float64)
    group_a = np.asarray([0, 1, 4], dtype=np.float64)
    group_b = np.asarray([5, 6, 90], dtype=np.float64)

    assert conditional_metrics(ref, source)["fpr95"] == 0.5
    assert conditional_metrics(ref, group_a)["fpr95"] == 0.0
    assert conditional_metrics(ref, group_b)["fpr95"] == 1.0


def test_bootstrap_is_deterministic_for_fixed_seed():
    values = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0])
    one = bootstrap_mean_ci(values, n_resamples=200, seed=42)
    two = bootstrap_mean_ci(values, n_resamples=200, seed=42)
    assert one == two
