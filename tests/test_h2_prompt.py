from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.h2_prompt_sensitivity import (
    compare_predictive_models,
    fit_association,
    wilson_interval,
)
from baselines.neglabel import (
    neglabel_prompt_scores,
    neglabel_scores,
)


def _normalized_tensor(
    generator: torch.Generator,
    rows: int,
    cols: int,
) -> torch.Tensor:
    x = torch.randn(
        rows,
        cols,
        generator=generator,
    )
    return x / x.norm(dim=1, keepdim=True)


def test_neglabel_prompt_matrix_matches_separate_scores():
    device = torch.device("cpu")
    g = torch.Generator().manual_seed(123)

    image = _normalized_tensor(g, 9, 6)
    positive_a = _normalized_tensor(g, 7, 6)
    positive_b = _normalized_tensor(g, 7, 6)
    negative = _normalized_tensor(g, 20, 6)

    matrix = neglabel_prompt_scores(
        image.numpy().astype(np.float32),
        [positive_a, positive_b],
        negative,
        device=device,
        ngroup=4,
        batch_size=4,
    )

    expected_a = neglabel_scores(
        image.numpy().astype(np.float32),
        positive_a,
        negative,
        device=device,
        ngroup=4,
        batch_size=4,
    )
    expected_b = neglabel_scores(
        image.numpy().astype(np.float32),
        positive_b,
        negative,
        device=device,
        ngroup=4,
        batch_size=4,
    )
    expected = np.column_stack(
        [expected_a, expected_b]
    )

    np.testing.assert_allclose(
        matrix,
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_h2_statistics_detect_clear_incremental_sigma_signal():
    rng = np.random.default_rng(42)
    n = 2500
    mu = rng.normal(size=n)
    sigma = rng.lognormal(
        mean=-1.0,
        sigma=0.55,
        size=n,
    )
    latent = (
        -0.5
        + 0.15 * mu
        + 3.0 * (sigma - sigma.mean())
        + rng.normal(scale=0.55, size=n)
    )
    error = (
        latent
        > np.quantile(latent, 0.72)
    ).astype(np.int32)

    association = fit_association(
        error,
        mu,
        sigma,
        n_bootstrap=150,
        seed=7,
    )
    predictive = compare_predictive_models(
        error,
        mu,
        sigma,
        folds=5,
        n_bootstrap=150,
        seed=7,
    )

    assert association.beta_sigma > 0.0
    assert association.beta_sigma_ci_low > 0.0
    assert predictive.delta_logloss > 0.0
    assert predictive.delta_logloss_ci_low > 0.0


def test_wilson_interval_contains_observed_fraction():
    low, high = wilson_interval(25, 100)
    assert 0.0 <= low < 0.25 < high <= 1.0
