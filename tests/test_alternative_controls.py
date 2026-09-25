from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.alternative_controls import (
    nearest_cosine_similarity,
    oof_similarity_risk,
)


def test_nearest_cosine_similarity_matches_direct_max():
    device = torch.device("cpu")
    reference = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.asarray(
        [
            [1.0, 0.0],
            [0.0, -1.0],
            [np.sqrt(0.5), np.sqrt(0.5)],
        ],
        dtype=np.float32,
    )

    actual = nearest_cosine_similarity(
        query,
        reference,
        device,
        batch_size=2,
    )
    expected = (query @ reference.T).max(axis=1)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_oof_similarity_risk_recovers_similarity_signal():
    rng = np.random.default_rng(17)
    n = 3000
    visual = rng.normal(size=n)
    semantic = rng.normal(size=n)
    latent = (
        1.2 * visual
        + 0.9 * semantic
        + 0.6 * visual * semantic
        + rng.normal(scale=0.8, size=n)
    )
    threshold = np.quantile(latent, 0.70)
    error = (latent >= threshold).astype(np.int32)

    result = oof_similarity_risk(
        error,
        visual,
        semantic,
        folds=5,
        seed=42,
    )

    assert result.probabilities.shape == (n,)
    assert np.all(
        (result.probabilities >= 0.0)
        & (result.probabilities <= 1.0)
    )
    assert result.auc > 0.80
    assert result.logloss < 0.55
