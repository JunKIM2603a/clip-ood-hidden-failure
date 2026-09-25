from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass(frozen=True)
class SimilarityRiskResult:
    probabilities: np.ndarray
    auc: float
    logloss: float


@torch.no_grad()
def nearest_cosine_similarity(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Return max cosine similarity to the reference set for every query.

    Feature caches are expected to be L2 normalized. The function validates
    that assumption to avoid silently turning the dot product into another
    quantity.
    """
    query = np.asarray(query_features, dtype=np.float32)
    reference = np.asarray(reference_features, dtype=np.float32)
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError("query_features and reference_features must be 2-D")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query/reference feature dimensions differ")
    if len(query) == 0 or len(reference) == 0:
        raise ValueError("query/reference features must be non-empty")

    q_norm = np.linalg.norm(query, axis=1)
    r_norm = np.linalg.norm(reference, axis=1)
    if not np.allclose(q_norm, 1.0, rtol=1e-3, atol=1e-3):
        raise ValueError("query features are not L2 normalized")
    if not np.allclose(r_norm, 1.0, rtol=1e-3, atol=1e-3):
        raise ValueError("reference features are not L2 normalized")

    ref = torch.from_numpy(reference).to(device)
    out = np.empty(len(query), dtype=np.float32)
    for start in range(0, len(query), batch_size):
        feat = torch.from_numpy(query[start : start + batch_size]).to(device)
        sim = feat @ ref.T
        out[start : start + len(feat)] = (
            sim.max(dim=1).values.cpu().numpy().astype(np.float32, copy=False)
        )
    return out


def _risk_pipeline() -> Pipeline:
    return Pipeline([
        (
            "poly",
            PolynomialFeatures(
                degree=2,
                include_bias=False,
            ),
        ),
        ("scale", StandardScaler()),
        (
            "logistic",
            LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=2000,
            ),
        ),
    ])


def oof_similarity_risk(
    error: np.ndarray,
    visual_similarity: np.ndarray,
    semantic_similarity: np.ndarray,
    *,
    folds: int = 5,
    seed: int = 42,
) -> SimilarityRiskResult:
    """Cross-fitted failure risk using only frozen similarity proxies."""
    y = np.asarray(error, dtype=np.int32).reshape(-1)
    visual = np.asarray(visual_similarity, dtype=np.float64).reshape(-1)
    semantic = np.asarray(semantic_similarity, dtype=np.float64).reshape(-1)
    if not (len(y) == len(visual) == len(semantic)):
        raise ValueError("error and similarity arrays must have equal length")
    if np.unique(y).size != 2:
        raise ValueError("error must contain both 0 and 1")
    if not np.isfinite(visual).all() or not np.isfinite(semantic).all():
        raise ValueError("similarity predictors must be finite")

    x = np.column_stack([visual, semantic])
    probabilities = np.empty(len(y), dtype=np.float64)
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    for train, test in cv.split(x, y):
        model = _risk_pipeline()
        model.fit(x[train], y[train])
        probabilities[test] = model.predict_proba(x[test])[:, 1]

    return SimilarityRiskResult(
        probabilities=probabilities,
        auc=float(roc_auc_score(y, probabilities)),
        logloss=float(log_loss(y, probabilities, labels=[0, 1])),
    )
