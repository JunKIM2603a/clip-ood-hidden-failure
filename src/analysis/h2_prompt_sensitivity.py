from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .h1_predefined import bootstrap_mean_ci


@dataclass(frozen=True)
class LogisticAssociation:
    beta_mu: float
    beta_sigma: float
    odds_ratio_sigma: float
    beta_sigma_ci_low: float
    beta_sigma_ci_high: float


@dataclass(frozen=True)
class PredictiveComparison:
    logloss_m0: float
    logloss_m1: float
    delta_logloss: float
    delta_logloss_ci_low: float
    delta_logloss_ci_high: float
    auc_m0: float
    auc_m1: float
    delta_auc: float
    delta_auc_ci_low: float
    delta_auc_ci_high: float
    oof_m0: np.ndarray
    oof_m1: np.ndarray


def _validate_binary(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    values = np.unique(y)
    if not np.array_equal(values, np.array([0, 1], dtype=np.int32)):
        raise ValueError(
            f"Expected both binary labels 0 and 1, got {values.tolist()}"
        )
    return y


def _zscore(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("Predictor has zero/non-finite standard deviation")
    return (values - mean) / std, mean, std


def _new_logistic() -> LogisticRegression:
    return LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
    )


def fit_association(
    error: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> LogisticAssociation:
    """Fit error ~ z(mu) + z(sigma) and bootstrap beta_sigma.

    Predictors are standardized once on the full condition so beta_sigma is
    interpreted per one original-condition SD of prompt dispersion.
    """
    y = _validate_binary(error)
    z_mu, _, _ = _zscore(mu)
    z_sigma, _, _ = _zscore(sigma)
    x = np.column_stack([z_mu, z_sigma])

    model = _new_logistic()
    model.fit(x, y)
    beta_mu = float(model.coef_[0, 0])
    beta_sigma = float(model.coef_[0, 1])

    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    n = len(y)
    done = 0
    attempts = 0
    max_attempts = n_bootstrap * 10
    while done < n_bootstrap:
        if attempts >= max_attempts:
            raise RuntimeError("Too many degenerate bootstrap samples")
        attempts += 1
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if np.unique(y_b).size < 2:
            continue
        m = _new_logistic()
        m.fit(x[idx], y_b)
        boot[done] = float(m.coef_[0, 1])
        done += 1

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot, [alpha, 1.0 - alpha])
    return LogisticAssociation(
        beta_mu=beta_mu,
        beta_sigma=beta_sigma,
        odds_ratio_sigma=float(np.exp(beta_sigma)),
        beta_sigma_ci_low=float(low),
        beta_sigma_ci_high=float(high),
    )


def _oof_probabilities(
    error: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = _validate_binary(error)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    if not (len(y) == len(mu) == len(sigma)):
        raise ValueError("error, mu, sigma must have the same length")

    oof0 = np.empty(len(y), dtype=np.float64)
    oof1 = np.empty(len(y), dtype=np.float64)
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    for train, test in cv.split(mu, y):
        mu_mean = float(mu[train].mean())
        mu_std = float(mu[train].std(ddof=0))
        sigma_mean = float(sigma[train].mean())
        sigma_std = float(sigma[train].std(ddof=0))
        if mu_std <= 0.0 or sigma_std <= 0.0:
            raise ValueError(
                "Zero predictor variance inside a CV training fold"
            )

        z_mu_train = (mu[train] - mu_mean) / mu_std
        z_mu_test = (mu[test] - mu_mean) / mu_std
        z_sigma_train = (sigma[train] - sigma_mean) / sigma_std
        z_sigma_test = (sigma[test] - sigma_mean) / sigma_std

        m0 = _new_logistic()
        m0.fit(z_mu_train[:, None], y[train])
        oof0[test] = m0.predict_proba(z_mu_test[:, None])[:, 1]

        m1 = _new_logistic()
        x1_train = np.column_stack([z_mu_train, z_sigma_train])
        x1_test = np.column_stack([z_mu_test, z_sigma_test])
        m1.fit(x1_train, y[train])
        oof1[test] = m1.predict_proba(x1_test)[:, 1]

    return oof0, oof1


def _binary_log_loss_contributions(
    y: np.ndarray,
    prob: np.ndarray,
) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    p = np.clip(
        np.asarray(prob, dtype=np.float64),
        eps,
        1.0 - eps,
    )
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _bootstrap_auc_delta(
    y: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    values = np.empty(n_bootstrap, dtype=np.float64)
    done = 0
    attempts = 0
    max_attempts = n_bootstrap * 10
    while done < n_bootstrap:
        if attempts >= max_attempts:
            raise RuntimeError("Too many degenerate AUC bootstrap samples")
        attempts += 1
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if np.unique(y_b).size < 2:
            continue
        values[done] = (
            roc_auc_score(y_b, p1[idx])
            - roc_auc_score(y_b, p0[idx])
        )
        done += 1

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return float(low), float(high)


def compare_predictive_models(
    error: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    folds: int = 5,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> PredictiveComparison:
    """Compare M0:error~mu and M1:error~mu+sigma with fixed OOF predictions."""
    y = _validate_binary(error)
    oof0, oof1 = _oof_probabilities(
        y,
        mu,
        sigma,
        folds=folds,
        seed=seed,
    )

    loss0_i = _binary_log_loss_contributions(y, oof0)
    loss1_i = _binary_log_loss_contributions(y, oof1)
    delta_i = loss0_i - loss1_i
    delta_ci = bootstrap_mean_ci(
        delta_i,
        n_resamples=n_bootstrap,
        confidence=confidence,
        seed=seed + 101,
    )

    auc0 = float(roc_auc_score(y, oof0))
    auc1 = float(roc_auc_score(y, oof1))
    auc_ci = _bootstrap_auc_delta(
        y,
        oof0,
        oof1,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed + 202,
    )

    return PredictiveComparison(
        logloss_m0=float(log_loss(y, oof0, labels=[0, 1])),
        logloss_m1=float(log_loss(y, oof1, labels=[0, 1])),
        delta_logloss=float(delta_i.mean()),
        delta_logloss_ci_low=delta_ci[0],
        delta_logloss_ci_high=delta_ci[1],
        auc_m0=auc0,
        auc_m1=auc1,
        delta_auc=auc1 - auc0,
        delta_auc_ci_low=auc_ci[0],
        delta_auc_ci_high=auc_ci[1],
        oof_m0=oof0,
        oof_m1=oof1,
    )


def wilson_interval(
    successes: int,
    n: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    if successes < 0 or successes > n:
        raise ValueError("successes must be in [0,n]")

    if confidence != 0.95:
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + confidence / 2.0))
    else:
        z = 1.959963984540054

    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (
        z
        * np.sqrt(
            p * (1.0 - p) / n
            + z * z / (4.0 * n * n)
        )
        / denom
    )
    return float(center - half), float(center + half)
