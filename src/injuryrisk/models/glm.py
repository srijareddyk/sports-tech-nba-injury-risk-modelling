"""Small numpy estimators with player-clustered standard errors.

scikit-learn gives point estimates only, and statsmodels is not a dependency.
Effect sizes need honest uncertainty, and with repeated rows per player that
means cluster-robust (sandwich) covariance, clustered on player.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from math import erf, sqrt


def _cluster_meat(scores: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, int]:
    codes, uniq = pd.factorize(clusters)
    summed = np.zeros((len(uniq), scores.shape[1]))
    np.add.at(summed, codes, scores)
    return summed.T @ summed, len(uniq)


def _pvalue(z: np.ndarray) -> np.ndarray:
    return np.array([2 * (1 - 0.5 * (1 + erf(abs(v) / sqrt(2)))) for v in z])


def logit_irls(X: np.ndarray, y: np.ndarray, clusters: np.ndarray, names: list[str],
               max_iter: int = 50, tol: float = 1e-9) -> pd.DataFrame:
    """Logistic regression by IRLS. ``X`` must include an intercept column."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.mean() / (1 - y.mean()))
    for _ in range(max_iter):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        hess = X.T @ (X * w[:, None])
        step = np.linalg.solve(hess, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    p = 1 / (1 + np.exp(-(X @ beta)))
    bread = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    meat, g = _cluster_meat(X * (y - p)[:, None], clusters)
    cov = bread @ meat @ bread * g / (g - 1)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    return pd.DataFrame({"term": names, "coef": beta, "se": se, "z": z, "p_value": _pvalue(z),
                         "odds_ratio": np.exp(beta), "or_lo": np.exp(beta - 1.96 * se),
                         "or_hi": np.exp(beta + 1.96 * se)})


def within_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray, clusters: np.ndarray,
               names: list[str]) -> pd.DataFrame:
    """Linear probability model with group fixed effects (within transform)."""
    df = pd.DataFrame(np.column_stack([y, X]))
    demeaned = df - df.groupby(np.asarray(groups)).transform("mean")
    yd = demeaned.iloc[:, 0].to_numpy()
    Xd = demeaned.iloc[:, 1:].to_numpy()
    keep = Xd.std(axis=0) > 0
    Xd = Xd[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    xtx_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = xtx_inv @ Xd.T @ yd
    u = yd - Xd @ beta
    meat, g = _cluster_meat(Xd * u[:, None], clusters)
    n, k = Xd.shape
    n_groups = len(np.unique(groups))
    cov = xtx_inv @ meat @ xtx_inv * (g / (g - 1)) * ((n - 1) / (n - k - n_groups))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = beta / se
    return pd.DataFrame({"term": names, "coef": beta, "se": se, "z": z, "p_value": _pvalue(z),
                         "ci_lo": beta - 1.96 * se, "ci_hi": beta + 1.96 * se})
