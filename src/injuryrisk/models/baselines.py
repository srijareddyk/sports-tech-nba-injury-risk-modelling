"""Mandatory baselines (spec 6.1), reported next to every model.

1. ``base_rate``: the training positive rate for every row.
2. ``history_only``: logistic on days since last injury, prior injury count,
   games missed last season. If the full model can't beat this by more than the
   noise floor, workload adds nothing, and that is a reportable result.
3. ``history_age_role``: + age and role.
4. ``full_logistic``: every feature, linear. Separates "more features" from
   "nonlinear model" when the GBM is compared against (2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

HISTORY_ONLY = ["days_since_last_injury", "prior_injury_count", "missed_lag1"]
HISTORY_AGE_ROLE = HISTORY_ONLY + ["age", "mpg_prior", "mpg_last10_played", "usage_prior"]

#: Long-tailed columns logged before a linear model sees them.
LOG_PREFIXES = ("days_since", "player_days_since", "season_min_cum", "travel_km",
                "missed_career", "games_since_return", "inj_events", "prior_injury_count",
                "last_spell_length")


def log_heavy(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if c.startswith(LOG_PREFIXES):
            X[c] = np.log1p(X[c].clip(lower=0))
    return X


class BaseRate:
    """Predicts the training positive rate."""

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


def logistic(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("log", FunctionTransformer(log_heavy, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=3000)),
    ])
