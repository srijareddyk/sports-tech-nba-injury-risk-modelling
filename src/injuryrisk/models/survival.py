"""Discrete-time hazard model (spec 7, option 4).

The H = 1 risk set already has survival structure: one row per game the player
is at risk, ``y = 1`` when a new spell starts next game, and the player leaves
the risk set while out. A logistic regression on it is a discrete-time hazard
model. Time since return enters as a spline (baseline hazard), alongside
rolling load and history as time-varying covariates.

Two uses:

* ``hazard_model``: a predictive pipeline scored in the forward chain like any
  other model.
* ``risk_factor_table``: mutually adjusted odds ratios per 1 SD with
  player-clustered 95% CIs. This is goal 2, "ranked risk factors with honest
  effect sizes". These are **associations**: minutes are endogenous (coaches
  cut minutes for players who look hurt), so a load coefficient is not the
  effect of changing load.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer, StandardScaler

from injuryrisk.models.baselines import log_heavy
from injuryrisk.models.glm import logit_irls

SPLINE_TERMS = ["games_since_return", "age"]
HAZARD_COVARIATES = [
    "min_mean_3", "min_mean_15", "acwr_3_15", "games_last_7d", "b2b", "travel_km", "abs_tz_shift",
    "prior_injury_count", "days_since_last_injury", "inj_events_365d", "prior_major_structure",
    "missed_lag1", "mpg_prior", "usage_prior", "height_in", "bmi", "shots_per_min_10",
]
HAZARD_FEATURES = SPLINE_TERMS + HAZARD_COVARIATES


def _gsr(X: pd.DataFrame) -> pd.DataFrame:
    """Games since return: never-injured players sit at a fixed far value."""
    X = X.copy()
    X["games_since_return"] = np.log1p(X["games_since_return"].fillna(500).clip(0, 500))
    return X


def hazard_model(C: float = 1.0) -> Pipeline:
    spline = Pipeline([
        ("fill", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=5, degree=3)),
    ])
    linear = Pipeline([
        ("log", FunctionTransformer(log_heavy, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer([("spline", spline, SPLINE_TERMS), ("linear", linear, HAZARD_COVARIATES)])
    return Pipeline([
        ("gsr", FunctionTransformer(_gsr, feature_names_out="one-to-one")),
        ("pre", pre),
        ("lr", LogisticRegression(C=C, max_iter=3000)),
    ])


def risk_factor_table(df: pd.DataFrame, label: str, features: list[str] | None = None) -> pd.DataFrame:
    """Adjusted odds ratios per 1 SD (log scale for long-tailed columns)."""
    features = features or HAZARD_FEATURES
    d = df.dropna(subset=[label])
    X = d[features].copy()
    if "games_since_return" in X:
        X["games_since_return"] = X["games_since_return"].fillna(500).clip(0, 500)
    X = log_heavy(X)
    X = X.fillna(X.median())
    sd = X.std().replace(0, 1)
    Z = (X - X.mean()) / sd
    design = np.column_stack([np.ones(len(Z)), Z.to_numpy()])
    tab = logit_irls(design, d[label].to_numpy(), d["PLAYER_ID"].to_numpy(), ["intercept", *features])
    tab = tab[tab["term"] != "intercept"].copy()
    tab["scale"] = ["log1p, per SD" if f.startswith(("days_since", "games_since", "travel", "inj_events",
                                                       "prior_injury_count"))
                    else "per SD" for f in tab["term"]]
    tab["sd_raw_units"] = sd.reindex(tab["term"]).to_numpy()
    tab = tab.assign(abs_log_or=np.abs(np.log(tab["odds_ratio"])))
    return tab.sort_values("abs_log_or", ascending=False).drop(columns="abs_log_or").reset_index(drop=True)
