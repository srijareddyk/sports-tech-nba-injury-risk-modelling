"""Interpretation on the fixed label (M6).

* Grouped permutation importance (PR-AUC drop) for the GBM on the last test
  season. Permuting whole groups gives a fair answer when features inside a
  group are strongly correlated (min_mean_7 vs min_mean_15).
* Per-feature permutation importance for the top features.
* SHAP, if ``shap`` is installed.
* Minor vs major: separate forward chains, compared on PR-AUC lift over base
  rate and over the history-only baseline.
* Age, re-read: observed rate and mean prediction by age band. The old
  notebook's age reading came from a SHAP misreading and a contaminated label
  (young players stashed on the IL, survivorship of durable veterans).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from injuryrisk.evaluate.metrics import safe_ap
from injuryrisk.features.build import FEATURE_GROUPS


def permutation_importance(model, X: pd.DataFrame, y: np.ndarray, columns: dict[str, list[str]],
                           repeats: int = 3, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = safe_ap(y, model.predict_proba(X)[:, 1])
    rows = []
    for name, cols in columns.items():
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]      # same permutation across the group
            drops.append(base - safe_ap(y, model.predict_proba(Xp)[:, 1]))
        rows.append({"name": name, "n_features": len(cols), "pr_auc_drop": float(np.mean(drops)),
                     "drop_sd": float(np.std(drops)), "base_pr_auc": base})
    return pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False).reset_index(drop=True)


def grouped_importance(model, X, y, features, repeats=3, seed=7) -> pd.DataFrame:
    groups = {g: [f for f in fs if f in features] for g, fs in FEATURE_GROUPS.items()}
    return permutation_importance(model, X, y, {g: c for g, c in groups.items() if c}, repeats, seed)


def feature_importance(model, X, y, features, repeats=3, seed=7) -> pd.DataFrame:
    return permutation_importance(model, X, y, {f: [f] for f in features}, repeats, seed)


def shap_summary(model, X: pd.DataFrame, n: int = 3000, seed: int = 7) -> pd.DataFrame | None:
    try:
        import shap
    except ImportError:
        return None
    sample = X.sample(min(n, len(X)), random_state=seed)
    values = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(values, list):
        values = values[1]
    return pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(values).mean(axis=0)}) \
        .sort_values("mean_abs_shap", ascending=False)


def age_table(pred: pd.DataFrame, table: pd.DataFrame, label: str, pcol: str = "p_gbm") -> pd.DataFrame:
    df = pred.merge(table[["PLAYER_ID", "GAME_ID", "age"]], on=["PLAYER_ID", "GAME_ID"], how="left")
    df["age_band"] = pd.cut(df["age"], [17, 22, 25, 28, 31, 34, 45],
                            labels=["<=22", "23-25", "26-28", "29-31", "32-34", "35+"])
    return df.groupby("age_band", observed=True).agg(
        rows=(label, "size"), observed_rate=(label, "mean"), mean_pred=(pcol, "mean"),
        players=("PLAYER_ID", "nunique")).reset_index()
