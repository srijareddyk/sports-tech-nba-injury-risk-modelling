"""Gradient-boosted trees. scikit-learn's histogram GBM: native NaN handling,
no extra dependency. Trained on log-loss with no class reweighting, so the
probabilities stay close to calibrated (``scale_pos_weight=63`` in the old
notebook pushed every score toward 0.7).

The search space is the fixed ``models.gbm_grid`` in the config; every config is
scored on the validation season and logged.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier


def gbm(params: dict, seed: int = 7) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss", early_stopping=False, random_state=seed, **params,
    )
