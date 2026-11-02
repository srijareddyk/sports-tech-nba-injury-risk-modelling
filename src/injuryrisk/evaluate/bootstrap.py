"""Player-level bootstrap CIs (spec 6.3).

Rows of one player are correlated, so rows are not independent draws. Each
replicate resamples *players* with replacement and keeps all their rows.
Paired differences use the same replicate for both models, which is what
makes "model A beats model B" testable against the noise floor.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from injuryrisk.evaluate.metrics import safe_ap
from sklearn.metrics import brier_score_loss


class PlayerBootstrap:
    def __init__(self, players: np.ndarray, reps: int, seed: int = 7):
        codes, uniq = pd.factorize(np.asarray(players))
        order = np.argsort(codes, kind="stable")
        bounds = np.cumsum(np.bincount(codes, minlength=len(uniq)))[:-1]
        self.groups = np.split(order, bounds)
        rng = np.random.default_rng(seed)
        self.draws = [rng.integers(0, len(uniq), len(uniq)) for _ in range(reps)]

    def indices(self):
        for draw in self.draws:
            yield np.concatenate([self.groups[i] for i in draw])

    def run(self, stat: Callable[[np.ndarray], float]) -> np.ndarray:
        return np.array([stat(idx) for idx in self.indices()])


def ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return (np.nan, np.nan)
    a = (1 - level) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))


def model_cis(y: np.ndarray, preds: dict[str, np.ndarray], players: np.ndarray, reps: int,
              seed: int, reference: str | None = None, flags: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """PR-AUC / Brier CIs per model, and paired PR-AUC differences vs ``reference``.

    ``flags`` (optional, per model): fixed alert sets whose precision is bootstrapped.
    """
    boot = PlayerBootstrap(players, reps, seed)
    y = np.asarray(y, float)
    rows = []
    stats = {}
    for name, p in preds.items():
        ap = np.empty(reps)
        br = np.empty(reps)
        pr = np.empty(reps)
        f = None if flags is None else flags.get(name)
        for r, idx in enumerate(boot.indices()):
            ap[r] = safe_ap(y[idx], p[idx])
            br[r] = brier_score_loss(y[idx], p[idx])
            if f is not None:
                fi = f[idx]
                pr[r] = y[idx][fi].mean() if fi.any() else np.nan
        stats[name] = ap
        row = {"model": name}
        row["pr_auc_lo"], row["pr_auc_hi"] = ci(ap)
        row["brier_lo"], row["brier_hi"] = ci(br)
        if f is not None:
            row["precision_lo"], row["precision_hi"] = ci(pr)
        rows.append(row)
    out = pd.DataFrame(rows)
    if reference and reference in stats:
        diffs = {n: stats[n] - stats[reference] for n in stats}
        out["pr_auc_diff_vs_ref"] = [np.nanmean(diffs[n]) for n in out["model"]]
        lohi = [ci(diffs[n]) for n in out["model"]]
        out["diff_lo"] = [x[0] for x in lohi]
        out["diff_hi"] = [x[1] for x in lohi]
        out["beats_ref"] = out["diff_lo"] > 0
    return out
