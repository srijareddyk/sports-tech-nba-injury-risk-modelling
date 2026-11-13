"""Metrics that match the decision (spec 6.4).

* PR-AUC and Brier lead. ROC-AUC is reported but misleading at 1-3% base rates.
* Calibration table: staff act on probabilities, so they must be calibrated.
* Precision/recall at a fixed alert budget: flag the top-k players per team per
  game. This mirrors a real decision better than a best-F1 threshold (the old
  notebook's best F1 meant 8.8% precision, ~10 false alarms per injury).
* Expected cost of acting on the alerts, and the break-even precision.

Cost model. Each alert rests one player for ``cost_rest`` games. A rested
player who would have been hurt avoids the injury with probability
``efficacy``, saving ``cost_injury`` games:

    cost(alerts) = (TP + FP) * cost_rest + (FN + (1 - efficacy) * TP) * cost_injury
    cost(none)   = positives * cost_injury

One more alert at precision p changes cost by ``cost_rest - p * efficacy * cost_injury``,
so alerts pay only above ``break_even = cost_rest / (efficacy * cost_injury)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def safe_auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if 0 < np.mean(y) < 1 else np.nan


def safe_ap(y, p) -> float:
    return float(average_precision_score(y, p)) if np.sum(y) > 0 else np.nan


def brier_skill(y, p) -> float:
    """1 - Brier / Brier of predicting the observed base rate. >0 beats base rate."""
    ref = brier_score_loss(y, np.full(len(y), np.mean(y)))
    return float(1 - brier_score_loss(y, p) / ref) if ref > 0 else np.nan


def budget_flags(frame: pd.DataFrame, score: np.ndarray, k: int,
                 group_cols: tuple[str, ...] = ("TEAM_ID", "GAME_ID")) -> np.ndarray:
    """True for the top-``k`` scores within each team-game."""
    df = frame[list(group_cols)].copy()
    df["_s"] = score
    rank = df.groupby(list(group_cols))["_s"].rank(method="first", ascending=False)
    return (rank <= k).to_numpy()


def break_even_precision(cost_rest: float, cost_injury: float, efficacy: float) -> float:
    return cost_rest / (efficacy * cost_injury)


def expected_cost(y: np.ndarray, flags: np.ndarray, cost_rest: float, cost_injury: float,
                  efficacy: float) -> dict:
    y = np.asarray(y).astype(bool)
    tp = int((flags & y).sum())
    fp = int((flags & ~y).sum())
    fn = int((~flags & y).sum())
    cost = (tp + fp) * cost_rest + (fn + (1 - efficacy) * tp) * cost_injury
    none = y.sum() * cost_injury
    return {"alerts": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "cost_alerts": cost, "cost_no_alerts": float(none), "cost_saving": float(none - cost)}


def score_all(y: np.ndarray, p: np.ndarray, frame: pd.DataFrame, eval_cfg: dict,
              cost_injury: float) -> dict:
    """Every headline metric for one model on one test set."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    k = int(eval_cfg["alert_budget_per_team_game"])
    flags = budget_flags(frame, p, k)
    yb = y.astype(bool)
    out = {
        "n": len(y), "positives": int(y.sum()), "base_rate": float(y.mean()),
        "pr_auc": safe_ap(y, p), "roc_auc": safe_auc(y, p),
        "brier": float(brier_score_loss(y, p)), "brier_skill": brier_skill(y, p),
        "mean_pred": float(p.mean()),
        f"precision_at_{k}": float(yb[flags].mean()) if flags.any() else np.nan,
        f"recall_at_{k}": float(flags[yb].mean()) if yb.any() else np.nan,
    }
    c_rest = float(eval_cfg["cost_rest_games"])
    eff = float(eval_cfg["rest_efficacy"])
    out.update(expected_cost(y, flags, c_rest, cost_injury, eff))
    out["break_even_precision"] = break_even_precision(c_rest, cost_injury, eff)
    return out


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability curve on quantile bins of the prediction."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    return df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"),
                                 observed=("y", "mean")).reset_index()


def calibration_plot(tables: dict[str, pd.DataFrame], path) -> bool:
    """Write a reliability plot if matplotlib is installed. Returns success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(5, 5))
    hi = max(max(t["mean_pred"].max(), t["observed"].max()) for t in tables.values())
    ax.plot([0, hi], [0, hi], ls="--", c="grey", lw=1)
    for name, t in tables.items():
        ax.plot(t["mean_pred"], t["observed"], marker="o", label=name)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True
