"""The label (spec section 4.3).

Row = a player-game *t* the player played. The prediction is made at tip-off of
*t*: features come from games strictly before *t* plus the schedule of *t*.

``y_{target}_h{H} = 1`` if a new absence spell of that kind starts in the next
``H`` team games (t+1 .. t+H) of the same stint. Getting hurt in game *t* and
missing *t+1* is a positive for H = 1.

* Rows where the player is already out are not at risk and are never rows: the
  risk set is played games only. Once a spell starts the player leaves the risk
  set until he plays again (discrete-time survival logic).
* A negative needs H observed games after *t*. Rows with fewer (end of season,
  trade, waiver) are censored to NaN unless a positive was already seen.

Targets: ``any`` (minor or major), ``minor``, ``major``. Severity comes from
``attribute_spells``: major = a major-class note, or a minor note on a spell of
at least ``major_min_games`` games.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = ("any", "minor", "major")


def label_column(target: str, horizon: int) -> str:
    return f"y_{target}_h{horizon}"


def attach_targets(panel: pd.DataFrame, spells: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Add ``spell_start_severity`` and every ``y_*`` column to ``panel``."""
    panel = panel.copy()
    starts = spells.set_index("spell_id")["severity"]
    first_row = panel["spell_id"].ne(panel.groupby("stint_id")["spell_id"].shift()) & (panel["spell_id"] >= 0)
    panel["spell_start_severity"] = np.where(
        first_row, panel["spell_id"].map(starts).fillna("none"), "none")
    panel["spell_start_cause"] = np.where(
        first_row, panel["spell_id"].map(spells.set_index("spell_id")["cause"]).fillna("unknown"), "")

    sev = panel["spell_start_severity"]
    flags = {
        "any": sev.isin(["minor", "major"]).astype(float),
        "minor": (sev == "minor").astype(float),
        "major": (sev == "major").astype(float),
    }
    grp = panel.groupby("stint_id")
    pos_in_stint = grp.cumcount()
    remaining = grp["GAME_ID"].transform("size") - pos_in_stint - 1

    for target, flag in flags.items():
        shifted = {k: flag.groupby(panel["stint_id"]).shift(-k) for k in range(1, max(horizons) + 1)}
        for h in horizons:
            window = pd.concat([shifted[k] for k in range(1, h + 1)], axis=1)
            y = (window.fillna(0).max(axis=1) > 0).astype(float)
            y[(remaining < h) & (y == 0)] = np.nan          # censored negative
            y[panel["played"] == 0] = np.nan                 # not at risk
            panel[label_column(target, h)] = y
    return panel


def label_summary(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    risk = panel[panel["played"] == 1]
    rows = []
    for season, g in risk.groupby("season"):
        row = {"season": season, "at_risk_rows": len(g)}
        for t in TARGETS:
            for h in horizons:
                col = label_column(t, h)
                row[f"{col}_rate"] = g[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)
