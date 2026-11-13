"""Schedule-driven variation: the second night of a back-to-back (M7, spec 7 option 2).

Minutes are endogenous, so a classifier can't answer "what if we rest him?".
Back-to-backs, 3-in-4 and 4-in-6 stretches are set by the league schedule, not
the coach, so comparing the same player's injury rate on dense vs rested nights
is closer to a natural experiment.

Estimator: linear probability model with player fixed effects and season
dummies, standard errors clustered by player. Outcome: a new injury spell
starts the next game (``y_any_h1``), i.e. the player got hurt in this game.
Controls: home, travel distance, time-zone shift, age.

Threat to validity, measured not assumed: the risk set is games the player
*played*. If coaches rest exactly the at-risk players on B2B nights, the
players left in the B2B sample are healthier, and the estimate is biased toward
zero. ``selection_check`` reports how often players sit (rest-coded or
unexplained absences) on B2B second nights vs other nights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from injuryrisk.models.glm import within_ols

CONTROLS = ["is_home", "travel_km", "abs_tz_shift", "age"]


def estimate(table: pd.DataFrame, treatment: str, label: str = "y_any_h1") -> dict:
    d = table.dropna(subset=[label, treatment]).copy()
    d["travel_km"] = d["travel_km"].fillna(0) / 1000.0           # per 1,000 km
    d["abs_tz_shift"] = d["abs_tz_shift"].fillna(0)
    d["age"] = d["age"].fillna(d["age"].median())
    seasons = pd.get_dummies(d["season"], prefix="s", drop_first=True, dtype=float)
    X = pd.concat([d[[treatment, *CONTROLS]].astype(float), seasons], axis=1)
    y = d[label].to_numpy(float)
    fe = within_ols(y, X.to_numpy(), d["PLAYER_ID"].to_numpy(), d["PLAYER_ID"].to_numpy(), list(X.columns))
    row = fe[fe["term"] == treatment].iloc[0]

    treated = d[treatment] == 1
    base = y[~treated.to_numpy()].mean()
    return {
        "treatment": treatment, "label": label, "rows": len(d), "treated_rows": int(treated.sum()),
        "rate_treated": float(y[treated.to_numpy()].mean()), "rate_control": float(base),
        "naive_diff_pp": 100 * float(y[treated.to_numpy()].mean() - base),
        "fe_effect_pp": 100 * row["coef"], "fe_ci_lo_pp": 100 * row["ci_lo"],
        "fe_ci_hi_pp": 100 * row["ci_hi"], "fe_p_value": row["p_value"],
        "fe_relative_effect": row["coef"] / base if base > 0 else np.nan,
    }


def selection_check(panel: pd.DataFrame, team_sched: pd.DataFrame, spells: pd.DataFrame,
                    seasons: list[int]) -> pd.DataFrame:
    """Share of rostered player-games sat out, by B2B status and absence cause."""
    p = panel[panel["season"].isin(seasons)][["TEAM_ID", "GAME_ID", "played", "spell_id"]]
    p = p.merge(team_sched[["TEAM_ID", "GAME_ID", "b2b"]], on=["TEAM_ID", "GAME_ID"], how="left")
    cause = spells.set_index("spell_id")["cause"]
    p["cause"] = np.where(p["played"] == 1, "played", p["spell_id"].map(cause).fillna("unknown"))
    rows = []
    for b2b, g in p.groupby("b2b"):
        n = len(g)
        rows.append({
            "b2b_second_night": int(b2b), "rostered_player_games": n,
            "sat_any": float((g["cause"] != "played").mean()),
            "sat_rest_coded": float((g["cause"] == "rest").mean()),
            "sat_unexplained": float((g["cause"] == "unknown").mean()),
            "sat_injury": float(g["cause"].isin(["injury_minor", "injury_major"]).mean()),
        })
    return pd.DataFrame(rows)
