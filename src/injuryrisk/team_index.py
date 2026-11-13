"""Team Health Index (spec 8, M8).

For a team at decision point *g* (before tip-off), over the next N games:

* per active player i: per-game hazard ``h_i`` (the H = 1 model), so
  ``P_i(new spell within N) = 1 - (1 - h_i)^N``
* ``predicted_lost_minutes = Σ_i proj_min_i · h_i · N · L``, with L the
  training mean injury-spell length capped at N (games missed per spell)
* ``expected_available_minutes = Σ_i proj_min_i · N - predicted_lost_minutes``
* ``fragility`` = share of projected minutes held by players with
  ``P_i(spell within N) > threshold``

``proj_min_i`` is the player's minutes per game over his last 10 team games
(prior-season MPG as a fallback). Players already out at *g* are not in the
risk set and not in the index: this measures new risk to the healthy roster.

Validation: correlate the index with *actual* minutes lost to new injury
spells over the next N games (``Σ_i proj_min_i · injured games``), using
non-overlapping windows (every N-th game) and a team-level bootstrap CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def actual_injury_games(panel: pd.DataFrame, spells: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per (stint, game_num g): injured games for that player in team games g+1..g+N
    from spells that *start* after g (new injuries only)."""
    inj = spells[spells["severity"].isin(["minor", "major"])][["spell_id", "start_num"]]
    p = panel[["stint_id", "game_num", "spell_id"]].merge(inj, on="spell_id", how="left")
    p["injured"] = p["start_num"].notna().astype(float)
    rows = []
    # For each row, sum injured flags over the next N rows of the stint whose
    # spell started after the decision game.
    p = p.sort_values(["stint_id", "game_num"]).reset_index(drop=True)
    for k in range(1, horizon + 1):
        nxt_inj = p.groupby("stint_id")["injured"].shift(-k)
        nxt_start = p.groupby("stint_id")["start_num"].shift(-k)
        rows.append(np.where(nxt_start > p["game_num"], nxt_inj.fillna(0), 0.0))
    p["actual_injured_games"] = np.sum(rows, axis=0)
    return p[["stint_id", "game_num", "actual_injured_games"]]


def build(pred: pd.DataFrame, panel: pd.DataFrame, spells: pd.DataFrame, horizon: int,
          threshold: float, mean_spell_len: float, pcol: str) -> pd.DataFrame:
    act = actual_injury_games(panel, spells, horizon)
    d = pred.merge(act, on=["stint_id", "game_num"], how="left")
    d["proj_min"] = d["mpg_last10_played"].fillna(d["mpg_prior"]).fillna(0)
    h = d[pcol].clip(0, 1)
    d["p_spell_n"] = 1 - (1 - h) ** horizon
    L = min(mean_spell_len, horizon)
    d["pred_lost"] = d["proj_min"] * h * horizon * L
    d["actual_lost"] = d["proj_min"] * d["actual_injured_games"]
    d["fragile_min"] = np.where(d["p_spell_n"] > threshold, d["proj_min"], 0.0)
    team = d.groupby(["season", "TEAM_ID", "GAME_ID", "GAME_DATE"], as_index=False).agg(
        players=("PLAYER_ID", "size"), proj_minutes=("proj_min", "sum"),
        predicted_lost_minutes=("pred_lost", "sum"), actual_lost_minutes=("actual_lost", "sum"),
        fragile_minutes=("fragile_min", "sum"),
    )
    team["expected_available_minutes"] = team["proj_minutes"] * horizon - team["predicted_lost_minutes"]
    team["fragility"] = team["fragile_minutes"] / team["proj_minutes"].replace(0, np.nan)
    return team


def non_overlapping(team: pd.DataFrame, horizon: int) -> pd.DataFrame:
    team = team.sort_values(["season", "TEAM_ID", "GAME_DATE"]).copy()
    idx = team.groupby(["season", "TEAM_ID"]).cumcount()
    return team[idx % horizon == 0]


def validate(team: pd.DataFrame, col: str, reps: int = 300, seed: int = 7) -> dict:
    d = team.dropna(subset=[col, "actual_lost_minutes"])
    x, y = d[col].to_numpy(), d["actual_lost_minutes"].to_numpy()
    out = {"index": col, "windows": len(d), "pearson": pearsonr(x, y).statistic,
           "spearman": spearmanr(x, y).statistic}
    # Resample teams (clusters of windows) for the CI.
    keys = d["TEAM_ID"].to_numpy()
    uniq = np.unique(keys)
    groups = [np.flatnonzero(keys == k) for k in uniq]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(reps):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(uniq), len(uniq))])
        boots.append(spearmanr(x[idx], y[idx]).statistic)
    out["spearman_lo"], out["spearman_hi"] = np.quantile(boots, [0.025, 0.975])
    return out
