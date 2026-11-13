"""Workload, role and in-game intensity features, from the player panel.

The panel has one row per team game in the player's stint, absences included,
so "minutes over the last 7 games" counts a missed game as 0 minutes rather
than skipping it. Absent rows carry NaN box scores and are never imputed (the
old notebook filled 129,930 DNPs with the median).

Every rolling window covers rows t-k .. t-1 and resets each season. With
``box_score_lag = 1`` game *t*'s own box score never enters a feature for game *t*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from injuryrisk.features.common import lagged_mean, lagged_window

WORKLOAD_FEATURES = [
    "min_last1", "min_mean_3", "min_mean_7", "min_mean_15", "acwr_3_15",
    "min_z10", "min_delta", "games_last_7d", "season_min_cum", "season_games_played",
]
ROLE_FEATURES = ["mpg_prior", "gp_prior", "usage_prior", "mpg_last10_played", "starter_proxy"]
INTENSITY_FEATURES = ["shots_per_min_10", "reb_per_min_10", "fta_per_min_10", "pf_per_min_10"]
AVAIL_INSEASON_FEATURES = ["absences_last_10", "season_missed_to_date"]


def add_workload(panel: pd.DataFrame) -> pd.DataFrame:
    """Adds workload, intensity and in-season availability columns to ``panel``."""
    p = panel
    gid = p["PLAYER_ID"].astype("int64") * 10_000 + p["season"].astype("int64")
    played = p["played"].astype(float)
    minutes = p["MIN"].where(p["played"] == 1, 0.0).fillna(0.0).astype(float)

    prev1 = minutes.groupby(gid).shift(1)
    prev2 = minutes.groupby(gid).shift(2)
    p["min_last1"] = prev1
    for k in (3, 7, 15):
        p[f"min_mean_{k}"] = lagged_mean(minutes, gid, k)
    p["acwr_3_15"] = (p["min_mean_3"] / p["min_mean_15"]).where(p["min_mean_15"] > 0)

    # z-score of the last game against the 10 games before it (t-11 .. t-2).
    s, n = lagged_window(prev1, gid, 10)
    sq, _ = lagged_window(prev1 ** 2, gid, 10)
    mean10 = s / n
    sd10 = np.sqrt((sq / n - mean10 ** 2).clip(lower=0))
    p["min_z10"] = ((prev1 - mean10) / sd10).where((n >= 5) & (sd10 > 1.0))
    p["min_delta"] = prev1 - prev2

    cum_min = minutes.groupby(gid).cumsum()
    p["season_min_cum"] = cum_min - minutes
    p["season_games_played"] = played.groupby(gid).cumsum() - played
    absent = 1.0 - played
    p["season_missed_to_date"] = absent.groupby(gid).cumsum() - absent
    p["absences_last_10"], _ = lagged_window(absent, gid, 10)

    # Appearances in the 7 days before tip-off (any stint).
    p["games_last_7d"] = _appearances_last_days(p, 7)

    min_s, _ = lagged_window(minutes, gid, 10)
    played_s, _ = lagged_window(played, gid, 10)
    p["mpg_last10_played"] = (min_s / played_s).where(played_s > 0)
    p["starter_proxy"] = (p["mpg_last10_played"] >= 24).astype(float).where(p["mpg_last10_played"].notna())

    for name, num in (
        ("shots_per_min_10", p["FGA"].fillna(0) + p["FTA"].fillna(0)),
        ("reb_per_min_10", p["REB"].fillna(0)),
        ("fta_per_min_10", p["FTA"].fillna(0)),     # proxy for fouls drawn
        ("pf_per_min_10", p["PF"].fillna(0)),
    ):
        num_s, _ = lagged_window(num.astype(float), gid, 10)
        p[name] = (num_s / min_s).where(min_s >= 30)
    return p


def _appearances_last_days(p: pd.DataFrame, days: int) -> np.ndarray:
    from injuryrisk.features.common import window_count

    apps = p.loc[p["played"] == 1, ["PLAYER_ID", "GAME_DATE"]].rename(columns={"GAME_DATE": "date"})
    return window_count(p[["PLAYER_ID", "GAME_DATE"]], apps, days)


def prior_season_role(panel: pd.DataFrame, shrink_games: float = 10.0) -> pd.DataFrame:
    """Last season's MPG, games and usage proxy, shrunk toward the league mean.

    Shrinkage stops "poisoned priors": 35.0 MPG from a single game.
    """
    played = panel[panel["played"] == 1]
    s = played.groupby(["PLAYER_ID", "season"]).agg(
        gp=("GAME_ID", "size"), min_sum=("MIN", "sum"),
        fga=("FGA", "sum"), fta=("FTA", "sum"), tov=("TOV", "sum"),
    ).reset_index()
    league = s.groupby("season").agg(min_sum=("min_sum", "sum"), gp=("gp", "sum"),
                                     fga=("fga", "sum"), fta=("fta", "sum"), tov=("tov", "sum"))
    league_mpg = (league["min_sum"] / league["gp"]).rename("league_mpg")
    league_usage = ((league["fga"] + 0.44 * league["fta"] + league["tov"]) / league["min_sum"]).rename("league_usage")
    s = s.join(league_mpg, on="season").join(league_usage, on="season")
    k = shrink_games
    s["mpg_prior"] = (s["min_sum"] + k * s["league_mpg"]) / (s["gp"] + k)
    poss = s["fga"] + 0.44 * s["fta"] + s["tov"]
    s["usage_prior"] = (poss + k * s["league_mpg"] * s["league_usage"]) / (s["min_sum"] + k * s["league_mpg"])
    s["gp_prior"] = s["gp"]
    s["season"] = s["season"] + 1          # last season's numbers describe this season
    return s[["PLAYER_ID", "season", "mpg_prior", "gp_prior", "usage_prior"]]
