"""Schedule density and travel. Known in advance from the league schedule.

These are the only features a coach does not choose, which is what makes them
usable for the causal slice in ``causal/b2b.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from injuryrisk.features.common import haversine_km, NO_HISTORY_DAYS
from injuryrisk.paths import DATA_EXTERNAL

SCHEDULE_FEATURES = [
    "team_days_rest", "b2b", "three_in_four", "four_in_six", "road_trip_len",
    "travel_km", "tz_shift", "abs_tz_shift", "is_home", "player_days_since_last_game", "player_b2b",
]


def load_arenas() -> pd.DataFrame:
    return pd.read_csv(DATA_EXTERNAL / "arenas.csv")


def team_schedule_features(team_games: pd.DataFrame, arenas: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (TEAM_ID, GAME_ID): rest, density, road trip, travel, time zones."""
    arenas = load_arenas() if arenas is None else arenas
    tg = team_games.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).copy()
    unknown = set(tg["home_abbr"]) - set(arenas["abbr"])
    assert not unknown, f"arenas.csv is missing abbreviations {sorted(unknown)}"

    key = [tg["TEAM_ID"], tg["season"]]
    d = tg["GAME_DATE"]
    days = lambda a, b: (a - b).dt.days
    tg["team_days_rest"] = days(d, d.groupby(key).shift(1))
    tg["b2b"] = (tg["team_days_rest"] == 1).astype(int)
    tg["three_in_four"] = (days(d, d.groupby(key).shift(2)) <= 3).astype(int)
    tg["four_in_six"] = (days(d, d.groupby(key).shift(3)) <= 5).astype(int)

    away = (tg["is_home"] == 0).astype(int)
    run = (away != away.groupby(key).shift()).groupby(key).cumsum()
    tg["road_trip_len"] = np.where(away == 1, away.groupby([tg["TEAM_ID"], tg["season"], run]).cumsum(), 0)

    a = arenas.set_index("abbr")
    lat, lon, tz = (tg["home_abbr"].map(a[c]) for c in ("lat", "lon", "utc_offset"))
    plat, plon, ptz = (s.groupby(key).shift(1) for s in (lat, lon, tz))
    tg["travel_km"] = haversine_km(plat, plon, lat, lon)
    tg["tz_shift"] = tz - ptz
    tg["abs_tz_shift"] = tg["tz_shift"].abs()
    return tg[["TEAM_ID", "GAME_ID", "team_days_rest", "b2b", "three_in_four", "four_in_six",
               "road_trip_len", "travel_km", "tz_shift", "abs_tz_shift"]]


def add_player_schedule(panel: pd.DataFrame) -> pd.DataFrame:
    """Days since the player's own last appearance, and a player-level B2B."""
    p = panel
    last_app = p["GAME_DATE"].where(p["played"] == 1)
    last_app = last_app.groupby(p["PLAYER_ID"]).shift(1)
    last_app = last_app.groupby(p["PLAYER_ID"]).ffill()
    gap = (p["GAME_DATE"] - last_app).dt.days
    p["player_days_since_last_game"] = gap.fillna(NO_HISTORY_DAYS).clip(upper=NO_HISTORY_DAYS)
    prev_played = p.groupby("stint_id")["played"].shift(1)
    p["player_b2b"] = ((p["b2b"] == 1) & (prev_played == 1)).astype(int)
    return p
