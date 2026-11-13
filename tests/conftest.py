"""Synthetic game logs and events, small enough to reason about by hand."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from injuryrisk.labels.classify import classify_events

BOS, LAL = 1610612738, 1610612747
BOX = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK",
       "TOV", "PF", "PTS", "PLUS_MINUS"]


def make_logs(games: list[tuple], rng: np.random.Generator) -> pd.DataFrame:
    """``games``: (season, game_id, date, home_team, [(player_id, team_id), ...] who played)."""
    rows = []
    for season, gid, date, home, players in games:
        for pid, team in players:
            abbr = "BOS" if team == BOS else "LAL"
            opp = "LAL" if team == BOS else "BOS"
            matchup = f"{abbr} vs. {opp}" if team == home else f"{abbr} @ {opp}"
            row = {"season": season, "PLAYER_ID": pid, "PLAYER_NAME": f"Player {pid}", "TEAM_ID": team,
                   "TEAM_ABBREVIATION": abbr, "GAME_ID": gid, "GAME_DATE": pd.Timestamp(date),
                   "MATCHUP": matchup, "MIN": float(rng.integers(5, 40))}
            row.update({c: float(rng.integers(0, 10)) for c in BOX})
            rows.append(row)
    return pd.DataFrame(rows)


def make_events(rows: list[tuple]) -> pd.DataFrame:
    """``rows``: (player_id, date, note)."""
    ev = pd.DataFrame(rows, columns=["PLAYER_ID", "date", "notes"])
    ev["date"] = pd.to_datetime(ev["date"])
    ev["direction"] = "out"
    ev["season"] = ev["date"].dt.year + (ev["date"].dt.month >= 8)
    return classify_events(ev)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
