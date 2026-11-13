"""Player game logs, one ``LeagueGameLog`` request per season.

Ported from ``nba_fantasy_model``'s ``historical_game_logs.py``. ``LeagueGameLog``
returns every player-game row of a season (~25k rows) in a single call, so twenty
seasons is twenty requests.

The endpoint lists only players who appeared. A player on the roster who did not
play has no row, which is exactly why absences are rebuilt from the team
schedule (``labels/absences.py``) rather than imputed.

Team schedules come from the same rows: every team game has players in it, so the
unique ``(TEAM_ID, GAME_ID)`` pairs are the schedule, with no second request.

    python run.py collect
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from injuryrisk.config import season_label
from injuryrisk.paths import DATA_RAW

KEEP = [
    "SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS",
]

#: Columns downstream code reads. NBA Stats spells turnovers ``TOV``; other
#: sources use ``TO``, and a silent mismatch went unnoticed for months in the
#: fantasy project. Assert rather than guard.
REQUIRED = ["PLAYER_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "MIN",
            "FGA", "FTA", "REB", "TOV", "PF"]

SLEEP_BETWEEN_SEASONS = 1.0
MAX_RETRIES = 3
RETRY_SLEEP = 3.0


def fetch_season(end_year: int, *, timeout: int = 120) -> pd.DataFrame:
    """Every regular-season player-game row for one season."""
    from nba_api.stats.endpoints import leaguegamelog

    label = season_label(end_year)
    last: Exception | None = None
    for _ in range(MAX_RETRIES):
        try:
            raw = leaguegamelog.LeagueGameLog(
                season=label,
                player_or_team_abbreviation="P",
                season_type_all_star="Regular Season",
                timeout=timeout,
            ).get_data_frames()[0]
            break
        except Exception as exc:  # network/timeout: retry, then give up
            last = exc
            time.sleep(RETRY_SLEEP)
    else:
        raise RuntimeError(f"LeagueGameLog failed for {label}: {last!r}")
    if raw.empty:
        raise RuntimeError(f"LeagueGameLog returned no rows for {label}")

    missing = [c for c in REQUIRED if c not in raw.columns]
    assert not missing, f"{label}: LeagueGameLog is missing columns {missing}"
    df = raw[[c for c in KEEP if c in raw.columns]].copy()
    df["season"] = int(end_year)
    return df.sort_values(["PLAYER_ID", "GAME_DATE"], kind="stable").reset_index(drop=True)


def path_for(end_year: int, raw_dir: Path | None = None) -> Path:
    return (raw_dir or DATA_RAW) / f"game_logs_{end_year}.csv"


def pull(seasons: list[int], *, raw_dir: Path | None = None, overwrite: bool = False) -> pd.DataFrame:
    """Fetch and write one CSV per season. Returns a per-season summary."""
    raw_dir = raw_dir or DATA_RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in seasons:
        path = path_for(year, raw_dir)
        if path.exists() and not overwrite:
            n = sum(1 for _ in open(path, encoding="utf-8")) - 1
            print(f"  game logs {season_label(year)}: on disk ({n} rows)")
            rows.append({"season": year, "rows": n, "status": "existing"})
            continue
        df = fetch_season(year)
        df.to_csv(path, index=False)
        print(f"  game logs {season_label(year)}: {len(df):>6} rows, "
              f"{df['PLAYER_ID'].nunique()} players, {df['GAME_ID'].nunique()} games, "
              f"{df['TEAM_ID'].nunique()} teams")
        rows.append({"season": year, "rows": len(df), "status": "fetched"})
        time.sleep(SLEEP_BETWEEN_SEASONS)
    return pd.DataFrame(rows)


def load_all(seasons: list[int], raw_dir: Path | None = None) -> pd.DataFrame:
    """Stack the season files. Fails loudly if a season is missing."""
    frames = []
    for year in seasons:
        path = path_for(year, raw_dir)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run `python run.py collect` first.")
        frames.append(pd.read_csv(path, dtype={"GAME_ID": str, "SEASON_ID": str}))
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED if c not in df.columns]
    assert not missing, f"game logs missing columns {missing}"
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
    return df
