"""Per-season player bio: age, height, weight, draft position, usage.

Ported from ``nba_fantasy_model``'s ``player_bio.py``. ``LeagueDashPlayerBioStats``
is one request per season and reports age as of that season, keyed on
``PLAYER_ID``. Do not rebuild age from the fantasy ``player_master.csv``: it was
missing age for 31-64% of rows.

Position is not in the bio endpoint. ``fetch_positions`` takes it from
``PlayerIndex`` (one request, all players, historical included). If that call
fails, the position columns stay empty and the pipeline carries on.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from injuryrisk.config import season_label
from injuryrisk.paths import DATA_RAW

KEEP = [
    "PLAYER_ID", "PLAYER_NAME", "AGE", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT",
    "DRAFT_YEAR", "DRAFT_ROUND", "DRAFT_NUMBER", "USG_PCT", "TS_PCT",
]

#: "Undrafted" is a string in a numeric column; scored as a late pick.
UNDRAFTED_PICK = 61
SLEEP = 0.6
MAX_RETRIES = 3
POSITIONS_FILE = "player_positions.csv"


def fetch_season(end_year: int, *, timeout: int = 90) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashplayerbiostats

    last: Exception | None = None
    for _ in range(MAX_RETRIES):
        try:
            raw = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
                season=season_label(end_year),
                season_type_all_star="Regular Season",
                timeout=timeout,
            ).get_data_frames()[0]
            break
        except Exception as exc:
            last = exc
            time.sleep(2.0)
    else:
        raise RuntimeError(f"bio stats failed for {season_label(end_year)}: {last!r}")

    df = raw[[c for c in KEEP if c in raw.columns]].copy()
    for col in ("PLAYER_WEIGHT", "PLAYER_HEIGHT_INCHES", "DRAFT_YEAR", "DRAFT_ROUND", "DRAFT_NUMBER"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["DRAFT_NUMBER"] = df["DRAFT_NUMBER"].fillna(UNDRAFTED_PICK)
    df["season"] = int(end_year)
    return df.drop_duplicates(subset=["PLAYER_ID"], keep="first").reset_index(drop=True)


def fetch_positions(*, timeout: int = 90) -> pd.DataFrame:
    """``PLAYER_ID -> POSITION`` (G, F, C, G-F, F-C...) for every player on record."""
    from nba_api.stats.endpoints import playerindex

    raw = playerindex.PlayerIndex(historical_nullable="1", timeout=timeout).get_data_frames()[0]
    id_col = "PERSON_ID" if "PERSON_ID" in raw.columns else "PLAYER_ID"
    out = raw[[id_col, "POSITION"]].rename(columns={id_col: "PLAYER_ID"})
    return out.drop_duplicates("PLAYER_ID")


def pull(seasons: list[int], *, raw_dir: Path | None = None, overwrite: bool = False) -> None:
    raw_dir = raw_dir or DATA_RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in seasons:
        path = raw_dir / f"player_bio_{year}.csv"
        if path.exists() and not overwrite:
            print(f"  bio {season_label(year)}: on disk")
            continue
        df = fetch_season(year)
        df.to_csv(path, index=False)
        print(f"  bio {season_label(year)}: {len(df)} players")
        time.sleep(SLEEP)

    pos_path = raw_dir / POSITIONS_FILE
    if pos_path.exists() and not overwrite:
        return
    try:
        pos = fetch_positions()
        pos.to_csv(pos_path, index=False)
        print(f"  positions: {len(pos)} players")
    except Exception as exc:  # optional source; physical.py tolerates its absence
        print(f"  positions: PlayerIndex failed ({exc!r}); position features will be empty")


def load_all(raw_dir: Path | None = None) -> pd.DataFrame:
    """Every season's bio rows, one row per ``(PLAYER_ID, season)``."""
    raw_dir = raw_dir or DATA_RAW
    frames = [pd.read_csv(p) for p in sorted(raw_dir.glob("player_bio_*.csv"))]
    if not frames:
        raise FileNotFoundError("No player_bio_*.csv files. Run `python run.py collect` first.")
    return pd.concat(frames, ignore_index=True)


def load_positions(raw_dir: Path | None = None) -> pd.DataFrame | None:
    path = (raw_dir or DATA_RAW) / POSITIONS_FILE
    return pd.read_csv(path) if path.exists() else None
