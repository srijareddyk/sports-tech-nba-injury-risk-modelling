"""Availability and injury history, as of tip-off.

Three sources:

* **Season availability** (ported from ``availability_history.py``): games
  missed per season = schedule minus appearances, lagged 1-3 seasons, plus
  career totals. Players with no logged prior season stay NaN, never 0: a 0
  would read as "never missed a game".
* **Absence spells** (``labels/absences.py``): prior injury spells, days since
  the last one, games since return. A spell enters history only once its
  cause was knowable (``known_date``), so a PST note logged after tip-off can't
  leak in.
* **Classified PST events** (pattern from ``injury_raw_events.py``): days since
  the last injury event per body-region group, counts within 30/90/180/365
  days, prior major structure (ACL, Achilles, patellar/quad tendon), prior
  surgery. No severity judgement: the model gets raw counts and recency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from injuryrisk.features.common import NO_HISTORY_DAYS, asof_count, asof_last, days_between, window_count
from injuryrisk.labels.classify import INJURY_CLASSES
from injuryrisk.labels.vocab import REGION_GROUPS

RECENCY_HORIZONS = (30, 90, 180, 365)

AVAILABILITY_FEATURES = [
    "missed_lag1", "missed_lag2", "missed_lag3", "missed_career",
    "career_avail_rate", "seasons_missing_20plus",
]
SPELL_FEATURES = [
    "prior_injury_count", "prior_major_spells", "days_since_injury_spell",
    "last_spell_length", "games_since_return",
]
EVENT_FEATURES = [
    "days_since_injury_event", "inj_events_career",
    *(f"inj_events_{h}d" for h in RECENCY_HORIZONS),
    *(f"days_since_{g}" for g in REGION_GROUPS),
    *(f"inj_events_{g}_365d" for g in REGION_GROUPS),
    "max_region_events_365d", "prior_major_structure", "prior_surgery",
]
COMBINED_FEATURES = ["days_since_last_injury"]
INJURY_HISTORY_FEATURES = SPELL_FEATURES + EVENT_FEATURES + COMBINED_FEATURES


def season_history(season_table: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Availability-history block for each (PLAYER_ID, season) in ``rows``."""
    t = season_table.copy()
    keys = rows[["PLAYER_ID", "season"]].drop_duplicates()
    out = keys.copy()
    for lag in (1, 2, 3):
        lagged = t[["PLAYER_ID", "season", "missed"]].assign(season=t["season"] + lag)
        out = out.merge(lagged.rename(columns={"missed": f"missed_lag{lag}"}),
                        on=["PLAYER_ID", "season"], how="left")

    # Career totals over every season strictly before the row's season.
    t = t.sort_values(["PLAYER_ID", "season"])
    g = t.groupby("PLAYER_ID")
    t["c_missed"] = g["missed"].cumsum()
    t["c_played"] = g["played"].cumsum()
    t["c_sched"] = g["scheduled"].cumsum()
    t["c_heavy"] = (t["missed"] >= 20).astype(int).groupby(t["PLAYER_ID"]).cumsum()
    t["season_next"] = t["season"] + 1
    career = t[["PLAYER_ID", "season_next", "c_missed", "c_played", "c_sched", "c_heavy"]]
    # As-of merge: the latest logged season before the row's season.
    out = out.sort_values("season")
    career = career.sort_values("season_next")
    out = pd.merge_asof(out, career, left_on="season", right_on="season_next", by="PLAYER_ID",
                        direction="backward")
    out["missed_career"] = out["c_missed"]
    out["career_avail_rate"] = out["c_played"] / out["c_sched"]
    out["seasons_missing_20plus"] = out["c_heavy"]
    return out[["PLAYER_ID", "season", *AVAILABILITY_FEATURES]]


def spell_history(rows: pd.DataFrame, panel: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    """History from attributed injury spells, using ``known_date`` for visibility."""
    q = rows[["PLAYER_ID", "GAME_DATE"]].reset_index(drop=True)
    inj = spells[spells["severity"].isin(["minor", "major"])].copy()
    out = pd.DataFrame(index=q.index)
    out["prior_injury_count"] = asof_count(q, inj, edate="known_date")
    out["prior_major_spells"] = asof_count(q, inj[inj["severity"] == "major"], edate="known_date")

    last = asof_last(q, inj, ["start_date", "end_date", "length"], edate="known_date")
    out["days_since_injury_spell"] = days_between(q["GAME_DATE"], last["start_date"])
    out["last_spell_length"] = last["length"].fillna(0).to_numpy()

    apps = panel.loc[panel["played"] == 1, ["PLAYER_ID", "GAME_DATE"]].rename(columns={"GAME_DATE": "date"})
    before_now = asof_count(q, apps)
    q_end = pd.DataFrame({"PLAYER_ID": q["PLAYER_ID"], "GAME_DATE": last["end_date"]})
    has = q_end["GAME_DATE"].notna().to_numpy()
    before_end = np.zeros(len(q))
    if has.any():
        before_end[has] = asof_count(q_end[has].reset_index(drop=True), apps)
    out["games_since_return"] = np.where(has, before_now - before_end, np.nan)
    return out


def event_history(rows: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """History from classified PST injury events dated before tip-off."""
    q = rows[["PLAYER_ID", "GAME_DATE"]].reset_index(drop=True)
    ev = events[(events["direction"] == "out") & events["event_class"].isin(INJURY_CLASSES)]
    ev = ev.dropna(subset=["PLAYER_ID"]).copy()
    out = pd.DataFrame(index=q.index)

    last = asof_last(q, ev, ["notes"])
    out["days_since_injury_event"] = days_between(q["GAME_DATE"], last["_edate"])
    out["inj_events_career"] = asof_count(q, ev)
    for h in RECENCY_HORIZONS:
        out[f"inj_events_{h}d"] = window_count(q, ev, h)
    for group in REGION_GROUPS:
        gev = ev[ev["region_group"] == group]
        glast = asof_last(q, gev, ["notes"])
        out[f"days_since_{group}"] = days_between(q["GAME_DATE"], glast["_edate"])
        out[f"inj_events_{group}_365d"] = window_count(q, gev, 365)
    out["max_region_events_365d"] = out[[f"inj_events_{g}_365d" for g in REGION_GROUPS]].max(axis=1)
    out["prior_major_structure"] = (asof_count(q, ev[ev["major_structure"]]) > 0).astype(float)
    out["prior_surgery"] = (asof_count(q, ev[ev["surgery"]]) > 0).astype(float)
    return out


def combine(df: pd.DataFrame) -> pd.DataFrame:
    df["days_since_last_injury"] = np.minimum(
        df["days_since_injury_spell"].fillna(NO_HISTORY_DAYS),
        df["days_since_injury_event"].fillna(NO_HISTORY_DAYS),
    )
    return df
