"""Absences from the schedule, not from transactions (spec section 4.2).

A player's games missed = his team's schedule minus his own appearances. Both
are in the game logs, so the transaction log is used only to explain *why* a
player was out, never to decide *whether* he was.

Pipeline
--------
1. ``build_team_games``: each club's real schedule, counted, never assumed to be
   82 (2011-12 was 66, 2019-20 was 64-75, 2020-21 was 72).
2. ``build_stints``: roster spans, first to last appearance per team stint.
   A stint whose last appearance is followed within ``span_extension_days`` by an
   injury event is extended to the end of the season (or to the player's next
   stint): a season-ending injury happens *after* the last appearance, and
   without this the biggest injuries would never produce an absence.
3. ``build_panel``: one row per (player, team game) inside a stint, with
   ``played`` 0/1. A team game inside the span with no player row is an absence.
4. ``find_spells``: consecutive absences within a stint form a spell.
5. ``attribute_spells``: each spell takes the class of the nearest classified
   event within ``event_window_days`` of its first missed game (as-of join),
   falling back to any informative event inside the spell. No event: ``unknown``
   (mostly coach's decisions, which PST does not log).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from injuryrisk.labels.classify import CAUSE_PRIORITY, INJURY_CLASSES

BOX_COLS = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB",
            "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS"]
UNINFORMATIVE = ("stash", "other")


def build_team_games(logs: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game), numbered within the team's season."""
    tg = logs[["season", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP"]]
    tg = tg.drop_duplicates(["TEAM_ID", "GAME_ID"]).copy()
    tg["is_home"] = tg["MATCHUP"].str.contains(" vs. ", regex=False).astype(int)
    tg["opp_abbr"] = tg["MATCHUP"].str.split(" ").str[-1]
    tg["home_abbr"] = np.where(tg["is_home"] == 1, tg["TEAM_ABBREVIATION"], tg["opp_abbr"])
    tg = tg.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    tg["game_num"] = tg.groupby(["TEAM_ID", "season"]).cumcount() + 1
    tg["team_games"] = tg.groupby(["TEAM_ID", "season"])["GAME_ID"].transform("size")
    return tg


def schedule_lengths(team_games: pd.DataFrame) -> pd.DataFrame:
    """Min/median/max club schedule by season, for checking the short seasons."""
    per_team = team_games.groupby(["season", "TEAM_ID"])["GAME_ID"].nunique()
    return per_team.groupby("season").agg(["min", "median", "max"])


def build_stints(
    logs: pd.DataFrame,
    team_games: pd.DataFrame,
    events: pd.DataFrame | None = None,
    extension_days: int = 7,
) -> pd.DataFrame:
    """Roster spans per (player, season, team), extended for late injuries."""
    app = logs[["PLAYER_ID", "TEAM_ID", "GAME_ID", "season", "GAME_DATE"]].drop_duplicates(
        ["PLAYER_ID", "GAME_ID"])
    app = app.merge(team_games[["TEAM_ID", "GAME_ID", "game_num", "team_games"]],
                    on=["TEAM_ID", "GAME_ID"], how="left")
    st = app.groupby(["PLAYER_ID", "season", "TEAM_ID"], as_index=False).agg(
        first_num=("game_num", "min"), last_num=("game_num", "max"),
        first_date=("GAME_DATE", "min"), last_date=("GAME_DATE", "max"),
        appearances=("GAME_ID", "size"), team_games=("team_games", "first"),
    )
    st = st.sort_values(["PLAYER_ID", "season", "first_date"]).reset_index(drop=True)
    st["stint_id"] = np.arange(len(st))
    st["next_first_date"] = st.groupby(["PLAYER_ID", "season"])["first_date"].shift(-1)
    st["end_num"] = st["last_num"]
    st["extended"] = False

    if events is not None and len(events):
        inj = events[(events["direction"] == "out") & events["event_class"].isin(INJURY_CLASSES)]
        inj = inj.dropna(subset=["PLAYER_ID"])[["PLAYER_ID", "date"]].copy()
        inj["PLAYER_ID"] = inj["PLAYER_ID"].astype("int64")
        inj = inj.sort_values("date")
        probe = st[["stint_id", "PLAYER_ID", "last_date"]].sort_values("last_date")
        probe["PLAYER_ID"] = probe["PLAYER_ID"].astype("int64")
        hit = pd.merge_asof(
            probe, inj.rename(columns={"date": "evt_date"}),
            left_on="last_date", right_on="evt_date", by="PLAYER_ID",
            direction="forward", tolerance=pd.Timedelta(days=extension_days),
        )
        ext_ids = set(hit.loc[hit["evt_date"].notna(), "stint_id"])
        ext = st["stint_id"].isin(ext_ids) & (st["last_num"] < st["team_games"])
        st.loc[ext, "end_num"] = st.loc[ext, "team_games"]
        st.loc[ext, "extended"] = True
    return st


def build_panel(logs: pd.DataFrame, team_games: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, team game) inside a stint. ``played`` is 0 or 1.

    Box-score columns are NaN on absence rows: a DNP is not a median game.
    """
    app = logs[["PLAYER_ID", "GAME_ID", *BOX_COLS]].drop_duplicates(["PLAYER_ID", "GAME_ID"])
    tg_cols = ["season", "TEAM_ID", "GAME_ID", "GAME_DATE", "game_num", "team_games",
               "is_home", "home_abbr", "TEAM_ABBREVIATION"]
    parts = []
    for season, st in stints.groupby("season"):
        tg = team_games.loc[team_games["season"] == season, tg_cols]
        p = st[["PLAYER_ID", "TEAM_ID", "stint_id", "first_num", "end_num", "last_num",
                "next_first_date"]].merge(tg, on="TEAM_ID")
        keep = (p["game_num"] >= p["first_num"]) & (p["game_num"] <= p["end_num"])
        keep &= p["next_first_date"].isna() | (p["GAME_DATE"] < p["next_first_date"])
        p = p[keep]
        p = p.assign(extension_row=p["game_num"] > p["last_num"])
        parts.append(p.drop(columns=["first_num", "end_num", "last_num", "next_first_date"]))
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.merge(app, on=["PLAYER_ID", "GAME_ID"], how="left", indicator=True)
    panel["played"] = (panel.pop("_merge") == "both").astype(int)
    panel = panel.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return panel


def find_spells(panel: pd.DataFrame) -> pd.DataFrame:
    """Runs of consecutive absences within a stint. Adds ``spell_id`` to ``panel``."""
    absent = panel["played"] == 0
    change = (panel["played"] != panel.groupby("stint_id")["played"].shift()) | (
        panel["stint_id"] != panel["stint_id"].shift())
    run = change.cumsum()
    panel["spell_id"] = np.where(absent, run, -1)
    stint_last = panel.groupby("stint_id")["game_num"].transform("max")
    rows = panel[absent].assign(_stint_last=stint_last[absent])
    spells = rows.groupby("spell_id", as_index=False).agg(
        PLAYER_ID=("PLAYER_ID", "first"), season=("season", "first"), TEAM_ID=("TEAM_ID", "first"),
        stint_id=("stint_id", "first"), start_date=("GAME_DATE", "min"), end_date=("GAME_DATE", "max"),
        start_num=("game_num", "min"), end_num=("game_num", "max"), length=("game_num", "size"),
        stint_last=("_stint_last", "first"),
    )
    # Ends on the stint's last game: the true length is unknown (season ended).
    spells["censored"] = spells["end_num"] == spells["stint_last"]
    return spells.drop(columns="stint_last")


def attribute_spells(
    spells: pd.DataFrame,
    events: pd.DataFrame,
    window_days: int = 3,
    within_spell: bool = True,
    major_min_games: int = 12,
    known_absences: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Give each spell a cause: the best classified event near its start."""
    ev = events[(events["direction"] == "out")].dropna(subset=["PLAYER_ID"])
    ev = ev[["PLAYER_ID", "date", "event_class", "notes", "region_group", "major_structure", "surgery"]].copy()
    ev["PLAYER_ID"] = ev["PLAYER_ID"].astype("int64")

    sp = spells[["spell_id", "PLAYER_ID", "start_date", "end_date"]].copy()
    sp["PLAYER_ID"] = sp["PLAYER_ID"].astype("int64")
    m = sp.merge(ev, on="PLAYER_ID", how="inner")
    lo = m["start_date"] - pd.Timedelta(days=window_days)
    hi = m["start_date"] + pd.Timedelta(days=window_days)
    if within_spell:
        hi = np.maximum(hi, m["end_date"])
    m = m[(m["date"] >= lo) & (m["date"] <= hi)].copy()
    m["_uninformative"] = m["event_class"].isin(UNINFORMATIVE).astype(int)
    m["_dist"] = (m["date"] - m["start_date"]).dt.days.abs()
    m["_prio"] = m["event_class"].map(CAUSE_PRIORITY).fillna(99)
    best = m.sort_values(["spell_id", "_uninformative", "_dist", "_prio"]).drop_duplicates("spell_id")
    best = best.rename(columns={"date": "cause_event_date", "event_class": "cause",
                                "notes": "cause_notes", "region_group": "cause_region_group",
                                "major_structure": "cause_major_structure", "surgery": "cause_surgery"})
    out = spells.merge(
        best[["spell_id", "cause", "cause_event_date", "cause_notes", "cause_region_group",
              "cause_major_structure", "cause_surgery"]],
        on="spell_id", how="left",
    )
    out["cause"] = out["cause"].fillna("unknown")

    if known_absences is not None and len(known_absences):
        ka = known_absences.dropna(subset=["player_id"])
        for row in ka.itertuples(index=False):
            hit = (out["PLAYER_ID"] == int(row.player_id)) & (out["season"] == int(row.season)) & (
                out["cause"] == "unknown")
            reason = {"surgery": "injury_major", "injury": "injury_minor"}.get(row.reason, row.reason)
            out.loc[hit, "cause"] = reason

    # When the absence became attributable: never before the event was logged.
    out["known_date"] = out[["start_date", "cause_event_date"]].max(axis=1)
    out["severity"] = "none"
    injury = out["cause"].isin(INJURY_CLASSES)
    major = injury & ((out["cause"] == "injury_major") | (out["length"] >= major_min_games))
    out.loc[injury, "severity"] = "minor"
    out.loc[major, "severity"] = "major"
    return out


def season_availability(panel: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Per player-season: games played, scheduled, missed (spec 4.2).

    A traded player is measured against the longest schedule he was part of, so
    84 appearances in an 82-game season is not a negative absence.
    """
    played = panel[panel["played"] == 1].groupby(["PLAYER_ID", "season"])["GAME_ID"].nunique()
    sched = panel.groupby(["PLAYER_ID", "season"])["team_games"].max()
    out = pd.DataFrame({"played": played, "scheduled": sched}).fillna({"played": 0}).reset_index()
    out["missed"] = (out["scheduled"] - out["played"]).clip(lower=0)
    return out
