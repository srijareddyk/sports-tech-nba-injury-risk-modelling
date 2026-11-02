"""Assemble the model table: one row per at-risk player-game, features + labels.

Every expected feature is asserted to exist and to be non-empty after the build.
The old notebook guarded with ``if "minutes" in df.columns`` when the column was
``numMinutes``, so every workload feature was silently skipped. Here a missing
feature stops the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from injuryrisk.features.common import NO_HISTORY_DAYS
from injuryrisk.features.history import (
    AVAILABILITY_FEATURES, INJURY_HISTORY_FEATURES, combine, event_history, season_history, spell_history,
)
from injuryrisk.features.physical import PHYSICAL_FEATURES, physical_table
from injuryrisk.features.schedule import SCHEDULE_FEATURES, add_player_schedule, team_schedule_features
from injuryrisk.features.workload import (
    AVAIL_INSEASON_FEATURES, INTENSITY_FEATURES, ROLE_FEATURES, WORKLOAD_FEATURES, add_workload,
    prior_season_role,
)
from injuryrisk.labels.absences import season_availability

FEATURE_GROUPS: dict[str, list[str]] = {
    "workload": WORKLOAD_FEATURES,
    "schedule": SCHEDULE_FEATURES,
    "availability": AVAILABILITY_FEATURES + AVAIL_INSEASON_FEATURES,
    "injury_history": INJURY_HISTORY_FEATURES,
    "physical": PHYSICAL_FEATURES,
    "role": ROLE_FEATURES,
    "intensity": INTENSITY_FEATURES,
}

ID_COLS = ["PLAYER_ID", "season", "TEAM_ID", "GAME_ID", "GAME_DATE", "stint_id", "game_num"]

#: Features allowed to be entirely NaN in a small or synthetic table.
OPTIONAL = {"pos_guard", "pos_forward", "pos_center"}


def feature_list(groups: list[str]) -> list[str]:
    unknown = set(groups) - set(FEATURE_GROUPS)
    assert not unknown, f"unknown feature groups {sorted(unknown)}"
    return [f for g in groups for f in FEATURE_GROUPS[g]]


def build_panel_features(panel: pd.DataFrame, team_games: pd.DataFrame,
                         arenas: pd.DataFrame | None = None) -> pd.DataFrame:
    """Row-wise features that need the full panel (absences included)."""
    p = panel.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    sched = team_schedule_features(team_games, arenas)
    p = p.merge(sched, on=["TEAM_ID", "GAME_ID"], how="left")
    p = add_workload(p)
    p = add_player_schedule(p)
    return p


def build_model_table(
    panel: pd.DataFrame,
    team_games: pd.DataFrame,
    spells: pd.DataFrame,
    events: pd.DataFrame,
    bio: pd.DataFrame,
    positions: pd.DataFrame | None,
    groups: list[str],
    shrink_games: float = 10.0,
    arenas: pd.DataFrame | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    p = build_panel_features(panel, team_games, arenas)
    rows = p[p["played"] == 1].reset_index(drop=True)

    seasons = season_availability(panel, team_games)
    rows = rows.merge(season_history(seasons, rows), on=["PLAYER_ID", "season"], how="left")
    rows = rows.merge(prior_season_role(panel, shrink_games), on=["PLAYER_ID", "season"], how="left")
    rows = rows.merge(physical_table(bio, positions), on=["PLAYER_ID", "season"], how="left")

    hist = pd.concat([spell_history(rows, panel, spells), event_history(rows, events)], axis=1)
    rows = pd.concat([rows, hist], axis=1)
    rows = combine(rows)
    for col in [c for c in rows.columns if c.startswith("days_since")]:
        rows[col] = rows[col].fillna(NO_HISTORY_DAYS)

    features = feature_list(groups)
    assert_features(rows, features, strict=strict)
    label_cols = [c for c in rows.columns if c.startswith("y_")]
    keep = ID_COLS + ["TEAM_ABBREVIATION", "MIN"] + label_cols + features
    return rows[keep]


def assert_features(df: pd.DataFrame, features: list[str], strict: bool = True) -> None:
    """Fail loudly: every feature exists, and (strict) none is entirely empty."""
    missing = [f for f in features if f not in df.columns]
    assert not missing, f"features never built: {missing}"
    if strict:
        empty = [f for f in features if f not in OPTIONAL and df[f].notna().sum() == 0]
        assert not empty, f"features built but entirely NaN: {empty}"
        constant = [f for f in features if f not in OPTIONAL and df[f].nunique(dropna=True) <= 1]
        assert not constant, f"features with a single value: {constant}"


def coverage_report(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rep = pd.DataFrame({
        "feature": features,
        "non_null": [df[f].notna().mean() for f in features],
        "mean": [pd.to_numeric(df[f], errors="coerce").mean() for f in features],
    })
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    rep.insert(0, "group", rep["feature"].map(group_of))
    return rep
