"""Leakage test: every feature for game t is computable before tip-off of t.

For a sample of at-risk rows, rebuild every feature from a world that ends at
tip-off of t: all panel rows after t removed, every event dated on or after t's
date removed, and game t's own box score replaced with noise. Each feature must
come out identical. A feature that reads the current row's box score, a future
game, or a not-yet-logged injury note fails here.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import BOS, BOX, LAL, make_events, make_logs
from injuryrisk.features.build import FEATURE_GROUPS, build_model_table, feature_list
from injuryrisk.labels import absences

NOTES = ["placed on IL with sprained left ankle", "placed on IL with sore right knee",
         "placed on IL with strained hamstring", "placed on IL with torn ACL in left knee (out for season)"]


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(1)
    games, events = [], []
    roster = {BOS: [101, 102, 103, 104, 105, 106], LAL: [201, 202, 203, 204, 205, 206]}
    out_until: dict[int, int] = {}
    for season, start in ((2015, "2014-10-28"), (2016, "2015-10-27")):
        date = pd.Timestamp(start)
        for i in range(30):
            date += pd.Timedelta(days=int(rng.integers(1, 4)))
            gid = f"{season}{i:03d}"
            played = []
            for team, pids in roster.items():
                for pid in pids:
                    if out_until.get(pid, -1) >= i:
                        continue
                    if rng.random() < 0.06:          # a new absence spell starts
                        out_until[pid] = i + int(rng.integers(0, 4))
                        if rng.random() < 0.7:
                            events.append((pid, date + pd.Timedelta(days=int(rng.integers(-1, 2))),
                                           NOTES[int(rng.integers(0, len(NOTES)))]))
                        continue
                    played.append((pid, team))
            games.append((season, gid, date, BOS if i % 2 else LAL, played))
        out_until.clear()
    logs = make_logs(games, rng)
    ev = make_events(events)
    ev["PLAYER_ID"] = ev["PLAYER_ID"].astype(float)
    bio = pd.DataFrame([{"PLAYER_ID": pid, "season": s, "AGE": 20 + pid % 15, "PLAYER_HEIGHT_INCHES": 75 + pid % 8,
                         "PLAYER_WEIGHT": 200 + pid % 40, "DRAFT_NUMBER": pid % 60}
                        for pids in roster.values() for pid in pids for s in (2015, 2016)])
    return logs, ev, bio


def _pipeline(panel, tg, ev, bio):
    panel = panel.drop(columns=[c for c in ("spell_id",) if c in panel.columns])
    spells = absences.attribute_spells(absences.find_spells(panel), ev)
    return build_model_table(panel, tg, spells, ev, bio, None, list(FEATURE_GROUPS), strict=False)


def test_features_use_only_the_past(world):
    logs, ev, bio = world
    tg = absences.build_team_games(logs)
    stints = absences.build_stints(logs, tg, ev)
    panel = absences.build_panel(logs, tg, stints)
    full = _pipeline(panel.copy(), tg, ev, bio)
    feats = feature_list(list(FEATURE_GROUPS))
    full = full.set_index(["PLAYER_ID", "GAME_ID"])

    rng = np.random.default_rng(2)
    risk = panel.index[(panel["played"] == 1) & (panel["season"] == 2016)]
    checked = 0
    for idx in rng.choice(risk, size=25, replace=False):
        row = panel.loc[idx]
        d = row["GAME_DATE"]
        past = panel[(panel["GAME_DATE"] < d) | (panel.index == idx)].copy()
        past.loc[idx, ["MIN", *BOX]] = rng.integers(0, 60, size=len(BOX) + 1).astype(float)
        tg_past = tg[tg["GAME_DATE"] <= d]
        ev_past = ev[ev["date"] < d]
        trunc = _pipeline(past, tg_past, ev_past, bio).set_index(["PLAYER_ID", "GAME_ID"])

        key = (row["PLAYER_ID"], row["GAME_ID"])
        a, b = full.loc[key, feats], trunc.loc[key, feats]
        leaked = [f for f in feats if not (np.isclose(a[f], b[f], equal_nan=True))]
        assert not leaked, f"row {key}: features depend on the current row or the future: {leaked}"
        checked += 1
    assert checked == 25


def test_every_group_is_nonempty():
    for group, feats in FEATURE_GROUPS.items():
        assert feats, group
    assert len(feature_list(list(FEATURE_GROUPS))) == len(set(feature_list(list(FEATURE_GROUPS))))
