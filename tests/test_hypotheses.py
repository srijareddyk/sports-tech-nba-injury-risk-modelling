"""Helpers behind the hypothesis stage."""
import numpy as np
import pandas as pd

from injuryrisk.evaluate.metrics import budget_flags
from injuryrisk.hypotheses import holm, load_spec, ramp_ratio, team_status
from injuryrisk.labels.vocab import note_tissue_group


def test_holm_step_down():
    adj = holm({"A": 0.01, "B": 0.04, "C": 0.03})
    assert np.isclose(adj["A"], 0.03)
    assert np.isclose(adj["C"], 0.06)
    assert np.isclose(adj["B"], 0.06)       # monotone: never below the previous step


def test_team_status():
    panel = pd.DataFrame({
        "PLAYER_ID": [1, 1, 1, 2, 3, 3],
        "season":    [2015, 2016, 2016, 2016, 2015, 2016],
        "TEAM_ID":   [10, 10, 20, 10, 10, 30],
    })
    s = team_status(panel).set_index(["PLAYER_ID", "season", "TEAM_ID"])["team_status"]
    assert s[(1, 2016, 10)] == "same"
    assert s[(1, 2016, 20)] == "new"        # traded mid-season to a team he hadn't played for
    assert s[(2, 2016, 10)] == "no_prior"   # no previous season: in neither group
    assert s[(3, 2016, 30)] == "new"
    assert s[(1, 2015, 10)] == "no_prior"


def test_constant_score_flags_are_not_lowest_ids():
    frame = pd.DataFrame({"TEAM_ID": 1, "GAME_ID": np.repeat(np.arange(200), 10),
                          "PLAYER_ID": np.tile(np.arange(10), 200)})
    flags = budget_flags(frame, np.full(len(frame), 0.03), 2)
    per_player = pd.Series(flags).groupby(frame["PLAYER_ID"]).mean()
    assert flags.sum() == 400
    assert per_player.max() < 0.4           # row order no longer decides who is flagged


def test_spec_has_thresholds():
    spec = load_spec()
    for h in ("A", "B", "C", "D"):
        assert "threshold" in spec["hypotheses"][h]
    assert spec["multiple_testing"]["method"] == "holm"
    assert spec["multiple_testing"]["primary"] == ["A", "B", "C", "D"]
    d = spec["hypotheses"]["D"]
    assert d["threshold"] == {"statistic": "effect_pp_per_0.25", "direction": "greater", "value": 0.0}
    assert (d["baseline_games"], d["baseline_min_games"], d["baseline_min_mpg"], d["ramp_cap"]) == (15, 5, 10, 1.5)
    assert d["games_back"] == [1, 10]
    assert spec["hypotheses"]["A"]["secondary_analyses"]["tissue_type"]["in_holm_family"] is False


def test_tissue_mapping():
    cases = {
        "sprained left ankle": "ligament",
        "bruised right knee": "soft_tissue",
        "knee contusion": "soft_tissue",
        "torn ACL in right knee": "ligament",
        "torn right hamstring": "soft_tissue",
        "dislocated left shoulder": "joint",
        "fractured right wrist": "bone",
        "broken hand": "bone",
        "sore knee": "unspecified",
        "strained left calf": "soft_tissue",
        "torn meniscus": "joint",
        "ruptured achilles tendon": "soft_tissue",
        "patellar tendinitis": "soft_tissue",
        "plantar fasciitis": "soft_tissue",
        "stress fracture in left tibia": "bone",
        "torn ligament in thumb": "ligament",
        "torn knee": "unspecified",           # a tear with no muscle and no structure names no tissue
        "illness": "unspecified",
        # Rule changes 2026-11-27 (configs/hypotheses.yaml, amendments):
        "bone bruise in left knee": "bone",
        "sore right hamstring": "soft_tissue",
        "left calf injury": "soft_tissue",
        "tightness in groin": "soft_tissue",
        "left adductor injury": "soft_tissue",
        "sprained groin": "ligament",         # a sprain keeps its rule
        "sore right ankle": "unspecified",    # not a muscle region
    }
    for note, group in cases.items():
        assert note_tissue_group(note) == group, note


def _ramp_world(future_minutes: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.to_datetime
    pre = pd.date_range("2020-01-01", periods=20, freq="2D")       # pre-injury games
    post = pd.date_range("2020-03-01", periods=6, freq="2D")        # games after the return
    apps = pd.DataFrame({"PLAYER_ID": 1, "GAME_DATE": list(pre) + list(post),
                         "MIN": [20.0] * 5 + [30.0] * 15 + [10.0, 20.0, 30.0, future_minutes, 99.0, 99.0]})
    rows = pd.DataFrame({"PLAYER_ID": 1, "GAME_DATE": [post[3]],
                         "prev_start": [d("2020-02-15")], "prev_end": [d("2020-02-25")]})
    return rows, apps


def test_ramp_ratio_uses_only_games_before_t():
    rows, apps = _ramp_world(future_minutes=0.0)
    r = ramp_ratio(rows, apps, baseline_games=15, cap=1.5).iloc[0]
    assert r["baseline_games"] == 15 and np.isclose(r["baseline_mpg"], 30.0)   # last 15 before the spell
    assert r["games_back"] == 3                                                # games 0, 1, 2 since return
    assert np.isclose(r["ramp_mpg"], 20.0) and np.isclose(r["ramp_ratio"], 20 / 30)
    # Minutes at game t and after must not move the ratio.
    rows2, apps2 = _ramp_world(future_minutes=48.0)
    assert np.isclose(ramp_ratio(rows2, apps2).iloc[0]["ramp_ratio"], r["ramp_ratio"])


def test_ramp_ratio_cap_and_short_baseline():
    rows, apps = _ramp_world(future_minutes=0.0)
    apps = apps.copy()
    apps.loc[apps["GAME_DATE"] < "2020-02-15", "MIN"] = 5.0
    r = ramp_ratio(rows, apps, cap=1.5).iloc[0]
    assert np.isclose(r["ramp_ratio"], 1.5)                                    # 20 / 5 = 4, capped
    r3 = ramp_ratio(rows, apps.iloc[17:], cap=1.5).iloc[0]
    assert r3["baseline_games"] == 3                                           # caller drops < 5
