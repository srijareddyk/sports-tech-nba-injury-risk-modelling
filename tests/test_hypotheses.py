"""Helpers behind the hypothesis stage."""
import numpy as np
import pandas as pd

from injuryrisk.evaluate.metrics import budget_flags
from injuryrisk.hypotheses import holm, load_spec, team_status


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
    for h in ("A", "B", "C"):
        assert "threshold" in spec["hypotheses"][h]
    assert spec["multiple_testing"]["method"] == "holm"
