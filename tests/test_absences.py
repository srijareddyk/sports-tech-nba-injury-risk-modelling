"""Schedule-minus-appearances: traded players, short seasons, spells, extension."""
import pandas as pd

from conftest import BOS, LAL, make_events, make_logs
from injuryrisk.labels import absences, target


def _schedule(season, n, start, team_players):
    """n BOS-LAL games, one every 2 days. ``team_players(i)`` -> who played game i."""
    games = []
    for i in range(n):
        date = pd.Timestamp(start) + pd.Timedelta(days=2 * i)
        games.append((season, f"{season}{i:03d}", date, BOS if i % 2 == 0 else LAL, team_players(i)))
    return games


def _build(logs, events=None, ext=7):
    tg = absences.build_team_games(logs)
    stints = absences.build_stints(logs, tg, events, ext)
    panel = absences.build_panel(logs, tg, stints)
    spells = absences.find_spells(panel)
    return tg, stints, panel, spells


def test_short_season_schedule_is_counted_not_assumed(rng):
    logs = make_logs(
        _schedule(2012, 5, "2012-01-01", lambda i: [(1, BOS), (2, LAL)])
        + _schedule(2013, 8, "2012-11-01", lambda i: [(1, BOS), (2, LAL)]), rng)
    tg = absences.build_team_games(logs)
    lengths = absences.schedule_lengths(tg)
    assert lengths.loc[2012, "max"] == 5
    assert lengths.loc[2013, "max"] == 8


def test_gap_inside_span_is_an_absence(rng):
    # Player 1 misses games 3 and 4 (0-based 2, 3) of 8.
    logs = make_logs(_schedule(2013, 8, "2012-11-01",
                               lambda i: [(2, LAL), (5, BOS)] + ([] if i in (2, 3) else [(1, BOS)])), rng)
    _, _, panel, spells = _build(logs)
    p1 = spells[spells["PLAYER_ID"] == 1]
    assert len(p1) == 1
    assert p1["length"].iloc[0] == 2
    assert not p1["censored"].iloc[0]
    assert panel[panel["PLAYER_ID"] == 1]["played"].sum() == 6


def test_traded_player_measured_against_longest_schedule(rng):
    # Player 3 plays games 0-3 for BOS, then games 4-7 for LAL. No absence.
    def who(i):
        base = [(1, BOS), (2, LAL)]
        return base + [(3, BOS if i < 4 else LAL)]
    logs = make_logs(_schedule(2013, 8, "2012-11-01", who), rng)
    tg, stints, panel, spells = _build(logs)
    assert len(stints[stints["PLAYER_ID"] == 3]) == 2
    assert spells[spells["PLAYER_ID"] == 3].empty
    season = absences.season_availability(panel, tg)
    row = season[season["PLAYER_ID"] == 3].iloc[0]
    assert row["played"] == 8 and row["scheduled"] == 8 and row["missed"] == 0


def test_season_ending_injury_extends_the_span(rng):
    # Player 4 plays games 0-2 of 8, then an injury note one day after game 2.
    logs = make_logs(_schedule(2013, 8, "2012-11-01",
                               lambda i: [(1, BOS), (2, LAL)] + ([(4, BOS)] if i <= 2 else [])), rng)
    last = logs.loc[logs["PLAYER_ID"] == 4, "GAME_DATE"].max()
    ev = make_events([(4, last + pd.Timedelta(days=1), "placed on IL with torn ACL in left knee (out for season)")])
    _, stints, panel, spells = _build(logs, ev)
    assert stints.loc[stints["PLAYER_ID"] == 4, "extended"].iloc[0]
    s = absences.attribute_spells(spells, ev)
    s4 = s[s["PLAYER_ID"] == 4].iloc[0]
    assert s4["length"] == 5 and s4["censored"]
    assert s4["cause"] == "injury_major" and s4["severity"] == "major"

    # Without the note, no extension and no absence.
    _, stints2, _, spells2 = _build(logs, None)
    assert not stints2.loc[stints2["PLAYER_ID"] == 4, "extended"].iloc[0]
    assert spells2[spells2["PLAYER_ID"] == 4].empty


def test_cause_attribution_and_target(rng):
    logs = make_logs(_schedule(2013, 10, "2012-11-01",
                               lambda i: [(2, LAL), (5, BOS)] + ([] if i in (3, 4, 7) else [(1, BOS)])), rng)
    dates = sorted(logs["GAME_DATE"].unique())
    ev = make_events([
        (1, dates[3] + pd.Timedelta(days=1), "placed on IL with sprained right ankle"),
        (1, dates[7], "placed on IL for rest"),
    ])
    _, _, panel, spells = _build(logs, ev)
    s = absences.attribute_spells(spells, ev).sort_values("start_num")
    assert list(s.loc[s["PLAYER_ID"] == 1, "cause"]) == ["injury_minor", "rest"]
    panel = target.attach_targets(panel, s, [1, 3])
    p1 = panel[(panel["PLAYER_ID"] == 1) & (panel["played"] == 1)].set_index("game_num")
    # Game 3 (1-based) precedes the injury spell at game 4.
    assert p1.loc[3, "y_any_h1"] == 1
    assert p1.loc[2, "y_any_h1"] == 0 and p1.loc[2, "y_any_h3"] == 1
    # The rest spell at game 8 is not an injury.
    assert p1.loc[7, "y_any_h1"] == 0
    # Not enough games left for H = 3 and no positive: censored.
    assert pd.isna(p1.loc[9, "y_any_h3"])
