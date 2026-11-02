"""Word-boundary bugs found in the fantasy project, pinned so they can't return."""
from injuryrisk.labels import vocab as V
from injuryrisk.labels.classify import classify_note


def test_washington_is_not_a_shin():
    assert V.find_term("traded to Washington", V.REGION_TERMS) is None
    assert not V.contains_any("Washington Wizards", ("shin",))
    assert classify_note("placed on IL (Washington)")["region"] is None


def test_oracle_debacle_spectacle_are_not_acls():
    for text in ("played at Oracle Arena", "a debacle", "quite a spectacle"):
        assert V.find_term(text, V.STRUCTURE_TERMS) is None, text
    assert V.find_term("torn ACL in left knee", V.STRUCTURE_TERMS) == "ACL"


def test_rest_of_season_is_not_a_rest_night():
    for note in ("placed on IL with sprained left ankle (out for the rest of the season)",
                 "placed on IL with sore knee, will miss the rest of the season",
                 "placed on IL (out for season)"):
        assert classify_note(note)["event_class"] == "injury_major", note


def test_rest_is_rest():
    assert classify_note("placed on IL for rest")["event_class"] == "rest"
    assert classify_note("placed on IL with knee injury / rest")["event_class"] == "rest"
    assert classify_note("placed on IL with injury management")["event_class"] == "rest"


def test_structural_damage_beats_rest_language():
    assert classify_note("placed on IL with fractured hand (rest)")["event_class"] == "injury_major"


def test_classes():
    cases = {
        "placed on IL with sprained left ankle": "injury_minor",
        "placed on IL with torn ACL in right knee (out for season)": "injury_major",
        "placed on IL recovering from surgery on left knee": "injury_major",
        "placed on IL with chip fracture in wrist": "injury_minor",
        "placed on IL with illness": "illness",
        "placed on IL with flu": "illness",
        "placed on IL with NBA health and safety protocols": "protocol",
        "placed on IL with concussion (concussion protocol)": "injury_minor",
        "placed on IL for personal reasons": "personal",
        "placed on IL (suspension)": "suspension",
        "placed on IL": "stash",
        "placed on IL (P)": "stash",
        "placed on IL (date approximate)": "stash",
    }
    for note, expected in cases.items():
        assert classify_note(note)["event_class"] == expected, (note, classify_note(note)["event_class"])


def test_region_groups():
    assert classify_note("placed on IL with sore left knee")["region_group"] == "knee"
    assert classify_note("placed on IL with torn Achilles tendon")["region_group"] == "lower_leg"
    assert classify_note("placed on IL with torn Achilles tendon")["major_structure"]


def test_parse_absence_days():
    assert V.parse_absence_days("expected to miss 4-6 weeks") == 42
    assert V.parse_absence_days("will be out two months") == 60
    assert V.parse_absence_days("missed 52 games last season") is None
