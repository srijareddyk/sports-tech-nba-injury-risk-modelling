"""Closed vocabularies for reading injury notes, matched on word boundaries.

The clinical tables (``PATHOLOGY_TERMS``, ``STRUCTURE_TERMS``, ``REGION_TERMS``,
``SIDE_TERMS``), ``SEASON_ENDING_LANGUAGE``, ``LOAD_MANAGEMENT_LANGUAGE`` and
``parse_absence_days`` are ported from ``nba_fantasy_model``'s
``collect/news/injury_events.py``. The class vocabularies (protocol, suspension,
personal, illness) are new.

Every lookup goes through :func:`contains_any` / :func:`find_term`, which match on
word boundaries. Plain substring matching produced these bugs in the fantasy
project, and ``tests/test_vocab_boundaries.py`` pins each one:

* "shin" inside *Washington* read as a fracture
* "acl" inside *oracle*, *debacle*, *spectacle*
* "rest" inside "out for the **rest** of the season" filed a season-ending
  injury as a rest night. Boundaries alone don't fix this one: the word really
  is "rest". :func:`strip_season_ending` removes those phrases before any rest
  test.
"""
from __future__ import annotations

import re

Table = tuple[tuple[str, tuple[str, ...]], ...]

#: Injury kind, most severe first; the first hit wins.
PATHOLOGY_TERMS: Table = (
    ("torn", ("torn", "tore", "tear", "rupture", "ruptured")),
    ("fractured", ("fractured", "fracture", "broken", "break", "stress reaction")),
    ("dislocated", ("dislocated", "dislocation", "subluxation")),
    ("sprained", ("sprained", "sprain", "hyperextended", "hyperextension")),
    ("strained", ("strained", "strain", "pull", "pulled")),
    ("bruised", ("bruised", "bruise", "contusion", "bone bruise")),
    ("inflamed", ("tendinitis", "tendonitis", "tendinopathy", "bursitis",
                  "inflammation", "inflamed", "impingement", "fasciitis", "fasciaitis")),
    ("sore", ("sore", "soreness", "tightness", "discomfort", "spasms", "spasm", "stiffness")),
)

#: Anatomical structures, the clinically decisive ones first.
STRUCTURE_TERMS: Table = (
    ("achilles tendon", ("achilles",)),
    ("ACL", ("acl", "anterior cruciate")),
    ("patellar tendon", ("patellar tendon", "patella tendon", "patellar", "patella")),
    ("meniscus", ("meniscus", "meniscal")),
    ("quadriceps tendon", ("quadriceps tendon", "quad tendon")),
    ("rotator cuff", ("rotator cuff",)),
    ("labrum", ("labrum", "labral")),
    ("plantar fascia", ("plantar fascia", "plantar fasciitis")),
    ("MCL", ("mcl", "medial collateral")),
    ("PCL", ("pcl", "posterior cruciate")),
    ("UCL", ("ucl", "ulnar collateral")),
    ("tibia", ("tibia", "tibial")),
    ("fibula", ("fibula", "fibular")),
    ("metatarsal", ("metatarsal",)),
    ("navicular", ("navicular",)),
    ("scaphoid", ("scaphoid",)),
    ("ligament", ("ligament",)),
    ("tendon", ("tendon",)),
    ("cartilage", ("cartilage", "microfracture")),
)

#: Body regions, most specific first so "lower back" beats "back".
REGION_TERMS: Table = (
    ("lower back", ("lower back", "lumbar")),
    ("core muscle", ("core muscle", "abdominal", "abdomen", "oblique", "sports hernia")),
    ("knee", ("knee",)),
    ("ankle", ("ankle",)),
    ("shin", ("shin",)),
    ("foot", ("foot", "feet", "heel", "arch")),
    ("toe", ("toe", "big toe", "turf toe")),
    ("hamstring", ("hamstring", "hammy")),
    ("groin", ("groin", "adductor")),
    ("quadriceps", ("quadriceps", "quadricep", "quad")),
    ("calf", ("calf",)),
    ("hip", ("hip", "hip flexor")),
    ("thigh", ("thigh",)),
    ("shoulder", ("shoulder",)),
    ("elbow", ("elbow",)),
    ("wrist", ("wrist",)),
    ("thumb", ("thumb",)),
    ("finger", ("finger", "pinky", "index finger")),
    ("hand", ("hand",)),
    ("forearm", ("forearm",)),
    ("back", ("back",)),
    ("neck", ("neck", "cervical")),
    ("rib", ("rib", "ribs")),
    ("chest", ("chest", "pectoral", "sternum")),
    ("head", ("head", "concussion", "face", "facial")),
    ("jaw", ("jaw",)),
    ("nose", ("nose", "nasal")),
    ("eye", ("eye", "orbital")),
    ("leg", ("leg",)),
    ("arm", ("arm",)),
)

SIDE_TERMS: Table = (
    ("left", ("left",)),
    ("right", ("right",)),
    ("bilateral", ("bilateral", "both")),
)

#: Region groups big enough to hold counts. From ``injury_raw_events.py``.
REGION_GROUPS: dict[str, tuple[str, ...]] = {
    "knee": ("knee",),
    "lower_leg": ("ankle", "shin", "foot", "toe", "calf", "leg"),
    "upper_leg": ("hamstring", "groin", "quadriceps", "hip", "thigh"),
    "back_core": ("lower back", "back", "core muscle", "rib", "chest"),
    "upper_body": ("shoulder", "elbow", "wrist", "hand", "finger", "thumb", "arm", "forearm", "neck"),
    "head": ("head", "jaw", "nose", "eye"),
}

#: Structures that name no region still belong to a group.
STRUCTURE_GROUPS: dict[str, str] = {
    "achilles tendon": "lower_leg", "tibia": "lower_leg", "fibula": "lower_leg",
    "plantar fascia": "lower_leg", "metatarsal": "lower_leg", "navicular": "lower_leg",
    "ACL": "knee", "patellar tendon": "knee", "meniscus": "knee", "MCL": "knee", "PCL": "knee",
    "quadriceps tendon": "upper_leg",
    "rotator cuff": "upper_body", "labrum": "upper_body", "UCL": "upper_body", "scaphoid": "upper_body",
}

#: Structures whose damage is a long layoff on its own (prior-major flag).
MAJOR_STRUCTURES: tuple[str, ...] = (
    "achilles tendon", "ACL", "patellar tendon", "quadriceps tendon",
)

SEASON_ENDING_LANGUAGE: tuple[str, ...] = (
    "out for season", "out indefinitely", "out for the season", "season-ending",
    "season ending", "out for the year", "miss the season", "miss the entire season",
    "miss the rest of the season", "rest of the season", "rest of season",
    "done for the season", "sidelined for the season", "will not play this season",
    "ruled out for the season", "remainder of the season", "remainder of season",
)

SURGERY_LANGUAGE: tuple[str, ...] = (
    "surgery", "surgical", "arthroscopic", "arthroscopy", "repair", "repaired",
    "reconstruction", "reconstructive", "procedure", "operation",
)

#: Day-surgery wording: a scope or clean-up is not a major layoff.
MINOR_PROCEDURE_LANGUAGE: tuple[str, ...] = (
    "cleanup", "clean-up", "clean up", "debridement", "debrided",
    "loose bodies", "loose body", "loose cartilage", "scoped", "trim procedure",
)

#: A chip, hairline or avulsion fracture is often played through.
MINOR_FRACTURE_LANGUAGE: tuple[str, ...] = ("chip", "chipped", "hairline", "avulsion")

#: Rest and maintenance: a decision, not an outcome. "personal reasons" and
#: "conditioning" moved out of the fantasy list into their own classes.
LOAD_MANAGEMENT_LANGUAGE: tuple[str, ...] = (
    "load management", "injury management", "maintenance", "rest", "resting",
    "rested", "minutes restriction", "minutes limit", "dnp-rest", "dnp - rest",
    "dnp-cd", "coach's decision", "coaches decision", "precautionary",
    "conditioning", "not injury related",
)

#: Bare "protocol" is deliberately absent: "concussion protocol" is an injury.
PROTOCOL_LANGUAGE: tuple[str, ...] = (
    "health and safety", "health & safety", "covid", "covid-19", "coronavirus",
)

SUSPENSION_LANGUAGE: tuple[str, ...] = ("suspension", "suspended", "suspend")

PERSONAL_LANGUAGE: tuple[str, ...] = (
    "personal", "personal reasons", "bereavement", "family", "birth", "paternity",
    "leave of absence", "away from team", "not with team", "religious",
    "league discipline", "legal", "jail", "immigration", "visa", "death", "funeral",
    "arrest", "late for", "late to",
)

ILLNESS_LANGUAGE: tuple[str, ...] = (
    "illness", "ill", "flu", "flu-like", "virus", "viral", "infection", "infected",
    "sick", "food poisoning", "gastroenteritis", "stomach", "pneumonia", "strep",
    "dehydration", "mononucleosis", "mono", "appendicitis", "appendectomy",
    "migraine", "migraines", "headache", "fever", "cold", "bronchitis",
    "respiratory", "kidney stone", "kidney stones", "blood clot", "blood clots",
    "heart", "cardiac", "vertigo", "dental", "tooth", "wisdom teeth",
    "anemia", "asthma", "allergic", "allergy", "heartbeat", "tonsillitis", "sinus",
    "salmonella", "gastrointestinal", "root canal", "ailment",
)

#: Words that mark an injury even with no named body part.
INJURY_LANGUAGE: tuple[str, ...] = (
    "injury", "injured", "injuries", "concussion", "laceration", "lacerated", "bone spur",
    "hernia", "wound", "rehab", "rehabilitation", "recovering",
    "bone spurs", "stress reaction", "strained", "sprained", "sore",
)

#: Note padding PST adds that says nothing about the cause.
_BOILERPLATE_RE = re.compile(
    r"placed on (il|ir|injured list|inactive list)|activated from il|activated|ineligible|"
    r"\((f|p|cbc|cbc p|not cbc|dtd|out indefinitely|date approximate|date unsure|per [^)]*)\)",
    re.IGNORECASE,
)


def _pattern(surface: str) -> re.Pattern:
    return re.compile(rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9])")


_PATTERNS: dict[str, re.Pattern] = {}


def _rx(surface: str) -> re.Pattern:
    if surface not in _PATTERNS:
        _PATTERNS[surface] = _pattern(surface)
    return _PATTERNS[surface]


def contains_any(text: str | None, phrases: tuple[str, ...]) -> bool:
    """Word-boundary phrase search, case-insensitive."""
    low = (text or "").lower()
    return any(_rx(p).search(low) for p in phrases)


def find_term(text: str | None, table: Table) -> str | None:
    """First canonical term in ``table`` whose surface form appears in ``text``."""
    low = (text or "").lower()
    for canonical, surfaces in table:
        for surface in surfaces:
            if _rx(surface).search(low):
                return canonical
    return None


def strip_season_ending(text: str | None) -> str:
    """Remove season-ending phrases so the word "rest" inside them can't match."""
    low = (text or "").lower()
    for phrase in sorted(SEASON_ENDING_LANGUAGE, key=len, reverse=True):
        low = _rx(phrase).sub(" ", low)
    return low


def strip_boilerplate(text: str | None) -> str:
    """The part of a note that says something about the cause."""
    s = _BOILERPLATE_RE.sub(" ", (text or "").lower())
    s = re.sub(r"\(\s*\)", " ", s)
    s = re.sub(r"\b(with|for|on|a|an|the)\b", " ", s)
    return re.sub(r"[\s().,;:-]+", " ", s).strip()


def region_group(region: str | None, structure: str | None) -> str | None:
    for group, regions in REGION_GROUPS.items():
        if region in regions:
            return group
    return STRUCTURE_GROUPS.get(structure) if structure else None


# ---------------------------------------------------------------------------
# Stated absence duration (ported)
# ---------------------------------------------------------------------------

_DURATION_UNITS = {"day": 1, "week": 7, "month": 30, "game": 2}
_DURATION_RE = re.compile(
    r"(\d{1,3})\s*(?:to|-|–|or)?\s*(\d{1,3})?\s*(day|week|month|game)s?", re.IGNORECASE,
)
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
}
_WORD_DURATION_RE = re.compile(
    r"(" + "|".join(_WORD_NUMBERS) + r")\s*(?:to|-|–|or)?\s*("
    + "|".join(_WORD_NUMBERS) + r")?\s*(day|week|month|game)s?",
    re.IGNORECASE,
)
FORWARD_ABSENCE_CUES: tuple[str, ...] = (
    "expected to", "will be out", "will miss", "out for", "out at least",
    "return in", "returns in", "re-evaluated", "reevaluated", "timetable",
    "projected", "targeting", "sidelined for", "to miss", "ruled out for", "out ",
)
BACKWARD_ABSENCE_MARKERS: tuple[str, ...] = (
    "missed", "kept him out", "limited him", "played", "has been out",
    "was out", "had been out", "spent", "sat out", "returned",
)


def parse_absence_days(text: str | None) -> int | None:
    """Longest forward-looking stated absence in ``text``, in days, or None.

    The upper end of a range is taken: "4-6 weeks" -> 42 days.
    """
    if not text or not isinstance(text, str):
        return None
    low = text.lower()
    if any(m in low for m in BACKWARD_ABSENCE_MARKERS):
        return None
    if not any(c in low for c in FORWARD_ABSENCE_CUES):
        return None
    best: int | None = None
    for regex, lookup in ((_DURATION_RE, None), (_WORD_DURATION_RE, _WORD_NUMBERS)):
        for match in regex.finditer(text):
            lo, hi, unit = match.group(1), match.group(2), match.group(3).lower()
            values = [v for v in (lo, hi) if v]
            numbers = [lookup[v.lower()] for v in values] if lookup else [int(v) for v in values]
            if not numbers:
                continue
            days = max(numbers) * _DURATION_UNITS[unit]
            best = days if best is None else max(best, days)
    return best
