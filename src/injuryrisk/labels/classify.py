"""Event -> exactly one class (spec section 4.1).

==================  =======================================  ==============
class               example notes                            in target?
==================  =======================================  ==============
``injury_major``    torn ACL, fracture, surgery, out for     yes (major)
                    season, stated return >= 4 weeks
``injury_minor``    sprained ankle, sore knee, strain        yes (minor)
``rest``            rest, load/injury management, DNP-CD     no: a decision
``illness``         illness, flu, virus                      no
``protocol``        health and safety protocols, COVID       no
``suspension``      suspension                               no
``personal``        personal reasons, bereavement            no
``stash``           "placed on IL" and nothing else          no
``other``           text the vocabulary does not recognise   no
``activation``      returns ("activated from IL")            n/a
==================  =======================================  ==============

Precedence: protocol > suspension > personal > illness > rest > major > minor
> stash > other. Two exceptions: structural damage (surgery, tear, fracture)
beats rest, because "rest" in such notes is never the cause. And illness yields
to injury when a body part is named ("sore knee, flu-like symptoms" is still a
knee).

Run ``python run.py audit-sample`` to draw a stratified sample for hand
labelling, then ``python run.py audit-score`` for per-class precision (M2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from injuryrisk.labels import vocab as V

INJURY_CLASSES = ("injury_major", "injury_minor")
EXCLUDED_CLASSES = ("rest", "illness", "protocol", "suspension", "personal", "stash", "other")
ALL_CLASSES = INJURY_CLASSES + EXCLUDED_CLASSES

#: Ranking used when several events could explain one absence.
CAUSE_PRIORITY = {
    "injury_major": 0, "injury_minor": 1, "illness": 2, "protocol": 3, "suspension": 4,
    "personal": 5, "rest": 6, "other": 7, "stash": 8,
}

MAJOR_STATED_DAYS = 28


def classify_note(note: str | None) -> dict:
    """Parse one note. Returns the class plus the vocabulary hits behind it."""
    text = (note or "").lower()
    pathology = V.find_term(text, V.PATHOLOGY_TERMS)
    structure = V.find_term(text, V.STRUCTURE_TERMS)
    region = V.find_term(text, V.REGION_TERMS)
    side = V.find_term(text, V.SIDE_TERMS)
    season_ending = V.contains_any(text, V.SEASON_ENDING_LANGUAGE)
    surgery = V.contains_any(text, V.SURGERY_LANGUAGE)
    minor_procedure = V.contains_any(text, V.MINOR_PROCEDURE_LANGUAGE)
    minor_fracture = V.contains_any(text, V.MINOR_FRACTURE_LANGUAGE)
    stated = V.parse_absence_days(note)

    body_named = bool(region or structure)
    injury_named = body_named or bool(pathology) or V.contains_any(text, V.INJURY_LANGUAGE)
    structural = (
        (surgery and not minor_procedure)
        or pathology == "torn"
        or (pathology == "fractured" and not minor_fracture)
    )
    rest_text = V.strip_season_ending(text)

    if V.contains_any(text, V.PROTOCOL_LANGUAGE):
        cls = "protocol"
    elif V.contains_any(text, V.SUSPENSION_LANGUAGE):
        cls = "suspension"
    elif V.contains_any(text, V.PERSONAL_LANGUAGE) and not injury_named:
        cls = "personal"
    elif V.contains_any(text, V.ILLNESS_LANGUAGE) and not body_named:
        cls = "illness"
    elif V.contains_any(rest_text, V.LOAD_MANAGEMENT_LANGUAGE) and not structural:
        cls = "rest"
    elif structural or season_ending or (stated is not None and stated >= MAJOR_STATED_DAYS):
        cls = "injury_major"
    elif injury_named or surgery:
        cls = "injury_minor"
    elif not V.strip_boilerplate(text):
        cls = "stash"
    else:
        cls = "other"

    return {
        "event_class": cls,
        "pathology": pathology,
        "structure": structure,
        "region": region,
        "region_group": V.region_group(region, structure),
        "side": side,
        "surgery": bool(surgery and not minor_procedure),
        "season_ending": bool(season_ending),
        "major_structure": structure in V.MAJOR_STRUCTURES,
        "stated_days": stated,
    }


def classify_events(events: pd.DataFrame) -> pd.DataFrame:
    """Classify every ``out`` event; ``in`` events become ``activation``."""
    out = events.copy()
    notes = out["notes"].fillna("")
    cache = {n: classify_note(n) for n in notes.unique()}
    parsed = pd.DataFrame([cache[n] for n in notes], index=out.index)
    out = pd.concat([out, parsed], axis=1)
    is_in = out["direction"] == "in"
    out.loc[is_in, "event_class"] = "activation"
    out["is_injury"] = out["event_class"].isin(INJURY_CLASSES)
    return out


def class_report(events: pd.DataFrame, first_season: int) -> pd.DataFrame:
    window = events[(events["season"] >= first_season) & (events["direction"] == "out")]
    rep = window["event_class"].value_counts().rename_axis("event_class").reset_index(name="events")
    rep["share"] = rep["events"] / rep["events"].sum()
    return rep


# ---------------------------------------------------------------------------
# Hand-labelled audit (M2)
# ---------------------------------------------------------------------------

def audit_sample(events: pd.DataFrame, first_season: int, n: int = 240, seed: int = 7) -> pd.DataFrame:
    """Stratified sample of distinct notes for hand labelling.

    Equal allocation across classes (capped by availability), so rare classes
    get enough rows for a per-class precision. Fill ``true_class`` by hand.
    """
    pool = events[(events["season"] >= first_season) & (events["direction"] == "out")]
    pool = pool.drop_duplicates("notes")
    classes = sorted(pool["event_class"].unique())
    per = int(np.ceil(n / len(classes)))
    parts = [
        g.sample(min(len(g), per), random_state=seed)
        for _, g in pool.groupby("event_class")
    ]
    sample = pd.concat(parts)
    # Small classes can't fill their share; top up from the rest of the pool.
    short = n - len(sample)
    if short > 0:
        rest = pool.drop(sample.index)
        sample = pd.concat([sample, rest.sample(min(short, len(rest)), random_state=seed)])
    cols = ["date", "team_name", "player_name", "notes", "event_class"]
    sample = sample[cols].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sample["true_class"] = ""
    return sample


def audit_score(labelled: pd.DataFrame) -> pd.DataFrame:
    """Per-class precision of the classifier against the hand labels."""
    df = labelled[labelled["true_class"].fillna("").astype(str).str.strip() != ""].copy()
    if df.empty:
        raise ValueError("No rows have true_class filled in yet.")
    df["true_class"] = df["true_class"].str.strip()
    df["correct"] = df["true_class"] == df["event_class"]
    rep = df.groupby("event_class").agg(labelled=("correct", "size"), precision=("correct", "mean"))
    recall = df.groupby("true_class")["correct"].mean().rename("recall")
    rep = rep.join(recall, how="outer").reset_index().rename(columns={"index": "class"})
    total = pd.DataFrame([{"event_class": "ALL", "labelled": len(df), "precision": df["correct"].mean()}])
    return pd.concat([rep, total], ignore_index=True)
