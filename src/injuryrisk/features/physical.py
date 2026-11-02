"""Age, build and position, keyed on ``PLAYER_ID`` (never on names)."""
from __future__ import annotations

import numpy as np
import pandas as pd

PHYSICAL_FEATURES = ["age", "height_in", "weight_lb", "bmi", "draft_number",
                     "pos_guard", "pos_forward", "pos_center"]


def physical_table(bio: pd.DataFrame, positions: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (PLAYER_ID, season). Missing ages are filled from the player's
    other seasons (age + season difference), not from a league median."""
    b = bio.rename(columns={"AGE": "age", "PLAYER_HEIGHT_INCHES": "height_in",
                            "PLAYER_WEIGHT": "weight_lb", "DRAFT_NUMBER": "draft_number"})
    b = b[["PLAYER_ID", "season", "age", "height_in", "weight_lb", "draft_number"]].copy()
    b["birth_proxy"] = b["season"] - b["age"]
    b["birth_proxy"] = b.groupby("PLAYER_ID")["birth_proxy"].transform("median")
    b["age"] = b["age"].fillna(b["season"] - b["birth_proxy"])
    for col in ("height_in", "weight_lb", "draft_number"):
        b[col] = b[col].fillna(b.groupby("PLAYER_ID")[col].transform("median"))
    b["bmi"] = 703.0 * b["weight_lb"] / b["height_in"] ** 2
    if positions is not None:
        pos = positions.set_index("PLAYER_ID")["POSITION"].astype(str)
        p = b["PLAYER_ID"].map(pos).fillna("")
        b["pos_guard"] = p.str.contains("G").astype(float)
        b["pos_forward"] = p.str.contains("F").astype(float)
        b["pos_center"] = p.str.contains("C").astype(float)
        unknown = p == ""
        b.loc[unknown, ["pos_guard", "pos_forward", "pos_center"]] = np.nan
    else:
        b[["pos_guard", "pos_forward", "pos_center"]] = np.nan
    return b.drop(columns="birth_proxy")
