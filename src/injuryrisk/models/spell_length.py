"""Spell length: games missed per new injury spell (spec 4.3, separate target).

Rows are the at-risk games followed by an injury spell (``y_any_h1 = 1``).
Two variants:

* ``pre_injury``: features known before tip-off only. "If he gets hurt, how long?"
* ``with_note``: adds what the injury note says once it is logged (major class,
  region group, surgery, major structure). "Now that it's reported, how long?"

Censored spells (running into season end) have unknown length and are dropped.
That biases lengths down for late-season injuries; say so when reporting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from injuryrisk.evaluate.splits import forward_chain
from injuryrisk.labels.vocab import REGION_GROUPS

NOTE_FEATURES = ["note_major", "note_surgery", "note_major_structure",
                 *(f"note_region_{g}" for g in REGION_GROUPS)]


def spell_rows(table: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    """At-risk rows whose next game starts an injury spell, joined to that spell."""
    pos = table[table["y_any_h1"] == 1].copy()
    pos["start_num"] = pos["game_num"] + 1
    sp = spells[spells["severity"].isin(["minor", "major"])]
    sp = sp[["stint_id", "start_num", "length", "censored", "cause", "cause_surgery",
             "cause_major_structure", "cause_region_group"]]
    rows = pos.merge(sp, on=["stint_id", "start_num"], how="inner")
    rows["note_major"] = (rows["cause"] == "injury_major").astype(float)
    rows["note_surgery"] = rows["cause_surgery"].fillna(False).astype(float)
    rows["note_major_structure"] = rows["cause_major_structure"].fillna(False).astype(float)
    for g in REGION_GROUPS:
        rows[f"note_region_{g}"] = (rows["cause_region_group"] == g).astype(float)
    return rows


def run(table: pd.DataFrame, spells: pd.DataFrame, cfg: dict, features: list[str]) -> pd.DataFrame:
    rows = spell_rows(table, spells)
    rows = rows[~rows["censored"]]
    out = []
    for split in forward_chain(cfg, sorted(table["season"].unique())):
        fit = rows[rows["season"].isin(split.fit)]
        test = rows[rows["season"] == split.test]
        if len(test) < 20:
            continue
        y = test["length"].to_numpy(float)
        preds = {
            "median_all": np.full(len(test), fit["length"].median()),
            "median_by_note_class": test["note_major"].map(
                fit.groupby("note_major")["length"].median()).fillna(fit["length"].median()).to_numpy(),
        }
        for name, cols in (("gbm_pre_injury", features), ("gbm_with_note", features + NOTE_FEATURES)):
            m = HistGradientBoostingRegressor(loss="poisson", max_iter=200, learning_rate=0.05,
                                              max_leaf_nodes=15, min_samples_leaf=100, random_state=7)
            m.fit(fit[cols], fit["length"])
            preds[name] = m.predict(test[cols])
        for name, p in preds.items():
            rho = spearmanr(p, y).statistic if np.std(p) > 0 else np.nan
            out.append({"test_season": split.test, "model": name, "spells": len(test),
                        "mean_length": y.mean(), "median_length": np.median(y),
                        "mae": np.mean(np.abs(p - y)), "spearman": rho})
    return pd.DataFrame(out)
