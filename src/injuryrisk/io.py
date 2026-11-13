"""Read/write processed tables. Pickle: fast, exact dtypes, no extra dependency."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from injuryrisk.paths import DATA_PROCESSED

#: Stage outputs, and the command that produces each one.
PRODUCED_BY = {
    "events": "events",
    "team_games": "labels",
    "panel": "labels",
    "spells": "labels",
    "model_table": "features",
    "predictions": "model",
}


def path_for(name: str) -> Path:
    return DATA_PROCESSED / f"{name}.pkl"


def save(df: pd.DataFrame, name: str) -> Path:
    path = path_for(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return path


def load(name: str) -> pd.DataFrame:
    path = path_for(name)
    if not path.exists():
        stage = PRODUCED_BY.get(name, "?")
        raise FileNotFoundError(f"{path} missing. Run `python run.py {stage}` first.")
    return pd.read_pickle(path)
