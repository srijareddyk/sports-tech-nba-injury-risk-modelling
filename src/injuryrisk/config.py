"""Load ``configs/base.yaml`` and expose a few derived helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from injuryrisk.paths import CONFIGS


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else CONFIGS / "base.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = str(path)
    return cfg


def config_hash(cfg: dict) -> str:
    """Short stable hash of the config, written next to every logged result."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha1(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:10]


def season_label(end_year: int) -> str:
    """2006 -> '2005-06'. Seasons are named by end year everywhere."""
    return f"{end_year - 1}-{str(end_year)[2:]}"


def all_seasons(cfg: dict) -> list[int]:
    return list(range(int(cfg["seasons"]["first"]), int(cfg["seasons"]["last"]) + 1))


def date_to_season(dates):
    """Season end year for a date: August onward belongs to the next season."""
    return dates.dt.year + (dates.dt.month >= 8).astype(int)
