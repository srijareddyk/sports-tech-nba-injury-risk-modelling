"""Pro Sports Transactions (PST) injury log: load, clean, map to ``PLAYER_ID``.

PST identifies players by name and teams by nickname. Both are mapped to NBA ids
here, once, so nothing downstream ever joins on a name (the old notebook lost
~1,000 rows to name mismatches).

Name cleanup:
* the 2018-23 export prefixes names with a ``• `` bullet, sometimes mis-decoded;
  any leading non-word characters are stripped
* ``"Raulzinho Neto / Raul Neto"`` lists aliases; every alias is tried
* ``"John Collins (Martin)"`` carries a disambiguator; the base name is tried
* accents, punctuation and Jr./Sr./II/III suffixes are folded away

Matching runs in stages, strictest first, and the stage that matched is kept on
every row so the unmapped count and match quality are both reportable.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from injuryrisk.config import date_to_season
from injuryrisk.paths import DATA_RAW

#: PST nickname -> NBA team id. "Hornets" is resolved by date below.
TEAM_IDS: dict[str, int] = {
    "Hawks": 1610612737, "Celtics": 1610612738, "Cavaliers": 1610612739,
    "Pelicans": 1610612740, "Bulls": 1610612741, "Mavericks": 1610612742,
    "Nuggets": 1610612743, "Warriors": 1610612744, "Rockets": 1610612745,
    "Clippers": 1610612746, "Lakers": 1610612747, "Heat": 1610612748,
    "Bucks": 1610612749, "Timberwolves": 1610612750, "Nets": 1610612751,
    "Knicks": 1610612752, "Magic": 1610612753, "Pacers": 1610612754,
    "76ers": 1610612755, "Sixers": 1610612755, "Suns": 1610612756,
    "Blazers": 1610612757, "Trail Blazers": 1610612757, "Kings": 1610612758,
    "Spurs": 1610612759, "Thunder": 1610612760, "Sonics": 1610612760,
    "SuperSonics": 1610612760, "Raptors": 1610612761, "Jazz": 1610612762,
    "Grizzlies": 1610612763, "Wizards": 1610612764, "Bullets": 1610612764,
    "Pistons": 1610612765, "Bobcats": 1610612766,
}
NEW_ORLEANS_ID = 1610612740
CHARLOTTE_ID = 1610612766
#: New Orleans played as the Hornets through 2012-13; Charlotte took the name in 2014-15.
HORNETS_SWITCH = pd.Timestamp("2013-07-01")

_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def fold_name(name: str) -> str:
    """Lowercase, strip accents, punctuation and generational suffixes."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z ]", "", s)
    s = _SUFFIX_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_raw_name(name) -> str:
    """Strip the leading bullet (in whatever encoding it survived as) and spaces."""
    if not isinstance(name, str):
        return ""
    return re.sub(r"^[^\w(]+", "", name.strip()).strip()


def name_candidates(name: str) -> list[str]:
    """Folded spellings to try for one PST name, most literal first."""
    if not name:
        return []
    raw = [name]
    if " (" in name:
        raw.append(name.split(" (")[0])
    for part in name.split(" / "):
        raw.append(part)
        if " (" in part:
            raw.append(part.split(" (")[0])
    out: list[str] = []
    for r in raw:
        f = fold_name(r)
        if f and f not in out:
            out.append(f)
    return out


def _read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"cannot decode {path}")


def load_pst(files: list[str], raw_dir: Path | None = None) -> pd.DataFrame:
    """Union of the PST exports, one row per (date, team, player, direction, note)."""
    raw_dir = raw_dir or DATA_RAW
    frames = []
    for name in files:
        path = raw_dir / name
        if not path.exists():
            raise FileNotFoundError(f"PST export not found: {path}")
        df = _read_csv_any(path)
        df = df[[c for c in ("Date", "Team", "Acquired", "Relinquished", "Notes") if c in df.columns]]
        df["source_file"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["date"])

    rel = df["Relinquished"].map(clean_raw_name)
    acq = df["Acquired"].map(clean_raw_name)
    # A "Relinquished" row is a player leaving the active roster (placed on IL);
    # an "Acquired" row is a return (activated from IL).
    out = pd.DataFrame({
        "date": df["date"],
        "team_name": df["Team"].astype(str).str.strip(),
        "player_name": np.where(rel != "", rel, acq),
        "direction": np.where(rel != "", "out", "in"),
        "notes": df["Notes"].fillna("").astype(str).str.strip(),
        "source_file": df["source_file"],
    })
    out = out[out["player_name"] != ""]
    out["_key"] = out["player_name"].map(fold_name) + "|" + out["notes"].str.lower()
    out = out.drop_duplicates(subset=["date", "team_name", "direction", "_key"]).drop(columns="_key")
    out["season"] = date_to_season(out["date"])
    out["team_id"] = out["team_name"].map(TEAM_IDS)
    hornets = out["team_name"] == "Hornets"
    out.loc[hornets, "team_id"] = np.where(
        out.loc[hornets, "date"] < HORNETS_SWITCH, NEW_ORLEANS_ID, CHARLOTTE_ID
    )
    return out.sort_values("date").reset_index(drop=True)


def _build_lookups(logs: pd.DataFrame) -> dict[str, dict]:
    """Name keys -> sets of PLAYER_IDs, from the game logs (the id authority)."""
    pairs = logs[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "season"]].drop_duplicates()
    pairs = pairs.assign(fname=pairs["PLAYER_NAME"].map(fold_name))
    lookups: dict[str, dict] = {"nts": {}, "ns": {}, "n": {}}
    for pid, fname, team, season in pairs[["PLAYER_ID", "fname", "TEAM_ID", "season"]].itertuples(index=False):
        lookups["nts"].setdefault((fname, int(team), int(season)), set()).add(int(pid))
        lookups["ns"].setdefault((fname, int(season)), set()).add(int(pid))
        lookups["n"].setdefault(fname, set()).add(int(pid))
    return lookups


def map_player_ids(events: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Attach ``PLAYER_ID`` and ``match_stage``. Unique matches only, never a guess."""
    lk = _build_lookups(logs)

    def resolve(name: str, team, season: int) -> tuple[float, str]:
        cands = name_candidates(name)
        team = int(team) if pd.notna(team) else None
        stages = []
        if team is not None:
            stages.append(("name_team_season", [(lk["nts"], (c, team, season)) for c in cands]))
            stages.append(("name_team_adjacent",
                           [(lk["nts"], (c, team, season + d)) for c in cands for d in (-1, 1)]))
        stages.append(("name_season", [(lk["ns"], (c, season)) for c in cands]))
        stages.append(("name_only", [(lk["n"], c) for c in cands]))
        for stage, probes in stages:
            ids: set[int] = set()
            for table, key in probes:
                ids |= table.get(key, set())
            if len(ids) == 1:
                return float(next(iter(ids))), stage
            if len(ids) > 1:
                return np.nan, f"ambiguous_{stage}"
        return np.nan, "unmapped"

    keys = events[["player_name", "team_id", "season"]].drop_duplicates()
    resolved = {
        (n, t, s): resolve(n, t, int(s))
        for n, t, s in keys.itertuples(index=False)
    }
    out = events.copy()
    res = [resolved[(n, t, s)] for n, t, s in out[["player_name", "team_id", "season"]].itertuples(index=False)]
    out["PLAYER_ID"] = [r[0] for r in res]
    out["match_stage"] = [r[1] for r in res]
    return out


def mapping_report(events: pd.DataFrame, first_season: int) -> pd.DataFrame:
    """Match-stage counts for events inside the modelled seasons."""
    window = events[events["season"] >= first_season]
    rep = window["match_stage"].value_counts().rename_axis("match_stage").reset_index(name="events")
    rep["share"] = rep["events"] / rep["events"].sum()
    return rep
