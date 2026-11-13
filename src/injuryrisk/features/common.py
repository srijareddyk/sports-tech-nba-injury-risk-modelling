"""Time-strict helpers shared by the feature builders.

Two primitives carry the leakage rule:

* :func:`lagged_window` sums a column over the ``k`` panel rows *before* the
  current one, within a group. The current row never enters.
* :func:`asof_count` / :func:`asof_last` look up events dated strictly before
  the query date. An event on game day is not visible at tip-off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NO_HISTORY_DAYS = 3650.0


def lagged_window(x: pd.Series, gid: pd.Series, k: int) -> tuple[pd.Series, pd.Series]:
    """(sum, count of non-null) of ``x`` over rows t-k .. t-1 within ``gid``.

    ``x`` must be ordered by time within each group. Implemented with group
    cumulative sums, so it is exact and fast on a million rows.
    """
    prev = x.groupby(gid).shift(1)
    filled = prev.fillna(0.0)
    present = prev.notna().astype(float)
    cs = filled.groupby(gid).cumsum()
    cn = present.groupby(gid).cumsum()
    s = cs - cs.groupby(gid).shift(k).fillna(0.0)
    n = cn - cn.groupby(gid).shift(k).fillna(0.0)
    return s, n


def lagged_mean(x: pd.Series, gid: pd.Series, k: int, min_count: int = 1) -> pd.Series:
    s, n = lagged_window(x, gid, k)
    return (s / n).where(n >= min_count)


def _prep(query: pd.DataFrame, key: str, date: str) -> pd.DataFrame:
    q = query[[key, date]].copy()
    q["_order"] = np.arange(len(q))
    q[key] = q[key].astype("int64")
    q[date] = pd.to_datetime(q[date]).astype("datetime64[ns]")
    return q.sort_values(date, kind="stable")


def asof_count(query: pd.DataFrame, events: pd.DataFrame, key: str = "PLAYER_ID",
               qdate: str = "GAME_DATE", edate: str = "date", offset_days: int = 0) -> np.ndarray:
    """Events with ``edate < query date - offset_days``, per key. Aligned to ``query``."""
    q = _prep(query, key, qdate)
    if offset_days:
        q[qdate] = q[qdate] - pd.Timedelta(days=offset_days)
        q = q.sort_values(qdate, kind="stable")
    if events.empty:
        return np.zeros(len(query))
    ev = events[[key, edate]].dropna().copy()
    ev[key] = ev[key].astype("int64")
    ev[edate] = pd.to_datetime(ev[edate]).astype("datetime64[ns]")
    ev = ev.sort_values([key, edate])
    ev["_cum"] = ev.groupby(key).cumcount() + 1
    # Equal dates within a key: keep the highest count so ties are all-or-nothing.
    ev = ev.drop_duplicates([key, edate], keep="last").sort_values(edate)
    m = pd.merge_asof(q, ev.rename(columns={edate: "_edate"}), left_on=qdate, right_on="_edate",
                      by=key, allow_exact_matches=False, direction="backward")
    return m.sort_values("_order")["_cum"].fillna(0).to_numpy(dtype=float)


def window_count(query: pd.DataFrame, events: pd.DataFrame, days: int, **kw) -> np.ndarray:
    """Events dated in ``[d - days, d)``: count(< d) - count(< d - days)."""
    return asof_count(query, events, **kw) - asof_count(query, events, offset_days=days, **kw)


def asof_last(query: pd.DataFrame, events: pd.DataFrame, value_cols: list[str],
              key: str = "PLAYER_ID", qdate: str = "GAME_DATE", edate: str = "date") -> pd.DataFrame:
    """The latest event strictly before each query date, aligned to ``query``.

    Returns ``value_cols`` plus ``_edate`` (the matched event's date).
    """
    q = _prep(query, key, qdate)
    ev = events[[key, edate, *[c for c in value_cols if c not in (key, edate)]]].dropna(subset=[key, edate]).copy()
    ev[key] = ev[key].astype("int64")
    ev["_edate"] = pd.to_datetime(ev[edate]).astype("datetime64[ns]")
    if edate not in value_cols:
        ev = ev.drop(columns=edate)
    ev = ev.sort_values("_edate")
    m = pd.merge_asof(q, ev, left_on=qdate, right_on="_edate", by=key,
                      allow_exact_matches=False, direction="backward")
    m = m.sort_values("_order").reset_index(drop=True)
    return m[["_edate", *[c for c in value_cols if c != key]]]


def days_between(later: pd.Series, earlier: pd.Series, fill: float = NO_HISTORY_DAYS) -> np.ndarray:
    d = (pd.to_datetime(later).to_numpy() - pd.to_datetime(earlier).to_numpy()) / np.timedelta64(1, "D")
    d = np.where(np.isnan(d), fill, d)
    return np.minimum(d, fill)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))
