"""The three pre-stated hypotheses (``configs/hypotheses.yaml``).

* **A, the return window (headline).** Is a player's risk of a new injury
  spell raised in his first games back from injury, against his own later
  baseline? Player fixed-effects hazard on the H = 1 risk set.
* **B, acquired players.** For a player in his first season with a new team,
  does public injury history rank risk better than age, minutes and role?
  Paired player-bootstrap PR-AUC difference, pooled forward-chain test seasons.
* **C, schedule density.** Is a player more likely to miss the next game with
  a new injury spell when it is the next day? Player fixed effects, dose
  response over the gap, illness placebo.

Every primary test yields a one-sided p-value against its stated threshold;
Holm is applied across A, B and C. Every result is written to ``reports/``,
null or not.
"""
from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from injuryrisk.causal.b2b import BASE_CONTROLS, NEXT_GAP_DUMMIES, estimate, fe_fit, prepare
from injuryrisk.evaluate.bootstrap import PlayerBootstrap, ci
from injuryrisk.evaluate.metrics import safe_ap
from injuryrisk.evaluate.splits import forward_chain, usable_seasons
from injuryrisk.features.common import asof_last
from injuryrisk.models.baselines import HISTORY_AGE_ROLE, BaseRate, logistic
from injuryrisk.paths import CONFIGS

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

#: Games since return: label, lowest, highest. 41+ is the reference.
A_BINS = [("0", 0, 0), ("1-2", 1, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-40", 21, 40)]
A_EARLY = [("0-5", 0, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-40", 21, 40)]
A_REF_MIN = 41
#: Null-benchmark simulations per Hypothesis A variant. Added after the first
#: run showed implausible negative ratios on short panels (see ``_null_coef``);
#: it adjusts robustness rows only and leaves the pre-stated primary test as is.
NULL_REPS = 20

#: What a front office knows about a player it doesn't own, without injury data.
FRONT_OFFICE = ["age", "mpg_prior", "mpg_last10_played", "usage_prior"]
#: Rest before the current game, the control in Hypothesis C.
C_REST = ["b2b", "three_in_four"]
C_CONTROLS = BASE_CONTROLS + C_REST

LENGTH_BINS = [("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-9", 5, 9), ("10+", 10, 10_000)]


def load_spec(path: str | Path | None = None) -> dict:
    with open(Path(path) if path else CONFIGS / "hypotheses.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def phi(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    adj, running = {}, 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[name]))
        adj[name] = running
    return adj


def add_spell_context(table: pd.DataFrame, panel: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    """Previous-spell severity/region/length and the next spell's region/length, per row.

    The previous spell uses the same visibility rule as the features
    (``known_date`` strictly before tip-off). The next spell is the one whose
    start makes ``y_*_h1`` positive: read only for outcomes, never as a feature.
    """
    d = table.reset_index(drop=True)
    inj = spells[spells["severity"].isin(["minor", "major"])]
    last = asof_last(d[["PLAYER_ID", "GAME_DATE"]], inj, ["severity", "cause_region_group", "length"],
                     edate="known_date")
    d["prev_severity"] = last["severity"].to_numpy()
    d["prev_region"] = last["cause_region_group"].to_numpy()

    p = panel.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"])
    nxt = p.groupby("stint_id")["spell_id"].shift(-1)
    sp = spells.set_index("spell_id")
    nx = pd.DataFrame({"PLAYER_ID": p["PLAYER_ID"].to_numpy(), "GAME_ID": p["GAME_ID"].to_numpy(),
                       "next_region": nxt.map(sp["cause_region_group"]).to_numpy(),
                       "next_length": nxt.map(sp["length"]).to_numpy()})
    d = d.merge(nx, on=["PLAYER_ID", "GAME_ID"], how="left")

    y = d["y_any_h1"]
    same = (y == 1) & d["prev_region"].notna() & (d["next_region"] == d["prev_region"])
    d["y_same_region_h1"] = np.where(y.isna(), np.nan, same.astype(float))
    d["y_diff_region_h1"] = np.where(y.isna(), np.nan, ((y == 1) & ~same).astype(float))
    return d


def label_drift(d: pd.DataFrame) -> pd.DataFrame:
    """New injury spells per 1,000 at-risk player-games, by eventual spell length (Step 0.1)."""
    rows = []
    for season, g in d.dropna(subset=["y_any_h1"]).groupby("season"):
        row = {"season": int(season), "at_risk_rows": len(g)}
        pos = g[g["y_any_h1"] == 1]
        for name, lo, hi in LENGTH_BINS:
            row[f"len_{name}"] = 1000 * float(pos["next_length"].between(lo, hi).sum()) / len(g)
        row["all"] = 1000 * float(len(pos)) / len(g)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A. The return window
# ---------------------------------------------------------------------------

def _a_design(d: pd.DataFrame, bins, minutes: bool) -> tuple[pd.DataFrame, list[str]]:
    d = d.copy()
    gsr = d["games_since_return"]
    terms = []
    for name, lo, hi in bins:
        d[f"gsr_{name}"] = gsr.between(lo, hi).astype(float)
        terms.append(f"gsr_{name}")
    d["no_prior_injury"] = gsr.isna().astype(float)
    d["log_prev_len"] = np.log1p(d["last_spell_length"].fillna(0))
    d["age"] = d["age"].fillna(d["age"].median())
    terms += ["no_prior_injury", "log_prev_len", "age"]
    if minutes:
        d["min_now"] = d["MIN"].fillna(0)
        d["min_mean_3"] = d["min_mean_3"].fillna(d["min_mean_3"].median())
        terms += ["min_now", "min_mean_3"]
    return d, terms


def a_fit(d: pd.DataFrame, label: str, bins=A_BINS, minutes: bool = False) -> pd.DataFrame:
    """Per-bin within-player effects, as risk differences and risk ratios vs 41+."""
    d = d.dropna(subset=[label])
    ref = d.loc[d["games_since_return"] >= A_REF_MIN, label]
    ref_rate = float(ref.mean())
    X, terms = _a_design(d, bins, minutes)
    fe = fe_fit(X, label, terms)
    out = []
    for name, lo, hi in bins:
        r = fe[fe["term"] == f"gsr_{name}"]
        n = int(X[f"gsr_{name}"].sum())
        if r.empty:
            out.append({"bin": name, "lo": lo, "hi": hi, "rows": n})
            continue
        r = r.iloc[0]
        out.append({
            "bin": name, "lo": lo, "hi": hi, "rows": n,
            "positives": int(X.loc[X[f"gsr_{name}"] == 1, label].sum()),
            "raw_rate": float(X.loc[X[f"gsr_{name}"] == 1, label].mean()) if n else np.nan,
            "coef": r["coef"], "se": r["se"], "p_value": r["p_value"],
            "rr": 1 + r["coef"] / ref_rate, "rr_lo": 1 + r["ci_lo"] / ref_rate, "rr_hi": 1 + r["ci_hi"] / ref_rate,
        })
    return pd.DataFrame(out).assign(label=label, ref_rate=ref_rate, ref_rows=len(ref),
                                    fit_rows=int(fe["rows"].iloc[0]))


def a_early(d: pd.DataFrame, label: str, threshold: float, minutes: bool = False) -> dict:
    """Primary statistic: risk ratio in games 0-5 back vs 41+, and p against ``threshold``."""
    t = a_fit(d, label, A_EARLY, minutes)
    r = t[t["bin"] == "0-5"].iloc[0]
    if pd.isna(r.get("coef", np.nan)):
        return {"label": label, "rows": int(t["fit_rows"].iloc[0])}
    ref = float(r["ref_rate"])
    z = (r["coef"] - (threshold - 1) * ref) / r["se"]
    return {"label": label, "rows": int(t["fit_rows"].iloc[0]), "window_rows": int(r["rows"]),
            "window_positives": int(r["positives"]), "ref_rate": ref, "coef_pp": 100 * r["coef"],
            "rr": r["rr"], "rr_lo": r["rr_lo"], "rr_hi": r["rr_hi"], "p_one_sided": 1 - phi(z)}


def _null_coef(d: pd.DataFrame, label: str, rng: np.random.Generator) -> tuple[float, float]:
    """One draw of the 0-5 bin coefficient when there is no return window at all.

    Each player keeps his real number of at-risk games, his starting games
    since return, and a constant hazard (his own rate, shrunk toward the
    league's). Injuries arrive at that constant rate, games since return is
    rebuilt from the simulated injuries, and the same within-player model is
    fitted. Games since return is a function of past outcomes, so with a
    finite panel the within-player estimate is biased even when the truth is
    no effect; this measures that bias (Nickell-type) on the real panel shape.

    Games since return always comes from *any* injury spell. For a narrower
    outcome (durable, same region) each simulated injury counts toward it with
    the player's own share of that outcome among his injuries, also constant.
    """
    d = d.dropna(subset=[label, "y_any_h1"])
    g = d.groupby("PLAYER_ID", sort=False)
    n = g.size().to_numpy()
    pos = g["y_any_h1"].sum().to_numpy()
    sub = g[label].sum().to_numpy()
    start = g["games_since_return"].first().to_numpy()
    mean_rate, prior = pos.sum() / n.sum(), 200.0
    h = (pos + prior * mean_rate) / (n + prior)
    share_mean = sub.sum() / max(pos.sum(), 1)
    q = (sub + 5 * share_mean) / (pos + 5)
    gsr_parts, y_parts = [], []
    for ni, hi, s, qi in zip(n, h, start, q):
        gaps = rng.geometric(hi, size=int(ni * hi * 3) + 10)
        while gaps.sum() < ni:
            gaps = np.append(gaps, rng.geometric(hi, size=len(gaps)))
        ends = np.cumsum(gaps)
        k = int(np.searchsorted(ends, ni)) + 1
        seg = np.repeat(np.arange(k), gaps[:k])[:ni]
        within = np.arange(ni) - np.concatenate([[0], ends[:k - 1]])[seg]
        gsr = within.astype(float)
        first = seg == 0
        gsr[first] = np.nan if np.isnan(s) else s + within[first]
        y = np.zeros(ni)
        y[ends[:k][ends[:k] <= ni] - 1] = 1.0
        if qi < 1:
            y *= rng.random(ni) < qi
        gsr_parts.append(gsr)
        y_parts.append(y)
    sim = pd.DataFrame({"games_since_return": np.concatenate(gsr_parts), label: np.concatenate(y_parts)})
    sim["PLAYER_ID"] = np.repeat(g.size().index.to_numpy(), n)
    sim["season"] = 0
    terms = []
    for name, lo, hi_ in A_EARLY:
        sim[f"gsr_{name}"] = sim["games_since_return"].between(lo, hi_).astype(float)
        terms.append(f"gsr_{name}")
    sim["no_prior_injury"] = sim["games_since_return"].isna().astype(float)
    fe = fe_fit(sim, label, terms + ["no_prior_injury"])
    raw = sim.loc[sim["gsr_0-5"] == 1, label].mean() / sim.loc[sim["games_since_return"] >= A_REF_MIN, label].mean()
    return float(fe.loc[fe["term"] == "gsr_0-5", "coef"].iloc[0]), float(raw)


def null_benchmark(d: pd.DataFrame, label: str, reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = np.array([_null_coef(d, label, rng) for _ in range(reps)])
    return {"null_coef_mean": float(draws[:, 0].mean()), "null_coef_sd": float(draws[:, 0].std(ddof=1)),
            "null_raw_ratio": float(draws[:, 1].mean())}


def window_length(bins: pd.DataFrame) -> str | None:
    """First bin, in order, whose 95% interval includes no effect."""
    for r in bins.itertuples():
        if r.rr_lo <= 1:
            return r.bin
    return None


def run_a(d: pd.DataFrame, spells: pd.DataFrame, seasons: list[int], spec: dict, seed: int = 7) -> dict:
    hA = spec["hypotheses"]["A"]
    thr = float(hA["threshold"]["value"])
    conf = int(spec["holdout"]["confirmation_season"])
    flagged = spec["label"]["flagged_seasons"]
    du = d[d["season"].isin(seasons)]

    curves = pd.concat([a_fit(du, f"y_{t}_h1").assign(target=t) for t in ("any", "dur3", "dur2")],
                       ignore_index=True)
    primary = a_early(du, "y_any_h1", thr)

    variants = [
        ("primary", du, "y_any_h1", False),
        ("with minutes", du, "y_any_h1", True),
        ("durable 3+ games", du, "y_dur3_h1", False),
        ("durable 2+ games", du, "y_dur2_h1", False),
        ("previous spell minor", du[du["prev_severity"] == "minor"], "y_any_h1", False),
        ("previous spell major", du[du["prev_severity"] == "major"], "y_any_h1", False),
        ("new spell, same region", du, "y_same_region_h1", False),
        ("new spell, different region", du, "y_diff_region_h1", False),
        ("era 2006-17", du[du["season"] <= 2017], "y_any_h1", False),
        ("era 2018-23", du[du["season"] >= 2018], "y_any_h1", False),
        ("without 2021-22", du[~du["season"].isin(flagged)], "y_any_h1", False),
        (f"confirmation season {conf} only", du[du["season"] == conf], "y_any_h1", False),
    ]
    rows = []
    for i, (name, dd, lab, m) in enumerate(variants):
        r = {"variant": name, **a_early(dd, lab, thr, minutes=m)}
        n = dd.dropna(subset=[lab]).groupby("PLAYER_ID").size()
        r["games_per_player"] = float((n ** 2).sum() / n.sum())    # row-weighted panel length
        if "coef_pp" in r:
            nb = null_benchmark(dd, lab, NULL_REPS, seed + i)
            ref = r["ref_rate"]
            r["rr_null"] = 1 + nb["null_coef_mean"] / ref
            r["rr_bias_adjusted"] = 1 + (r["coef_pp"] / 100 - nb["null_coef_mean"]) / ref
            r["null_sd_rr"] = nb["null_coef_sd"] / ref
            r["rr_null_raw_pooled"] = nb["null_raw_ratio"]
        rows.append(r)
    rob = pd.DataFrame(rows)

    # Practitioner units: extra spells in the raised-risk window -> games lost per team-season.
    any_curve = curves[curves["target"] == "any"].reset_index(drop=True)
    end = window_length(any_curve)
    raised = any_curve if end is None else any_curve.iloc[:int(any_curve.index[any_curve["bin"] == end][0])]
    extra_spells = float((raised["rows"] * raised["coef"]).sum())
    inj = spells[spells["season"].isin(seasons) & spells["severity"].isin(["minor", "major"]) & ~spells["censored"]]
    mean_len = float(inj["length"].mean())
    team_seasons = du.groupby(["TEAM_ID", "season"]).ngroups
    units = {
        "window_ends_at_bin": end, "raised_bins": ",".join(raised["bin"]),
        "extra_spells_total": extra_spells, "team_seasons": team_seasons,
        "mean_games_per_injury_spell": mean_len,
        "extra_spells_per_team_season": extra_spells / team_seasons,
        "extra_games_lost_per_team_season": extra_spells * mean_len / team_seasons,
    }
    return {"curves": curves, "robustness": rob, "primary": primary, "units": units}


# ---------------------------------------------------------------------------
# B. Acquired players
# ---------------------------------------------------------------------------

def team_status(panel: pd.DataFrame) -> pd.DataFrame:
    """Per (player, season, team): ``new`` / ``same`` vs last season, or ``no_prior``."""
    pairs = panel[["PLAYER_ID", "season", "TEAM_ID"]].drop_duplicates()
    prev = pairs.assign(season=pairs["season"] + 1, _same=1)
    had = pairs[["PLAYER_ID", "season"]].drop_duplicates().assign(season=lambda x: x["season"] + 1, _had=1)
    out = pairs.merge(prev, on=["PLAYER_ID", "season", "TEAM_ID"], how="left")
    out = out.merge(had, on=["PLAYER_ID", "season"], how="left")
    out["team_status"] = np.select([out["_had"].isna(), out["_same"] == 1], ["no_prior", "same"], "new")
    return out[["PLAYER_ID", "season", "TEAM_ID", "team_status"]]


def b_predictions(table: pd.DataFrame, panel: pd.DataFrame, cfg: dict, label: str) -> pd.DataFrame:
    """Forward-chain the three models; test rows of players with a prior season."""
    status = team_status(panel)
    data = table.dropna(subset=[label]).merge(status, on=["PLAYER_ID", "season", "TEAM_ID"], how="left")
    C = cfg["models"]["logistic_C"]
    frames = []
    for split in forward_chain(cfg, sorted(table["season"].unique())):
        fit = data[data["season"].isin(split.fit)]
        test = data[(data["season"] == split.test) & data["team_status"].isin(["new", "same"])]
        out = test[["PLAYER_ID", "season", "TEAM_ID", "stint_id", "game_num", "team_status", label]].copy()
        y = fit[label].to_numpy()
        out["p_base_rate"] = BaseRate().fit(fit[FRONT_OFFICE], y).predict_proba(test[FRONT_OFFICE])[:, 1]
        out["p_front_office"] = logistic(C).fit(fit[FRONT_OFFICE], y).predict_proba(test[FRONT_OFFICE])[:, 1]
        out["p_history"] = logistic(C).fit(fit[HISTORY_AGE_ROLE], y).predict_proba(test[HISTORY_AGE_ROLE])[:, 1]
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def b_compare(pred: pd.DataFrame, label: str, reps: int, seed: int) -> dict:
    """Paired player-bootstrap PR-AUC gain of history over front office, new vs same team."""
    y = pred[label].to_numpy(float)
    ph, pf, pb = (pred[c].to_numpy() for c in ("p_history", "p_front_office", "p_base_rate"))
    new = (pred["team_status"] == "new").to_numpy()
    same = ~new

    def gain(idx, mask):
        i = idx[mask[idx]]
        return safe_ap(y[i], ph[i]) - safe_ap(y[i], pf[i])

    boot = PlayerBootstrap(pred["PLAYER_ID"].to_numpy(), reps, seed)
    all_idx = np.arange(len(y))
    g_new, g_same = [], []
    for idx in boot.indices():
        g_new.append(gain(idx, new))
        g_same.append(gain(idx, same))
    g_new, g_same = np.array(g_new), np.array(g_same)
    rows = []
    for name, mask, reps_ in (("new team", new, g_new), ("same team", same, g_same)):
        i = all_idx[mask]
        lo, hi = ci(reps_)
        rows.append({"group": name, "label": label, "rows": int(mask.sum()), "positives": int(y[i].sum()),
                     "players": int(pred.loc[mask, "PLAYER_ID"].nunique()), "base_rate": float(y[i].mean()),
                     "pr_auc_base_rate": safe_ap(y[i], pb[i]), "pr_auc_front_office": safe_ap(y[i], pf[i]),
                     "pr_auc_history": safe_ap(y[i], ph[i]), "gain": gain(all_idx, mask),
                     "gain_lo": lo, "gain_hi": hi,
                     "p_one_sided": float(np.mean(reps_[~np.isnan(reps_)] <= 0))})
    diff = g_new - g_same
    lo, hi = ci(diff)
    rows.append({"group": "new minus same", "label": label,
                 "gain": rows[0]["gain"] - rows[1]["gain"], "gain_lo": lo, "gain_hi": hi,
                 "p_one_sided": float(np.mean(diff[~np.isnan(diff)] <= 0))})
    return pd.DataFrame(rows)


def b_terciles(pred: pd.DataFrame, spells: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Games missed to new injury spells in the stint, by tercile of risk at the first game."""
    first = pred.sort_values("game_num").drop_duplicates("stint_id")
    inj = spells[spells["severity"].isin(["minor", "major"])]
    missed = inj.groupby("stint_id").agg(games_missed=("length", "sum"), spells=("spell_id", "size"))
    rostered = panel.groupby("stint_id").size().rename("rostered_games")
    first = first.join(missed, on="stint_id").join(rostered, on="stint_id")
    first[["games_missed", "spells"]] = first[["games_missed", "spells"]].fillna(0)
    rows = []
    for status, g in first.groupby("team_status"):
        for model in ("p_history", "p_front_office"):
            t = pd.qcut(g[model].rank(method="first"), 3, labels=["bottom", "middle", "top"])
            agg = g.groupby(t, observed=True).agg(
                stints=("stint_id", "size"), mean_predicted=(model, "mean"),
                games_missed=("games_missed", "mean"), injury_spells=("spells", "mean"),
                rostered_games=("rostered_games", "mean")).reset_index(names="tercile")
            rows.append(agg.assign(team_status=status, model=model.removeprefix("p_")))
    out = pd.concat(rows, ignore_index=True)
    out["games_missed_per_82"] = 82 * out["games_missed"] / out["rostered_games"]
    return out


def run_b(table, panel, spells, cfg, spec) -> dict:
    reps, seed = int(cfg["evaluation"]["bootstrap_reps"]), int(cfg["evaluation"]["seed"])
    comps, terc = [], None
    for target in ("any", "dur3"):
        label = f"y_{target}_h1"
        pred = b_predictions(table, panel, cfg, label)
        comps.append(b_compare(pred, label, reps, seed))
        if target == "any":
            terc = b_terciles(pred, spells, panel)
            conf = int(spec["holdout"]["confirmation_season"])
            comps.append(b_compare(pred[pred["season"] == conf], label, reps, seed)
                         .assign(label=f"{label} ({conf} only)"))
    comp = pd.concat(comps, ignore_index=True)
    p = comp[(comp["label"] == "y_any_h1") & (comp["group"] == "new team")].iloc[0]
    return {"comparison": comp, "terciles": terc,
            "primary": {"gain": p["gain"], "gain_lo": p["gain_lo"], "gain_hi": p["gain_hi"],
                        "p_one_sided": p["p_one_sided"], "rows": p["rows"], "positives": p["positives"]}}


# ---------------------------------------------------------------------------
# C. Schedule density
# ---------------------------------------------------------------------------

def c_effects(d: pd.DataFrame, label: str, variant: str) -> list[dict]:
    rows = []
    r = estimate(d, "next_gap_1", label, C_CONTROLS, prepared=True)
    z = r["fe_effect_pp"] / r["fe_se_pp"]
    rows.append({"variant": variant, "model": "next day vs any other", "term": "next_gap_1", **r,
                 "p_one_sided": 1 - phi(z)})
    dose = fe_fit(d.dropna(subset=[label]), label, NEXT_GAP_DUMMIES + C_CONTROLS)
    for t in NEXT_GAP_DUMMIES:
        q = dose[dose["term"] == t].iloc[0]
        rows.append({"variant": variant, "model": "dose vs 4+ days", "term": t, "label": label,
                     "rows": int(q["rows"]), "rate_control": float(q["mean_y"]),
                     "fe_effect_pp": 100 * q["coef"], "fe_ci_lo_pp": 100 * q["ci_lo"],
                     "fe_ci_hi_pp": 100 * q["ci_hi"], "fe_p_value": q["p_value"], "fe_se_pp": 100 * q["se"]})
    return rows


def run_c(table: pd.DataFrame, seasons: list[int], spec: dict) -> dict:
    conf = int(spec["holdout"]["confirmation_season"])
    flagged = spec["label"]["flagged_seasons"]
    d = prepare(table[table["season"].isin(seasons)])
    rows = []
    for target in ("any", "dur3", "dur2", "minor", "major", "illness"):
        rows += c_effects(d, f"y_{target}_h1", "all usable seasons")
    rows += c_effects(d[~d["season"].isin(flagged)], "y_any_h1", "without 2021-22")
    rows += c_effects(d[d["season"] == conf], "y_any_h1", f"confirmation season {conf} only")
    res = pd.DataFrame(rows)
    p = res[(res["variant"] == "all usable seasons") & (res["label"] == "y_any_h1")
            & (res["term"] == "next_gap_1") & (res["model"] == "next day vs any other")].iloc[0]
    return {"effects": res, "primary": {"effect_pp": p["fe_effect_pp"], "ci_lo": p["fe_ci_lo_pp"],
                                        "ci_hi": p["fe_ci_hi_pp"], "p_one_sided": p["p_one_sided"],
                                        "rows": p["rows"]}}


# ---------------------------------------------------------------------------
# Summary and figures
# ---------------------------------------------------------------------------

def summary(a: dict, b: dict, c: dict, spec: dict) -> pd.DataFrame:
    alpha = float(spec["multiple_testing"]["alpha"])
    H = spec["hypotheses"]
    rows = [
        {"hypothesis": "A", "name": H["A"]["name"], "role": H["A"]["role"],
         "statistic": "risk ratio, games 0-5 back vs 41+", "estimate": a["primary"]["rr"],
         "ci_lo": a["primary"]["rr_lo"], "ci_hi": a["primary"]["rr_hi"],
         "threshold": float(H["A"]["threshold"]["value"]), "p_one_sided": a["primary"]["p_one_sided"]},
        {"hypothesis": "B", "name": H["B"]["name"], "role": H["B"]["role"],
         "statistic": "PR-AUC gain, history over front office, new-team rows", "estimate": b["primary"]["gain"],
         "ci_lo": b["primary"]["gain_lo"], "ci_hi": b["primary"]["gain_hi"],
         "threshold": float(H["B"]["threshold"]["value"]), "p_one_sided": b["primary"]["p_one_sided"]},
        {"hypothesis": "C", "name": H["C"]["name"], "role": H["C"]["role"],
         "statistic": "next-day game effect, pp", "estimate": c["primary"]["effect_pp"],
         "ci_lo": c["primary"]["ci_lo"], "ci_hi": c["primary"]["ci_hi"],
         "threshold": float(H["C"]["threshold"]["value"]), "p_one_sided": c["primary"]["p_one_sided"]},
    ]
    df = pd.DataFrame(rows)
    df["p_two_sided_equiv"] = np.minimum(1.0, 2 * df["p_one_sided"])
    adj = holm(dict(zip(df["hypothesis"], df["p_two_sided_equiv"])))
    df["holm_p"] = df["hypothesis"].map(adj)
    df["ci_clears_threshold"] = df["ci_lo"] > df["threshold"]
    df["holds_after_holm"] = df["holm_p"] < alpha
    return df


# Reference palette (dataviz skill), light surface.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _axes(plt, figsize=(7, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=22, fontweight="bold")
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=INK2)


def figures(a: dict, b: dict, c: dict, out_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths = []

    # A: within-player risk ratio by games since return.
    fig, ax = _axes(plt)
    cur = a["curves"]
    labels = [b_ for b_, *_ in A_BINS] + ["41+"]
    x = np.arange(len(labels))
    for i, (target, name) in enumerate((("any", "any new injury spell"), ("dur3", "spells of 3+ games"))):
        t = cur[cur["target"] == target]
        rr = np.append(t["rr"].to_numpy(), 1.0)
        lo = np.append(t["rr_lo"].to_numpy(), 1.0)
        hi = np.append(t["rr_hi"].to_numpy(), 1.0)
        xs = x + (i - 0.5) * 0.18
        ax.errorbar(xs[:-1], rr[:-1], yerr=[rr[:-1] - lo[:-1], hi[:-1] - rr[:-1]], fmt="o", ms=7,
                    color=SERIES[i], ecolor=SERIES[i], elinewidth=2, capsize=0, label=name,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.plot(xs[:-1], rr[:-1], color=SERIES[i], lw=2, alpha=0.5)
    ax.plot([x[-1]], [1.0], marker="o", ms=7, color=INK2)
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.axhline(1.5, color=INK2, lw=1, ls=":")
    ax.text(x[-1] + 0.3, 1.5, "1.5x threshold", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_xlabel("games played since returning from the last injury spell (41+ = player's own baseline)",
                  fontsize=9, color=INK2)
    ax.set_ylabel("risk vs own baseline (x)", fontsize=9, color=INK2)
    _title(ax, "A. Risk is raised in the first games back",
           "Within-player risk ratio of a new spell next game, 95% CI. Unadjusted: the design's bias "
           "pulls these toward zero")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    p = out_dir / "fig_a_return_window.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)

    # B: games missed by tercile of predicted risk, acquired players.
    fig, ax = _axes(plt)
    t = b["terciles"]
    t = t[t["team_status"] == "new"]
    x = np.arange(3)
    for i, (model, name) in enumerate((("history", "ranked with public injury history"),
                                       ("front_office", "ranked on age, minutes, role only"))):
        g = t[t["model"] == model].set_index("tercile").reindex(["bottom", "middle", "top"])
        bars = ax.bar(x + (i - 0.5) * 0.36, g["games_missed"], width=0.34, color=SERIES[i], label=name,
                      edgecolor=SURFACE, linewidth=2)
        for rect, v in zip(bars, g["games_missed"]):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8, color=INK2)
    ax.set_xticks(x, ["bottom third", "middle third", "top third"])
    ax.set_xlabel("predicted risk at the player's first game with the new team", fontsize=9, color=INK2)
    ax.set_ylabel("games missed to new injuries, that stint", fontsize=9, color=INK2)
    _title(ax, "B. Acquired players: games lost by predicted-risk tercile",
           "Players in their first season with a new team, pooled test seasons")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    p = out_dir / "fig_b_acquired_terciles.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)

    # C: dose response over the next-game gap, with the illness placebo.
    fig, ax = _axes(plt)
    e = c["effects"]
    e = e[(e["variant"] == "all usable seasons") & (e["model"] == "dose vs 4+ days")]
    x = np.arange(4)
    for i, (label, name) in enumerate((("y_any_h1", "new injury spell"), ("y_dur3_h1", "injury spell of 3+ games"),
                                       ("y_illness_h1", "illness spell (placebo)"))):
        g = e[e["label"] == label].set_index("term").reindex(NEXT_GAP_DUMMIES)
        eff = np.append(g["fe_effect_pp"].to_numpy(), 0.0)
        lo = np.append(g["fe_ci_lo_pp"].to_numpy(), 0.0)
        hi = np.append(g["fe_ci_hi_pp"].to_numpy(), 0.0)
        xs = x[:-1] + (i - 1) * 0.2
        ax.errorbar(xs, eff[:-1], yerr=[eff[:-1] - lo[:-1], hi[:-1] - eff[:-1]], fmt="o", ms=7,
                    color=SERIES[i], elinewidth=2, capsize=0, label=name,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.plot([x[-1]], [0.0], marker="o", ms=7, color=INK2)
    ax.axhline(0, color=INK2, lw=1, ls="--")
    ax.set_xticks(x, ["1 (next day)", "2", "3", "4+ (reference)"])
    ax.set_xlabel("days until the team's next game", fontsize=9, color=INK2)
    ax.set_ylabel("change in P(spell starts next game), pp", fontsize=9, color=INK2)
    _title(ax, "C. Next-game gap: dose response and placebo",
           "Player fixed effects, season dummies, rest-before-game controls; 95% CI")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    p = out_dir / "fig_c_schedule_dose.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)
    return paths


def run_all(table, panel, spells, cfg, spec=None) -> dict:
    spec = spec or load_spec()
    seasons = usable_seasons(cfg, sorted(table["season"].unique()))
    d = add_spell_context(table, panel, spells)
    print("  A: return window")
    a = run_a(d, spells, seasons, spec, int(cfg["evaluation"]["seed"]))
    print("  B: acquired players")
    b = run_b(table, panel, spells, cfg, spec)
    print("  C: schedule density")
    c = run_c(table, seasons, spec)
    return {"A": a, "B": b, "C": c, "summary": summary(a, b, c, spec), "label_drift": label_drift(d)}
