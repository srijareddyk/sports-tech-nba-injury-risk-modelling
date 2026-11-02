"""Assemble ``reports/REPORT.md`` from the stage outputs (M9).

Every number comes from a CSV written by an earlier stage; the prose around
them is conditional on those numbers, so a rerun can't leave a stale claim.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from injuryrisk.config import season_label
from injuryrisk.paths import REPORTS


def _read(name: str) -> pd.DataFrame | None:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else None


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _missing(stage: str) -> str:
    return f"_Not run yet: `python run.py {stage}`._\n"


def write_report(cfg: dict) -> Path:
    k = int(cfg["evaluation"]["alert_budget_per_team_game"])
    tgt, hor = cfg["models"]["primary_target"], int(cfg["models"]["primary_horizon"])
    tests = ", ".join(season_label(int(s)) for s in cfg["split"]["test_seasons"])
    out: list[str] = ["# NBA Injury Risk: Results\n",
                      f"Generated from `{Path(cfg['_path']).name}`. Test seasons (forward-chained, "
                      f"one at a time): {tests}. COVID seasons 2019-20 and 2020-21: "
                      f"{cfg['covid_mode']}. Labels end {season_label(int(cfg['label_last_season']))} "
                      f"(PST log coverage).\n"]

    # 1. Headline ---------------------------------------------------------
    out.append("## 1. Per-player injury probability\n")
    summ = _read("model_summary.csv")
    by_season = _read("model_by_season.csv")
    if summ is None:
        out.append(_missing("model"))
    else:
        out.append(f"Target: a new injury absence spell (minor or major) starts within the next H "
                   f"games, for a player who played tonight. Means over test seasons; "
                   f"`diff vs history` is the paired player-bootstrap PR-AUC difference against "
                   f"the history-only logistic baseline.\n")
        for h in cfg["labels"]["horizons"]:
            s = summ[(summ["target"] == tgt) & (summ["horizon"] == h)]
            cols = ["model", "pr_auc", "roc_auc", "brier_skill", f"precision_at_{k}", f"recall_at_{k}",
                    "pr_auc_diff_vs_ref", "seasons_beating_ref", "seasons"]
            t = s[cols].rename(columns={"pr_auc_diff_vs_ref": "diff vs history",
                                        "seasons_beating_ref": "seasons CI>0"})
            out.append(f"**H = {h} game(s)**, base rate "
                       f"{by_season[(by_season['target'] == tgt) & (by_season['horizon'] == h)]['base_rate'].mean():.2%}\n")
            out.append(md_table(t, 4) + "\n")

        g = by_season[(by_season["target"] == tgt) & (by_season["horizon"] == hor) & (by_season["model"] == "gbm")]
        wins = int(g["beats_ref"].sum())
        out.append("### Does workload add anything beyond injury history?\n")
        out.append(md_table(g[["test_season", "base_rate", "pr_auc", "pr_auc_lo", "pr_auc_hi",
                               "pr_auc_diff_vs_ref", "diff_lo", "diff_hi"]], 4) + "\n")
        if wins == len(g):
            verdict = ("The full GBM beats history-only in every test season, with the 95% CI of the "
                       "paired difference above zero each time. The gain is real but small in absolute "
                       "PR-AUC terms.")
        elif wins == 0:
            verdict = ("The full GBM never clears the noise floor against history-only. Workload, "
                       "schedule and physical features add nothing detectable. That is a valid result.")
        else:
            verdict = (f"The full GBM clears the noise floor in {wins} of {len(g)} test seasons. The "
                       f"evidence that workload adds signal beyond history is mixed.")
        out.append(verdict + "\n")

        out.append("### Is the alert list worth acting on?\n")
        pm = by_season[(by_season["target"] == tgt) & (by_season["horizon"] == hor)]
        dec = pm.groupby("model")[[f"precision_at_{k}", "break_even_precision", "alerts", "tp",
                                   "cost_saving"]].mean().reset_index()
        out.append(f"Alert budget: top {k} players per team per game. Break-even precision = "
                   f"cost_rest / (efficacy x mean injury-spell games), with rest efficacy "
                   f"{cfg['evaluation']['rest_efficacy']} (an optimistic upper bound). "
                   f"`cost_saving` is games saved vs no alerts; negative means acting costs more.\n")
        out.append(md_table(dec, 3) + "\n")
        best = dec.loc[dec[f"precision_at_{k}"].idxmax()]
        if best[f"precision_at_{k}"] < best["break_even_precision"]:
            out.append(f"Best precision ({best[f'precision_at_{k}']:.1%}, {best['model']}) is below "
                       f"break-even ({best['break_even_precision']:.1%}). **Resting players on these "
                       f"alerts would cost more games than it saves.** The probabilities are still useful "
                       f"for ranking and team-level planning, not for game-by-game rest calls.\n")
        else:
            out.append(f"Best precision ({best[f'precision_at_{k}']:.1%}) clears break-even "
                       f"({best['break_even_precision']:.1%}) under these cost assumptions.\n")

    cal = _read("calibration.csv")
    if cal is not None:
        out.append("### Calibration (pooled over test seasons, deciles of predicted risk)\n")
        c = cal[(cal["target"] == tgt) & (cal["horizon"] == hor) & cal["model"].isin(["history_only", "gbm"])]
        c = c.groupby(["model", "bin"]).agg(mean_pred=("mean_pred", "mean"), observed=("observed", "mean")).reset_index()
        piv = c.pivot(index="bin", columns="model", values=["mean_pred", "observed"])
        piv.columns = [f"{b}_{a}" for a, b in piv.columns]
        out.append(md_table(piv.reset_index(), 4) + "\n")
        if (REPORTS / "calibration_primary.png").exists():
            out.append("![calibration](calibration_primary.png)\n")

    # 2. Risk factors -----------------------------------------------------
    out.append("## 2. Ranked risk factors (associations)\n")
    rf = _read("risk_factors.csv")
    if rf is None:
        out.append(_missing("risk-factors"))
    else:
        r = rf[rf["target"] == "any"].head(15)
        out.append("Discrete-time hazard (logistic on the at-risk set, H = 1), all covariates in one "
                   "model, odds ratio per 1 SD, 95% CIs clustered by player. **Associations, not "
                   "effects:** minutes are endogenous (coaches cut minutes for players who look hurt).\n")
        out.append(md_table(r[["term", "scale", "odds_ratio", "or_lo", "or_hi", "p_value"]], 3) + "\n")
    imp = _read("importance_groups.csv")
    if imp is not None:
        out.append("**Grouped permutation importance (GBM, last test season, PR-AUC drop):**\n")
        piv = imp.pivot(index="name", columns="target", values="pr_auc_drop").reset_index()
        out.append(md_table(piv.sort_values(piv.columns[1], ascending=False), 4) + "\n")
    feat = _read("importance_features.csv")
    if feat is not None:
        f = feat[feat["target"] == tgt].head(12)[["name", "pr_auc_drop", "drop_sd"]]
        out.append("**Top single features (target `any`):**\n")
        out.append(md_table(f, 4) + "\n")
    age = _read("age_effect.csv")
    if age is not None:
        out.append("**Age, re-read on the fixed label (test seasons):**\n")
        out.append(md_table(age, 4) + "\n")

    # 3. Minor vs major ---------------------------------------------------
    out.append("## 3. Minor vs major injuries\n")
    mvm = _read("minor_vs_major.csv")
    if mvm is None:
        out.append(_missing("interpret"))
    else:
        out.append("`lift_over_base` = PR-AUC / base rate. The fantasy project found recurring minor "
                   "absences persist (r = 0.30) and major spells barely do (r = 0.05).\n")
        out.append(md_table(mvm, 4) + "\n")
    sl = _read("spell_length.csv")
    if sl is not None:
        out.append("**Spell length (games missed per new injury spell, uncensored spells):**\n")
        out.append(md_table(sl.groupby("model")[["mae", "spearman"]].mean().reset_index(), 3) + "\n")

    # 4. Causal -----------------------------------------------------------
    out.append("## 4. Rest and schedule: the causal slice\n")
    cb = _read("causal_b2b.csv")
    if cb is None:
        out.append(_missing("causal"))
    else:
        out.append("Linear probability model, player fixed effects + season dummies, controls for home, "
                   "travel, time-zone shift and age; SEs clustered by player. Effects in percentage points "
                   "of P(new injury spell starts next game).\n")
        out.append(md_table(cb[["treatment", "label", "rate_treated", "rate_control", "naive_diff_pp",
                                "fe_effect_pp", "fe_ci_lo_pp", "fe_ci_hi_pp", "fe_p_value"]], 3) + "\n")
        sel = _read("causal_selection_check.csv")
        if sel is not None:
            out.append("**Selection check:** share of rostered player-games sat out, by B2B status. "
                       "If at-risk players are rested on B2B nights, the estimate above is biased "
                       "toward zero.\n")
            out.append(md_table(sel, 4) + "\n")

    # 5. Team index -------------------------------------------------------
    out.append("## 5. Team Health Index\n")
    ti = _read("team_index_validation.csv")
    if ti is None:
        out.append(_missing("team-index"))
    else:
        n = cfg["team_index"]["horizon_games"]
        out.append(f"Correlation of the index (computed before tip-off) with actual minutes lost to new "
                   f"injury spells over the next {n} games, non-overlapping windows, team-bootstrap CI.\n")
        out.append(md_table(ti[ti["test_season"] == "all"][["model", "index", "windows", "pearson", "spearman",
                                                             "spearman_lo", "spearman_hi"]], 3) + "\n")

    # 6. Data and labels --------------------------------------------------
    out.append("## 6. Data and labels\n")
    for name, title in (("data_mapping.csv", "PST -> PLAYER_ID mapping (modelled seasons)"),
                        ("event_classes.csv", "Event classes"),
                        ("spell_causes.csv", "Absence spells by attributed cause"),
                        ("label_summary.csv", "Label base rates by season")):
        df = _read(name)
        if df is not None:
            out.append(f"**{title}**\n")
            out.append(md_table(df, 4) + "\n")
    audit = _read("event_audit_precision.csv")
    out.append("**Event classifier audit:** " + (
        "\n" + md_table(audit, 3) + "\n" if audit is not None else
        "hand-labelled sample drawn to `data/external/event_audit_sample.csv`; per-class precision is "
        "pending until `true_class` is filled in and `python run.py audit-score` is run.\n"))
    cc = _read("config_count.csv")
    if cc is not None:
        out.append(f"**Configurations tried:** {int(cc['fits'][0])} logged fits, "
                   f"{int(cc['distinct_configs'][0])} distinct (model, params) configs "
                   f"(`experiments/log.csv`).\n")

    # 7. Limits -----------------------------------------------------------
    out.append("## 7. Limits\n")
    out.append("""- **No tracking load.** Public box scores have no distance, speed, jump counts, force-plate or
  sleep data. Minutes are a coarse proxy for load. This caps any public-data model.
- **Association vs causation.** Everything except section 4 is predictive. A risk factor's odds ratio is
  not the effect of changing it. Section 4 uses only schedule-driven variation, and even that is exposed
  to selective resting.
- **Noise floor.** Differences between models are judged by paired player-bootstrap CIs, one test season
  at a time. The search space was fixed in the config before scoring; every fit is logged.
- **Labels are transactions, cleaned.** PST notes are classified by a closed vocabulary. Bare "placed on
  IL" (stash) events, rest, illness, protocols, suspensions and personal absences are excluded from the
  target. Absences with no nearby event ("unknown", mostly coach's decisions) are not injuries.
- **Label drift.** Logged injury rates rise sharply from 2017-18 on (more granular PST logging plus the
  load-management era). Forward-chaining exposes this; later test seasons have higher base rates.
- **Coverage.** Labels stop at 2022-23. Seasons before 2005-06 contribute injury-event history only.
- **Censoring.** Spells that run into season end have unknown length and are excluded from spell-length
  evaluation; end-of-stint rows without H observed games are dropped from the risk set.
""")
    path = REPORTS / "REPORT.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path
