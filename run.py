"""One command per stage. Run from the repo root:

    python run.py collect        # M1  game logs + bio from NBA Stats (cached in data/raw)
    python run.py events         # M1  PST log -> PLAYER_ID, classified events
    python run.py audit-sample   # M2  stratified sample for hand labelling
    python run.py audit-score    # M2  per-class precision once labelled
    python run.py labels         # M2  roster spans, absence spells, targets
    python run.py features       # M4  model table, every feature asserted
    python run.py baselines      # M3  base rate / history-only, per test season
    python run.py model          # M5  all models x targets x horizons, bootstrap CIs
    python run.py risk-factors   #     adjusted hazard odds ratios (goal 2)
    python run.py interpret      # M6  permutation importance, minor vs major, age
    python run.py spell-length   #     games missed per new spell
    python run.py causal         # M7  B2B second-night effect, player fixed effects
    python run.py team-index     # M8  team health index vs actual minutes lost
    python run.py report         # M9  reports/REPORT.md
    python run.py all            # everything after collect, in order

Several stages run in order: ``python run.py model interpret report``.
Options: --config PATH (default configs/base.yaml).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from injuryrisk import io  # noqa: E402
from injuryrisk.config import all_seasons, load_config  # noqa: E402
from injuryrisk.paths import DATA_EXTERNAL, REPORTS, ensure_dirs  # noqa: E402

AUDIT_FILE = DATA_EXTERNAL / "event_audit_sample.csv"


def _write(df: pd.DataFrame, name: str) -> None:
    path = REPORTS / name
    df.to_csv(path, index=False)
    print(f"  -> {path.relative_to(REPORTS.parent)}")


# ---------------------------------------------------------------------------
# Data and labels
# ---------------------------------------------------------------------------

def stage_collect(cfg):
    from injuryrisk.collect import bio, game_logs
    seasons = all_seasons(cfg)
    game_logs.pull(seasons)
    bio.pull(seasons)


def stage_events(cfg):
    from injuryrisk.collect import game_logs, transactions
    from injuryrisk.labels.classify import class_report, classify_events
    logs = game_logs.load_all(all_seasons(cfg))
    ev = transactions.load_pst(cfg["pst_files"])
    ev = transactions.map_player_ids(ev, logs)
    ev = classify_events(ev)
    io.save(ev, "events")
    first = int(cfg["seasons"]["first"])
    rep = transactions.mapping_report(ev, first)
    print(rep.to_string(index=False))
    _write(rep, "data_mapping.csv")
    unmapped = ev[(ev["season"] >= first) & ev["PLAYER_ID"].isna()]
    _write(unmapped[["date", "team_name", "player_name", "notes", "match_stage"]], "unmapped_events.csv")
    cls = class_report(ev, first)
    print(cls.to_string(index=False))
    _write(cls, "event_classes.csv")


def stage_audit_sample(cfg):
    from injuryrisk.labels.classify import audit_sample
    if AUDIT_FILE.exists():
        print(f"  {AUDIT_FILE} exists; not overwriting hand labels. Delete it to redraw.")
        return
    sample = audit_sample(io.load("events"), int(cfg["seasons"]["first"]))
    sample.to_csv(AUDIT_FILE, index=False)
    print(f"  {len(sample)} notes -> {AUDIT_FILE}. Fill `true_class`, then run audit-score.")


def stage_audit_score(cfg):
    from injuryrisk.labels.classify import audit_score
    rep = audit_score(pd.read_csv(AUDIT_FILE))
    print(rep.to_string(index=False))
    _write(rep, "event_audit_precision.csv")


def stage_labels(cfg):
    from injuryrisk.collect import game_logs
    from injuryrisk.collect.transactions import fold_name
    from injuryrisk.labels import absences, target
    lc = cfg["labels"]
    logs = game_logs.load_all(all_seasons(cfg))
    ev = io.load("events")
    tg = absences.build_team_games(logs)
    _write(absences.schedule_lengths(tg).reset_index(), "schedule_lengths.csv")
    stints = absences.build_stints(logs, tg, ev, lc["span_extension_days"])
    panel = absences.build_panel(logs, tg, stints)
    spells = absences.find_spells(panel)

    known = pd.read_csv(DATA_EXTERNAL / "known_absences.csv")
    if len(known) and known["player_id"].isna().any():
        ids = logs.assign(f=logs["PLAYER_NAME"].map(fold_name)).drop_duplicates("f").set_index("f")["PLAYER_ID"]
        known["player_id"] = known["player_id"].fillna(known["player"].map(fold_name).map(ids))
    spells = absences.attribute_spells(spells, ev, lc["event_window_days"], lc["attribute_within_spell"],
                                       lc["major_min_games"], known)
    panel = target.attach_targets(panel, spells, lc["horizons"])
    # No PST coverage after label_last_season: labels unknown, not zero.
    ycols = [c for c in panel.columns if c.startswith("y_")]
    panel.loc[panel["season"] > int(cfg["label_last_season"]), ycols] = np.nan

    io.save(tg, "team_games")
    io.save(panel, "panel")
    io.save(spells, "spells")
    print(f"  panel {len(panel):,} rows ({panel['played'].mean():.1%} played), "
          f"{int(stints['extended'].sum())} stints extended past last appearance")
    causes = spells.groupby("cause").agg(spells=("spell_id", "size"), median_games=("length", "median"),
                                         censored=("censored", "mean")).reset_index()
    print(causes.to_string(index=False))
    _write(causes, "spell_causes.csv")
    summ = target.label_summary(panel[panel["season"] <= int(cfg["label_last_season"])], lc["horizons"])
    _write(summ, "label_summary.csv")


def stage_features(cfg):
    from injuryrisk.collect import bio
    from injuryrisk.features.build import build_model_table, coverage_report, feature_list
    groups = cfg["features"]["groups"]
    if int(cfg["features"]["box_score_lag"]) != 1:
        raise ValueError("box_score_lag must be 1: game t's box score is not known at tip-off of t")
    table = build_model_table(io.load("panel"), io.load("team_games"), io.load("spells"), io.load("events"),
                              bio.load_all(), bio.load_positions(), groups,
                              cfg["features"]["prior_season_shrink_games"])
    io.save(table, "model_table")
    feats = feature_list(groups)
    print(f"  model table: {len(table):,} at-risk rows x {len(feats)} features (all asserted present)")
    _write(coverage_report(table, feats), "feature_coverage.csv")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _chain(cfg, targets, horizons, models, stage):
    from injuryrisk.models.runner import run_forward_chain
    table, spells = io.load("model_table"), io.load("spells")
    results = {}
    for t in targets:
        for h in horizons:
            results[(t, h)] = run_forward_chain(table, spells, cfg, t, h, models=models, stage=stage)
    return results


def stage_baselines(cfg):
    from injuryrisk.models.runner import summarise
    k = int(cfg["evaluation"]["alert_budget_per_team_game"])
    res = _chain(cfg, [cfg["models"]["primary_target"]], cfg["labels"]["horizons"],
                 ("base_rate", "history_only", "history_age_role"), "baselines")
    metrics = pd.concat([r["metrics"] for r in res.values()], ignore_index=True)
    _write(metrics, "baselines_by_season.csv")
    cal = pd.concat([r["calibration"] for r in res.values()], ignore_index=True)
    _write(cal, "baselines_calibration.csv")
    print(summarise(metrics, k).round(4).to_string(index=False))


def stage_model(cfg):
    from injuryrisk.evaluate.log import config_count
    from injuryrisk.evaluate.metrics import calibration_plot
    from injuryrisk.models.runner import summarise
    k = int(cfg["evaluation"]["alert_budget_per_team_game"])
    res = _chain(cfg, ["any", "minor", "major"], cfg["labels"]["horizons"],
                 ("base_rate", "history_only", "history_age_role", "full_logistic", "hazard", "gbm"), "model")
    metrics = pd.concat([r["metrics"] for r in res.values()], ignore_index=True)
    _write(metrics, "model_by_season.csv")
    summary = summarise(metrics, k)
    _write(summary, "model_summary.csv")
    cal = pd.concat([r["calibration"] for r in res.values()], ignore_index=True)
    _write(cal, "calibration.csv")

    primary = (cfg["models"]["primary_target"], int(cfg["models"]["primary_horizon"]))
    io.save(res[primary]["predictions"], "predictions")
    params = {str(s): i["gbm_params"] for s, i in res[primary]["info"].items()}
    (REPORTS / "gbm_selected_params.json").write_text(json.dumps(params, indent=2))

    pooled = cal[(cal["target"] == primary[0]) & (cal["horizon"] == primary[1])]
    pooled = pooled.groupby(["model", "bin"]).agg(mean_pred=("mean_pred", "mean"),
                                                   observed=("observed", "mean")).reset_index()
    tables = {m: g for m, g in pooled.groupby("model") if m in ("history_only", "full_logistic", "gbm")}
    if calibration_plot(tables, REPORTS / "calibration_primary.png"):
        print("  -> reports/calibration_primary.png")
    count = config_count()
    _write(pd.DataFrame([count]), "config_count.csv")
    print(summary[summary["target"] == primary[0]].round(4).to_string(index=False))
    print(f"  experiments/log.csv: {count['fits']} fits, {count['distinct_configs']} distinct configs")


def stage_risk_factors(cfg):
    from injuryrisk.evaluate.splits import usable_seasons
    from injuryrisk.labels.target import label_column
    from injuryrisk.models.survival import risk_factor_table
    table = io.load("model_table")
    seasons = usable_seasons(cfg, sorted(table["season"].unique()))
    rows = []
    for target in ("any", "minor", "major"):
        label = label_column(target, 1)
        tab = risk_factor_table(table[table["season"].isin(seasons)], label)
        rows.append(tab.assign(target=target))
        if target == "any":
            print(tab[["term", "odds_ratio", "or_lo", "or_hi", "p_value"]].round(3).to_string(index=False))
    _write(pd.concat(rows, ignore_index=True), "risk_factors.csv")


def stage_interpret(cfg):
    from injuryrisk import interpret
    from injuryrisk.evaluate.splits import forward_chain
    from injuryrisk.features.build import feature_list
    from injuryrisk.labels.target import label_column
    from injuryrisk.models.gbm import gbm
    table = io.load("model_table")
    feats = feature_list(cfg["features"]["groups"])
    split = forward_chain(cfg, sorted(table["season"].unique()))[-1]
    params = json.loads((REPORTS / "gbm_selected_params.json").read_text())[str(split.test)]
    seed = int(cfg["evaluation"]["seed"])

    rows_g, rows_f, rows_s = [], [], []
    for target in ("any", "minor", "major"):
        label = label_column(target, int(cfg["models"]["primary_horizon"]))
        d = table.dropna(subset=[label])
        fit, test = d[d["season"].isin(split.fit)], d[d["season"] == split.test]
        model = gbm(params, seed).fit(fit[feats], fit[label])
        X, y = test[feats], test[label].to_numpy()
        rows_g.append(interpret.grouped_importance(model, X, y, feats, seed=seed).assign(target=target))
        rows_f.append(interpret.feature_importance(model, X, y, feats, seed=seed).head(25).assign(target=target))
        shap = interpret.shap_summary(model, X, seed=seed)
        if shap is not None:
            rows_s.append(shap.assign(target=target))
        print(f"  [{target}] test {split.test}: top groups "
              f"{rows_g[-1].head(3)[['name', 'pr_auc_drop']].round(4).values.tolist()}")
    _write(pd.concat(rows_g, ignore_index=True), "importance_groups.csv")
    _write(pd.concat(rows_f, ignore_index=True), "importance_features.csv")
    if rows_s:
        _write(pd.concat(rows_s, ignore_index=True), "shap_mean_abs.csv")
    else:
        print("  shap not installed: SHAP skipped (pip install shap), permutation importance reported")

    metrics = pd.read_csv(REPORTS / "model_by_season.csv")
    mm = metrics[metrics["model"].isin(["history_only", "gbm"])].copy()
    mm["lift_over_base"] = mm["pr_auc"] / mm["base_rate"]
    mvm = mm.groupby(["target", "horizon", "model"]).agg(
        base_rate=("base_rate", "mean"), pr_auc=("pr_auc", "mean"), lift_over_base=("lift_over_base", "mean"),
        pr_auc_diff_vs_history=("pr_auc_diff_vs_ref", "mean"),
        seasons_beating_history=("beats_ref", "sum")).reset_index()
    _write(mvm, "minor_vs_major.csv")
    print(mvm.round(4).to_string(index=False))

    pred = io.load("predictions")
    label = label_column(cfg["models"]["primary_target"], int(cfg["models"]["primary_horizon"]))
    age = interpret.age_table(pred, table, label)
    _write(age, "age_effect.csv")
    print(age.round(4).to_string(index=False))


def stage_spell_length(cfg):
    from injuryrisk.features.build import feature_list
    from injuryrisk.models import spell_length
    res = spell_length.run(io.load("model_table"), io.load("spells"), cfg,
                           feature_list(cfg["features"]["groups"]))
    _write(res, "spell_length.csv")
    print(res.groupby("model")[["mae", "spearman"]].mean().round(3).to_string())


def stage_causal(cfg):
    from injuryrisk.causal import b2b
    from injuryrisk.evaluate.splits import usable_seasons
    from injuryrisk.features.schedule import team_schedule_features
    table = io.load("model_table")
    seasons = usable_seasons(cfg, sorted(table["season"].unique()))
    d = table[table["season"].isin(seasons)]
    rows = [b2b.estimate(d, t, label) for t in cfg["causal"]["treatments"]
            for label in ("y_any_h1", "y_minor_h1", "y_major_h1")]
    res = pd.DataFrame(rows)
    _write(res, "causal_b2b.csv")
    print(res[["treatment", "label", "rate_treated", "rate_control", "naive_diff_pp", "fe_effect_pp",
               "fe_ci_lo_pp", "fe_ci_hi_pp"]].round(3).to_string(index=False))
    sched = team_schedule_features(io.load("team_games"))
    sel = b2b.selection_check(io.load("panel"), sched, io.load("spells"), seasons)
    _write(sel, "causal_selection_check.csv")
    print(sel.round(4).to_string(index=False))


def stage_team_index(cfg):
    from injuryrisk import team_index
    from injuryrisk.models.runner import injury_cost
    tc = cfg["team_index"]
    n = int(tc["horizon_games"])
    pred, panel, spells = io.load("predictions"), io.load("panel"), io.load("spells")
    L = injury_cost(spells, sorted(pred["season"].unique()), "any")
    rows, samples = [], []
    for pcol in ("p_gbm", "p_history_only"):
        team = team_index.build(pred, panel, spells, n, float(tc["fragility_threshold"]), L, pcol)
        windows = team_index.non_overlapping(team, n)
        for col in ("predicted_lost_minutes", "fragility"):
            for season, g in [("all", windows), *windows.groupby("season")]:
                r = team_index.validate(g, col, reps=int(cfg["evaluation"]["bootstrap_reps"]))
                rows.append({"model": pcol.removeprefix("p_"), "test_season": season, **r})
        if pcol == "p_gbm":
            samples.append(team)
    res = pd.DataFrame(rows)
    _write(res, "team_index_validation.csv")
    _write(samples[0], "team_index_by_game.csv")
    print(res[res["test_season"] == "all"].round(3).to_string(index=False))


def stage_report(cfg):
    from injuryrisk.report import write_report
    path = write_report(cfg)
    print(f"  -> {path.relative_to(REPORTS.parent)}")


STAGES = {
    "collect": stage_collect, "events": stage_events, "audit-sample": stage_audit_sample,
    "audit-score": stage_audit_score, "labels": stage_labels, "features": stage_features,
    "baselines": stage_baselines, "model": stage_model, "risk-factors": stage_risk_factors,
    "interpret": stage_interpret, "spell-length": stage_spell_length, "causal": stage_causal,
    "team-index": stage_team_index, "report": stage_report,
}
ALL_ORDER = ["events", "labels", "features", "baselines", "model", "risk-factors", "interpret",
             "spell-length", "causal", "team-index", "audit-sample", "report"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NBA injury risk pipeline")
    ap.add_argument("stage", nargs="+", choices=[*STAGES, "all"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ensure_dirs()
    for name in (ALL_ORDER if "all" in args.stage else args.stage):
        t = time.time()
        print(f"== {name}")
        STAGES[name](cfg)
        print(f"   ({time.time() - t:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
