"""Forward-chain every model for one (target, horizon) and score it.

Per test season S:

* baselines, full logistic and hazard model: fit on train + val, score S
* GBM: each grid config fit on train and scored on val (logged); the best by
  validation PR-AUC is refit on train + val and scored once on S

Outputs per-season metrics, player-bootstrap CIs, paired PR-AUC differences
against ``history_only`` (the noise-floor test), calibration tables, and the
test predictions (used by the team index and interpretation stages).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from injuryrisk.config import config_hash
from injuryrisk.evaluate import log as xlog
from injuryrisk.evaluate.bootstrap import model_cis
from injuryrisk.evaluate.metrics import budget_flags, calibration_table, safe_ap, safe_auc, score_all
from injuryrisk.evaluate.splits import Split, forward_chain
from injuryrisk.features.build import feature_list
from injuryrisk.labels.target import label_column
from injuryrisk.models.baselines import HISTORY_AGE_ROLE, HISTORY_ONLY, BaseRate, logistic
from injuryrisk.models.gbm import gbm
from injuryrisk.models.survival import HAZARD_FEATURES, hazard_model
from sklearn.metrics import brier_score_loss

REFERENCE = "history_only"
ALL_MODELS = ("base_rate", "history_only", "history_age_role", "full_logistic", "hazard", "gbm")


def injury_cost(spells: pd.DataFrame, seasons, target: str) -> float:
    """Mean games missed per uncensored injury spell of this kind, in ``seasons``."""
    sev = {"any": ["minor", "major"], "minor": ["minor"], "major": ["major"]}[target]
    s = spells[spells["season"].isin(seasons) & spells["severity"].isin(sev) & ~spells["censored"]]
    return float(s["length"].mean()) if len(s) else 5.0


def _log(cfg, stage, target, horizon, test, eval_on, model, params, y, p):
    xlog.log_fit(config_hash=config_hash(cfg), stage=stage, target=target, horizon=horizon,
                 test_season=test, eval_on=eval_on, model=model, params=params, n=len(y),
                 positives=int(np.sum(y)), pr_auc=safe_ap(y, p),
                 brier=float(brier_score_loss(y, p)), roc_auc=safe_auc(y, p))


def select_gbm(train: pd.DataFrame, val: pd.DataFrame, features: list[str], label: str, cfg: dict,
               stage: str, target: str, horizon: int, test: int) -> tuple[dict, list[dict]]:
    seed = int(cfg["evaluation"]["seed"])
    results = []
    for params in cfg["models"]["gbm_grid"]:
        m = gbm(params, seed).fit(train[features], train[label])
        p = m.predict_proba(val[features])[:, 1]
        ap = safe_ap(val[label].to_numpy(), p)
        results.append({**params, "val_pr_auc": ap})
        _log(cfg, stage, target, horizon, test, f"val_{val['season'].iloc[0]}", "gbm", params,
             val[label].to_numpy(), p)
    best = max(results, key=lambda r: r["val_pr_auc"])
    return {k: v for k, v in best.items() if k != "val_pr_auc"}, results


def run_split(table: pd.DataFrame, split: Split, label: str, cfg: dict, models, stage: str,
              target: str, horizon: int, features: list[str]) -> tuple[pd.DataFrame, dict]:
    data = table.dropna(subset=[label])
    train = data[data["season"].isin(split.train)]
    val = data[data["season"] == split.val]
    fit = data[data["season"].isin(split.fit)]
    test = data[data["season"] == split.test]
    y_fit = fit[label].to_numpy()

    specs = {
        "base_rate": (BaseRate, HISTORY_ONLY),
        "history_only": (lambda: logistic(cfg["models"]["logistic_C"]), HISTORY_ONLY),
        "history_age_role": (lambda: logistic(cfg["models"]["logistic_C"]), HISTORY_AGE_ROLE),
        "full_logistic": (lambda: logistic(cfg["models"]["logistic_C"]), features),
        "hazard": (lambda: hazard_model(cfg["models"]["logistic_C"]), HAZARD_FEATURES),
    }
    preds = {}
    info = {}
    for name in models:
        if name == "gbm":
            params, grid = select_gbm(train, val, features, label, cfg, stage, target, horizon, split.test)
            model = gbm(params, int(cfg["evaluation"]["seed"])).fit(fit[features], y_fit)
            info["gbm_params"] = params
            info["gbm_grid"] = grid
            info["gbm_model"] = model
            cols = features
        else:
            factory, cols = specs[name]
            model = factory().fit(fit[cols], y_fit)
        p = model.predict_proba(test[cols])[:, 1]
        preds[name] = p
        _log(cfg, stage, target, horizon, split.test, f"test_{split.test}", name,
             info.get("gbm_params", {}) if name == "gbm" else {"features": len(cols)},
             test[label].to_numpy(), p)

    out = test[["PLAYER_ID", "season", "TEAM_ID", "GAME_ID", "GAME_DATE", "stint_id", "game_num",
                "mpg_last10_played", "mpg_prior", label]].copy()
    for name, p in preds.items():
        out[f"p_{name}"] = p
    return out, info


def run_forward_chain(table: pd.DataFrame, spells: pd.DataFrame, cfg: dict, target: str, horizon: int,
                      models=ALL_MODELS, stage: str = "model") -> dict:
    label = label_column(target, horizon)
    features = feature_list(cfg["features"]["groups"])
    splits = forward_chain(cfg, sorted(table["season"].unique()))
    ev = cfg["evaluation"]
    k = int(ev["alert_budget_per_team_game"])

    metric_rows, ci_rows, cal_rows, pred_frames, infos = [], [], [], [], {}
    for split in splits:
        print(f"  [{target} h{horizon}] test {split.test}: train {split.train[0]}-{split.train[-1]}, "
              f"val {split.val}")
        preds, info = run_split(table, split, label, cfg, models, stage, target, horizon, features)
        infos[split.test] = info
        pred_frames.append(preds)
        y = preds[label].to_numpy()
        cost_inj = (float(ev["cost_injury_games"]) if ev["cost_injury_games"] != "mean_spell"
                    else injury_cost(spells, split.fit, target))
        flags = {}
        for name in models:
            p = preds[f"p_{name}"].to_numpy()
            m = score_all(y, p, preds, ev, cost_inj)
            metric_rows.append({"target": target, "horizon": horizon, "test_season": split.test,
                                "model": name, "cost_injury_games": cost_inj, **m})
            flags[name] = budget_flags(preds, p, k)
            cal = calibration_table(y, p, int(ev["calibration_bins"]))
            cal_rows.append(cal.assign(target=target, horizon=horizon, test_season=split.test, model=name))
        cis = model_cis(y, {n: preds[f"p_{n}"].to_numpy() for n in models}, preds["PLAYER_ID"].to_numpy(),
                        int(ev["bootstrap_reps"]), int(ev["seed"]), reference=REFERENCE, flags=flags)
        ci_rows.append(cis.assign(target=target, horizon=horizon, test_season=split.test))

    metrics = pd.DataFrame(metric_rows)
    cis = pd.concat(ci_rows, ignore_index=True)
    metrics = metrics.merge(cis, on=["target", "horizon", "test_season", "model"], how="left")
    return {
        "metrics": metrics,
        "calibration": pd.concat(cal_rows, ignore_index=True),
        "predictions": pd.concat(pred_frames, ignore_index=True),
        "info": infos,
        "label": label,
        "features": features,
    }


def summarise(metrics: pd.DataFrame, k: int) -> pd.DataFrame:
    """Mean over test seasons, and how many seasons each model beat the reference."""
    cols = ["pr_auc", "roc_auc", "brier", "brier_skill", f"precision_at_{k}", f"recall_at_{k}",
            "cost_saving", "pr_auc_diff_vs_ref"]
    agg = metrics.groupby(["target", "horizon", "model"])[cols].mean()
    agg["seasons_beating_ref"] = metrics.groupby(["target", "horizon", "model"])["beats_ref"].sum()
    agg["seasons"] = metrics.groupby(["target", "horizon", "model"])["test_season"].nunique()
    order = {m: i for i, m in enumerate(ALL_MODELS)}
    agg = agg.reset_index()
    return agg.sort_values(["target", "horizon", "model"], key=lambda s: s.map(order) if s.name == "model" else s)
