# sports-tech-nba-injury-risk-modelling

Short-horizon NBA injury risk from public box scores and the Pro Sports Transactions (PST) injury log,
built to the spec in [nba_injury_risk_STARTER.md](nba_injury_risk_STARTER.md). It replaces the
notebook pipeline in `nba-final report.ipynb` and fixes the ten defects listed in spec section 2.

## Goal

The project tests what public, league-wide data adds to what a team already has, rather than trying
to out-predict a medical staff. Four hypotheses, written down before testing in
[configs/hypotheses.yaml](configs/hypotheses.yaml):

| Hypothesis | What public data supplies | Output |
|---|---|---|
| **A. Return window** (headline) | League-wide base rates for risk after returning from injury | `reports/hyp_a_*.csv`, `fig_a_return_window.png` |
| **B. Acquired players** | Injury history on players a team doesn't own yet | `reports/hyp_b_*.csv`, `fig_b_acquired_terciles.png` |
| **C. Schedule density** | Risk tied to a schedule known months ahead | `reports/hyp_c_effects.csv`, `fig_c_schedule_dose.png` |
| **D. Minutes ramp-up** (added after A-C were seen) | Not a data gap: turns A's window into a minutes decision staff control | `reports/hyp_d_*.csv`, `fig_d_ramp.png` |
| A, secondary: **tissue type** (outside Holm) | A's window split by soft tissue / ligament / joint / bone / unspecified | `reports/hyp_a_tissue.csv`, `fig_a_tissue.png`; audit sample `data/external/tissue_audit_sample.csv` |

D and the tissue analysis were pre-registered on 2026-11-26, after the A-C results had been seen.

`python run.py hypotheses` runs all four plus the tissue analysis, applies a Holm correction across A-D (`reports/hypotheses_summary.csv`)
and draws the figures; `python run.py report` puts them at the front of `reports/REPORT.md`.

## Predictive pipeline (context)

| Goal | Output |
|---|---|
| Per-player P(new injury spell) in the next 1 / 3 games | `run.py model` → `reports/model_summary.csv`, `model_by_season.csv`, `calibration.csv` |
| Ranked risk factors with honest effect sizes | `run.py risk-factors` (adjusted hazard ORs, player-clustered CIs), `run.py interpret` (permutation importance) |
| Team health index | `run.py team-index` → validated against actual minutes lost |
| Rest / minutes guidance (causal) | `run.py causal` → B2B second-night effect with player fixed effects, plus a selection check |
| Write-up | `run.py report` → `reports/REPORT.md` |

## Quick start

```bash
pip install -e .[dev]            # pandas, numpy, scikit-learn, scipy, matplotlib, pyyaml, nba_api, pytest
# optional: pip install shap

# PST exports go in data/raw/ (filenames listed in configs/base.yaml):
#   NBA Player Injury Stats(1951 - 2023).csv, nba_injuries_2018_2023.csv

python run.py collect            # game logs + bio, 2005-06 .. 2024-25 (40 API calls, cached)
python run.py all                # events -> labels -> features -> models -> hypotheses -> report
python -m pytest                 # vocab boundaries, absences, leakage
```

Each stage is one command (`python run.py --help` lists them) and reads and writes `data/processed/`.
Everything is configured in [configs/base.yaml](configs/base.yaml): seasons, horizons, the COVID-season
policy, the fixed GBM search grid, alert budget, cost assumptions.

## How the label is built

1. **Absences come from the schedule.** Games missed = the team's schedule minus the player's
   appearances, inside roster spans (first to last appearance per team stint). Stints get extended
   to season end when an injury note follows the last appearance.
2. **Causes come from PST.** Each absence spell takes the class of the nearest classified PST note
   within ±3 days (as-of join), falling back to any note inside the spell. Notes are classified by
   a closed, word-boundary vocabulary into `injury_major`, `injury_minor`, `rest`, `illness`,
   `protocol`, `suspension`, `personal`, `stash`, `other`.
3. **Target.** `y = 1` if a player who played game *t* starts a new minor or major injury spell within
   the next H games. Robustness targets: `dur3` / `dur2` (the spell lasts 3+ / 2+ games, a guard against
   rest days logged as soreness) and `illness` (the placebo outcome for Hypothesis C). Players already out aren't in the risk set. Features use only games before *t*
   and *t*'s schedule. `tests/test_no_leakage.py` rebuilds features from a world truncated at tip-off
   and checks they're identical.

## Evaluation rules

- Forward chaining by season. Hyperparameters are chosen on the season before the test season, never on the test season itself.
- Every model is reported next to the base-rate and history-only baselines, with player-level
  bootstrap CIs and paired differences (the noise floor).
- Metrics: PR-AUC, Brier, calibration, precision/recall at a fixed alert budget (top 2 per team per
  game), and expected cost against a break-even precision.
- Every fit is appended to `experiments/log.csv`.

## Layout

```
configs/base.yaml          pre-registered settings
configs/hypotheses.yaml    hypotheses A-D, tissue analysis, thresholds, Holm, holdout decision
data/raw/                  game logs, bio, PST exports (gitignored)
data/external/             arenas.csv (coords, time zones), known_absences.csv, event + tissue audit samples
data/processed/            panel, spells, model table, predictions (gitignored)
src/injuryrisk/
  collect/                 game_logs, bio, transactions (PST -> PLAYER_ID)
  labels/                  vocab (incl. tissue_group), classify, absences, target
  features/                workload, schedule, history, physical, build (asserts every feature)
  models/                  baselines, gbm, survival (hazard + risk factors), glm, runner, spell_length
  evaluate/                splits, metrics, bootstrap, log
  causal/b2b.py            schedule-driven causal slice
  hypotheses.py            tests A-D, tissue analysis, null benchmarks, Holm, figures
  team_index.py, interpret.py, report.py, report_hypotheses.py
tests/                     vocab boundaries, absences, targets, leakage, hypothesis helpers
run.py                     one command per stage
```

## Known limits

No tracking data (distance, speed, jumps, sleep), so load is proxied by minutes. Labels end in
2022-23, when the PST export stops. Logged injury rates rise sharply from 2017-18, so later test seasons have higher base rates. Hypotheses A, C and D use within-player (fixed-effects) designs; everything else is association.
The event classifier still needs a hand-labelled audit: `run.py audit-sample` draws the sample
to `data/external/event_audit_sample.csv`; fill in `true_class`, then run `run.py audit-score`.
