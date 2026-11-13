# sports-tech-nba-injury-risk-modelling

Short-horizon NBA injury risk from public box scores and the Pro Sports Transactions (PST) injury log,
built to the spec in [nba_injury_risk_STARTER.md](nba_injury_risk_STARTER.md). It replaces the
notebook pipeline in `nba-final report.ipynb` and fixes the ten defects listed in spec section 2.

## Deliverables

| Goal | Output |
|---|---|
| Per-player P(new injury spell) in the next 1 / 3 games | `run.py model` → `reports/model_summary.csv`, `model_by_season.csv`, `calibration.csv` |
| Ranked risk factors with honest effect sizes | `run.py risk-factors` (adjusted hazard ORs, player-clustered CIs), `run.py interpret` (permutation importance) |
| Team health index | `run.py team-index` → validated against actual minutes lost |
| Rest / minutes guidance (causal) | `run.py causal` → B2B second-night effect with player fixed effects, plus a selection check |
| Write-up | `run.py report` → `reports/REPORT.md` |

## Quick start

```bash
pip install -e .[dev]            # pandas, numpy, scikit-learn, pyyaml, nba_api, pytest
# optional: pip install matplotlib shap

# PST exports go in data/raw/ (filenames listed in configs/base.yaml):
#   NBA Player Injury Stats(1951 - 2023).csv, nba_injuries_2018_2023.csv

python run.py collect            # game logs + bio, 2005-06 .. 2024-25 (40 API calls, cached)
python run.py all                # events -> labels -> features -> models -> report
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
   the next H games. Players already out aren't in the risk set. Features use only games before *t*
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
data/raw/                  game logs, bio, PST exports (gitignored)
data/external/             arenas.csv (coords, time zones), known_absences.csv, audit sample
data/processed/            panel, spells, model table, predictions (gitignored)
src/injuryrisk/
  collect/                 game_logs, bio, transactions (PST -> PLAYER_ID)
  labels/                  vocab, classify, absences, target
  features/                workload, schedule, history, physical, build (asserts every feature)
  models/                  baselines, gbm, survival (hazard + risk factors), glm, runner, spell_length
  evaluate/                splits, metrics, bootstrap, log
  causal/b2b.py            schedule-driven causal slice
  team_index.py, interpret.py, report.py
tests/                     vocab boundaries, absences, leakage
run.py                     one command per stage
```

## Known limits

No tracking data (distance, speed, jumps, sleep), so load is proxied by minutes. Labels end in
2022-23, when the PST export stops. Logged injury rates rise sharply from 2017-18, so later test seasons have higher base rates. Only section 4 of the report makes causal claims; everything else is association.
The event classifier still needs a hand-labelled audit: `run.py audit-sample` draws the sample
to `data/external/event_audit_sample.csv`; fill in `true_class`, then run `run.py audit-score`.
