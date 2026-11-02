# NBA Injury Risk: Starter Spec

Starting point for a new repo. It draws on two sources: findings from `nba_fantasy_model` (a 2023-24 fantasy ranking project with a full injury/availability pipeline) and an audit of `nba-final report.ipynb`, the prior LightGBM injury notebook.

Pitch goals, restated as testable deliverables:

1. Per-player injury probability over a short horizon (next game / next 7 days).
2. Ranked risk factors with honest effect sizes.
3. A team-level health index that aggregates player risk.
4. Guidance on rest and minutes. **This one is causal, not predictive. See section 7.**

---

## 1. What the evidence says is predictable

From `nba_fantasy_model` (seasons 2019-20 to 2023-24, NBA Stats game logs):

| Quantity | Value | Meaning |
|---|---|---|
| Games-played share of season-total variance | 49.8% | Availability matters more than production |
| GP year-over-year persistence | r = 0.13 to 0.16 | Last season's GP barely predicts this season's |
| MPG persistence | r = 0.68 to 0.77 | Role is stable |
| Per-minute production persistence | r = 0.87 to 0.89 | Skill is very stable |
| Players under 41 GP | 11.4% of players, 53% of GP variance | A few wipeout seasons dominate |
| AUC of any predictor for a wipeout season | 0.50 to 0.60 | Season-level wipeouts are near coin-flip |
| Persistence of **isolated** absences (1-3 game misses) | r = 0.303 | Recurring minor absences are predictable |
| Persistence of **injury spells** (multi-week) | r = 0.050 | Major injuries are close to random |
| "Reaches 70 games" base rate | ~54% | Near-balanced, tractable |
| Season-ending wipeout base rate | ~9% | Rare, hard |
| Precision a risk flag needed to help (fantasy context) | ≥ 60% | Below that, acting on the flag costs more than it saves |

Implications:

- **Short horizons beat season horizons.** The notebook's AUC 0.72 at a 7-day horizon doesn't contradict the 0.50 to 0.60 season result. Different horizons give different difficulty.
- **Most of the short-horizon signal is likely autocorrelation**: the player is already hurt, or is in an existing spell. Prove otherwise with the baselines in section 6.
- **Split the target.** Model "minor recurring absence" and "major injury spell" separately. They behave differently.

---

## 2. Audit of `nba-final report.ipynb`: fix before reusing anything

| # | Defect | Location | Fix |
|---|---|---|---|
| 1 | **No workload features were ever built.** The guard is `if "minutes" in df.columns`, but the column is `numMinutes`. ACWR, acute/chronic load, workload z-score, minutes deltas and `games_last_7` are all skipped. The FAQ's claims about "workload spikes" are unsupported. | Feature cell | Assert every expected feature exists after building. Fail loudly. |
| 2 | **The label is roster transactions.** Pro Sports Transactions (PST) "placed on IL" mixes real injuries, healthy stashes (historically the injured/inactive list often needed no injury; verify the era rules), rest, suspensions and COVID protocols. | Injury CSV | Classify every event (section 4). |
| 3 | **Exact-date join drops most events.** Injury rows join only when the transaction date equals a date the player played. Only 388 of 781k rows matched. IL moves land on days the player *didn't* play. | Merge cell | Use schedule-based absences plus as-of joins (section 4). |
| 4 | **Leakage through the injury game.** Positives cover the 7 days *before* the transaction, which includes the game where the player got hurt and left early. That game's low minutes/points flag the injury itself, not a precursor. | Label loop | Build features only from games strictly before the prediction point, and lag box scores by one game. |
| 5 | **DNPs imputed as normal games.** 129,930 NaN `numMinutes` rows get the median, which erases absences. | Imputation cell | NaN minutes = did not play. Make it a feature, never impute it. |
| 6 | **Selection on outcome.** The ≥24-game filter (cell 52 also overwrites `season` with the calendar year) removes the injured players. | Filter cells | Don't filter on games played. If you must, filter on rostered days. |
| 7 | **COVID contamination of the test set.** Split date 2022-01-01 falls in the Omicron wave. Test positive rate is 3.05% vs 1.55% in train. | Split cell | Drop or separately label health-and-safety-protocol events. Exclude 2019-20 and 2020-21 or treat them as their own population. |
| 8 | **SHAP age reading is wrong.** SHAP is a per-row conditional effect, so how many young players exist doesn't bias it. Likely causes: young players stashed on IL (#2), and survivorship (old players still in the league are durable). | Interpretation | Re-read after the label is fixed. |
| 9 | **The operating point is unusable.** Best F1 gives 8.8% precision at 38.7% recall, about 10 false alarms per true injury. | Threshold table | Report precision at a decision-relevant threshold and expected cost (section 6). |
| 10 | **Not reproducible.** Cell 50 uses `df`, but cell 49 loads `dataset`. Colab Drive paths are hardcoded. | Whole notebook | Use scripts plus config, one command per stage. |

---

## 3. Data sources

| Source | Use | Access | Notes |
|---|---|---|---|
| NBA Stats `LeagueGameLog` (via `nba_api`) | Game-by-game box scores and team schedules | One request per season, ~25k rows | Port `historical_game_logs.py` (section 8) |
| NBA Stats `LeagueDashPlayerBioStats` | Height, weight, age, draft | One request per season | Port `player_bio.py` |
| Pro Sports Transactions injury log | Candidate injury events with notes | Kaggle export (1951-2023), or the 2018-2023 export in the fantasy repo: `data/external/nba_injuries_2018_2023.csv`, 10,343 rows | Notes need classification. Names carry a leading `• ` bullet; strip it |
| News corpus (optional) | Injury detail PST lacks: body region, structure, stated timeline | Fantasy repo `data/raw/news_corpus_2024.csv`, 4,573 articles | LLM extraction costs ~$4.90 per season of text |
| Official NBA injury report PDFs (2021-22 onward, optional) | Game-day status: Out / Questionable / Rest | Public PDFs | Cleanest rest-vs-injury separation available |

Use the NBA `PLAYER_ID` as the key everywhere. Don't join on names: the notebook lost ~1,000 rows to name mismatches.

---

## 4. Target definition

### 4.1 Event classification

Classify every PST/news event into exactly one class:

| Class | Example notes | In target? |
|---|---|---|
| `injury_major` | torn ACL, fracture, surgery, "out for season", stated return ≥ 4 weeks | Yes (spell model) |
| `injury_minor` | sprained ankle, sore knee, strain, no surgery | Yes (minor model) |
| `rest` | rest, load management, injury management, maintenance, DNP-CD | **No.** This is a decision, not an outcome |
| `illness` | illness, flu | Separate or exclude |
| `protocol` | health and safety protocols, COVID | Exclude |
| `suspension` / `personal` | suspension, personal reasons | Exclude |
| `stash` | "placed on IL" with no reason | Exclude |

Port the vocabularies from `injury_events.py`: `PATHOLOGY_TERMS`, `STRUCTURE_TERMS`, `REGION_TERMS`, `LOAD_MANAGEMENT_LANGUAGE`, `SEASON_ENDING_LANGUAGE`, `parse_absence_days`. **Match on word boundaries.** Bugs already found with plain substring matching:

- "shin" matches *Washington*, so it read as a fracture
- "acl" matches *oracle*, *debacle*, *spectacle*
- "rest" matches "out for the **rest** of the season", so a season-ending injury was filed as a rest night

### 4.2 Absences from the schedule, not transactions

From `availability_history.py`: games missed = the team's actual schedule minus the player's appearances.

- Count the schedule per team per season. Don't assume 82: 2019-20 was 64-75 games, 2020-21 was 72, 2011-12 was 66.
- Measure a traded player against the longest schedule they were part of.
- For game-level labels, build **roster spans** (first to last appearance per team stint). A team game inside the span with no player row is an absence. Then attach the nearest classified event (as-of join, ±3 days) to assign the absence a cause.

### 4.3 Label

`y = 1` if the player **starts a new injury absence spell** (class `injury_minor` or `injury_major`) within the next H games, given they played the current game.

- Start with H = 1 and H = 3 games. Days are noisier because of uneven schedule gaps.
- Rows where the player is already out are **not at risk**. Exclude them. This is survival-analysis logic: once a spell starts, the player leaves the risk set until they return.
- Fit spell length as a separate target: games missed per new spell.

---

## 5. Features

**Leakage rule: every feature must be computable before tip-off of the game being predicted.** Lag box-score stats by one game. Add an automated test that shifts each feature forward and confirms nothing uses the current row.

| Group | Features | Source |
|---|---|---|
| Workload | minutes last 1/3/7/15 games, acute:chronic ratio (3 vs 15), minutes z-score vs own 10-game mean, minutes delta, games in last 7 days, cumulative season minutes | Game logs |
| Schedule | days since last game, back-to-back, 3-in-4, 4-in-6, road trip length, travel distance (arena coordinates), time-zone shifts | Game logs plus arena table |
| Availability history | `missed_lag1..3`, `missed_career`, `career_avail_rate`, `seasons_missing_20plus` | `availability_history.py` |
| Injury history | days since last injury per body region (horizons 30/90/180/365), count of prior events per region, prior major structure (ACL, Achilles), prior surgery flag | `injury_raw_events.py` |
| Re-injury | same-region injury within 90 days, games since return from last spell | Derived |
| Physical | age, height, weight, BMI, position | Bio |
| Role | prior-season MPG, usage, starter flag | Game logs |
| In-game intensity (proxy) | FGA + FTA per minute, rebounds per minute, fouls drawn | Game logs |

Public box scores have no tracking load (distance, speed, jumps) and no force-plate or sleep data. Say so in the report. It caps what any public-data model can do.

---

## 6. Evaluation protocol

### 6.1 Baselines (mandatory, report next to every model)

1. Base rate.
2. **History-only**: `days_since_last_injury` + `prior_injury_count` + `missed_lag1` in a logistic model.
3. History + age + role.
4. Full model.

If (4) doesn't beat (2) by more than the noise floor (6.3), workload adds nothing. That's a valid, reportable result.

### 6.2 Splits

- Forward-chaining by season: train ≤ S, test S+1. Never a random split, and never a split date inside a season.
- Report each test season separately. COVID seasons get a separate flag.

### 6.3 Noise floor and multiple comparisons

The lesson that governed the fantasy project: **73 configurations scored against one benchmark. Per-variant noise was 0.81 ranks, and best-of-73 under a pure null lands about 1.9 ranks better than baseline by chance.** Every "winner" after that was indistinguishable from luck.

- Pre-register the feature set and model before looking at the test season.
- Bootstrap the test metric (resample players, not rows) and report 95% CIs.
- Log every config tried. Report the count.
- Fit hyperparameters on a validation season, never the test season. Hindsight-fitted curves lost 1.3 to 3.0 ranks out of sample in the fantasy project.

### 6.4 Metrics

- **PR-AUC** and **Brier score**. ROC-AUC is misleading at 1-3% base rates.
- **Calibration plot** (reliability curve). Staff act on probabilities, so they must be calibrated.
- **Precision and recall at a fixed alert budget**, e.g. "flag the top 2 players per team per game". This mirrors a real decision better than a best-F1 threshold.
- **Expected cost**: `cost = FP * cost_of_rest + FN * cost_of_injury_games`. Show the break-even precision. The fantasy project's break-even was 60%.

---

## 7. Rest and minutes recommendations need causal methods

Minutes are **endogenous**. Coaches cut minutes for players who look hurt and rest players who are sore, so in observational data high minutes look *protective* (healthy players play more). A classifier can't answer "what if we rest him?" because that counterfactual is never observed.

Options, in increasing rigor:

1. **Report associations only**, labelled as associations. This is the honest minimum.
2. **Schedule-driven variation**: back-to-backs and 4-in-6 stretches come from the league schedule, not coach choice. Compare injury rates on the second night of a B2B vs rested games for the same player (player fixed effects).
3. **Matched designs**: match player-games on prior workload and injury history, then compare outcomes when the player was rested vs played. Treat "rest" events from 4.1 as the treatment.
4. **Survival model with time-varying covariates** (Cox or discrete-time hazard): hazard of a new spell as a function of rolling load, controlling for history.

---

## 8. Team Health Index

Without calibration it's just a sum of guesses. Define:

- `expected_available_minutes(team, next N games) = Σ_players P(available) × projected minutes`
- `fragility = minutes share held by players with P(new spell, next 10 games) > threshold`
- Validate: correlate the predicted index with *actual* minutes lost per team over the following N games. Report the correlation with a CI.

`durability.py` frames availability as "does the player reach 70 games" (base rate ~54%). It's a ready season-level component for roster planning.

---

## 9. Code to port from `nba_fantasy_model`

All modules depend on `nbafantasy.paths` (`DATA_RAW`, `DATA_PROCESSED`, `DATA_EXTERNAL`). Copy `paths.py` first and rename the package. Strip fantasy-specific columns (`fantasy_points`, `FPPM`).

| Source file | Lines | Take | Leave |
|---|---|---|---|
| `src/nbafantasy/paths.py` | ~30 | All of it (finds the repo root via `pyproject.toml`) | |
| `src/nbafantasy/collect/historical_game_logs.py` | 165 | `fetch_season`, `pull`, CLI. One `LeagueGameLog` call per season | |
| `src/nbafantasy/collect/player_bio.py` | 118 | `fetch_season`, `load_all` | |
| `src/nbafantasy/features/availability_history.py` | 128 | `_season_table`, `history`, `attach`, `schedule_lengths`. Schedule-minus-appearances logic | Season granularity only; extend to game level per 4.2 |
| `src/nbafantasy/collect/news/injury_events.py` | 1,185 | `PATHOLOGY_TERMS`, `STRUCTURE_TERMS`, `REGION_TERMS`, `SIDE_TERMS`, `LOAD_MANAGEMENT_LANGUAGE`, `SEASON_ENDING_LANGUAGE`, `parse_absence_days`, `classify_injury`, `_contains_any` | Rules-engine note synthesis (`synthesize_notes`) unless you need PST-format output |
| `src/nbafantasy/features/injury_raw_events.py` | 291 | `build_event_features`, `RECENCY_HORIZONS`, `REGION_GROUPS`, region/structure codes. Raw event features with no severity judgement | |
| `src/nbafantasy/features/injury_fopc_rules.py` | 1,327 | PST IL-log parsing, `SEASON_ENDING_PHRASES`, `MAJOR_STRUCTURE_TIERS`, `FRACTURE_MARKERS` | `RISK_LEVEL_DELTAS` and hand-set tier constants. These were measured wrong by ~10x |
| `src/nbafantasy/features/injury_absence_model.py` | 477 | `training_table`, `fit_absence_model` pattern: fits severity → realised availability with role controls (`minutes_per_game_prior`, `games_played_prior`) | Fantasy factor bounds |
| `src/nbafantasy/features/durability.py` | 108 | `expected_games`, the `AVAILABLE_GAMES = 70` framing | `availability_lift` (fantasy-specific) |
| `src/nbafantasy/features/known_absences.py` | 128 | CSV schema `season, player, games_missed, reason, known_on, source` with `reason ∈ {suspension, surgery, injury, personal, other}`. Use it to exclude non-injury absences | |
| `src/nbafantasy/collect/news/injury_passages.py` | 386 | `strip_boilerplate`, `split_sentences`, `build_passages` (only if using news) | |
| `src/nbafantasy/collect/news/injury_extract.py` | 564 | Batch LLM extraction with JSONL cache keyed by passage hash (interruptible, never re-bills) | Update `DEFAULT_MODEL`. Narrow the prompt (section 10, item 6) |

Data worth copying:

| File | Rows | Contents |
|---|---|---|
| `data/external/nba_injuries_2018_2023.csv` | 10,343 | PST log, `Date,Team,Acquired,Relinquished,Notes` |
| `data/external/NBA Player Injury Stats(1951 - 2023).csv` | ~18k | Same source as the notebook |
| `data/processed/injury_events_union_2024.csv` | 10,529 | PST plus news-extracted events, PST format |
| `data/processed/game_level_dataset_{2019..2025}.csv` | ~25k/season | Game logs with `days_since_last_game`, `back_to_back`, `games_last_7` already computed |
| `data/raw/player_bio_{2006..2024}.csv` | per season | Bio data (2012 missing) |
| `data/raw/news_corpus_2024.csv` | 4,573 | Preseason articles (optional) |

Known data issues carried over:

- Age missing for 31 to 64% of rows in the fantasy `player_master.csv` (316 players vs 530+ per season). Rebuild age from `player_bio` by `PLAYER_ID`; don't reuse `player_master`.
- "Poisoned priors": per-game averages from tiny samples (e.g. 35.0 MPG from 1 game). Shrink or require a minimum games count on any prior-season rate feature.
- The turnovers column is `TOV` in NBA Stats and `TO` elsewhere. A silent mismatch went unnoticed for months. Assert column presence.

---

## 10. Lessons that cost time in the fantasy project

1. **Accuracy and ranking can dissociate.** The best minutes model by correlation was the worst for ranking, and vice versa. Pick the metric that matches the decision before building.
2. **Feature guards that silently skip** (`if col in df.columns`) hid broken features in both projects. Use asserts instead.
3. **Post-hoc injury corrections hurt** when precision is below break-even, and more recall made it worse.
4. **Coarse outputs can beat continuous ones.** A step-function tier curve acted as useful shrinkage; the continuous version lost.
5. **The residual head ("predict the change") looked good only on small training sets.** Its edge correlated +0.989 with training size. Check that gains survive at full data.
6. **LLM extraction prompts can over-suppress.** The rules in the prompt dropped 123 of 169 cohort players; the passage gate only dropped 6. Audit drop counts at every pipeline stage.
7. **Team-name mapping.** "Philadelphia Sixers" vs "76ers" silently dropped 19 players. Map by team ID.

---

## 11. Suggested repo layout

```
nba-injury-risk/
  pyproject.toml
  README.md
  configs/
    base.yaml              # seasons, horizon H, feature groups, split
  data/
    raw/                   # game logs, bio, PST, news (gitignored)
    external/              # hand-curated CSVs (tracked): known_absences, arena coords
    processed/             # derived tables (gitignored)
  src/injuryrisk/
    paths.py
    collect/
      game_logs.py         # from historical_game_logs.py
      bio.py               # from player_bio.py
      transactions.py      # PST loader, bullet/name cleanup, PLAYER_ID mapping
    labels/
      vocab.py             # term tables from injury_events.py
      classify.py          # event -> class (section 4.1)
      absences.py          # roster spans, schedule-minus-appearances, spells
      target.py            # y for horizon H, at-risk set
    features/
      workload.py
      schedule.py
      history.py           # from availability_history.py, injury_raw_events.py
      physical.py
    models/
      baselines.py
      gbm.py
      survival.py
    evaluate/
      splits.py            # forward-chaining by season
      metrics.py           # PR-AUC, Brier, calibration, precision@budget, cost
      bootstrap.py         # player-level bootstrap CIs
    team_index.py
  tests/
    test_no_leakage.py     # shift-forward test on every feature
    test_vocab_boundaries.py  # Washington/shin, oracle/acl, rest-of-season
    test_absences.py       # traded player, short seasons
  experiments/
    log.csv                # every config tried, with test metric
```

---

## 12. Milestones

- [ ] **M1 Data**: game logs and bio for 2005-06 to 2024-25 keyed on `PLAYER_ID`. PST log mapped to `PLAYER_ID`, with the unmapped count reported.
- [ ] **M2 Labels**: event classifier with a hand-labelled audit sample (≥ 200 events, report per-class precision). Roster spans, absence spells, target `y` for H = 1 and H = 3.
- [ ] **M3 Baselines**: base rate and history-only logistic, with forward-chaining PR-AUC, Brier and calibration per season.
- [ ] **M4 Features**: workload, schedule, physical. Leakage test passes. Every feature asserted to exist.
- [ ] **M5 Model**: GBM plus discrete-time hazard. Compared to baselines with bootstrap CIs. Config count logged.
- [ ] **M6 Interpretation**: SHAP on the fixed label. Minor vs major models compared.
- [ ] **M7 Causal slice**: B2B second-night effect with player fixed effects (section 7, option 2).
- [ ] **M8 Team index**: predicted vs actual minutes lost per team, with CI.
- [ ] **M9 Report**: state limits plainly. No tracking data, association vs causation, the noise floor.
