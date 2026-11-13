# NBA Injury Risk: Results

Generated from `base.yaml`. Test seasons (forward-chained, one at a time): 2015-16, 2016-17, 2017-18, 2018-19, 2021-22, 2022-23. COVID seasons 2019-20 and 2020-21: exclude. Labels end 2022-23 (PST log coverage).

## 1. Per-player injury probability

Target: a new injury absence spell (minor or major) starts within the next H games, for a player who played tonight. Means over test seasons; `diff vs history` is the paired player-bootstrap PR-AUC difference against the history-only logistic baseline.

**H = 1 game(s)**, base rate 2.01%

| model | pr_auc | roc_auc | brier_skill | precision_at_2 | recall_at_2 | diff vs history | seasons CI>0 | seasons |
|---|---|---|---|---|---|---|---|---|
| base_rate | 0.0201 | 0.5000 | -0.0044 | 0.0263 | 0.2475 | -0.0221 | 0 | 6 |
| history_only | 0.0414 | 0.6588 | 0.0051 | 0.0369 | 0.3463 | 0.0000 | 0 | 6 |
| history_age_role | 0.0420 | 0.6608 | 0.0055 | 0.0378 | 0.3513 | 0.0004 | 0 | 6 |
| full_logistic | 0.0423 | 0.6613 | 0.0066 | 0.0377 | 0.3546 | 0.0009 | 3 | 6 |
| hazard | 0.0410 | 0.6580 | 0.0063 | 0.0371 | 0.3457 | -0.0005 | 1 | 6 |
| gbm | 0.0409 | 0.6502 | 0.0057 | 0.0368 | 0.3347 | -0.0006 | 1 | 6 |

**H = 3 game(s)**, base rate 5.54%

| model | pr_auc | roc_auc | brier_skill | precision_at_2 | recall_at_2 | diff vs history | seasons CI>0 | seasons |
|---|---|---|---|---|---|---|---|---|
| base_rate | 0.0554 | 0.5000 | -0.0124 | 0.0746 | 0.2562 | -0.0529 | 0 | 6 |
| history_only | 0.1077 | 0.6621 | 0.0128 | 0.1019 | 0.3493 | 0.0000 | 0 | 6 |
| history_age_role | 0.1102 | 0.6676 | 0.0147 | 0.1044 | 0.3545 | 0.0026 | 2 | 6 |
| full_logistic | 0.1109 | 0.6694 | 0.0172 | 0.1038 | 0.3570 | 0.0032 | 2 | 6 |
| hazard | 0.1084 | 0.6659 | 0.0165 | 0.1043 | 0.3565 | 0.0007 | 1 | 6 |
| gbm | 0.1127 | 0.6541 | 0.0163 | 0.1010 | 0.3383 | 0.0054 | 2 | 6 |

### Does workload add anything beyond injury history?

| test_season | base_rate | pr_auc | pr_auc_lo | pr_auc_hi | pr_auc_diff_vs_ref | diff_lo | diff_hi |
|---|---|---|---|---|---|---|---|
| 2016 | 0.0098 | 0.0186 | 0.0142 | 0.0258 | -0.0108 | -0.0217 | -0.0026 |
| 2017 | 0.0127 | 0.0194 | 0.0164 | 0.0239 | -0.0052 | -0.0116 | -0.0016 |
| 2018 | 0.0186 | 0.0315 | 0.0265 | 0.0400 | 0.0014 | -0.0016 | 0.0059 |
| 2019 | 0.0194 | 0.0414 | 0.0326 | 0.0540 | 0.0008 | -0.0050 | 0.0068 |
| 2022 | 0.0308 | 0.0677 | 0.0581 | 0.0786 | 0.0015 | -0.0069 | 0.0089 |
| 2023 | 0.0291 | 0.0665 | 0.0567 | 0.0790 | 0.0088 | 0.0030 | 0.0171 |

The full GBM clears the noise floor in 1 of 6 test seasons and is significantly *worse* than history-only in 2. Workload, schedule and physical features add no reliable signal beyond injury history at this horizon.

### Is the alert list worth acting on?

Alert budget: top 2 players per team per game. Break-even precision = cost_rest / (efficacy x mean injury-spell games), with rest efficacy 1.0 (an optimistic upper bound). `cost_saving` is games saved vs no alerts; negative means acting costs more.

| model | precision_at_2 | break_even_precision | alerts | tp | cost_saving |
|---|---|---|---|---|---|
| base_rate | 0.026 | 0.153 | 4860.000 | 127.667 | -4036.760 |
| full_logistic | 0.038 | 0.153 | 4860.000 | 183.333 | -3676.592 |
| gbm | 0.037 | 0.153 | 4860.000 | 179.000 | -3707.042 |
| hazard | 0.037 | 0.153 | 4860.000 | 180.167 | -3697.251 |
| history_age_role | 0.038 | 0.153 | 4860.000 | 183.667 | -3674.904 |
| history_only | 0.037 | 0.153 | 4860.000 | 179.333 | -3701.980 |

Best precision (3.8%, history_age_role) is below break-even (15.3%). **Resting players on these alerts would cost more games than it saves.** The probabilities are still useful for ranking and team-level planning, not for game-by-game rest calls.

### Calibration (pooled over test seasons, deciles of predicted risk)

| bin | gbm_mean_pred | history_only_mean_pred | gbm_observed | history_only_observed |
|---|---|---|---|---|
| 0 | 0.0037 | 0.0049 | 0.0082 | 0.0080 |
| 1 | 0.0054 | 0.0053 | 0.0101 | 0.0101 |
| 2 | 0.0068 | 0.0074 | 0.0112 | 0.0106 |
| 3 | 0.0085 | 0.0103 | 0.0132 | 0.0129 |
| 4 | 0.0104 | 0.0122 | 0.0140 | 0.0148 |
| 5 | 0.0126 | 0.0142 | 0.0199 | 0.0167 |
| 6 | 0.0153 | 0.0166 | 0.0192 | 0.0212 |
| 7 | 0.0189 | 0.0195 | 0.0237 | 0.0254 |
| 8 | 0.0252 | 0.0239 | 0.0331 | 0.0318 |
| 9 | 0.0475 | 0.0343 | 0.0481 | 0.0493 |

## 2. Ranked risk factors (associations)

Discrete-time hazard (logistic on the at-risk set, H = 1), all covariates in one model, odds ratio per 1 SD, 95% CIs clustered by player. **Associations, not effects:** minutes are endogenous (coaches cut minutes for players who look hurt).

| term | scale | odds_ratio | or_lo | or_hi | p_value |
|---|---|---|---|---|---|
| prior_injury_count | log1p, per SD | 1.358 | 1.294 | 1.426 | 0.000 |
| days_since_last_injury | log1p, per SD | 0.747 | 0.706 | 0.790 | 0.000 |
| inj_events_365d | log1p, per SD | 1.178 | 1.128 | 1.231 | 0.000 |
| min_mean_15 | per SD | 1.152 | 1.070 | 1.241 | 0.000 |
| min_mean_3 | per SD | 0.923 | 0.859 | 0.991 | 0.027 |
| shots_per_min_10 | per SD | 1.074 | 1.029 | 1.120 | 0.001 |
| age | per SD | 0.933 | 0.896 | 0.971 | 0.001 |
| missed_lag1 | per SD | 1.047 | 1.016 | 1.078 | 0.002 |
| b2b | per SD | 0.955 | 0.928 | 0.983 | 0.002 |
| usage_prior | per SD | 0.969 | 0.930 | 1.009 | 0.131 |
| mpg_prior | per SD | 1.029 | 0.988 | 1.071 | 0.166 |
| prior_major_structure | per SD | 1.028 | 1.002 | 1.056 | 0.037 |
| games_since_return | log1p, per SD | 1.026 | 0.969 | 1.087 | 0.381 |
| acwr_3_15 | per SD | 1.021 | 0.983 | 1.061 | 0.283 |
| games_last_7d | per SD | 0.981 | 0.951 | 1.011 | 0.215 |

**Grouped permutation importance (GBM, last test season, PR-AUC drop):**

| name | any | major | minor |
|---|---|---|---|
| injury_history | 0.0340 | 0.0025 | 0.0288 |
| workload | 0.0059 | 0.0005 | 0.0032 |
| role | 0.0012 | 0.0002 | 0.0001 |
| availability | 0.0001 | 0.0004 | 0.0001 |
| intensity | -0.0000 | -0.0004 | -0.0021 |
| physical | -0.0006 | 0.0005 | -0.0008 |
| schedule | -0.0021 | 0.0001 | -0.0015 |

**Top single features (target `any`):**

| name | pr_auc_drop | drop_sd |
|---|---|---|
| inj_events_180d | 0.0034 | 0.0007 |
| days_since_injury_spell | 0.0024 | 0.0011 |
| season_min_cum | 0.0023 | 0.0005 |
| acwr_3_15 | 0.0014 | 0.0000 |
| season_missed_to_date | 0.0012 | 0.0001 |
| days_since_upper_leg | 0.0011 | 0.0004 |
| days_since_last_injury | 0.0010 | 0.0001 |
| days_since_injury_event | 0.0008 | 0.0003 |
| inj_events_career | 0.0007 | 0.0003 |
| last_spell_length | 0.0007 | 0.0003 |
| fta_per_min_10 | 0.0006 | 0.0002 |
| season_games_played | 0.0005 | 0.0007 |

**Age, re-read on the fixed label (test seasons):**

| age_band | rows | observed_rate | mean_pred | players |
|---|---|---|---|---|
| <=22 | 25546 | 0.0183 | 0.0116 | 369 |
| 23-25 | 42048 | 0.0180 | 0.0133 | 620 |
| 26-28 | 35965 | 0.0197 | 0.0154 | 442 |
| 29-31 | 27239 | 0.0201 | 0.0171 | 306 |
| 32-34 | 14705 | 0.0280 | 0.0227 | 171 |
| 35+ | 7691 | 0.0231 | 0.0201 | 81 |

## 3. Minor vs major injuries

`lift_over_base` = PR-AUC / base rate. The fantasy project found recurring minor absences persist (r = 0.30) and major spells barely do (r = 0.05).

| target | horizon | model | base_rate | pr_auc | lift_over_base | pr_auc_diff_vs_history | seasons_beating_history |
|---|---|---|---|---|---|---|---|
| any | 1 | gbm | 0.0201 | 0.0409 | 1.9580 | -0.0006 | 1 |
| any | 1 | history_only | 0.0201 | 0.0414 | 2.1113 | 0.0000 | 0 |
| any | 3 | gbm | 0.0554 | 0.1127 | 1.9238 | 0.0054 | 2 |
| any | 3 | history_only | 0.0554 | 0.1077 | 1.9453 | 0.0000 | 0 |
| major | 1 | gbm | 0.0043 | 0.0070 | 1.6469 | -0.0007 | 1 |
| major | 1 | history_only | 0.0043 | 0.0077 | 1.8721 | 0.0000 | 0 |
| major | 3 | gbm | 0.0115 | 0.0187 | 1.5773 | -0.0010 | 0 |
| major | 3 | history_only | 0.0115 | 0.0200 | 1.8227 | 0.0000 | 0 |
| minor | 1 | gbm | 0.0158 | 0.0344 | 2.1023 | -0.0007 | 1 |
| minor | 1 | history_only | 0.0158 | 0.0350 | 2.2867 | 0.0000 | 0 |
| minor | 3 | gbm | 0.0441 | 0.0909 | 1.9384 | 0.0012 | 1 |
| minor | 3 | history_only | 0.0441 | 0.0901 | 2.0243 | 0.0000 | 0 |

**Spell length (games missed per new injury spell, uncensored spells):**

| model | mae | spearman |
|---|---|---|
| gbm_pre_injury | 4.877 | 0.250 |
| gbm_with_note | 4.449 | 0.301 |
| median_all | 4.162 |  |
| median_by_note_class | 3.991 | 0.195 |

## 4. Rest and schedule: the causal slice

Linear probability model, player fixed effects + season dummies, controls for home, travel, time-zone shift and age; SEs clustered by player. Effects in percentage points of P(new injury spell starts next game).

| treatment | label | rate_treated | rate_control | naive_diff_pp | fe_effect_pp | fe_ci_lo_pp | fe_ci_hi_pp | fe_p_value |
|---|---|---|---|---|---|---|---|---|
| b2b | y_any_h1 | 0.013 | 0.015 | -0.233 | -0.099 | -0.189 | -0.010 | 0.029 |
| b2b | y_minor_h1 | 0.009 | 0.012 | -0.216 | -0.108 | -0.185 | -0.032 | 0.006 |
| b2b | y_major_h1 | 0.003 | 0.004 | -0.017 | 0.009 | -0.037 | 0.055 | 0.700 |
| three_in_four | y_any_h1 | 0.013 | 0.015 | -0.213 | -0.086 | -0.167 | -0.005 | 0.037 |
| three_in_four | y_minor_h1 | 0.010 | 0.012 | -0.201 | -0.098 | -0.167 | -0.030 | 0.005 |
| three_in_four | y_major_h1 | 0.003 | 0.004 | -0.011 | 0.012 | -0.029 | 0.053 | 0.570 |
| four_in_six | y_any_h1 | 0.013 | 0.015 | -0.213 | -0.084 | -0.163 | -0.005 | 0.038 |
| four_in_six | y_minor_h1 | 0.010 | 0.012 | -0.187 | -0.084 | -0.152 | -0.015 | 0.016 |
| four_in_six | y_major_h1 | 0.003 | 0.004 | -0.025 | 0.000 | -0.039 | 0.039 | 0.994 |

**Selection check:** share of rostered player-games sat out, by B2B status. If at-risk players are rested on B2B nights, the estimate above is biased toward zero.

| b2b_second_night | rostered_player_games | sat_any | sat_rest_coded | sat_unexplained | sat_injury |
|---|---|---|---|---|---|
| 0 | 420340 | 0.2500 | 0.0003 | 0.1205 | 0.0800 |
| 1 | 118104 | 0.2501 | 0.0009 | 0.1189 | 0.0824 |

## 5. Team Health Index

Correlation of the index (computed before tip-off) with actual minutes lost to new injury spells over the next 10 games, non-overlapping windows, team-bootstrap CI.

| model | index | windows | pearson | spearman | spearman_lo | spearman_hi |
|---|---|---|---|---|---|---|
| gbm | predicted_lost_minutes | 1620 | 0.255 | 0.317 | 0.273 | 0.359 |
| gbm | fragility | 1620 | 0.218 | 0.232 | 0.182 | 0.280 |
| history_only | predicted_lost_minutes | 1620 | 0.307 | 0.338 | 0.294 | 0.379 |
| history_only | fragility | 1620 | 0.239 | 0.258 | 0.216 | 0.305 |

## 6. Data and labels

**PST -> PLAYER_ID mapping (modelled seasons)**

| match_stage | events | share |
|---|---|---|
| name_team_season | 30241 | 0.9743 |
| unmapped | 308 | 0.0099 |
| name_team_adjacent | 279 | 0.0090 |
| name_season | 119 | 0.0038 |
| name_only | 91 | 0.0029 |
| ambiguous_name_only | 1 | 0.0000 |

**Event classes**

| event_class | events | share |
|---|---|---|
| injury_minor | 6803 | 0.4133 |
| stash | 5972 | 0.3628 |
| injury_major | 1562 | 0.0949 |
| illness | 903 | 0.0549 |
| protocol | 553 | 0.0336 |
| personal | 354 | 0.0215 |
| rest | 269 | 0.0163 |
| other | 26 | 0.0016 |
| suspension | 17 | 0.0010 |

**Absence spells by attributed cause**

| cause | spells | median_games | censored |
|---|---|---|---|
| illness | 787 | 2.0000 | 0.0000 |
| injury_major | 908 | 13.0000 | 0.6311 |
| injury_minor | 6028 | 3.0000 | 0.0715 |
| other | 17 | 3.0000 | 0.0000 |
| personal | 222 | 2.0000 | 0.0090 |
| protocol | 417 | 5.0000 | 0.0000 |
| rest | 199 | 1.0000 | 0.0000 |
| stash | 3268 | 4.0000 | 0.0006 |
| suspension | 9 | 2.0000 | 0.0000 |
| unknown | 38628 | 1.0000 | 0.0004 |

**Label base rates by season**

| season | at_risk_rows | y_any_h1_rate | y_any_h3_rate | y_minor_h1_rate | y_minor_h3_rate | y_major_h1_rate | y_major_h3_rate |
|---|---|---|---|---|---|---|---|
| 2006 | 24917 | 0.0108 | 0.0305 | 0.0082 | 0.0233 | 0.0027 | 0.0073 |
| 2007 | 25086 | 0.0148 | 0.0414 | 0.0112 | 0.0316 | 0.0036 | 0.0098 |
| 2008 | 24830 | 0.0121 | 0.0342 | 0.0097 | 0.0276 | 0.0024 | 0.0067 |
| 2009 | 24625 | 0.0118 | 0.0332 | 0.0081 | 0.0229 | 0.0038 | 0.0103 |
| 2010 | 24806 | 0.0105 | 0.0300 | 0.0075 | 0.0217 | 0.0030 | 0.0083 |
| 2011 | 25145 | 0.0137 | 0.0378 | 0.0101 | 0.0282 | 0.0037 | 0.0097 |
| 2012 | 20758 | 0.0105 | 0.0295 | 0.0076 | 0.0213 | 0.0029 | 0.0082 |
| 2013 | 25757 | 0.0086 | 0.0243 | 0.0062 | 0.0175 | 0.0024 | 0.0068 |
| 2014 | 25590 | 0.0086 | 0.0238 | 0.0056 | 0.0154 | 0.0030 | 0.0084 |
| 2015 | 25971 | 0.0107 | 0.0302 | 0.0074 | 0.0213 | 0.0033 | 0.0090 |
| 2016 | 26078 | 0.0098 | 0.0267 | 0.0072 | 0.0192 | 0.0027 | 0.0075 |
| 2017 | 26139 | 0.0127 | 0.0354 | 0.0099 | 0.0278 | 0.0028 | 0.0076 |
| 2018 | 26107 | 0.0186 | 0.0515 | 0.0131 | 0.0367 | 0.0056 | 0.0150 |
| 2019 | 26101 | 0.0194 | 0.0549 | 0.0147 | 0.0421 | 0.0046 | 0.0129 |
| 2020 | 22393 | 0.0250 | 0.0699 | 0.0196 | 0.0556 | 0.0054 | 0.0144 |
| 2021 | 23054 | 0.0265 | 0.0705 | 0.0200 | 0.0538 | 0.0065 | 0.0169 |
| 2022 | 26031 | 0.0308 | 0.0838 | 0.0250 | 0.0695 | 0.0058 | 0.0146 |
| 2023 | 25895 | 0.0291 | 0.0804 | 0.0247 | 0.0692 | 0.0043 | 0.0114 |

**Event classifier audit:** hand-labelled sample drawn to `data/external/event_audit_sample.csv`; per-class precision is pending until `true_class` is filled in and `python run.py audit-score` is run.

**Configurations tried:** 562 logged fits, 9 distinct (model, params) configs (`experiments/log.csv`).

## 7. Limits

- **No tracking load.** Public box scores have no distance, speed, jump counts, force-plate or
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
