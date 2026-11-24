# Recommendations: Testing What Public Data Adds

## The goal

The project is no longer trying to build a model that beats a team's medical staff. Teams have
biometric, tracking and medical data we will never see, and no public-data pipeline will out-predict
them. The goal is to **test whether some part of this work gives a sports professional something
they don't already have.**

A team's biometrics tell it a great deal about its own players today. Three things are harder for a
team to get on its own, and this project has all three:

| What a team lacks | What this project has | Hypothesis |
|---|---|---|
| League-wide base rates (one roster is a small sample) | 20 seasons, every team | **A. The return window** (headline) |
| History on players it doesn't own yet | Public injury history for every player | **B. Acquired players** (supporting) |
| A view of risk tied to the schedule, months ahead | The league schedule | **C. Schedule density** (supporting) |

The report's story becomes: *public, league-wide data supplies three things biometrics can't, and
here is how much each one is worth.*

Numbers marked **(check)** come from one-off read-only scripts run against `data/processed/`. They
are not saved in the repo, so treat them as pointers to verify, not results to cite.

---

## Step 0. Fix the labels first

All three hypotheses use the same label: "a new injury spell starts". If the label is wrong, all
three results are wrong in the same way. Do these before any test.

### 0.1 Separate rest days from injuries

The rise in logged injuries is mostly one-game absences. New spells per 1,000 at-risk
player-games, by eventual length **(check)**:

| Spell length | 2013–17 avg | 2022–23 avg | Ratio |
|---|---|---|---|
| 1 game | 2.1 | 11.5 | 5.5x |
| 2 games | 1.4 | 4.4 | 3.2x |
| 3–4 games | 1.8 | 4.5 | 2.5x |
| 5–9 games | 2.1 | 4.3 | 2.1x |
| 10+ games | 2.6 | 4.7 | 1.8x |

Rest-coded PST notes drop from 62 in 2017-18 to 0 in 2018-19 and stay near zero, exactly when load
management grew. So some recent "minor injuries" are probably rest days logged as soreness.

**Do:** add a durable-injury target, a spell of 3+ games (also report 2+), and run every
hypothesis on it as well as on the current target. A result that holds on both is about injuries;
one that holds only on the current target may be about rest decisions.

### 0.2 Finish the classifier audit

`true_class` is blank in all 240 rows of
[event_audit_sample.csv](data/external/event_audit_sample.csv). Errors already visible in the
sample:

- "strained stomach muscle" is classed as illness, because `stomach` is in the illness list at
  [vocab.py:173](src/injuryrisk/labels/vocab.py#L173).
- "dehydraftion" (a typo), "dizziness" and "vestibular condition" fall into `other`.
- "right knee injury / conditioning" and "rest / sore left knee" are classed as rest.

**Do:** fill in the sample and run `python run.py audit-score`. Report per-class precision in the
write-up; a practitioner will ask how trustworthy the labels are.

### 0.3 Fix `missed_lag1`

It counts games before a player joined a roster as missed
([absences.py:201](src/injuryrisk/labels/absences.py#L201)). It averages 30.0 games against 17.6
in-stint absences, and 27% of player-seasons differ by 10+ games **(check)**. It is one of the three
history features, so Hypothesis B depends on it directly.

**Do:** count only absences inside roster spans.

### 0.4 Decide how to treat 2021-22

16% of rotation players' absence spells that season are protocol-coded (Omicron) **(check)**, but
[base.yaml:16](configs/base.yaml#L16) lists only 2019-20 and 2020-21 as COVID seasons. Protocol
spells are already excluded from the target, so the label is probably fine. Two things still need
stating: the at-risk population that season is unusual, and 2021-22 has no usable prior season
(see Hypothesis B).

**Do:** keep it, flag it, and run a sensitivity check without it.

---

## Step 1. Write the hypotheses down before testing

Each hypothesis below has a statement, a direction and a threshold. Write them into the repo (for
example a `configs/hypotheses.yaml` or a section of the report) **before** running the tests, then
report every outcome, null or not. Your own spec explains why: the fantasy project scored 73
configurations, and every "winner" was indistinguishable from best-of-73 luck.

**Multiple testing.** Three primary tests means a Holm correction across A, B and C: the smallest
p-value must clear 0.05/3, the next 0.05/2, the last 0.05. A is the headline either way.

**Holdout.** No labelled season is fully untouched:
- 2023-24 and 2024-25 have features but no labels, because the PST log stops in April 2023.
  Labelling them (newer PST data, or the official NBA injury reports from 2021-22 onward) gives a
  genuine holdout. The official reports also mark rest vs injury, which helps 0.1.
- If that isn't feasible, keep 2022-23 as the confirmation season and say plainly that it was
  scored earlier, in the `smoke` runs in `experiments/log.csv` and in the original report.

---

## Hypothesis A (headline): the return window

### Statement

> A player's risk of a new injury spell is raised in the first games after he returns from an
> injury, compared with **his own** later baseline. The raised risk is at least 1.5x in the first 5
> games back and fades within a measurable number of games.

### Why a professional would use it

The decision is how fast to ramp a returning player's minutes. Medical staff judge whether this
player is ready. This hypothesis gives them something one roster can't produce: a league-wide
estimate of how long the raised-risk window lasts and how large it is. That is a reference point,
not a replacement for their judgement.

### Evidence so far

Risk of a new spell next game, by games played since returning from the last injury spell
(usable seasons) **(check)**:

| Games since return | Risk |
|---|---|
| 0 (first game back) | 4.6% |
| 1–2 | 3.8% |
| 3–5 | 3.0% |
| 6–10 | 2.6% |
| 11–20 | 2.3% |
| 21–40 | 1.9% |
| 41–82 | 1.5% |
| 83+ | 1.05% |
| No recorded injury | 0.7% |

Sample: 50,132 at-risk rows within 10 games of a return, with 1,589 new spells **(check)**. That is
well powered.

### Why the curve alone doesn't prove it

Injury-prone players fill the early bins, because they are the ones who keep returning. The curve
could just mean "injury-prone players get hurt more", which teams already know.

### Test design

1. **Compare each player to himself.** Fit a discrete-time hazard model on the H = 1 risk set with
   games-since-return bins (0, 1–2, 3–5, 6–10, 11–20, 21–40, 41+) and player fixed effects.
   [glm.py](src/injuryrisk/models/glm.py) already has a fixed-effects linear probability model
   (`within_ols`) with player-clustered errors; a fixed-effects logistic model is the stricter
   version. The reference bin is 41+ games since return.
2. **Controls.** Season dummies (logging drift), age, and the length of the previous spell. A longer
   absence probably means a worse injury and a longer recovery.
3. **Decide what to do with minutes on return.** Returning players often play restricted minutes,
   and coaches choose that because of perceived risk. Fit the model with and without recent
   minutes. If the effect survives without them, minutes are not driving it. If it survives with
   them, the window is not just low-minute games.
4. **Primary result:** the risk ratio for 0–5 games back vs 41+, with a 95% confidence interval.
   The hypothesis holds if the interval is above 1.5.
5. **Secondary result:** the first bin whose interval includes 1. That is the estimated length of
   the window.

### Robustness checks

- **Durable injuries only** (0.1). If the window disappears on the 3+ game target, early "re-injuries"
  are mostly one-game rest days after a return, which is a different and weaker finding.
- **Minor vs major** previous injury: does a longer layoff mean a longer window?
- **Same region vs different region.** A new spell in the same body region soon after returning
  looks like an incomplete recovery ("came back too early"); a different region looks like
  compensation or deconditioning. Both are useful to staff, but they are different messages. The
  region groups already exist in [history.py](src/injuryrisk/features/history.py).
- **By era** (2006–17 vs 2018–23), given the logging drift.

### What to present

- One figure: the within-player risk ratio by games since return, with confidence bands.
- One sentence in practitioner units, for example "the first N games back carry X times a player's
  normal risk; across the league that is about Y extra games lost per team per season."

---

## Hypothesis B (supporting): acquired players

### Statement

> For a player in his first season with a new team, public injury history ranks his risk of new
> injury spells better than what a front office would know without it (age, minutes, role).

### Why a professional would use it

When a team trades for or signs a player, it doesn't have the previous team's biometric or medical
data. Public injury history is what it has. If that history ranks acquired players' risk, it is
directly usable in trade and free-agency evaluation.

### Sample

Among rows with a prior season: 88,854 new-team rows, 929 players, 1,135 new spells **(check)**.
That is about 100 spells per test season, so **pool the test seasons**; per-season tests will be
too noisy to read. 2021-22 drops out because its prior season (2020-21) is excluded as a COVID
season, unless you allow 2020-21 as history only (features, never labels). That would be a
reasonable rule to write down in Step 1.

### Test design

1. Define "new team" as a stint with a team the player did not appear for in the previous season.
2. Compare three models on new-team rows only, using the existing forward chain:
   - **What a front office knows without injury data:** age, prior-season minutes per game, recent
     minutes, usage (the `history_age_role` features minus the history terms).
   - **That plus public injury history:** `history_age_role` as it is.
   - Base rate.
3. **Primary result:** the paired player-bootstrap difference in PR-AUC between the second and first
   models, pooled across test seasons. The hypothesis holds if the 95% interval is above zero.
4. Run the same comparison on same-team rows. If history helps just as much there, the finding is
   simply "history helps", which is fine. If it helps *more* on new-team rows, that is the stronger
   claim, because the team would have its own data for players it already had.

### What to present

Expected games missed in the first season, by tercile of predicted risk, for acquired players. For
example: "acquired players in the top third of public-history risk missed X games to new injuries,
against Y for the bottom third."

**A null result is still usable.** "Public injury history does not help evaluate acquired players
beyond age and role" is something a front office would want to know.

---

## Hypothesis C (supporting): schedule density

### Statement

> Players are more likely to miss the next game when it is on the next day, after controlling for
> rest before the current game.

### Why a professional would use it

The schedule is known months ahead, while biometrics describe the present. A schedule effect lets
staff plan minutes around the dense stretches in advance.

### Important caution

This hypothesis measures **availability**, not necessarily injury. A team choosing to sit a sore
player before a next-day game looks the same in the data as an injury. The current evidence points
that way: the effect appears for minor spells only.

### What's already known

- The back-to-back "protective effect" in the current report is an artifact. After the second
  night of a back-to-back, the next game is almost never the next day: 0.6% of rows, against 28.6%
  otherwise **(check)**. Adding the gap to the next game as a control moves the back-to-back
  estimate from −0.099pp [−0.189, −0.010] to −0.013pp [−0.108, +0.081], p = 0.78 **(check)**.
- With that control, a next-day game adds +0.25pp [0.10, 0.40] to the chance of a new spell for
  minor injuries only, and nothing for major ones **(check)**.

**Do:** add the next-game gap to `CONTROLS` in [b2b.py:26](src/injuryrisk/causal/b2b.py#L26),
and correct report section 4, which currently states the artifact as a finding. The same bias
affects the `b2b` odds ratio of 0.955 in report section 2.

### Test design

1. Player fixed effects and season dummies, as now, plus the next-game gap.
2. **Dose response:** days until the next game as 1, 2, 3, 4+ rather than one flag. A real effect
   should shrink steadily as the gap grows.
3. **Placebo outcome:** illness spells. The schedule should not cause illness. If next-day games
   "raise" illness absences too, the design is picking up something else, probably teams managing
   availability.
4. **Durable target** (0.1). If the effect disappears on 3+ game spells, it is about short absences
   and decisions to sit, not injuries.
5. **Primary result:** the next-day-game effect on the current target, with player-clustered 95%
   interval. The hypothesis holds if the interval is above zero.

### What to present

State it as an availability finding unless the durable-target and placebo checks say otherwise:
"next-day games raise the chance a player misses the following game by X points; most of that is
short absences."

---

## Presenting the results

- **Lead with figures.** The report currently has none (matplotlib isn't installed). Three would
  carry the story: the return-window curve (A), expected games missed by risk tercile for acquired
  players (B), and the next-game-gap dose response with the illness placebo (C).
- **Use practitioner units.** Games lost per team per season, or "X times a player's normal risk".
  Odds ratios per standard deviation and PR-AUC belong in an appendix.
- **Say what biometrics do better.** A short section on where this sits next to a team's own data:
  it is a league-wide reference and a tool for players outside the team's monitoring, not a
  replacement for individual readiness assessment.
- **Keep the current model results as context.** The finding that box-score workload adds nothing
  beyond injury history is worth one paragraph: it tells a professional not to rely on public load
  proxies, and it motivates looking elsewhere.

---

## Lower priority under the new goal

These were in the earlier version of this file. They are still real issues but no longer central,
because the project is not competing on predictive accuracy.

| Item | Why it drops | Still worth doing? |
|---|---|---|
| Alert tie-break bug ([metrics.py:46](src/injuryrisk/evaluate/metrics.py#L46)): a constant score flags the lowest `PLAYER_ID`s, inflating the base-rate baseline | The alert list is no longer a deliverable | Yes if the alert table stays in the report; it's a one-line fix |
| GBM overfitting (training PR-AUC 0.158 vs 0.019 on validation) | Not competing on accuracy | No |
| Probabilities too low in recent seasons | Same | No |
| Team health index inflated by pooling seasons (0.317 pooled, 0.199 within-season) | Not a hypothesis deliverable | Only if the index stays in the report; then report the within-season figure |
| Spell-length scoring mismatch (Poisson loss, absolute-error scoring) | Medical staff estimate return timelines far better | No |
| Tighter alert budgets | Same as the alert list | No |
| `known_absences.csv` is empty and would match by season, not date ([absences.py:176](src/injuryrisk/labels/absences.py#L176)) | Unused | Only if you start filling it |
| Missing tests for metrics, bootstrap, splits and name mapping | Engineering hygiene | Add tests for any new code written for A, B and C |
| `scipy` undeclared in [pyproject.toml](pyproject.toml); versions unpinned | Engineering hygiene | Yes, quick |

---

## Checked and found clean

- **Note-timing leakage:** only 0.3% of positives have their cause note dated before tip-off.
- **Stash spells:** for rotation players there are just 319, median 1 game, so they are not a
  hidden injury pool.

---

## Suggested order of work

1. Step 0: durable target, classifier audit, `missed_lag1` fix, 2021-22 decision.
2. Step 1: write the three hypotheses and thresholds into the repo; decide the holdout.
3. Hypothesis C's control fix (it corrects a finding the report currently states).
4. Hypothesis A, then its robustness checks.
5. Hypothesis B.
6. Apply the Holm correction, make the three figures, rewrite the report around the umbrella story.
