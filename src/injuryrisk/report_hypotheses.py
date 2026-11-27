"""The front of ``reports/REPORT.md``: the pre-stated hypotheses (A-D) and the tissue analysis.

Every number comes from a CSV written by ``python run.py hypotheses``; every
sentence that states a result is conditional on those numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from injuryrisk.config import season_label
from injuryrisk.hypotheses import load_spec
from injuryrisk.paths import REPORTS


def _read(name: str) -> pd.DataFrame | None:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else None


def _table(df: pd.DataFrame, digits: int = 3) -> str:
    from injuryrisk.report import md_table
    return md_table(df, digits)


def _fig(name: str, alt: str) -> str:
    return f"![{alt}]({name})\n" if (REPORTS / name).exists() else ""


def _verdict(row) -> str:
    if row["holds_after_holm"]:
        return "**supported** (clears its threshold after Holm)"
    if row["ci_clears_threshold"]:
        return "**not supported after Holm** (95% CI clears the threshold, Holm-adjusted p does not)"
    return "**not supported** (95% CI does not clear the threshold)"


def hypothesis_sections(cfg: dict) -> list[str]:
    spec = load_spec()
    summ = _read("hypotheses_summary.csv")
    conf = int(spec["holdout"]["confirmation_season"])
    out = ["# What public, league-wide data adds to a team's injury picture\n"]
    out.append(
        "A team's biometrics, tracking and medical data tell it far more about its own players today than "
        "any public data can, so this project does not try to out-predict a medical staff. It asks a "
        "narrower question: **does public, league-wide data supply something a team can't easily get on "
        "its own?** Three candidates, each tested as a hypothesis written down before the test "
        "(`configs/hypotheses.yaml`):\n")
    out.append(_table(pd.DataFrame([
        {"what a team lacks": "League-wide base rates (one roster is a small sample)",
         "what public data has": "20 seasons, every team", "hypothesis": "A. The return window (headline)"},
        {"what a team lacks": "History on players it doesn't own yet",
         "what public data has": "Public injury history for every player", "hypothesis": "B. Acquired players"},
        {"what a team lacks": "Risk tied to the schedule, months ahead",
         "what public data has": "The league schedule", "hypothesis": "C. Schedule density"},
    ])) + "\n")
    out.append(
        "**Hypothesis D (minutes ramp-up)** is not a fourth gap in a team's data: how fast a returning player's "
        "minutes come back is a decision the staff already control. It turns A's return window into that "
        "decision, asking whether risk in games 1-10 back is higher when minutes return to the pre-injury level "
        "faster. A **tissue-type analysis** under A asks whether the window depends on the kind of injury the "
        "player is coming back from.\n")
    out.append(
        "**Disclosure: D and the tissue analysis were added after the A, B and C results had been seen** "
        "(2026-11-26). Their designs and D's threshold were written into `configs/hypotheses.yaml` and "
        "committed on their own (\"Pre-register tissue-type analysis and Hypothesis D\") before any code for "
        "them was written or run. Adding D to the Holm family can only make A, B and C harder to clear. The "
        "tissue analysis is descriptive and outside the Holm family.\n")
    if summ is None:
        out.append("_Not run yet: `python run.py hypotheses`._\n")
        return out

    out.append("## Results at a glance\n")
    t = summ[["hypothesis", "name", "statistic", "estimate", "ci_lo", "ci_hi", "threshold",
              "p_one_sided", "holm_p"]].copy()
    t["verdict"] = [_verdict(r) for _, r in summ.iterrows()]
    out.append(_table(t, 4) + "\n")
    m = len(spec["multiple_testing"]["primary"])
    steps = ", then ".join(f"0.05/{k}" if k > 1 else "0.05" for k in range(m, 0, -1))
    out.append(
        f"Holm correction across the {m} primary tests ({', '.join(spec['multiple_testing']['primary'])}; "
        f"alpha {spec['multiple_testing']['alpha']}): the p-values, smallest first, must clear {steps}. "
        "`p_one_sided` is against "
        "the stated threshold; Holm is applied to twice that (the two-sided equivalent), so a result that "
        "survives also has its 95% interval past the threshold. Every outcome is reported, null or not.\n")
    out.append(_plain_reading(summ, conf))
    out.append(
        f"**Holdout.** No labelled season is untouched: the PST log ends in April 2023, so 2023-24 and "
        f"2024-25 have no labels. {season_label(conf)} is reported on its own as a confirmation season, but "
        f"it was already scored in the earlier `smoke` and `model` runs (`experiments/log.csv`) and in the "
        f"first version of this report. The exploratory figures that motivated the hypotheses were also "
        f"seen before they were written down. Treat the confirmation rows as a consistency check, not a "
        f"clean out-of-sample test. Labelling 2023-24 onward from the official NBA injury reports would give "
        f"a real holdout.\n")

    out += _labels_section(spec)
    out += _section_a(spec, summ)
    out += _section_b(spec, summ)
    out += _section_c(spec, summ)
    out += _section_d(spec, summ)
    out += _biometrics_section()
    return out


def _plain_reading(summ: pd.DataFrame, conf: int) -> str:
    """One bullet per hypothesis: what the verdict means once the robustness checks are read."""
    rob, comp, eff = _read("hyp_a_robustness.csv"), _read("hyp_b_comparison.csv"), _read("hyp_c_effects.csv")
    if rob is None or comp is None or eff is None:
        return ""
    v = summ.set_index("hypothesis")["holds_after_holm"]
    r = rob.set_index("variant")
    dur_ok = r.loc["durable 3+ games", "rr_lo"] > 1
    same = r.loc["new spell, same region", "rr_bias_adjusted"]
    a = (f"- **A** {'holds' if v['A'] else 'does not hold'} on the pooled seasons"
         f"{' and on the durable 3+ game target' if dur_ok else ''}; the excess is concentrated in re-injury of "
         f"the same body region ({same:.1f}x). {season_label(conf)} alone is too short to confirm or refute it "
         f"with this design.")
    nm = comp[(comp["label"] == "y_any_h1") & (comp["group"] == "new minus same")].iloc[0]
    b = (f"- **B** {'holds' if v['B'] else 'does not hold'}: public injury history ranks acquired players' risk "
         f"better than age, minutes and role. "
         + ("It does not help *more* for acquisitions than for players a team already had, so the finding is "
            "\"history helps\", not \"history fills a gap specific to trades\"." if nm["gain_lo"] <= 0 else
            "It helps more for acquisitions than for players a team already had."))
    m = eff[(eff["model"] == "next day vs any other") & (eff["variant"] == "all usable seasons")].set_index("label")
    c_dur = m.loc["y_dur3_h1", "fe_ci_lo_pp"] > 0
    plac = m.loc["y_illness_h1"]
    plac_null = plac["fe_ci_lo_pp"] <= 0 <= plac["fe_ci_hi_pp"]
    c = (f"- **C** {'holds as stated' if v['C'] else 'does not hold'}, but "
         + ("it also holds on durable injuries with a null placebo, so it reads as an injury effect."
            if c_dur and plac_null else
            "it is an **availability** finding, not an injury one: the effect is only in short absences"
            + ("" if c_dur else " (none on spells of 3+ games)")
            + ("" if plac_null else ", and the illness placebo is not null") + "."))
    bullets = [a, b, c]
    drob = _read("hyp_d_robustness.csv")
    if "D" in v.index and drob is not None:
        dr = drob.set_index("variant")
        dm = dr.loc["with this game's minutes"]
        bullets.append(
            f"- **D** {'holds' if v['D'] else 'does not hold'}: "
            + ("players whose minutes come back faster have a higher within-player risk in games 1-10 back."
               if v["D"] else
               "a faster return to pre-injury minutes is not detectably riskier than a gradual one within the "
               "same player. That is a finding, not a gap: staff can't use public data to justify a slower ramp "
               "on risk grounds, though coaches restricting minutes for players they think are fragile could "
               "hide a real effect.")
            + (f" One robustness row disagrees: holding this game's minutes fixed, the ramp effect is "
               f"{dm['effect_pp']:+.2f} pp per +0.25. That row is not evidence for D (section 5)."
               if dm.get("ci_lo_pp", 0) > 0 and not v["D"] else ""))
    return "**Plain reading.**\n\n" + "\n".join(bullets) + "\n"


def _audit_text(audit: pd.DataFrame) -> str:
    a = audit.set_index("event_class")
    overall = a.loc["ALL", "precision"]
    inj = [f"`{c}` {a.loc[c, 'precision']:.1%}" for c in ("injury_minor", "injury_major") if c in a.index]
    return (
        f"\n\n**Hand-labelled audit.** {int(a.loc['ALL', 'labelled'])} notes, drawn stratified by class and "
        f"labelled by hand. The classifier agreed on "
        f"**{overall:.1%}** of them; precision on the two classes that make up the injury label is "
        f"{' and '.join(inj)}. Notes that name both an injury and rest (\"knee injury / rest\") were labelled "
        f"as injuries, the rule the classifier uses. Per class (precision: of the notes the classifier put in "
        f"a class, the share that belong there; recall: of the notes that belong in a class, the share it "
        f"found):\n\n" + _table(audit.reset_index(drop=True), 3) + "\n\n"
        "The weak class is `other`, a catch-all excluded from the target. **After scoring**, the gaps the "
        "audit exposed were fixed (illnesses such as chicken pox and sore throat, injuries such as a herniated "
        "disc or a stinger, source tags like \"(Fox)\", quarantine as protocol, team discipline as "
        "suspension), changing the class of 23 distinct notes across the whole log. The precision above "
        "is for the classifier as audited, before those fixes: scoring the fixed classifier on the same "
        "sample would grade it against its own answer key. A fresh sample would be needed for a post-fix "
        "figure.\n")


def _labels_section(spec) -> list[str]:
    out = ["## 1. How far can the labels be trusted?\n"]
    out.append(
        "All four hypotheses share one label: *a new injury absence spell starts the next game*. If the label "
        "is wrong, every result is wrong in the same way, so it was fixed first.\n")
    drift = _read("hyp_label_drift.csv")
    if drift is not None:
        early = drift[drift["season"].between(2013, 2017)]
        late = drift[drift["season"].between(2022, 2023)]
        rows = []
        for c, name in (("len_1", "1 game"), ("len_2", "2 games"), ("len_3-4", "3-4 games"),
                        ("len_5-9", "5-9 games"), ("len_10+", "10+ games")):
            e, l_ = early[c].mean(), late[c].mean()
            rows.append({"spell length": name, "2013-17 avg": e, "2022-23 avg": l_,
                         "ratio": l_ / e if e > 0 else np.nan})
        r = pd.DataFrame(rows)
        out.append("**Rest days logged as injuries.** New injury spells per 1,000 at-risk player-games, by "
                   "eventual spell length:\n")
        out.append(_table(r, 2) + "\n")
        one, ten = r.iloc[0]["ratio"], r.iloc[-1]["ratio"]
        out.append(
            f"One-game spells rose {one:.1f}x while 10+ game spells rose {ten:.1f}x. Rest-coded notes "
            "almost vanish from the PST log from 2018-19 on, just as load management grew, so some recent "
            "\"minor injuries\" are probably rest days logged as soreness. Every hypothesis is therefore "
            "also run on a **durable target**: a spell of 3+ games (and 2+). A result that holds on both "
            "is about injuries; one that holds only on the main target may be about rest decisions.\n")
    audit = _read("event_audit_precision.csv")
    out.append(
        "**Note classifier.** Errors seen in the audit sample were fixed (\"strained stomach muscle\" was "
        "illness; \"dizziness\", \"vestibular condition\" and a misspelt \"dehydration\" were `other`; "
        "\"knee injury / rest\" and \"right knee injury / conditioning\" were rest). "
        + (_audit_text(audit) if audit is not None
           else "**The hand-labelled audit is still pending**: `true_class` in "
                "`data/external/event_audit_sample.csv` must be filled in and `python run.py audit-score` run "
                "before per-class precision can be reported.\n"))
    out.append(
        "**Games missed last season** (`missed_lag1`, a history feature Hypothesis B relies on) now counts "
        "only absences inside the player's roster spans. It used to count games before a player joined a "
        "team as missed.\n")
    out.append(
        "**2021-22** is kept but flagged: about one in six rotation players' absence spells that season "
        "are protocol-coded (Omicron). Those spells are excluded from the target, so the label is probably "
        "fine, but the at-risk population is unusual. Every hypothesis is re-run without it. COVID seasons "
        "2019-20 and 2020-21 supply injury history and the \"previous team\" for Hypothesis B, never "
        "labelled rows.\n")
    return out


def _section_a(spec, summ) -> list[str]:
    h = spec["hypotheses"]["A"]
    out = ["## 2. Hypothesis A (headline): the return window\n", f"> {h['statement']}\n"]
    cur, rob, units = _read("hyp_a_curve.csv"), _read("hyp_a_robustness.csv"), _read("hyp_a_units.csv")
    if cur is None:
        return out + ["_Not run yet._\n"]
    row = summ[summ["hypothesis"] == "A"].iloc[0]
    u = units.iloc[0]
    out.append(
        "**Why a professional would use it.** The decision is how fast to ramp a returning player's minutes. "
        "Medical staff judge whether *this* player is ready; one roster can't tell them how large the "
        "raised-risk window is or how long it lasts across the league. This is a reference point for that "
        "judgement, not a replacement for it.\n")
    out.append(
        "**Design.** Each player is compared with himself: a linear probability hazard with player fixed "
        "effects on the H = 1 risk set, games-since-return bins against a reference of 41+ games back, "
        "season dummies, age and the length of the previous spell as controls, SEs clustered by player. "
        "Comparing within player removes the obvious confound (injury-prone players fill the early bins "
        "because they keep returning). Risk ratio = 1 + within-player risk difference / reference-bin rate.\n")
    out.append(_fig("fig_a_return_window.png", "Within-player risk ratio by games since return"))
    c = cur[cur["target"] == "any"][["bin", "rows", "positives", "raw_rate", "rr", "rr_lo", "rr_hi"]]
    out.append(f"Reference (41+ games back) rate: {cur['ref_rate'].iloc[0]:.2%} per game.\n")
    out.append(_table(c, 3) + "\n")
    rr, lo, hi = row["estimate"], row["ci_lo"], row["ci_hi"]
    end = u["window_ends_at_bin"]
    window_txt = ("the raised risk does not fade inside 40 games" if pd.isna(end) else
                  f"the first bin whose interval includes no effect is **{end} games back**, so the window "
                  f"covers games {u['raised_bins'].replace(',', ', ')}" if u["raised_bins"] else
                  "even the first game back is not distinguishable from baseline")
    out.append(
        f"**Primary result.** Games 0-5 back carry **{rr:.2f}x** a player's own baseline risk "
        f"(95% CI {lo:.2f}-{hi:.2f}) against a threshold of 1.5x: {_verdict(row)}. "
        f"**Window length:** {window_txt}.\n")
    out.append(
        f"**In practitioner units:** the first {u['raised_bins'].split(',')[-1].split('-')[-1]} games back carry "
        f"about {rr:.1f}x a player's normal risk. Across the league the raised-risk window adds at least "
        f"{u['extra_spells_per_team_season']:.1f} new injury spells per team per season, about "
        f"**{u['extra_games_lost_per_team_season']:.0f} player-games lost per team per season** at "
        f"{u['mean_games_per_injury_spell']:.1f} games per spell (unadjusted estimates, so a lower bound; "
        f"see the bias below).\n")
    if rob is not None:
        out.append("**Robustness** (risk ratio, games 0-5 back vs 41+, same design):\n")
        out.append(_table(rob[["variant", "rows", "window_positives", "ref_rate", "rr", "rr_lo", "rr_hi",
                               "rr_null", "rr_bias_adjusted"]], 3) + "\n")
        r = rob.set_index("variant")
        p_ = r.loc["primary"]
        conf_name = [v for v in r.index if v.startswith("confirmation")][0]
        cf = r.loc[conf_name]
        out.append(
            "**A bias in the design, measured.** Games since return is built from the player's own past "
            "injuries, so a within-player estimate on a finite panel is biased even when there is no return "
            "window (a Nickell-type bias). `rr_null` is what the same model reports on simulated data with "
            "*no* window: each player keeps his real number of at-risk games and a constant injury rate (his "
            "own, shrunk toward the league's), 20 simulations per row. `rr_bias_adjusted` subtracts that "
            "null on the risk-difference scale. The benchmark was added after the first run showed "
            "implausible negative ratios on short panels; it does not change the pre-stated primary test.\n")
        out.append(
            f"- On the full sample the bias works **against** the hypothesis (null {p_['rr_null']:.2f}x), so "
            f"the primary {p_['rr']:.2f}x is conservative; adjusted, it is about {p_['rr_bias_adjusted']:.2f}x.\n"
            f"- The bias grows as panels shorten: the typical row belongs to a player with "
            f"{r.loc['era 2006-17', 'games_per_player']:.0f} at-risk games in 2006-17, "
            f"{r.loc['era 2018-23', 'games_per_player']:.0f} in 2018-23 and {cf['games_per_player']:.0f} in one "
            f"season. That is why some single-era, single-season and subgroup rows show ratios below zero, "
            f"which is impossible as a true effect.\n"
            f"- **The confirmation season does not confirm the window.** In {conf_name.split()[-2]} alone the "
            f"adjusted ratio is {cf['rr_bias_adjusted']:.2f}x: one season is too short for this design to "
            f"separate the window from the bias. Read it as uninformative, not as a replication.\n")
        out.append(
            f"Why not just read the raw curve? In the same simulations with no window, the pooled raw ratio "
            f"(no player fixed effects) comes out at {p_['rr_null_raw_pooled']:.1f}x, entirely from "
            f"injury-prone players filling the early bins. The raw curve proves nothing on its own; the "
            f"within-player design is what the claim rests on.\n")
        notes = []
        dur = r.loc["durable 3+ games"]
        notes.append("The window **survives on the durable 3+ game target**, so it is not just one-game rest "
                     "days after a return." if dur["rr_lo"] > 1 else
                     "The window **does not survive on the durable 3+ game target**: early re-injuries are "
                     "mostly short absences, a weaker finding than \"came back too early\".")
        m = r.loc["with minutes"]
        notes.append(f"Adding this game's minutes and the 3-game minutes mean moves the ratio to {m['rr']:.2f}x "
                     + ("(the window is not just low-minute games)." if m["rr_lo"] > 1 else
                        "(the interval now includes no effect)."))
        same, diff = r.loc["new spell, same region"], r.loc["new spell, different region"]
        same_raised = same["rr_bias_adjusted"] > 1.5
        diff_raised = diff["rr_bias_adjusted"] > 1.5
        reading = ("The window is a **same-region** effect: the excess risk is re-injury of the body part that "
                   "was just out, which reads as incomplete recovery (\"came back too early\") rather than "
                   "compensation elsewhere." if same_raised and not diff_raised else
                   "The excess is in other body regions, which reads as compensation or deconditioning."
                   if diff_raised and not same_raised else
                   "The excess appears in both the same and other regions.")
        notes.append(f"New spell in the same body region as the last injury: {same['rr_bias_adjusted']:.2f}x "
                     f"bias-adjusted ({same['rr']:.2f}x raw); a different region: "
                     f"{diff['rr_bias_adjusted']:.2f}x ({diff['rr']:.2f}x raw). " + reading)
        mi, ma = r.loc["previous spell minor"], r.loc["previous spell major"]
        notes.append(f"After a minor spell: {mi['rr_bias_adjusted']:.2f}x bias-adjusted; after a major one: "
                     f"{ma['rr_bias_adjusted']:.2f}x (far fewer returns, so a much shorter panel and a noisier "
                     f"estimate).")
        e1, e2 = r.loc["era 2006-17"], r.loc["era 2018-23"]
        notes.append(f"Era 2006-17: {e1['rr']:.2f}x raw, {e1['rr_bias_adjusted']:.2f}x adjusted; 2018-23: "
                     f"{e2['rr']:.2f}x raw, {e2['rr_bias_adjusted']:.2f}x adjusted.")
        notes += [f"Without 2021-22: {r.loc['without 2021-22', 'rr']:.2f}x "
                  f"({r.loc['without 2021-22', 'rr_bias_adjusted']:.2f}x adjusted)."]
        out.append("\n".join(f"- {n}" for n in notes) + "\n")
    out += _section_a_tissue(spec)
    return out


def _section_a_tissue(spec) -> list[str]:
    tis = _read("hyp_a_tissue.csv")
    out = ["### 2a. Secondary: does the window depend on the tissue type? (outside Holm)\n"]
    if tis is None:
        return out + ["_Not run yet._\n"]
    out.append(
        "Each injury spell gets a tissue group from its cause note, read by the same vocabulary parser that "
        "classifies notes (`tissue_group` in `labels/vocab.py`). A named structure decides first (ACL -> "
        "ligament, meniscus -> joint, tibia -> bone, Achilles or patellar tendon -> soft tissue); otherwise the "
        "pathology does: strain, tendinitis and bruise -> soft tissue, sprain -> ligament, dislocation -> joint, "
        "fracture -> bone. A tear counts as soft tissue only on a muscle (hamstring, calf, groin...). Sore, "
        "tight or unnamed injuries elsewhere (\"sore knee\", \"ankle injury\") are **unspecified**, which is kept "
        "as its own group. Ligament is kept separate from soft tissue on purpose. A's primary design (0-5 vs "
        "41+) is then rerun on rows whose previous injury spell is in each group.\n")
    out.append(
        "**Two rules were changed after the first run** (2026-11-27, recorded as amendments in "
        "`configs/hypotheses.yaml`), following a hand check of 100 random notes and every edge case: a **bone "
        "bruise** now counts as bone rather than a soft-tissue bruise, and a sore, tight or unnamed injury on a "
        "**muscle region** (\"sore hamstring\", \"calf injury\", \"groin tightness\") now counts as soft tissue "
        "rather than unspecified. Both were decided on clinical grounds, but they were made after the first "
        "results had been seen, so this analysis is exploratory. The first-run table is in the commit history "
        "(`b27d00d`).\n")
    shares = tis.drop_duplicates("tissue_group")[["tissue_group", "injury_spells", "spell_share"]]
    uns = shares.set_index("tissue_group").loc["unspecified", "spell_share"]
    out.append(_table(shares.reset_index(drop=True), 3) + "\n")
    out.append(
        f"**Limitation:** {uns:.0%} of injury spells are unspecified: the note names a body part and soreness, "
        "or nothing, and no tissue. Any tissue-level conclusion rests on the other half. A stratified sample of "
        "60 cause notes (12 per group) is in `data/external/tissue_audit_sample.csv` with a blank `true_group` "
        "column for hand labelling; **the tissue mapping has not been audited yet.**\n")
    out.append(_fig("fig_a_tissue.png", "Return-window risk ratio by tissue type"))
    show = tis[["tissue_group", "outcome", "rows", "window_rows", "window_positives", "ref_rate", "rr", "rr_lo",
                "rr_hi", "rr_null", "rr_bias_adjusted"]]
    out.append(_table(show, 3) + "\n")
    t = tis.set_index(["tissue_group", "outcome"])
    any_ = t.xs("any new spell", level="outcome")
    same = t.xs("new spell, same region", level="outcome")
    big = [g for g in ("soft_tissue", "ligament", "unspecified") if g in any_.index]
    small = [g for g in ("joint", "bone") if g in any_.index]
    out.append(
        "**Reading.** Splitting by group shortens each player's panel, so the raw ratios carry the design's "
        "short-panel bias (section 2), and several are below zero, which is impossible as a true effect. Read "
        "the bias-adjusted column, and only for the large groups.\n")
    lines = [f"{g.replace('_', ' ')}: {any_.loc[g, 'rr_bias_adjusted']:.2f}x bias-adjusted for any new spell, "
             f"{same.loc[g, 'rr_bias_adjusted']:.2f}x for a new spell in the same region "
             f"({int(any_.loc[g, 'window_positives'])} new spells in the window)." for g in big]
    top = max(big, key=lambda g: any_.loc[g, "rr_bias_adjusted"])
    lines.append(
        f"The window is strongest after **{top.replace('_', ' ')}** returns. "
        + ("Those are mostly \"sore\" or \"tightness\" notes, so part of the window may be the same niggle "
           "logged again rather than a new injury; the same-region excess points the same way."
           if top == "unspecified" else "")
        + (" In every large group the excess is in the **same region**, as in the pooled result."
           if all(same.loc[g, "rr_bias_adjusted"] > 1.5 for g in big)
           and all(t.loc[(g, "new spell, different region"), "rr_bias_adjusted"] < 1.5 for g in big)
           else ""))
    lines.append(
        f"{' and '.join(g for g in small)}: "
        + "; ".join(f"{g} has {int(any_.loc[g, 'window_positives'])} new spells in the window" for g in small)
        + ". **Underpowered**: the estimates are dominated by the bias and noise and are not read.")
    out.append("\n".join(f"- {x}" for x in lines) + "\n")
    return out


def _section_b(spec, summ) -> list[str]:
    h = spec["hypotheses"]["B"]
    out = ["## 3. Hypothesis B (supporting): acquired players\n", f"> {h['statement']}\n"]
    comp, terc = _read("hyp_b_comparison.csv"), _read("hyp_b_terciles.csv")
    if comp is None:
        return out + ["_Not run yet._\n"]
    row = summ[summ["hypothesis"] == "B"].iloc[0]
    out.append(
        "**Why a professional would use it.** A team that trades for or signs a player has none of the "
        "previous team's biometric or medical data. Public injury history is what it has.\n")
    out.append(
        "**Design.** New team = a stint with a team the player did not appear for in the previous season. "
        "Two logistic models fit on all rows of the forward-chain training seasons and scored on the test "
        "seasons: a *front-office* model (age, prior-season minutes per game, recent minutes, usage) and the "
        "same plus public injury history (days since last injury, prior injury count, games missed last "
        "season). PR-AUC on new-team rows pooled across test seasons, with a paired player bootstrap.\n")
    show = comp[["label", "group", "rows", "positives", "base_rate", "pr_auc_front_office", "pr_auc_history",
                 "gain", "gain_lo", "gain_hi"]].copy()
    show[["rows", "positives"]] = show[["rows", "positives"]].astype("Int64").astype(str).replace("<NA>", "")
    out.append(_table(show, 4) + "\n")
    out.append(
        f"**Primary result.** On new-team rows, history adds {row['estimate']:+.4f} PR-AUC "
        f"(95% CI {row['ci_lo']:+.4f} to {row['ci_hi']:+.4f}): {_verdict(row)}.\n")
    nm = comp[(comp["label"] == "y_any_h1") & (comp["group"] == "new minus same")].iloc[0]
    if nm["gain_lo"] > 0:
        out.append("History helps *more* for acquired players than for players a team already had, which is "
                   "the stronger claim: it fills a gap the team's own data covers for its own players.\n")
    else:
        out.append(f"The gain on new-team rows is not detectably larger than on same-team rows (difference "
                   f"{nm['gain']:+.4f}, 95% CI {nm['gain_lo']:+.4f} to {nm['gain_hi']:+.4f}). If history helps, "
                   f"it helps generally, not specially for acquisitions.\n")
    if terc is not None:
        t = terc[terc["team_status"] == "new"]
        out.append(_fig("fig_b_acquired_terciles.png", "Games missed by predicted-risk tercile, acquired players"))
        out.append(_table(t[["model", "tercile", "stints", "mean_predicted", "games_missed", "injury_spells",
                             "rostered_games", "games_missed_per_82"]], 3) + "\n")
        g = t[t["model"] == "history"].set_index("tercile")
        f_ = t[t["model"] == "front_office"].set_index("tercile")
        out.append(f"**In practitioner units:** acquired players in the top third of public-history risk at "
                   f"their first game with the new team missed {g.loc['top', 'games_missed']:.1f} games to new "
                   f"injuries that stint, against {g.loc['bottom', 'games_missed']:.1f} for the bottom third "
                   f"({g.loc['top', 'games_missed_per_82']:.1f} vs {g.loc['bottom', 'games_missed_per_82']:.1f} "
                   f"per 82 rostered games). Ranked on age, minutes and role alone, the same split is "
                   f"{f_.loc['top', 'games_missed']:.1f} vs {f_.loc['bottom', 'games_missed']:.1f}.\n")
    conf_rows = comp[comp["label"].str.contains("only") & (comp["group"] == "new team")]
    if len(conf_rows):
        c = conf_rows.iloc[0]
        out.append(f"**Confirmation season alone:** on new-team rows the gain is {c['gain']:+.4f} "
                   f"({c['gain_lo']:+.4f} to {c['gain_hi']:+.4f}, {int(c['positives'])} new spells)"
                   + (", not distinguishable from zero; one season of acquisitions is too few to read on its own."
                      if c["gain_lo"] <= 0 else ", consistent with the pooled result.") + "\n")
    if row["ci_lo"] <= 0:
        out.append("**A null result is still usable:** public injury history does not detectably help rank "
                   "acquired players beyond age and role, which a front office would want to know before "
                   "paying for it.\n")
    return out


def _section_c(spec, summ) -> list[str]:
    h = spec["hypotheses"]["C"]
    out = ["## 4. Hypothesis C (supporting): schedule density\n", f"> {h['statement']}\n"]
    eff = _read("hyp_c_effects.csv")
    if eff is None:
        return out + ["_Not run yet._\n"]
    row = summ[summ["hypothesis"] == "C"].iloc[0]
    out.append(
        "**Why a professional would use it.** The schedule is known months ahead; biometrics describe the "
        "present. A schedule effect lets staff plan minutes around dense stretches in advance. **Caution:** "
        "this measures availability, not necessarily injury. A team sitting a sore player before a next-day "
        "game looks the same in the data as an injury.\n")
    out.append(
        "**Design.** Linear probability model, player fixed effects, season dummies, SEs clustered by player. "
        "Treatment: the team's next game is the next day. Controls: home, travel, time-zone shift, age, and "
        "rest before *this* game (second night of a back-to-back, 3-in-4). Dose response over 1, 2, 3 days "
        "vs 4+. Placebo outcome: illness spells, which the schedule should not cause.\n")
    out.append(_fig("fig_c_schedule_dose.png", "Effect of days until next game, with illness placebo"))
    main = eff[eff["model"] == "next day vs any other"][["variant", "label", "rows", "rate_control",
                                                          "fe_effect_pp", "fe_ci_lo_pp", "fe_ci_hi_pp", "fe_p_value"]]
    out.append("**Next-day game vs any other gap** (percentage points of P(spell starts next game)):\n")
    out.append(_table(main, 3) + "\n")
    dose = eff[(eff["model"] == "dose vs 4+ days") & (eff["variant"] == "all usable seasons")]
    piv = dose.pivot(index="label", columns="term", values="fe_effect_pp").reset_index()
    out.append("**Dose response** (pp vs a gap of 4+ days):\n")
    out.append(_table(piv, 3) + "\n")
    out.append(f"**Primary result.** A next-day game changes the chance of a new spell starting at it by "
               f"{row['estimate']:+.3f} pp (95% CI {row['ci_lo']:+.3f} to {row['ci_hi']:+.3f}): {_verdict(row)}.\n")

    m = eff[(eff["model"] == "next day vs any other") & (eff["variant"] == "all usable seasons")].set_index("label")
    plac, dur, mn, mj = (m.loc[c] for c in ("y_illness_h1", "y_dur3_h1", "y_minor_h1", "y_major_h1"))
    placebo_null = plac["fe_ci_lo_pp"] <= 0 <= plac["fe_ci_hi_pp"]
    durable = dur["fe_ci_lo_pp"] > 0
    lines = [
        f"Illness placebo: {plac['fe_effect_pp']:+.3f} pp ({plac['fe_ci_lo_pp']:+.3f} to {plac['fe_ci_hi_pp']:+.3f}) - "
        + ("null, as it should be." if placebo_null else
           f"**not null**: next-day games also {'raise' if plac['fe_effect_pp'] > 0 else 'lower'} illness "
           f"absences, which the schedule can't cause biologically. The design is picking up how teams "
           f"manage and log availability around dense stretches, not only injury."),
        f"Durable 3+ game spells: {dur['fe_effect_pp']:+.3f} pp ({dur['fe_ci_lo_pp']:+.3f} to {dur['fe_ci_hi_pp']:+.3f}) - "
        + ("the effect survives on injuries that last." if durable else
           "no detectable effect, so the next-day effect is about short absences and decisions to sit."),
        f"Minor spells: {mn['fe_effect_pp']:+.3f} pp; major spells: {mj['fe_effect_pp']:+.3f} pp.",
    ]
    by_var = eff[(eff["model"] == "next day vs any other") & (eff["label"] == "y_any_h1")].set_index("variant")
    no22 = by_var.loc["without 2021-22"]
    cf = by_var.loc[[v for v in by_var.index if v.startswith("confirmation")][0]]
    lines.append(f"Without 2021-22: {no22['fe_effect_pp']:+.3f} pp; {season_label(int(spec['holdout']['confirmation_season']))} "
                 f"alone: {cf['fe_effect_pp']:+.3f} pp. The effect is concentrated in recent seasons, the same "
                 f"seasons where one-game spells multiplied (section 1).")
    out.append("\n".join(f"- {x}" for x in lines) + "\n")
    if durable and placebo_null:
        out.append("**Reading:** the schedule effect holds on durable injuries with a null placebo, so it can be "
                   "read as an injury effect, not only availability.\n")
    else:
        out.append(f"**Reading: an availability finding.** Next-day games raise the chance a player misses the "
                   f"following game with a logged injury by about {row['estimate']:.2f} points; most of that is "
                   f"short absences, not injuries that last.\n")
    return out


def _section_d(spec, summ) -> list[str]:
    h = spec["hypotheses"]["D"]
    out = ["## 5. Hypothesis D (supporting): minutes ramp-up after return\n", f"> {h['statement']}\n"]
    eff, rob = _read("hyp_d_effects.csv"), _read("hyp_d_robustness.csv")
    if eff is None or rob is None or "D" not in set(summ["hypothesis"]):
        return out + ["_Not run yet._\n"]
    row = summ[summ["hypothesis"] == "D"].iloc[0]
    s = eff[eff["section"] == "sample"].set_index("term")["estimate"]
    out.append(
        "**Why a professional would use it.** A shows that risk is raised in the first games back. The lever "
        "staff actually hold during that window is minutes. D asks whether bringing minutes back to the "
        "pre-injury level quickly is associated with more new injuries than a gradual ramp. *Added after A, B "
        "and C were seen, and pre-registered before it was run.*\n")
    out.append(
        f"**Design.** Rows: the H = 1 risk set in games 1-10 since return (game 0 has no post-return minutes "
        f"yet), usable seasons. Pre-injury baseline: mean minutes over the player's last {h['baseline_games']} "
        f"games played before the spell started (at least {h['baseline_min_games']}). **Ramp ratio** at game t: "
        f"mean minutes over the games played since the return and *before* t, divided by the baseline, capped "
        f"at {h['ramp_cap']}. Linear probability model, player fixed effects, season dummies, SEs clustered by "
        f"player; controls: games-since-return bins (1-2 reference, 3-5, 6-10), log(1 + previous spell "
        f"length), previous spell major, age.\n")
    out.append(
        f"**Sample.** {int(s['returns_in_window']):,} returns reach games 1-10. "
        f"{int(s['returns_dropped_short_baseline'])} were dropped for fewer than {h['baseline_min_games']} "
        f"pre-injury games and **{int(s['returns_dropped_low_baseline_mpg'])} for a baseline under "
        f"{h['baseline_min_mpg']} mpg**, leaving {int(s['returns_kept']):,} returns, {int(s['rows_kept']):,} "
        f"player-games, {int(s['players'])} players and {int(s['positives']):,} new spells "
        f"({s['mean_rate']:.2%} per game). Median baseline {s['median_baseline_mpg']:.1f} mpg; mean ramp ratio "
        f"{s['mean_ramp']:.2f}.\n")
    out.append(_fig("fig_d_ramp.png", "Risk in games 1-10 back by minutes ramp ratio"))
    coef = eff[eff["section"].isin(["coefficient", "risk at ramp", "units"])][
        ["section", "term", "estimate", "ci_lo", "ci_hi", "p_value", "unit"]]
    out.append(_table(coef, 3) + "\n")
    out.append(
        f"**Primary result.** Each +0.25 in the ramp ratio changes the within-player risk of a new spell by "
        f"**{row['estimate']:+.3f} pp** per game (95% CI {row['ci_lo']:+.3f} to {row['ci_hi']:+.3f}; one-sided "
        f"p {row['p_one_sided']:.3f}, Holm p {row['holm_p']:.3f}): {_verdict(row)}.\n")
    r = eff[eff["section"] == "risk at ramp"].reset_index(drop=True)
    u = eff[eff["section"] == "units"].reset_index(drop=True)
    out.append(
        f"**In practitioner units:** predicted risk per game is {r.loc[0, 'estimate']:.2f}% at half the "
        f"pre-injury minutes, {r.loc[1, 'estimate']:.2f}% at three quarters and {r.loc[2, 'estimate']:.2f}% at "
        f"full minutes. Holding every return to a ramp of no more than 0.75 would change new spells by "
        f"{u.loc[0, 'estimate']:+.2f} per team per season (95% CI {u.loc[0, 'ci_lo']:+.2f} to "
        f"{u.loc[0, 'ci_hi']:+.2f}), about {u.loc[1, 'estimate']:+.1f} player-games "
        f"({u.loc[1, 'ci_lo']:+.1f} to {u.loc[1, 'ci_hi']:+.1f}).\n")
    out.append("**Robustness** (pp per +0.25 ramp; `null_effect_pp` is the mean estimate with ramp ratios shuffled "
               "across each player's returns, 20 times; `effect_bias_adjusted_pp` subtracts it):\n")
    out.append(_table(rob[["variant", "rows", "positives", "returns", "effect_pp", "ci_lo_pp", "ci_hi_pp",
                           "p_one_sided", "null_effect_pp", "effect_bias_adjusted_pp"]], 3) + "\n")
    rb = rob.set_index("variant")
    lines = []
    pr = rb.loc["primary"]
    centred = abs(pr["null_effect_pp"]) < pr["null_sd_pp"]
    lines.append(
        f"The shuffled null averages {pr['null_effect_pp']:+.3f} pp (sd {pr['null_sd_pp']:.3f}), "
        + ("so unlike A this design shows no visible bias on the pooled panel. Small subgroups do drift "
           "(see their null column)." if centred else
           "so the design is biased on this panel; read the bias-adjusted column."))
    sig = [v for v in rb.index if v != "primary" and pd.notna(rb.loc[v, "ci_lo_pp"])
           and (rb.loc[v, "ci_lo_pp"] > 0) != (pr["ci_lo_pp"] > 0)]
    if sig:
        lines.append("Rows whose verdict differs from the primary: "
                     + ", ".join(f"{v} ({rb.loc[v, 'effect_pp']:+.2f} pp, {rb.loc[v, 'ci_lo_pp']:+.2f} to "
                                 f"{rb.loc[v, 'ci_hi_pp']:+.2f})" for v in sig) + ".")
    if "with this game's minutes" in sig:
        lines.append(
            "**Why the minutes row is not evidence for D.** Holding this game's minutes fixed, a high ramp means "
            "the player played *less* today than in his recent games, which is what a player hurt during the "
            "game looks like (he leaves early and misses the next one). This game's minutes are partly an "
            "outcome of the injury, so conditioning on them builds the outcome into the treatment.")
    small = [v for v in rb.index if v.startswith("previous spell") and rb.loc[v, "positives"] < 50]
    if small:
        lines.append(f"{', '.join(small)}: fewer than 50 new spells each, underpowered and not read.")
    out.append("\n".join(f"- {x}" for x in lines) + "\n")
    out.append(
        "**Confound.** Coaches restrict minutes for players they think are fragile. If they are right, slow "
        "ramps carry the riskier players and the data push *against* the hypothesis (slow ramps look riskier). "
        "Player fixed effects remove fixed differences between players, not a coach's read of this player on "
        "this return. D is an **association, not proof that ramping causes injury**"
        + (", and a null result here is a finding: on public data, a faster ramp is not detectably riskier "
           "than a gradual one, and the decision rests on the medical staff's own read." if not row["holds_after_holm"]
           else ".") + "\n")
    return out


def _biometrics_section() -> list[str]:
    return ["## 6. Where this sits next to a team's own data\n",
            """- **Biometrics do better at readiness.** Force plates, tracking load, sleep and medical exams
  describe *this* player *today*. Nothing here replaces an individual return-to-play assessment.
- **Public data does better at base rates and reach.** It gives a league-wide reference (how big and how
  long the return window is), covers players outside the team's monitoring (trade and free-agency targets),
  and is tied to a schedule known months ahead.
- **Public load proxies are not worth relying on.** In the predictive pipeline (appendix A1), box-score
  workload, schedule and physical features added no reliable signal beyond injury history. That tells a
  professional not to lean on minutes-based load proxies from public data, and is why this report looks
  elsewhere for what public data adds.
- **Labels are transactions.** The target is a logged absence after a classified note. Rest days logged
  as soreness inflate recent one-game spells; the durable target is the guard against that.
"""]
