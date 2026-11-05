# 12 GPS gaps as an underground-rail signal (SHL, WAIIS)

## Question

Can GPS loss while moving mark underground rail, so that the suggestion engine proposes metro where it now shows
unknown GPS-only legs or reads metro minutes as walking or still? GPS loss while moving means no fix, or only coarse
fixes (> 500 m), with displacement across the gap at vehicle speed.

## Plan

**Models.** 10's **E1** as saved (the tool's default, 11) and **E2** (descriptive). There is no retraining. The gap
signal is applied after the model, in `tmd.legs` and `tmd.suggest`. Everything else is as 11 approved:
- product fusion, then HMM forward-backward;
- GPS rule C1;
- leg rules: E2 `s_veh 0, s_other 3, g 10, l_min 2, c 0.5`; E1 raw HMM legs, `c 0.5`;
- output format `tmd-legs/0.1`.

Seed 0 is the model; seeds 1–2 are the spread (as 11).

**A gap** is the time between two consecutive good fixes (accuracy ≤ 500 m) of one recording or session, measured
on the 20 s track that WAIIS records.
- **Speed across** = the great-circle distance from the last good fix before to the first good fix after, divided by
  the time between them.
- **Moving gap**: length ≥ `g_min` and speed across in [`v_lo`, `v_hi`).
- A minute is *in* the gap if its centre (`ts + 30`) lies strictly inside it.
- Missing GPS alone never counts; there must be movement across the gap.

**Stop-point 1 approved 2026-11-05**, with five changes:
1. The WAIIS train leg was a commuter / regional train with some stations underground. So R2 boosts all rail
   columns, and R1's hint is `underground_rail`, not `metro`. Where accelerometer data exist, the rail sub-mode hint
   stays the model's.
2. The gap definition is fixed by reasoning from the stop-point 1 statistics, not tuned. Only `k` is tuned, per
   model; the other gap settings are reported as sensitivity.
3. The choice rule, the guards and the R1 proposal are approved as proposed.
4. Once, after the rule is fixed: SHL Users 2–3, plus false rail time on the ExtraSensory test users (the guard,
   checked once, no choice made on it). The WAIIS days stay descriptive.
5. The code moves are approved as proposed, with 11's leg scores checked unchanged.

### Stop-points

1. **Plan** (this section; Results, stop-point 1). Approved 2026-11-05.
2. **Dev results** and the proposed `k` per model. Approved 2026-11-05: `k` = 2 for both models, rule on; train and
   subway boosted, not tram; re-run clean with that change, which must keep `k` = 2 within all three limits.
3. **Once:** SHL Users 2–3 (all positions), false rail time on the ExtraSensory test users, and the three WAIIS app
   days with 11's lenient diary comparison (descriptive). Then Results, draft Conclusion, update STATUS.md.

### Gap definition (fixed, approved)

**Moving gap:** length **≥ 5 min**, speed across in **[2, 25) m/s**.
- **5 min:** keeps 79 of 102 subway gaps ≥ 2 min and drops 36 of 48 bus gaps (bus median 3.2 min).
- **2 m/s:** drops the indoor still and walking gaps (median ≤ 0.12 m/s).
- **25 m/s:** keeps train (p90 23 m/s) and drops ExtraSensory's GPS jumps (still at ≥ 40 m/s).

Read as GPS only, this keeps 0.485 of subway minutes on SHL User 1, at rail precision 0.770 and with 31 ExtraSensory
minutes.

### Rule (approved)

- **R1, GPS-only stretches** (11's `gps_only_legs`, no accelerometer). A moving gap inside a GPS-only moving
  stretch becomes its own leg, from the last good fix before the gap to the first good fix after. The rest of the
  stretch stays as 11 made it (still or unknown by its speed across, if ≥ 5 min).
  - The gap leg gets mode `unknown`, hint `underground_rail`, and reason `gps_gap_moving` (gap minutes, speed
    across, the two fix times).
  - Mode `rail` flagged would be proposed only if the gap setting's rail precision on SHL User 1 (GPS only) were
    ≥ 0.8. It is 0.770, so R1 stays `unknown`.
- **R2, minutes with accelerometer data in a moving gap.** Before the HMM decodes, the raw fused probabilities of
  the train and subway columns are multiplied by `k`, then renormalised. Tram is not boosted (changed at the
  stop-point 2 approval: mostly above ground, not checkable on dev, E1's tram class barely fires). There is one `k`
  for both columns: the top level is rail either way, and the gap doesn't separate metro from train.
  - The rail sub-mode hint is the model's, as in 11.
  - A leg that contains gap minutes gets the reason `gps_gap_moving`.
- **R3:** both.

**Tuned:** `k` ∈ {1, 2, 4, 8, 16, 32} per model, on SHL User 1 dev (all positions), seed 0. `k` = 1 is the rule off.
The chosen `k` and `k` = 1 are re-run with seeds 1–2.

**Sensitivity** (reported, never chosen): at each model's chosen `k`, the stop-point 1 grid's other gap settings:
`g_min` 3, 5, 10 × `v_lo` 1, 2, 3 × `v_hi` 15, 25.

**Choice**, per model, on SHL User 1, all positions pooled, seed 0:
- **Objective:** fewest edits per hour.
- **Constraints:**
  - (a) the ExtraSensory guard (below);
  - (b) extra rail legs per hour rise by ≤ 0.05;
  - (c) false rail time on still, walking, car and bus rises by ≤ 0.5 percentage points (pp) each.
- **Tie-break:** within 0.01 edits per hour, the smaller `k`.
- **Off:** if no `k` > 1 beats `k` = 1, the rule is off for that model.

**ExtraSensory guard.** False rail time is the share of labelled non-rail minutes (all classes but train, subway and
tram) inside predicted rail legs, using the leg modes after 11's leg rules, mean of seeds 0–2.
- **Limit:** it may rise by at most **0.02 pp** absolute per model, about 49 of 242,687 minutes.
- **Baseline** (minute argmax after the HMM): E1 0.22%, E2 0.09%.
- **Why 0.02 pp:** read as GPS only, the stop-point 1 grid put 12–111 ExtraSensory minutes in moving gaps (0.005–0.05
  pp). The limit allows less than half of the largest.

<details><summary>Stop-point 1 proposal (superseded by the approval above)</summary>

Grid `g_min` 3, 5, 10 min × `v_lo` 1, 2, 3 m/s × `v_hi` 15, 25 m/s × `k` 2–32 × boosted columns (subway only / all
rail); R1 hint `metro`; tie-break smaller `k`, then subway only, then larger `g_min`.
</details>

### Data

| Use | Data | GPS |
|---|---|---|
| dev, choice | SHL User 1, out of fold (day folds), seeds 0–2. Hips: all days. Hand, torso, bag: the 4phones days only (1,988 minutes each) | 1 Hz hips track thinned to 20 s (as 11), rule C1 for the model; all good thinned fixes for gaps. SHL's GPS is the hips phone's for every position |
| guard | ExtraSensory dev, out of fold | `gps_per_minute` (≤ 1 fix per minute) for both the model (10's per-minute GPS) and the gaps; per session |
| once, after the rule is fixed | SHL Users 2–3 (reserved; all positions) | 20 s, C1 (as 11's `shl23.py`) |
| once, after the rule is fixed | ExtraSensory test users (reserved): false rail time only (guard) | `gps_per_minute`, as dev |
| once, after the rule is fixed | the three WAIIS app days | 11's reader (coarse fixes > 500 m are missing) |

The reserved set is excluded everywhere except the once-only check. The WAIIS days are not looked at before
stop-point 3.

### Metrics

Leg building, IoU ≥ 0.5 matching and edit counting are as in 11 (`tmd.legs.score`), at the tool's top level, plus:
- **Rail legs (subway):** truth rail legs whose fine majority is subway.
  - **Leg recall:** share matched to a predicted rail leg.
  - **Leg precision:** share of predicted rail legs matched to a truth rail leg; reported for all rail and for
    subway truth legs.
  - **Hint accuracy:** share of matched subway legs with hint `metro` (descriptive; the hint is the model's).
- **Extra rail legs per hour:** predicted rail legs that overlap no truth rail leg, per hour of labelled time.
- **False rail time:** per truth class (still, walking, car, bus), the share of its labelled minutes inside predicted
  rail legs.
- **11's metrics:** edits per hour, time-weighted accuracy, wrong unflagged time, leg F1.
- **R1, GPS only:** time-weighted rail precision, subway precision and subway recall of the gap minutes (SHL User 1),
  and the non-rail minutes it touches (ExtraSensory dev).
- **Reporting:** per position, and for all positions pooled.
- **WAIIS (stop-point 3):** 11's lenient vehicle-leg table and paired comparison, with and without the rule.
  - Rail legs separately, the number of the 5 no-data metro legs that get the `underground_rail` hint, and any
    extra rail legs.
  - Run time per day (single thread).

### Code (proposed, before the dev runs)

- **Move 11's dev helpers to `src/tmd/models/e10.py`.** That covers 11's `dev.py` (10's dev frames, saved
  out-of-fold arrays, fold models, fusion + HMM, leg data and scoring) and `gps_candidates.thinned_track`. 12 is
  their second user. 11's scripts then import from there, and an unchanged leg score for one 11 configuration
  checks the move.
- **New in `tmd.gps`:** `gaps(track, by)`, which finds the gaps between good fixes with their speed across, and
  `minute_gaps(minutes, gaps, by)`.
- **New in `tmd.legs`:** a `GapRule(g_min, v_lo, v_hi, k)`, the boost before the HMM, the gap leg inside
  GPS-only stretches, and the `gps_gap_moving` reason.
- **`tmd.suggest`:** the rule is applied in `minute_probs` and `suggest`; `RULES_VERSION` becomes `legs-v2 (12)`.
- **`tmd.legs_adapter`:** the new reason code; the format stays `tmd-legs/0.1` (one optional reason code added).
- **Tests:** gap finding (distance and speed across on synthetic tracks), the boost (k = 1 is the identity), and the
  GPS-only gap leg.

## Results

### Stop-point 1 (2026-11-05): gap statistics

`uv run python gap_stats.py` (log `logs/gap_stats.log`, full output `logs/gap_stats.out`, tables
`results/tables/gap_*.csv`, `guard_baseline_false_rail.csv`). Descriptive, with no MLflow run. The guard baseline
reads 10's saved out-of-fold arrays; the dev runs at stop-point 2 log the baseline as `k` = 1. For now the script
imports 11's `dev.py` read-only, which the code move above replaces.

**Tracks.**
- **SHL User 1:** 1,377,605 fixes at 1 Hz, thinned to 70,259 at 20 s; 23,445 labelled minutes; 1,091 gaps > 60 s.
  SHL has no fix coarser than 400 m, so its gaps are fix-free time only.
- **ExtraSensory dev:** 245,064 fixes ≤ 500 m (none coarser), 250,094 labelled minutes, 42,140 gaps (≥ 1 minute
  without a fix).

**SHL User 1, per class** (labelled minutes; "in gap ≥ 2 min, v ≥ x" = share of the class's minutes inside a gap of
≥ 2 min whose speed across is ≥ x m/s):

| Class | Minutes | Covered | In gap ≥ 2 min | v ≥ 1 | v ≥ 2 | v ≥ 3 | Speed across p10 / p50 / p90 (m/s) |
|---|---|---|---|---|---|---|---|
| subway | 2,421 | 0.33 | 0.69 | 0.59 | 0.52 | 0.45 | 0.6 / 4.8 / 7.6 |
| train | 3,381 | 0.54 | 0.47 | 0.46 | 0.46 | 0.46 | 11.0 / 16.7 / 23.4 |
| car | 4,087 | 0.84 | 0.16 | 0.16 | 0.16 | 0.15 | 3.7 / 17.5 / 26.6 |
| bus | 3,069 | 0.92 | 0.09 | 0.08 | 0.08 | 0.06 | 1.7 / 4.1 / 9.1 |
| walk | 3,101 | 0.87 | 0.14 | 0.03 | 0.03 | 0.02 | 0.02 / 0.06 / 4.5 |
| still | 3,425 | 0.41 | 0.52 | 0.02 | 0.01 | 0.005 | 0.005 / 0.02 / 0.12 |
| bike, run | 2,900, 1,061 | 0.99, 1.00 | 0 | 0 | 0 | 0 | |

**SHL User 1, per gap ≥ 2 min** (by the majority class of its minutes; the median majority share is 1.0, so gaps are
almost always one class):

| Class | Gaps | ≥ 5 min | Length median / p90 (min) | Speed across p50 (m/s) | v ≥ 2 m/s |
|---|---|---|---|---|---|
| subway | 102 | 79 | 13.5 / 44.0 | 5.7 | 88 |
| train | 109 | 58 | 5.6 / 49.8 | 16.7 | 100 |
| car | 28 | 22 | 26.3 / 51.1 | 12.3 | 25 |
| bus | 48 | 12 | 3.2 / 10.7 | 3.4 | 38 |
| still | 110 | 90 | 16.6 / 56.5 | 0.05 | 4 |
| walk | 35 | 20 | 7.6 / 31.8 | 0.12 | 3 |

**Readings.**

1. **Subway is the class GPS loses most.** 69% of subway minutes are in a gap ≥ 2 min, 52% in one with speed
   across ≥ 2 m/s. Speeds across are low (median 4.8 m/s) because the gap includes walking in the stations and the
   first fix after it comes late.
2. **Train also loses GPS:** 46% of minutes, at 11–23 m/s. That is still top-level rail, but the wrong hint for
   metro.
3. **Road vehicles do have moving gaps,** but they don't look like tunnels:
   - **Bus:** 9% of minutes, mostly short (median 3.2 min) at 2–5 m/s, the same speed range as subway.
   - **Car:** 16% of minutes, in long gaps (median 26 min, up to 67 min, up to 84 km) at 12–27 m/s. They look like
     GPS outages in the car rather than tunnels.
   - So speed and length alone don't separate subway from bus. In R2 the accelerometer has to do that.
4. **Still and walking lose GPS indoors without moving.** 52% of still minutes are in gaps, but at ~0 m/s; only
   1.1% of still minutes and 2.5% of walking minutes are in gaps with speed across ≥ 2 m/s. The movement condition
   is what keeps them out.
5. **SHL logs few fixes underground:** only 34% of subway minutes have any 1 Hz fix. The gaps are real GPS loss,
   not an effect of thinning. Accuracy p50 / p90: subway 12 / 32 m, train 16 / 32 m, others 6–8 / 12–24 m.

**ExtraSensory dev** (no subway; train 190 minutes):
- Missing location is common (covered: Android 0.74, iPhone 0.90; the reason for every missing minute is
  `no_updates`). But **moving gaps are rare:** ≥ 2 min with speed across ≥ 3 m/s holds for 0.10% of Android minutes
  and 0.27% of iPhone minutes. This confirms that missing GPS alone must not count.
- Per class, in a gap ≥ 2 min with v ≥ 2 m/s: still 0.2%, walk 0.2%, car 0.9%, bus 1.2%, bike 0.8%.
- 366 still minutes have a speed across ≥ 25 m/s (352 of them ≥ 40 m/s): GPS jumps. Hence the cap `v_hi`.

**R1 read as GPS only** (every minute in a moving gap, as if it had no accelerometer data; full grid in
`gap_r1_gps_only.csv`):

| `g_min` | `v_lo` | `v_hi` | SHL minutes | Rail precision | Metro precision | Metro recall | Bus / car / walk / still minutes | ExtraSensory minutes |
|---|---|---|---|---|---|---|---|---|
| 3 | 1 | 15 | 2,574 | 0.771 | 0.547 | 0.582 | 198 / 256 / 84 / 52 | 111 |
| 5 | 2 | 15 | 2,214 | 0.775 | 0.531 | 0.485 | 148 / 249 / 68 / 34 | 31 |
| 5 | 3 | 15 | 1,941 | 0.791 | 0.519 | 0.416 | 112 / 225 / 53 / 15 | 19 |
| 10 | 3 | 15 | 1,732 | 0.783 | 0.506 | 0.362 | 94 / 221 / 51 / 10 | 12 |
| 5 | 2 | 25 | 3,293 | 0.770 | 0.357 | 0.485 | 148 / 504 / 73 / 34 | 31 |

- Rail precision stays at 0.77–0.79 over the whole grid, and metro precision falls to about 0.35 when `v_hi` is 25
  (car and train come in).
- That is why R1 is proposed as `unknown` + hint `metro`, not as rail (the hint became `underground_rail` at the approval).

**Headroom for R2** (`gap_headroom_shl.csv`; what the models say now, seed 0, top-level argmax after the HMM, on
minutes in gaps ≥ 3 min with 1 ≤ v < 15 m/s):

| Position | Subway minutes (in gap) | E1 rail now: all / in gap | E2 rail now: all / in gap | E1 in gap, other predictions |
|---|---|---|---|---|
| hips | 2,420 (1,407) | 0.957 / 0.950 | 0.967 / 0.974 | bus 32, walking 16 |
| hand | 151 (99) | 0.636 / 0.525 | 0.371 / 0.182 | bus 17, walking 12 |
| torso | 151 (99) | 0.748 / 0.626 | 0.669 / 0.586 | bus 23, walking 8 |
| bag | 151 (99) | 0.967 / 0.970 | 0.881 / 0.909 | walking 3 |

1. **On hips, SHL User 1's subway is already ~96% rail.** The WAIIS failure (metro read as walking or still) doesn't
   show there, and R2 can gain little. SHL User 1 is also in E1's and E2's training (out of fold, by day).
2. **The headroom is hand and torso** (E1: 47 and 37 gap minutes not rail; E2 hand: 81). That is close to the WAIIS
   case (phone not on the hips). But it is small: 4phones days only, 99 gap minutes and a few subway legs per
   position. The dev choice will rest on few legs, and Results will say so.
3. **Risks inside moving gaps on hips:**
   - car 256 minutes (E1 already 5% rail there);
   - bus 198 minutes (0% rail now);
   - walking 84 minutes;
   - still 52 minutes (46% rail for E1, 69% for E2; probably station stops inside subway trips).
   Constraint (c) watches these.

**Guard baseline** (10's saved out-of-fold p, mean of seeds 0–2, minute argmax after the HMM; at stop-point 2 it is
recomputed on leg modes):

| | Labelled minutes | False rail minutes | Share of non-rail | Still | Walk | Car | Bus |
|---|---|---|---|---|---|---|---|
| ExtraSensory dev, E1 | 242,687 | 542 | 0.22% | 0.18% | 0.35% | 0.81% | 2.55% |
| ExtraSensory dev, E2 | 242,687 | 215 | 0.09% | 0.04% | 0.33% | 0.54% | 1.68% |
| SHL User 1 hips, E1 | 23,426 | 81 | 0.46% | 1.98% | 0 | 0.32% | 0 |
| SHL User 1 hips, E2 | 23,426 | 320 | 1.82% | 9.15% | 0 | 0.16% | 0 |

### Open questions (stop-point 1)

- **WAIIS coarse fixes.** WAIIS may log coarse network fixes (> 500 m) underground where SHL logs nothing. Both
  count as missing, so the rule is the same; SHL can't test it.
- **Late first fix.** The first good fix after a subway ride can come minutes after leaving the station. The gap
  then ends late and lowers the speed across (subway p10 0.6 m/s). `v_lo` = 1 keeps most of these; 2–3 drops a
  quarter to a third of the subway gap minutes.

### Stop-point 2 (2026-11-05): code, dev results, proposed `k`

**Code** (approved moves).
- **`src/tmd/models/e10.py`:** 11's `dev.py` helpers and `gps_candidates.thinned_track` (as `shl_thinned_track`).
  11's `dev.py` now keeps its folder constants and re-exports the moved helpers, so 11's scripts run unchanged.
- **Move check** (`check_move.py old|new`, `logs/check_move_{old,new}.json`): run on 11's code before the move and
  on `tmd.models.e10` after it.
  - Inputs: SHL User 1 hips with C1 at 20 s, recomputed end to end, plus ExtraSensory dev; E1 and E2, seed 0.
  - Leg scores (11's approved rules and raw legs): all 50 values identical.
  - The recomputed probabilities differ from 11's saved ones by at most 3e-8.
- **New code:**
  - `tmd.gps.gaps` / `minute_gaps`;
  - `tmd.legs.GapRule`, `boost_rail`, the `in_gap` argument of `predict_legs` (reason `gps_gap_moving` with
    `gap_model_minutes`), and R1 in `gps_only_legs(..., gap_rule)`;
  - `tmd.suggest`: `Spec.gap`, `in_gap`, gaps per day; `RULES_VERSION` `legs-v2 (12)`;
  - `tmd.legs_adapter`: `REASON_CODES`, and `RULE_VERSION` per rule source.
- **Off by default:** with `gap=None` (still the default in `SPECS`) every output is as in 11.
- **Tests:** `tests/test_gaps.py` (7) covers gaps, speed across, group boundaries, minute assignment, the moving
  rule, the boost (k = 1 is the identity) and the R1 gap leg with its reason. The full suite passes (125).
- **R1 detail, not a deviation:** the gap part of a stretch without model minutes becomes a gap leg even when 11
  made no leg for that stretch (fewer than 2 fixes inside: a metro trip with neither accelerometer data nor GPS).
  The parts of the stretch before and after the gap leg get 11's legs if they qualify (≥ 5 min, ≥ 2 fixes).

**Dev run.** `uv run python dev_k.py grid` (log `logs/dev_k.log`, table `results/tables/dev_k.csv`, choice
`dev_k_choice.json`), then `uv run python dev_k.py sensitivity` (`sensitivity.csv`).
- 72 MLflow runs (`dev_k_<model>_k<k>_seed<s>`, `sens_<model>_<rule>`), all clean at commit `bc60bfd`.
- **Re-run after the stop-point 2 approval.** The first run (all rail columns boosted, tram included for E1) was
  dirty. At the approval its 72 runs were deleted from MLflow, the boost was restricted to train and subway, and
  everything was re-run clean with the code committed.
  - Choice and feasible set unchanged: `k` = 2 for both, only `k` 1–2 feasible.
  - Only E1 at `k` ≥ 4 moved: `k` = 4 edits per hour 0.887 → 0.882; `k` = 16 car false rail 4.98% → 4.96%.
  - Everything at `k` ≤ 2, the sensitivity, and all of E2 are unchanged. E1's tram probability is near zero on
    SHL.
  - The tables below are the clean run's.
- Checks before scoring:
  - The other positions and ExtraSensory reproduce 10's saved `p` exactly from 10's saved raw arrays (max |diff| 0;
    one HMM sequence per position).
  - SHL hips at `k` = 1 reproduces 11 (E1 edits per hour 0.8427).
- Moving gaps (approved definition): SHL User 1 has 143 of 1,091 gaps. Hips: 3,463 minutes in them, 1,174 of them
  subway. ExtraSensory dev has 11 of 53,454 gaps (48 minutes).
- Truth: 70 subway legs pooled: 61 on hips and 3 per other position (the 4phones days).

**`k` grid, SHL User 1, all positions pooled, seed 0** (the choice). False rail is the share of the class's minutes
inside rail legs. The ExtraSensory column is the mean of seeds 0–2.

| Model | k | Edits / h | Time accuracy | Subway leg recall | Rail leg precision | Extra rail legs / h | False rail car | False rail bus | ExtraSensory false rail |
|---|---|---|---|---|---|---|---|---|---|
| E1 | 1 | 0.903 | 0.958 | 0.686 | 0.738 | 0.007 | 0.28% | 0 | 0.22% |
| E1 | **2** | **0.876** | 0.962 | 0.771 | 0.801 | 0.009 | 0.28% | 0.17% | 0.22% |
| E1 | 4 | 0.882 | 0.961 | 0.771 | 0.787 | 0.016 | 0.92% ✗ | 0.28% | 0.22% |
| E1 | 8 | 0.887 | 0.959 | 0.757 | 0.772 | 0.023 | 1.49% ✗ | 0.62% ✗ | 0.22% |
| E1 | 16 | 0.932 | 0.950 | 0.757 | 0.742 | 0.045 | 4.96% ✗ | 2.44% ✗ | 0.23% |
| E1 | 32 | 0.937 | 0.941 | 0.757 | 0.729 | 0.052 ✗ | 9.48% ✗ | 4.02% ✗ | 0.23% |
| E2 | 1 | 0.962 | 0.946 | 0.686 | 0.692 | 0.030 | 0.44% | 0.17% | 0.09% |
| E2 | **2** | **0.950** | 0.948 | 0.743 | 0.714 | 0.032 | 0.72% | 0.20% | 0.09% |
| E2 | 4 | 0.923 | 0.949 | 0.814 | 0.758 | 0.034 | 2.01% ✗ | 0.25% | 0.09% |
| E2 | 8 | 0.944 | 0.948 | 0.786 | 0.743 | 0.041 | 2.27% ✗ | 0.53% | 0.09% |
| E2 | 16 | 0.960 | 0.943 | 0.786 | 0.727 | 0.052 | 4.70% ✗ | 1.24% ✗ | 0.09% |
| E2 | 32 | 0.993 | 0.931 | 0.743 | 0.691 | 0.070 ✗ | 9.86% ✗ | 3.65% ✗ | 0.09% |

✗ = breaks a constraint ((b): extra rail legs per hour up > 0.05; (c): false rail on a class up > 0.5 pp).
False rail on still: E1 2.39% → 2.51% at `k` = 2 and 2.90% from `k` = 4 (+0.51 pp, just over the limit); E2 9.70% →
9.83–10.14%. Walking ≤ 0.4% at every `k`.

**Proposed: `k` = 2 for both models.** It is the only `k` > 1 that meets all constraints, and it beats `k` = 1 for
both. The rule is on.

| Seed-0 choice, mean of seeds 0–2 | E1 k = 1 | E1 k = 2 | E2 k = 1 | E2 k = 2 |
|---|---|---|---|---|
| Edits per hour (seeds 0 / 1 / 2) | 0.903 / 0.910 / 0.941 | 0.876 / 0.894 / 0.910 | 0.962 / 0.955 / 0.964 | 0.950 / 0.939 / 0.944 |
| Edits per hour, mean | 0.918 | 0.893 | 0.960 | 0.944 |
| Time accuracy | 0.958 | 0.962 | 0.947 | 0.950 |
| Wrong unflagged time | 4.19% | 3.82% | 5.27% | 4.98% |
| Leg F1 | 0.821 | 0.830 | 0.816 | 0.821 |
| Subway leg recall (70 legs) | 0.710 | 0.776 | 0.695 | 0.752 |
| Subway legs with hint `metro` (matched) | 0.94 | 0.91 | 0.99 | 0.94 |
| Rail leg recall / precision | 0.829 / 0.750 | 0.859 / 0.799 | 0.820 / 0.699 | 0.846 / 0.720 |
| Extra rail legs per hour | 0.008 | 0.009 | 0.028 | 0.032 |
| ExtraSensory false rail (minutes) | 0.22% (533) | 0.22% (533) | 0.09% (206) | 0.09% (206) |

**Per position, seed 0, E1, `k` 1 → 2** (subway legs: hips 61, each other position 3):

| Position | Edits per hour | Subway leg recall | Time accuracy |
|---|---|---|---|
| hips | 0.843 → 0.835 | 0.721 → 0.754 | 0.965 → 0.966 |
| hand | 1.782 → 1.545 | 1 → 2 of 3 | 0.835 → 0.876 |
| torso | 1.193 → 0.954 | 1 → 3 of 3 | 0.932 → 0.969 |
| bag | 1.129 → 1.069 | 2 → 3 of 3 | 0.947 → 0.948 |

E2, `k` 1 → 2: hips 0.896 → 0.894, hand 2.139 → 2.139 (0 → 1 of 3 subway legs), torso 1.312 → 1.074, bag
0.951 → 0.951.

**Sensitivity** (`k` = 2, seed 0; all 18 settings: `g_min` 3, 5, 10 × `v_lo` 1, 2, 3 × `v_hi` 15, 25):
- E1 edits per hour 0.873–0.878 (approved setting 0.876), E2 0.946–0.953 (0.950).
- Subway leg recall is the same in every setting (E1 0.771, E2 0.743).
- Extra rail legs and false rail on car / bus are within ±0.002 / ±0.0007.
- At `k` = 2 the result doesn't depend on the gap definition.

**Readings.**
1. **The gain is real but small, and carried by few legs.**
   - E1, `k` = 2: −0.025 edits per hour (−2.7%, every seed), subway leg recall +0.07 (+3 to +6 of 70 legs by seed).
   - Most of the per-position gain is on torso, hand and bag, but that is 3 subway legs each (the 4phones days).
     Hips gains little (subway legs 44 → 46 of 61), as stop-point 1 expected.
   - E2 gains less (−0.016 edits per hour).
2. **Car limits `k`.** SHL User 1's long car GPS outages (stop-point 1, reading 3) turn into rail from `k` = 4 (E1
   car false rail 0.3% → 0.9%; at 32, 9.5%). Bus follows at higher `k`. Constraint (c) binds; without it, E2 would
   take `k` = 4 (0.923 edits per hour, subway recall 0.81).
3. **ExtraSensory is untouched.** With only 11 moving gaps, false rail time doesn't change at any `k` ≤ 8 (E1 0.22%,
   E2 0.09%; +0.01 pp at 16–32). The guard never binds. Missing GPS in ExtraSensory is not movement across a gap.
4. **Hint.** Among matched subway legs, `metro` drops slightly with the boost (E1 0.94 → 0.91), because boosting all
   rail columns also raises train. The top level is unaffected.
5. **R1 isn't exercised on dev** (SHL has accelerometer data throughout). Its evidence stays the stop-point 1
   GPS-only reading for the approved setting: rail precision 0.770, subway recall 0.485, 31 ExtraSensory minutes.

### Open questions (stop-point 2)

None. Approved 2026-11-05 (see Stop-points).

### Stop-point 3 (2026-11-05): SHL Users 2–3, ExtraSensory test, WAIIS days (once, descriptive)

**Commands** (commit `4dbc4df`, rule on in `tmd.suggest.SPECS` = `GAP_RULE`):
- `uv run python sp3.py shl23|es_test|waiis` (logs `logs/sp3_*.log`, `*.out`);
- `uv run python src/tmd/tracking.py export 12_gps-gap-rail_shl-waiis`, which writes `results/runs.csv`;
- `uv run python tables.py > results/tables_from_runs.md`.

The export has 84 runs, all clean: 72 dev runs at `bc60bfd`, 8 stop-point 3 runs at `4dbc4df`, and 4
WAIIS runs with `k` = 1 at `a5a7fac` (below). Its
`k_per_class` column is `tmd.tracking`'s numeric copy of the param `k`, here the boost factor.

Each part ran with the rule off and on, from the same inputs.
- **Checks:**
  - **Rule off reproduces 11:** SHL Users 2–3 C1 leg metrics equal 11's `shl23_legs.csv`, and the WAIIS legs equal
    11's output files.
  - **10's saved test arrays:** the HMM on them reproduces 10's saved test `p` exactly (ExtraSensory test).
- **WAIIS outputs** (rule on, `tmd-legs/0.1`, reasons `gps_gap_moving`) and the personal tables are under
  `<processed>/12_gps-gap-rail_shl-waiis/waiis/` (not in git).

**SHL Users 2–3** (reserved; all positions; 10's final models, seed 0; C1 at 20 s). There are 10 moving gaps (of 67)
and 285 minutes per position: subway 127, train 69, car 25, bus 22, still 12, walking 6, unlabelled 24. The gaps
are 4 subway, 2 train, 2 bus, 1 car, and 1 still at 20.5 m/s (a GPS jump). The 6 subway legs are the same legs in
each position.

| All positions pooled (off → on) | E1 | E2 |
|---|---|---|
| Edits per hour | 1.163 → 1.194 | 1.150 → 1.181 |
| Time accuracy | 0.901 → 0.900 | 0.883 → 0.889 |
| Leg F1 | 0.819 → 0.814 | 0.812 → 0.813 |
| Subway leg recall (24 = 6 legs × 4 positions) | 0.708 → 0.667 | 0.667 → 0.708 |
| Rail leg precision | 0.732 → 0.739 | 0.746 → 0.766 |
| Extra rail legs per hour | 0.056 → 0.063 | 0.019 → 0.025 |
| False rail: car / bus | 0.7% → 2.3% / 6.5% → 8.5% | 0 → 0 / 3.4% → 7.7% |

- On hips, E1's car false rail goes from 2.6% to 9.4% (the 25-minute car gap). E2's bus false rail goes from 0 to
  8.5%.
- Hand gains for E2 (edits per hour 1.475 → 1.425, subway leg recall 0.50 → 0.67); E1 on hand is unchanged.
- The rule doesn't hold up here: edits per hour rise for both models, by about 2.7%.

**ExtraSensory test users** (reserved, 12 users, 809 h; false rail = share of labelled non-rail minutes inside rail
legs). There are 6 moving gaps (of 10,309), 90 minutes, all Android.

| Off → on | E1 | E2 |
|---|---|---|
| False rail, all | 0.181% → 0.227% (88 → 110 min) | 0.068% → 0.070% (33 → 34 min) |
| — iPhone / Android | 0.061% → 0.061% / 0.325% → 0.424% | 0 → 0 / 0.149% → 0.153% |
| False rail on walking | 0 → 0.53% (Android 0.84%) | 0 → 0 |
| Edits per hour | 1.330 → 1.332 | 1.310 → 1.310 |

- E1 rises by **0.046 pp**, above the 0.02 pp limit fixed for dev; E2 rises by 0.002 pp. This is descriptive, so
  no choice is made on it (approved plan). The rise is 22 minutes, all on Android, where walking false rail goes
  from 0 to 0.84%.

**WAIIS days** (11's lenient comparison; 19 diary vehicle legs, 14 of them rail: 13 metro, 1 tram). Moving gaps per
day: 2, 7 and 2 (of 22, 39 and 24 gaps). They hold only 11, 8 and 3 model minutes.

| Off → on | E1 | E2 |
|---|---|---|
| Vehicle legs right / wrong / no data | 8 / 6 / 5 → 8 / 6 / 5 | 6 / 8 / 5 → 7 / 7 / 5 |
| Vehicle core-minute accuracy | 0.807 → 0.807 | 0.699 → 0.747 |
| Rail legs right (of 14) | 3 → 3 | 1 → 2 |
| No-data metro legs with an `underground_rail` gap leg | 0 → 4 of 5 (mean share of the leg covered 0.63) | same (R1 doesn't depend on the model) |
| Rail legs touched by a gap leg | 0 → 7 | 0 → 7 |
| Extra vehicle / rail / gap legs | 0 / 0 / 0 → 0 / 0 / 0 | 0 / 0 / 0 → 0 / 0 / 0 |
| Run time per day, single thread (mean / max) | 0.171 / 0.198 s → 0.173 / 0.200 s | 0.167 / 0.196 s → 0.168 / 0.195 s |

- **R1:** 4 of the 5 metro legs without accelerometer data now show a gap leg (`unknown`, hint `underground_rail`)
  covering 45–100% of the diary leg. The fifth (2026-10-31 D3) has no good fix before or after it inside the day's
  track that makes a moving gap (longest gap 15 min at 1.9 m/s, below `v_lo`). No gap leg falls outside a diary rail
  leg (±10 min).
- **R2:** one leg changes, E2's 2026-11-03 D1 metro, from car to rail. Its hint is `train`, because train is boosted
  too. E1 doesn't change.
- **The 5 metro legs read as walking or still** (with accelerometer data) stay wrong for both models. GPS during the
  diary rail legs (counts only; personal detail not kept):
  - Three are short rides (4–7 min: 2026-10-30 D5, 2026-10-31 D9, 2026-11-03 D4) whose gaps are 3.0–4.6 min, under
    `g_min`.
  - 2026-10-31 D5: none of its model minutes falls inside the moving gap.
  - 2026-10-31 D7: 6 of its 9 model minutes are in a moving gap. E1's raw fused rail probability there is 0.2–0.9,
    but the HMM decodes still: rail is 0.01 off and 0.19 at `k` = 2. E2's raw rail is ≤ 0.11 there, and nothing
    changes.
  - **Coarse fixes:** WAIIS logs coarse fixes (> 500 m) underground in 8 of the 13 metro legs (open question of
    stop-point 1). They count as missing, so the rule saw them as gaps as planned.
- These days are not unseen (11). No choice was made on them.

**R2 switched off after the once-check (2026-11-05).**
- **Why:** `k` = 2 was chosen on dev, then failed on SHL Users 2–3 (both models, edits per hour +2.7%) and on the
  ExtraSensory test guard (E1, +0.046 pp against the 0.02 pp limit).
- **Change:** `tmd.suggest.GAP_RULE` now has `k` = 1 (commit `cef3b91`). R1 stays on: the `gps_gap_moving` gap legs
  with hint `underground_rail`, and the reason on model legs holding gap minutes. Switching off returns to the
  untuned setting; no other `k` is chosen from these results.
- **WAIIS re-run** (`sp3.py waiis` at `a5a7fac`; 4 runs, `gap_rule` `gap5min_v2-25_k1`; the `k` = 2 runs stay in
  the export as evidence; output `logs/sp3_waiis.out`, the `k` = 2 output kept as `logs/sp3_waiis_k2.out`):
  - The vehicle-leg table with the rule on equals the rule off and 11's lenient statuses, for every leg and both
    models: E1 8 / 6 / 5, E2 6 / 8 / 5 (right / wrong / no data).
  - The same 4 of 5 no-data metro legs get an `underground_rail` gap leg (mean share of the leg covered 0.63).
  - 7 rail legs are touched by a gap leg; there are no extra vehicle, rail or gap legs.
  - Run time 0.15–0.16 s per day.
- **Export:** 84 runs, all clean.

### Deviations

1. **R1 also covers stretches 11 left without a leg** (stop-point 2): a gap leg is made even when the stretch had
   fewer than 2 fixes (`tmd/legs.py`, `gps_only_legs`). This follows the approved definition, but 11's stretch rule
   alone would have dropped it.
2. **The other positions' C1 GPS is the hips track** (SHL logs GPS on the hips phone), as in 10 and 11.
3. **`sp3.py` imports 11's `lenient.py`, `review.py` and `shl23.py`** (read-only) for the lenient core, the diary
   reader and the SHL Users 2–3 frames. They are 11's once-only helpers, so they were not moved to `src/tmd`.
4. **R2 (`k` = 2) switched off after the once-check** (`src/tmd/suggest.py`, `GAP_RULE`). The Plan chose `k` on
   dev and fixed it for the once-check. `k` = 2 failed there (SHL Users 2–3, both models; ExtraSensory test guard,
   E1) and was set back to 1, the untuned setting. No other `k` is chosen from these results. R1 is unchanged.

### Open questions (stop-point 3)

- **R2:** switched off (`k` = 1) after the once-check; see above.
- **HMM vs rail evidence on WAIIS metro** (2026-10-31 D7): raw rail evidence is overridden by the HMM's still
  persistence. Raising it inside gaps far enough (`k` ≥ 4) costs car false rail on SHL. A gap-aware transition
  (leaving still is easier inside a moving gap) would be a different rule, not tested here.

## Conclusion

**Draft — not approved.**

**Question.** Can GPS loss while moving mark underground rail, so the suggestion engine proposes metro where it
showed unknown GPS-only legs or read metro minutes as walking or still?

**Answer: yes for the first, barely for the second.**

- **Where there's no accelerometer data (R1), the gap is a useful hint.** On the WAIIS days, 4 of the 5 metro legs
  that 11 couldn't suggest at all now show an `unknown` leg with the hint `underground_rail` and the reason
  `gps_gap_moving` (gap minutes, speed across, the two fix times). There are no false hints. The leg stays
  `unknown` because, read as GPS only, the gaps' rail precision on SHL User 1 is 0.77 and metro precision about 0.5.
  The movement condition matters: missing GPS alone is common and mostly an Android effect in ExtraSensory, where
  moving gaps are 0.1–0.3% of minutes.
- **Where there is accelerometer data (R2), the gain is small and doesn't carry over.**
  - **Dev (SHL User 1, all positions):** `k` = 2 cut edits per hour by 2.7% for E1 and 1.6% for E2, with subway leg
    recall up 0.06–0.07. Most of the gain came from 9 subway legs off the hips. Larger `k` turns SHL's long car GPS
    outages and bus gaps into rail.
  - **SHL Users 2–3:** edits per hour went up for both models (E1 1.163 → 1.194, E2 1.150 → 1.181), through car and
    bus false rail.
  - **ExtraSensory test:** E1's false rail rose by 0.046 pp, above the 0.02 pp limit set for dev (22 minutes,
    Android, walking). E2's rose by 0.002 pp.
  - **WAIIS days:** one metro leg changed (E2), and E1 is unchanged. The metro minutes read as walking or still are
    mostly short rides with gaps under 5 min, or minutes where the HMM keeps still against the rail evidence.
- **Road vehicles do lose GPS while moving.** SHL car outages last up to an hour and bus gaps are short, so gap length
  and speed don't separate them from metro. Train loses GPS as often as metro (46% of its minutes), which is why the
  hint is `underground_rail`, not `metro`.
- **Cost:** about 0.17 s per WAIIS day, unchanged.

**In the engine** (`tmd.suggest`, rules version `legs-v2 (12)`): **R1 only.**
- Both models get the R1 gap legs (`unknown`, hint `underground_rail`, reason `gps_gap_moving`) for moving gaps of
  ≥ 5 min at 2–25 m/s. The output format stays `tmd-legs/0.1` with the new reason code.
- **R2 was switched off after the once-check.** `k` = 2 was chosen on dev, then failed on SHL Users 2–3 (both
  models) and on the ExtraSensory test guard (E1). Switching off returns to the untuned setting (`k` = 1), and no
  other `k` is chosen from these results. This is recorded as deviation 4.
- With R1 only, the WAIIS vehicle-leg suggestions equal 11's, plus the 4 hinted gap legs.

**Limits.**
- The decisive evidence is a few legs: 9 subway legs off the hips on dev, 6 on SHL Users 2–3, and 13 metro legs on
  three WAIIS days of one person, one iPhone.
- SHL logs no coarse fixes, so the WAIIS case of coarse fixes underground is untested on dev.
- R1 can't be scored on dev (SHL always has accelerometer data).

**Next.**
1. **Follow-up for the retraining with labelled WAIIS days:** WAIIS metro minutes with raw fused rail around 0.4
   are decoded as still by the HMM (2026-10-31 D7, E1). The retrained model and its HMM should be checked on such
   minutes.
2. Fixing metro read as walking or still is a model question, not a gap rule. It belongs to the next model
   experiment (E1 + Collecty, then labelled WAIIS days), with metro minutes from WAIIS phones off the hips.
