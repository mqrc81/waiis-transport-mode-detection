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
2. **Dev results** and the proposed `k` per model. Wait for OK.
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
  all rail columns (train and subway, plus tram for E1) are multiplied by `k`, then renormalised.
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
