# 11 Leg suggestions on the WAIIS app days

## Question

What leg suggestions (start, end, mode) do E1 and E2 give on the WAIIS app days, how good are they on dev, and how
should WAIIS GPS feed them? The result is the first suggestion engine for the WAIIS labelling tool, adapted to the
tool's format once its repo is available.

## Plan

Models: 10's **E2** (final) and **E1** (descriptive in 10; has a tram class), as saved; no retraining. Minute
probabilities → HMM (forward-backward, as in 10) → legs. Dev data: SHL User 1 (hips, day folds, out of fold) and
ExtraSensory dev (out of fold); ExtraSensory's label timing is coarse, so its boundaries are reported, never tuned on.
WAIIS days are review days, not a test set.

**Stop-point 1 approved 2026-11-04:** all proposals as written, with four changes (platform confirmed; the WAIIS app
days described here as one dataset; the speed across in the reason of moving GPS-only legs; the device-selection
note as what the manifest records). "Proposed" below marks the approved proposals.

### Stop-points

1. **Inventory and plan** (this section and Results, stop-point 1). Wait for OK.
2. **Dev results:** GPS input rule, leg rules and thresholds, both models. Wait for OK and for the diaries before any
   model runs on the WAIIS days.
3. **WAIIS days:** E1 and E2 on the three days, review sheets; SHL Users 2–3 once (leg metrics, descriptive). Wait for
   the review. Then Results, draft Conclusion, STATUS.md.

### Models as saved (experiment 10, `<data>/processed/10_gps-acc-baseline_waiis-bench-v2/models/`)

| Use | E2 | E1 |
|---|---|---|
| dev, out of fold | acc `E2_seed<k>_fold<f>.txt`, GPS `A0_gps_seed<k>_fold<f>.txt` (E2's GPS model is A0's), HMM refit per fold by `tmd.models.minutes.hmm_params` (deterministic) | acc `E1_seed<k>_fold<f>.txt`, same GPS models, HMM from ExtraSensory + SHL User 1 (9 states) |
| saved out-of-fold probabilities | `oof/E2_seed<k>.npz` (acc, gps, raw, p; 360,313 rows) | `oof/E1_seed<k>.npz` |
| WAIIS days, SHL Users 2–3 | `test_E2_seed0.txt`, `test_gps_seed0.txt`, `test_E2_seed0_hmm.npz` | `test_E1_seed0.txt`, same GPS, `test_E1_seed0_hmm.npz` |

- All fold models were saved, so **no refit is needed**. Still, before any dev run, the loaded fold models are checked
  against 10's saved out-of-fold probabilities (acc and GPS, seed 0, SHL User 1 and ExtraSensory rows; expected
  bit-identical or within float tolerance).
- **Seed** (proposed, approved): seed 0 is *the* model (one model id per output file); dev choices are made on seed 0, and seeds
  1–2 are reported as spread. (An average of the 3 seeds would be a new model.)
- Fusion and smoothing exactly as 10: product fusion on GPS-covered minutes, acc model elsewhere; E1's GPS tram factor
  = vehicle mean (`tmd.models.fusion.product`); HMM forward-backward, 60 s step, restart after 10 min
  (`tmd.models.minutes.smooth`).

### WAIIS input

**Reader** `tmd.readers.waiis` (new): one day folder (`<data>/raw/waiis/user=<key>/date=<YYYY-MM-DD>/`) → accelerometer
(t [s], xyz in the Android convention, m/s²), GPS fixes, sessions, minute windows.

- **Platform:** not recorded anywhere in the data (Results). The reader takes it from the project-level device table
  `metadata/waiis_devices.csv` (in git; layout in `metadata/README.md`): one row per user key and device period
  (`valid_from` / `valid_to`), with platform, phone model and the axis convention kept apart from its evidence. It
  fails when a recorded day has no row or several. Current row: `sub_0c0b…`, iphone, iPhone 16, axis convention
  `ios` (strongly indicated, not confirmed by a face-up recording), confirmed 2026-11-04. When the app stores the
  platform, the reader takes it from the data.
- **Axis signs:** v2's rule A1 (iOS → Android convention). WAIIS iPhone data are already in m/s², and iOS's signs
  kept is strongly indicated given the iPhone, not confirmed by a face-up recording (Results), so the rule is `× −1`
  (`tmd.readers.waiis.axis_sign`, from the device table's `axis_convention`). E1's and E2's features are invariant
  to it. `tests/test_waiis_sign_invariance.py` checks that end to end on a WAIIS day (minute windows flip exactly,
  identical 580 features, identical E1 / E2 acc-model probabilities); `tests/test_acc_features.py` already covers the
  feature functions on synthetic windows.
- **Resampling and windows:** as `waiis_bench_v2`: the whole day's samples go through `tmd.candidates.trip_windows`
  (distinct times, gap-free stretches resampled 100 Hz → 25 Hz, one 15 s window per clock minute, real rate ≥ 24 Hz, no
  gap > 0.5 s); only status `ok` windows become model minutes. Features: `tmd.features.acc.features(w, sub=True,
  grav=False)` (05's 580).
- **Bursts → sessions** (proposed, approved): bursts whose pause is **≤ 120 s** join into one session; a longer pause starts a
  new one. The count is the same for any pause from 10 s to 120 s on all three days (pauses are either 1–10 s or many
  minutes; Results), and 120 s is `tmd.gps.SESSION_GAP_S`. Within a session the HMM bridges missing minutes with Aᵏ (≤
  10 min, as 10). GPS fixes belong to the session whose span ± 150 s (gps-v2's half window) contains them.

**GPS → E2's GPS input.** What gps-v2 actually uses (`tmd.features.gps`): only `time`, `lat`, `lon` per fix, grouped by
(user, session), with each fix assigned to a minute `ts`. From those it derives pair speeds (consecutive fixes ≤ 150 s
apart), acceleration, stop count (1.0 m/s threshold), turn angles (bearings over displacements ≥ 20 m), over the fixes
in [ts − 150, ts + 150] s. **It never reads reported speed, heading, accuracy or fix counts.** So:
- WAIIS's missing reported speed (29%) doesn't matter to the model. Reported speed is used only on the review sheets,
  next to derived speed.
- Accuracy matters only in two places: fixes with accuracy > 500 m are dropped (missing) before every rule, and it is
  the ranking key for the best-accuracy rules. **14.25 m** (a third of WAIIS fixes, probably a default) is kept as a
  value; ties are broken by distance to the minute centre. SHL has no such default, so the dev choice can't test this
  quirk. On the WAIIS days, how often a 14.25 m fix is picked over a fix with a real value is reported (descriptive).

Candidates (one fix, or one position, per minute `ts`, minute centre `ts + 30`):

| | Rule | Fix time used |
|---|---|---|
| C1 | fix nearest the minute centre, within the minute | its own |
| C2 | best-accuracy fix within ±7.5 s of the centre (SHL's training rule, `tmd.bench.build.shl_tracks`) | its own |
| C3 | best-accuracy fix within the minute | its own |
| C4 | median lat and median lon of the minute's fixes | median of their times |
| C5 | every fix as it comes, each assigned to its clock minute (comparison only; expected worse) | its own |
| ref | 10's per-minute GPS from SHL's 1 Hz track (what the model was trained on) | upper reference |
| ref | no GPS (acc model + HMM) | lower reference |

- **Dev set for the choice:** SHL User 1's 1 Hz track (`tmd.bench.shl_gps_1hz`) thinned per recording to a fixed 20 s
  cadence (`tmd.gps.thin` with a constant 20 s interval: first fix, then each fix ≥ 20 s after the last kept; natural
  dropouts stay gaps), sessions from the bench labels. Features are gps-v2. E2's GPS model is used out of fold (A0's GPS
  fold models), fused with E2's saved out-of-fold acc probabilities, HMM per fold.
- **Choice:** the highest **six-class macro-F1 on SHL User 1 (official minutes) after fusion and HMM**, seed 0, with
  seeds 1–2 as spread. Within 0.005 of the best, the candidate with more GPS-covered minutes wins, then the simpler one
  (C1 < C3 < C2 < C4). Also reported: GPS-model-alone six-class F1 on covered minutes, coverage, and raw-HMM edits per
  hour. ExtraSensory has no dense GPS and keeps its own per-minute GPS.

**Optional boundary refinement with the 20 s fixes (leg rules only, never model input).** A boundary between a slow
leg (still, walking) and a fast one (bike, car, bus, rail) is moved, within ±1 min of the minute boundary, to the 20 s
fix where derived speed crosses halfway between the two legs' median speeds. **Margin, fixed now:** adopted only if, on
the same 20 s SHL dev with SHL's per-sample labels (`stream.parquet`) as truth at 1 s resolution, it lowers the
median boundary error of matched legs by **≥ 0.5 min** and edits per hour by **≥ 5%**, with constraints (a) and (b) still met.
Otherwise boundaries stay on the minute grid.

**Stretches without accelerometer data** (the app records only while moving). Rule as given: no data, no leg.
- **GPS only, speed across < 0.5 m/s** (proposed, approved): **emit a `still` leg, coverage "GPS only", flagged**, model id
  `rule:gps_only_still`. Conditions: no accelerometer minute in the stretch; fixes with accuracy ≤ 500 m at both ends;
  displacement / time across the stretch < 0.5 m/s; length ≥ 5 min. The leg ends at the neighbouring legs (or at the
  last / first fix). Not checked on dev (SHL and ExtraSensory have no such stretches); the diaries are its first check,
  which is why it is flagged. Overnight stretches are clipped to the UTC day of the file.
- **GPS only, moving (≥ 0.5 m/s across), not in the brief** (proposed, approved). On the three days, 16 / 35 / 19
  minutes have a fix but no accelerometer: mostly isolated fixes, plus the rail-like legs with only coarse fixes.
  Each such stretch becomes **an `unknown` leg, coverage "GPS only", with no mode and no top-2**, so the timeline
  shows that a trip happened. **Its flag reason carries the GPS speed across the stretch** (displacement between its
  end fixes / time, m/s), e.g. `{"code": "gps_only_moving", "speed_across_mps": 7.9}`. The GPS model alone is never
  used for a suggestion (10 never evaluated it alone).

### Labels (the tool's tree, from memory; one mapping table in the adapter)

- Top level, one per leg: still, walking, running, bike, e_scooter, motorcycle, car, bus, rail, unknown.
- Mapping: still → still, walk → walking, run → running, bike → bike, car → car, bus → bus, train / subway / tram →
  rail. **Top-level minute probability = sum of the fine probabilities mapped to it.** Legs are built on the top level,
  so a train / subway flip doesn't split a leg; walk / run does.
- **Sub-mode hint** (rail only): train / metro (= subway) / tram, the largest of them, with its share of the rail
  probability over the leg. E2 has no tram, so its hint is train or metro; E1's can be tram.
- **Blind spots (never predicted):** walking purposes, car driver / passenger, train types, e_scooter, motorcycle,
  phone position.
  - e_scooter: no training data. Most likely **bike**: road vibration without a pedalling rhythm and bike-like speeds.
    car is the second candidate.
  - motorcycle: ExtraSensory has a `motorbike` flag. Stop-point 2 reports E2's predictions on those dev minutes, if
    any; expected **car**.
- **unknown:** a leg whose confidence is below the threshold gets mode = unknown, with its top-2 modes still listed,
  so a one-click accept never stores a guess.

### Legs and leg rules

**Raw HMM legs** (the reference for constraint b): per minute the top-level argmax of the forward-backward posterior;
a leg = a run of one mode within a session, broken by a gap > 10 min without minutes. No other rule, no unknown.

**Rules, applied in this order** (one set per model; dev only):
1. **Stops inside vehicle legs:** a still run of ≤ `S_veh` min between two legs of the same vehicle mode (car, bus,
   rail) is absorbed into it. Grid `S_veh` ∈ {0, 2, 5, 10}.
2. **How stays split legs:** a still run of ≤ `S_other` min between two legs of the same non-vehicle mode (walking,
   running, bike) is absorbed; longer still runs are stays and split. Grid `S_other` ∈ {0, 1, 2, 3}. A data gap
   (no minutes) between two legs of one mode joins them if ≤ `G` min, else splits them. Grid `G` ∈ {2, 10}.
3. **Minimum leg length:** legs shorter than `L_min` min are merged, shortest first, into the neighbour whose mode has
   the higher mean probability over the short leg's minutes. If both neighbours have the same mode, all three join.
   A short leg without a neighbour (gap on both sides) stays and is flagged `short`. Grid `L_min` ∈ {1, 2, 3, 5}.
4. **Confidence** of a leg = the mean over its minutes of its mode's top-level probability. Below `c` → mode unknown,
   flag reason `low_confidence`. Grid `c` ∈ {off, 0.5, 0.6, 0.7, 0.8, 0.9}.
5. **Second mode shown** when its leg probability ≥ `τ2`, `τ2` ∈ {0.1, 0.2, 0.3}. Chosen after rules 1–4: the highest
   `τ2` that shows the true mode for ≥ 50% of SHL legs whose top-1 is wrong. The output always lists top-2; `τ2` only
   sets the "show" flag.

Other flag reasons (informational, they don't set unknown): `short` (< 3 min), `no_gps` on a car / bus / rail leg (road
vs rail from the accelerometer alone is weak), `rail_hint_uncertain` (hint share < 0.6), `gps_only_still` (GPS-only
still legs). GPS-only moving legs are unknown by construction, with reason `gps_only_moving` and the speed across.

**Grid:** 4 × 4 × 2 × 4 × 6 = 768 combinations per model, then `τ2`. **Objective:** fewest edits per hour on SHL User
1 (seed 0), subject to
- (a) time with a wrong, unflagged mode ≤ 5% (SHL User 1);
- (b) ExtraSensory leg F1 and time-weighted accuracy within 0.01 of the raw HMM legs. **Reading (proposed, approved):** (b) scores
  the top-1 mode of every leg, including unknown legs. It checks that merging and minimum length don't damage
  ExtraSensory. The threshold `c` only flags and is governed by (a) and edits. Scoring unknown as wrong in (b) would
  allow almost no threshold. Both versions are reported.

Ties within 1% of the fewest edits go to the simpler combination: smaller `S_veh`, `S_other`, `L_min`, `G`, in that
order, then the higher `c` (more flagging). If no combination meets (a), that is the stop-point 2 result; nothing is
relaxed without OK.

### Metric definitions (dev; legs on the top level; minutes = 1-min model steps)

- **Truth legs:** labelled minutes (`tmd.models.minutes.labelled_minutes(fine=True)`; SHL mixed minutes by majority)
  mapped to the top level, cut into legs by the same run and gap rules as the raw HMM legs. Unlabelled minutes are
  never scored. Predicted legs may extend over them.
- **Matched legs:** same top-level mode and **IoU ≥ 0.5** over labelled minutes, one-to-one, greedy by IoU (segmental
  F1@0.5). Leg precision, recall, F1. Unknown legs never match.
- **Boundary error** of matched legs: |start difference| and |end difference| in minutes, median and p90, excluding
  ends where the truth leg meets a session edge or a data gap. SHL also in seconds against `stream.parquet` labels
  for the refinement check. ExtraSensory: reported, never tuned on.
- **Extra / missed legs per hour:** unmatched predicted / unmatched truth legs per hour of labelled time.
- **Time-weighted accuracy:** share of labelled minutes whose leg mode equals the truth mode. Reported with unknown
  counting as wrong and on top-1.
- **Edits** (per hour and per day; SHL day = recording day, ExtraSensory day = user-day). Per session, each predicted leg
  is mapped to the truth leg it overlaps most, and each truth leg to the predicted leg it overlaps most:
  - merges = Σ over truth legs of (predicted legs mapped to it − 1);
  - splits = Σ over predicted legs of (truth legs mapped to it − 1);
  - mode changes = truth legs whose mapped predicted leg has another mode (unknown included);
  - boundary moves = internal truth boundaries between two truth legs mapped to different predicted legs whose
    predicted boundary is off by > 1 min (a split places its own boundary, so no extra move).
- **Wrong but unflagged:** minutes in non-unknown legs whose mode ≠ truth (share of labelled time); legs whose
  majority truth mode ≠ their mode; share of legs flagged (unknown).
- Reported for **short truth legs (< 5 min)** and **per mode** (recall by truth mode, precision by predicted mode,
  boundary errors).
- **Range mode:** each truth leg as a selected range; the mean of its minutes' top-level probabilities; top-1 and top-2
  accuracy; calibration of the top-1 probability (reliability table, 10 bins, ECE; Brier score).
- **SHL Users 2–3 (stop-point 3, once):** 10's saved test probabilities (`oof/test_E{1,2}_seed0.npz`, per-minute GPS) and the chosen 20 s
  rule (test GPS model on SHL 2–3's thinned track), all positions, leg metrics with the fixed rules.

**Run time:** per WAIIS day, the wall time of reading, windows, features, both models, HMM and legs, single thread,
with minutes of data. Inference latency and model size per CLAUDE.md come from 10 (the models are unchanged) and are
repeated only for the WAIIS path.

### Output (provisional; `tmd.legs_adapter` is the only module that knows the format)

One JSON file per model per day: `<data>/processed/11_leg-suggestions_waiis-app-days/waiis/user=<key>/date=<day>/legs_<model>.json`
(personal data, never in git).

```
{"format": "tmd-legs/0.1", "user_key": ..., "date": "2026-11-03", "time": "UTC epoch ms",
 "model": {"id": "E2", "version": "10:test_E2_seed0", "rules": "legs-v1 (11)", "code_commit": "..."},
 "input": {"platform": "iphone", "platform_source": "self-reported", "gps_rule": "C?", "acc_rate_hz": 25},
 "legs": [{"start_ms", "end_ms", "mode", "sub_mode_hint": {"mode": "metro", "share": 0.71} | null,
           "top2": [{"mode", "p"}, {"mode", "p"}], "show_second": bool, "confidence",
           "low_confidence": bool, "flag_reasons": [{"code", ...details, e.g. "speed_across_mps"}], "coverage": "acc+gps" | "acc" | "gps",
           "n_minutes", "model_id", "model_version"}],
 "minutes": [{"t_ms", "coverage", "p": {top-level mode: prob}, "p_fine": {class: prob}}]}
```

- `tmd.legs` (new): `predict_legs(minutes, probs, rules)` and the leg metrics (reusable for MobilityNet and WAIIS
  labels later). `tmd.readers.waiis` (new): the reader. Scripts in this folder: `inventory.py`, `app_days/`, then
  `check_models.py`, `gps_candidates.py`, `leg_grid.py`, `waiis_days.py`, `review_sheet.py` (names provisional).
- MLflow (experiment `11_leg-suggestions_waiis-app-days`): one run per GPS candidate and seed, one per leg-rule grid
  and model (the full grid as an artifact table), one per WAIIS day and model, one for SHL Users 2–3. Datasets:
  `shl_user1_dev`, `extrasensory_dev_v2`, `shl_user23_test_v3`. The WAIIS days get a new `tracking.SPLITS` entry backed
  by `splits/waiis_days.json` in this folder (user key and dates only).

### WAIIS days (stop-point 3)

- Before any model runs on them: a rough diary per day (approximate times and modes, no places), written by the person
  who recorded the days.
- Review sheet per day (HTML or PNG plus a CSV, under `<data>/processed/11_.../waiis/`): a timeline of E1's and E2's legs against the
  diary, with confidence, coverage and GPS speed (derived and reported; no map); counts of legs that match the diary
  (same mode, IoU ≥ 0.5 with the diary leg), missed, or extra. Each leg is then reviewed as right / wrong / unsure.
- **Tram legs in the diaries** (added at the OK of stop-point 2): for each diary tram leg, per model, the top-level
  mode, the sub-mode hint, the minute shares of tram / train / subway / bus / car (argmax fine class per minute) and
  the GPS coverage of its minutes. E2 has no tram class: a tram leg is right at the top level only if it becomes rail.
  E1's first tram check with GPS. Descriptive; no model choice is made from it.
- Descriptive: the package's own predictions (`predictions/*.parquet`: `fusion_late__hmm_viterbi`, `lightgbm_gps`,
  `inceptiontime_acc(_25hz)`, `hart_acc`, `rules_speed`, 18 s rows) mapped to the top level, against the diary, beside E1
  and E2.
- No rule or threshold changes because of these days without a re-check on dev, recorded as a deviation.

### Run-time estimate

| Step | Compute (estimate) |
|---|---|
| model check against 10's saved probabilities | ~5 min |
| GPS candidates: 5 rules × SHL User 1 (~37k minutes) gps-v2 features, 3 seeds × 5 folds, fusion, HMM | ~30–45 min |
| leg-rule grid: 768 combinations × 2 models on SHL User 1 + ExtraSensory dev (291k minutes), seed 0; seeds 1–2 for the chosen rules | ~1 h |
| boundary refinement (optional), range mode, per-mode tables | ~15 min |
| WAIIS days: 3 days × 2 models (2–3 h of accelerometer per day) | < 1 min per day |
| SHL Users 2–3 once | ~15 min |

Long steps run in the background with `tmd.progress`, and the log path is given at launch.

## Results

### Stop-point 1: the WAIIS app days (2026-11-04; approved 2026-11-04)

Three unlabelled days of WAIIS app recordings, **one dataset**: one person, one phone, UTC days 2026-10-30, 10-31 and
11-03. Descriptive; no models, no MLflow runs. Figures are pooled over the three days unless a day is named; per-day
figures are in the last table.

**Commands** (outputs in `logs/`; summaries and local times of gaps, never coordinates):
```
cd experiments/11_leg-suggestions_waiis-app-days/app_days     # data path as optional argument
uv run python {describe,gaps,imu_timing,noise_floor}.py              > ../logs/app_days_<script>.log
# per day: describe, gaps, imu_timing on a temp folder holding one symlinked day  > ../logs/app_days_<script>_<day>.log
cd .. && uv run python inventory.py                                  > logs/inventory.log
```
The four `app_days/` scripts were moved here from 08's WAIIS day scripts before stop-point 2; reading, fix states
and sessions now come from `tmd.readers.waiis`. Rerun on the committed code, their output equals the first run's except
one gap distance on 10-31 (2,438.8 → 2,438.9 m: `tmd.gps.haversine_m` uses the mean Earth radius 6,371,008.8 m, the
old copy 6,371,000 m); no count changes.
Definitions (the scripts' own): a fix's **state** comes from its derived speed over a 60 s baseline (displacement to
the first fix ≥ 60 s ahead, else behind, at most 180 s away): still < 0.5 m/s, walk 0.5–2.5, 2.5–4, vehicle > 4,
unknown without a baseline. **Still 5 s blocks:** 125 samples inside one burst with std |acc| < 0.05 m/s². **Minute
windows** as `waiis_bench_v2` (one 15 s window per clock minute, 25 Hz, `tmd.candidates.trip_windows`); **quiet:**
std |acc| < 0.03 m/s².

#### Files and fields

- **Layout** per UTC day, `<data>/raw/waiis/user=<key>/date=<YYYY-MM-DD>/`: `gps.parquet`, `imu.parquet`
  (`t_ms, ax, ay, az, gap`), `_manifest.json` (build provenance: source objects, hashes, `nominal_hz` /
  `effective_hz` 25), `segments/h3r10_dwell_sandwich-1-1.parquet` (stays / trips), `context/osm*.parquet`
  (OpenStreetMap context per fix and segment), `predictions/*.parquet` (the package's own models, 18 s rows:
  `fusion_late__hmm_viterbi`, `lightgbm_gps__hmm_viterbi`, `inceptiontime_acc__hmm_viterbi`,
  `inceptiontime_acc_25hz__hmm_viterbi`, `hart_acc__hmm_viterbi`, `rules_speed__majority_window`; classes still /
  walking / running / bike / car / bus / subway / train; plus `point_quality_rules_v1`). One user key; no README.
- **GPS columns:** `t_ms`, `lat`, `lon`, `accuracy_m`, `speed_mps`, `speed_accuracy_mps`, `heading_deg`,
  `altitude_m`, `source`, `activity`, `activity_confidence`, `is_moving`, `event`. Always empty:
  `speed_accuracy_mps`, `activity_confidence`, `event`. Heading missing for 31%. Units m, m/s, degrees.
- **Time:** UTC epoch ms, GPS always whole seconds, sorted, no duplicates. Local time Europe/Madrid, UTC+1 on these
  dates. Folders are UTC days.
- **The phone's own activity flag is useless:** 95% of fixes say `in_vehicle` with `is_moving = true`; the median
  reported speed is 1.0 m/s.
- **Platform and phone model:** iPhone 16, confirmed 2026-11-04 (`metadata/waiis_devices.csv`). **Not recorded in the data:**
  not in the manifest, the parquet columns or the parquet key-value metadata; GPS `source` is always `waiis`.
- **Device selection:** the manifest records `selected_device_rule = greatest_data_object_bytes`. With one user and
  one phone, what happens with a second device is not observable here; open question for the data build.

#### GPS (1,243 fixes)

- **Cadence: a fixed ~20 s while recording.** 80% of intervals are 20–30 s, 5.6% under 20 s; median 20 s in every
  state.

  | State (fixes) | p50 | p75 | p95 | share ≤ 20 s |
  |---|---|---|---|---|
  | still (474) | 20 s | 24 s | 79 s | 0.62 |
  | walk (475) | 20 | 20 | 31 | 0.86 |
  | 2.5–4 m/s (80) | 20 | 20 | 47 | 0.86 |
  | vehicle (157) | 20 | 24 | 102 | 0.68 |
  | unknown (57) | 588 | 1,013 | 8,167 | 0.04 |

- **Gaps over 60 s: 103**, split by the speed across the gap (displacement between its two end fixes / time):

  | Speed across | Gaps | Median length | Total | Ending at a coarse fix (> 500 m) |
  |---|---|---|---|---|
  | < 0.5 m/s, stationary | 52 | 17.5 min | 37.5 h | 2 |
  | 0.5–2 m/s | 13 | 3.3 min | 0.9 h | 6 |
  | ≥ 2 m/s, moving | 38 | 4.8 min | 2.8 h | 26 |

  - Stationary gaps are still periods: overnight and long stays show isolated fixes every 10–20 min.
  - Moving gaps are vehicle legs without good GPS, mostly ending at a **coarse fix** every ~4–5 min (32 coarse fixes,
    all without speed; 1,414 m ten times, 2,000 m twice, probably a cap). The pattern fits underground rail; there
    are no labels to confirm it.
  - Short vehicle stops don't show as gaps; they stay on the 20 s cadence.
- **Reported speed:** missing for 26% of fixes (still 35%, walk 12%, 2.5–4 m/s 11%, vehicle 27%, every coarse fix).
  Where present it agrees with derived speed: correlation 0.93 against consecutive fixes ≤ 30 s apart, median
  difference −0.08 m/s. Medians, reported vs 60 s derived: still 0.1 / 0.1, walk 1.1 / 1.1, 2.5–4 m/s 3.1 / 3.1,
  vehicle 7.4 / 6.5 m/s. Maximum reported 26.1 m/s.
- **Accuracy:** median 14.2 m, p95 107 m, 9.5% over 50 m. In a vehicle either very good or very poor (p25 6.6 m, p75
  44 m, p95 937 m; 24% over 50 m). **33% of fixes report exactly 14.25 m** (407 of 1,243; 14.24 m another 25), which
  looks like a default rather than an estimate.

#### Accelerometer (627,618 samples, 7.0 h)

- **Unit and gravity:** m/s², phone frame, gravity included. Median |acc| over 158 still 5 s blocks is 9.73 m/s²
  (per day 9.73 / 9.73 / 9.74), about 0.8% below 9.81, a small scale offset.
- **Rate:** 24.9 Hz real rate in every burst (p5 24.8). Integer-ms timestamps, 99.7–99.8% of steps 40 or 41 ms, so
  samples drift from a 40 ms grid over a burst (median 0.7–0.8 s, p99 2.7–6.3 s per day). Outside that: 465 steps
  under 39 ms, 594 of 42–200 ms, 14 over 200 ms; the largest step inside a burst is 0.97 s.
- **Bursts:** 91, median ~5 min (capped near 300 s, a few ~600 s), pauses of a few seconds between bursts while
  moving (median 3.2–3.8 s per day), then minutes to hours when the app stops. Recording only while moving.
- **Saturation (±65.5 m/s²):** 40 samples (6 / 6 / 28 per day).
- **Values on a 0.002 m/s² grid** (the app or the platform rounds them); 10× below the quiet threshold.
- **Noise floor** (median std |acc| of quiet minute windows, 25 Hz): **0.019 m/s²**, from only 3 quiet minute windows
  (9 quiet 15 s tiles), all inside GPS-covered stretches (2 at still fixes, 1 at a walk fix), none in a moving gap.
  For scale (same definitions, `noise_floor.py`): ExtraSensory iPhone dev users median 0.011 (WAIIS is above 27 of
  28), ExtraSensory Android 0.014, the low-noise Android user 4E98F91F 0.0070, NOR-TMD subway 0.019. So the WAIIS
  phone is not in 4E98F91F's low-noise regime (10's false-rail case), and a resting phone without any location did
  not occur on these days.
- **Axis signs: iOS signs kept (scaled to m/s², not converted to Android's) is strongly indicated given the iPhone,
  not confirmed by a face-up recording.**
  - Still blocks by gravity axis: y negative 102 of 158, z negative 27, z positive 22, x positive 7.
  - Upright (top up), Android gives y = +9.8 and raw iOS y negative: y ≈ −9.8, the most common still orientation,
    fits iOS signs with an upright phone; on Android it would need the phone mostly upside down.
  - Flat blocks: z negative in 28 of 44, positive in 16. Face up, Android gives z = +9.8 and raw iOS z negative; face
    down reverses both, so the flat blocks alone can't tell the conventions apart.
  - To confirm: a 30 s recording with the phone face up on a table (expected z ≈ −9.8 if iOS signs are kept).
  - v2's rule A1 (iOS × −9.80665 to the Android convention) applies to this phone as `× −1`. E1's and E2's features
    are invariant to it (test planned).

#### GPS and accelerometer together

- **Minutes** (clock minutes; GPS = a fix with accuracy ≤ 500 m): 383 with both an `ok` accelerometer window and GPS,
  34 with an accelerometer window only, 70 with GPS but no accelerometer sample. Minute windows: 417 `ok`, 10 with a
  real rate below 24 Hz, 1 with a gap > 0.5 s, 1 at the edge.
- **Vehicle-state fixes with accelerometer data within 30 s:** 98% / 57% / 99% per day. On 10-31 most vehicle legs
  lie in moving GPS gaps; walk and still fixes have accelerometer data nearby on every day (92–98%).
- **Bursts → sessions:** the pause between bursts is either a few seconds or many minutes, so any joining limit from
  10 s to 120 s gives the same sessions (8 / 8 / 9 per day; 6 / 8 / 8 with 600 s).

#### Per day

| | 2026-10-30 | 2026-10-31 | 2026-11-03 |
|---|---|---|---|
| GPS fixes | 388 | 364 | 491 |
| GPS span / accelerometer span (UTC) | 12:27–23:24 / 16:09–22:57 | 01:22–22:30 / 12:14–20:28 | 07:18–23:03 / 12:33–21:14 |
| interval p50; share 20–30 s; share < 20 s | 20 s; 83%; 4.9% | 20 s; 75%; 3.3% | 20 s; 82%; 8.0% |
| fixes by state: still / walk / 2.5–4 m/s / vehicle / unknown | 163 / 143 / 0 / 65 / 17 | 166 / 141 / 5 / 21 / 31 | 145 / 191 / 75 / 71 / 9 |
| median 60 s derived speed, vehicle-state fixes with reported speed | 18.8 m/s | (1 fix) | 5.4 m/s |
| gaps > 60 s: stationary / 0.5–2 m/s / moving | 16 (8.1 h) / 2 / 8 (0.5 h) | 21 (17.1 h) / 4 / 24 (1.8 h) | 15 (12.3 h) / 7 / 6 (0.4 h) |
| coarse fixes (> 500 m); moving gaps ending at one | 9; 5 of 8 | 14; 17 of 24 | 9; 4 of 6 |
| reported speed missing | 23% | 36% | 22% |
| reported vs consecutive-fix derived speed, corr. | 0.97 | 0.63 | 0.64 |
| accuracy exactly 14.25 m | 33% | 32% | 33% |
| accuracy p50 / p95; share > 50 m | 14.2 / 66 m; 5.9% | 14.2 / 154 m; 12.9% | 14.2 / 87 m; 9.8% |
| accelerometer samples; hours | 206,333; 2.3 h | 180,893; 2.0 h | 240,392; 2.7 h |
| bursts; median length; median pause | 30; 295 s; 3.2 s | 26; 262 s; 3.8 s | 35; 297 s; 3.2 s |
| largest step inside a burst | 0.90 s | 0.20 s | 0.97 s |
| saturated samples | 6 | 6 | 28 |
| still 5 s blocks; median \|acc\| | 27; 9.73 | 89; 9.73 | 42; 9.74 |
| still blocks by gravity axis | y− 21, z+ 6 | y− 55, z− 27, x+ 7 | y− 26, z+ 16 |
| flat still blocks: z sign | 6 of 6 positive | 26 of 26 negative | 10 of 12 positive |
| vehicle-state fixes with accelerometer within 30 s | 98% | 57% | 99% |
| minute windows `ok`; other | 138; 3 rate | 119; 3 rate, 1 gap, 1 edge | 160; 4 rate |
| minutes: acc + GPS / acc only / GPS only | 123 / 15 / 16 | 108 / 11 / 35 | 152 / 8 / 19 |
| sessions (pauses ≤ 10 s / 120 s / 600 s join) | 8 / 8 / 6 | 8 / 8 / 8 | 9 / 9 / 8 |
| quiet minute windows; noise floor | 1; 0.017 | 2; 0.022 | 0; – |

- **10-30** has the fast vehicle legs with good GPS (vehicle median 18.8 m/s, reported and derived speed agree at 0.97).
- **10-31** is the hard day for coverage: 24 moving gaps (1.8 h), 17 of them ending at coarse fixes, only 57% of
  vehicle-state fixes with accelerometer data nearby, and 35 GPS-only minutes.
- **11-03** has slow movement: 75 fixes at 2.5–4 m/s and vehicle-state fixes at a median of 5.4 m/s, which fits
  cycling or slow urban traffic (the diary will tell). It has few moving gaps (0.4 h) and the most sub-20 s intervals (8%).
- The lower reported / derived correlation on 10-31 and 11-03 comes with little speed range and consecutive-fix noise;
  gps-v2 doesn't read reported speed (Plan).

### Stop-point 2: dev results (2026-11-04)

Commits before the runs: `2082bd3` (stop-point 1 state), `0309cdf` (`tmd.legs`, dev helpers, checks), `c19774c` (grid
script). MLflow experiment `11_leg-suggestions_waiis-app-days`; outputs under `<data>/processed/11_.../` (minute
probabilities, GPS features) and `results/tables/` (git-ignored until the export).

**Before the dev runs** (as asked at the OK of stop-point 1):
- `app_days/`: the four WAIIS day scripts moved here from 08 and rewired to `tmd.readers.waiis` (Results, stop-point 1).
- Device table moved to `metadata/waiis_devices.csv` (project level, in git; one row per user key and device period;
  layout in `metadata/README.md`). Reader `tmd.readers.waiis` and tests `tests/test_waiis_reader.py`,
  `tests/test_waiis_sign_invariance.py`.
- **Sign invariance passes** on a WAIIS day (10-30, first 40 minute windows): windows read with and without the
  `× −1` rule are exact negatives; the 580 features and E1 / E2's acc-model probabilities are identical (tolerance
  1e-9).

#### Step 0: 10's saved fold models (`check_models.py`, `logs/check_models.log`)

Loaded fold models (acc `E2/E1_seed<k>_fold<f>`, GPS `A0_gps_seed<k>_fold<f>`) predicting 10's dev frame, then fusion +
HMM per fold, against 10's saved out-of-fold arrays, ExtraSensory dev and SHL User 1, seeds 0–2: **reproduced**, worst
max |difference| acc / GPS 3.0e-8, raw 2.3e-7, p 1.6e-7 (float32 storage); six-class F1 identical to 4 decimals in
all 12 cases (e.g. SHL User 1 E2 seed 0: 0.9532; ExtraSensory E2 seed 0: 0.6498). No refit was needed.

#### Step 1: WAIIS GPS → E2's GPS input (`gps_candidates.py`, `logs/gps_candidates.log`, MLflow `gps_<rule>_seed<k>`)

SHL User 1's 1 Hz track (1,377,605 fixes) thinned per recording to 20 s: 70,259 fixes (median interval 20.1 s; 96%
of intervals 20–21 s); none above 500 m. 10's per-minute GPS features recomputed from the bench table equal its saved
features (sanity check). E2's acc probabilities = 10's saved out-of-fold; GPS model out of fold; HMM per fold.

Six-class macro-F1 on SHL User 1 official minutes after fusion + HMM (seed 0 decides; mean and range over seeds 0–2):

| Rule | seed 0 | mean [min, max] | E1 (mean) | before HMM (mean) | GPS model alone, covered minutes | GPS-covered official minutes | raw legs: edits / h, leg F1 (mean) |
|---|---|---|---|---|---|---|---|
| **C1** nearest the centre | 0.9529 | 0.9547 [0.9529, 0.9562] | 0.9642 | 0.8566 | 0.703 | 75.5% | 0.90, 0.826 |
| C2 best accuracy ±7.5 s | **0.9550** | 0.9560 [0.9550, 0.9567] | 0.9660 | 0.8369 | 0.680 | 58.2% | 0.91, 0.822 |
| C3 best accuracy in the minute | 0.9533 | 0.9534 [0.9523, 0.9544] | 0.9623 | 0.8553 | 0.693 | 75.5% | 0.92, 0.818 |
| C4 median position | 0.9537 | 0.9542 [0.9537, 0.9552] | 0.9638 | 0.8560 | 0.703 | 75.5% | 0.90, 0.825 |
| C5 every fix | 0.9487 | 0.9498 [0.9487, 0.9520] | 0.9596 | 0.8427 | 0.601 | 76.4% | 1.01, 0.807 |
| ref: 10's per-minute GPS (1 Hz, training rule) | 0.9532 | 0.9553 [0.9532, 0.9577] | 0.9634 | 0.8547 | 0.710 | 73.9% | 0.93, 0.820 |
| ref: no GPS | 0.9493 | 0.9505 [0.9493, 0.9512] | 0.9659 | 0.7878 | – | 0% | 0.97, 0.808 |

- **Chosen: C1** (fix nearest the minute centre) by the pre-set rule: C1–C4 lie within 0.005 of the best (C2, 0.9550);
  C1, C3 and C4 have the most GPS-covered minutes (75.5%); C1 is the simplest.
- **The choice carries little signal.** SHL User 1 is near its ceiling after the HMM: all one-fix rules are within
  0.002 of the per-minute reference (mean over seeds), and even no GPS is only 0.005 below. Before the HMM, GPS is worth
  +0.07 (0.79 → 0.86) and C1 / C3 / C4 match the reference. C2 loses a quarter of the covered minutes (a 20 s fix
  falls within ±7.5 s of the centre only ~3 minutes in 4) and is the worst one-fix rule before the HMM; it wins by
  0.002 after the HMM on seed 0.
- **C5 (every fix) is worse, as expected:** the GPS model alone drops from 0.70 to 0.60 on covered minutes (bus recall
  after HMM 0.91 vs 0.94), and raw legs need 11% more edits. That is the cost of feeding 20 s features to a model
  trained on one fix per minute.
- E1 gains nothing from GPS on SHL User 1 (no GPS 0.966, C1 0.964): its acc model alone is already at the ceiling there.
- Limits: one SHL person (hips) and its 1 Hz GPS thinned to a clean 20 s cadence; WAIIS's coarse fixes, 14.25 m
  accuracies and dropouts aren't in it.

**Motorcycle** (`motorbike.py`, `logs/motorbike.log`): no ExtraSensory dev minute is flagged `motorbike`. The only
flagged minutes (86, one user) are in the reserved test set, so they are not used. What a motorcycle would be predicted
as stays a reasoned guess (car).

**Deviations and readings so far:**
1. *Edits, tie-break* (`src/tmd/legs.py`, `score`): when two legs overlap the leg they are mapped to by the same number
   of minutes, the one with the same mode wins, then the earlier. Without it, a tie could count a mode change and a
   split the user wouldn't make (found by `tests/test_legs.py::test_edits_counted`).
2. *Constraint (b)*: read as "not more than 0.01 below the raw HMM legs"; a rule set that improves ExtraSensory by more
   than 0.01 is not excluded.
3. *Day key*: ExtraSensory has no recording day; edits per day use user + UTC date. UTC midnight is about 16:00 local
   for most ExtraSensory users (US West Coast), so a user-day is cut in the afternoon. This affects only edits per day,
   not the objective (edits per hour).
4. *Internal boundaries*: a truth leg end counts for boundary errors only where the next truth leg starts at exactly
   that minute; ends at unlabelled time or data gaps don't. On SHL User 1 that is 172 boundaries.

#### Step 2: leg rules and thresholds (`leg_grid.py`, `logs/leg_grid.log`, MLflow `leg_grid_E2`, `leg_grid_E1`)

Minute probabilities: SHL User 1 with C1 (20 s), ExtraSensory dev with its own per-minute GPS (10's saved `p`). 768
combinations per model on seed 0, then the chosen rules and the raw legs on seeds 1–2. Full grids in
`results/tables/leg_grid_<model>.csv`; per mode in `leg_per_mode_<model>.csv`; seeds in `leg_chosen_seeds_<model>.csv`.

**Chosen (by the pre-set objective and tie rules):**

| | E2 | E1 |
|---|---|---|
| rules | `s_veh` 0, `s_other` 3, `g` 10, `l_min` 2, `c` 0.5 | `s_veh` 0, `s_other` 0, `g` 2, `l_min` 3, `c` 0.5 |
| combinations meeting (a) | 736 of 768 | 768 of 768 |
| second mode shown (`τ2`) | none reaches the 50% rule | none |

**Seed 0, raw HMM legs → chosen rules** (seeds 1–2: chosen SHL edits / h 0.886–0.896 for E2, 0.838–0.871 for E1;
ExtraSensory 1.379–1.384 and 1.456–1.461):

| | E2 SHL User 1 | E2 ExtraSensory | E1 SHL User 1 | E1 ExtraSensory |
|---|---|---|---|---|
| hours of labelled time; days | 390 h; 74 | 4,045 h; 354 user-days | 390 h; 74 | 4,045 h; 354 |
| truth legs (short < 5 min) | 1,016 (149) | 10,379 (5,384) | 1,016 (149) | 10,379 (5,384) |
| **edits / h** | 0.909 → **0.896** | 1.398 → 1.384 | 0.843 → **0.838** | 1.404 → 1.461 |
| edits / day | 4.80 → 4.73 | 15.97 → 15.81 | 4.45 → 4.42 | 16.04 → 16.69 |
| merges / splits / mode changes / boundary moves | 173 / 76 / 91 / 10 | 1,181 / 1,942 / 2,022 / 452 | 166 / 71 / 82 / 8 | 1,448 / 1,962 / 2,035 / 464 |
| leg F1 (top-1) | 0.823 → 0.824 (0.824) | 0.718 → 0.718 (0.722) | 0.830 → 0.836 (0.838) | 0.718 → 0.705 (0.709) |
| time-weighted accuracy (top-1) | 0.952 → 0.952 (0.952) | 0.917 → 0.916 (0.917) | 0.965 → 0.964 (0.965) | 0.915 → 0.915 (0.915) |
| boundary error median / p90 [min] | 0 / 3 | 1.9 / 13.6 | 0 / 3 | 2.0 / 15.8 |
| extra / missed legs per h | 0.60 / 0.36 | 0.60 / 0.79 | 0.56 / 0.33 | 0.67 / 0.81 |
| **time wrong and unflagged** | 4.8% | 8.3% | 3.5% | 8.5% |
| legs wrong and unflagged | 139 | 1,727 | 129 | 1,729 |
| legs flagged (unknown) | 0.0% | 1.0% | 0.3% | 1.0% |
| short-leg recall | 0.58 | 0.66 | 0.58 | 0.66 |
| range mode: top-1 / top-2 | 0.912 / 0.957 | 0.808 / 0.943 | 0.921 / 0.960 | 0.808 / 0.942 |
| range mode: ECE / Brier | 0.049 / 0.158 | 0.118 / 0.335 | 0.042 / 0.142 | 0.119 / 0.335 |

**What the grid shows:**
- **The leg rules hardly matter.** Over the whole grid, SHL edits per hour range 0.889–1.370 (E2), but the chosen
  rules gain only 1.4% (E2) and 0.6% (E1) on the raw HMM legs, about the size of the seed spread. The HMM already
  removes the short flips the rules target. Larger `l_min` (5) and `c` 0.9 cost edits.
- **E1's choice costs on ExtraSensory:** edits +4% (1.40 → 1.46), leg F1 top-1 −0.009, just inside (b). Rules chosen on
  one SHL person don't carry over. E2's choice is neutral there (−1% edits).
- **Constraint (a) doesn't bind on SHL User 1:** the raw legs are at 4.8% (E2) and 3.5% (E1). So the objective picks the
  lowest threshold, `c` 0.5, which flags almost nothing (0–1% of legs). **On ExtraSensory, wrong legs are mostly
  confidently wrong:** with E2, `c` 0.9 flags 19.5% of legs and 2.8% of time, yet only takes wrong-unflagged time from
  8.3% to 6.8%. No threshold in the grid reaches 5% there.
- **The second mode rarely holds the truth** when top-1 is wrong: SHL 37% / 14% / 7% of those legs at `τ2` 0.1 / 0.2 /
  0.3 (ExtraSensory 25% / 14% / 7%), so no `τ2` meets the 50% rule. Top-2 is still listed in the output; the "show"
  flag would stay off.
- **Range mode** (the user selects a true leg): top-1 0.91 / top-2 0.96 on SHL, 0.81 / 0.94 on ExtraSensory. Averaged
  minute probabilities are calibrated on SHL (ECE 0.05) and overconfident on ExtraSensory (0.12).
- **Per mode** (E2, chosen, `results/tables/leg_per_mode_E2.csv`):
  - SHL User 1: leg recall ≥ 0.92 for walking, running, bike, car, bus; rail 0.84, still 0.67 (short stills inside
    trips: short-leg recall 0.22). Precision lowest for car (0.69), rail (0.72) and walking (0.72): extra short legs.
  - ExtraSensory: still 0.82; walking 0.37 (walk read as still: time recall 0.57), bus 0.32, rail 0.41 (37 legs),
    car 0.55, bike 0.52. Boundary p90 of still legs 30 min (coarse label timing; reported, not tuned on).
- **Edits per day:** about 5 on SHL User 1 (5.3 h of labelled time per day), about 16 per ExtraSensory user-day (11
  h, many short self-reported legs).

#### Step 3: 20 s boundary refinement (`refine.py`, `logs/refine.log`, MLflow `refine_E2`, `refine_E1`)

1 s truth for all 172 internal SHL truth boundaries (median shift from the minute boundary 10.5 s, max 29 s). The rule
moved 568 of 1,022 slow / fast boundaries (E2).

| E2 (E1) against the 1 s truth | minute boundaries | refined |
|---|---|---|
| boundary error median [min] | 0.35 (0.33) | 0.55 (0.49) |
| boundary error p90 [min] | 2.81 (3.08) | 2.84 (3.02) |
| edits / h | 0.940 (0.866) | 0.955 (0.881) |

**Not adopted:** it makes boundaries worse (median +0.21 min, edits +1.6%), against a margin of −0.5 min and −5%.
Boundaries stay on the minute grid. A 20 s pair speed crossing halfway between leg speeds is noisier than the
minute-level HMM boundary.

#### E1 vs E2 on dev (seeds 0–2; E2 with its chosen rules, E1 with the raw HMM legs as approved)

| mean [min, max] | E2 SHL User 1 | E1 SHL User 1 | E2 ExtraSensory | E1 ExtraSensory |
|---|---|---|---|---|
| edits / h | 0.893 [0.886, 0.896] | **0.858** [0.843, 0.879] | **1.382** [1.379, 1.384] | 1.405 [1.404, 1.406] |
| time-weighted accuracy | 0.954 [0.952, 0.956] | **0.964** [0.963, 0.965] | 0.916 | 0.915 |
| time wrong and unflagged | 4.6% [4.4, 4.8] | **3.6%** [3.5, 3.7] | 8.3% | 8.5% |
| leg F1 | 0.827 | 0.830 | 0.720 | 0.717 |
| time recall still / rail | 0.82 / 0.974 | **0.90** / 0.976 | 0.960 / 0.29 | 0.959 / 0.40 |

**E1 is better on SHL User 1 in every seed** (edits per hour, time accuracy, wrong unflagged time). The gain is in
**still**, not rail: still time recall 0.90 vs 0.82, rail equal (0.976 vs 0.974), and E1's subway minute recall is
lower (0.87–0.88 vs 0.92–0.93). E1's NOR-TMD-trained acc model reads SHL User 1's still minutes better. **On
ExtraSensory they are tied** (E2 −0.02 edits per hour; E1's rail time recall 0.40 vs 0.29 rests on 37 rail legs).
No choice between them is made on dev; both run on the WAIIS days.

#### Confidence does not catch the main errors

On ExtraSensory about **8% of labelled time is wrong and unflagged** (E2 8.3%, E1 8.5%). The errors are mainly
**walking read as still** (walking time recall 0.57) and **bus** (0.43). Most of these minutes are confident: `c` 0.9
would flag 19.5% of legs and 2.8% of time for only −1.5 points of wrong unflagged time (8.3% → 6.8%; E2). On SHL User
1 the threshold has nothing to do (raw legs 4.8% / 3.5%). **A high-confidence suggestion is not a checked one:** `c`
0.5 marks almost no leg unknown, and a one-click accept of a confident leg can still store a wrong mode.

#### Stop-point 2 approved 2026-11-04

1. **GPS input C1** (fix nearest the minute centre); **second mode off** (top-2 still in the output); **no 20 s
   boundary refinement**.
2. **Leg rules:** E2 as chosen (`s_veh` 0, `s_other` 3, `g` 10, `l_min` 2, `c` 0.5). **E1 uses the raw HMM legs**
   (`g` 10, no other rule) with `c` 0.5.
3. **Threshold `c` 0.5** for both models, with the caveat above.
4. Readings 1–4 accepted (with the UTC note on ExtraSensory user-days).

**Deviation 5 (from the Plan's per-model choice, `leg_grid.py` / `results/tables/leg_choice.json`):** E1's leg rules
chosen by the objective (`s_veh` 0, `s_other` 0, `g` 2, `l_min` 3) are not used. They gain 0.6% edits per hour on SHL
User 1 (within seed spread) and cost 4% edits per hour (1.40 → 1.46) and 0.009 leg F1 on ExtraSensory. E1 uses the
raw HMM legs instead, with `c` 0.5.

**Open questions:**
- Platform and phone model should come from the app or the data build once more users label. Until then the reader
  uses `metadata/waiis_devices.csv`.
- Axis signs of the WAIIS iPhone: a face-up recording (no effect on E1 / E2).
- Device selection in the data build (`selected_device_rule = greatest_data_object_bytes`) with a second device.

**Next (stop-point 3), after the diaries:** E1 and E2 on the three WAIIS days with these rules, review sheets
(including the tram check), SHL Users 2–3 once.

### Stop-point 3: the WAIIS days and SHL Users 2–3 (2026-11-05)

**Diaries** (written by the recording person before any model ran on the days; received 2026-11-05): 28 legs (10 /
11 / 7), saved as `<processed>/11_.../waiis/diaries/<date>.csv` (local time, tool top level, rail sub-mode, `gt` =
start / end / both confirmed by public transport card validation). Rough times otherwise. Metro / train validation
can come a few minutes before boarding, bus / tram a few minutes after; for public transport only the start is known.
**Walking was not tracked from 10-31 on** (beyond the listed legs), so walking / running legs on 10-31 and 11-03 are
"not in diary", not extra. Modes: metro → rail (hint metro), tram → rail (hint tram), bicycle → bike, run → running.

**Code** (commits `b7dba28`, `4379b1a`, `6ddef89`): `tmd.suggest` (WAIIS path: reader → minute windows → 580 features →
C1 GPS → saved model → fusion → HMM → `tmd.legs.predict_legs` + `gps_only_legs`), `tmd.legs_adapter` (the output
format), `waiis_days.py` (both models, three days, output files, review sheets, MLflow `waiis_<date>_<model>`),
`review.py` (diary matching, tram check, package legs, sheet), `rail_minutes.py`, `shl23.py`. Tests:
`tests/test_suggest.py`; all 118 tests pass. Outputs (personal, outside git) under `<processed>/11_.../waiis/`:
`user=<key>/date=<day>/legs_<model>.json`, `review/<day>_sheet.png`, `review/<day>_legs_review.csv` (one row per leg
with empty `review` / `review_note` columns for right / wrong / unsure), `review/diary_matching.csv`,
`review/tram_check.csv`, `review/summary.csv`.

**Deviations at stop-point 3:**
6. *GPS-only stretches* (`tmd.legs.gps_only_legs`): the 5 min minimum length applies to moving stretches too (the
   Plan set it for still ones), so the short pauses between legs of one session don't become unknown legs.
7. *Diary times parsed at second resolution* (`review.py`, `diary`): the first dry run (no MLflow) matched nothing
   because pandas parsed the times at second resolution; fixed before any logged run.
8. *Walking on 11-03*: the first logged runs counted walking legs on 11-03 as extra; the diary note says walking
   wasn't tracked from 10-31 on. Re-run; the six first runs are kept in MLflow with tag `superseded`.

#### Run time (single thread, `logs/waiis_days.log`)

Reading, minute windows, features, both GPS steps, model, HMM and legs: **0.12–0.18 s per day per model** for 2.0–2.7
h of accelerometer data (119–160 minute windows), about 1 ms per minute window. Model sizes and per-window latency
are 10's (unchanged models).

#### Suggestions per day (E2 / E1)

| | 2026-10-30 | 2026-10-31 | 2026-11-03 |
|---|---|---|---|
| minute windows; GPS-covered | 138; 89% | 119; 91% | 160; 95% |
| model legs | 11 / 13 | 10 / 10 | 21 / 21 |
| GPS-only legs: still / moving (unknown) | 4 / 3 | 4 / 5 | 5 / 4 |
| model legs below `c` 0.5 (unknown) | 0 / 0 | 0 / 0 | 0 / 0 |
| diary legs matched (same mode, IoU ≥ 0.5) | 2 / **5** of 10 | 0 / 0 of 11 | 2 / 2 of 7 |
| extra legs (moving mode, < 50% inside a same-mode diary leg) | 2 / 2 | 1 / 1 | 3 / 2 |
| package (`fusion_late__hmm_viterbi`): matched, extra | 3, 5 | 0, 1 | 1, 10 |

**By diary mode, all three days** (E2 / E1; same-mode share = share of the diary leg's time under legs of its mode;
data = share under model legs, i.e. with accelerometer minutes):

| diary mode | legs | matched | same-mode share | data |
|---|---|---|---|---|
| rail (13 metro, 2 tram) | 15 | **0 / 1** | 0.02 / 0.09 | 0.48 |
| walking | 8 | 1 / 3 | 0.42 / 0.40 | 0.46 |
| bus | 1 | 0 / 0 | 0.46 / 0.45 | 0.51 |
| bike | 2 | 2 / 2 | 0.78 | 0.81 |
| car | 1 | 1 / 1 | 0.95 | 0.95 |
| running | 1 | 0 / 0 | 0 | 0 |

- **Missing accelerometer data is the first problem.** Half of the diary's rail time has no model minute: the app
  didn't record. 7 of the 13 metro legs have no accelerometer minute at all. Those stretches become GPS-only unknown
  legs with their speed across (e.g. 10-31: 14:04–14:31 at 5.5 m/s, 17:58–18:11 at 8.8 m/s), so the trip shows but has
  no mode. The two late metros on 10-31 (22:51, 23:13) come after the accelerometer stopped (21:28) and fall into one
  2 h unknown leg (1.1 m/s across).
- **Where metro minutes exist, they are rarely rail** (`rail_minutes.py`, `logs/rail_minutes.log`; 55 model minutes in
  6 metro legs, 76% GPS-covered): **E1 rail 20 of 55** (subway 16, train 4), **E2 rail 4 of 55**; otherwise walking
  (E2 28, E1 19) and still (16 / 16). Some walking minutes are likely real: a GT start is the card validation, before
  the walk to the platform. Without GPS, E1 says rail on 9 of 13 minutes. The one E1-matched metro is 10-30 21:39–21:52
  (hint metro); E2 has a 4 min rail leg on 10-31 17:54 (IoU too small to match).
- **Bike and car are right** (both bike legs and the car leg matched by both models, 0.95–1.00 confidence where data
  exist). **Bus** (11-03, 18:05–19:10, rough times) is found as two bus legs (18:31–18:39, 18:47–19:09) split by a
  6 min GPS-only gap; IoU 0.34, not matched.
- **Confident errors, as on dev:** the extras are confident (bike 0.95, bus 0.93–0.95 for the trams; car 0.68 at the
  start of the 11-03 metro; walking ≥ 0.90 near the 10-30 car trip, plausibly unlisted walks). No model leg fell below
  `c` 0.5 on any day.
- **The package** (`fusion_late__hmm_viterbi`, 18 s rows; descriptive): 4 of 28 matched (E2 4, E1 7); no rail at all,
  more extra legs on 11-03 (10, mostly short flips).

#### Tram check (`review/tram_check.csv`; descriptive, no model choice)

| diary tram leg (11-03) | model minutes; GPS | E2: leg, minute argmax | E1: leg, minute argmax |
|---|---|---|---|
| 14:01–14:10 (GT start) | 6; 100% | bike; bike 6 | bike; bike 6 |
| 16:57–17:05 (GT start) | 8; 100% | bus; bus 8 | bus; bus 8 |

- **Neither tram leg is right at the top level** (both should be rail). E1, with a tram class and GPS, gives tram 0 of
  14 minutes (mean rail probability 0.00); the share of tram / train / subway is 0 in both legs for both models.
- This is E1's first tram check with GPS: on these two legs its tram class doesn't fire.

#### SHL Users 2–3, once (`shl23.py`, `logs/shl23.log`, `results/tables/shl23_legs.csv`, MLflow `shl23_E2/E1`)

10's final models reproduce 10's saved test probabilities for these rows exactly (max |difference| 0). 40 h of
labelled time, 113 truth legs per position. C1: 1 Hz track thinned to 20 s (5,190 fixes), 65% of hips minutes
covered.

| E2 / E1 | GPS | leg F1 | edits / h | time accuracy | wrong unflagged time | rail leg recall |
|---|---|---|---|---|---|---|
| hips | per minute | 0.823 / **0.861** | 1.03 / 0.95 | 0.923 / 0.941 | 7.7% / 5.9% | 0.80 / 0.93 |
| hips | C1, 20 s | 0.830 / 0.853 | 1.08 / 1.00 | 0.926 / 0.935 | 7.4% / 6.5% | 0.80 / 0.93 |
| bag | per minute | 0.858 / 0.886 | 0.85 / 0.73 | 0.928 / 0.928 | 7.2% / 7.2% | 0.80 / 1.00 |
| torso | per minute | 0.863 / 0.812 | 0.83 / 1.10 | 0.893 / 0.897 | 10.7% / 10.2% | 0.87 / 0.87 |
| hand | per minute | 0.785 / 0.767 | 1.40 / 1.45 | 0.806 / 0.850 | 19.2% / 14.8% | 0.80 / 0.80 |

(C1 for bag / torso / hand: `results/tables/shl23_legs.csv`; within ±0.07 leg F1 of per minute, edits +0–0.35 / h.)

- E1 is ahead on the hips (leg F1 +0.04, rail leg recall 0.93 vs 0.80), as on SHL User 1; mixed elsewhere.
- Hand is the weakest position for both (wrong unflagged time 15–19%); flagging stays near 0–2% of legs.
- C1 at 20 s against the per-minute GPS the model was trained on: hips leg F1 +0.007 (E2) / −0.008 (E1), edits +5%.

**Waiting for the review** of each leg in `review/<day>_legs_review.csv` (right / wrong / unsure). Then Results,
draft Conclusion, STATUS.md.
