# Status

*As of 2026-11-05. Project facts and rules: [CLAUDE.md](CLAUDE.md). Decisions: [DECISIONS.md](DECISIONS.md).*

## 1. Data

**`waiis_bench_v2`** is the frozen benchmark every model reads (`tmd.bench` default since 2026-11-03), built at
commit `6d51dc2` (experiment 09, approved).
- v1 + NOR-TMD (Android; 14 reserved installations) + Collecty; 9 fine classes (tram added); six-class primary
  score (still, foot, bike, car, bus, rail) on ExtraSensory dev, pooled out of fold; guard NOR-TMD dev (bus, car,
  rail). Reserved set `splits/reserved_test_v3.json` (`tmd.splits.reserved_test()` default).
- Dev rail 11,127 windows (v1: 5,931), bus 9,985 (4,376). iOS converted to the Android axis convention.
- Contents and counts: [experiment 09's README](experiments/09_frozen-benchmark_es-shl-nor-collecty/README.md);
  decisions: [09's decision register](experiments/09_frozen-benchmark_es-shl-nor-collecty/decision_register.md),
  which defers to [04's](experiments/04_frozen-benchmark_es-shl/decision_register.md) for ExtraSensory and SHL.
- Splits: [splits/README.md](splits/README.md).
- Not rebuildable from raw at the moment: the Collecty and NOR-TMD zips were removed after the build (the build
  reads `collecty.zip` for the duplicate check). The built files verify against the manifest.

**`waiis_bench_v1`** (experiment 04, built at `e51ea79`) stays readable and unchanged with `version="waiis_bench_v1"`;
05–08 pin it. Rebuilt hash-identical and 06's dev out-of-fold probabilities bit-identical after every v2 change.

## 2. Current results

**Reference model on `waiis_bench_v2`: experiment 10's E2** (below; Conclusion approved 2026-11-03, its DECISIONS.md
entry still **draft, not approved**): 06's configuration trained on ExtraSensory + SHL User 1 + Collecty.

**On `waiis_bench_v1`: experiment 06**, accelerometer + GPS fusion
([README](experiments/06_gps-acc-baseline_waiis-bench/README.md), approved 2026-10-30). It replaced **05**
(accelerometer only; [README](experiments/05_acc-baseline_waiis-bench/README.md), approved 2026-10-30), which stays
the accelerometer-only baseline. E2 keeps 06's model, fusion and smoothing (below).

| `waiis_bench_v1` | 8-class macro-F1 (06 / 05) | family macro-F1 (06 / 05) |
|---|---|---|
| ExtraSensory test, 12 users (mean of 3 seeds) | **0.707** / 0.666 | **0.747** / 0.698 |
| — iPhone / Android | 0.764 / 0.682 (05: 0.762 / 0.665) | 0.847 / 0.708 (05: 0.801 / 0.690) |
| ExtraSensory dev, pooled out-of-fold | 0.613 / 0.551 | 0.681 / 0.615 |
| SHL Users 2–3, rail check | 0.847 / 0.795 | 0.944 / 0.932 |

- **v2's six-class score** (rescored by 09, no retraining, v1 labels): dev 0.642 / 0.575, test 0.686 / 0.642
  (06 / 05). On v2's labels the run flag changes the dev set, so the first v2 model rescores 06 on v2 as its
  baseline.
- **Model (06):**
  - acc model: 05's LightGBM (580 features, `tmd.features.acc`);
  - GPS model: LightGBM on 13 GPS features from one fix per minute (`tmd.features.gps`, gps-v2: speeds derived
    from positions, acceleration, stops, heading change), trained with SHL User 1;
  - fusion: product of the two models' probabilities, with the acc model alone where there is no GPS;
  - smoothing: HMM forward-backward per session.
  - Cost: 1.4 + 3.8 MB; ~12 ms per single window, ~1 ms per window over whole sessions.
- **GPS platform effects** are judged by a cross-platform transfer test (DECISIONS.md, 2026-10-30): GPS features
  identify iPhone vs Android within a class, but the fusion's gain carried over fully to a platform the GPS model
  never saw.
- **What mattered:**
  - the HMM (05: +0.12 family);
  - GPS late fusion (06: +0.06 8-class on dev, +0.04 on test);
  - SHL User 1 in training, for both models.
  - Not adopted: stacking, early fusion, a lower learning rate, gravity, low-pass, context features.
- **Weak spots:** bus (test 0.25), ExtraSensory train / rail (2 Android test users), walk (test recall falls with
  GPS, 0.715 → 0.650), no 8-class gain on iPhone test users.
- **Evaluation:** user-bootstrap CIs redraw resamples that lack a class of the evaluation set
  (`tmd.evaluate.user_resamples`).

### First models on v2 (10; Conclusion approved 2026-11-03)

**10 GPS + acc baseline on `waiis_bench_v2`** ([README](experiments/10_gps-acc-baseline_waiis-bench-v2/README.md),
2026-11-03): 06's configuration on v2 with ablations of the new sources, dataset-shortcut checks, test scored once.
- **Final configuration E2:** 06's configuration (8 classes) trained on ExtraSensory + SHL User 1 + Collecty
  (Collecty's tram minutes unlabelled). **Reference A0:** 06 retrained on v2's labels (ExtraSensory + SHL User 1).
- **No NOR-TMD and no tram class in it.**

| `waiis_bench_v2`, six-class macro-F1, mean of 3 seeds | E2 (final) | A0 (reference) |
|---|---|---|
| ExtraSensory test, 12 users | **0.685** | 0.667 |
| — iPhone / Android | 0.756 / 0.656 | 0.757 / 0.634 |
| ExtraSensory dev, pooled out-of-fold | 0.650 | 0.650 |
| NOR-TMD test, bus / car / rail (14 installations) | 0.469 | 0.463 |
| SHL Users 2–3 rail check, hips | 0.945 | 0.942 |

- **E2 vs A0:** tied on dev; +0.018 on the test, all on Android rail and bus (2 and 7 users), within A0's test seed
  spread (0.024). Collecty's users +0.07 on dev. A safe addition, not a proven gain.
- **NOR-TMD (descriptive E1):** NOR-TMD test 0.778 (subway 0.97) and better subway at every SHL position, but
  ExtraSensory dev −0.013 to −0.020 in every variant and seed (seeds don't resample users; E1 −0.013 [−0.032,
  +0.011]). False rail on still minutes without GPS, concentrated in one Android device without location data
  (4E98F91F, 115 of 185 minutes) plus scattered sitting minutes; the mechanism is not shown (not a NOR-TMD label
  convention; fusion overrules it where GPS exists). Also false bus on minutes with GPS (bus precision −0.084).
  Descriptive: 4E98F91F's accelerometer has the lowest noise floor of the ExtraSensory Android users, otherwise
  ordinary; without it E1 − A0 is −0.000 to −0.005 on dev, the bus precision cost stays. Not adopted; a follow-up.
- **Tram:** not learnable on v2 without NOR-TMD (Collecty's 4 users: recall 0 on its own held-out users); a balanced
  tiny tram class also halves the acc model's early stopping (found in 10; leave tram out of the stopping fold when
  it is a class: `tmd.models.lgbm.paired_cv(stop_ok=...)`).
- **Unchanged from 06:** fusion +0.06 to +0.08 six-class on dev in every row; walk → still 27% of test walk minutes;
  no gain on iPhone; cost 1.65 + 4.09 MB, ~12 ms per window, ~1 ms per window over sessions. SHL weight 0.5 and SHL's
  other positions: nothing.
- **Weak spots:** ExtraSensory bus (test recall 0.29) and train (2 users); subway off the hips on SHL (hand 0.08,
  torso 0.18); NOR-TMD bus / subway without NOR-TMD in training (0.39 / 0.11).

### Label curve (07, development data only)

**07 user label curve** of 06's final configuration ([README](experiments/07_user-label-curve_waiis-bench/README.md)),
approved 2026-11-02. No test users were read.
- **Domain step:** with SHL User 1 in training, 8-class dev F1 is 0.452 with no ExtraSensory user, 0.586 with 3
  (+0.13), 0.613 with all ~38 (+0.03 more). Past n = 3 it gains about +0.01 per doubling (flattens at n = 3).
- Without SHL User 1: 0.544 with all ~38 users; at that point 59% of the gap is train (6 non-rail classes: 0.635 vs
  0.668).
- Still gaining at the end: **bus** (most), walk, bike. Flat: still, car, run. Rail can't be measured.
- Reading for WAIIS: label a few users per mode and per platform first (the reference model is WAIIS's n = 0), then
  target bus and train. DECISIONS.md entry 2026-10-30, approved 2026-11-04; per platform needs the platform stored
  with each recording.
- Its open run question (dev recall 0.26 vs 0.74 on test) was traced in 09 to one dev user whose run moves like
  walking; v2 drops run labels of such users.

### Candidate datasets (08) and benchmark v2 (09)

**08 quality check** of NOR-TMD, Collecty and MobilitApp ([README](experiments/08_candidate-datasets-qc/README.md))
and **09 build of `waiis_bench_v2`** ([README](experiments/09_frozen-benchmark_es-shl-nor-collecty/README.md)), both
approved 2026-11-02. No models.
- **Verdicts:** NOR-TMD usable, Android only (iOS records at 5 Hz); Collecty a small training addition (≤ 5 users per
  class except car); MobilitApp not usable (gravity removed, 56% of files untimed). Commercial use of NOR-TMD and
  Collecty verified (`LICENSES.md`).
- **None has GPS**, and both add only vehicles + bike / run: within-class source identification from 05's features
  has AUC 0.82–0.96 (tram 0.52), so shortcut checks belong in the first model on v2.
- **iOS sign:** iOS accelerometers report in g with the opposite sign to Android (also in ExtraSensory). v2 converts
  iOS to the Android convention and m/s²; 05 / 06's features are sign-invariant, so v1 results are unaffected.

### Leg suggestions for the labelling tool (11, 12)

**11 leg suggestions on the WAIIS app days** ([README](experiments/11_leg-suggestions_waiis-app-days/README.md),
2026-11-05): the first suggestion engine (`tmd.readers.waiis`, `tmd.suggest`, `tmd.legs`, `tmd.legs_adapter`; one
file per model and day; ~0.15 s per day). Conclusion and DECISIONS.md entry (E1 as the tool's default, E2
selectable) approved 2026-11-05.
- **GPS input:** one fix per minute, nearest the minute centre (C1); all ~20 s fixes cost accuracy.
- **Legs:** the HMM makes them; extra leg rules gain ~1% edits per hour. Confidence doesn't catch the main errors
  (ExtraSensory ~8% of time wrong and unflagged; `c` 0.5 flags almost nothing): a suggestion is not a checked label.
- **E1 vs E2:** E1 better on SHL User 1 (every seed) and SHL Users 2–3 hips, tied on ExtraSensory dev; on the WAIIS
  days (lenient comparison with diaries) 8 vs 6 of 19 vehicle legs right, paired 2–0, all metro.
- **WAIIS limits:** 5 of 19 vehicle legs (metros) have no accelerometer data (app recording gap); metro minutes often
  read as walking / still; the one tram leg reads as bus. Bike, car and bus right; no false vehicle legs.
- The three WAIIS days are used and no longer unseen.

**12 GPS gaps as an underground-rail signal** ([README](experiments/12_gps-gap-rail_shl-waiis/README.md),
2026-11-05; Conclusion **draft, not approved**): a moving GPS gap (no good fix for ≥ 5 min, speed across 2–25 m/s)
added to the engine (`tmd.legs.GapRule`, rules version `legs-v2 (12)`). **Only R1 is on** (`tmd.suggest.GAP_RULE`,
`k` = 1, both models). With it, the WAIIS vehicle-leg suggestions equal 11's, plus the hinted gap legs.
- **R1, no accelerometer data:** an `unknown` leg with hint `underground_rail` and reason `gps_gap_moving`. On the
  WAIIS days, 4 of the 5 metro legs 11 couldn't suggest now get it, with no false hints. It stays `unknown`: read as
  GPS only, gaps are rail 0.77 and metro ~0.5 of the time on SHL User 1.
- **R2, accelerometer data:** train and subway probabilities × `k` = 2 before the HMM.
  - Dev (SHL User 1): edits per hour −2.7% for E1 and −1.6% for E2, from few legs off the hips.
  - Held-out: worse on SHL Users 2–3 (+2.7% edits per hour, car / bus false rail), and E1's ExtraSensory test false
    rail +0.046 pp (dev limit 0.02 pp).
  - WAIIS: one metro leg (E2).
  - Switched off after the once-check (`k` = 1, the untuned setting); no other `k` chosen.
- **Missing GPS alone is no signal:** in ExtraSensory it is mostly Android, and moving gaps are 0.1–0.3% of minutes.
  SHL car outages and bus gaps look like metro by length and speed. Train loses GPS as often as metro.
- **WAIIS logs coarse fixes (> 500 m) underground** (8 of 13 metro legs). Metro read as walking or still remains,
  mostly short rides (gaps < 5 min) or the HMM keeping still against raw rail evidence.

### WAIIS app data (own days, descriptive)

Three unlabelled days of WAIIS app recordings (2026-10-30, 10-31, 11-03; one person, one iPhone 16), described as one
dataset in [experiment 11's README](experiments/11_leg-suggestions_waiis-app-days/README.md) (Results, stop-point 1).
- **GPS:** a fixed ~20 s cadence while recording (80% of intervals 20–30 s), not one fix per minute. Vehicle legs
  without good GPS (likely underground rail) get only a coarse fix (> 500 m) every ~5 min with no speed. Short
  vehicle stops stay on the 20 s cadence. Reported speed is missing for 26% of fixes and matches derived speed where
  present; `activity` / `is_moving` are useless; 33% of accuracies are exactly 14.25 m.
- **Accelerometer:** m/s², gravity included, 24.9 Hz real rate in ~5 min bursts while moving.
- **Platform:** iPhone 16, confirmed 2026-11-04; not recorded in the data. **Axis signs:** iOS signs kept (scaled to
  m/s², not converted to Android's) is strongly indicated given the iPhone, not confirmed by a face-up recording. v2's
  iOS sign rule applies to WAIIS iPhone data as `× −1`; E1 / E2's features are sign-invariant. Device table (platform,
  phone model, axis convention per user and period): `metadata/waiis_devices.csv` (in git).
- **Diaries** of the three days (rough times, card-validation GT starts) and the review outputs:
  `<data>/processed/11_leg-suggestions_waiis-app-days/waiis/` (personal, outside git).

## 3. Superseded experiments

- **01 UniMTS few-shot on SHL.**
  - Outdated: one SHL user, a day split, acc + gyro comparisons.
  - Still holds: pretrained UniMTS beats LightGBM only at k ≤ 2 and is weak on vehicles; not adopted.
- **02 family baseline on SHL + ExtraSensory.**
  - Outdated: acc + gyro features, the old splits and labels, numbers not on the benchmark.
  - Still holds:
    - HMM smoothing (Viterbi) is the cheapest gain (+0.09–0.11 family F1);
    - SHL alone doesn't transfer to ExtraSensory, so training data matter more than the model;
    - context features built from the context length leak the dataset;
    - car vs bus and train vs subway stay weak without GPS.
- **03 label curve on SHL User 1.**
  - Outdated: a same-person day split, acc + gyro.
  - Still holds: the curve hadn't flattened by k = 1000, and the remaining gain is in the vehicle classes.

## 4. Next experiment

1. **E1 + Collecty**, then with labelled WAIIS days added (11's draft). Chosen on `waiis_bench_v2` dev, not on the
   three WAIIS days used in 11. Carries the NOR-TMD follow-up: keep NOR-TMD's vehicle gains (its test 0.778 vs 0.469)
   without the false rail and false bus on ExtraSensory (e.g. a lower NOR-TMD weight, still minutes of many devices
   as hard negatives, per-user false-rail counts, 4E98F91F's resting signal), with a tram-aware HMM; metro minutes
   read as walking / still on WAIIS (a model question, not a GPS-gap rule: 12), with labelled WAIIS days: check
   metro minutes with raw rail ~0.4 that the HMM decodes as still (12); ExtraSensory test only for the model it chooses. Leg metrics with
   `tmd.legs` beside the six-class score.
2. **GPS track:** Geolife, MTL Trajet, SHL 1 Hz (MOBIS if granted); cadence-robust gps-v3 for WAIIS's ~20 s fixes
   and ~5 min coarse fixes on vehicle legs without good GPS; leg-level classification of vehicle legs.

Later: InceptionTime / LIMU-BERT on v2, walk-aware fusion, map context for bus vs car, driver vs passenger.
Open WAIIS questions: the app's recording gap in the metro (5 of 19 vehicle legs without accelerometer data in 11;
12's gap legs mark 4 of them);
axis signs (iOS signs strongly indicated on the confirmed iPhone 16, to confirm with a face-up recording); platform
and phone model stored by the app or the data build once more users label; the data build's device selection with a
second device; the labelling tool's real format (one adapter module); modes needed. Answered by the app-days
description (11): recording in ~5 min bursts while moving, m/s² with gravity, GPS stays on the 20 s cadence at
vehicle stops.
