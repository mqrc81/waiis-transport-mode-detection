# Status

*As of 2026-11-03. Project facts and rules: [CLAUDE.md](CLAUDE.md). Decisions: [DECISIONS.md](DECISIONS.md).*

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

**Reference model: experiment 06**, accelerometer + GPS fusion
([README](experiments/06_gps-acc-baseline_waiis-bench/README.md), approved 2026-10-30), on `waiis_bench_v1`. It
replaces **05** (accelerometer only; [README](experiments/05_acc-baseline_waiis-bench/README.md), approved
2026-10-30), which stays the accelerometer-only baseline. No model has been trained on v2 yet.

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

### Label curve (07, development data only)

**07 user label curve** of 06's final configuration ([README](experiments/07_user-label-curve_waiis-bench/README.md)),
approved 2026-11-02. No test users were read.
- **Domain step:** with SHL User 1 in training, 8-class dev F1 is 0.452 with no ExtraSensory user, 0.586 with 3
  (+0.13), 0.613 with all ~38 (+0.03 more). Past n = 3 it gains about +0.01 per doubling (flattens at n = 3).
- Without SHL User 1: 0.544 with all ~38 users; at that point 59% of the gap is train (6 non-rail classes: 0.635 vs
  0.668).
- Still gaining at the end: **bus** (most), walk, bike. Flat: still, car, run. Rail can't be measured.
- Reading for WAIIS: label a few users per mode and per platform first (06 is WAIIS's n = 0), then target bus and
  train. The DECISIONS.md entry is still **draft, not approved**.
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

### WAIIS app data (own days, descriptive)

Two unlabelled days of WAIIS app recordings ([report](experiments/08_candidate-datasets-qc/waiis_app_days/README.md),
2026-11-03; no models).
- **GPS:** a fixed ~20 s cadence while recording (79% of intervals 20–30 s), not one fix per minute. Vehicle legs
  without good GPS (likely underground rail) get only a coarse fix (> 500 m) every ~5 min with no speed. Short
  vehicle stops stay on the 20 s cadence. Reported speed is missing for 29% of fixes (40% in vehicles) and matches
  derived speed where present; `activity` / `is_moving` are useless; 33% of accuracies are exactly 14.25 m.
- **Accelerometer:** m/s², gravity included, 24.9 Hz real rate in ~5 min bursts while moving. Platform isn't
  recorded; the phone was probably an iPhone 16 (not certain). If so, the app keeps iOS's axis signs (scaled to
  m/s², not converted to Android's), so v2's iOS sign rule would apply to WAIIS iPhone data. Unconfirmed.

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

1. **06 on v2:** 06 as-is vs retrained with NOR-TMD and Collecty; SHL weight ablation; per-source and shortcut
   checks (with/without new datasets, NOR-TMD transfer). GPS model unchanged.
2. **Leg suggestions** for the labelling tool (in parallel): `predict_legs` with 06, evaluated on SHL, my own
   labelled WAIIS days and later MobilityNet; harness ready for WAIIS labels.
3. **GPS track:** Geolife, MTL Trajet, SHL 1 Hz (MOBIS if granted); cadence-robust gps-v3 for WAIIS's ~20 s fixes
   and ~5 min coarse fixes on vehicle legs without good GPS; leg-level classification of vehicle legs.

Later: InceptionTime / LIMU-BERT on v2, walk-aware fusion, map context for bus vs car, driver vs passenger.
Open WAIIS questions: axes and platform (not recorded; probably iOS signs on an iPhone 16, to confirm with a
face-up recording and the phone model stored with the data), modes needed. Answered by the app-days report: recording in ~5 min bursts while moving, m/s² with gravity, GPS stays on
the 20 s cadence at vehicle stops.
