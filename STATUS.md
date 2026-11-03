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
([README](experiments/06_gps-acc-baseline_waiis-bench/README.md)). Conclusion **drafted, not yet approved**.
- It replaces **05** (accelerometer only; [README](experiments/05_acc-baseline_waiis-bench/README.md), approved
  2026-10-30), which stays the accelerometer-only baseline.

| `waiis_bench_v1` | 8-class macro-F1 (06 / 05) | family macro-F1 (06 / 05) |
|---|---|---|
| ExtraSensory test, 12 users (mean of 3 seeds) | **0.707** / 0.666 | **0.747** / 0.698 |
| — iPhone / Android | 0.764 / 0.682 (05: 0.762 / 0.665) | 0.847 / 0.708 (05: 0.801 / 0.690) |
| ExtraSensory dev, pooled out-of-fold | 0.613 / 0.551 | 0.681 / 0.615 |
| SHL Users 2–3, rail check | 0.847 / 0.795 | 0.944 / 0.932 |

- **Model (06):**
  - acc model: 05's LightGBM (580 features, `tmd.acc_features`);
  - GPS model: LightGBM on 13 GPS features from one fix per minute (`tmd.gps_features`, gps-v2: speeds derived
    from positions, acceleration, stops, heading change), trained with SHL User 1;
  - fusion: product of the two models' probabilities, with the acc model alone where there is no GPS;
  - smoothing: HMM forward-backward per session.
  - Cost: 1.4 + 3.8 MB; ~12 ms per single window, ~1 ms per window over whole sessions.
- **Adopted despite a failed platform check** (a recorded deviation): GPS features identify iPhone vs Android
  within a class. A cross-platform transfer test showed the gain doesn't depend on it.
  - A DECISIONS.md entry (draft) makes this transfer test the criterion for GPS features.
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

**07 user label curve** of 06's final configuration ([README](experiments/07_user-label-curve_waiis-bench/README.md)).
Conclusion **drafted, not yet approved** (revised at review). No test users were read.
- **Domain step:** with SHL User 1 in training, 8-class dev F1 is 0.452 with no ExtraSensory user, 0.586 with 3
  (+0.13), 0.613 with all ~38 (+0.03 more). Past n = 3 it gains about +0.01 per doubling (flattens at n = 3).
- Without SHL User 1: 0.544 with all ~38 users; at that point 59% of the gap is train (6 non-rail classes: 0.635 vs
  0.668).
- Still gaining at the end: **bus** (most), walk, bike. Flat: still, car, run. Rail can't be measured.
- Draft reading for WAIIS: label a few users per mode and per platform first (06 is WAIIS's n = 0), then target bus
  and train (draft DECISIONS.md entry).
- Open dataset question: dev run recall 0.26 at every n against 0.74 on test.

### Candidate datasets for waiis_bench_v2 (08, quality check)

**08 quality check** of NOR-TMD, Collecty and MobilitApp ([README](experiments/08_candidate-datasets-qc/README.md)).
No models, no MLflow runs. Verdicts, Conclusion and DECISIONS.md entry **approved 2026-11-02**.
- **NOR-TMD** (Oslo, Bodø; 152 installations): usable for training, **Android only**: 12,988 clean v1 windows,
  219 h; bus 70 users, subway 27, train 17, car 13, bike 6. iOS records at 5 Hz, so 214 h aren't clean. As an
  external test it covers vehicles and bike only (open: reserve ~20% of its users as a second test set).
- **Collecty** (Zagreb; 15 users, Android): a small training addition (9,026 clean v1 windows, mostly car, train and
  bus from ≤ 13 users, top-user shares up to 0.98). Clock repairs and dedup approved.
- **MobilitApp** (Barcelona): **not usable** (final). Gravity is removed; 56% of files have no sample times.
- Commercial use of NOR-TMD and Collecty verified (2026-11-02, `LICENSES.md`).
- **None has GPS.** Both usable sets add only vehicles + bike / run: the within-class dataset-identification check
  is required in v2. Without MobilitApp, tram (37 users, 2 datasets) passes the new-class rule; e-scooter (4)
  doesn't.
- **iOS sign:** iOS accelerometers report in g with the opposite sign to Android, in NOR-TMD and in ExtraSensory
  (still minutes: iPhone mean z −9.4, Android +9.1 m/s²). v1 results are unaffected (05 / 06 features are
  sign-invariant, checked on real windows); raw-axis models would not be. v2 rule: iOS → Android convention and m/s²
  in every reader.

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
3. **GPS track:** Geolife, MTL Trajet, SHL 1 Hz (MOBIS if granted); cadence-robust gps-v3 for WAIIS's ~10 s fixes;
   leg-level classification of vehicle legs.

Later: InceptionTime / LIMU-BERT on v2, walk-aware fusion, map context for bus vs car, driver vs passenger.
Open WAIIS questions: recording mode, units and axes per platform, modes needed, GPS at vehicle stops.
