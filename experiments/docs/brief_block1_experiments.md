# Brief: block 1 experiments (SHL + ExtraSensory)

> **Parked after experiment 02 (2026-10-28).** The next experiment follows a new plan; 03–06 below are not scheduled.

## Goal
Find the best model for transport mode from phone data, input = accelerometer + gyroscope at 25 Hz, arbitrary orientation. The phone records and uploads; the model runs on an edge server (decided 2026-10-25).
- Production output: the family `still / foot / bike / road / rail`.
- Car vs bus and train vs subway: always trained as separate labels and reported, but not relied on. A separate GPS + map stage (experiment 05) tries to resolve them per leg; GPS never enters the sensor model's features.
- Main question: which model holds up on **unseen users and an unseen dataset**. In-dataset scores are secondary.

### Metrics
Model size is not a hard limit (edge server; the earlier 2 MB phone ceiling is dropped). Every candidate reports exactly these:
1. macro-F1;
2. confusion matrix and per-class recall;
3. inference time per window: features + model end to end, single thread, after warm-up, median and p95;
4. model size.

Work as five experiments, in this order: 02 → 03 → 04 → 05 → 06. Each answers one question. Shared code (readers, resampler, windows, label mapping, split search, features, HMM, eval) goes in `../../src/tmd`. The reserved final test is `../../splits/reserved_test.json`; each experiment's own folds stay in its folder (02: `../02_family-baseline_shl-es/splits/`).

## Data
Paths: `<data>` is the data root in `CLAUDE.local.md` (`/Users/ms/Projects/waiis/data`), not `~/data`. Items marked
*(verified 02)* were checked in the 02 data audit (`experiments/02_family-baseline_shl-es/README.md`, Results step 1).

### SHL (`<data>/raw/shl/release`)
- `User1`: about 391 h, hips phone only, over 7 months.
- `User{1,2,3}_4phones`: 3 days per user, 4 positions (Hand, Hips, Torso, Bag).
- Days in `User1_4phones` may also be in `User1`. Deduplicate by recording date; one day is never in two partitions.
  *(verified 02)* 220617, 260617 and 270617 are in both, as identical copies; keep the `User1_4phones` copy (4 positions).
  `User1` has 90 usable recordings on 76 calendar days (`m<day>` = a second recording of the same day). `160517` is
  empty, and `180417` / `200317` lack `00inf.txt`. User 1 hips: 391.3 labelled hours.
- *(verified 02)* The motion files are already on an exact 100 Hz grid (resampled by SHL), so the phones' original
  jitter is not visible. Units are m/s² and rad/s.
- Inspect the folder structure first. Check the column order of the motion and label files against the SHL documentation; don't trust memory. *(verified 02 against `doc/doc_dataset.pdf` §3.2 / §3.11: correct.)* Expected order: time, acc xyz, gyro xyz, mag xyz, orientation, gravity, linear acc, pressure, …; coarse label 1–8 = Still, Walk, Run, Bike, Car, Bus, Train, Subway; 0 = null, which is dropped.
- Use **raw acc (with gravity) and gyro** only. Don't use SHL's orientation, gravity or linear-acc channels: the phone won't send them. Separate gravity yourself (low-pass ≈ 0.3 Hz).
- Magnetometer and pressure are used only in experiment 03. Location/GPS files, if present, are used only in experiment 05.

### ExtraSensory (`<data>/raw/extrasensory`)
- Per-user features+labels CSVs: use them **only for labels, phone position and `label_source`**. Never report results from their motion summary features: they were computed on the native-rate snippets (≈ 34–100 Hz, see below), not at 25 Hz.
- Original (uncleaned) label files: needed too. Cleaning dropped rare labels, including `ON_A_TRAIN` (344 minutes), so a train minute can appear in the cleaned CSV as e.g. `SITTING`. *(verified 02: 344 minutes in 5 users; on 342 of them the cleaned CSV says still. The files were downloaded on 2026-10-24 to `per_uuid_original_labels/`.)*
- Raw files: `raw_acc` (includes gravity), `proc_gyro` (calibrated), optionally `raw_magnet`. One snippet per minute, matched by user + timestamp. Some files are missing or contain only "nan": skip them and count them in the audit.
- *(verified 02)* **Not 40 Hz / 20 s.** Every acc snippet has 800 samples:
  - iPhone acc: ≈ 34 Hz, ≈ 23 s, **in g**.
  - Android acc: mostly ≈ 50 Hz, ≈ 16 s, in m/s². Some users are partly at ≈ 100 Hz (8 s).
  - Gyro: ≈ 40 Hz (iPhone), 40–100 Hz (Android).

  Units must be decided **per snippet**, not per platform: one Android user (BEF6C611) is in g for half of their
  minutes. Rule: median |acc| < 3 → g → × 9.80665. Gyro is rad/s on both platforms.
- *(verified 02)* **No gyroscope:** 61359772, CCAF77F0 and F50235E0 (all Android). 617 other minutes have no gyro
  file, and 290 minutes have no acc file.
- *(verified 02)* About 1.3k acc snippets and 1.2k gyro snippets (mostly Android, 33 users) have time running backwards
  or a span over 40 s. **Drop them** and count them.
- Resample each snippet to 25 Hz on its own; never interpolate across the gaps between snippets (≈ 37–52 s, depending on the snippet length).
- A label NaN means "not reported", not 0.

### Common rules
- Every signal is resampled to 25 Hz (anti-aliasing low-pass, then decimation) before any other step. The only exception is the rate comparison in 03.
- Windows: 5 s, stride 2.5 s. The label is the majority label; mixed windows are dropped from training and kept for evaluation.
- Long context: 60 s on SHL. On ExtraSensory, context = consecutive minutes (with gaps). Develop long-context models on SHL first.
- Inputs that don't depend on orientation: acc magnitude, vertical acc (projected on gravity), horizontal acc magnitude, gyro magnitude. Add random-rotation augmentation for the deep models.

## Labels
- Train on the 8 SHL classes. Report both the 8 classes and the family: foot = walk + run, road = car + bus, rail = train + subway.
- ExtraSensory → classes, one per minute, in this priority order:
  1. `ON_A_TRAIN` (original labels) → train. San Diego has both commuter rail and a light-rail trolley, so trust it at family level (rail) and don't read much into train vs subway here.
  2. `ON_A_BUS` → bus
  3. `IN_A_CAR` / `DRIVE_*` → car. `DRIVE_-_I_M_A_PASSENGER` doesn't name the vehicle: with `ON_A_BUS` / `ON_A_TRAIN` it is bus / train, not a conflict *(decided 2026-10-25; 475 bus minutes)*.
  4. `BICYCLING` → bike
  5. `FIX_running` → run
  6. `FIX_walking` / `STAIRS_*` → walk
  7. `SITTING` / `OR_standing` / `LYING_DOWN` → still. `SLEEPING` minutes are dropped (phone on the nightstand; 83k minutes) *(labels v2, decided 2026-10-26)*.
- Drop minutes whose labels conflict (e.g. walking + a vehicle, or car + bus) and `ELEVATOR`. A conflict means two or more of train / bus / car / bike / run / walk in one minute. Car means `IN_A_CAR` or `I'M_THE_DRIVER` (passenger: see above). A still label never conflicts; the moving class wins.
- Scan the original labels for other transport labels dropped in cleaning (e.g. motorbike, plane, tram). Map one to a class only if it clearly is one of ours; otherwise drop those minutes. List them in the audit.
  *(decided 2026-10-25)* Dropped: `ON_A_PLANE`, `ON_A_BOAT`, `SKATEBOARDING`, `STATIONARY_BIKE`, `TREADMILL`, `ELLIPTICAL_MACHINE`, `WHEELCHAIR` (and `RIDING_AN_ANIMAL`, `USING_CRUTCHES`, which have 0 minutes).
- **Motorbike** (86 minutes, one user: 33A85C34, a dev user): never trained on, never a 9th class. It is kept as a separate labelled check: train without it, then report what each model predicts on it (bike? road?), from the fold where that user is held out. This decides whether motorbike can be folded into road. LIMU-BERT-X's "riding" class (04) was scooters and motorcycles. **Open with the company:** do motorbikes matter for anti-fraud? If so, they should be labelled in its recordings.
- *(labels v2, decided 2026-10-26)* **Phone not carried** = `PHONE_ON_TABLE` or original `PHONE_AWAY_FROM_ME`:
  - walk / run / bike with the phone not carried → dropped (the label describes the person, not the phone);
  - car / bus / train keep them (the phone rides along);
  - still: not-carried minutes are capped per user at 20% of that user's still minutes (fixed seed), in training and in evaluation. Resting phones are real still states in production, so they are kept, not removed.
  *(verified 02: before v2, 44% of still minutes had the phone not carried and 41% had no position label.)*
- Every ExtraSensory table reports still twice: "still (v2)" (main macro-F1) and "still, carried phone only" (pocket / hand / bag reported, not resting). Still predicted as rail is broken down by carried / not carried / unknown position.
- ExtraSensory has very little rail (344 `ON_A_TRAIN` minutes). Report its rail recall together with that count, and check rail false positives (rail predicted on road or still minutes) in the confusion matrix.

## Splits (fixed once, committed as files, never changed)
| Partition | SHL | ExtraSensory |
|---|---|---|
| Unlabelled pre-training | User 1, all days and positions | 48 dev users |
| Training + tuning | User 1, folds of whole days (every class in each fold) | 48 dev users, 5 folds split by user |
| Final test (experiment 06 only) | Users 2 and 3, all 4 positions | 12 test users |

Files: final test in `../../splits/reserved_test.json` (the reference); 02's folds in `../02_family-baseline_shl-es/splits/`.

- Before choosing the 12 ExtraSensory test users, count minutes per user × class and the platform. Then pick: ≥3–4 users with bus, car and bike, and a balance of iOS and Android. If the train minutes come from ≥2 users, put at least one of them in the test set. Fix the seed.
  *(decided 2026-10-25)* This means per class: ≥ 4 test users with ≥ 5 usable minutes for each of bus, car and bike, and about 20% of each class's minutes. One train user in test is enough, because SHL users 2–3 are the real rail test. Only users with a gyroscope are candidates.
- ExtraSensory has only 4 dev users with train, so one dev fold has no train. "Every class in each fold" applies to classes that occur in ≥ 5 groups.
- With 4–6 test users per class, ExtraSensory test metrics depend on who is in the test set: **report them with a confidence interval from resampling test users** (bootstrap over users).
- Test users never enter any stage before 06, self-supervised pre-training included.
- During development the cross-user signal comes from the ExtraSensory dev folds, and rail is checked mainly on held-out User 1 days and other positions (ExtraSensory's few train minutes are too thin to decide on).

Training setups, used where an experiment says so:
- **R1:** SHL only.
- **R2:** SHL + ExtraSensory dev users.
- **R3:** R1 model + k labelled ExtraSensory dev users, k ∈ {1, 3, 10, all}. This is the label-efficiency curve, a stand-in for the first company labels.

Extra breakdowns in every table: per dataset and per phone position.

---

## 02_family-baseline_shl-es
**Question:** How well do hand-crafted features + LightGBM + HMM classify the family and the 8 classes at 25 Hz, across days, positions, users and datasets?

**Plan**
1. **Data audit.**
   - Structure, actual rates, jitter, gaps, units, missing files.
   - Days shared between `User1` and `User1_4phones`.
   - Hours per user × position × class (SHL); minutes per user × class × platform × position (ExtraSensory, classes from cleaned + original labels).
   - Propose the 12 ExtraSensory test users. **Stop-point: wait for my approval of the splits.**
2. **Features per 5 s window** on the orientation-independent channels:
   - Time domain: mean, std, percentiles, zero crossings, autocorrelation peak.
   - Welch band energies 0–1, 1–3, 3–6, 6–12 Hz; dominant frequency; spectral entropy.
   - Use the SHL 2024 winner's rotation-invariant features as a reference (arXiv 2407.11048; tsflex).
   - Run R1 with cross-validation. **Stop-point.**
3. Add 60 s context features (stats of the window features over the previous 60 s) and HMM smoothing, with transitions estimated from training sequences only. Compare with step 2. **Stop-point.**
4. R2 vs R1 on the ExtraSensory dev folds.
5. Export ONNX; measure size and latency (features and model separately).

## 03_inputs-sensors-rate_shl
**Question:** Do magnetometer and barometer, or a different sampling rate, change the result enough to request them in the company's recordings?

**Plan**
- Best configuration from 02, SHL only, CV on User 1 days plus the other positions.
- Sensors:
  - acc + gyro
  - + magnetometer (magnitude features only)
  - + pressure: Δpressure features over 5 s and 60 s, never absolute pressure
- Rates: 10 Hz, 20 Hz, 25 Hz, native 100 Hz.
- Report per-class recall for car / bus / train / subway, not just macro-F1.
- The Conclusion should give a clear recommendation for the company's data request.

## 04_limubertx-transfer_shl-es
**Question:** Does the pre-trained LIMU-BERT-X beat the 02 baseline at equal cost, especially across datasets and with few labels?

**Facts to verify in the code, not assume**
- Repo: `github.com/WANDS-HKUST/LIMU-BERT_Experience` (MIT) → clone to `../ext/`, record the commit.
- Weights: `weights/limu_bert_x.pt` (227 KB). Config `base_v4`: 6 channels, hidden 72, 4 layers, 4 heads, `seq_len` 20.
- Default data version `10_20`, i.e. probably 10 Hz × 20 samples = 2 s. The paper (arXiv 2509.24303) says 10 Hz with 6 s windows. Establish which one the shipped weights expect before using them.
- Normalisation: acc / 9.8 (`utils.py`).
- First reproduce the repo's documented HHAR fine-tuning as shipped, in its own environment. **Stop-point.**

**Plan**
1. Frozen encoder: average the embeddings per window, pool them over 60 s, train logistic regression. Compare with the same probe on 02's features, same windows.
2. GRU or small TCN over 60 s of embeddings (frozen encoder).
3. Continued self-supervised pre-training from the X weights on the unlabelled dev partitions (SHL User 1 + ExtraSensory dev), with rotation augmentation. Then repeat 2. **Stop-point.**
4. Full fine-tuning (lower learning rate for the encoder than for the head).
5. Control: the same architecture with random initialisation, trained the same way.
6. R3 label-efficiency curve for: baseline 02, X frozen, X fine-tuned, random-init control.
7. If 03 shows that 10 Hz costs the bus class: add a branch with the 3–6 / 6–12 Hz band energies at 25 Hz, joined to the embeddings at the head.
8. ONNX export, size, latency for the whole chain (resampling + encoder + head).

## 05_gps-legs_shl-es
**Question:** At leg level, how much do GPS + OpenStreetMap (OSM), on top of the sensor family, add for car vs bus and train vs subway?

**Plan**
1. Check which location data exists: SHL location/GPS files; ExtraSensory has per-minute location summary features in the CSV, and maybe raw coordinates (check). Report the point density. **Stop-point** if it's too sparse.
2. Legs: change points on the smoothed family output from 02 (or 04).
3. Features per leg:
   - Sensor side, no GPS: sequence pattern (walk → wait at a stop → ride → walk vs no walking access).
   - GPS side: speed percentiles; stops per km; dwell times; share of stops within ~30 m of an OSM bus stop; distance to rail lines; OSM railway type (`subway` / `rail` / `light_rail`); GPS gaps (underground).
   - OSM extracts for the SHL area (Brighton / London) and San Diego. No live vehicle positions or GTFS needed.
4. Report sensor-only per leg vs sensor + GPS + OSM, as separate rows.

## 06_final-test_shl-es
**Question:** How do the chosen candidates perform on users never seen before?
- Take each candidate's final configuration from 02 / 04 (and 05 for the leg stage) and run it **once** on SHL users 2–3 and the 12 ExtraSensory test users. No decisions are taken from these numbers.
- Output: accuracy vs size / latency table and Pareto plot, plus the full per-class / per-position / per-dataset breakdown.
