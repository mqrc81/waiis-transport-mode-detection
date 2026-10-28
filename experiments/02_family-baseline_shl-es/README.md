# 02 Family baseline on SHL + ExtraSensory

## Question

How well do hand-crafted features + LightGBM + HMM classify the family and the 8 classes at 25 Hz, across days,
positions, users and datasets?

*(From `docs`, which also holds the data, label, split and setup rules
(R1/R2/R3) that this experiment follows.)*

## Plan

1. **Data audit.**
   - Structure, actual rates, jitter, gaps, units, missing files.
   - Days shared between `User1` and `User1_4phones`.
   - Hours per user × position × class (SHL); minutes per user × class × platform × position (ExtraSensory, classes
     from cleaned + original labels).
   - Propose the 12 ExtraSensory test users. **Stop-point: wait for my approval of the splits.**
2. **Features per 5 s window** on the orientation-independent channels:
   - Time domain: mean, std, percentiles, zero crossings, autocorrelation peak.
   - Welch band energies 0–1, 1–3, 3–6, 6–12 Hz; dominant frequency; spectral entropy.
   - Use the SHL 2024 winner's rotation-invariant features as a reference (arXiv 2407.11048; tsflex).
   - Run R1 with cross-validation. **Stop-point.**
3. Add 60 s context features (stats of the window features over the previous 60 s) and HMM smoothing, with
   transitions estimated from training sequences only. Compare with step 2. **Stop-point.**
4. R2 vs R1 on the ExtraSensory dev folds.
5. Export ONNX; measure size and latency (features and model separately).

## Results

### Step 1: data audit and proposed splits (2026-10-24, revised 2026-10-25) — **stop-point**

Status: **approved 2026-10-25.** Label rules and test criteria: see *Decisions* at the end of this step. The revised
splits are written to `splits/shl_days.csv` and `splits/es_users.csv` (see `splits/README.md`) and committed.
*(2026-10-28: both files moved unchanged to this folder's `splits/`. The reserved final test is now
`splits/reserved_test.json` at the project root, the reference from now on.)*

No model was trained, so nothing from this step is logged to MLflow (one dummy logging run, see Decisions). Everything below comes from the CSVs in `results/audit/`.

**Commands** (run from this folder; short output goes to `logs/*.out`, the full log to `logs/*.log`):

```
uv run python audit_shl.py --workers 5      # results/audit/shl_recordings.csv, shl_duplicates.csv  (~4 min)
uv run python audit_es.py --workers 5       # <data>/processed/es_minutes_audit.parquet (per minute, not in git),
                                            # results/audit/es_users.csv, es_original_labels.csv, es_train_minutes.csv
uv run python summarise_audit.py            # shl_hours_user_position_class.csv, es_minutes_user_class_position.csv
uv run python propose_splits.py --seed 0    # proposed_shl_day_folds.csv, proposed_es_users.csv  (~3 min)
```

Shared code added in `src/tmd`:
- `paths.py`: data root (`$TMD_DATA`, default `../data`).
- `labels.py`: the 8 classes, the family and `to_family`.
- `shl.py`: SHL readers.
- `extrasensory.py`: ExtraSensory readers and the minute → class mapping.
- `splits.py`: seeded random search for balanced folds and the test pick.

#### Deviations and setup notes
- The brief is at `docs`, not `docs/brief_block1.md`. The brief's `~/data/raw/...` is
  `<data>/raw/...`, where `<data>` = `/Users/ms/Projects/waiis/data` (`CLAUDE.local.md`).
- **The ExtraSensory original labels were missing** from the data root. I downloaded
  `http://extrasensory.ucsd.edu/data/additional_data_files/ExtraSensory.per_uuid_original_labels.zip` (sha256
  `7c017f75cbf3ca6d157a6ecfd8aacb1173493d9864772ab648ba27ac81373e79`) to `<data>/raw/extrasensory/` and unzipped it to
  `per_uuid_original_labels/`: 60 files, one row per minute, identical timestamps to the cleaned CSV.
- The local unzip of the cleaned CSVs left one folder per file (`<uuid>.features_labels.csv/<uuid>.features_labels.csv`).
  `extrasensory._cleaned_path` handles both layouts.
- Only 13 GB of disk are free on the data volume, so nothing is cached at 100 Hz. The audit streams the SHL text
  (~70 GB).

#### SHL
- **Structure.** `release/User1/`: 91 folders.
  - `160517` has no `Motion` or `Label` file and is skipped.
  - `180417` and `200317` have no `00inf.txt`. Their data are fine; `shl.info` returns `{}`.
  - The 90 usable recordings cover 76 calendar days. 14 days also have an `m<day>` recording (`shl.Recording.day`
    strips the `m`).
  - `User{1,2,3}_4phones/`: 3 days each, all 4 positions present.
- **Columns** were checked against `doc/doc_dataset.pdf` §3.2 and §3.11 and match the brief: time ms, acc xyz [m/s²],
  gyro xyz [rad/s], mag xyz [µT], orientation wxyz, gravity xyz, linear acc xyz, pressure [hPa], altitude,
  temperature. Coarse label 0 = Null, 1–8 = Still … Subway. `Label.txt` times equal `Motion.txt` times row by row (asserted
  for all 126 files).
- **Rate and jitter.** SHL ships the data already resampled to a regular 100 Hz grid (doc §3). Every one of the 126
  files has dt = 10 ms exactly, so the phones' original jitter can't be seen here.
  - The share of identical consecutive acc samples is ≤ 0.1% in most files, but higher in User 2/3: 12% (U2 140617
    Hand, Hips) and 21% (U3 070717 Torso). Those look like held values.
- **Gaps** (acc or gyro NaN): 0–2 gaps over 1 s per file, the longest 137 s. In total, 0.4 h of User 1 hips data
  has no acc or gyro. Mag and pressure are missing exactly where acc is.
- **Units.** The median |acc| on Still is 9.56–10.11 m/s² and the median |gyro| on Still is 0.004–0.15 rad/s, so no
  conversion is needed.
- **Duplicates.** `User1` 220617, 260617 and 270617 are copies of the `User1_4phones` hips data (not compared byte by byte): same
  size and sha1 of the first 50 MB of `Hips_Motion.txt`, and same `Label.txt` size (`shl_duplicates.csv`). They are
  kept once, from `User1_4phones` (all 4 positions).
- **Hours per user × position × class** (labelled, acc + gyro present, after deduplication;
  `shl_hours_user_position_class.csv`):

  | user | position | days | still | walk | run | bike | car | bus | train | subway | null |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | User1 | Hips | 76 | 57.2 | 51.7 | 17.7 | 48.3 | 68.2 | 51.1 | 56.2 | 40.4 | 223.3 |
  | User1 | Hand / Torso / Bag | 3 | 2.4 | 2.2 | 0.4 | 1.4 | 2.8 | 2.7 | 2.5 | 2.5 | 16.2 |
  | User2 | each of 4 | 3 | 3.4 | 4.1 | 0.3 | 2.3 | 4.3 | 1.2 | 2.2 | 2.9 | 5.3 |
  | User3 | each of 4 | 3 | 4.9 | 3.1 | 0.5 | 1.1 | 1.4 | 1.4 | 3.8 | 3.1 | 5.0 |

  User 1 hips has 391.3 labelled hours, which matches the brief. In User 2/3 the positions are synchronous phones on the same trip, so their
  hours are identical. User 1 has the other positions on only 3 days.
- Days of User 1 containing each class (> 3 min): still 67, walk 65, run 37, bike 43, car 27, bus 40, train 27,
  subway 16.

#### ExtraSensory
- **Structure.** There are 60 users (34 iPhone, 26 Android; platform from `cv_5_folds`) and 377,346 minutes in the
  cleaned CSVs.
  - `raw_acc`: 60 user folders. 290 minutes have no file. No file is empty or all-NaN.
  - `proc_gyro`: 57 user folders. **No gyroscope: 61359772…, CCAF77F0…, F50235E0…** (all Android). Another 617
    minutes in other users have no gyro file.
  - There are no raw files without a CSV row, and no original-label rows without a cleaned row.
- **Actual rates differ from the brief's "40 Hz, ~20 s"** (per-snippet medians, `es_users.csv`). Every acc snippet has
  exactly 800 samples.

  | | acc dt (median, p5–p95) | acc span | acc units | gyro dt (median, p5–p95) |
  |---|---|---|---|---|
  | iPhone | 29.6 ms (29.1–32.7), ≈ 34 Hz | 23.4 s | **g** (\|acc\| still ≈ 1.00) | 25.0 ms (23.5–27.4), ≈ 40 Hz |
  | Android | 20.0 ms (11.0–25.2), ≈ 50 Hz, some users partly ≈ 100 Hz (8 s snippets) | 16.0 s | m/s² | 24.7 ms (10.0–25.2) |

  - One Android user (BEF6C611…) is in **g for 1713 of their 3451 minutes**. Units therefore have to be decided per
    snippet, not per platform. The plan: |acc| median < 3 → g → × 9.80665.
  - Gyro magnitudes are consistent with rad/s on both platforms: still median 0.004 (iPhone) and 0.018 (Android);
    p95 over all minutes 0.7–0.8.
  - **Bad timestamps:** 1289 acc snippets (1253 Android, in 33 users) and 1226 gyro snippets have time going
    backwards or a span over 40 s. The resampler in step 2 must split or drop these. The proposal is to drop them and
    count them.
- **Labels** (`extrasensory.minute_classes`):
  - Priority order as in the brief.
  - `ON_A_TRAIN` comes from the original labels; everything else from the cleaned CSV.
  - NaN = not reported, never positive.
  - **Conflict** = two or more of {train, bus, car, bike, run, walk} in one minute. A still label never conflicts;
    a moving class wins, e.g. `SITTING` on a train → train.
  - **`DRIVE_-_I_M_A_PASSENGER` counts as car only without `ON_A_BUS` / `ON_A_TRAIN`** (approved 2026-10-25): it
    doesn't name the vehicle. Only bus/train + `IN_A_CAR` or `DRIVE_-_I_M_THE_DRIVER` is a conflict.

  Outcome over 377,346 minutes: ok 305,418; unlabelled 70,353; conflict 18; dropped 1,471; motorbike check 86. Usable
  for the model (class ok + acc + gyro present): **289,019 minutes**.

  | | still | walk | car | bike | bus | run | train |
  |---|---|---|---|---|---|---|---|
  | usable minutes | 249,171 | 21,430 | 10,882 | 4,599 | 1,699 | 896 | 342 |

  - **Conflicts:** 15 run + walk, 2 train + walk, 1 bus + car. Before the passenger rule, 476 bus + car minutes were
    conflicts (475 of them `ON_A_BUS` + passenger, 3 users), which cost 28% of the bus minutes. There was no
    train + passenger minute, so the rule changes nothing for train today.
  - **Other transport labels in the original labels** (`es_original_labels.csv`). These minutes are dropped
    (`extrasensory.DROP`). Minutes and users: `ON_A_PLANE` 630 (4), `ELLIPTICAL_MACHINE` 233 (2), `ELEVATOR` 200 (12,
    as the brief says), `TREADMILL` 164 (2), `SKATEBOARDING` 131 (3), `STATIONARY_BIKE` 86 (2; the cleaned labels had
    them as bike), `ON_A_BOAT` 18 (1), `WHEELCHAIR` 9 (1). `RIDING_AN_ANIMAL` and `USING_CRUTCHES` have 0 minutes. Not
    dropped: `STROLLING` 806 (574 → walk via `FIX_walking`, 193 → car), `LIMPING` 62.
  - **`MOTORBIKE`: 86 minutes, one user (33A85C34, Android, a dev user in fold 4; all minutes have acc + gyro).** The
    cleaned labels had them as car. They are kept out of training but labelled (`reason = check:MOTORBIKE`,
    `extrasensory.CHECK`). Each model is trained without them and we report what it predicts on them, using the fold-4
    held-out model so the user is unseen. That shows whether motorbike can be folded into road or needs its own class.
  - **`ON_A_TRAIN`:** 344 minutes in 5 users (`es_train_minutes.csv`). 342 are mapped to train (2 are train + walk
    conflicts), and all have acc + gyro. On 342 of the 344 the cleaned CSV says still, which confirms the brief.

    | user | platform | train min |
    |---|---|---|
    | 1DBB0F6F | android | 115 |
    | B7F9D634 | android | 97 |
    | 3600D531 | iphone | 69 |
    | 33A85C34 | android | 37 |
    | A76A5AF5 | iphone | 24 |
- **`label_source`** (ok minutes): history interface 2 = 260,310; active feedback 0/1 = 23,809; notification replies
  3–6 = 21,299.
- **Phone position** (usable minutes, `es_minutes_user_class_position.csv`):

  | | in_bag | in_hand | in_pocket | on_table | none reported |
  |---|---|---|---|---|---|
  | iPhone | 6,682 | 10,117 | 13,544 | 64,616 | 82,411 |
  | Android | 3,062 | 5,202 | 7,603 | 42,442 | 53,340 |

  - 47% of usable minutes have no position label.
  - **`PHONE_ON_TABLE`: 106,655 of 249,171 usable still minutes (43%).** The cap and its ratio will be set in step 2,
    on training users only.
  - Train has 112 on-table minutes.

#### Proposed splits (seed 0, revised 2026-10-25 with the new label rules)

**SHL User 1, 5 folds of whole days** (`proposed_shl_day_folds.csv`). Unit = calendar day: all recordings of a day
(`m` and plain) and all its positions go in the same fold. Every class is in every fold. Folds are chosen by random
search (20,000 draws) for the class shares closest to 1/5. Hips hours per fold:

| fold | days | still | walk | run | bike | car | bus | train | subway |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 15.0 | 9.5 | 3.8 | 10.6 | 15.8 | 9.3 | 12.2 | 8.0 |
| 1 | 15 | 9.8 | 9.5 | 3.1 | 8.5 | 14.8 | 10.7 | 10.4 | 7.9 |
| 2 | 15 | 10.0 | 9.2 | 2.5 | 7.0 | 12.1 | 10.6 | 10.9 | 7.3 |
| 3 | 15 | 10.2 | 11.5 | 3.9 | 11.1 | 8.3 | 10.7 | 8.7 | 7.5 |
| 4 | 15 | 12.2 | 12.1 | 4.4 | 11.1 | 17.2 | 9.9 | 14.1 | 9.6 |

The 3 `User1_4phones` days fall in folds by day like any other. Users 2 and 3 are final test only (experiment 06).

**ExtraSensory, 12 test users** (`proposed_es_users.csv`). How they were chosen:
- Candidates are the 57 users with a gyroscope.
- Among 200,000 random 12-subsets (seed 0) that meet the constraints, the one whose per-class test share is
  closest to 12/60 wins.
- Constraints:
  - each of bus, car and bike has ≥ 4 test users with ≥ 5 usable minutes. This is my reading of "≥ 3–4 users with
    bus, car and bike": per class, not all three in the same user.
  - iPhone count 6–7 (dataset ratio 34:26).
  - ≥ 1 train user in test and ≥ 1 in dev.
  - dev keeps ≥ 50% of every class.

| uuid | platform | still | walk | run | bike | car | bus | train |
|---|---|---|---|---|---|---|---|---|
| 098A72A5-E3E5-4F54-A152-BBDA0DF7B694 | iphone | 4789 | 187 | 33 | 120 | 142 | 0 | 0 |
| 1155FF54-63D3-4AB2-9863-8385D0BD0A13 | android | 2425 | 158 | 1 | 0 | 55 | 42 | 0 |
| 1538C99F-BA1E-4EFB-A949-6C7C47701B20 | iphone | 5764 | 662 | 0 | 0 | 0 | 0 | 0 |
| 3600D531-0C55-44A7-AE95-A7A38519464E | iphone | 3156 | 132 | 0 | 0 | 410 | 0 | 69 |
| 481F4DD2-7689-43B9-A2AA-C8772227162B | android | 4860 | 1133 | 0 | 0 | 16 | 43 | 0 |
| 59EEFAE0-DEB0-4FFF-9250-54D2A03D0CF2 | android | 6614 | 612 | 0 | 0 | 149 | 121 | 0 |
| 5EF64122-B513-46AE-BCF1-E62AAC285D2C | iphone | 2951 | 132 | 0 | 0 | 265 | 0 | 0 |
| 74B86067-5D4B-43CF-82CF-341B76BEA0F4 | android | 5964 | 15 | 88 | 692 | 81 | 0 | 0 |
| 96A358A0-FFF2-4239-B93E-C7425B901B47 | android | 1572 | 20 | 0 | 0 | 0 | 62 | 0 |
| C48CE857-A0DD-4DDB-BEA5-3A25449B2153 | iphone | 4018 | 268 | 0 | 0 | 79 | 46 | 0 |
| CA820D43-E5E2-42EF-9798-BE56F776370B | iphone | 5711 | 352 | 0 | 27 | 280 | 0 | 0 |
| D7D20E2E-FC78-405D-B346-DBD3FD8FC92B | iphone | 4061 | 265 | 37 | 60 | 433 | 24 | 0 |

Result:
- 7 iPhone and 5 Android.
- Test share of each class's usable minutes: still 20.8%, walk 18.4%, run 17.7%, bike 19.5%, car 17.6%, bus 19.9%,
  train 20.2%.
- Users with ≥ 5 min: bus 6, car 10, bike 4, run 3. Train: 1 user (3600D531, 69 min).
- Only 3 of the 12 users are the same as in the first proposal (2026-10-24, before the passenger rule). The bus counts
  changed, so the random search landed elsewhere.
- With only 4–6 users per class, test metrics swing with who is in the test set. Test metrics are therefore reported
  with a confidence interval from resampling test users (bootstrap over users) (approved 2026-10-25).

**ExtraSensory, 48 dev users in 5 folds by user.** Folds are dealt round-robin within platform, with the same
search. Usable minutes:

| fold | users | iPhone | still | walk | run | bike | car | bus | train |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 10 | 6 | 35,222 | 3,305 | 74 | 559 | 1,694 | 191 | 0 |
| 1 | 9 | 5 | 46,130 | 4,272 | 287 | 655 | 2,463 | 184 | 97 |
| 2 | 9 | 5 | 30,383 | 2,802 | 153 | 782 | 1,590 | 427 | 24 |
| 3 | 10 | 5 | 35,580 | 3,439 | 55 | 824 | 977 | 234 | 115 |
| 4 | 10 | 6 | 49,971 | 3,676 | 168 | 880 | 2,248 | 325 | 37 |

- Only 4 dev users have train, so **fold 0 has no train**. `splits.balanced_folds` requires a class in every fold
  only when it occurs in ≥ k groups (deviation from "every class in each fold", which the brief states for SHL).
- The 3 users without a gyroscope are dev users (folds 1, 3, 2) and contribute 0 acc + gyro minutes.
  They count only for the unlabelled pre-training partition if acc-only data is ever used there.

#### Decisions (2026-10-25, from Marc)
1. **Bus / train + passenger → bus / train**, as above. Implemented in `extrasensory.minute_classes` (`PASSENGER`).
2. **Motorbike** is kept out of training and used as a separate labelled check, as above. SHL has none, and in
   ExtraSensory it comes from one user, so a 9th class would learn that person and can't be tested on unseen users.
   The check tells us whether motorbike folds into road. It also fits experiment 04 (LIMU-BERT-X's "riding" class was
   scooters and motorcycles). **To ask the company:** do motorbikes matter for anti-fraud? If so, they should be
   labelled in the company's recordings. The other `DROP` labels stay dropped.
3. **Test-user criteria** as read (≥ 4 users per class for bus, car and bike; ~20% of each class). One train user is
   enough, because SHL users 2–3 are the real rail test. Test metrics get a confidence interval from resampling test
   users.
4. **Snippets with backwards timestamps (or a span over 40 s) are dropped** and counted (to implement in the step-2
   resampler).
5. **Logging fixed.**
   - `.env` was wrong: `REPO` pointed to `.../transport-mode`, and `$REPO_URI` was undefined. Also, python-dotenv only
     expands `${VAR}`, not `$VAR`. It now reads `REPO=${HOME}/Projects/waiis/transport-mode-detection`,
     `MLFLOW_TRACKING_URI=sqlite:///${REPO}/mlflow.db` and `MLFLOW_ARTIFACTS=${REPO}/mlartifacts`.
   - `.env` is not tracked: it is ignored by `.gitignore:159` and has never been in the history.
   - `src/tmd/tracking.py` now loads `.env` by path, not from the cwd (the cwd broke runs started from an experiment
     folder). It also wraps `REPO` / `ARTIFACTS` in `Path`, with the repo-root defaults as fallback.
   - 2026-10-28: replaced by `src/tmd/paths.py` (`PROJECT`, `MLFLOW_DB`, `MLARTIFACTS`). `.env` is no longer read,
     `python-dotenv` was removed, and `REPO` became `PROJECT` (also in `backfill_dataset_inputs.py`). The tracking
     store is the same.
   - Dummy run `tracking-check-2026-10-25` in MLflow experiment `00_smoke-test` (run `95205245c57b4659afc9f957807b32be`,
     started from this folder, dirty tree). Params, metrics, a figure artifact (`mlartifacts/00_smoke-test/...`), and
     the `git_commit` / `dirty` tags were all logged. `tracking.py export 00_smoke-test --out <scratchpad>` worked.

Splits approved 2026-10-25 and written to `splits/` (one row per SHL day; one row per ExtraSensory user; now in this folder's `splits/`).

### Step 2: features per 5 s window, R1 with cross-validation (2026-10-25) — **stop-point**

**Commands** (from this folder):

```
uv run python make_features.py --workers 6   # <data>/processed/{shl_u1,es_dev}_25hz_w5s_s2.5s_feat-v1.parquet (~6 min)
uv run python r1_cv.py --seed 0              # MLflow run R1_lgbm_feat-v1_seed0; preds and models in
                                             # <data>/processed/02_family-baseline_shl-es/r1_25hz_w5s_s2.5s_feat-v1_seed0/
```

MLflow run: `c615aedd62da40159a0b65113820aad1` (experiment `02_family-baseline_shl-es`, commit `7f435c5`,
**dirty**: the step-2 code below was not yet committed when it ran). Window counts are in
`results/features_counts.json`.

**Shared code added in `src/tmd`:**
- `resample.py`: everything goes onto a uniform 100 Hz grid, then `resample_poly(x, 1, 4, padtype="line")`
  (Kaiser FIR low-pass + decimation to 25 Hz).
  - SHL: per NaN-free stretch.
  - ExtraSensory: per snippet, with linear interpolation onto 100 Hz inside the snippet only, over the acc ∩ gyro
    time range. Acc units are decided per snippet (median |acc| < 3 → g → × 9.80665).
- `windows.py`: 5 s windows; label = majority; pure = one label throughout.
- `features.py`: features v1.
- `evaluate.py`: macro-F1, recall, confusion matrices, user bootstrap, latency.

**Windows:**
- 5 s = 125 samples. The 2.5 s stride is 62.5 samples, so starts are ⌊k · 62.5⌋ (strides alternate 62 / 63).
- Windows whose majority label is Null are removed.
- SHL User 1 (dev days only): 635,987 windows, 99.5% pure. Hips 562,925; Hand / Torso / Bag about 24,350 each
  (3 days). Training uses pure windows; evaluation uses all of them.
- ExtraSensory dev users (48): 1,364,126 windows from 227,128 minutes (labelled `ok` + motorbike check).

**Features v1** (`tmd.features`): 145 = 5 channels × 29.
- Channels, all independent of phone orientation: |a|; vertical a (projected on gravity); horizontal |a|; |ω|; jerk
  |da/dt|. Gravity is a 2nd-order Butterworth low-pass at 0.3 Hz, run forward-backward **inside each window**, so a
  window's features depend on that window alone.
- Per channel:
  - mean, std, min, max, p5/25/50/75/95, IQR, skew, kurtosis, mean |diff|, mean-crossing rate;
  - ACF peak and its lag over 0.2–2 s; Hjorth mobility and complexity;
  - Welch PSD (64-sample segments, 50% overlap): log and relative energy in 0–1 / 1–3 / 3–6 / 6–12.5 Hz, dominant
    frequency, spectral centroid, normalised spectral entropy.
- A test rotating a noisy synthetic window changes every feature by < 4e-6 relative.
- Python cost: 0.04 ms per window in batch.

**Reference: arXiv 2407.11048** ("Signal Sleuths", SHL 2024; the paper itself doesn't say it won). Their setup:
- 100 Hz, non-overlapping 5 s windows.
- acc, gyro, mag; per modality, the magnitude of the raw signal, of its 1st and 2nd gradient, and of its integral;
  raw axes pooled across x, y, z as mean/std or sorted.
- About 70 features per signal (980 in total): 17 PSD/DCT bands from 0.1 to 50 Hz on z-normalised signals, centroid
  and bandwidth, ACF stats, zero-crossing rate and slope sign change, entropies, Hjorth, Katz fractal dimension.
- Magnitude-based time-domain features (quantiles, mean, std) are deliberately left out.
- CatBoost (1000 iterations, balanced class weights), no smoothing.

What I took from it: the jerk-magnitude channel and Hjorth / spectral centroid. Differences, on purpose:
- 25 Hz caps the bands at 12.5 Hz.
- We keep magnitude statistics. They separate still from vehicles, and 02's test is whether that holds across datasets.
- We use vertical/horizontal projections (brief) instead of gradient/integral magnitudes.
- No magnetometer (experiment 03).

**Model and protocol (R1).**
- LightGBM multiclass on the 8 classes:
  - learning rate 0.1, 31 leaves, min_child_samples 50, feature/bagging fraction 0.8, L2 1;
  - class weights balanced on the training windows; seed 0.
- Family probability = sum of the class probabilities; family prediction = argmax of that.
- Per fold k:
  - number of rounds: early stopping (patience 30, multi-logloss) on day-fold (k+1) mod 5, trained on the other 3
    training folds. Only training data is used.
  - Then a refit on all 4 training folds with that number of rounds.
- No hyperparameter search; the values above are fixed defaults.
- ExtraSensory dev users are never trained on in R1, so every one of them is cross-dataset. They are scored with
  the **mean of the 5 fold models' probabilities**. The 95% CI comes from resampling the 48 users (1000 draws).
- Macro-F1 averages only the classes present in the evaluated data (subway is absent from ExtraSensory). Predicting
  an absent class still counts as an error.

#### R1 results

| eval set | n windows | macro-F1 8 classes | macro-F1 family |
|---|---|---|---|
| SHL User 1, out-of-fold, all positions | 635,987 | **0.755** (folds 0.754 ± 0.016) | **0.842** (folds 0.841 ± 0.019) |
| SHL, pure windows only | 632,869 | 0.756 | 0.843 |
| SHL Hips (76 days) | 562,925 | 0.776 | 0.859 |
| SHL Hips, the 3 four-phone days only | 24,358 | 0.725 | 0.827 |
| SHL Torso (3 days) | 24,352 | 0.597 | 0.726 |
| SHL Bag (3 days) | 24,356 | 0.692 | 0.807 |
| SHL Hand (3 days) | 24,354 | 0.510 | 0.590 |
| **ExtraSensory dev users (cross-dataset)** | 1,363,710 | **0.259** (95% CI over users 0.223–0.307) | **0.320** (0.286–0.350) |
| ES iPhone | 989,746 | 0.284 | 0.352 |
| ES Android | 373,964 | 0.225 | 0.276 |
| ES in pocket | 108,052 | 0.311 | 0.399 |
| ES in bag | 58,006 | 0.282 | 0.390 |
| ES in hand | 71,771 | 0.320 | 0.285 |
| ES on table | 499,760 | 0.092 | 0.101 |
| ES no position | 626,121 | 0.260 | 0.328 |

The single fold models on ExtraSensory give family 0.320 ± 0.017, the same as the ensemble.

**Per-class recall:**

| | still | walk | run | bike | car | bus | train | subway |
|---|---|---|---|---|---|---|---|---|
| SHL (out-of-fold) | 0.835 | 0.946 | 0.979 | 0.929 | 0.748 | 0.565 | 0.574 | 0.472 |
| ExtraSensory | 0.273 | 0.430 | 0.087 | 0.579 | 0.367 | 0.245 | 0.352 (1,051 windows / 342 min, 4 users) | – |

| family recall | still | foot | bike | road | rail |
|---|---|---|---|---|---|
| SHL | 0.782 | 0.957 | 0.925 | 0.798 | 0.757 |
| ExtraSensory | 0.188 | 0.420 | 0.569 | 0.611 | 0.660 |

**Confusion matrices:**
- MLflow figures `cm_{shl,es}_{class,family}.png`; counts in `tables.json`.
- SHL: rail ↔ road (19k rail → road, 31k road → rail) and rail ↔ still (19k rail → still, 15k still → rail).
- SHL within-family errors: bus is recalled at 0.57 (rest mostly → car); train ↔ subway.
- **Rail false positives on ExtraSensory are the main failure:**
  - 64.8% of still windows and 23.7% of road windows are predicted rail.
  - Family confusion of ES still (1.17M windows): rail 760k, road 164k, still 220k, foot 22k, bike 7k.

**Why ExtraSensory still fails (diagnosis, not tuned).** Units and resampling look right:
- ES train and car feature medians match SHL train and car (acc std 0.172 vs 0.175 for train, 0.386 vs 0.509 for
  car; |a| mean ≈ 9.81 everywhere).
- ExtraSensory still is a different physical state. 42% of its still windows have the phone on a table, and those
  are far quieter than any SHL still window: median acc std 0.015 vs 0.029 m/s², |ω| 0.003 vs 0.024 rad/s, jerk 0.39
  vs 1.15. SHL's phones are always on the body, so a resting phone is outside the training range, and the trees send
  it to rail or road.
- ES still in a pocket is close to SHL still (acc std 0.056, |ω| 0.046), and its family F1 is the best (0.40). Still
  recall there is only 0.40, though: pocket still in ES is livelier than SHL still.
- This is the cross-dataset gap R2 (step 4) is meant to measure. It also shows that SHL alone doesn't cover a phone
  resting on a surface.

**Motorbike check** (86 minutes, 416 windows, one user, never trained on):
- Family: road 269, rail 71, foot 41, bike 28, still 7.
- Classes: car 180, bus 91, subway 48, walk 41, bike 34, train 13, still 9.
- So R1 mostly calls it road (65%). This is one model and one user, and that user's still is also misread, so it's
  only a first hint that motorbike folds into road.

**Size and latency** (fold models; ONNX in step 5):
- Model size: LightGBM text model 6.5 MB on average (3.1–12.9 MB), 1,851 trees on average (8 classes × 111–458
  rounds). **This is over the 2 MB ceiling** (DECISIONS 2026-10-23).
- Latency per window, single thread, after 50 warm-up calls, 500 calls (Apple Silicon, Python):
  - features: median 2.10 ms, p95 2.14 ms (numpy/scipy per-call overhead; 0.04 ms per window in batch);
  - model: median 0.045 ms, p95 0.048 ms.

#### Deviations (step 2)
- **Extra ExtraSensory snippet drops**, besides the approved ones (time not increasing: 759; span over 40 s: 122):
  - hole over 0.5 s inside a snippet: 1,482;
  - acc ∩ gyro overlap under 5 s: 418;
  - acc or gyro file missing: 16,261 (mostly the 3 no-gyro users).

  Of the 246,170 labelled dev minutes, 227,128 remain (`resample.es_snippet_to_25hz`).
- **The top band is 6–12.5 Hz**, not 6–12, so it runs to Nyquist.
- **One seed** (0). The fold spread (± 0.02) is reported instead of a seed spread.
- **Fixed LightGBM hyperparameters** (no search). Only the number of rounds comes from the inner fold.
- **PHONE_ON_TABLE is not subsampled**: R1 doesn't train on ExtraSensory, and evaluation keeps every minute. The cap
  applies from step 4 (R2).

#### Open questions (step 2)
1. ~~Model size~~ **Answered 2026-10-25 (Marc):** the model runs on an **edge server**, so 6.5 MB is well within
   bounds. No shrinking; size and latency are still reported. CLAUDE.md, the block-1 brief ("phone model") and the
   draft DECISIONS entry of 2026-10-23 (UniMTS "too large" against a 2 MB ceiling) predate this and are not yet
   updated.
2. **Python feature latency** (2.1 ms) is mostly call overhead. Step 5 measures it again (features and model
   separately).
3. **SHL Hand is weak** (family 0.59). It has only 3 days, in folds 1, 2 and 4, so each held-out Hand day is predicted
   by a model trained on the other 2 Hand days. Step 3 keeps reporting it separately.

### Step 2b: ExtraSensory label review per class (2026-10-25) — **v2 approved 2026-10-26**

Asked by Marc: a phone lying on a table should not make up 42% of still. Commands (from this folder):

```
uv run python es_label_review.py      # results/audit/es_label_review.csv: class x source label x position x label_source
uv run python rescore_es_labels.py    # rules v2: results/audit/es_label_rules_v2_counts.csv; MLflow eval-only run
```

**What each class is made of** (all 60 users, `ok` minutes under the rules of step 1; `es_label_review.csv`).
"Resting" = `PHONE_ON_TABLE` or original `PHONE_AWAY_FROM_ME`: the phone is not carried.

| class | minutes | resting | position not reported | notes |
|---|---|---|---|---|
| still | 264,158 | 44% | 41% | SLEEPING 83,055 (31%): 93% of them on a table or unknown position, 5% away (a nightstand) |
| walk | 22,173 | 932 | 51% | resting walk = the person walks, the phone stays behind |
| run | 908 | 146 (16%) | 75% | same |
| bike | 4,878 | 0 | 70% | |
| car | 11,166 | 15.5% | 54% | phone on a seat or dashboard still rides along |
| bus | 1,793 | 0 | 39% | |
| train | 342 | 33% (112 on a table) | 12% | |

`label_source` = 2 (history interface, labelled afterwards) for 86% of still and 54–82% of the moving classes (train 54%, the rest 69–82%).

**Rules v2** (implemented in `tmd.extrasensory`):
- `SLEEPING` still minutes → `drop:SLEEPING`.
- Walk / run / bike with a resting phone → `drop:not_carried`.
- Vehicles keep resting minutes.
- Still: resting minutes are capped per user at 20% of that user's still minutes (`cap_resting_still`, seed 0), in
  training and in evaluation. Uncapped resting still is kept as a diagnostic row.
- **Rules v1 (step 1) are no longer in the current code:** v2 replaced them in place in `tmd.extrasensory`. Rebuild
  v1 from commit `1f52628` (`git show 1f52628:src/tmd/extrasensory.py`).

Counts, all users (`es_label_rules_v2_counts.csv`):
- still 264,158 → 181,103 minutes after dropping SLEEPING → 118,604 after the cap. The resting share is 13.9% (the
  cap is per user, so users with few carried still minutes lose more).
- walk −932 and run −146 (not carried).
- Other classes unchanged.

**R1 rescored on the ExtraSensory dev users under v2** (MLflow `32b5199f1bed4fb7bf62ea7493ac9799`, eval-only, same
predictions as `c615aedd…`, dirty):

| ES dev | macro-F1 8 classes | macro-F1 family | still recall (family) | rail FP on still | rail FP on road |
|---|---|---|---|---|---|
| step 1 rules | 0.259 | 0.320 (0.286–0.350) | 0.188 | 64.8% | 23.7% |
| rules v2 | 0.295 (0.254–0.353) | 0.366 (0.328–0.395) | 0.236 | 57.6% | 23.7% |
| v2 + still only with a reported carried position (variant B) | – | 0.418 | 0.283 | – | – |

**What v2 does not fix.** Still is still mostly predicted rail. Two reasons, both in the dev windows:
- **Unknown position is often a resting phone too.** 49% of still windows with no position label are as quiet as a
  phone on a table (median |ω| < 0.006 rad/s). For bag it's 57%. In SHL only 6.8% of still windows are that quiet
  (≤ 1.8% for any other class). Quiet still windows are predicted rail 77% of the time.
- **Even carried still differs from SHL still.** With the phone in a pocket (not resting), only 40% are predicted
  still and 40% rail. That is a cross-dataset gap no label rule removes. R2 (step 4) adds ExtraSensory to training.

**Variant B** (not implemented, for decision): still only when a carried position (pocket / hand / bag) is reported,
plus resting minutes capped at 20%. Still minutes with no position label are left out of training and reported as
their own row.
- Dev still drops to 31,207 minutes, which is still more than walk.
- Family F1 rises to 0.418 (still recall 0.283).
- Cost: 72% of still minutes, and a still class built only from users who report positions.

**Decision (2026-10-26, Marc):**
- v2 approved with the 20% cap. Training uses v2, not B: resting phones are real still states in production.
- Every ExtraSensory table reports still twice, "still (v2)" and "still, carried phone only" (B's subset). The main
  macro-F1 uses v2.
- Step 4 (R2) runs before step 3.

The splits are not affected: they were chosen on bus / car / bike / train / platform, which v2 leaves unchanged.

### Step 4 (run before step 3, as decided 2026-10-26): R2 = SHL + ExtraSensory dev users (2026-10-26) — **stop-point**

**Commands** (from this folder):

```
uv run python r2_cv.py --seed 0     # MLflow runs R2_lgbm_feat-v1_seed0 and R1_lgbm_feat-v1_seed0_es-v2-foldmatched (~40 min)
```

- MLflow: R2 `d668fa2e279d4776a65bc7e8f2f4977a`; R1 fold-matched reference `e37b328555da4c0e8d40b94b1450629c`. Both
  at commit `cb8a024`, **dirty** (`eval_tables.py` and `r2_cv.py` were not yet committed).
- Predictions and models: `<data>/processed/02_family-baseline_shl-es/r2_25hz_w5s_s2.5s_feat-v1_seed0/`.

**Protocol.**
- **Paired folds:** fold k holds out SHL User 1 day-fold k **and** ExtraSensory user-fold k (`splits/`). Every SHL
  window and every ExtraSensory dev window is predicted once, by a model that saw neither that day nor that user.
- **Training data:** SHL pure windows, plus ExtraSensory windows under labels v2 with resting still capped at 20% per
  user (seed 0). Motorbike minutes are never used.
  - Per fold: about 0.48–0.53M SHL + 0.54–0.64M ES windows.
  - ES windows usable under v2 + cap: 726,038 of 1,364,126 (sleeping drops 369k).
- **Model:** the same LightGBM and features as R1. Class weights are balanced over the combined training set.
  Rounds come from early stopping on the inner paired fold (k+1) mod 5, then a refit on all training folds.
- **Comparison (R1 fold-matched):** the saved R1 fold-k model is scored on ES user-fold k. Same ES windows, same
  tables, no retraining. R1's SHL numbers are from run `c615aedd…` (unchanged).
- **Not-carried / unknown / carried** = the ES position labels: resting (`PHONE_ON_TABLE` or `PHONE_AWAY_FROM_ME`),
  none reported, or pocket / hand / bag.

#### Macro-F1 (8 classes / family)

| eval set | n windows | R1 | R2 |
|---|---|---|---|
| **SHL User 1, out-of-fold, all positions** | 635,987 | 0.755 / **0.842** | 0.723 / **0.784** |
| SHL Hips | 562,925 | 0.776 / 0.859 | 0.741 / 0.797 |
| SHL Hand | 24,354 | 0.510 / 0.590 | 0.504 / 0.593 |
| SHL Torso | 24,352 | 0.597 / 0.726 | 0.587 / 0.677 |
| SHL Bag | 24,356 | 0.692 / 0.807 | 0.652 / 0.755 |
| **ES dev, labels v2 (main)** | 726,038 | 0.288 (0.247–0.348) / **0.362** (0.324–0.393) | 0.400 (0.349–0.451) / **0.489** (0.447–0.521) |
| ES dev, still carried phone only (B subset) | 328,963 | 0.332 / 0.425 (0.390–0.454) | 0.423 / 0.513 (0.478–0.542) |
| ES carried (pocket / hand / bag) | 219,180 | 0.362 / 0.397 | 0.395 / 0.490 |
| ES not carried (resting, capped) | 87,081 | 0.224 / 0.262 | 0.523 / 0.531 |
| ES position unknown | 419,777 | 0.269 / 0.342 | 0.393 / 0.478 |
| ES in pocket | 106,685 | 0.307 / 0.398 | 0.357 / 0.465 |
| ES in hand | 63,594 | 0.327 / 0.299 | 0.334 / 0.387 |
| ES in bag | 52,195 | 0.268 / 0.386 | 0.402 / 0.534 |
| ES on table | 77,562 | 0.211 / 0.248 | 0.542 / 0.542 |
| ES iPhone | 517,747 | 0.314 / 0.395 | 0.427 / 0.517 |
| ES Android | 208,291 | 0.244 / 0.299 | 0.346 / 0.421 |

ES intervals are 95% CIs from resampling the 48 dev users. The R1 column for ES is the fold-matched reference; it is
close to the ensemble numbers of step 2b (family 0.366).

#### Per-class recall

| | still (v2) | still, carried only | walk | run | bike | car | bus | train | subway |
|---|---|---|---|---|---|---|---|---|---|
| SHL, R1 | 0.835 | – | 0.946 | 0.979 | 0.929 | 0.748 | 0.565 | 0.574 | 0.472 |
| SHL, R2 | **0.556** | – | 0.950 | 0.982 | 0.925 | 0.719 | 0.563 | 0.582 | 0.508 |
| ES, R1 | 0.305 | 0.396 | 0.441 | 0.107 | 0.579 | 0.368 | 0.247 | 0.328 | – |
| ES, R2 | **0.787** | **0.706** | 0.564 | 0.190 | 0.603 | 0.515 | 0.184 | 0.221 | – |

| family recall | still (v2) | still, carried only | foot | bike | road | rail |
|---|---|---|---|---|---|---|
| SHL, R1 | 0.782 | – | 0.957 | 0.925 | 0.798 | 0.757 |
| SHL, R2 | **0.429** | – | 0.960 | 0.919 | 0.792 | 0.792 |
| ES, R1 | 0.249 | 0.334 | 0.433 | 0.569 | 0.608 | 0.616 |
| ES, R2 | **0.754** | **0.662** | 0.559 | 0.595 | 0.605 | **0.414** |

- Counts: ES still (v2) 541,556 windows, of which carried 144,481.
- ES rail is 1,051 windows (342 minutes) from 4 dev users, so its recall rests on very little data.

#### ES still predicted as rail, by carry state (family level)

| carry state | windows | R1 → rail | R1 → still | R2 → rail | R2 → still |
|---|---|---|---|---|---|
| carried | 144,481 | 43.2% | 33.4% | 9.8% | 66.2% |
| not carried (capped) | 77,352 | 61.4% | 22.4% | 3.9% | 87.1% |
| unknown | 319,723 | 58.5% | 21.6% | 7.0% | 76.7% |

Rail false positives over all ES still: 54.8% → 7.3%. On ES road: 23.8% → 9.2%.

#### Confusion matrices (R2, family; counts)
MLflow figures `cm_{shl,es}_{class,family}.png`; counts in `tables.json`.

| true \ pred | still | foot | bike | road | rail |
|---|---|---|---|---|---|
| SHL still | 39,854 | 2,065 | 457 | 5,710 | **44,722** |
| SHL rail | 9,825 | 2,320 | 602 | 20,707 | 127,415 |
| ES still | 408,124 | 48,472 | 5,722 | 39,751 | 39,487 |
| ES rail | **239** | 134 | 11 | 232 | 435 |

In R1, SHL still went 15,381 → rail; in R2 it is 44,722.

#### Reading

- **Adding ExtraSensory fixes ExtraSensory still:**
  - recall 0.25 → 0.75 (carried only: 0.33 → 0.66);
  - still → rail 55% → 7%;
  - ES family macro-F1 0.362 → 0.489, and the user-bootstrap CIs don't overlap (0.324–0.393 vs 0.447–0.521).
  - Walk, bike and car also gain on ES (recall +0.12, +0.02, +0.15).
- **It costs SHL still and ES rail:**
  - SHL still recall falls 0.835 → 0.556. Nearly half of SHL still now goes to rail, and SHL family F1 drops
    0.842 → 0.784. SHL Hand is unchanged (0.59), Torso and Bag drop.
  - ES rail recall falls 0.62 → 0.41, mostly to still.
- **Interpretation (not tested further):** with no context, one 5 s window of a still phone and one of a smooth
  train look alike. The two datasets put the still / rail boundary in different places: ES still is quieter than SHL
  still (step 2), and ES has ~1k rail windows against ~550k still. Balanced class weights don't change the fact that
  ES's quiet still now owns that region of feature space. Step 3's 60 s context and HMM are the natural place for
  still vs rail (stops, accelerations); per-dataset weighting is another lever.
- **ES bus gets worse** (0.25 → 0.18): bus goes to still and car. Bus has 1.3k ES dev minutes.
- **Motorbike** (fold 4 model, user unseen): R1 road 316 / foot 43 / bike 30 / still 16 / rail 11; R2 road 237 /
  foot 77 / still 68 / bike 34. Road stays the majority (57% under R2; classes car 172, walk 81, still 78, bus 46,
  bike 39).

**Size and inference time** (R2 fold models):
- Model size: 6.95 MB on average (2.7–16.9 MB), 1,960 trees on average (rounds 94–599).
- Single thread, after warm-up, fold-0 model:
  - features: median 2.14 ms, p95 2.22 ms;
  - model: median 0.028 ms, p95 0.031 ms.
- **Inference time per window** (metric c: features + model, single thread, 50 warm-up + 500 timed calls, mean
  over the 5 fold models; `bench_inference.py`, logged into the R1 and R2 runs on 2026-10-26):

  | | median | p95 | model size (mean) |
  |---|---|---|---|
  | R1 | 2.11 ms | 2.30 ms | 6.55 MB |
  | R2 | 2.15 ms | 2.75 ms | 6.95 MB |

  Python feature extraction is ≈ 2.1 ms of it. The model takes 0.03–0.05 ms, even at 4,792 trees.

**Metrics from now on (2026-10-26, Marc):** (a) macro-F1, (b) confusion matrix and per-class recall, (c) inference
time per window (features + model, single thread, median and p95), (d) model size. CLAUDE.md and the brief were
updated, and `tmd.evaluate.inference_latency` / `model_size_bytes` compute (c) and (d).

**Deviations (step 4):**
- Step 4 ran before step 3, as decided.
- The ES training data are capped per user, so the not-carried still share stays ≤ 20%.
- ES and SHL folds are paired by index (fold k with fold k). The pairing is arbitrary but fixed.
- One seed.

**Open questions (step 4):**
1. The still ↔ rail trade-off between datasets. Should step 3 (context + HMM) be run on R2 as the base, and be judged
   on SHL still and ES rail as well as macro-F1? Or should dataset weighting be tried first?
2. ~~**The brief's still labels conflict with v2.** The current brief (your edit, uncommitted) maps only
   `SITTING` / `OR_standing` → still. `LYING_DOWN` is gone, which v2 keeps. Which is right? It affects about 23k
   non-sleeping lying minutes (all users).~~ **Closed:** decided to keep `LYING_DOWN` in still, as in v2; the brief
   was updated in commit `02e321a` (its Labels item 7).

### ExtraSensory-only model (before step 3, asked 2026-10-26) — reported with step 3

**Command:** `uv run python r2_cv.py --seed 0 --setup ES`
- MLflow `c3800bfa1eb240c0b46e336ec8621668` (commit `30aeaeb`, dirty: the `--setup` option and the new still/rail
  metrics in `eval_tables.py` were not yet committed).
- Same paired folds and tables as R2, but trained on ES dev user-folds only (labels v2 + cap). SHL day-fold k is
  scored by the fold-k model, so for this model SHL is an unseen dataset.

| | R1 (SHL only) | ES only | R2 (SHL + ES) |
|---|---|---|---|
| **SHL macro-F1, 8 classes / family** | 0.755 / 0.842 | 0.480 / 0.602 | 0.723 / 0.784 |
| **ES macro-F1, 8 classes / family (v2, main)** | 0.288 / 0.362 | 0.331 / 0.403 (CI 0.361–0.438) | 0.400 / 0.489 (CI 0.447–0.521) |
| ES, still carried only, family | 0.425 | 0.443 | 0.513 |
| SHL still recall (family) | 0.782 | 0.554 | 0.429 |
| SHL rail recall (family) | 0.757 | 0.172 | 0.792 |
| **SHL still → rail** | 16.6% | 6.3% | 48.2% |
| SHL rail → still | 11.7% | 11.3% | 6.1% |
| ES still (v2) recall (family) | 0.249 | 0.583 | 0.754 |
| ES still, carried only, recall (family) | 0.334 | 0.423 | 0.662 |
| ES rail recall (family; 1,051 windows, 4 users) | 0.616 | **0.009** | 0.414 |
| **ES still → rail** | 54.8% | 2.9% | 7.3% |
| ES rail → still | 6.8% | 21.0% | 22.7% |
| model size (mean of folds) | 6.55 MB | 0.35 MB | 6.95 MB |
| inference per window, median / p95 | 2.11 / 2.30 ms | 2.04 / 2.11 ms | 2.15 / 2.75 ms |

R1 and R2 ES columns are the fold-matched / paired-fold numbers of step 4.

ES-only, more detail:
- Per class (ES recall): still 0.674, walk 0.528, run 0.242, bike 0.592, car 0.507, bus 0.270, train 0.029.
- On SHL: still 0.697, walk 0.733, run 0.981, bike 0.677, car 0.432, bus 0.417, train 0.285, subway 0.000.
- By carry state (ES family F1): carried 0.397, not carried 0.447, unknown 0.401.
- ES still → rail: carried 3.8%, not carried 3.5%, unknown 2.3%.
- Motorbike: road 335, foot 45, still 16, rail 12, bike 8.

**Reading:**
- **ES alone is not enough, even on ES.** ES family F1 is 0.403 (CI 0.361–0.438), below R2 (0.489, CI 0.447–0.521).
  It barely learns rail: train comes from 4 dev users, and in each fold 1–2 of them are held out.
- **Without SHL, still is rarely called rail on either dataset** (SHL 6%, ES 3%). Rail is almost never predicted.
  The high still → rail of R2 on SHL (48%) therefore comes from combining the two still definitions with SHL's rail.
- **SHL as an unseen dataset for the ES model:** 0.602 family. Train and subway are almost never recognised (SHL rail
  recall 0.17).
- **Caveat: early stopping stopped ES-only very early**, at 7–31 rounds (56–248 trees; R2: 94–599 rounds). The inner
  fold is a handful of held-out users with balanced class weights, so the validation loss turns up after a few rounds.
  ES-only may be underfit by this rule. This was not tuned further: no test-informed change.

### Step 3 (on R2): 60 s context features + HMM (2026-10-26) — **stop-point**

**Commands** (from this folder):

```
uv run python make_features.py --workers 4 --all   # all-window tables *_all.parquet (incl. SHL Null, ES unlabelled/dropped minutes)
uv run python step3_context_hmm.py --seed 0          # 6 MLflow runs step3_{base,base+fwd,base+vit,ctx,ctx+fwd,ctx+vit}_feat-v1_seed0
```

- MLflow runs: base `7db74d8c822b453f9120a17b60cb9d27`, base+fwd `cfdf3eb45283465fbe7b5de25b62ac17`, base+vit
  `3b5fdbda11ef49ddac5f30521b75c715`, ctx `3a47f096d01a4b078c4590fd73870fe9`, ctx+fwd
  `4de65e9be527484f8928d0a15f8272e9`, ctx+vit `adc06318617746a7860aac50db377029`.
- All runs are at commit `30aeaeb` and **dirty** (step-3 code uncommitted when run).
- Outputs: `<data>/processed/02_family-baseline_shl-es/step3_25hz_w5s_s2.5s_feat-v1_seed0/`.

**Protocol** (code: `context.py` in this folder, moved from `tmd.context` on 2026-10-28; `tmd.hmm`):
- **All windows.** Context and HMM run over **every window the phone would have produced**, so the context is not
  filtered by labels. Evaluation uses exactly the R2 windows.
  - SHL: 1,027,556 windows including Null.
  - ES dev: 1,716,072 windows from 284,996 minutes, labelled or not.
  - `make_features.py --all` drops the same broken snippets as before.
- **Context.** For each window, the mean and std over (t − 60 s, t] (causal) of 65 window features: 13 per channel
  (mean, std, p50, p95, mean-crossing rate, ACF peak, Hjorth mobility, dominant frequency, spectral entropy, 4 band
  log-energies). Plus `ctx_n` (windows in the context) and `ctx_span_s`. That's 132 extra features, 277 in total.
  - Sequences: SHL = recording × position; ES = user, with minutes chained by time.
  - ES window time = minute timestamp + 2.5 s × window index. The snippet's offset within the minute is unknown.
- **Context coverage.** SHL: every evaluated window has a full 60 s context (24 windows). ES: on average **6.4
  windows** in context, and **77.5%** of evaluated windows reach back into the previous minute (chained minutes).
  The rest only see the earlier windows of their own snippet.
- **ctx model.** LightGBM retrained on 277 features. Same training rows (SHL pure + ES v2 capped), paired folds,
  inner early stopping and class weights as R2. Rounds: 42–205.
- **HMM.**
  - States = 8 classes. Emissions = classifier posteriors (balanced class weights, so treated as scaled likelihoods
    under a uniform prior).
  - Transition matrix per 2.5 s step, per fold, from **SHL training days only**: consecutive labelled windows,
    +1 pseudo-count. ES labels are one per minute, so within a minute all ES transitions are self-transitions.
  - A gap of k steps uses A^k; a gap over 60 s restarts the chain. ES's ~40 s gaps between snippets are bridged with
    A^16.
  - Two decoders: **forward filtering** (causal, per window, deployable as a stream) and **Viterbi** (offline,
    whole sequence).
- **Inference time (metric c).** Features + streaming 60 s context (`StreamingContext`) + model, plus one forward
  step for +fwd. Single thread, after warm-up, fold-0 model. Viterbi is offline and has no per-window cost of its own.

#### Results (paired folds; ES = labels v2, main)

| config | SHL macro-F1 (8 / family) | ES macro-F1 (8 / family) [95% CI over users, family] | ES still carried only (family F1) | inference p50 / p95 | size |
|---|---|---|---|---|---|
| base (= R2) | 0.723 / 0.784 | 0.400 / 0.489 [0.447–0.521] | 0.513 | 2.22 / 2.46 ms | 6.95 MB |
| base + forward | 0.837 / 0.879 | 0.476 / 0.566 [0.523–0.602] | 0.589 | 2.15 / 2.20 ms | 6.95 MB |
| base + Viterbi | 0.853 / 0.890 | 0.494 / 0.583 [0.541–0.620] | 0.605 | (offline) | 6.95 MB |
| ctx | 0.851 / 0.898 | 0.430 / 0.553 [0.506–0.604] | 0.569 | 2.14 / 2.18 ms | 3.03 MB |
| ctx + forward | 0.875 / 0.917 | 0.447 / 0.582 [0.530–0.654] | 0.590 | 2.14 / 2.21 ms | 3.03 MB |
| ctx + Viterbi | **0.891 / 0.930** | 0.459 / **0.599** [0.549–0.673] | 0.603 | (offline) | 3.03 MB |

Inference time is dominated by Python feature extraction (≈ 2.1 ms). Streaming context and one forward step add
< 0.1 ms (the differences between rows are timing noise).

**Still and rail, separately per dataset** (family level). SHL is the main reference for still vs rail; ES rail is
1,051 windows from 4 self-reported users.

| config | SHL still recall | SHL rail recall | SHL still → rail | SHL rail → still | ES still recall (v2 / carried) | ES rail recall | ES still → rail (carried / not / unknown) | ES rail → still |
|---|---|---|---|---|---|---|---|---|
| base (= R2) | 0.429 | 0.792 | 48.2% | 6.1% | 0.754 / 0.662 | 0.414 | 7.3% (9.8 / 3.9 / 7.0) | 22.7% |
| base + forward | 0.633 | 0.863 | 32.5% | 5.9% | 0.852 / 0.801 | 0.366 | 2.1% (2.0 / 1.0 / 2.4) | 24.7% |
| base + Viterbi | 0.649 | 0.876 | 32.0% | 5.9% | 0.863 / 0.813 | 0.368 | 1.7% (1.5 / 0.8 / 1.9) | 24.6% |
| ctx | 0.698 | 0.901 | 26.0% | 2.9% | 0.850 / 0.778 | **0.041** | 0.0% | 39.5% |
| ctx + forward | 0.774 | 0.907 | 17.7% | 2.9% | 0.871 / 0.803 | **0.059** | 0.0% | 42.7% |
| ctx + Viterbi | **0.804** | **0.917** | **15.8%** | 2.8% | 0.880 / 0.812 | **0.055** | 0.0% | 39.3% |

For reference: R1 (SHL only, no context) had SHL still 0.782, rail 0.757, still → rail 16.6%.

**Per-class recall (8 classes):**

| config | data | still | walk | run | bike | car | bus | train | subway |
|---|---|---|---|---|---|---|---|---|---|
| base | SHL | 0.556 | 0.950 | 0.982 | 0.925 | 0.719 | 0.563 | 0.582 | 0.508 |
| base + Viterbi | SHL | 0.649 | 0.988 | 0.991 | 0.975 | 0.900 | 0.845 | 0.768 | 0.718 |
| ctx + Viterbi | SHL | 0.804 | 0.982 | 0.996 | 0.988 | 0.900 | 0.841 | 0.862 | 0.766 |
| base | ES | 0.787 | 0.564 | 0.190 | 0.603 | 0.515 | 0.184 | 0.221 | – |
| base + Viterbi | ES | 0.863 | 0.633 | 0.175 | 0.674 | 0.660 | 0.242 | 0.195 | – |
| ctx + Viterbi | ES | 0.880 | 0.668 | 0.101 | 0.661 | 0.666 | **0.061** | **0.055** | – |

The other configs are in the MLflow runs (`tables.json`, `cm_*.png`).

**Per position (family F1):**
- ctx + Viterbi on SHL: Hips 0.946, Hand 0.618, Torso 0.856, Bag 0.930.
- ctx + Viterbi on ES: carried 0.616, not carried 0.594, unknown 0.565; iPhone 0.603, Android 0.531.
- base + Viterbi on SHL: Hips 0.904, Hand 0.722, Torso 0.777, Bag 0.856.

**Motorbike** (fold 4, user unseen): road 222–246 of 416 windows in every config (53–59%), then foot, still.

#### Reading

1. **The HMM helps everywhere and costs nothing per window.**
   - Forward filtering, which is causal, gives +0.10 family F1 on SHL and +0.08 on ES over base. Viterbi adds another
     ~0.01–0.02.
   - SHL bus recall 0.56 → 0.85 and train 0.58 → 0.77 (base → base + Viterbi).
   - ES still → rail 7% → 2%.
2. **60 s context helps SHL most, and reduces the still/rail confusion there:**
   - SHL still → rail falls from 48% (base) to 26% (ctx) and 16% (ctx + Viterbi), back to R1's level (17%). SHL
     still recall is 0.80 and rail 0.92 for ctx + Viterbi.
3. **But on ES the context model has learned a dataset shortcut.**
   - `ctx_n`, the number of windows in the context, is the **most important feature** of the ctx models (11% of
     the gain, rank 1 of 277).
   - It is always 24 on SHL and ~6 on ES, so it tells the model which dataset a window comes from. ES has almost no
     rail and little bus.
   - So under ctx, ES rail recall drops 0.41 → 0.04–0.06 (mostly → still and road), ES bus 0.18–0.24 → 0.06–0.10,
     and ES still → rail goes to exactly 0%.
   - ES macro-F1 still rises (still, walk and car improve), but **ctx's ES numbers are not trustworthy as a
     cross-user result**: part of the gain is "this is ES, so not rail".
   - Production records continuously, so it would look like SHL (full context). ES's gappy context is the odd one
     out.
4. **Without context** (base + Viterbi), ES rail stays at 0.37 and ES still → rail at 1.7%. SHL still → rail is
   32%, though.

**Deviations (step 3):**
- Step 3 ran on R2 after step 4.
- Context and HMM use all windows (also unlabelled), so the feature tables were rebuilt with `--all`. The labelled
  rows are identical; base reproduces R2 exactly.
- HMM transitions come from SHL training days only, not from ES.
- No HMM or context hyperparameters were tuned: horizon 60 s, +1 smoothing, restart after 60 s.
- One seed.

**Open questions (step 3):**
1. **Context shortcut.** Proposal: remove `ctx_n` / `ctx_span_s`, and when training, compute SHL context on an
   ES-like duty cycle (~20 s on / 40 s off) for part of the SHL sequences. Then the context's shape doesn't reveal
   the dataset. Evaluate SHL with both full and duty-cycled context. Or report base + HMM as the 02 candidate and
   leave context for later?
2. **Which decoder is the candidate?** Forward filtering is causal and per window. Viterbi needs the whole sequence
   (or a fixed delay). The edge server could run Viterbi with a delay; is a delay acceptable in production?
3. ES-only stopped early at 7–31 rounds (see above). Is a rerun with a fixed number of rounds wanted, for a fairer
   ES-only line?

### Status: parked (2026-10-26)

Marc's decision: the next experiment is a label-efficiency study (LightGBM label curve with and without the
gyroscope; LIMU-BERT pre-training + fine-tuning on a new day split). 02 is parked at this point.

**Current candidate: base + HMM Viterbi on R2.** R2 window-only LightGBM (features v1), Viterbi with SHL-estimated
transitions:
- SHL User 1: 8-class 0.853 / family 0.890 (day folds, one user).
- ExtraSensory dev users: 8-class 0.494 / family 0.583 (CI over users 0.541–0.620).
- ES rail recall 0.37, ES still → rail 1.7%; SHL still → rail 32%.
- Inference ~2.2 ms per window (Python), model 6.95 MB.

**The context model's ExtraSensory numbers are invalid** (step 3, ctx / ctx + forward / ctx + Viterbi). The models
use `ctx_n` (rank 1 of 277) as a dataset shortcut: always 24 windows on SHL, ~6 on ES. Their SHL numbers stand as
within-dataset results. Their ES numbers must not be compared with the base configurations or used to choose a model.

**Code added after step 3, not yet used in any run:**
- `tmd.hmm.fixed_lag`: fixed-lag smoothing, P(state_t | windows up to t + lag). Checked on a synthetic sequence:
  - lag 0 equals `forward` exactly;
  - an unbounded lag equals full forward-backward exactly;
  - a gap over 60 s splits the sequence into independent segments.
- `derive_step2_tables.py`: rebuilds the two step-2 feature tables from the `*_all` tables. `--check` found them
  identical: same rows and metadata, features equal with NaN = NaN.
  - The step-2 tables were then deleted (approved 2026-10-26).
  - `r1_cv.py`, `r2_cv.py` and `rescore_es_labels.py` read those tables, so run
    `uv run python derive_step2_tables.py` before re-running them.
- **Known issue in features v1:** `gyro_norm__skew` / `gyro_norm__kurt` are NaN on 1,996 of 1,716,072 ES windows
  with an exactly constant gyroscope. `skew * (std > 1e-9)` keeps NaN, since NaN × 0 is NaN. LightGBM treats it as
  missing. Not changed, to keep features v1 fixed.

**Open items (not done):**
1. **Step 3b, context fix:**
   - remove `ctx_n` / `ctx_span_s`;
   - train with ES-like duty-cycled context (≈ 20 s on / 40 s off) on part of the SHL sequences;
   - check for remaining leaks with a dataset classifier on context features only (report its AUC);
   - score SHL with continuous context and with ES-like gaps;
   - compare forward filtering, fixed lag ~60 s and full Viterbi (a delay is acceptable on the server; use fixed lag
     as default if it gets close to Viterbi).
   - The fixed context model replaces the candidate only if it beats base + Viterbi on both datasets and stays honest
     on ES rail and bus recall.
2. **ExtraSensory-only rerun** with a fixed number of rounds (about R2's). Early stopping stopped it at 7–31 rounds.
3. **Majority-vote comparison** (as a simpler smoother than the HMM).
4. **Plan step 5:** ONNX export, and inference time and size measured on it. The numbers so far are Python
   (LightGBM text model, numpy/scipy features).
5. **MLflow export:** `results/runs.csv` is not exported yet (CLAUDE.md: once, when the experiment finishes). Until
   then MLflow is the only record. All 02 runs are tagged dirty.

**Clean-up before the next experiment (2026-10-28).** 02-specific files moved out of the project-wide folders
(`git mv`, history kept); no behaviour or feature definition changed.
- `splits/shl_days.csv`, `splits/es_users.csv`, `splits/README.md` → this folder's `splits/` (unchanged bytes).
  The reserved final test is now `splits/reserved_test.json` (`tmd.splits.reserved_test()`), the reference from now on.
- `src/tmd/context.py` → `context.py` here (`step3_context_hmm.py` imports it locally).
- `tmd.splits.shl_days()` / `es_users()` → `make_features.py` (their only user), reading this folder's `splits/`.
- `tmd.tracking.SPLITS` points `*_dev` to this folder's files and `*_test` to `splits/reserved_test.json`.
- `backfill_dataset_inputs.py` reads the split file from `splits/` at commits before the move.

**MLflow dataset inputs, backfilled (2026-10-26, asked by Marc).** `tmd.tracking.run()` now takes
`datasets=[DataInput(...)]` and logs one MLflow dataset input per dataset part (metadata only). The 11 existing runs
got the same inputs with `MlflowClient.log_inputs`. Metrics, params, tags and artifacts were not touched.

```
uv run python backfill_dataset_inputs.py           # dry run: run -> inputs
uv run python backfill_dataset_inputs.py --apply   # logs/backfill_dataset_inputs.log
```

- Datasets: `shl_user1_dev` (digest `af25db55` = sha256 of `splits/shl_days.csv`) and `extrasensory_dev`
  (`649c32d5`, `splits/es_users.csv`). Since 2026-10-28 the files are in this folder's `splits/`; the inputs
  in MLflow keep the old source path (MLflow can't change it), and the digests still match. The script checked that each split file at the run's commit equals today's.
- Contexts: R1 → SHL training, ES evaluation (labels v1 = step-1 rules). The two R1 rescoring runs → ES evaluation
  (v2); SHL training with `trained_in_run=c615…`. R2 and step 3 → both training. ES-only → ES training, SHL
  evaluation.
- Input tags: `split`, `label_rules` (`v1` / `v2` / `shl-coarse`), `features` (`feat-v1`, `feat-v1+ctx60s` for the ctx
  runs), `rate_hz`, `window_s`, `stride_s`. Also `eval=out-of-fold` on cross-validated training data,
  `trained_in_run` where the run reused models, and `windows=all…` for step 3.
- MLflow links a dataset to a run only once (a second `log_input` with another context is dropped silently). So
  cross-validated data are one `training` input tagged `eval=out-of-fold`. `run()` refuses duplicates.
- Check: `search_runs` with `dataset.digest = 'af25db55'` returns all 11 runs. Dummy run
  `dataset-inputs-check-2026-10-26` in `00_smoke-test` (`1cc3817a183f40e287644fab8d876391`, dirty) filled both inputs.
- Search caveat: `dataset.name = 'x' AND dataset.context = 'evaluation'` matches each condition on any input of the run,
  so ES-only (ES training, SHL evaluation) also matches.
- Re-running `r1_cv.py`, `r2_cv.py`, `rescore_es_labels.py` or `step3_context_hmm.py` now needs `datasets=` added to
  their `run()` calls (not done).

## Conclusion

**Draft — not approved**

*Written from the Results section only; `runs.csv` is not exported yet (02 is parked, not finished).*

**Answer to the question.** Hand-crafted orientation-independent features + LightGBM at 25 Hz, smoothed by an HMM,
classify transport mode well within one SHL user, but much less well across users and datasets:

| | 8 classes | family |
|---|---|---|
| SHL User 1, held-out days (R2 + Viterbi) | 0.853 | 0.890 |
| ExtraSensory dev users, held-out users (R2 + Viterbi) | 0.494 | 0.583 (CI 0.541–0.620) |

- **Across datasets, training data matter more than the model.**
  - SHL alone does not transfer to ExtraSensory: family 0.362. ExtraSensory still is quieter than anything in SHL,
    and is read as rail 55% of the time.
  - Adding ExtraSensory users (R2) fixes this on ExtraSensory (family 0.489, still → rail 7%). It costs SHL still:
    48% → rail without HMM, 32% with Viterbi.
  - ExtraSensory alone is worse on ExtraSensory (family 0.403) and almost never predicts rail. Its early stopping
    may have underfit it.
- **The HMM is the cheapest gain.** Viterbi adds +0.11 family F1 on SHL and +0.09 on ExtraSensory over the
  window-only model, at no per-window cost. It raises SHL bus recall 0.56 → 0.85 and train 0.58 → 0.77.
- **60 s context helps within SHL** (family 0.930 with Viterbi; SHL still → rail 16%). As built, though, it encodes
  the dataset through the context length, so its cross-dataset value is unknown until step 3b.
- **Label rules matter for ExtraSensory still.** Without v2, a phone lying on a table made up 44% of still and
  sleeping 31%.
- **Car vs bus and train vs subway stay weak across datasets.** ES bus recall is 0.24 and ES rail recall 0.37 with
  Viterbi. This supports the brief's plan to resolve them per leg with GPS (05). Rail on ExtraSensory rests on 4
  self-reported users, so SHL is the reference for still vs rail.
- **Cost:** about 2.2 ms per window in Python, almost all feature extraction; the model takes 0.03 ms. The
  LightGBM model is 6.95 MB (edge server: no size limit).

**Limits.**
- SHL is one user (split by day).
- ExtraSensory rail comes from 4 self-reported users.
- One seed, no hyperparameter search.
- Inference times are from Python, not ONNX.
- All runs are dirty.
- The context-model ES numbers are invalid.

No `DECISIONS.md` entry: choosing the candidate doesn't change the project's direction.
