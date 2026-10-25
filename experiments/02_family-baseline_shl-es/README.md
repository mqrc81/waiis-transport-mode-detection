# 02 Family baseline on SHL + ExtraSensory

## Question

How well do hand-crafted features + LightGBM + HMM classify the family and the 8 classes at 25 Hz, across days,
positions, users and datasets?

*(From `experiments/docs/brief_block1_experiments.md`, which also holds the data, label, split and setup rules
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
- The brief is at `experiments/docs/brief_block1_experiments.md`, not `docs/brief_block1.md`. The brief's `~/data/raw/...` is
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
   - Dummy run `tracking-check-2026-10-25` in MLflow experiment `00_smoke-test` (run `95205245c57b4659afc9f957807b32be`,
     started from this folder, dirty tree). Params, metrics, a figure artifact (`mlartifacts/00_smoke-test/...`), and
     the `git_commit` / `dirty` tags were all logged. `tracking.py export 00_smoke-test --out <scratchpad>` worked.

Splits approved 2026-10-25 and written to `splits/` (one row per SHL day; one row per ExtraSensory user).

## Conclusion

*(not yet)*
