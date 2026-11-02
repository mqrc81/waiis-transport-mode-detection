# 08 Quality check of candidate datasets for waiis_bench_v2

## Question

Are three public datasets usable for `waiis_bench_v2`, for training or as an external test set: **NOR-TMD**
(the "D1" dataset of doi:10.1080/15472450.2025.2515567), **MobilitApp** (doi:10.34810/DATA1948) and **Collecty**
(doi:10.1016/j.dib.2023.109481)? What would each add (users, platforms, classes, gyroscope, GPS), and what would it
cost in label mapping and cleaning?

This is a quality check. There are no models and no MLflow runs; findings are recorded here only.

## Plan

*From the brief of 2026-11-02.*

- **Data:** `<data root>/raw/<dataset>/`. Report the size before any download over 5 GB. If a site blocks access or
  the data needs a request, stop on that dataset; it is downloaded by hand.
- **Stop-point 1 (access and layout, before any reader).** Per dataset: licence and citation (`LICENSES.md`); file
  layout and columns, sensor names, timestamp unit and reference; how users, devices, platform, trips and labels are
  encoded; label vocabulary; whether acceleration includes gravity, and its unit. Wait for OK.
- **Readers:** one per dataset in `src/tmd`, returning the same frame as the v1 readers. Window tables go in
  `processed/`, named by their parameters.
- **Checks** (per dataset, same definitions everywhere):
  1. inventory: users, devices, platform, device models, recording period; trips, hours per class per user;
     single-mode trips or legs;
  2. units and plausibility: median |acc| when still; axis signs per platform; clipping (max |acc| per device); share
     failing v1's accelerometer plausibility check;
  3. timing: real rate per device and session (median, p5, p95 of intervals); duplicates, non-monotonic timestamps,
     gaps > 0.5 s; fixed-rate vs event-driven; background throttling;
  4. window yield: clean 15 s windows (single label, real accelerometer rate ≥ 25 Hz over the window, no gap > 0.5 s,
     plausibility passed; v1's rules where they exist, deviations stated); one window per labelled minute (v1's grid)
     and total clean hours;
  5. other sensors per clean window: gyroscope (≥ 25 Hz, same rule), magnetometer, barometer;
  6. GPS, if present: coverage as in 06, fix cadence, fields, repeated positions, median derived speed per class;
     location only as aggregates (no maps of traces, no raw coordinates in git);
  7. distributions of clean windows: class × platform; users per class with ≥ 5 windows, top user's share; phone
     position;
  8. labels: mapping proposal to v1's 8 classes, modes outside v1 with hours and users; segment lengths; plausibility
     flags (walk with GPS speed > 3 m/s, still moving, vehicles at walking speed); extra labels;
  9. duplicates: identical files or sessions.
- **Results:** one table per dataset; one cross-dataset table with v1's ExtraSensory and SHL User 1 for comparison;
  a verdict per dataset (training / external test / not usable); open decisions for v2.
- **Stop-point 2:** Results and verdicts here, `STATUS.md` updated. Wait for review.

**Answers at stop-point 1 (approved 2026-11-02):**
1. **Downloads:** NOR-TMD and Collecty are downloaded; the zips are kept. Before converting, estimate from a sample
   the Parquet size of only the needed columns (acc, gravity if separate, gyro, timestamps, user / device, platform,
   label). If it fits the free space with a 10 GB margin, convert each dataset in one streaming pass to Parquet
   partitioned by user / device, and run all checks on that. Nothing is deleted without asking.
2. **Labels:** v1's 8 classes are kept. Labels that conflict with them are excluded: NOR-TMD `outside` and
   `inside`, Collecty `Walk` (it includes standing); their hours are still reported. Every mode outside v1 is
   reported with hours, users and datasets. A new class is a candidate only with **≥ 10 users across ≥ 2 datasets**.
3. **Reader output:** as proposed: a per-minute labels table, one 15 s × 25 Hz window per minute with gravity
   included, and a raw-stream accessor. Collecty: total acceleration = `acc*` + `grav*`; the real rate comes from the
   change-time columns (repeated rows are held values, not samples).
4. **MobilitApp: inventory only** (checks 1, 3 and 8 on the timed files). No plausibility check, no window tables.
   Untimed files are counted in the inventory, never as clean. Gravity removal goes into the open decisions: it only
   matters if WAIIS sends linear acceleration.
5. **NOR-TMD iOS:** its real rate is reported. Below 25 Hz its windows don't count as clean; the hours this excludes
   are reported.
6. **Results also give**, per dataset, the v1 classes it contributes cleanly and the other modes present, and flag
   datasets that contribute only vehicle classes (a possible dataset shortcut, to be measured in v2).

**Answers at stop-point 2 (approved 2026-11-02, with changes before committing):**
1. Rules approved: the Collecty 12-hour clock repair and splits at backward steps (overlapping windows never
   clean); the dedup rule (one copy within a user, both dropped across users); listing only non-empty windows.
2. NOR-TMD and Collecty data verified as usable commercially (2026-11-02); `LICENSES.md` updated.
3. MobilitApp: final verdict "not usable"; tram and e-scooter users recounted without it. Tram windows stay built
   and flagged; the class decision waits for which modes WAIIS needs.
4. iOS sign check, no change to v1: which 05 / 06 features depend on a raw axis or its sign; mean z on still
   minutes of ExtraSensory iPhone vs Android; whether v1 results could be affected. v2 reader rule: every iOS
   source converted to the Android axis convention and m/s².
5. Open decisions added: a ~20% NOR-TMD test set for vehicles and rail; a per-user cap per class (Collecty, check
   NOR-TMD); excluded foot labels kept as unlabelled context minutes; the within-class dataset-identification
   check as a required v2 check.

## Results

### Stop-point 1: access and layout (2026-11-02)

**Access**

| | NOR-TMD (D1) | MobilitApp | Collecty |
|---|---|---|---|
| Host | Kaggle `scholarone/nor-tmd`, v1 (2025-01-12) | CORA.RDR Dataverse, v1.0 (2024-12-24) | Kaggle `merdelic/dataset-for-multimodal-transport-analytics`, v4 (2023-07-15) |
| Licence | CC BY 4.0 (Kaggle page) | CC0 1.0 | "Unknown" on Kaggle; article CC BY 4.0 |
| Size | 1 CSV, **73.6 GB** unzipped; zip **7.36 GB** | zip 833 MB (2.8 GB unzipped), 8,676 CSV | 454 TXT, **24.9 GB** unzipped; zip **5.90 GB** |
| State | **not downloaded** (> 5 GB) | downloaded, md5 matches Dataverse (`bc535c20…`) | **not downloaded** (> 5 GB) |
| Account needed | no: the Kaggle API serves the archive without login (checked 2026-11-02) | no | no (same) |

- **NOR-TMD data availability:** the publisher page (tandfonline.com) returns 403, so the formal data availability
  statement was not read. The paper's text says "This dataset has been made publicly available on Kaggle (Skretting &
  Grønli, 2025)", and its references give `https://www.kaggle.com/datasets/scholarone/nor-tmd`. Source of the
  paper's text: the copy in A. Skretting's PhD thesis, Kristiania 2025:3 (`kristiania.no`, PDF, pp. 214–235).
- **Collecty data availability (article):** Kaggle, doi:10.34740/kaggle/dsv/4976711; "The data can be downloaded
  upon registering."
- **Disk:** 43 GB free on the data volume. NOR-TMD unzipped does not fit. Both Kaggle sets would be kept zipped,
  and their readers would stream from the zip.
- Layout of NOR-TMD and Collecty was read from the **first 4 MB of each zip** (HTTP range request, start of the
  deflate stream inflated): 227k NOR-TMD rows (one Android trip) and 62k Collecty rows (file `17_1.txt`). Statements
  below marked *(sample)* rest on that alone.
- Licences and citations are in `LICENSES.md`.

**NOR-TMD (D1)**

- **Collection** (paper): 101 participants, 57 device models, Android and iOS (about twice as many Android users),
  Oslo and Bodø, one month; trips started and stopped by hand in a custom app, labels reviewed before upload.
  Sample rows are from 2021-09-28/29.
- **Layout:** one long-format CSV, **one row per sensor event**, columns `installationId, journeyNumber, tripNumber,
  manufacturer, model, timestamp, transportType, typeInteger, typeString, deviceLocation, value_0 … value_5,
  activity_recognition, OS`.
- **Encoding:**
  - user / device: `installationId` (UUID; an app installation, not checked against persons yet);
  - device model: `manufacturer`, `model` (e.g. HUAWEI VOG-L29);
  - platform: `OS` (`ANDROID` in the sample; iOS spelling unknown);
  - trips: `journeyNumber`, `tripNumber` (a journey has trips, i.e. legs; one mode per trip, to be confirmed);
  - label: `transportType` per row;
  - phone position: `deviceLocation` (`HAND` in the sample);
  - sensor: `typeString` (Android sensor names, `typeInteger` = Android sensor type).
- **Sensors** (sample, Android): `android.sensor.accelerometer` (type 1), `linear_acceleration`, `gravity`,
  `gyroscope`, `gyroscope_uncalibrated`, `magnetic_field`, `magnetic_field_uncalibrated`, `rotation_vector`,
  `game_rotation_vector`, `geomagnetic_rotation_vector`, plus `ACTIVITY_RECOGNITION` rows (`ENTER VEHICLE` /
  `EXIT VEHICLE`). iOS (paper, Table 1): accelerometer, gravity, rotation rate, quaternion, gyroscope, magnetic field,
  pressure, activity recognition. `value_3…5` are only used by uncalibrated and rotation sensors.
- **Timestamp:** string `YYYY-MM-DD HH:MM:SS.ffffff`, millisecond resolution, **no time zone** (reference not stated;
  local vs UTC to be checked). Rows are **not sorted** within a sensor *(sample)*.
- **Acceleration:** type 1 accelerometer **includes gravity**, m/s² (median |a| 9.82 *(sample)*); linear acceleration
  and gravity are also there. iOS accelerometer unit unknown (CoreMotion reports g); to check.
- **Rate:** the app requested 5 Hz (paper); Android accelerometer events come at a median 10 ms (~100 Hz) *(sample)*.
  iOS: 3.86 M accelerometer events in total against 71.5 M on Android (paper, Table 1); **whether iOS reaches 25 Hz is
  open.**
- **GPS:** none (not in the paper's sensor table, none in the sample).
- **Labels (paper):** `bicycle, boat, bus, car, e-scooter, inside, metro, outside, train, tram`. "Inside" = inside a
  building, not in a vehicle; "outside" = outdoors but not in a vehicle, "for instance, walking or standing outside".
  **There is no walk, run or still label.** No e-scooter on iOS. Trips over 1 h and under 60 s were removed by the
  authors. Spelling in the file: `BUS` *(sample)*; the other spellings are not seen yet.

**MobilitApp**

- **Collection** (`Readme.txt`): Android app MobilitApp, Barcelona metropolitan area, "50 smartphones", "298 h",
  23 March 2023 – 27 June 2024; volunteers start and stop single-mode trips.
- **Layout:** one flat folder of CSV files with a quoted header `Timestamp, ACCX, ACCY, ACCZ, MAGX, MAGY, MAGZ, GYRX,
  GYRY, GYRZ`. File name `<id16>_[<NNNN>_]<mode>_<sensors>_<YYYYMMDDhhmmss>_<seg6>.csv`; the optional 4-digit counter
  (3,517 files) is not in the Readme.
  - 8,633 `ACC-MAG-GYR` files, 37 `PRESS` (barometer, hPa in `ACCX`) and 6 `STEP` (step count in `ACCX`).
- **Encoding:**
  - **user / device:** the 16-character id. The Readme says it changes with every trip "for the sake of privacy",
    but in the files it does not: **119 ids**, up to 548 trips and 68 days per id (median 4 trips, 1 day). So an id
    is a phone (or app installation) over time, and 119 ids against 50 phones means some phones have several ids.
    A split by id is possible; a split by person is not guaranteed.
  - **trip:** (id, start stamp): 3,272 trips, **each a single mode**, so no legs.
  - **segment:** consecutive files of a trip, back to back (median gap 0 s between files; about 3,000 samples per
    file, so ~24 s at ~125 Hz or ~55 s at ~55 Hz).
  - **platform:** Android only. **No device model** anywhere.
  - **label:** in the file name only. **No GPS** and no phone position.
- **Label vocabulary** (files, `ACC-MAG-GYR`): Car 1,678 (+ Cotxe 22), Metro 1,398, Bus 1,345, Walk 1,125
  (+ Caminar 44), Tram 897, Train 457 (+ Tren 25), Stationary 422 (+ Estacionario 17, Estacionari 1), e-Scooter 312
  (+ E-Scooter 74, e_Scooter 3), Moto 234, Run 227, Bici 226 (+ Bicycle 24), e-Bicing 58 (+ e-Bike 34, e_Bicycle 6,
  e-Bicicleta 4). That is 12 modes in English, Catalan and Spanish spellings (`e-Bicing` = Barcelona's shared e-bike).
- **Timestamp** (all 8,633 files checked): the format changes with the app version.

  | Format | Files | Months | Usable sample time |
  |---|---|---|---|
  | epoch **ms** (UTC; file-name stamp is local time, −1 h / −2 h) | 3,828 | 2022-11 – 2023-06 | yes |
  | `99999999999` on every row (fill value) | 4,076 | 2023-04 – 2024-01 | **no** |
  | `YYYYMMDDhhmmss` | 133 | 2024-01 – 2024-03 | **no**: one value for the whole file |
  | `YYYY-MM-DD hh:mm:ss` | 598 | 2024-05 – 2024-06 | **no**: one value for the whole file |

  So **56% of the files have no usable sample times**: rate, gaps and duplicates can't be measured there. The ms
  files start in November 2022, before the Readme's start date.
- **Timing in the ms files** (sample of 508 files): median interval 8 ms (p5 7, p95 10; ~125 Hz) on most phones,
  20 ms on others; 0.4% repeated timestamps (median per file), no backwards steps, gaps > 0.5 s rare (max 3 per file).
  Recorded time in the ms files: **39.7 h**. At ~37 s per file, all 8,633 files are about 90 h, **not 298 h**.
- **Acceleration: mostly gravity-free.** The Readme says the app records Android linear acceleration (m/s²), and the
  data agree: median |a| per file is 0.44 m/s² (ms files), Stationary 0.17, Walk 3.4, Run 14.3. But 43 of 1,200
  sampled files (3.6%) have a median |a| of 7–13 m/s², and 7 of 79 sampled ids look gravity-included throughout:
  probably phones without a linear-acceleration sensor. **No gravity channel is stored, so gravity can't be
  restored** for the rest.
- **Other sensors:** gyroscope (rad/s) and magnetometer (µT) on every row; barometer only in 37 files.

**Collecty**

- **Collection** (article): app Collecty, **Android**, Zagreb, November 2021 – March 2022, 15 users; only routes the
  user confirmed are published; no device models given.
- **Layout:** 454 files `archive/<user>_<trajectory>.txt` (CSV with header): 15 user ids (17, 18, 20, 23–29, 31, 35,
  37, 39, 40), trajectories per user 1–168 (user 18: 168, user 29: 71, user 23: 59, user 37: 45, user 17: 40; six users
  have ≤ 2).
- **Columns:** `sampleTime, accX, accY, accZ, lChangeTimeacc, gyroX, gyroY, gyroZ, lChangeTimegyro, gravX, gravY,
  gravZ, lChangeTimegrav, magnX, magnY, magnZ, lChangeTimemagn, label`.
- **Encoding:** user and trajectory in the file name; label per row; a trajectory **can contain several modes**
  (legs, with the change marked by the user). No platform column (all Android), no device model, no phone position, no
  extra labels.
- **Timestamp:** strings `YYYY-MM-DD hh:mm:ss[.ffffff]` (24-hour clock, though the article says 12-hour; the fraction
  is left out when it is .000), millisecond resolution, **no time zone**. The sample trip starts at 01:22, so UTC vs
  local is open.
- **Sampling:** a 10 ms timer copies the latest value of each sensor (sample-and-hold), and `lChangeTime*` is the
  time of that sensor's last new event. So the **real rate comes from the distinct `lChangeTime*` values**, not from
  the rows. *(sample)*: ticks every 10 ms (one 957 ms gap); distinct accelerometer events ~89 Hz (interval p5 9, median
  10, p95 14 ms); gyroscope, gravity and magnetometer similar.
- **Acceleration:** `acc*` is **linear acceleration** (gravity removed), m/s²; `grav*` is the gravity vector, so
  **acc + grav gives the total acceleration**: |acc + grav| median 9.79 m/s² *(sample)*.
- **Labels** (article): `Walk, Run, Bike, Car, Bus, Tramway, Train, E scooter, Unknown` (Unknown on 37 of 83 M
  rows). **Standing (e.g. waiting for a bus) is labelled Walk**, and there is no still label. Bike and e-scooter
  come from 2 users only.
- **GPS:** recorded by the app but **not published**.

**What this means for the checks (to decide before the readers)**

1. **Downloads:** NOR-TMD (7.36 GB zip, 73.6 GB unzipped) and Collecty (5.90 GB zip, 24.9 GB unzipped) are over
   5 GB. I download them myself if approved; both stay zipped and are streamed.
2. **"The same frame as the v1 readers":** v1 has no common reader frame (`tmd.shl` returns arrays, `tmd.extrasensory`
   snippets). Proposal: each reader returns what `tmd.bench` exposes, a per-minute `labels` table (user, ts, class
   and flags) and a 15 s × 25 Hz × 3 accelerometer window per minute in m/s² with gravity, plus a per-trip
   raw-stream accessor for the timing checks.
3. **Gravity-free accelerometers:** v1's plausibility check (window median |acc| within 7–20 m/s², rule U3 of 04's
   decision register; also `quiet`, std |acc| < 0.03 m/s², and X3) and the gravity features assume total
   acceleration. Collecty can be rebuilt (acc + grav). MobilitApp can't. Proposal: report MobilitApp
   with the bound replaced by a check on |linear acc| (e.g. still/vehicle median < 3 m/s²; numbers to be fixed
   before looking at yields), and flag it as unusable for v1's feature set, which assumes gravity.
4. **Untimed MobilitApp files:** count only epoch-ms files as clean (the timing rules can't be checked otherwise);
   the 4,807 untimed files go into the inventory only.
5. **Collecty rate:** measured from distinct `lChangeTimeacc` values, not from the 100 Hz rows; held repeats are
   dropped before resampling.
6. **Labels without a v1 counterpart:** NOR-TMD has no walk / run / still (`outside`, `inside` cover them in part);
   Collecty folds standing into walk. Both affect the mapping in check 8, so neither can give a still class as
   v1 defines it.
7. **GPS:** none of the three publishes GPS, so check 6 and the GPS-based label flags of check 8 drop out for all
   three.
8. **Users:** MobilitApp ids are phones or installations (119 ids, 50 phones), NOR-TMD `installationId` the same.
   Splits by id are possible in both; the persons behind the ids are unknown.

### Stop-point 2: checks (2026-11-02)

**Commands** (from this folder; logs in `logs/`, tables in `results/tables/`, every table also in
`results/tables/report.md`):

```
# downloads (Kaggle API, no login; resumable): logs/download_{nor_tmd,collecty}.log
uv run python convert.py collecty          # zip -> Parquet, one pass (22 s)
uv run python convert.py nor_tmd           # zip -> Parquet, one streaming pass (~13 min)
uv run python build_windows.py collecty    # window tables in <data>/processed/collecty_windows_25hz_w15s/
uv run python build_windows.py nor_tmd     # ... nor_tmd_windows_25hz_w15s/
uv run python mobilitapp_inventory.py      # results/tables/mobilitapp_*.csv
uv run python report.py                    # all check tables
uv run python sign_check.py                # iOS sign check on v1 (dev users only)
```

Code: `src/tmd/candidates.py` (window rules, timing, legs, clipping: one definition for all datasets),
`src/tmd/{nor_tmd,collecty}.py` (readers), `mobilitapp.py` here (moved out of `src/tmd` after 08: not used
beyond it), the scripts above.

**Size check before converting** (answer 1): from the two 4 MB samples, written in the converted schema: NOR-TMD
8.7 B per kept row, 273 M kept rows ≈ 2.4 GB; Collecty ≈ 3.0 GB. Free space after both zips: ~30 GB, so even at
twice the estimate the 10 GB margin held. Actual: **NOR-TMD 2.77 GB, Collecty 1.52 GB**. Free space went from 43 GB to
~18 GB: zips 13.3 GB, Parquet 4.3 GB, the rest is macOS swap (12 GB in use during the parallel jobs) and three local
Time Machine snapshots of today (not touched).

**Conversion check:** NOR-TMD 387,464,900 rows, the paper's Table 1 event counts exactly per sensor and platform;
Collecty 83,227,415 rows, the article's count exactly.

**Deviations and rules added during the checks** (2, 4 and 5 approved 2026-11-02; the others recorded):
1. **Resampling** (`src/tmd/candidates.py`, `resample_window`): 100 Hz grid over the window ± 1 s, not over the whole
   stretch as in 04 (`build.py`, `shl_rec`). Same interpolation and `decimate_100_to_25`.
2. **Windows without any event are not rows** (approved) (`trip_windows`): empty minutes and tiles inside a trip are not
   listed (a broken clock made one trip span 52 years before the parsing fix below).
3. **Clean** = single label, rate ≥ 25 Hz, no gap > 0.5 s, U3 (window median |acc| 7–20 m/s²) and not X3 (quiet
   walk / run / bike), as v1. `edge` = the window is not fully inside the trip.
4. **Collecty clocks** (approved) (`src/tmd/collecty.py`, `repair_time`; `build_windows.py`, `collecty_trip`):
   - `sampleTime` and the change times use a **12-hour clock without AM / PM**: 12:59:59.998 is followed by
     01:00:00.009. 36 files have this −12 h step; it is undone (+12 h after it). A whole file recorded in the
     afternoon reads as morning, so the time of day is not reliable (only intervals are used).
   - 40 files (users 17, 18 mostly) have **other backward steps** (median 3 s, up to 2,885 per file, a sawtooth
     clock). The trajectory is cut into pieces at each backward step; a window that overlaps another piece's time
     span gets status `time_overlap` and is never clean.
   - Timing and legs are computed per file in file order, windows per piece.
5. **Collecty duplicates** (approved) (`collecty.duplicates`): 13 pairs of byte-identical files (26 files), 2 of the
   pairs under two different users (17_8 = 29_14, 23_11 = 40_4). **Rule:** within a user keep the lower trajectory number,
   across users drop both (owner unknown): 15 files dropped. All Collecty numbers below are after this drop.
6. **NOR-TMD timestamps** (`src/tmd/nor_tmd.py`, `_t_ms`): iOS rows have 4-digit fractions and **unpadded hours
   and minutes** (`2021-10-25 14:0:00.0090`). Arrow's ISO parser rejects those, so 17 of 275 batches go through a
   pandas fallback (`format="ISO8601"`). My first conversion took that fallback with the wrong unit and put 1,149
   trips in 1970 (my bug, not the data: no raw row has a 1970 date). Fixed and reconverted; 0 unparsed timestamps;
   the batches and example strings are in `timestamp_fallback.parquet` next to the events.
7. **NOR-TMD sensors** (`build_windows.py`): accelerometer = Android `accelerometer` (TYPE_ACCELEROMETER, m/s²,
   gravity included) or iOS `CMAccelerometerData` (g, gravity included; converted per piece by
   `tmd.resample.to_ms2`, the v1 rule). Gyroscope = the denser of `gyroscope` / `gyroscope_uncalibrated` (Android)
   or `CMGyroData` / `CMDeviceMotion.rotationRate` (iOS). Magnetometer and barometer (`pressure`, `CMAltitudeData`):
   at least one event in the window.
8. **No still label** in NOR-TMD or Collecty: the |acc| level "when still" (check 2) is measured on windows with
   std |acc| < 0.1 m/s² (phone at rest), whatever the label.
9. **Check 6 (GPS) and the GPS-based flags of check 8 don't apply:** none of the three has GPS. The acc-only flag
   "still with high acc energy" uses a threshold fixed from v1 before looking at the candidates: p95 of window std
   |acc| over v1's official still examples, dev users only (ExtraSensory + SHL User 1): **1.278 m/s²**.
10. **MobilitApp hours of untimed files** are estimated as samples / the id's median rate over its timed trips
    (dataset median if the id has none); they are inventory only.

**Per-dataset results.** All tables: `results/tables/report.md` (CSV per table beside it). "Clean" = rule 3 above;
"v1" = a raw label mapped to one of v1's 8 classes (answer 2). "Users" are app installations / ids, not checked
against persons.

*NOR-TMD*

| Check | Result |
|---|---|
| 1 Inventory | 152 installations (paper: 101 participants), Android 107 / iOS 44 (+1 without accelerometer); 57 device models; 2021-03-25 – 2022-02-28 (paper: one month); 3,149 trips, 9 with two labels; 613.5 h |
| 2 Units | Android TYPE_ACCELEROMETER m/s², gravity included: median \|acc\| at rest 9.8; iOS `CMAccelerometerData` **in g** (median 1.00) with the **opposite sign**: trip-mean gravity along −z in 52% of iOS trips, +z in 52% of Android trips. U3 fails 0%. Max \|acc\| per device median 68.5 m/s² (max 132.6); no plateaus (≤ 368 events at an axis maximum) |
| 3 Timing | **Android: event-driven**, accelerometer median 51.5 Hz per trip (p5 5.8, p95 98.8), 15 of 108 devices below 25 Hz; 55 repeated times per 1k events; 1.2 gaps > 0.5 s per h. **iOS: fixed 5 Hz** (200 ms on all 44 devices; the app's setting). No backward steps; no throttling (rate after 5 min / first 5 min = 1.00) |
| 4 Yield | Android 17,547 clean minute windows, **12,988 with a v1 class**; clean hours 296.8, **219.1 with a v1 class**. **iOS 0: 214.2 h excluded (5 Hz)**; Android below 25 Hz: 51.9 h |
| 5 Sensors | clean v1 windows: gyroscope 78.1%, magnetometer 100%, barometer 58.3% |
| 6 GPS | none |
| 7 Distribution | bus 7,081 windows (70 users ≥ 5), subway 3,908 (27), car 804 (13), train 603 (17), bike 592 (6); top user's share ≤ 0.48 (bike), 0.06 (bus). Position: hand 65%, pocket 26%, other 9% |
| 8 Labels | bicycle → bike, bus, car, metro → subway, train; outside v1: **boat 76.3 h / 19 users, tram 40.4 h / 33, e-scooter 3.3 h / 1**; excluded: outside 38.8 h / 59, inside 7.6 h / 29. Leg median 8 min (bus, metro) to 23 min (bicycle). X3 0.2% of bike windows; quiet (std < 0.03) boat 17.7%, car 7.5% |
| 9 Duplicates | none (windows, trips) |

*Collecty* (after dropping 15 duplicate files)

| Check | Result |
|---|---|
| 1 Inventory | 15 users (13 with clean v1 windows), Android, no device models; 2021-11-24 – 2022-04 plus 9 files from 2023-01/02 (article: Nov 2021 – Mar 2022); 439 trajectories, **217 with several legs**; 233.6 h |
| 2 Units | acc + grav, m/s²: median \|acc\| at rest 9.8; U3 fails 0.008%. **4 devices clip at ±8 g** (axis maximum 78.4 m/s², ≤ 616 events); the others stay below |
| 3 Timing | real events median 88 Hz (p5 77, p95 91) per trajectory, interval 10 ms, event-driven within a 10 ms timer (regular 0.70); 24.6 gaps > 0.5 s per h; **clock faults**: 36 files with a 12 h wrap, 40 with other backward steps (17,312 minute windows `time_overlap`); no throttling |
| 4 Yield | 13,313 clean minute windows, **9,026 with a v1 class**; clean hours 222.5, **150.7 with a v1 class**; below 25 Hz: 1.1 h |
| 5 Sensors | gyroscope 100%, magnetometer 94.3%, barometer 0% |
| 6 GPS | recorded by the app, not published |
| 7 Distribution | car 3,899 windows (13 users ≥ 5, top 0.43), train 2,664 (5, top 0.84), bus 1,705 (5, top 0.75), bike 562 (2, top 0.98), run 196 (1). No position |
| 8 Labels | Run, Bike, Car, Bus, Train mapped; outside v1: tram 7.6 h / 4 users, e-scooter 6.3 h / 3; **excluded: Walk 62.5 h / 14 users**; Unknown dropped. Leg median 7 min (tram) to 38 min (train). No X3; quiet ≤ 4.2% |
| 9 Duplicates | 13 pairs of identical files, 2 across users; 15 dropped (deviation 5) |

*MobilitApp* (inventory only)

| Check | Result |
|---|---|
| 1 Inventory | 119 ids (71 with timed trips), Android, no device models; file stamps 2022-11-25 – 2024-06-27; 3,272 trips, all single-mode; 1,170 timed. **39.8 h timed + ~28.9 h untimed (estimated) ≈ 69 h, not 298 h** |
| 3 Timing (timed) | median 124 Hz per trip (p5 49), interval 8 ms (p95 of medians 20 ms); 4 ids below 25 Hz; 10 repeated times per 1k; no backward steps; 3.4 gaps > 0.5 s per h; regular 0.71; no throttling. Untimed: 4,076 files with a fill value, 731 with one second-resolution time per file |
| 8 Labels | still 4.7 h / 22 ids, walk 9.6 / 50, run 1.6 / 7, bike 1.4 / 9 (none timed), car 16.4 / 19, bus 9.4 / 34, train 3.8 / 27, subway 9.3 / 15 (all files, untimed estimated); outside v1: tram 5.6 h / 5, e-scooter 4.8 / 11, motorbike 1.5 / 4, e-bike 0.6 / 9. Trip median 1–3 min (timed). Flags on 15 s tiles: still with std > 1.278 m/s²: 1.1%; quiet walk / run / bike: 1.5% of walk. Linear acceleration, except train tiles (median \|acc\| 9.8: phones without a linear sensor) |
| 9 Duplicates | 4 byte-identical files; **45 files with identical accelerometer content** in 19 trips (10 groups), **6 groups under different modes** (e.g. one segment as Bus, Metro and Stationary) |

**Cross-dataset summary** (`cross_dataset.csv`, `contributions.csv`)

| Dataset | Clean v1 windows | Clean v1 hours | Users | iOS / Android | Gyroscope | GPS coverage | Users per class (≥ 5 clean windows) | Modes outside v1 |
|---|---|---|---|---|---|---|---|---|
| NOR-TMD | 12,988 | 219.1 | 89 | 0 / 100% | 78.1% | none | bike 6, car 13, bus 70, train 17, subway 27 | tram, ferry/boat, e-scooter |
| Collecty | 9,026 | 150.7 | 13 | 0 / 100% | 100% | none | run 1, bike 2, car 13, bus 5, train 5 | tram, e-scooter |
| MobilitApp (inventory) | – | – | 119 ids | 0 / 100% | all files | none | – (no windows) | tram, e-scooter, motorbike, e-bike |
| v1 ExtraSensory (official, dev + test) | 145,380 | 605.8 (minute grid) | 59 | 61.9 / 38.1% | not in benchmark | 83.2% | still 59, walk 57, run 13, bike 21, car 47, bus 26, train 5 | – |
| v1 SHL User 1 (official) | 23,076 | 96.2 (minute grid) | 1 | 0 / 100% | not in benchmark | 70.3% | all 8 classes, 1 user | – |

| Dataset | v1 classes contributed cleanly | Also present | Only vehicle classes? |
|---|---|---|---|
| NOR-TMD | bike, car, bus, train, subway | tram, boat, e-scooter (and outside / inside, excluded) | **yes, plus bike**: no still, walk or run |
| Collecty | run (1 user), bike (2), car, bus, train | tram, e-scooter (and Walk, excluded) | **yes, plus bike and run**: no still or walk |
| MobilitApp | none (inventory only) | all 8 v1 classes as labels; tram, e-scooter, motorbike, e-bike | – |

**Both usable sets contribute only vehicles plus bike / run.** In a pooled v2, dataset identity would predict "not
still / not walk", a possible dataset shortcut, to be measured in v2 (e.g. a dataset classifier within each class).

**Modes outside v1** (`modes_outside_v1.csv`; candidate class = ≥ 10 users across ≥ 2 datasets, answer 2;
MobilitApp not counted, answer 3 at stop-point 2)

| Mode | Datasets | Users | Hours | Per dataset | Candidate |
|---|---|---|---|---|---|
| tram | 2 | 37 | 48.1 | NOR-TMD 40.4 h / 33 users; Collecty 7.6 / 4 | **yes** |
| e-scooter | 2 | 4 | 9.6 | NOR-TMD 3.3 / 1; Collecty 6.3 / 3 | no |
| ferry / boat | 1 | 19 | 76.3 | NOR-TMD only | no |

Before the recount (with MobilitApp ids): tram 42 users / 3 datasets, e-scooter 15 / 3 (11 of them MobilitApp ids);
MobilitApp alone also had motorbike (4 ids) and e-bike (9 ids). Tram windows stay built and flagged in the window
tables (`label` = `tram` / `Tramway`, `cls` = ''); the class decision waits for open decision 1.

**Top-user shares per v1 class** (`nor_tmd_top_users.csv`, `collecty_top_users.csv`; users with ≥ 1 clean v1
window)

| Class | NOR-TMD users | top user | top 3 users | Collecty users | top user | top 3 users |
|---|---|---|---|---|---|---|
| bike | 7 | 0.48 | 0.75 | 2 | 0.98 | 1.00 |
| car | 16 | 0.30 | 0.66 | 13 | 0.43 | 0.76 |
| bus | 74 | 0.06 | 0.16 | 5 | 0.75 | 0.96 |
| train | 20 | 0.23 | 0.43 | 5 | 0.84 | 0.98 |
| subway | 30 | 0.15 | 0.31 | – | – | – |
| run | – | – | – | 1 | 1.00 | 1.00 |

### iOS sign check on v1 (stop-point 2 answers, item 4; no change to v1)

`sign_check.py` → `results/tables/es_still_mean_z.csv`. Dev users only (no reserved test user read).

- **05 / 06 features don't depend on a raw axis or its sign.** 06's accelerometer model uses 05's B,
  `af.names(True, False)` = BASE + SUB, 580 features (`experiments/06_gps-acc-baseline_waiis-bench/pipe.py:29`).
  Their 4 channels are |a|, a · ĝ (ĝ the 0.3 Hz low-pass of the same window, so it flips with a), the horizontal
  norm and |da/dt| (`src/tmd/acc_features.py`, `channels`). The GRAV block (not adopted in 05) is also built from
  angles and projections only.
  - `tests/test_acc_features.py` passes (4 tests).
  - On 500 real dev windows, all 580 features are **identical** with all axes negated (max relative change 0) and
    with x / y swapped.
- **ExtraSensory has the same convention split:** still minutes, 4,000 sampled windows per platform:

  | Platform | Users | Median window-mean z [m/s²] | p5 / p95 | z dominant, positive | z dominant, negative |
  |---|---|---|---|---|---|
  | Android | 19 | **+9.12** | −9.68 / +10.19 | 66.0% | 11.0% |
  | iPhone | 28 | **−9.40** | −9.90 / +9.30 | 9.0% | 66.5% |

- **v1 results (05, 06, 07) are not affected:** their features are sign-invariant. A model on raw axes (e.g.
  InceptionTime, LIMU-BERT from scratch, both v1 candidates in `STATUS.md`) would see the sign as a platform tell,
  so for those the convention matters even on v1.
- **v2 reader rule:** every iOS source is converted to the Android axis convention and m/s² in its reader
  (iOS × −9.80665, all three axes), before windows are cut. This covers ExtraSensory iPhones in v2 too.

**Open decisions for v2**

1. **Label mapping for new modes:** without MobilitApp, tram passes the candidate rule (37 users, 2 datasets,
   48.1 h, 33 of them from NOR-TMD); e-scooter (4 users) and boat (1 dataset) don't. Options for tram: a new class,
   merged into rail (train / subway) or into the rail family only, or excluded. Waits on which modes WAIIS needs.
2. **Platform balance:** every clean window from the candidates is Android. iOS comes only from ExtraSensory
   (61.9% of v1). Using NOR-TMD iOS (214 h, 44 devices) would need a separate rate rule (5 Hz), i.e. a second
   benchmark track.
3. **GPS availability:** none of the candidates publishes GPS. 06's fusion can't be trained or evaluated on them;
   their windows go through the accelerometer model only, and the GPS model stays on ExtraSensory + SHL.
4. **Leave-one-dataset-out feasibility** (users with ≥ 5 clean windows per class):
   - car: ExtraSensory 47, NOR-TMD 13, Collecty 13;
   - bus: ExtraSensory 26, NOR-TMD 70, Collecty 5;
   - train: ExtraSensory 5, NOR-TMD 17, Collecty 5, SHL 1;
   - subway: NOR-TMD 27, SHL 1;
   - bike: ExtraSensory 21, NOR-TMD 6, Collecty 2;
   - run: ExtraSensory 13, Collecty 1;
   - still and walk: ExtraSensory and SHL only.

   So leave-one-dataset-out is feasible for car and bus, marginal for train, and not possible for still, walk,
   run or subway.
5. **A second test set from NOR-TMD:** reserve ~20% of NOR-TMD users (installations) as a test set for vehicles
   and rail, stratified so each vehicle class has enough users. At 20% of the users with ≥ 5 clean windows that
   is about bus 14, subway 5, train 3–4, car 2–3, bike 1. So car and bike would need a higher share or a minimum
   count per class (e.g. ≥ 5 users each), which the stratification has to trade off against training data. Picked
   once, like `splits/reserved_test_v2.json`, before any v2 model reads NOR-TMD.
6. **Per-user cap per class:** Collecty's top user holds 0.75–0.98 of bus, train and bike (run: one user), so a cap
   per user and class is needed if Collecty is pooled. NOR-TMD's top-user shares are 0.06–0.48 (bike highest; car
   0.30), so a cap there would bind for bike and car only.
7. **Excluded foot labels stay as context:** NOR-TMD `outside` / `inside` (46.4 h) and Collecty `Walk` (62.5 h,
   14 users) stay in the sessions as **unlabelled usable minutes**, for context features and the HMM, never as
   labels.
8. **Dataset identification within class is a required v2 check:** e.g. for bus, a classifier ExtraSensory vs
   NOR-TMD on the model's features. Both candidates add only vehicles (+ bike / run), so dataset identity could
   stand in for "not still / not walk".
9. **Users vs persons:** NOR-TMD has 152 installations for 101 participants. A split by installation may put one
   person on both sides (also for the test set in 5).
10. **Gravity removal** matters only if WAIIS sends linear acceleration (MobilitApp is out either way).

**Note (2026-11-02, experiment 09):** `tmd.candidates` became waiis_bench_v2's window definition. Its minimum
rate is now 24 Hz (was 25), and it resamples per gap-free stretch as v1 does (was: window ± 1 s, deviation 1
above). Rerunning 08's `build_windows.py` / `report.py` therefore gives slightly different numbers from the tables
above, which were built at commit `4bf8f68`.

## Conclusion

**Answer.** Of the three candidates, **NOR-TMD (Android only) is the one worth adding to `waiis_bench_v2`**,
Collecty is a small optional addition, and MobilitApp is out. None of them adds still, walk, iOS or GPS, so v2 still
depends on ExtraSensory and SHL for those.

- **NOR-TMD** adds 12,988 clean v1 windows (219 h) where v1 is weakest: bus 70 users, subway 27, train 17, car
  13, bike 6. The data are clean (timing, no duplicates, top-user share ≤ 0.48) and come from 57 device models,
  three phone positions and two cities. Its iOS half (214 h, 44 devices) records at 5 Hz and doesn't pass the
  25 Hz rule.
- **Collecty** adds 9,026 clean v1 windows (151 h), but from ≤ 5 users for every class except car (13), with top
  users holding 0.75–0.98 of a class. It is usable only after the approved clock repair and dedup, and only with
  a per-user cap.
- **MobilitApp** is not usable (approved): gravity is removed and can't be restored, 56% of files have no sample
  times, and the volume is ~69 h, not 298 h.

**What this changes for v2.**
- **Classes:** v1's 8 classes stay. Tram is the only mode outside v1 with ≥ 10 users across ≥ 2 datasets (37 users,
  NOR-TMD + Collecty); e-scooter (4), boat (1 dataset) and motorbike (no usable user) are not candidates.
- **A dataset shortcut is the main risk.** Both usable sets bring only vehicles plus bike / run, so dataset identity
  can stand for "not still / not walk". The within-class dataset-identification check is required before pooling.
- **Evaluation:** NOR-TMD can give a second, vehicle and rail test set (~20% of its installations, with a minimum
  per class). Leave-one-dataset-out works only for car and bus.
- **iOS convention:** iOS accelerometers (NOR-TMD and ExtraSensory) report with the opposite sign to Android. v1's
  results are not affected (sign-invariant features), but every v2 reader converts iOS to the Android convention and
  m/s², which matters for raw-axis models.
- **GPS:** the candidates' windows can only go through the accelerometer model; 06's fusion stays trained and
  evaluated on ExtraSensory + SHL.

**Limits.**
- One quality check, no models: whether NOR-TMD's vehicle windows actually improve bus and train on the
  ExtraSensory test users is not measured.
- Users are app installations (NOR-TMD: 152 for 101 participants), so a split by installation may put one person on
  both sides.
- Collecty's time of day is unreliable (12-hour clock), and 17,312 of its minute windows are lost to clock
  overlaps.
- The labels were taken as given apart from the acc-only flags; without GPS, vehicles at walking speed and walk
  above 3 m/s could not be checked.

**Next.** The open decisions for v2 (Results): tram, the NOR-TMD test set, per-user caps, iOS track, then the v2
build with the dataset-identification check.
