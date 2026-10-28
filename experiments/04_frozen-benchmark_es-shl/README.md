# 04 Frozen benchmark on ExtraSensory (+ SHL check sets)

## Question

Build one frozen, versioned dataset (`es_bench_v1`) that every model reads (LightGBM, InceptionTime
\+ GPS, LIMU-BERT later), so that models are compared on exactly the same examples. The examples should look as much
as possible like what the WAIIS app sends, as measured in production:
- accelerometer only, ~25 Hz, recorded in fragments;
- GPS: a fix every ~10 s while moving, ~20 s when walking, minutes apart when still; ~23% of fixes have no speed;
  median accuracy ~8 m.

*(Brief from Marc, 2026-10-28. This is a data build, not a model experiment: it has no MLflow runs. Experiment 03's
README calls the planned LIMU-BERT experiment "04"; that one will get a later number.)*

## Plan

1. **Inventory.** What is on disk (absolute location?), snippet length per platform at 25 Hz, how many snippets the
   15 s window drops, location coverage per class, what the SHL location files contain.
   **Stop-point: wait for OK.**
2. **Build `es_bench_v1`** with one command (`build.py` in this folder; shared readers in `src/tmd`).
   - **Accelerometer.** `raw_acc` only. Units are decided per snippet (median |acc| < 3 → g → × 9.80665), as in 02.
     Each snippet is resampled to 25 Hz with an anti-aliasing filter, inside the snippet only. Windows never cross
     snippets. Same cleaning as 02 (time going backwards, span > 40 s, hole > 0.5 s, NaN), without requiring a
     gyroscope. One 15 s window (375 × 3) is cut from each snippet's centre; snippets with less than 15 s are dropped
     and counted per platform.
   - **Labels.** Rules v2 (`tmd.extrasensory`) unchanged. Every minute is kept, with flags `included_v2`,
     `not_carried`, `capped`, `motorbike`, `has_location`. The official example set is `included_v2`. The source
     label columns are kept so that rules can be re-derived.
   - **Split.** Test users from `splits/reserved_test.json`, unchanged. Dev folds are 02's folds by user, with the
     3 gyroscope-less users added without moving anyone. Report class shares per fold.
   - **GPS (ExtraSensory).** Per minute, keyed by user + minute like the labels:
     - from the main CSV, all `location:` / `location_quick_features:` columns;
     - lat/lon from the absolute-location files.

     The GPS track is built per user: one fix per minute, and a new session wherever two minutes are > 2 min apart.
     Each fix has time, lat, lon, speed (`max_speed`, else −1), accuracy (`best_horizontal_accuracy`, else −1) and
     bearing (from consecutive fixes, else −1), plus distance and speed between consecutive fixes.
   - **SHL.** These are separate files in the same format, hips only, accelerometer only, one 15 s window per minute.
     GPS comes from `Hips_Location.txt`, thinned to the WAIIS cadence, with speed removed on ~23% of fixes (fixed
     seed).
     - Users 2–3 are a rail check set, evaluation only.
     - User 1 is an optional training source (all of the metro data, most of the train data).
   - **Format.** Parquet tables `sessions`, `accelerometer`, `gps`, `labels`, joined on user + minute timestamp. The
     build writes a manifest with the sha256 of every output file.
   - **README.** Counts per class and per user, for all minutes and for moving minutes only. Counts per platform and
     phone position, location coverage per class, dropped counts with reasons, and every decision.
   **Stop-point.**

## Results

### Step 1: inventory (2026-10-28) — **stop-point**

```
uv run python inventory.py      # ~12 s; tables in results/inventory/, log in logs/inventory.log
```

Snippet statistics come from 02's per-minute audit of `raw_acc` (`<data>/processed/es_minutes_audit.parquet`, made by
02's `audit_es.py`). The build will recompute them from the raw files. Labels are rules v2 over all 60 users. The
still cap is applied as in 02 (`eval_tables.py`: `cap_resting_still` over the `ok` minutes of all users, seed 0), so
`included_v2` is the same set of minutes as in 02.

#### On disk
- **Absolute location is already in the data root**: `raw/extrasensory/ExtraSensory.per_uuid_absolute_location.zip`
  (2,303,359 bytes, sha256 `09d06c2bf1f69d306b2e529d50d5fe2c8c5d56a832ddd3b64c8ee3e4908b750e`, file date
  2026-10-23), unzipped to `per_uuid_absolute_location/`. It has 60 users with `timestamp, latitude, longitude`. As
  with the cleaned CSVs, the unzip left one folder per file.
  - There is one row for every one of the 377,346 CSV minutes, with lat/lon present in 305,747.
  - Website definition: NaN if there is no location update in the minute's recording, or if all updates have an
    accuracy worse than 200 m. Otherwise the coordinates are those of the latest update among the ones with the best
    accuracy.
  - Source: http://extrasensory.ucsd.edu/data/additional_data_files/ExtraSensory.per_uuid_absolute_location.zip
    (link on the saved dataset page; recorded in LICENSES.md at step 2).
- **LICENSES.md has no ExtraSensory entry yet** (02 used the dataset without adding one). The saved dataset page
  states no licence, only that published work must cite Vaizman et al. 2017. Added at step 2.
- **Free disk on the data volume: 2.8 GB** (`df`, 2026-10-28). See open question 2; 18 GB were free at step 2.

#### ExtraSensory accelerometer at 25 Hz
A 375-sample window at 25 Hz needs a snippet span of at least 14.96 s.

| samples at 25 Hz (clean snippets) | p1 | p5 | p10 | p25 | p50 | p90 |
|---|---|---|---|---|---|---|
| Android (153,140 snippets) | 168 | 233 | 394 | 399 | 400 | 413 |
| iPhone (220,681 snippets) | 572 | 573 | 574 | 578 | 586 | 630 |

- **iPhone** snippets are ~23 s long (800 samples at ≈ 34 Hz). Every clean one fits 15 s.
- **Android** snippets are ~16 s long (800 samples at 50 Hz). 9.5% of the clean ones are shorter than 15 s
  (`es_snippet_len25_hist.csv`). They come almost all from 6 users whose phones sample faster at times (800
  samples at ≈ 80–100 Hz = 8–10 s; `es_users_under_15s.csv`):

  | user | clean snippets | under 15 s | share |
  |---|---|---|---|
  | 1DBB0F6F | 7,107 | 4,055 | 57% |
  | 81536B0A | 6,179 | 3,395 | 55% |
  | 27E04243 | 4,764 | 2,135 | 45% |
  | CCAF77F0 (no gyro) | 8,390 | 1,954 | 23% |
  | 96A358A0 (test) | 5,748 | 1,612 | 28% |
  | F50235E0 (no gyro) | 2,183 | 1,117 | 51% |

**Status per snippet** (02's cleaning on acc alone, then the 15 s rule; the first failing reason counts):

| | all minutes, Android | all, iPhone | included_v2, Android | included_v2, iPhone |
|---|---|---|---|---|
| ok (15 s window) | 138,559 | 220,681 | 58,049 | 91,632 |
| under 15 s | 14,581 | 0 | 7,502 | 0 |
| hole > 0.5 s | 1,710 | 236 | 804 | 166 |
| time not increasing | 1,187 | 0 | 562 | 0 |
| span > 40 s | 66 | 36 | 46 | 16 |
| no file | 282 | 8 | 9 | 0 |
| total | 156,385 | 220,961 | 66,972 | 91,814 |

**By class** (included_v2 minutes; `es_acc_status_by_class.csv`):

| | still | walk | run | bike | car | bus | train |
|---|---|---|---|---|---|---|---|
| included_v2 | 118,604 | 21,241 | 762 | 4,878 | 11,166 | 1,793 | 342 |
| with a 15 s window | 112,927 | 19,468 | 708 | 4,376 | 10,334 | 1,626 | 242 |
| kept | 95.2% | 91.7% | 92.9% | 89.7% | 92.5% | 90.7% | **70.8%** |
| lost to the 15 s rule | 4,562 | 1,478 | 49 | 457 | 714 | 151 | 91 |

- **Train is hit hardest.** 1DBB0F6F (dev, fold 3) is the largest train user, and 91 of their 115 train minutes are
  under 15 s; only 15 remain. Train with a window per user: B7F9D634 97, 3600D531 (test) 69, 33A85C34 37,
  A76A5AF5 24, 1DBB0F6F 15. Dev train goes from 273 to 173 minutes, and fold 3 from 115 to 15. See open question 1.
- **Gyroscope-less users come back** (included_v2 with a window):

  | user | still | walk | run | bike | car | bus |
  |---|---|---|---|---|---|---|
  | 61359772 | 2,682 | 231 | 0 | 227 | 0 | 60 |
  | CCAF77F0 | 1,254 | 20 | 0 | 0 | 28 | 0 |
  | F50235E0 | 847 | 44 | 5 | 19 | 12 | 13 |

  **They already have folds**: 02's `splits/es_users.csv` assigned them as dev users in folds 1, 3 and 2
  (61359772, CCAF77F0 and F50235E0). Nothing needs to be added and nobody moves. The class shares per fold will be
  recomputed with them at step 2.

#### ExtraSensory location
- **The two sources agree.** Lat/lon is present in 305,747 minutes, and `location:num_valid_updates > 0` in 305,338.
  410 minutes have lat/lon but no location features, and 1 has features but no lat/lon.
- **Coverage per class** (included_v2; `es_location_coverage_by_class.csv`):

  | | still | walk | run | bike | car | bus | train | motorbike (check) | all minutes |
  |---|---|---|---|---|---|---|---|---|---|
  | has lat/lon | 82.8% | 78.9% | 80.7% | 93.7% | 88.0% | 77.2% | 91.2% | 100% | 81.0% |
  | iPhone | 92.4% | 95.9% | 100% | 99.4% | 95.8% | 96.1% | 100% | – | 92.1% |
  | Android | 69.3% | 60.7% | 70.2% | 87.3% | 71.6% | 60.3% | 88.0% | 100% | 65.4% |
- **Units and invalid values.**
  - `min_speed` / `max_speed` are in m/s. Medians of `max_speed`: walk 1.5, bike 6.5, car 14, bus 6–7.5 and
    train 21–23 m/s.
  - **Invalid speed is NaN in the CSV, not −1.** No speed or accuracy value is negative. The CSV evidently already
    turned iOS's −1 into NaN. Speed is NaN in 59% (iPhone) and 84% (Android) of the minutes that have location
    updates, so −1 will be written wherever it is NaN.
  - `best_horizontal_accuracy` is in m: median 10 m (iPhone) and 12.4 m (Android), max 199 m (the 200 m cut).
- **Speed missing per class** (minutes with updates; `es_location_speed_by_class.csv`). The production share is 23%.

  | speed valid | still | walk | run | bike | car | bus | train |
  |---|---|---|---|---|---|---|---|
  | iPhone | 33% | 65% | 60% | 89% | 91% | 90% | 92% |
  | Android | 9% | 40% | 42% | 79% | 79% | 58% | 72% |

  In vehicles ExtraSensory is close to production (8–28% missing, Android bus 42%). On foot and in still it is far
  more often missing. I don't plan to blank any extra speeds in ExtraSensory.
- **Cadence.** There is at most one fix per minute: the best-accuracy update within the ~20 s recording. Consecutive
  fixes are a median of 60 s apart (p90 72 s), and 2.4% of the gaps are over 2 min (session breaks). This is
  **sparser than production while moving** (10–20 s) and cannot be made denser. Speed and bearing from consecutive
  fixes therefore span ~60 s, while production spans 10–20 s.

#### SHL location files
`<Pos>_Location.txt` (doc 3.10) has 7 columns: time [ms], ignore, ignore, accuracy [m, 68% radius], latitude,
longitude, altitude. **There is no speed and no bearing column**, so both must be computed from consecutive fixes.
`<Pos>_GPS.txt` holds only satellite IDs, SNR and angles. Hips (`shl_hips_location.csv`):

| | recordings | fixes | hours | fix interval p50 | gaps > 60 s | accuracy p50 | lat/lon NaN |
|---|---|---|---|---|---|---|---|
| User1 | 90 | 1,377,679 | 592 | 1.0 s | 1,012 | 8 m | 0 |
| User2 (4 phones) | 3 | 64,709 | 26 | 1.0 s | 51 | 8 m | 0 |
| User3 (4 phones) | 3 | 36,815 | 17 | 1.0 s | 21 | 16 m | 0 |

The 1 Hz fixes can be thinned to the production cadence. Gaps over 60 s are presumably tunnels, the underground and
indoors (to check per class at step 2).

#### Open questions (for Marc; answered below)
1. **The 15 s rule costs 29% of train**, mostly from 1DBB0F6F (dev fold 3: 115 → 15 train minutes). Options:
   - (a) Keep it as specified.
   - (b) Lower the window to 10 s for everyone, which keeps nearly all Android snippets.
   - (c) Keep 15 s, plus a flagged second window table with the shorter snippets padded and a valid-length column.

   I recommend (a), because every model then sees the same fixed shape. Record the loss, and rely on SHL for train.
2. **Disk.**
   - The ES accelerometer table for every minute with a window is 359,240 × 375 × 3 float32 = 1.6 GB before
     compression. SHL adds ~0.2 GB, and sensor noise compresses poorly.
   - With 2.8 GB free, that is too tight to build safely.
   - Options:
     - free space, e.g. `processed/shl_u1_hips_w10s` (7.3 GB, from 01) or 02's feature parquets (1.6 GB). I won't
       delete anything without your OK;
     - store float16 (~0.8 GB; resolution ~0.008 m/s² near 9.8);
     - store accelerometer only for labelled minutes. That drops ~70k unlabelled minutes, which LIMU-BERT could use
       for pre-training.

   I recommend freeing space and keeping float32 for every minute.
3. **SHL GPS thinning: what sets the cadence?** If the cadence is chosen from the ground-truth label, the fix
   interval itself leaks the class.
   - I propose choosing it from the speed of the 1 Hz track, smoothed over 10 s: ≥ 3 m/s → a fix every 10 s,
     0.5–3 m/s → 20 s, below that → 180 s.
   - Speed is the distance between 1 Hz fixes over time, since SHL has no Doppler speed. It is then set to −1 on 23%
     of the kept fixes (seed 0).
   - Is 180 s right for "minutes apart"?
4. **SHL windows.** I propose one window centred in each clock minute. It gets a label only if all 375 samples have
   the same coarse label (≠ Null). Otherwise it is kept with a `mixed` / `null` flag, like the ES flags. User 1 would
   include its `User1_4phones` hips days, as in 02's day folds.

#### Decisions (2026-10-28, from Marc)
1. **Keep the 15 s window.** Record the loss per user and per class, and say that it is concentrated in 6 Android
   users with 80–100 Hz snippets (step 2 below).
2. **Disk.** Keep float32 and keep the accelerometer for every minute, unlabelled ones included. Delete
   `processed/shl_u1_hips_w10s` only if 01's README has a command to rebuild it.
   - At step 2 the folder was **already gone**: it is not in `processed/`, and nothing in this session deleted it.
     18 GB were free, so nothing was deleted.
   - For the record: 01 rebuilds it with `cache_raw.py` → `make_windows.py` → `make_kshot.py` (01's
     `COMMANDS.md`). `COMMANDS.md` still names the old output path `/Users/ms/data/shl_unimts/`, but
     `shl_common.py:12` writes to `processed/shl_u1_hips_w10s`.
3. **SHL GPS thinned by speed** from the 1 Hz track, with 180 s for slow. Each kept fix gets its speed and bearing
   from the 1 Hz track around the fix time (~3 s), not from consecutive thinned fixes. Speed is then blanked on 23%
   of the kept fixes (seed 0). Natural gaps stay gaps.
   - A **second variant has one fix per minute**, matching ExtraSensory's cadence. Otherwise a model trained on both
     datasets can tell them apart from the fix interval, the same kind of shortcut as `ctx_n` in 02.
4. **SHL windows** as proposed: one 15 s window per clock minute, with mixed or null labels flagged. User 1
   includes the `User1_4phones` hips days.
5. **ExtraSensory location NaN is ambiguous.** Location updates only on a significant change, so a still phone can
   give none. NaN can also mean location off or no fix, all updates worse than 200 m, or coordinates hidden near
   home (~500 m, per the paper).
   - Never forward-fill or impute: NaN stays NaN, and −1 in the track format.
   - Add a per-minute `location_missing_reason`, and check whether `withheld` occurs.
   - Report coverage and reasons per class, and note that missing location correlates with still.
   - Sessions come from minute timestamps (gap > 2 min), not from fixes.
6. Also: add ExtraSensory to LICENSES.md, and in 03's README replace "04" (LIMU-BERT) with "a later experiment".
   Both done.

### Step 2: build of `es_bench_v1` (2026-10-28) — **stop-point**

```
uv run python build.py --workers 8      # ~2 min; <data>/processed/es_bench_v1/, results/tables/*.csv, logs/build.log
uv run python build.py --sources shl_user23   # rebuild check: all 15 sha256 identical to the full build
```

**Shared code added or changed** (`src/tmd`):
- `bench.py`: the reader (`read`, `examples`, `windows`, `manifest`, `verify`).
- `gps.py`: haversine, bearing, sessions, speed and bearing from a dense track, cadence thinning.
- `resample.py`: `es_acc_to_25hz` (the acc-only path) and `centre`. The cleaning checks moved into
  `es_snippet_problem` and the unit rule into `to_ms2`. **02's `es_snippet_to_25hz` is unchanged**: its output was
  identical to the committed version on 90 sampled snippets.
- `extrasensory.py`: `read_location`, `location_missing_reason`.
- `shl.py`: `read_location`.

#### What was built

`<data>/processed/es_bench_v1/` is 1.77 GB, and `manifest.json` holds the parameters, the git commit and the
sha256, size and rows of every file. One folder per source:

| source | tables | role |
|---|---|---|
| `extrasensory` | `labels`, `sessions`, `accelerometer`, `gps_per_minute`, `location_features` | main benchmark; dev folds + reserved test users |
| `shl_user1` | `labels`, `sessions`, `accelerometer`, `gps_waiis_cadence`, `gps_per_minute` | optional training source (metro, most train) |
| `shl_user23` | same as `shl_user1` | rail check set, evaluation only |

- **Join key** is `user, ts` (minute timestamp, unix s). For SHL, `ts` is the start of the clock minute.
- **`labels`** has one row per minute, and every minute is kept. Columns:
  - split (`dev` / `test` / `rail_check`), `fold`, `session`, `cls`, `y`;
  - flags: `included_v2` (SHL: `included`), `has_acc`, `acc_status`, `has_location`, `location_missing_reason`,
    `not_carried`, `capped`, `motorbike`;
  - rule outputs (`reason`, `position`, `resting`, `moving`, `PHONE_*`);
  - ExtraSensory also keeps all source label columns: 51 `label:*` (cleaned CSV) and 116 `orig:*` (original
    labels), so rules can be re-derived.
- **`accelerometer`** has one row per minute that has a window. `x`, `y`, `z` are fixed-size lists of 375 float32
  values in m/s² at 25 Hz. ExtraSensory adds `unit` (detected `g` or `m/s2`), `raw_n`, `raw_rate_hz`, `span_s` and
  `start_offset_s`; SHL adds `t_start_ms`.
- **The official example set** is `tmd.bench.examples(source)`: included minutes that have a window. That is
  149,681 ExtraSensory minutes, 23,099 for SHL User 1 and 2,366 for SHL Users 2–3.
- **GPS tracks** have the columns `user, ts, session, time, lat, lon, speed, accuracy, bearing` (−1 = unknown),
  plus `dist_prev_m`, `dt_prev_s` and `speed_prev_mps` from the previous fix in the same session. SHL tracks also
  have `rec`, `speed_10s` and `speed_blanked`.
  - `gps_per_minute`: at most one fix per minute, which is ExtraSensory's native cadence.
    - ExtraSensory: the minute's absolute lat/lon, `speed` = `location:max_speed` and `accuracy` =
      `best_horizontal_accuracy`. `time` is the minute timestamp, because the fix's own time is not published.
      Bearing is computed from the previous fix.
    - SHL: inside the minute's 15 s window, the best-accuracy fix, latest on ties (the ExtraSensory rule). `time`
      is the real fix time.
  - `gps_waiis_cadence` (SHL only): the 1 Hz track thinned to the production cadence. A fix is kept when it is at
    least *interval* after the last kept fix. The interval comes from the speed over ~10 s at the candidate fix:
    ≥ 3 m/s → 10 s, ≥ 0.5 m/s → 20 s, otherwise 180 s.
  - In both SHL variants, speed and bearing come from the 1 Hz track: the first and last fix within ±1.5 s, never
    across a gap > 5 s. Speed **and bearing** are then set to −1 on 23% of the kept fixes (`rng(0)`, separately
    per table), as phones record a fix without speed (decision 2026-10-28, below).
  - **Use `gps_per_minute` when a model trains on ExtraSensory and SHL together.** With `gps_waiis_cadence`, the
    fix interval alone separates the datasets.
- **`sessions`**: per user, minutes sorted by time, with a new session at each gap > 2 min between minutes (not
  between fixes). The columns are start, end, minute count, minutes with a window and minutes with a location.
  - ExtraSensory has 8,729 sessions (median 4 minutes, max 7,875). SHL User 1 has 90 and Users 2–3 have 6, one per
    recording.
- **`location_features`** (ExtraSensory): all 17 `location:*` / `location_quick_features:*` columns plus lat/lon,
  per minute, unchanged (NaN kept).

#### ExtraSensory counts

Per class (`es_class_counts.csv`). "Examples" are included_v2 minutes with a 15 s window.

| | still | walk | run | bike | car | bus | train |
|---|---|---|---|---|---|---|---|
| minutes with that class (any rule outcome) | 265,154 | 22,323 | 1,090 | 5,020 | 11,252 | 1,794 | 344 |
| included_v2 | 118,604 | 21,241 | 762 | 4,878 | 11,166 | 1,793 | 342 |
| **examples** | **112,927** | **19,468** | **708** | **4,376** | **10,334** | **1,626** | **242** |
| iPhone / Android | 69,327 / 43,600 | 10,960 / 8,508 | 269 / 439 | 2,594 / 1,782 | 7,548 / 2,786 | 841 / 785 | 93 / 149 |
| dev / test | 94,148 / 18,779 | 15,862 / 3,606 | 585 / 123 | 3,477 / 899 | 8,432 / 1,902 | 1,357 / 269 | 173 / 69 |
| users | 60 | 59 | 22 | 23 | 49 | 28 | 5 |

- **Per user:**
  - all minutes: `es_user_class_all_minutes.csv` (unlabelled minutes included);
  - examples: `es_user_class_examples.csv` (with platform and split);
  - moving classes only: `es_user_class_examples_moving.csv`.

  Moving examples per user range from 1 (27E04243) and 2 (96A358A0, test) to 1,612 (0E6184E1).
- **Platform × position** (examples; `es_platform_position_class.csv`):

  | | none reported | in_pocket | in_hand | in_bag | on_table | total |
  |---|---|---|---|---|---|---|
  | iPhone | 52,650 | 13,347 | 9,028 | 5,843 | 10,764 | 91,632 |
  | Android | 38,120 | 7,963 | 3,722 | 3,020 | 5,224 | 58,049 |

**Dropped, with reasons.**
- **Label rules** (all 377,346 minutes; `es_label_reason.csv`):

  | reason | minutes |
  |---|---|
  | ok | 221,285 |
  | unlabelled | 70,353 |
  | `drop:SLEEPING` | 83,055 |
  | `drop:not_carried` | 1,078 |
  | `drop:ON_A_PLANE` | 630 |
  | `drop:ELLIPTICAL_MACHINE` | 233 |
  | `drop:ELEVATOR` | 200 |
  | `drop:TREADMILL` | 164 |
  | `drop:SKATEBOARDING` | 131 |
  | `check:MOTORBIKE` | 86 |
  | `drop:STATIONARY_BIKE` | 86 |
  | conflict | 18 |
  | `drop:ON_A_BOAT` | 18 |
  | `drop:WHEELCHAIR` | 9 |

  The still cap then removes 62,499 `ok` still minutes (`capped`).
- **Accelerometer**, included_v2 minutes (`es_acc_status_by_class.csv`; all minutes: `es_acc_status_all.csv`):

  | | still | walk | run | bike | car | bus | train | total |
  |---|---|---|---|---|---|---|---|---|
  | under 15 s | 4,562 | 1,478 | 49 | 457 | 714 | 151 | 91 | 7,502 |
  | hole > 0.5 s | 716 | 129 | 2 | 36 | 74 | 10 | 3 | 970 |
  | time not increasing | 344 | 162 | 3 | 6 | 40 | 6 | 1 | 562 |
  | span > 40 s | 47 | 3 | 0 | 3 | 4 | 0 | 5 | 62 |
  | no file | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 9 |
  | kept | 95.2% | 91.7% | 92.9% | 89.7% | 92.5% | 90.7% | **70.8%** | 94.3% |

  All 7,502 minutes lost to the 15 s rule are Android. Over all minutes there are 359,240 windows.
- **The 15 s loss is concentrated in 6 Android users whose phones record 800 samples at 80–100 Hz** (8–10 s
  snippets) for part of the time. These 6 lose 7,293 of the 7,502 included minutes (97%); every other user loses
  ≤ 100. Per user and class (`es_15s_loss_by_user.csv`, included_v2):

  | user | split | included | under 15 s | still | walk | run | bike | car | bus | train |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 81536B0A | dev 3 | 3,416 | 1,972 | 1,075 | 44 | 0 | 427 | 426 | 0 | 0 |
  | 1DBB0F6F | dev 3 | 2,927 | 1,951 | 1,050 | 704 | 44 | 0 | 36 | 26 | **91** |
  | CCAF77F0 (no gyro) | dev 3 | 2,641 | 1,337 | 879 | 269 | 0 | 0 | 188 | 1 | 0 |
  | F50235E0 (no gyro) | dev 2 | 2,050 | 1,037 | 799 | 135 | 5 | 28 | 54 | 16 | 0 |
  | 27E04243 | dev 3 | 1,113 | 600 | 293 | 262 | 0 | 0 | 0 | 45 | 0 |
  | 96A358A0 | test | 572 | 396 | 307 | 26 | 0 | 0 | 0 | 63 | 0 |

  (Folds from `es_users.csv`.)
  - **4 of the 6 are in dev fold 3.**
  - The 15 s rule causes **91 of the 100 dropped train minutes**, all from 1DBB0F6F.
  - It also costs **all 63 bus minutes of test user 96A358A0**. Test bus now has 5 users with examples (24–119
    each), not 6.

**Folds** (examples; `es_fold_class_counts.csv`, shares in `es_fold_class_share.csv`). These are 02's folds; the 3
gyroscope-less users were already in folds 1, 3 and 2, so nobody moved.

| fold | users | still | walk | run | bike | car | bus | train |
|---|---|---|---|---|---|---|---|---|
| 0 | 10 | 17,610 | 3,146 | 42 | 546 | 1,641 | 190 | 0 |
| 1 | 9 | 24,842 | 4,160 | 287 | 881 | 2,460 | 244 | 97 |
| 2 | 9 | 16,724 | 2,710 | 158 | 801 | 1,588 | 436 | 24 |
| 3 | 10 | 11,177 | 2,381 | **9** | 375 | **516** | 162 | **15** |
| 4 | 10 | 23,795 | 3,465 | 89 | 874 | 2,227 | 325 | 37 |
| test | 12 | 18,779 | 3,606 | 123 | 899 | 1,902 | 269 | 69 |

Share of each class's examples per dev fold: still 10–22%, walk 12–21%, run 1–41%, bike 9–20%, car 5–24%,
bus 10–27%, train 0–40%. Fold 3 is now thin in run, car and train, because 4 of the 6 high-rate
users are in it. Test holds 17–21% of every class except train (28%, one
user).

**Location** (examples; `es_location_by_class.csv`):

| | still | walk | run | bike | car | bus | train |
|---|---|---|---|---|---|---|---|
| has location | 82.8% | 78.6% | 80.1% | 93.6% | 87.8% | 78.0% | 87.6% |
| iPhone | 92.4% | 95.9% | 100% | 99.5% | 95.8% | 96.1% | 100% |
| Android | 67.6% | 56.4% | 67.9% | 85.0% | 66.3% | 58.6% | 79.9% |
| missing: `no_updates` | 17.2% | 21.4% | 19.9% | 6.4% | 12.2% | 22.0% | 12.4% |

- **Missing reasons, all 377,346 minutes:**
  - present 305,747;
  - `no_updates` 71,598;
  - `withheld` 1 (an unlabelled minute);
  - `poor_accuracy` 0;
  - `unknown` 0.
- **`withheld` effectively does not occur.** When coordinates are missing, the relative location features are
  missing too (except in that 1 minute). So the home hiding is either not in the released data, or it removed the
  features as well; we can't tell which.
- **`poor_accuracy` can't be told apart from `no_updates`.** The features never have an accuracy over 200 m (max
  199.3 m), so the CSV also seems to drop updates worse than 200 m. A minute whose only updates were poor therefore
  looks like a minute with no updates.
- **Missing location and still (possible shortcut).**
  - **In ExtraSensory the link is weak.** Still minutes miss location 17% of the time, which is less than walk
    (21%) and bus (22%) and more than bike (6%) and car (12%).
  - **The strong link is platform:** Android misses 32% of minutes, iPhone 8%.
  - **In SHL it is strong** (SHL counts below): no fix within the window for 54–61% of still minutes and 51–87% of
    train and subway minutes, against 0–20% for walk, run, bike, car and bus. Indoors and underground mean no fix.
  - A model that uses "has a fix" or the fix interval can learn still and rail from that. Experiments that use GPS
    should report results with and without the location-missing indicator.
- **Speed** is −1 on 67.5% of ExtraSensory fixes (the CSV has NaN; no negative values occur), and bearing on 29.1%
  (first fix of a session, or a move under 1 m). Production has ~23% without speed.

#### SHL counts

| | still | walk | run | bike | car | bus | train | subway | total |
|---|---|---|---|---|---|---|---|---|---|
| User 1 (`shl_user1`) | 3,366 | 3,015 | 1,047 | 2,844 | 4,058 | 3,028 | 3,353 | 2,388 | 23,099 |
| User 2 (`shl_user23`) | 197 | 240 | 14 | 136 | 255 | 68 | 133 | 174 | 1,217 |
| User 3 (`shl_user23`) | 290 | 188 | 31 | 63 | 84 | 82 | 227 | 184 | 1,149 |

- **Minutes and labels.**
  - User 1 has 36,968 minutes in 90 recordings (`release/User1` without 220617, 260617 and 270617, plus
    `User1_4phones`, as in 02). Users 2–3 have 3,024 minutes in 6 recordings. Platform `android`, position `hips`.
  - Label status, User 1: ok 23,100; null 13,056; mixed 694 (a transition inside the window); edge 118 (the window
    doesn't fit in the recording).
  - Accelerometer: 25 windows fall in a NaN gap, 1 of them on an `ok` minute.
  - Users 2–3: ok 2,366, null 586, mixed 65, edge 7, NaN gap 2.
- **Train and subway in User 1** have 3,353 and 2,388 examples (56 h and 40 h), against 242 ExtraSensory train
  examples.
- **GPS** (`shl_user1_gps_by_class.csv`, `shl_user23_gps_by_class.csv`; the 1 Hz track has 1,377,605 fixes for
  User 1 and 101,516 for Users 2–3). User 1:

  | `gps_waiis_cadence` | still | walk | run | bike | car | bus | train | subway |
  |---|---|---|---|---|---|---|---|---|
  | fixes | 874 | 7,565 | 3,424 | 14,478 | 18,070 | 10,787 | 7,994 | 3,119 |
  | interval to previous, p50 | 71 s | 20 s | 20 s | 10 s | 10 s | 10 s | 10 s | 10 s |
  | speed −1 | 25% | 24% | 23% | 23% | 23% | 24% | 26% | 24% |
  | accuracy p50 | 8 m | 6 m | 6 m | 6 m | 8 m | 8 m | 16 m | 12 m |

  - Totals: 80,892 fixes (User 1) and 5,550 (Users 2–3). 22.9% of speeds were blanked, and 24.2% are −1 in total
    (some fixes have no 1 Hz neighbour).
  - The cadence comes out as specified: 10 s in vehicles and on the bike, 20 s on foot. Still fixes are a median of
    71–87 s apart, and up to 180 s.

  | `gps_per_minute` | still | walk | run | bike | car | bus | train | subway |
  |---|---|---|---|---|---|---|---|---|
  | examples with a fix, User 1 | 39.4% | 85.9% | 99.8% | 99.3% | 83.2% | 89.7% | 49.0% | 30.6% |
  | examples with a fix, Users 2–3 | 46.4% | 86.0% | 100% | 89.9% | 91.7% | 80.0% | 38.1% | 13.4% |

  User 1 has 23,645 per-minute fixes, with a median of 60 s between them (as in ExtraSensory) and 22–25% of speeds
  at −1.

#### Deviations and decisions taken in the build
- **More tables than the four named in the brief.**
  - GPS has two SHL variants (Marc, decision 3), and ExtraSensory's track is called `gps_per_minute` so that it
    matches the SHL variant with the same cadence.
  - The location columns are a separate `location_features` table rather than part of `labels`.
- **The processed folder is named `es_bench_v1/`**, not by its parameters (CLAUDE.md: files in `processed/` are
  named by their parameters). The version stands for the parameter set, which is in `manifest.json`.
- **`y` is 0 for every minute that is not included** (ExtraSensory: not `included_v2`, so capped still minutes are
  0 too; SHL: not `ok`). In 02, `y` was set before the cap. `cls` keeps the rule's class name.
- **The still cap** is drawn as in 02 (`cap_resting_still` over the `ok` minutes of all 60 users, seed 0), so
  `included_v2` is exactly 02's minute set: 158,786 minutes, the same as at step 1.
- **ExtraSensory window position.** The window is the centre of the resampled snippet (`start_offset_s`), not
  aligned to the minute timestamp. SHL windows are centred on the clock minute (`ts` + 30 s).
- **SHL resampling.** Each NaN-free stretch is resampled 100 → 25 Hz as a whole (as 02's `make_features.py`), then
  the window is cut from it. The window must lie inside one stretch.
- **SHL cadence.** The interval is set by the speed at the candidate fix, so a fix arrives 10 s after moving off,
  not 180 s later. The phone's real trigger (distance filter or activity) is unknown.
- **SHL speed blanking** sets `speed` and `bearing` to −1 (Marc, 2026-10-28: production fixes on iPhone and
  Android without speed have no bearing either). The first build kept the bearing; the rebuild below replaced it.
- **Bearing is −1** for the first fix of a session, and wherever the move is under 1 m (ExtraSensory: from the
  previous fix; SHL: over ±1.5 s).
- **No MLflow run.** This is a data build. Model experiments that read it log `DataInput(name="extrasensory_dev" /
  "extrasensory_test", features=..., extra=(("benchmark", "es_bench_v1"),))`. The split files are still 02's, so
  `tracking.SPLITS` needs no new entry.
- **The manifest says `git_dirty: true`** (build from commit `96f1435` with this work uncommitted). The data don't
  depend on the commit; rebuilding after the commit gives the same sha256 (checked on `shl_user23`) and a clean
  manifest.
- The build prints a pandas `PerformanceWarning` (fragmented frame in `es_user`). It is harmless.

#### Output files (sha256 from `manifest.json`)

| file | rows | sha256 |
|---|---|---|
| `extrasensory/accelerometer.parquet` | 359,240 | `148a9f030bd2db44c823f6b4f187778ad1bac5cd835c1e4bd1b19f1970236e6d` |
| `extrasensory/gps_per_minute.parquet` | 305,747 | `308b2a7dfd04e2599049df9da20e35e640ad0de45ef4f44a559d1622ec6a1e32` |
| `extrasensory/labels.parquet` | 377,346 | `1a6579d6859ce792b77230a6d46b4d883b8fffab2409ace666bad474b9230a04` |
| `extrasensory/location_features.parquet` | 377,346 | `cc8e063862eaa46e331ca5990ed238e4c99cb5ad7a091f39635c262768228bbd` |
| `extrasensory/sessions.parquet` | 8,729 | `c5c2090161e0aeb3f44f5dbe7677691bc1f0b94372848b68be7bf9e443832b17` |
| `shl_user1/accelerometer.parquet` | 36,825 | `d1209fc862bd948ec50915121bc30625c36c1478e6e114609d88a1324e47fff1` |
| `shl_user1/gps_per_minute.parquet` | 23,645 | `9822639040bd2a869fa9e1f385cb2446af2bbf5c8ac80d288d59823faa9ce263` |
| `shl_user1/gps_waiis_cadence.parquet` | 80,892 | `36eb17667844970e7e76cd3dcc778ae3abc07a5e4549a60429ff18a91706c1be` |
| `shl_user1/labels.parquet` | 36,968 | `0eaf44ee479d4f9595a4f67d1eb4f0283199fa6b438633e761577977780cbd3b` |
| `shl_user1/sessions.parquet` | 90 | `22f67e501301acb6aea7372829308cacdd280e3265367f710273c1593f82b35d` |
| `shl_user23/accelerometer.parquet` | 3,015 | `debe6c6833c52bc5aebcec25ca93dc34bdae2ab21353f9d2579c0d35a3c0c4c7` |
| `shl_user23/gps_per_minute.parquet` | 1,749 | `a97a97544d1b63edb0c3ab76273f607301134cdc8af70effae1a754f29eb88d5` |
| `shl_user23/gps_waiis_cadence.parquet` | 5,550 | `eea127e414ec6f56dbb50d1a2a6a8f6e5b2d951f9e95478262f363a77d538631` |
| `shl_user23/labels.parquet` | 3,024 | `6920d88a6631be40f89278a321fe82f4de8c617c9ee2139f580ff9895e814684` |
| `shl_user23/sessions.parquet` | 6 | `890c107bf7aa3c7a0e0e908fd1ac1ca7e2bce9c4e7fdaf5419f56de5efeab686` |

#### Open questions (step 2)
1. ~~Bearing on fixes without speed.~~ Answered: blank the bearing wherever speed is blanked (rebuilt, see below).
2. **Fold 3 is thin** in run (9), car (516) and train (15) after the 15 s rule. The folds are 02's and fixed, but
   per-fold metrics for run and train will be noisy there.
3. ~~Commit, then rebuild for a clean manifest.~~ Answered: yes (see "Rebuild" below).
4. **ExtraSensory bearing without speed.** ExtraSensory's bearing is computed from consecutive fixes. It is still
   set on fixes whose speed is −1 (67.5% of fixes have no speed). To match production, it could be blanked there
   too; not done, because the brief asked for a bearing from consecutive fixes.
