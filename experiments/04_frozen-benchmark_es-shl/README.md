# 04 Frozen benchmark on ExtraSensory (+ SHL check sets)

## Question

Build one frozen, versioned dataset, **`waiis_bench_v1`**, that every model reads (LightGBM, InceptionTime + GPS,
LIMU-BERT later), so that models are compared on exactly the same examples. The examples should look as much as
possible like what the WAIIS app sends, as measured in production:
- accelerometer only, ~25 Hz, recorded in fragments;
- GPS: a fix every ~10 s while moving, ~20 s when walking, minutes apart when still; ~23% of fixes have no speed;
  median accuracy ~8 m.

*(This is a data build, not a model experiment: it has no MLflow runs.)*

## Plan

1. Inventory of the raw data (`inventory.py`). **Stop-point.**
2. Build the benchmark with one command (`build.py`; shared readers in `src/tmd`), verify it, and document every
   decision (`decision_register.md`). **Stop-point.**
3. Re-pick the ExtraSensory test users on the benchmark's counts (method in `splits/README.md`).
4. Final build, verification and documentation. **Stop-point.**

## Results

**The dataset is `waiis_bench_v1`**: `<data>/processed/waiis_bench_v1/`, 23 Parquet files, 4.8 GB.
- Built from commit `e51ea79` in a clean git worktree; `manifest.json` holds the parameters and the sha256 of
  every file.
- Read it with `tmd.bench`.
- Every decision that shaped it, with its effect, is in **`decision_register.md`**.
- The step-by-step log of the stop-points is in this README at commit `51e9319`.

```
uv run python inventory.py              # raw-data inventory; results/inventory/, logs/inventory.log
uv run python build.py --workers 8      # ~2 min; <data>/processed/waiis_bench_v1/, results/tables/, logs/build.log
uv run pytest tests                     # reserved and excluded users never reach a pre-training accessor
uv run python register_numbers.py       # results/tables/decision_details.csv (numbers behind the register)
```

### Contents

| source | tables | role |
|---|---|---|
| `extrasensory` | `labels`, `sessions`, `accelerometer`, `gps_per_minute`, `location_features`, `snippets` | main benchmark: dev folds + reserved test users |
| `shl_user1` | `labels`, `sessions`, `accelerometer`, `gps_waiis_cadence`, `gps_per_minute`, `gps_1hz`, `gps_unblanked`, `stream`, `accelerometer_other_positions` | optional training source (metro, most train) |
| `shl_user23` | as `shl_user1`, without `accelerometer_other_positions` | rail check set, evaluation only |

- **Join key** `user, ts`: the minute timestamp [unix s].
- **`labels`**: one row per minute, every minute kept, with flags and (ExtraSensory) all source label columns.
- **`official`** (in `labels`): the official example set. `tmd.bench.examples(source)` returns it.
  - official = included by the label rules, has a 15 s window, and none of `acc_implausible`, `label_implausible`,
    `user_excluded`.
- **`accelerometer`**: one 15 s window per minute (375 × 3 float32, m/s², 25 Hz, gravity included).
- **Never official, explicit accessors only** (reserved test users excluded unless `include_reserved=True`; excluded
  users always): full snippets incl. short and time-repaired ones, the SHL hips stream, SHL User 1's other phone
  positions, the SHL 1 Hz track and unblanked speeds.
- **GPS**: speed, accuracy and bearing are −1 when unknown; lat/lon are never imputed.
  - Models that train on ExtraSensory and SHL together use `gps_per_minute`. With `gps_waiis_cadence`, the fix
    interval alone separates the datasets.

### Official examples

| ExtraSensory | still | walk | run | bike | car | bus | train | all |
|---|---|---|---|---|---|---|---|---|
| dev, iPhone | 56,390 | 8,257 | 69 | 1,654 | 6,197 | 798 | 93 | 73,458 |
| dev, Android | 36,340 | 4,852 | 394 | 1,464 | 1,831 | 550 | 97 | 45,528 |
| test, iPhone | 13,041 | 1,372 | 38 | 646 | 1,365 | 47 | 0 | 16,509 |
| test, Android | 6,245 | 2,310 | 43 | 230 | 765 | 240 | 52 | 9,885 |
| **all** | **112,016** | **16,791** | **544** | **3,994** | **10,158** | **1,635** | **242** | **145,380** |
| users | 59 | 58 | 22 | 23 | 48 | 28 | 5 | |

| SHL | still | walk | run | bike | car | bus | train | subway | all |
|---|---|---|---|---|---|---|---|---|---|
| User 1 (`shl_user1`) | 3,366 | 3,011 | 1,030 | 2,842 | 4,058 | 3,028 | 3,353 | 2,388 | 23,076 |
| Users 2–3 (`shl_user23`) | 487 | 427 | 45 | 199 | 339 | 150 | 360 | 358 | 2,365 |

Per user, per position and per fold: `results/tables/` (`es_class_counts.csv`, `es_fold_class_counts.csv`,
`*_user_class_examples.csv`, …) and `results/tables/decision_details.csv`.

### Split

All current splits (also listed in `splits/README.md` at the project root):

| split | file | `tracking.SPLITS` name |
|---|---|---|
| test users: 12 ExtraSensory users, SHL Users 2–3 (rail check), approved 2026-10-28 | `splits/reserved_test_v2.json` (project root) | `extrasensory_test_v2`, `shl_user23_test_v2` |
| ExtraSensory dev: 5 folds by user | `splits/es_users_v2.csv` (here) | `extrasensory_dev_v2` |
| SHL User 1 dev: 5 folds of whole days | `splits/shl_days.csv` (here; a byte-identical copy of 02's file) | `shl_user1_dev` |

- **11 of the 12 test users were dev users in experiment 02** (training and model choices there).

| test user | platform | examples | still | walk | run | bike | car | bus | train |
|---|---|---|---|---|---|---|---|---|---|
| 098A72A5 | iPhone | 3,521 | 3,046 | 187 | 32 | 114 | 142 | 0 | 0 |
| 0BFC35E2 | Android | 1,919 | 1,349 | 308 | 34 | 106 | 113 | 9 | 0 |
| 1DBB0F6F | Android | 768 | 681 | 66 | 1 | 0 | 0 | 5 | 15 |
| 24E40C4C | iPhone | 2,776 | 1,897 | 373 | 0 | 56 | 420 | 30 | 0 |
| 27E04243 | Android | 507 | 506 | 0 | 0 | 0 | 0 | 1 | 0 |
| 33A85C34 | Android | 2,584 | 1,777 | 437 | 0 | 124 | 209 | 0 | 37 |
| 665514DE | Android | 2,239 | 1,016 | 1,056 | 8 | 0 | 20 | 139 | 0 |
| 8023FE1A | iPhone | 5,496 | 4,570 | 236 | 0 | 354 | 336 | 0 | 0 |
| A5CDF89D | iPhone | 2,496 | 2,138 | 189 | 0 | 0 | 169 | 0 | 0 |
| A7599A50 | Android | 1,868 | 916 | 443 | 0 | 0 | 423 | 86 | 0 |
| B09E373F | iPhone | 1,777 | 1,137 | 249 | 6 | 122 | 263 | 0 | 0 |
| ECECC2AB | iPhone | 443 | 253 | 138 | 0 | 0 | 35 | 17 | 0 |

- Test users with ≥ 5 examples: still 12, walk 11, car 10, bike 6, bus 6, run 4, train 2.
- iPhone share of examples: test 62.5%, dev 61.7%. Location coverage: test 85.6%, dev 82.7%.
- Dev folds hold 8–10 users with examples each. Train is in folds 0, 2 and 3 only.

| dev fold | still | walk | run | bike | car | bus | train |
|---|---|---|---|---|---|---|---|
| 0 | 23,745 | 2,889 | 288 | 342 | 1,431 | 174 | 69 |
| 1 | 16,873 | 2,405 | 104 | 892 | 1,807 | 277 | 0 |
| 2 | 17,083 | 2,498 | 19 | 600 | 1,791 | 207 | 97 |
| 3 | 22,462 | 2,881 | 23 | 768 | 1,405 | 279 | 24 |
| 4 | 12,567 | 2,436 | 29 | 516 | 1,594 | 411 | 0 |

### Location and missing speed

| official examples with a fix | still | walk | run | bike | car | bus | train | subway |
|---|---|---|---|---|---|---|---|---|
| ExtraSensory iPhone | 92% | 97% | 100% | 99% | 96% | 96% | 100% | – |
| ExtraSensory Android | 68% | 58% | 68% | 86% | 65% | 59% | 80% | – |
| SHL User 1 (per minute) | 39% | 86% | 100% | 99% | 83% | 90% | 49% | 31% |

- Missing location is mainly a platform effect in ExtraSensory, and a still / rail effect in SHL (indoors,
  underground).
- Missing speed is class-dependent in both datasets: highest when still, lowest in vehicles. SHL blanking is
  driven by speed (p = 0.95 · exp(−v / 1.954 m/s), 23% at the WAIIS cadence).
- Per class: `results/tables/es_speed_missing_by_class.csv` and `shl_user1_gps_by_class.csv`.

### Reporting rule for test results

- ExtraSensory test results are reported **overall and per platform** (iPhone, Android).
- "Without location" is not a separate breakdown: only 10 iPhone test examples lack a fix, so it would be
  Android-only.
- **Per-class results come with user-bootstrap 95% confidence intervals**: `tmd.evaluate.bootstrap_users_recall`
  for per-class recall, `bootstrap_users` for macro-F1, 1,000 resamples of the test users. Train (2 users, 52
  examples, Android only) and run (4 users, 81 examples) are tiny.
- SHL Users 2–3 are reported separately, as the rail check.

### Cross-validation rule (all future experiments)

- **ExtraSensory cross-validation metrics are computed on the pooled out-of-fold predictions:** macro-F1 and
  per-class recall over all 5 dev folds at once, with a user-bootstrap 95% CI (`tmd.evaluate.bootstrap_users`,
  `bootstrap_users_recall`). They are not a mean of per-fold scores.
- **Per-fold scores are reported only as spread**, e.g. min–max over folds.
- **Why:** the folds differ in size (17,553–28,938 examples). Folds 1 and 4 have no train, so a per-fold macro-F1
  there is computed over a different set of classes.

### Verification and rebuilding

- **Reference:** `results/manifests/waiis_bench_v1.json`, a committed copy of the build's `manifest.json`.
  `tmd.bench.verify()` checks every data file on disk (missing, extra, sha256) and the on-disk manifest against it.
- To rebuild from the build commit in a clean worktree (data root = the worktree's parent's `data/`, or `$TMD_DATA`):

  ```
  git worktree add ../tmd-waiis-bench e51ea79
  cd ../tmd-waiis-bench && uv run python experiments/04_frozen-benchmark_es-shl/build.py
  ```

- **Rebuild check (2026-10-29):** the full rebuild from `35257b4` in a separate worktree gave 23 data files whose
  sha256 and row counts all equal the manifest of the build at commit `cf1085c` (in `results/manifests/`), with
  identical build parameters. Only `manifest.json` itself differs (version, commit, date). The count tables in
  `results/tables/` are identical too.
- **Rebuilt from `e51ea79` (2026-10-29)**, so the manifest records the SHL split's new path (04's copy of `shl_days.csv`): all 23 data files keep their sha256, and only the commit and the `shl_splits` path changed in the manifest.
- `waiis_bench_v1` is the only benchmark on disk.

### Version

Built during development as es_bench_v4 (commit cf1085c); es_bench_v1–v3 were intermediate builds, never used by any model, deleted.

### Open questions
1. The 9 questions for WAIIS in `decision_register.md` (recording schedule, sampling, units, platforms, carrying,
   modes, GPS, labels, batching).
2. `label_implausible` removes 60% of iPhone run (161 of 269 windows). Is ExtraSensory's iPhone run label noise,
   or quiet treadmill-like running?

## Conclusion

**The deliverable is `waiis_bench_v1`**: the prepared files, the split, the counts and location coverage below.
Every decision taken is in [`decision_register.md`](decision_register.md); the numbers behind it are in
`results/tables/decision_details.csv`.

### Prepared files

- **Location:** `<data>/processed/waiis_bench_v1/`, 23 Parquet files, 4.8 GB.
- **Build:** from commit `e51ea79` with one command (`build.py`). A full rebuild reproduces every data file's
  sha256.
- **Reference:** `results/manifests/waiis_bench_v1.json`, a committed copy of the manifest. `tmd.bench.verify()`
  checks the files on disk against it.
- **Reading:** `tmd.bench.examples(source)` returns the official example set (the `official` column). The other
  tables are listed under Results → Contents.

| source | official examples | role |
|---|---|---|
| `extrasensory` | 145,380 (59 users) | main benchmark: 5 dev folds + 12 reserved test users |
| `shl_user1` | 23,076 | optional training source (all metro, most train) |
| `shl_user23` | 2,365 | rail check, evaluation only |

### Split

| split | file | `tracking.SPLITS` name |
|---|---|---|
| test users: 12 ExtraSensory users, SHL Users 2–3 | `splits/reserved_test_v2.json` (project root) | `extrasensory_test_v2`, `shl_user23_test_v2` |
| ExtraSensory dev: 5 folds by user | `splits/es_users_v2.csv` (here) | `extrasensory_dev_v2` |
| SHL User 1 dev: 5 folds of whole days | `splits/shl_days.csv` (here) | `shl_user1_dev` |

How the test users were picked: `splits/README.md`.

### Counts per class (official examples)

| ExtraSensory | still | walk | run | bike | car | bus | train | all |
|---|---|---|---|---|---|---|---|---|
| dev, iPhone | 56,390 | 8,257 | 69 | 1,654 | 6,197 | 798 | 93 | 73,458 |
| dev, Android | 36,340 | 4,852 | 394 | 1,464 | 1,831 | 550 | 97 | 45,528 |
| test, iPhone | 13,041 | 1,372 | 38 | 646 | 1,365 | 47 | 0 | 16,509 |
| test, Android | 6,245 | 2,310 | 43 | 230 | 765 | 240 | 52 | 9,885 |
| **all** | **112,016** | **16,791** | **544** | **3,994** | **10,158** | **1,635** | **242** | **145,380** |
| users with examples | 59 | 58 | 22 | 23 | 48 | 28 | 5 | |

| SHL | still | walk | run | bike | car | bus | train | subway | all |
|---|---|---|---|---|---|---|---|---|---|
| User 1 | 3,366 | 3,011 | 1,030 | 2,842 | 4,058 | 3,028 | 3,353 | 2,388 | 23,076 |
| User 2 | 197 | 240 | 14 | 136 | 255 | 68 | 133 | 174 | 1,217 |
| User 3 | 290 | 187 | 31 | 63 | 84 | 82 | 227 | 184 | 1,148 |

### Counts per fold (ExtraSensory, official examples)

| part | users (iPhone / Android) | examples (iPhone / Android) | still | walk | run | bike | car | bus | train | all | with location |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev 0 | 9 (5 / 4) | 20,633 / 8,305 | 23,745 | 2,889 | 288 | 342 | 1,431 | 174 | 69 | 28,938 | 86% |
| dev 1 | 10 (6 / 4) | 11,603 / 10,755 | 16,873 | 2,405 | 104 | 892 | 1,807 | 277 | 0 | 22,358 | 75% |
| dev 2 | 10 (6 / 4) | 13,997 / 8,298 | 17,083 | 2,498 | 19 | 600 | 1,791 | 207 | 97 | 22,295 | 89% |
| dev 3 | 10 (6 / 4) | 16,984 / 10,858 | 22,462 | 2,881 | 23 | 768 | 1,405 | 279 | 24 | 27,842 | 92% |
| dev 4 | 8 (5 / 3) | 10,241 / 7,312 | 12,567 | 2,436 | 29 | 516 | 1,594 | 411 | 0 | 17,553 | 65% |
| test | 12 (6 / 6) | 16,509 / 9,885 | 19,286 | 3,682 | 81 | 876 | 2,130 | 287 | 52 | 26,394 | 86% |
| **all** | 59 (34 / 25) | 89,967 / 55,413 | 112,016 | 16,791 | 544 | 3,994 | 10,158 | 1,635 | 242 | 145,380 | 83% |

- Users counts those with official examples. BEF6C611 (Android, dev fold 4) is excluded for inconsistent
  accelerometer processing (`decision_register.md`, X8).
- **Dev fold 4 is the smallest** (17,553 examples, against 22,295–28,938 in the other folds), for two reasons:
  - it contains BEF6C611, whose 1,544 windows are all excluded, so it has 8 users with examples instead of 9;
  - the folds were balanced on each class's share, not on total size. Fold 4 holds only 14% of the dev still
    examples (still is 78% of the dev examples), but the largest share of bus (31%).

  It also has no train and the lowest location coverage (65%).
- Per user: `results/tables/decision_details.csv`, IDs P1 (test) and P2 (dev), breakdown `user`.

### Location coverage (official examples with a fix)

| ExtraSensory | still | walk | run | bike | car | bus | train | all |
|---|---|---|---|---|---|---|---|---|
| iPhone | 92% | 97% | 100% | 99% | 96% | 96% | 100% | 93% |
| Android | 68% | 58% | 68% | 86% | 65% | 59% | 80% | 67% |

| ExtraSensory | iPhone | Android | all |
|---|---|---|---|
| dev | 92% | 68% | 83% |
| test | 100% | 62% | 86% |

| SHL (one fix per minute) | still | walk | run | bike | car | bus | train | subway | all |
|---|---|---|---|---|---|---|---|---|---|
| User 1 | 39% | 86% | 100% | 99% | 83% | 90% | 49% | 31% | 70% |
| Users 2–3 | 46% | 86% | 100% | 90% | 92% | 80% | 38% | 13% | 61% |

Missing location is mainly a platform effect in ExtraSensory, and a still / rail effect in SHL (indoors,
underground).

### Known limitations

- **Thin classes:** ExtraSensory train comes from only 5 users (2 in test, Android only, 52 examples), run from 22
  (4 in test, 81 examples). Per-class test results are reported with user-bootstrap intervals.
- **The test set isn't untouched:** 11 of its 12 users were dev users in experiment 02.
- **GPS is sparser than production and has shortcuts:** ExtraSensory has one fix per minute, and missing speed
  signals "slow" in both datasets.
- **The WAIIS conditions are assumed, not measured** (open questions in the register).
