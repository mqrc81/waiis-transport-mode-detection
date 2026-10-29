# 04 Frozen benchmark on ExtraSensory (+ SHL check sets)

## Question

Build one frozen, versioned dataset, **`waiis_bench_v1`**, that every model reads (LightGBM, InceptionTime + GPS,
LIMU-BERT later), so that models are compared on exactly the same examples. The examples should look as much as
possible like what the WAIIS app sends, as measured in production:
- accelerometer only, ~25 Hz, recorded in fragments;
- GPS: a fix every ~10 s while moving, ~20 s when walking, minutes apart when still; ~23% of fixes have no speed;
  median accuracy ~8 m.

*(Brief from Marc, 2026-10-28. This is a data build, not a model experiment: it has no MLflow runs.)*

## Plan

1. Inventory of the raw data (`inventory.py`). **Stop-point.**
2. Build the benchmark with one command (`build.py`; shared readers in `src/tmd`), verify it, and document every
   decision (`decision_register.md`). **Stop-point.**
3. Re-pick the ExtraSensory test users on the benchmark's counts (method in `splits/README.md`). **Stop-point: the
   supervisor's OK.**
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

**`waiis_bench_v1`** is the frozen benchmark that every transport-mode model in this project reads (`tmd.bench`).
- **Build:** commit `e51ea79`, `<data>/processed/waiis_bench_v1/`, 23 files, 4.8 GB.
- **Verified:** a full rebuild from that commit in a clean worktree gave the same sha256 for every data file
  (2026-10-29). It is the only benchmark on disk.
- **ExtraSensory:** 145,380 official examples (59 users, accelerometer only, 15 s windows at 25 Hz, labels v2),
  split into 5 dev folds (118,986) and 12 reserved test users (26,394).
- **SHL hips:** User 1 is an optional training source (23,076 examples, including all metro); Users 2–3 are the
  rail check (2,365).
- **GPS:** per minute for ExtraSensory; two cadences for SHL.

**Known limitations.**
- **Thin classes:** ExtraSensory has train from only 5 users (2 in test, Android only, 52 examples) and run from 22
  (4 in test, 81 examples). Per-class test results need user-bootstrap intervals.
- **The test set isn't untouched:** 11 of its 12 users were dev users in experiment 02.
- **GPS is sparser than production and has shortcuts:** ExtraSensory has one fix per minute. Missing location
  depends on platform (and in SHL on rail class), and missing speed signals "slow" in both datasets.
- **The WAIIS conditions the benchmark imitates are assumed, not measured:** recording schedule, units, cadence
  trigger and per-mode missing rates.

The full list, with numbers and risks, is in [`decision_register.md`](decision_register.md).
