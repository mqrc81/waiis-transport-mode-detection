# Splits shared by all experiments

## Current splits (used by `waiis_bench_v1`)

| split | file | `tracking.SPLITS` name |
|---|---|---|
| reserved test users: 12 ExtraSensory users, SHL Users 2–3 | `splits/reserved_test_v2.json` (here) | `extrasensory_test_v2`, `shl_user23_test_v2` |
| ExtraSensory dev folds: 5 folds by user | `experiments/04_frozen-benchmark_es-shl/splits/es_users_v2.csv` | `extrasensory_dev_v2` |
| SHL User 1 dev folds: 5 folds of whole days | `experiments/04_frozen-benchmark_es-shl/splits/shl_days.csv` | `shl_user1_dev` |

- `shl_days.csv` is a byte-identical copy of 02's file (sha256 `af25db55`); 02 keeps its own.
- Its `test` rows (Users 2–3) are the same as the SHL part of `reserved_test_v2.json`.
- Every other split file and name is legacy, kept only for the records of experiments 02 and 03
  (`tmd.tracking.LEGACY`, refused for new runs).

## Reserved test set

- `reserved_test_v2.json`: **the reserved final-test set, and the reference for it** (approved by the supervisor on
  2026-10-28; experiment 04). No experiment trains, tunes, selects or pre-trains on it, self-supervised
  pre-training included.
  - SHL: Users 2 and 3, all days and all 4 positions (6 `user, day` rows). Unchanged from the first set.
  - ExtraSensory: 12 users (`uuid, platform`), re-picked in experiment 04 (see below). The dev folds of the other
    users: `experiments/04_frozen-benchmark_es-shl/splits/es_users_v2.csv`.
  - **11 of the 12 ExtraSensory users were dev users in experiment 02** (training and model choices there);
    098A72A5 is the only one carried over.
  - Read it with `tmd.splits.reserved_test()`.
- The first set (approved 2026-10-25, **superseded**) is in `experiments/02_family-baseline_shl-es/splits/reserved_test.json`.
  It is unchanged, so its hash still matches 02's records; read it with `tmd.splits.reserved_test(1)`.

## How the test users were picked

- **Eligible users:** all 60 ExtraSensory users except those with more than 10% implausible accelerometer windows.
  At the time a window was implausible when its median |acc| was outside 7–13 m/s². That excluded only BEF6C611
  (79%).
- **Constraints:**
  - platform: 6–7 iPhone users, and the iPhone share of test examples within ±3 points of the share over all
    eligible users;
  - location coverage of test examples within ±3 points of the overall figure;
  - minimum users with ≥ 5 examples of a class: bus 5, bike 4, run 3, train 1–2 (at most 2), still / walk / car 5.
- **Selection rule:**
  - among random 12-user subsets that meet every constraint (3,000,000 draws; 9,972 passed), the one whose
    per-class share of test examples is closest to 12/59 (sum of squared deviations) wins;
  - the dev folds are then dealt from the remaining users: 5 folds by user, round-robin within platform,
    `tmd.splits.balanced_folds`.
- **Seed:** 0, for both the subset draws and the folds.
- **Exact reproduction:** run `propose_test_v2.py` as of commit `1d03c9d` (deleted after `4b09ec3`) with
  `--seed 0 --draws 3000000`, on the development build made at commit `b1e2a9c`. Its manifest is
  `experiments/04_frozen-benchmark_es-shl/results/manifests/es_bench_v2.json`.

An experiment's own folds (dev splits) live in its folder, e.g. `experiments/04_frozen-benchmark_es-shl/splits/`.
