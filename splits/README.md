# Splits shared by all experiments

- `reserved_test_v2.json`: **the reserved final-test set, and the reference for it** (approved by the supervisor on
  2026-10-28; experiment 04, step 3). No experiment trains, tunes, selects or pre-trains on it, self-supervised
  pre-training included.
  - SHL: Users 2 and 3, all days and all 4 positions (6 `user, day` rows). Unchanged from the first set.
  - ExtraSensory: 12 users (`uuid, platform`), re-picked in experiment 04 by
    `experiments/04_frozen-benchmark_es-shl/propose_test_v2.py`. Their dev folds:
    `experiments/04_frozen-benchmark_es-shl/splits/es_users_v2.csv`.
  - **11 of the 12 ExtraSensory users were dev users in experiment 02** (training and model choices there);
    098A72A5 is the only one carried over.
  - Read it with `tmd.splits.reserved_test()`.
- `reserved_test.json`: the first set (approved 2026-10-25 in experiment 02, step 1), **superseded**. Kept unchanged
  because 02's MLflow inputs record its hash; read it with `tmd.splits.reserved_test(1)`.

An experiment's own folds (dev splits) live in its folder, e.g. `experiments/04_frozen-benchmark_es-shl/splits/`.
