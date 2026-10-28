# Splits shared by all experiments

- `reserved_test.json`: **the reserved final-test set, and the reference for it from now on** (approved 2026-10-25 in
  experiment 02, step 1). No experiment trains, tunes, selects or pre-trains on it, self-supervised pre-training
  included.
  - SHL: Users 2 and 3, all days and all 4 positions (6 `user, day` rows).
  - ExtraSensory: 12 users (`uuid, platform`).
  - Read it with `tmd.splits.reserved_test()`.

An experiment's own folds (dev splits) live in its folder, e.g. `experiments/02_family-baseline_shl-es/splits/`.
