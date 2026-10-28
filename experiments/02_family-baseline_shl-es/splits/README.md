# Splits of experiment 02 (fixed, approved 2026-10-25, never changed)

Made in `experiments/02_family-baseline_shl-es` (Results, step 1) with `propose_splits.py --seed 0`. Rules are in
`../docs` (Splits). Moved here from `splits/` at the project root on 2026-10-28,
byte for byte (sha256 `af25db55…` and `649c32d5…`, the digests in 02's MLflow dataset inputs).

- `shl_days.csv`: `user, day, partition, fold`.
  - User 1: 76 calendar days in 5 folds (`dev`). A day means every recording of that date (`DDMMYY` and `mDDMMYY`)
    and every position.
  - Users 2 and 3, all 4 positions: `test`, fold −1.
- `es_users.csv`: `uuid, platform, partition, fold`.
  - 48 `dev` users in 5 folds by user.
  - 12 `test` users, fold −1.

**The reserved final test is `splits/reserved_test.json` at the project root** (read it with
`tmd.splits.reserved_test()`). It is the reference from now on. The `test` rows here are the same users and days,
kept so the files' hashes stay as recorded in MLflow.

Read the dev folds with `shl_days()` / `es_users()` in `make_features.py`.
