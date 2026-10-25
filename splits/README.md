# Splits (fixed, approved 2026-10-25, never changed)

Made in `experiments/02_family-baseline_shl-es` (Results, step 1) with `propose_splits.py --seed 0`. Rules are in
`experiments/docs/brief_block1_experiments.md` (Splits).

- `shl_days.csv`: `user, day, partition, fold`.
  - User 1: 76 calendar days in 5 folds (`dev`). A day means every recording of that date (`DDMMYY` and `mDDMMYY`)
    and every position.
  - Users 2 and 3, all 4 positions: `test`, fold −1, used in experiment 06 only.
- `es_users.csv`: `uuid, platform, partition, fold`.
  - 48 `dev` users in 5 folds by user.
  - 12 `test` users, fold −1, used in experiment 06 only. They are not used before 06 in any stage, self-supervised
    pre-training included.

Read them with `tmd.splits.shl_days()` / `tmd.splits.es_users()`.
