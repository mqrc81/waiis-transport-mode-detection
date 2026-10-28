# Splits of experiment 03 (approved 2026-10-28)

Experiment 04 (LIMU-BERT pre-training + fine-tuning) must use these files unchanged.

- `shl_u1_days.csv`: `user, day, partition, four_phones`. The 74 calendar days of SHL User 1 that have labelled
  hips windows go to `training` (44), `validation` (15) or `test` (15). A day means every recording of that date
  (`DDMMYY` and `mDDMMYY`). Days `180417` and `200317` have no labelled hips window and are in no partition.
  Made with `propose_split.py --seed 0` (method in its docstring).
- `samples_k.csv`: `seed, class_id, class, rank, day, rec, t_start_ms`. Made with `make_samples.py` (sampler:
  `tmd.sampling`). Per seed 0–4 and class, the first 1000 pure training-day hips windows in the round-robin order
  (days, then trips). **The (k, seed) set is `rank < k`**, so sets are nested. k = all = every pure training-day hips
  window of the class (not in the file). No class has fewer than 1000 windows.
- For 04: the unlabelled pre-training pool comes from training days only.

**Window ID** = (`rec`, `position` = `Hips`, `t_start_ms`): the SHL timestamp [ms] of the window's first sample on
the 25 Hz grid made by `tmd.resample.decimate_100_to_25` per NaN-free stretch, with starts from `tmd.windows.starts`
(5 s, stride 2.5 s), exactly as in `experiments/02_family-baseline_shl-es/make_features.py` (`shl_one`). Recordings:
`User1` without the 3 days that are also in `User1_4phones`, plus `User1_4phones` (02 audit, duplicates).
