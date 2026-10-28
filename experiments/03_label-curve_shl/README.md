# 03 Label curve on SHL User 1 (LightGBM, with and without gyroscope)

## Question

How many labelled windows per class does LightGBM need before macro-F1 stops improving, and does the gyroscope help?
Labels are expensive and unlabelled data is free; this is the label-curve baseline. Experiment 04 will pre-train
LIMU-BERT on unlabelled windows and fine-tune it with the **same** split, k values, seeds and sampled windows, so
everything 04 needs is saved as files. Model size is not a criterion.

## Plan

*(Brief from Marc, 2026-10-28.)*

**Data.**
- SHL User 1 only: `User1` plus `User1_4phones` (duplicates handled as in 02). Never read SHL Users 2–3 or the
  ExtraSensory test users (`splits/reserved_test.json`).
- Shared code unchanged: 25 Hz resampler, 5 s windows with a 2.5 s stride, features v1, the 8 SHL classes. Reuse 02's
  cached feature tables if they cover these windows.
- Main result: hips phone only. Other User 1 positions are a read-out only.
- Windows mixing two classes: dropped from training, kept in validation and test.

**Split (new, by day).** Whole days into training / validation / test, about 60 / 20 / 20 % of labelled hours. Every
class in validation and test, over several days. Report days and hours per class per partition. Fixed file in
`splits/`, fixed seed. **Test days are used once, for the final numbers only**; no choice (settings, flattening
point, feature subsets) looks at them.

**Sampling k windows per class.**
- k ∈ {1, 2, 5, 10, 50, 200, 1000, all}; seeds 0–4.
- Round-robin over training days (and separate trips within a day), random within. If a class has fewer than k
  windows, use all and record it.
- Nested per seed (the k = 5 set contains the k = 2 set, …). Window IDs per (k, seed) saved to a file for 04.
- "all" = all training-day windows of that class.

**Sensor sets** (paired: identical sampled windows). acc + gyro = all features v1; acc only = the features v1 that
use no gyroscope channel.

**LightGBM.** A small grid (min_data_in_leaf, num_leaves, learning rate) chosen per k on the validation days, early
stopping on the validation days. The validation set is fixed (all validation-day windows) and identical for every k;
report its size. Balanced class weights.

**Outputs.**
1. Plot: macro-F1 (8 classes) vs k (log axis), mean ± std over 5 seeds, one line per sensor set; second panel at
   family level.
2. Flattening point, rule fixed now: the smallest k whose mean **validation** macro-F1 is within 0.01 of k = all.
   Then its test value. Say so if the curve never flattens.
3. Per-class recall and confusion matrix at the flattening point and at k = all (especially car, bus, train, subway).
4. No smoothing for the curves. At the flattening point only: HMM forward filtering and Viterbi (`tmd.hmm`,
   transitions counted from training days).
5. Three short answers: where the curve flattens; whether the gyroscope helps (difference with spread over seeds,
   per class); what that means for how much to label.

Optional read-outs after the main results, reported separately, never used for decisions: (a) User 1's other
positions (hand / torso / bag) on the test days; (b) ExtraSensory dev users with label rules v2.

**Stop-points.**
1. Split proposal, sampling plan, gyro-free feature list, LightGBM grid. Wait for OK.
2. One seed of the full curve, both sensor sets, validation days only, as a sanity check.
3. All seeds, test numbers, plot, answers. Draft the Conclusion.

**Changes approved at stop-point 1 (Marc, 2026-10-28).** Split approved as proposed (seed 0, 44 / 15 / 15 days);
subway's day count is reported next to its numbers. The plan of step 1 below holds, with these changes:
1. **Smoothing is post-processing only**, never part of training.
   - Every run saves per-window class probabilities for validation and test days (with `rec`, `t_start_ms`,
     label), in time order.
   - The headline curve and the flattening point stay unsmoothed (single-window predictions).
   - Two extra read-outs from the saved probabilities:
     a. **One-parameter HMM:** stay probability p, all switches equally likely, uniform initial distribution and
        prior. p is picked once on validation from {0.9, 0.95, 0.98, 0.99, 0.995, 0.999}, pooled over all k (not
        per k). Plot validation macro-F1 vs p. Report Viterbi and forward filtering.
     b. **Counted HMM:** transitions counted from all training-day labels, Viterbi only. Next to the numbers: it
        uses far more than k labels.
   - HMMs run per recording, restarting at gaps. Mixed windows stay in the sequence and in the scoring.
2. **k = all uses 02's LightGBM parameters**, no grid (`r1_cv.py:30`: lr 0.1, 31 leaves, min_child_samples 50,
   feature / bagging fraction 0.8, L2 1). The grid stays for k ≤ 1000.
3. Checks before stop-point 2: (a) acc_vert / acc_hor must use an accelerometer-only gravity estimate, else the
   acc-only set drops them; (b) if feature_fraction is 1 at k = all, the five seeds are identical and that is said
   instead of a spread.
4. **Reduced-validation check:** at the flattening k, draw k windows per class from validation days with the same
   trip sampler; use only those for the grid choice and early stopping; report test macro-F1 as a separate row.
5. Validation and test macro-F1 are reported side by side per k, so the selection bias is visible.
6. For 04: the unlabelled pre-training pool comes from training days only.

## Results

### Step 1: split, sampling plan, feature subset, grid (2026-10-28) — **stop-point, approved 2026-10-28**

Approved with the changes listed at the end of the Plan. The "points to flag" below are answered there (2 → change
4, 4 → change 1b, 5 → change 2 and check 3b, 6 → change 6).

Nothing trained or logged to MLflow yet (one timing fit only, see Grid).

**Commands** (from this folder):

```
uv run python propose_split.py --seed 0    # splits/shl_u1_days.csv, results/split_{hours,days}.csv (~1 min)
```

**Cached features cover the windows.** `<data>/processed/shl_u1_25hz_w5s_s2.5s_feat-v1_all.parquet` (02 step 3,
`make_features.py --all`) holds every User 1 window of every position, including Null-majority windows (y = 0). Hips
with a label: 562,925 windows, the same count as 02's step-2 table, and the hours per class match 02's audit
(57.2 / 51.8 / 17.7 / 48.3 / 68.2 / 51.1 / 56.2 / 40.4 h). It contains User 1 only. The non-`_all` table of 02 is no
longer on disk; filtering `y > 0` gives the same windows. Window IDs (`rec`, `t_start_ms`) are unique. **No new
feature extraction needed.**

#### Proposed split (`splits/shl_u1_days.csv`, seed 0)

Method (`propose_split.py`): `tmd.splits.pick_test_groups` twice on whole days, balanced on hips hours per class.
First 15 test days, then 15 validation days from the remaining 59; each pick is the subset whose per-class hours
are closest to 20 % of the total (200,000 random draws). Constraints: every class on ≥ 3 days (> 3 min each) in the
picked set; at least one `User1_4phones` day in test (for read-out a).

*Tried first:* `balanced_folds` into 5 folds, then 3 / 1 / 1. Shares were uneven (bus 27 % in validation, still 25 %
in test), so I switched to the above.

Hips hours per class (labelled windows × 2.5 s):

| partition | days | still | walk | run | bike | car | bus | train | subway | total |
|---|---|---|---|---|---|---|---|---|---|---|
| training | 44 | 33.7 | 30.0 | 10.4 | 28.6 | 40.9 | 30.4 | 33.5 | 24.8 | 232.4 |
| validation | 15 | 12.0 | 10.7 | 3.8 | 10.3 | 13.9 | 9.8 | 11.1 | 8.1 | 79.8 |
| test | 15 | 11.5 | 11.1 | 3.5 | 9.4 | 13.4 | 10.9 | 11.6 | 7.5 | 78.8 |
| share val / test | | 21 / 20 % | 21 / 21 % | 22 / 20 % | 21 / 20 % | 20 / 20 % | 19 / 21 % | 20 / 21 % | 20 / 19 % | 20 / 20 % |

Days with > 3 min of the class:

| partition | still | walk | run | bike | car | bus | train | subway |
|---|---|---|---|---|---|---|---|---|
| training | 40 | 39 | 22 | 25 | 14 | 23 | 17 | 10 |
| validation | 12 | 14 | 8 | 7 | 6 | 8 | 5 | 3 |
| test | 15 | 12 | 7 | 10 | 7 | 9 | 5 | 3 |

- `User1_4phones` days: `220617` training, `260617` and `270617` test. So read-out (a) has 2 days of hand / torso / bag.
- **74 days, not 76:** `180417` and `200317` (the two without `00inf.txt`) have no labelled hips window, so they are
  in no partition.
- Subway is on only 16 days, so validation and test have 3 subway days each. Its numbers will swing with those days.

**Validation set (fixed for every k):** 114,883 hips windows = **79.8 labelled hours** (114,355 pure). Test: 113,456
windows (78.8 h). Training: 334,586 windows, 332,894 pure.

#### Sampling plan (`splits/samples_k.csv`, written after approval)

- Pool: pure hips windows on training days, per class.
- **Trip** = a maximal run of consecutive pure windows of the same class in one recording (a step of 2.5 s; a gap,
  a mixed window or another class ends it). The `m` and plain recordings of a day are separate trips.
- Per (seed, class), with `rng = default_rng([seed, class_id])`:
  1. shuffle the training days that have the class;
  2. within each day, shuffle its trips, and shuffle the windows within each trip; the day's queue takes one window
     from each trip in turn;
  3. the class order takes one window from each day's queue in turn (day 1, day 2, …, day 1, …).
- The (k, seed) set is the first k of each class's order, so the sets are nested by construction.
- File: one row per (seed, class, rank) for rank < 1000, with `day, rec, t_start_ms`. k = all is every pool window
  (no file needed; 04 gets it from the split and the window ID).
- Pool per class (training days): every class has ≥ 1000 windows, so no class falls short below k = all.

  | | still | walk | run | bike | car | bus | train | subway |
  |---|---|---|---|---|---|---|---|---|
  | windows | 48,191 | 42,825 | 14,908 | 40,957 | 58,660 | 43,686 | 48,144 | 35,523 |
  | days | 42 | 40 | 22 | 25 | 14 | 23 | 17 | 10 |
  | trips | 173 | 199 | 43 | 137 | 86 | 82 | 64 | 71 |

  (Days here count any pure window, so they are higher than the > 3 min counts above.) At k = 10 every class gets 10
  different days; at k = 50 subway gets 5 windows from each of its 10 days.

#### Feature sets

- **acc + gyro:** all 145 features v1 (5 channels × 29).
- **acc only:** 116 features. Kept: `acc_norm__*`, `acc_vert__*`, `acc_hor__*`, `jerk_norm__*` (jerk = |da/dt|, from
  acc). Dropped: the 29 `gyro_norm__*`. Per channel (29): mean, std, min, max, p05, p25, p50, p75, p95, iqr, skew,
  kurt, mad_diff, mcr, acf_peak, acf_lag_s, hjorth_mob, hjorth_comp, bandlog_{0_1, 1_3, 3_6, 6_12.5},
  bandrel_{0_1, 1_3, 3_6, 6_12.5}, dom_freq, spec_centroid, spec_entropy. No other feature mixes channels (gravity is
  estimated from acc only).

#### LightGBM grid

- Grid: `min_data_in_leaf` ∈ {1, 5, 20, 50} × `num_leaves` ∈ {7, 31} × `learning_rate` ∈ {0.05, 0.1} = 16
  settings. Values of `min_data_in_leaf` above k are skipped (k = 1, 2: only 1; k = 5: 1 and 5), since such leaves
  can't isolate a class.
- Fixed: multiclass, balanced class weights on the sampled windows, `feature_fraction` 0.8, **no bagging** (at k = 1
  a bag of 6 of 8 windows would drop whole classes), `lambda_l2` 1, `feature_pre_filter` false, up to 3,000 rounds,
  early stopping after 50 rounds without a gain in validation multi-logloss.
- Choice per (k, sensor set): the setting with the best **mean validation macro-F1 over the 5 seeds**. That setting's
  validation values make the curve and decide the flattening point.
- Family prediction = argmax of the summed class probabilities (as in 02).
- Cost: one fit on all training windows takes 56 s (lr 0.1, 345 rounds) to 123 s (lr 0.05, 840 rounds) on 10 threads
  (timing run, validation only). The whole grid is about 5 h, almost all of it at k = all.
- MLflow: one run per (sensor set, k, seed) with the chosen setting, and the full grid's validation scores as an
  artifact. Datasets: new `tracking.SPLITS` entries for this split's training / validation / test parts; the training
  input is tagged with `samples_k.csv`. Chosen models are saved to `<data>/checkpoints/03_label-curve_shl/`, so the
  test step only loads them.

#### Points to flag (please decide)

1. **Hips only for the main result: I agree.** The other positions have only 3 days (2 in test), too few to train on.
2. **The validation set costs more labels than the training set at small k.** It is 79.8 h; k = 1000 is 8,000
   windows = 5.6 h of training labels. The curve therefore answers "how many training labels, given a large labelled
   validation set". If that matters, a cheap add-on would repeat the flattening k with validation subsampled to the
   same k per class (reported separately). Not in the plan unless you want it.
3. **The validation curve is optimistic**, because the setting and the number of rounds are picked on it. The
   flattening point is still decided on it (as fixed); the test value is unbiased.
4. **HMM transitions from all training-day labels** use far more labels than k. I'll do as briefed and say so next
   to the numbers. (Transitions are cheap to get from trip diaries, so this is a fair read.)
5. **k = all is the same sample for every seed**, so its spread only reflects LightGBM's feature sampling.
6. **For 04 (not done here):** 04 re-cuts the raw windows from the window IDs with the same shared code. Its
   unlabelled pre-training pool should come from training days only (validation and test days are not "free" here).

### Step 2: one seed of the curve, validation days only (2026-10-28) — **stop-point; experiment closed here**

**Single seed (0), validation days only. The test days were never evaluated, so they remain unused.**

**Checks asked at stop-point 1.**
- (a) **acc_vert / acc_hor use an accelerometer-only gravity estimate:** `src/tmd/features.py:40-44`,
  `g = sosfiltfilt(_SOS_GRAV, acc)` on the acc channels; gyro enters only `gyro_norm`. Both stay in the acc-only set
  (116 features).
- (b) **k = all is not seed-identical:** 02's parameters have `feature_fraction` 0.8 and `bagging_fraction` 0.8
  (freq 1), so the five seeds give different models and a real (if small) spread.

**Commands** (from this folder; short output in `logs/*.out`, full log `logs/curve.log`):

```
uv run python make_samples.py                        # splits/samples_k.csv, results/samples_coverage.csv,
                                                     # <data>/processed/shl_u1_hips_25hz_w5s_s2.5s_windows.parquet (~2 s)
uv run python curve.py --seeds 0 --stage sp2         # results/grid/*.csv, results/curve_sp2.csv (~5 min)
```

**Shared code added:** `src/tmd/sampling.py` (trip sampler; reused for the reduced-validation check and by 04).
`src/tmd/tracking.py`: `SPLITS` entries `shl_user1_03_{training,validation,test}` → `splits/shl_u1_days.csv`.

**Samples** (`samples_k.csv`, 40,000 rows = 5 seeds × 8 classes × 1000 ranks). No class is short of k at any
k ≤ 1000. Coverage, seed 0 (`results/samples_coverage.csv`): up to k = 10 every class gets k different days and k
different trips; at k = 50 every class has windows from all its training days (subway: 10 days, 50 trips); at k = 1000
run, car, train and subway use all their trips.

**Window manifest for 04:** `<data>/processed/shl_u1_hips_25hz_w5s_s2.5s_windows.parquet`, every hips window
(884,330, Null included) with `day, rec, t_start_ms, y, pure, partition, trip`.

**MLflow:** 16 runs `sp2_<sensor>_k<k>_seed0` (param `stage = sp2`), exported to `results/runs.csv`. They are
tagged commit `4c77f14`, **dirty**: they ran before the 03 code was committed. Each holds validation metrics, model size and the grid's validation table (`grid_val.json`).
Models: `<data>/checkpoints/03_label-curve_shl/sp2/`; validation probabilities (rec, t_start_ms, day, y, pure,
p_<class>, time order): `<data>/processed/03_label-curve_shl/sp2/*_val_probs.parquet`. Test probabilities are not
computed yet: they come from the saved models once, at stop-point 3.

**Validation macro-F1, seed 0** (114,883 windows, mixed windows included; chosen grid setting per k on this one
seed only):

| k | acc + gyro, 8 classes | acc only, 8 classes | acc + gyro, family | acc only, family | setting acc + gyro (mdl / leaves / lr) | setting acc |
|---|---|---|---|---|---|---|
| 1 | 0.293 | 0.234 | 0.330 | 0.281 | 1 / 7 / 0.05 | 1 / 7 / 0.1 |
| 2 | 0.471 | 0.457 | 0.567 | 0.575 | 1 / 7 / 0.05 | 1 / 7 / 0.1 |
| 5 | 0.528 | 0.548 | 0.698 | 0.708 | 5 / 7 / 0.05 | 5 / 7 / 0.1 |
| 10 | 0.559 | 0.570 | 0.689 | 0.702 | 5 / 7 / 0.1 | 5 / 7 / 0.1 |
| 50 | 0.649 | 0.643 | 0.774 | 0.768 | 1 / 7 / 0.05 | 50 / 31 / 0.1 |
| 200 | 0.698 | 0.690 | 0.809 | 0.802 | 20 / 31 / 0.05 | 5 / 7 / 0.05 |
| 1000 | 0.735 | 0.725 | 0.834 | 0.826 | 1 / 31 / 0.05 | 1 / 31 / 0.05 |
| all | **0.779** | **0.769** | **0.857** | **0.849** | 02 params | 02 params |

k = all is 332,894 training windows (14,908–58,660 per class). Rounds (early stopping): 17–104 for k ≤ 1000, 378
(acc + gyro) and 428 (acc) at k = all. Model size: 58 kB (k = 1, acc) to 12.0 MB (k = all, acc).

**Per-class validation recall, acc + gyro** (acc only in `curve_sp2` MLflow runs; within ± 0.04 of these at
k ≥ 200):

| k | still | walk | run | bike | car | bus | train | subway (3 val days) |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.72 | 0.87 | 0.96 | 0.59 | 0.51 | 0.22 | 0.43 | 0.21 |
| 200 | 0.88 | 0.93 | 0.97 | 0.87 | 0.73 | 0.42 | 0.45 | 0.36 |
| 1000 | 0.88 | 0.94 | 0.97 | 0.89 | 0.78 | 0.51 | 0.47 | 0.46 |
| all | 0.83 | 0.97 | 0.98 | 0.93 | 0.83 | 0.60 | 0.55 | 0.55 |

**Sanity.**
- k = all, acc + gyro, 0.779 on validation days matches 02's out-of-fold hips result (0.776, `r1_cv.py`), with the
  same features and parameters on a different day split.
- The curve rises monotonically from k = 10 on; below that it is noisy (one seed; at k = 5 and 10 acc-only is ahead).
- Still, walk, run and bike are nearly saturated by k = 200. The gain from 1000 to all is in car, bus, train and
  subway. Still recall drops at k = all (0.88 → 0.83), probably because balanced weights over a much larger vehicle
  pool shift still windows to rail.
- Gyroscope: + 0.01 macro-F1 at k ≥ 50 on this seed; the spread over seeds decides at stop-point 3.

#### Deviations (step 2)
- **`min_data_in_bin` = 1 in the grid fits** (`curve.py`, `BASE`). With LightGBM's default 3, 8 training windows
  allow at most 2 bins per feature, so almost no split points. k = all uses 02's parameters unchanged (default 3).
- **`deterministic` = true** and fixed thread counts (2 per grid fit, 10 at k = all), so the final refit equals the
  grid fit.
- Grid fits are not separate MLflow runs. Their validation scores go in `results/grid/` and in each final run as
  `grid_val.json`.
- Inference latency is not measured yet; it is measured at stop-point 3 for the flattening point and k = all.

#### Open question (step 2)
**On seed 0, the curve does not flatten by k = 1000:** 0.735 vs 0.779 at k = all (gap 0.044 > 0.01), and 0.725 vs
0.769 for acc only. If the 5 seeds agree, the fixed rule gives "never flattens before k = all". The k grid jumps from
1000 to 15–59k per class, so the answer would not say where between them it flattens. The reduced-validation check
(change 4) also needs a flattening k below all. Option: add k = 2000, 5000 and 10000 (run has 14,908 training windows,
so all classes have enough). The sampled order is fixed, so extending `samples_k.csv` to rank < 10000 keeps every
existing (k, seed) set unchanged.
**Closed 2026-10-28 (Marc): not pursued.** No k values or seeds are added; 03 ends at this stop-point (see
Conclusion).

## Conclusion

**Draft — not approved**

- Stopped at stop-point 2: the direction changed (accelerometer only, final test = the 12 ExtraSensory test users,
  label curve to be redone on unseen ExtraSensory users; see `DECISIONS.md`, 2026-10-28).
- Seed 0, validation days only: the curve does not flatten by k = 1000 (validation macro-F1 0.735 vs 0.779 at
  k = all, gap 0.044; acc only 0.725 vs 0.769).
- Still, walk, run and bike are near saturation by k = 200; the remaining gain is in car, bus, train and subway.
- The gyroscope adds about + 0.01 macro-F1 at k ≥ 50, on one seed only, so this is not tested across seeds.

## Reusable

- `splits/shl_u1_days.csv`: SHL User 1 day split (44 / 15 / 15 days, about 60 / 20 / 20 % of hips hours per class);
  its test days are unused.
- `splits/samples_k.csv`: nested sampled training windows per (seed 0–4, class), rank < 1000; the (k, seed) set is
  rank < k.
- `src/tmd/sampling.py`: the round-robin sampler (days, then trips, random within a trip) that made it.
