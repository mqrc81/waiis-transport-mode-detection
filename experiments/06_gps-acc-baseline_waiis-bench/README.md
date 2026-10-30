# 06 GPS + accelerometer baseline on waiis_bench_v1 (LightGBM fusion + HMM)

## Question

How much do GPS features improve the 05 baseline (B + HMM forward-backward) on `waiis_bench_v1`, and which fusion
works best? The result becomes the new reference.

## Plan

*Approved 2026-10-30 with changes (stop-point 1); the changes are worked into this Plan.*

### Scope (as 05 unless stated)

- **Same as 05:**
  - training data: ExtraSensory dev + SHL User 1 at weight 1;
  - official examples, usable minutes, paired folds;
  - HMM forward-backward, with transitions from all labelled training minutes;
  - balanced class weights, and the paired user bootstrap with redraws (`tmd.evaluate`).
- **Primary score:** 8-class macro-F1 on pooled ExtraSensory dev out-of-fold, over 7 classes (no subway).
  Secondary: family macro-F1.
  - The 05 reference: 8-class 0.551, family 0.615 (seed 0: 0.552 / 0.615).
- **Test:** scored once, at stop-point 3, to confirm, never to choose.
- **GPS:** `gps_per_minute` only, so SHL and ExtraSensory have the same cadence (one fix per minute). No GPS extras
  (`gps_waiis_cadence`, `gps_1hz`, `gps_unblanked`).
- **No OSM way types.** `waiis_bench_v1` has no map or way-type column, so **step J is not run**.

### GPS fields per dataset and platform

| | position (lat / lon) | reported `speed` | speed derived from fixes (`speed_prev_mps`) |
|---|---|---|---|
| ExtraSensory iPhone | the minute's absolute location; 91–100% of official minutes per class | phone's `max_speed` over the minute; −1 when missing (32% of still minutes have it, 87% of car) | distance / time to the previous minute's fix (dt median 60 s) |
| ExtraSensory Android | same; 59–100% per class | same, but rarely present (5% of still, 63% of car) | same |
| SHL User 1 (hips) | best-accuracy fix within ±7.5 s of the minute centre; 31–100% per class | **derived**: local speed over ~3 s from the 1 Hz positions, then blanked with p = 0.95 · exp(−v / 1.954 m/s) | same as ExtraSensory |

**The two speeds don't mean the same thing in the two datasets** (dev official minutes, medians in m/s):

| | ES Android car | ES iPhone car | SHL car | ES Android train | ES iPhone train | SHL train |
|---|---|---|---|---|---|---|
| reported speed | 13.5 | 14.7 | 15.2 | 11.9 | 22.5 | 17.1 |
| derived speed | 4.9 | 6.9 | 11.9 | 0.5 | 17.8 | 13.0 |

- **ExtraSensory's derived speed is low.** Positions sometimes don't update between minutes (`no_updates`).
  - Identical consecutive positions: 25% of Android still, 13% of Android train, ≤ 2% of moving classes.
  - That doesn't explain the car gap; the rest is unexplained (position timing within the minute isn't published,
    register G5).
- **Reported speed is a minute's maximum in ExtraSensory** but a ~3 s instantaneous speed in SHL.
  - Its missingness depends on platform and class in both datasets (register G3, H5).
- **Derived speed is the only quantity computed identically in both datasets.**

**Approved:**
- The GPS feature set (`gps-v1`) uses **positions and derived speeds only**.
- Reported speed is an **optional** step **K**. If it runs, it must pass the same checks **and** row I's
  missingness check (within class, dataset and platform, ≤ 0.60): Android reports speed in 5% of still minutes
  against 63% of car.
- A derived-speed diagnostic ran before any training (Results, stop-point 1). If it had found a fixable artefact,
  the features would have been fixed; it found none.

### GPS coverage of official dev examples (descriptive check)

A minute is **GPS-covered** when at least one derived speed exists in its window (below): two consecutive fixes of
the same session ≤ 150 s apart, ending in minutes t − 2 … t + 2.

| covered | still | walk | run | bike | car | bus | train | subway |
|---|---|---|---|---|---|---|---|---|
| ExtraSensory iPhone | 0.91 | 0.96 | 1.00 | 0.99 | 0.96 | 0.96 | 1.00 | – |
| ExtraSensory Android | 0.67 | 0.58 | 0.69 | 0.85 | 0.73 | 0.59 | 0.91 | – |
| SHL User 1 | 0.43 | 0.89 | 1.00 | 0.99 | 0.85 | 0.94 | 0.57 | 0.36 |

- Coverage depends on platform (ExtraSensory) and on class, most of all in SHL: still, train and subway are often
  indoors or underground.
- The coverage table is reported per class, dataset and platform in Results. The test split's table comes at
  stop-point 3.

### GPS features (`gps-v1`, 16 features per minute)

**Inputs.**
- The minute's own fix and those of minutes t − 2 … t + 2 of the same session, labelled or not. This is a fixed
  ±2 min window, batch prediction as in 05.
- **Pair speeds:** v_i = `dist_prev_m` / `dt_prev_s` for each fix whose previous fix is ≤ 150 s earlier, in the
  window.
- **Displacements** between consecutive fixes, and their bearings when the displacement is ≥ 20 m (shorter ones are
  noise).
- Missing values are NaN. There is no fix count, no interval, no accuracy and no missingness column.

**Features.**

| group | features |
|---|---|
| speed | p10, p50, p90, max, mean of the pair speeds; v at the minute (pair ending at t) and the next (ending at t + 1) |
| acceleration | mean and max of \|Δv\| / Δt between consecutive pair speeds |
| stops | stop fraction (share of pair speeds < 0.5 m/s); number of stops (moving ≥ 2 m/s → stopped < 0.5 m/s) |
| heading change | mean and max absolute turn between consecutive displacement bearings [deg] |
| distance | path length (sum of displacements), net displacement (first to last fix), straightness (net / path) |

That is 7 + 2 + 2 + 2 + 3 = **16 features** (the stop-point-1 proposal said 17: an addition error). **K** would add the reported speed: p50 and max over the window, and
its value at the minute (3 more).

Code: `src/tmd/gps_features.py` (versioned `gps-v1`), cached per source in
`<data>/processed/06_gps-acc-baseline_waiis-bench/`.

### Shared code moved from 05

06 reuses 05's training, HMM and evaluation pieces. Per `CLAUDE.md` they move to `src/tmd`:
- `tmd.lgbm`: 05's LightGBM settings, balanced class weights, paired-fold training with inner early stopping;
- `tmd.minutes`: labelled minutes, HMM parameters, smoothing per session;
- `tmd.acc_features` is already shared.

05's own code stays unchanged; its README and results refer to it.

**Port check (before any 06 row):** the ported code retrains 05's B (seed 0) and must reproduce its out-of-fold
probabilities bit-identically (`oof/B_seed0.npz` of 05).

### LightGBM

- 05's settings (learning rate 0.1, 31 leaves, min_child_samples 50, feature and bagging fraction 0.8, L2 1).
  Balanced class weights for every model.
- Trees from early stopping on the inner validation fold (k + 1) mod 5, as in 05: patience 30 at lr 0.1.
- Seed 0 for the ablation, seeds 0–2 for the final configuration.
- **A′:**
  - learning rate 0.02 (÷ 5), patience 150 (× 5), at most 10,000 rounds;
  - Results confirm that no fold reaches the cap; if one does, the cap is raised and A′ rerun.
- **The GPS model and every later acc model use the learning rate of whichever of A and A′ has the higher pooled
  8-class F1** (the point estimate, even if A′ isn't adopted under the rule).

### Ablation

One MLflow run per row, seed 0. Each step builds on the best configuration so far.

**Keep rule (from the brief):** a step is adopted only if it
1. improves pooled ExtraSensory dev **8-class** macro-F1 by ≥ 0.01 (point estimate),
2. doesn't lower SHL User 1 8-class macro-F1 by more than 0.01, and
3. passes its checks.

Every row reports the paired user-bootstrap CI of the 8-class difference against the best row so far, and family
F1.

**The rows:**

- **A.** 05's final configuration (B + HMM forward-backward, seed 0), **rescored, no retraining**, from 05's saved
  out-of-fold probabilities, with 06's reporting (8-class primary, covered / uncovered split).
- **A′.** A with learning rate 0.02 (see LightGBM). Same features and HMM.
- **G.** GPS-only LightGBM on GPS-covered minutes (training and scoring), 16 features, then HMM forward-backward.
  - Descriptive, not a candidate.
  - Scored on covered minutes only, with the acc row's 8-class F1 on the same minutes alongside.
  - Run twice: **with and without SHL User 1 rows** in training (G, G_esonly), both scored on ExtraSensory dev.
  - H1 and H2 run with both GPS models, since they only need the GPS model's probabilities. **Which GPS model the
    fusion uses is chosen at stop-point 2.** I trains one model on all training rows, as in scope.
- **H1. Late fusion, product.** p ∝ p_acc · p_gps on covered minutes, renormalised. Both models are trained with
  balanced weights, so there is no prior correction. Uncovered minutes keep p_acc. Then HMM forward-backward.
- **H2. Late fusion, stacking.** A multinomial logistic regression on [log p_acc, log p_gps] (16 inputs, clipped at
  1e-6), fit on covered training minutes. The inner scheme is below.
- **I. Early fusion.** One LightGBM on the 580 acc features + 16 GPS features (NaN where missing), then HMM.
- ~~J. Best fusion + OSM way types~~: not run; the benchmark has no OSM data.
- **K (proposed, open question 1).** Best fusion + reported speed (3 features), with its own checks.

**H2 inner scheme (open question 2).** For outer fold k, with training folds T = {0..4} \ {k}:
1. For each j in T, train the acc and GPS models on T \ {j}, with the **number of rounds of the outer fold k's
   early stopping** (from training data only, no nested early stopping), and predict fold j.
   - For j = (k + 1) mod 5 this is **the outer early-stopping fit itself** (trained on T \ {j}, which predicts
     fold j), so it is reused. That leaves 3 new fits per model and outer fold.
   - Every training minute gets an inner out-of-fold probability.
2. Fit the logistic regression on the covered **official** training minutes:
   - inputs: the inner out-of-fold log-probabilities;
   - balanced class weights, so its outputs stay likelihood-like for the HMM;
   - **scikit-learn's default regularisation** (L2, C = 1.0), not tuned; lbfgs, at most 1,000 iterations.
3. Apply it to fold k's covered minutes, using the probabilities of the outer models (trained on all of T). Uncovered
   minutes keep p_acc. Then HMM forward-backward.
- The scored fold k never enters steps 1–2.
- The ExtraSensory parts of the inner folds are split by user, SHL by day, as everywhere.

**HMM** in every row: 05's setup (forward-backward, 1-min step, restart after 10 min or a new session, transitions
and initial distribution from all labelled training minutes per outer fold), over all usable minutes.

### Checks

**1. Dataset and platform from GPS features alone, within class (rows G, H1, H2, I, K).**
- A small fixed LightGBM (05's check model: 100 rounds, 15 leaves) predicts, out of fold:
  - the **dataset** (SHL vs ExtraSensory), on official covered minutes of walk and of car separately, with the
    paired folds;
  - the **platform** (iPhone vs Android), on ExtraSensory official covered minutes of walk and of car, with the
    user folds.
- The same AUC is computed from the 580 acc features on the same minutes.
- **Passes if each GPS AUC ≤ the acc AUC + 0.02** (4 comparisons).
- The rows share one GPS feature set, so the check is computed once for gps-v1 (and once for K) and reported with
  each row.

**2. Missingness pattern (row I).**
- AUC from the GPS-missingness pattern alone: one bit per GPS feature (NaN or not), for dataset and for platform.
- Computed **within class (walk and car), for dataset and for platform**, plus overall as a descriptive figure.
- **Passes if all four are ≤ 0.60.** Also applies to K.
- I is reported even if it fails, but not adopted then.
- Expect it to fail on platform: Android covers 58–73% of walk and car minutes, iPhone 96%.

**3. Descriptive:** GPS coverage per class, dataset and platform (table above; the test split's at stop-point 3).

### Reported for every row

- Pooled ExtraSensory dev 8-class and family macro-F1 with CIs and class lists.
  - The 8-class F1 is also split into **covered and uncovered** minutes.
  - The paired 8-class difference CI against the best row.
- Recall with user-bootstrap CIs for **bus, car, train and run**. Still recall by `quiet`.
- SHL User 1 8-class and family F1, and **train / subway recall** (same-person day split).
- **Inference time, end to end**: acc features + GPS features + both models + fusion (+ HMM in the pipeline
  number).
  - Single window: one minute's 15 s window + its ±2 min GPS fixes.
  - Per window over whole sessions: 200 fold-0 ExtraSensory sessions.
  - Single thread, after warm-up, median and p95.
- **Size of both models** (and the logistic regression for H2).

### Final configuration and test

- **Stop-point 2:** ablation results and the proposed final configuration. Wait for OK.
- **After OK:** the final configuration with seeds 1–2 on dev.
- **Stop-point 3:** the test protocol is written into this Plan and committed before scoring, as in 05. Then the
  12 test users and the SHL Users 2–3 rail check once, 3 seeds, and motorbike descriptive.

### MLflow

- Experiment `06_gps-acc-baseline_waiis-bench`.
- Inputs `extrasensory_dev_v2`, `shl_user1_dev` (training, `oof_eval=True`, window 15 s, stride 60 s); label rules
  `bench-v1`; features e.g. `acc-v1_whole_sub5s+gps-v1_w2min`.
- Params also include the fusion, the learning rate and the GPS feature set.

### Estimated run time

| step | estimate |
|---|---|
| port check (05 B retrained) | 5 min |
| GPS features, all sources | < 2 min |
| A (rescore) | 3 min |
| A′ (lr 0.02, ~5× rounds) | 25–30 min |
| G | 3 min |
| H1 (no training beyond G) | 3 min |
| H2 (inner fits: 20 acc + 20 GPS) | 15 min at lr 0.1; 60–75 min at lr 0.02 |
| I | 5 min at lr 0.1; 25 min at lr 0.02 |
| K | as the best fusion |
| checks and latency | 15 min |
| **ablation total** | **~1.5–3 h** |
| final seeds 1–2 | 0.5–2 h, depending on the fusion and learning rate |
| test | 15–30 min |

### Stop-points

1. ~~Plan.~~ Approved 2026-10-30 with changes.
2. Ablation results on development data and the proposed final configuration. Wait for OK before touching the test
   users.
3. Test protocol written in the Plan and committed. Then the final configuration (3 seeds) scored once on the 12
   test users and once on the SHL Users 2–3 rail check, plus motorbike (descriptive). Then Results, a draft
   Conclusion, and an update of STATUS.md.


## Results

*Nothing run yet.*

## Conclusion

*Not yet.*
