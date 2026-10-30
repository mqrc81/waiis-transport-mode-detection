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

### Stop-point 2 follow-up: gps-v2 and transfer test (approved 2026-10-30; fixed before running)

Option (b) of stop-point 2, run **once**. **No further feature changes after this run, whatever the checks say.**

**gps-v2** = gps-v1 with these changes only:
1. **Drop** `path_m`, `net_m` and `straightness`: they scale with the number of fixes in the window. That leaves
   13 features.
2. **Repeated positions (fixed rule, decided by a diagnostic before any v2 feature is computed):**
   - The diagnostic runs on ExtraSensory **Android** dev users, all labelled minutes with a pair ≤ 150 s, and counts
     the repeats (distance exactly 0):
     - the share of repeats in still minutes vs in moving-class minutes, and the repeat share per class;
     - for repeats in moving-class minutes, whether they sit at a stop: an adjacent pair (the previous or the next)
       of the same session has a derived speed < 1 m/s.
   - A repeat counts as a **real stop** if it is in a still minute, or in a moving minute at a stop.
   - **If ≥ 50% of Android repeats are real stops:** keep the repeats, and make stop detection uniform on both
     platforms with **one threshold, 1.0 m/s, fixed now**:
     - `stop_frac` = the share of pair speeds < 1.0 m/s;
     - `n_stops` = the number of consecutive pairs going from ≥ 1.0 m/s to < 1.0 m/s.
   - **Otherwise:** treat repeats as missing (no pair speed, no bearing), with v1's stop definitions.
3. Everything else as in gps-v1 (±2 min window, pairs ≤ 150 s, bearings ≥ 20 m, turns between consecutive valid
   bearings).

**Runs:** G and H1 with gps-v2, the GPS model trained with SHL User 1 at lr 0.02 (as G). H1 fuses with A. Check 1
(dataset and platform, within walk and car) on gps-v2.

**Transfer test** (gps-v1 and gps-v2; `transfer.py`):
- **GPS model:** trained on the covered official minutes of the **iPhone dev users + SHL User 1**, scored on the
  **Android dev users**; then the reverse (Android + SHL User 1 → iPhone).
  - LightGBM as G (lr 0.02, balanced class weights).
  - Rounds from early stopping on the training rows of dev fold 0 (ES users of the training platform in fold 0 +
    SHL fold-0 days), fit on the other training rows, then a refit on all training rows.
- **Fusion:** H1 with A's out-of-fold acc probabilities (05's B) on the held-out platform's covered minutes;
  uncovered minutes keep p_acc. HMM forward-backward per outer fold, as in A.
- **Reported, on the held-out platform's official minutes (8-class primary, family secondary):**
  - the transfer H1's gain over A;
  - the normal-fold H1's gain over A on the same platform (v1: H1; v2: H1_v2);
  - the difference between the two;
  - all with paired user-bootstrap CIs on that platform's users.
- One MLflow run per (version, direction).

**Decision (as instructed):**
- If gps-v2 passes the platform check **and** H1_v2 gains ≥ 0.01 8-class over A (with the SHL guard), H1 with
  gps-v2 is the proposed final configuration, under the rule as written.
- Otherwise all numbers are reported and the decision is left open. An override of the rule would be recorded as a
  deviation.
- Row I is not rerun; it keeps its dirty-tag note.

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

### Stop-point 1 (2026-10-30)

**Plan approved with changes** (worked into the Plan): derived speed only; K optional and subject to the missingness
check; the H2 scheme reuses the outer early-stopping fit, with default regularisation; the missingness check within
walk and car for dataset and platform; G with and without SHL; 05's code moves to `src/tmd`.

**Derived-speed diagnostic** (`diag_speed.py`; `logs/diag_speed.log`, `results/diag_speed_{ratio,repeats}.csv`;
dev users, official minutes). Not covered in 04's README.
1. **Derived / reported speed per minute, where both exist** (reported > 0.5 m/s), medians:
   - car: ES Android 0.73, ES iPhone 0.69, SHL 0.95;
   - bus: 0.66 / 0.51 / 0.67;
   - walk: 0.71 / 0.51 / 0.88;
   - train: 0.50 / 0.89 / 0.94.
   - ExtraSensory's reported speed is the minute's **maximum** (`location:max_speed`), SHL's a ~3 s speed at the
     fix, so a ratio below 1 is expected in ExtraSensory.
   - Where both exist, ExtraSensory car derived speed is 8.8–9.0 m/s (SHL 13.7). The much lower overall medians
     (4.9 / 6.9) come mostly from minutes **without** reported speed, which are slower (register G3).
2. **Fix age.** ExtraSensory fixes carry the minute timestamp (`time − ts` = 0 for all 305,747 fixes; the fix time
   is not published, register G5). SHL fixes are 22.5–37.5 s into the minute (median 36.8 s).
   - The derived speed uses the fixes' `time`: minute timestamps in ExtraSensory, fix times in SHL.
   - ExtraSensory dt is therefore uncertain by up to the fix's offset within the minute. That is noise, not a
     systematic bias.
3. **Repeated positions** (distance 0 to the previous fix ≤ 150 s earlier): ES Android still 25%, walk 10%,
   train 13%, car 2%; ES iPhone ≤ 9% (still), ≤ 2% when moving; SHL ≤ 4% (subway).
   - **Not cached fixes:** in ExtraSensory car / bus / train, the pair right after a repeat is slow (median
     0.39 m/s Android, 0.18 iPhone), not a catch-up jump.
- **Conclusion:** no fixable artefact found (no wrong timestamp, no cached-fix pattern). Features kept as planned.
  Low ExtraSensory derived speeds come from (a) max-vs-instant reported speed semantics, (b) slower minutes lacking
  reported speed, and (c) positions that don't update between minutes, mostly on Android.

**Port check** (`port_check.py`, `logs/port_check.log`): `tmd.lgbm` + `tmd.minutes` retrain 05's B (seed 0) and
reproduce 05's B and F_fb out-of-fold probabilities **bit-identically**.

**GPS features:** `gps-v1` has 16 features, not 17 (an addition error in the proposal), and `tests/test_gps_features.py`
covers them. The test found two bugs, fixed before any run:
- the path length included the pair whose start is outside the window;
- turns across a stop were lost.

### Stop-point 2: ablation on development data (2026-10-30)

**Commands** (this folder; full output in `logs/`):

```
uv run python make_gps.py                                   # gps-v1 tables, ~15 s
uv run python run.py --row A                                 # 05's B + fb rescored
uv run python run.py --row A2 --lr 0.02 --vs A
uv run python checks.py coverage
uv run python run.py --row G --lr 0.02 --acc-row A
uv run python run.py --row G_esonly --lr 0.02 --no-shl --acc-row A
uv run python checks.py gps && uv run python checks.py missing
uv run python run.py --row H1 [--name H1_esgps --gps-row G_esonly] --acc-row A --gps-row G --vs A
uv run python run.py --row H2 [--name H2_esgps --gps-row G_esonly] --acc-row A --gps-row G --vs A
uv run python run.py --row I --lr 0.02 --vs A
uv run python tables.py
```

- **Runs:** all seed 0. Clean at `64ced02` (A, A2) and `8e3edd8` (G … H2_esgps, the checks). **Row I is tagged
  dirty**; see "Concurrent edits" below.
- A first, dirty run of A at `4206fc5` and a chain stopped to fix a bug (H2's inner fits took one learning rate for
  both models; fixed in `8e3edd8` before any H2 run) are superseded.
- **Learning rate:** A2 (lr 0.02) is +0.002 [−0.002, +0.006] on 8-class, so **not adopted**. No fold reached the cap
  (118–375 rounds). Its point estimate is higher than A's, so under the Plan the GPS model and I use lr 0.02. The
  fused acc model stays A (lr 0.1).

**Ablation (pooled ExtraSensory dev out-of-fold, seed 0).**
- 8-class F1 over 7 classes (no subway), family F1 over 5 families. G rows are scored on covered minutes only.
- Δ = paired user-bootstrap difference against `vs`. SHL User 1 is a same-person day split.
- Generated by `tables.py`.

| row | vs | ES 8-class F1 [CI] | Δ 8-class [paired CI] | ES family F1 [CI] | Δ family | ES 8-class covered / uncovered | SHL 8-class | Δ SHL 8-class | SHL train / subway | single window p50 / p95 ms | pipeline p50 / p95 ms | acc / GPS model MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | – | 0.552 [0.467–0.638] | – | 0.615 [0.567–0.653] | – | 0.573 / 0.409 | 0.936 | – | 0.94 / 0.87 | 11.0 / 13.8 | 1.11 / 4.04 | 1.3 / – |
| A2 | A | 0.554 [0.469–0.640] | +0.002 [-0.002, +0.006] | 0.618 [0.568–0.656] | +0.003 | 0.574 / 0.422 | 0.937 | +0.001 | 0.94 / 0.88 | 10.9 / 11.6 | 1.19 / 4.09 | 6.8 / – |
| G | – | 0.502 [0.451–0.540] | – | 0.623 [0.570–0.655] | – | 0.502 / – | 0.815 | – | 0.81 / 0.79 | 0.9 / 1.1 | 0.20 / 0.60 | – / 3.82 |
| G_esonly | – | 0.460 [0.420–0.490] | – | 0.590 [0.544–0.614] | – | 0.460 / – | 0.486 | – | 0.13 / 0.00 | 1.0 / 3.6 | 0.19 / 0.58 | – / 1.69 |
| H1 | A | 0.613 [0.531–0.692] | +0.061 [+0.023, +0.088] | 0.681 [0.620–0.726] | +0.065 | 0.641 / 0.407 | 0.952 | +0.016 | 0.96 / 0.92 | 12.5 / 12.8 | 1.34 / 4.77 | 1.3 / 3.82 |
| H1_esgps | A | 0.582 [0.502–0.665] | +0.030 [+0.011, +0.049] | 0.640 [0.593–0.671] | +0.025 | 0.608 / 0.411 | 0.895 | -0.041 | 0.89 / 0.50 | 11.8 / 12.1 | 1.32 / 4.60 | 1.3 / 1.69 |
| H2 | A | 0.617 [0.534–0.699] | +0.065 [+0.019, +0.104] | 0.688 [0.619–0.737] | +0.073 | 0.644 / 0.408 | 0.937 | +0.000 | 0.94 / 0.94 | 11.9 / 12.4 | 1.29 / 4.56 | 1.3 / 3.82 |
| H2_esgps | A | 0.588 [0.510–0.667] | +0.036 [-0.004, +0.065] | 0.659 [0.608–0.705] | +0.044 | 0.611 / 0.408 | 0.931 | -0.006 | 0.93 / 0.89 | 11.8 / 12.0 | 1.29 / 4.50 | 1.3 / 1.69 |
| I | A | 0.584 [0.502–0.668] | +0.032 [+0.010, +0.058] | 0.634 [0.587–0.677] | +0.019 | 0.623 / 0.313 | 0.925 | -0.011 | 0.95 / 0.93 | 12.5 / 14.1 | 1.37 / 4.69 | 6.2 / – |

| row | ES recall bus [CI] | car [CI] | train [CI] | run [CI] | still recall quiet / not quiet (ES) | (SHL) | ES 8-class iPhone / Android | trees (mean rounds) |
|---|---|---|---|---|---|---|---|---|
| A | 0.34 [0.17–0.48] | 0.73 [0.63–0.81] | 0.06 [0.00–0.11] | 0.26 [0.03–0.79] | 0.996 / 0.736 | 0.888 / 0.699 | 0.589 / 0.519 | 45.6 |
| A2 | 0.33 [0.17–0.46] | 0.73 [0.63–0.81] | 0.06 [0.00–0.12] | 0.26 [0.03–0.79] | 0.996 / 0.737 | 0.846 / 0.712 | 0.591 / 0.521 | 240.2 |
| G | 0.34 [0.22–0.55] | 0.65 [0.57–0.71] | 0.38 [0.00–0.71] | 0.08 [0.00–0.50] | 0.984 / 0.869 | 0.603 / 0.632 | 0.545 / 0.443 | 137.0 |
| G_esonly | 0.29 [0.17–0.49] | 0.66 [0.59–0.72] | 0.00 [0.00–0.00] | 0.03 [0.00–0.21] | 0.971 / 0.855 | 0.542 / 0.563 | 0.478 / 0.448 | 68.8 |
| H1 | 0.43 [0.31–0.54] | 0.71 [0.61–0.79] | 0.22 [0.00–0.28] | 0.27 [0.03–0.81] | 0.999 / 0.842 | 0.929 / 0.801 | 0.652 / 0.574 | 45.6 |
| H1_esgps | 0.42 [0.28–0.53] | 0.73 [0.63–0.81] | 0.04 [0.00–0.07] | 0.26 [0.03–0.80] | 0.999 / 0.824 | 0.937 / 0.808 | 0.612 / 0.552 | 45.6 |
| H2 | 0.46 [0.32–0.59] | 0.72 [0.62–0.81] | 0.28 [0.00–0.45] | 0.28 [0.03–0.83] | 0.998 / 0.798 | 0.834 / 0.687 | 0.649 / 0.576 | 45.6 |
| H2_esgps | 0.49 [0.33–0.62] | 0.71 [0.61–0.79] | 0.15 [0.00–0.21] | 0.28 [0.03–0.83] | 0.998 / 0.796 | 0.849 / 0.700 | 0.588 / 0.581 | 45.6 |
| I | 0.42 [0.29–0.53] | 0.73 [0.62–0.81] | 0.09 [0.00–0.19] | 0.27 [0.03–0.81] | 0.995 / 0.793 | 0.807 / 0.613 | 0.615 / 0.546 | 218.2 |

| GPS only (covered minutes) | ES 8-class | ES family | SHL 8-class | acc row A on the same minutes: ES 8-class / family / SHL 8-class |
|---|---|---|---|---|
| G | 0.502 [0.451–0.540] | 0.623 [0.570–0.655] | 0.815 | 0.573 / 0.626 / 0.915 |
| G_esonly | 0.460 [0.420–0.490] | 0.590 [0.544–0.614] | 0.486 | 0.573 / 0.626 / 0.915 |

**Paired comparisons among the fusions** (`logs/compare_fusions_seed0.log`; ES 8-class [paired CI], ES family,
SHL 8-class):
- H2 − H1: +0.004 [−0.019, +0.023], +0.007, −0.015;
- I − H1: −0.029 [−0.045, +0.000], −0.047, −0.027;
- H1 − H1_esgps: +0.031 [−0.004, +0.045], +0.041, +0.057;
- H2 − H2_esgps: +0.029 [+0.002, +0.064], +0.029, +0.006.

**Checks.**

| check 1 (official covered minutes) | dataset AUC: GPS / acc | platform AUC within ES: GPS / acc |
|---|---|---|
| walk | 0.859 / 0.986 | **0.746** / 0.510 |
| car | 0.839 / 0.941 | **0.593** / 0.533 |

- **Check 1 fails on platform:** within walk and car, GPS features identify iPhone vs Android better than the acc
  features do (+0.236 and +0.060, against the +0.02 allowed). Dataset passes.
- Single features (`logs/diag_platform_gps.log`, descriptive): no feature dominates (best |AUC − 0.5| ≤ 0.10).
  For the same class, Android has lower derived speeds and shorter paths:
  - walk: v_p50 0.35 vs 0.50 m/s, path 89 vs 145 m;
  - car: 4.5 vs 7.0 m/s, 1,064 vs 1,807 m.
  - This fits Android positions updating less often (diagnostic 3).
  - `path_m` and `net_m` grow with the number of fixes in the window, so they partly act as a fix count.

| check 2 (row I): missingness pattern | dataset AUC | platform AUC |
|---|---|---|
| walk | **0.660** | **0.698** |
| car | 0.556 | 0.547 |
| all classes (descriptive) | 0.783 | 0.708 |

- **Check 2 fails** (walk > 0.60 for both): I is not adopted, whatever its F1.

**GPS coverage** (check 3, official dev minutes; `results/gps_coverage_dev.csv`): as in the Plan. ES iPhone 0.92
overall (0.91–1.00 per class); ES Android 0.67 (0.58–0.91); SHL User 1 0.74 (0.36 subway – 1.00 run).

**Decisions under the rule.**
- **F1 and SHL parts of the rule:**
  - **H1 passes** (+0.061 [+0.023, +0.088]; SHL +0.016);
  - **H2 passes** against A (+0.065; SHL +0.000), but **not against H1** (+0.004 < 0.01; SHL −0.015);
  - **I fails** (+0.032 against A, −0.029 against H1; SHL −0.011);
  - H1_esgps fails the SHL guard (−0.041).
- **Checks:** check 1 fails for every GPS row (G, H1, H2, I share gps-v1).
- **Strictly applied, no GPS row is adopted, and the final configuration stays A (05).** Open question 1.

**What GPS does (descriptive, H1 vs A):**
- the gain is on covered minutes (8-class 0.573 → 0.641; uncovered 0.409 → 0.407), and on **both platforms**
  (iPhone 0.589 → 0.652, Android 0.519 → 0.574);
- bus recall 0.34 → 0.43; ES train 0.06 → 0.22; not-quiet still 0.736 → 0.842; car 0.73 → 0.71.
- **GPS-only (G):** 0.502 on covered minutes, against 0.573 for A on the same minutes. GPS alone is weaker than acc
  but complementary. Without SHL (G_esonly) it is 0.460, and it never predicts train.

**Concurrent edits (row I).**
- While the chain ran, another Claude session (id 69f9d744…) edited the working tree without committing:
  - progress reporting: `src/tmd/progress.py`, `src/tmd/lgbm.py`, this folder's `run.py` and `pipe.py`;
  - `CLAUDE.md` (two rules), `.gitignore`, `pyproject.toml` / `uv.lock` (tqdm).
- A `git stash` / `stash pop` there replaced `logs/stage_ablation.out` while row I wrote to it, so I's summary line
  is missing from that file. I's metrics are in MLflow and `logs/run_I_seed0.log`.
- **Row I ran the committed code** (`8e3edd8`): its MLflow run started after training, as in the committed
  `run.py`, while the edited version starts it before. It is tagged dirty only because the edits were in the tree
  when it logged.
- The edits add hooks (progress, per-fold metrics) and restructure `run.py`. They don't change the models, the
  fusion or the evaluation.
- That session has since committed them as `b16ef11` (`tmd.progress`, hooks in `tmd.lgbm`, CLAUDE.md, tqdm) and
  `3ba8a44` (this folder's `run.py` / `pipe.py`). Every later run uses that code.

**Proposed final configuration** (depends on open question 1):
- if a GPS row is allowed: **H1** (product fusion; GPS model gps-v1 with SHL, lr 0.02; acc model A), the simplest
  fusion with the gain. H2 adds nothing over it (+0.004) and costs the inner fits.
- otherwise A (05), i.e. no new reference.

**Open questions (stop-point 2).**
1. **Check 1 fails on platform** for every GPS row. Options:
   - (a) apply the rule: no GPS fusion adopted, 06 reports +0.06 as not adoptable;
   - (b) a gps-v2 before deciding: drop `path_m` / `net_m` (they scale with the fix count) and treat repeated
     positions as missing, then rerun G, H1 and the checks. This is a Plan change;
   - (c) judge the platform check differently, e.g. accept it because H1 gains on both platforms (+0.063 iPhone,
     +0.055 Android).

   I'd recommend (b): it tests whether the platform signal comes from recording artefacts, and costs about 15 min.
2. **GPS model with or without SHL:** with SHL is better in both fusions (H1 +0.031, H2 +0.029 on ExtraSensory; SHL
   +0.057 / +0.006). Proposal: with SHL.
3. **Row I** is tagged dirty (see above). Rerun it clean at the current commit (~20 min)? It isn't a candidate
   (check 2 fails), so I'd leave it as is.


### Stop-point 2 follow-up: gps-v2 and transfer test (2026-10-30)

Plan section "Stop-point 2 follow-up", committed before running (`5c9f2f3`). All 7 runs are clean at `e919a3e`.

```
uv run python diag_repeats.py                                   # logs/diag_repeats.log
uv run python make_gps.py --version gps-v2                      # gps-v1 regenerated identical to the cache first
uv run python run.py --row G --name G_v2 --gps gps-v2 --lr 0.02 --acc-row A
uv run python checks.py gps --gps gps-v2
uv run python run.py --row H1 --name H1_v2 --gps gps-v2 --acc-row A --gps-row G_v2 --vs A
uv run python transfer.py --gps gps-v1 && uv run python transfer.py --gps gps-v2   # logs/stage_v2.out
```

**Repeats diagnostic** (ExtraSensory Android dev users, 68,844 labelled minutes with a pair):
- 12,646 repeats (18.4%); 94.2% are in still minutes, 5.8% in moving-class minutes.
- Repeat share per class: still 0.196, walk 0.159, train 0.134, bike 0.037, car 0.030, bus 0.017, run 0.007
  (iPhone: still 0.098, walk 0.030, others ≤ 0.022).
- Of the 739 repeats in moving minutes, 91.3% sit at a stop (an adjacent pair < 1 m/s). For comparison, 60.6% of
  moving minutes without a repeat are next to a pair < 1 m/s.
- **Real-stop share: 99.5% → by the pre-set rule, repeats are kept and stop detection is uniform at 1.0 m/s.**
- gps-v2 = 13 features (`tmd.gps_features`, `tests/test_gps_features.py`). Coverage is unchanged (ES 0.824, SHL
  0.739 of official minutes). gps-v1 tables regenerated with the edited code are identical to the cache.

**Rows** (pooled ExtraSensory dev, seed 0; 8-class over 7 classes, family over 5):

| row | ES 8-class F1 [CI] | Δ 8-class vs A [paired CI] | ES family F1 | Δ family | ES 8-class iPhone / Android | SHL 8-class | Δ SHL | bus / train recall | single window / pipeline p50 ms | GPS model MB |
|---|---|---|---|---|---|---|---|---|---|---|
| G (v1, covered) | 0.502 [0.451–0.540] | – | 0.623 | – | 0.545 / 0.443 | 0.815 | – | 0.34 / 0.38 | 0.9 / 0.20 | 3.82 |
| G_v2 (covered) | 0.512 [0.456–0.554] | – | 0.628 | – | – | 0.815 | – | – | 0.9 / 0.19 | – |
| H1 (v1) | 0.613 [0.531–0.692] | +0.061 [+0.023, +0.088] | 0.681 | +0.065 | 0.652 / 0.574 | 0.952 | +0.016 | 0.43 / 0.22 | 12.5 / 1.34 | 3.82 |
| **H1_v2** | **0.613 [0.533–0.693]** | **+0.061 [+0.024, +0.090]** | 0.682 | +0.067 | 0.655 / 0.571 | 0.952 | +0.016 | 0.43 / 0.22 | 12.5 / 1.35 | 3.80 |

- A on the covered minutes scores 0.573 (8-class).
- H1_v2's latency p95: 13.0 ms single window, 4.8 ms per window in the pipeline.

**Checks** (4 and 1 of the follow-up):

| check 1, within class | dataset AUC: GPS v1 / GPS v2 / acc | platform AUC (ES): GPS v1 / GPS v2 / acc |
|---|---|---|
| walk | 0.859 / 0.858 / 0.986 | **0.746 / 0.730** / 0.510 |
| car | 0.839 / 0.839 / 0.941 | **0.593 / 0.594** / 0.533 |

- The dataset AUCs apply to G and H1 alike (same GPS features). They pass in both versions: the GPS features
  identify the dataset less well than the acc features.
- **gps-v2 still fails the platform check** (walk +0.220, car +0.061 over acc; +0.02 allowed). Dropping the
  distance features and the uniform stop threshold lowered walk only from 0.746 to 0.730.

**Transfer test** (GPS model trained on one platform's dev users + SHL User 1, fused (H1) with A's out-of-fold acc
probabilities; scored on the other platform's dev users; paired user-bootstrap CIs over that platform's users):

| GPS | held-out platform (users) | 8-class: transfer H1 − A | normal-fold H1 − A | transfer − normal | family: transfer − A | normal − A | transfer − normal |
|---|---|---|---|---|---|---|---|
| v1 | Android (19), trained on iPhone (27) + SHL | +0.056 [+0.028, +0.083] | +0.055 [+0.027, +0.081] | +0.002 [−0.003, +0.011] | +0.053 [+0.024, +0.095] | +0.058 [+0.030, +0.100] | −0.006 [−0.017, +0.001] |
| v1 | iPhone (28), trained on Android (17) + SHL | +0.063 [+0.006, +0.105] | +0.063 [+0.009, +0.105] | +0.000 [−0.004, +0.003] | +0.074 [+0.018, +0.100] | +0.073 [+0.018, +0.099] | +0.001 [−0.002, +0.003] |
| v2 | Android (19) | +0.059 [+0.030, +0.089] | +0.052 [+0.025, +0.081] | +0.007 [−0.001, +0.025] | +0.053 [+0.024, +0.096] | +0.059 [+0.031, +0.100] | −0.006 [−0.018, +0.001] |
| v2 | iPhone (28) | +0.064 [+0.005, +0.108] | +0.066 [+0.012, +0.108] | −0.002 [−0.008, +0.003] | +0.077 [+0.018, +0.104] | +0.075 [+0.018, +0.102] | +0.002 [−0.002, +0.004] |

- Early stopping: 111–181 rounds, none at the cap. Users counts are those with official minutes on the held-out
  platform.
- **A GPS model that never saw the held-out platform gives the same gain as the normal folds**, in both directions
  and both versions. Every transfer − normal CI contains 0, and every point difference is ≤ 0.007.
- So the fusion's gain does not depend on learning the platform from the GPS features. The platform signal in GPS
  (check 1) is real (Android positions update less often), but the gain carries across platforms.

**Decision (as instructed):** gps-v2 fails the platform check, so H1_v2 is **not** the proposed final configuration
under the rule as written.
- The rule's other conditions hold (gain +0.061 ≥ 0.01, SHL +0.016), and the transfer test shows the gain
  transfers across platforms.
- **The decision is left open.** If the rule is overridden, it will be recorded here as a deviation.
- Row I is not rerun; its dirty-tag note stands.

## Conclusion

*Not yet.*
