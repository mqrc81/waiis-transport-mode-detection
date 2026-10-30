# 05 Accelerometer-only baseline on waiis_bench_v1 (LightGBM + HMM)

## Question

How well does an accelerometer-only LightGBM + HMM baseline perform on `waiis_bench_v1`, and which pre-processing
steps improve it? The result is the reference every later model (InceptionTime + GPS, LIMU-BERT) has to beat.

## Plan

*Approved 2026-10-29 with changes at stop-point 1, and again at stop-point 2; the changes are worked into this
Plan.*

### Scope and data

- **Accelerometer only, no GPS.** GPS fusion is a later experiment.
- **Official examples only** (`tmd.bench.examples`): one 15 s window (375 × 3, 25 Hz) per labelled minute. No full
  snippets, no SHL stream, no other SHL positions.
- **Classes:** train on the 8 classes.
  - Primary score: family macro-F1 (still / foot / bike / road / rail), family probability = sum of class
    probabilities (`tmd.evaluate.family_proba`).
  - Secondary score: 8-class macro-F1.
- **Training data:** ExtraSensory dev users (118,986 examples, 47 users) + SHL User 1 (23,076 examples, the only
  metro and most train).
  - SHL User 1 **always stays**; there is no drop rule.
  - Step A runs with and without it, as a diagnostic.
  - If SHL lowers pooled ExtraSensory dev family macro-F1 at A (point estimate), SHL rows are down-weighted. A is rerun
    with an SHL row weight of 0.5 and 0.25, seed 0, and the best on pooled ExtraSensory dev family macro-F1 is kept.
  - Class weights are computed **after** the SHL weighting (on weighted counts), so subway and train keep their full
    class weight.
- **Development scores (all runs):**
  - main: pooled out-of-fold predictions on the 5 ExtraSensory dev folds (`es_users_v2.csv`), with user-bootstrap
    95% CIs; per-fold scores only as spread;
  - secondary: SHL User 1, pooled over its 5 day folds (`shl_days.csv`). This is a **same-person day split**, so it
    is optimistic, and there is no CI. Every row reports SHL rail recall (train, subway), marked as such.
- **Paired folds** (as in 02): outer fold k trains on ExtraSensory folds ≠ k and SHL days of folds ≠ k, and scores
  ExtraSensory fold k and SHL fold k. 5 models per row.

### Macro-F1 and its class list

- Macro-F1 averages over the classes **present in the evaluation set's true labels**. A class that is predicted but
  absent from the true labels is not averaged; its predictions still count as errors of the true class.
  - This is what `tmd.evaluate.scores` has done since its first commit (`1f52628`), so earlier numbers (02, 03) are
    not affected.
- Every macro-F1 is reported with its class list:

| evaluation set | 8-class macro-F1 over | family macro-F1 over |
|---|---|---|
| ExtraSensory dev, pooled | still, walk, run, bike, car, bus, train (7; no subway) | 5 families (rail = train) |
| ExtraSensory dev folds 1 and 4 (spread only) | 6 (no subway, no train) | 4 (no rail) |
| SHL User 1, Users 2–3 | 8 | 5 |
| ExtraSensory test, overall and Android | 7 (no subway) | 5 |
| ExtraSensory test, iPhone | 6 (no subway, no train: test train is Android only) | 4 (no rail) |

### Usable minutes (context and HMM)

- Context (D) and the HMM (F) use **every usable minute of the session**, labelled or not.
- Scores are still computed on official examples only.
- **usable = `has_acc` and not `acc_implausible` and not `user_excluded`.** None of these depends on the label:
  - `has_acc`: a snippet of ≥ 15 s with a clean stretch;
  - `acc_implausible`: the window's median |acc| is outside 7–20 m/s² (register U3);
  - `user_excluded`: whole users (BEF6C611, register X8).
- Label-dependent flags are **not** used to choose minutes: `label_implausible`, `capped`, the drop reasons, SHL's
  mixed minutes.
- No labels of non-official minutes are used as inputs. Their labels are used only for HMM transitions (see F).
- Counts:
  - ExtraSensory dev: 291,155 usable minutes (118,986 official);
  - SHL User 1: 36,806 usable minutes (23,076 official).
- Neighbours are matched with ±30 s tolerance (D). 92% of official ExtraSensory minutes have a usable minute 1 min
  before them, and 93% 1 min after, in the same session; at ±3 min it is 91%. SHL User 1 has 100%.
  - The stop-point-1 figure of 65% counted exact 60 s matches only; ExtraSensory timestamps aren't on the minute
    grid.

### LightGBM

- 02's settings: multiclass (8), learning rate 0.1, 31 leaves, min_child_samples 50, feature and bagging fraction
  0.8 (bagging_freq 1), L2 1, 8 threads.
- **Balanced class weights.** w_c = W / (8 · W_c), with W_c the (SHL-weighted) training weight of class c. Every
  present class gets the same total weight (02's `class_weights`).
- **Number of trees, from an inner validation split by user.** In outer fold k:
  - inner validation = ExtraSensory users of fold (k + 1) mod 5 + SHL days of fold (k + 1) mod 5. SHL User 1 is one
    person, so its part is split by day;
  - fit on the other 3 training folds, with early stopping on multi-logloss (balanced weights, patience 30, at most
    1,000 rounds);
  - refit on all 4 training folds with that number of rounds. Fold k is never seen.
- **Seeds per row:** every ablation row uses seed 0 (D′'s gap patterns are drawn with seed 0). The final
  configuration runs seeds 0, 1 and 2; seed 0 is the ablation run itself, and D′'s patterns are re-drawn per seed.
- Known risk from 02: early stopping stopped ExtraSensory-only training at 7–31 rounds. If it happens again in A's
  ExtraSensory-only run, it is reported; the protocol doesn't change.

### Ablation rule (fixed before running)

Each step builds on the best configuration so far. One MLflow run per row; every row is reported.

**Adopt a step only if:**
1. **F1:** the paired user-bootstrap 95% CI of the difference in pooled ExtraSensory dev family macro-F1
   (step − best so far) excludes 0, i.e. its lower bound is > 0.
   - Same 47 dev users, 1,000 resamples with seed 0; both rows are scored on the same resampled users.
   - `tmd.evaluate.bootstrap_users_diff`.
   - Alternatively, the step is adopted if it is cheaper at equal F1: the CI contains 0 and it has a lower
     single-window inference time.
2. **SHL guard:** SHL User 1 family macro-F1 drops by no more than 0.01.
3. **D only:** its leak check passes. E's check is reported, but adoption is decided on F1.

The 8-class difference and its CI are reported too, but don't decide.

### Ablation steps

**A. Features v1, accelerometer subset, on the whole 15 s window (reference).** 116 features.
- It is exactly 03's accelerometer-only set: features v1 without `gyro_norm`, 4 channels × 29 statistics.
  - The 132 in the stop-point-1 proposal was a counting error (33 instead of 29 statistics per channel). There are
    no features beyond 03's 116.
- Channels:
  - `acc_norm` |a|;
  - `acc_vert` a · ĝ, with g a 0.3 Hz low-pass inside the window;
  - `acc_hor` |a − (a · ĝ) ĝ|;
  - `jerk_norm` |da/dt|.
- Per channel (29):
  - distribution: mean, std, min, max, p05, p25, p50, p75, p95, iqr, skew, kurt;
  - time domain: mad_diff, mcr, acf_peak, acf_lag_s, hjorth_mob, hjorth_comp;
  - spectrum: bandlog × 4 and bandrel × 4 (0–1, 1–3, 3–6, 6–12.5 Hz), dom_freq, spec_centroid, spec_entropy.
- Code: `tmd.features` unchanged except that the gyroscope channel is not computed. The window is 375 samples
  instead of 125, so Welch averages more segments.

**B. A + sub-window statistics.** +464 features (580 in total).
- The 116 features of A on each of the 5 sub-windows of 5 s with a 2.5 s stride inside the fragment (starts at
  samples 0, 62, 125, 187 and 250, as in `tmd.windows`).
- Then their mean, std, min and max across the 5 sub-windows: **116 features × 4 aggregations = 464.**
- A's 116 whole-window features stay.

**C. best + gravity features.** +21 features.
- Gravity is the window's mean direction: ĝ = mean(a) / |mean(a)| over 15 s.
- Features:
  - `grav_ratio` = |mean(a)| / mean(|a|). It is 1 when the phone keeps its orientation and lower when it turns
    (scale-free);
  - `tilt_*`: the angle between the 1 s moving mean of a and ĝ, in degrees. Mean, std, p95, max (4);
  - `subtilt_*`: the angles of the 5 sub-window mean directions, against ĝ (max, std) and between consecutive
    sub-windows (mean, max) (4);
  - projections on the fixed ĝ: `vfix` = a · ĝ and `hfix` = |a − (a · ĝ) ĝ|. Per channel: std, iqr, bandlog × 4
    (2 × 6 = 12).
- **Invariant to axis sign and axis order (iOS vs Android conventions).** Every C feature is an angle or a dot
  product between two vectors measured in the same phone frame. A sign flip of any axis, or a permutation of the
  axes, is an orthogonal transform: it changes both vectors alike and leaves angles and norms unchanged.
  - A unit test checks this: all 8 sign patterns × 6 axis orders give identical features.
  - Pose features (ĝ's own components, or |cos| of ĝ to each axis) are **not** included. Per-axis features change
    with the axis order, and they describe how the phone is held.

**D. best + context over neighbouring minutes in the same session** (past and future; batch prediction).
- **Base set (fixed, K = 40):** 10 of A's whole-window features per channel, 4 channels:
  - std, iqr, p50, acf_peak, dom_freq, spec_entropy;
  - bandlog × 4.
- **Neighbours:** the session's usable minutes at offsets −3 … −1 and +1 … +3 min.
  - The neighbour at offset j is the minute with |ts − (t + 60 j)| ≤ 30 s; ExtraSensory timestamps aren't on the
    minute grid.
- **Columns: 160, always.**
  - The value at −1 and at +1 (2 × 40).
  - The mean and the max over the available neighbours in ±3 min, excluding the minute itself (2 × 40).
- Missing neighbours stay NaN. There are no columns that count neighbours or describe the context length. If E is
  adopted, the base set is computed after the low-pass; the count doesn't change.
- **Check (dataset leak).** A small LightGBM (100 rounds, 15 leaves, fixed) predicts SHL vs ExtraSensory, out of
  fold on the paired folds, on three inputs:
  1. the 160 context columns alone;
  2. their missing-neighbour pattern alone (the 6 presence bits);
  3. the minute's own 40 base features (reference).

  **Passes if AUC(1) ≤ AUC(3) + 0.02 and AUC(2) ≤ 0.60.**
- **Fallback D′ (approved).** Used if the check fails.
  - In training, each SHL row gets a missing-neighbour pattern drawn from ExtraSensory: a **whole ±3-minute pattern
    (all 6 offsets jointly)** of a random ExtraSensory row of the **training folds only**.
  - The neighbours missing in that pattern are set to NaN before the context columns are computed.
  - The patterns are drawn with the row's seed (re-drawn per seed).
  - The check is rerun on D′.
  - SHL User 1 is scored both as recorded and masked; the SHL guard uses SHL as recorded.
  - D′ is its own row. Results will note that the real WAIIS gap pattern is unknown.
- **Descriptive check:** a classifier predicting the family from the missing-neighbour pattern alone, on
  ExtraSensory dev. Its macro-F1 is reported against the majority-class baseline, to see whether gaps carry the
  class.

**E. best + 10 Hz low-pass before feature extraction.**
- 4th-order Butterworth, zero-phase (`sosfiltfilt`), on the 3 axes of each 15 s window, before all features.
- **Check (platform leak, reported; not a gate).** A small LightGBM (fixed as in D) predicts iPhone vs Android on
  ExtraSensory dev only.
  - Out of fold by user (the dev folds), on the full feature set, with and without the low-pass.
  - The AUC is computed **within class, for walk and still**, plus their example-weighted mean.
  - E "passes" if the low-pass doesn't raise the weighted mean; the AUC change is reported.
  - Adoption is decided on F1 (rule above).

**F. best + HMM smoothing within sessions** (`tmd.hmm`).
- Rows: **forward-backward** and **Viterbi**, plus forward-only (causal), which is optional and cheap.
- States: the 8 classes.
- **Emissions:** the LightGBM probabilities as they are, with no division by class priors. LightGBM is trained with
  balanced class weights (see LightGBM), so its probabilities already act as likelihoods under a uniform prior.
- **Initial-state distribution:** class frequencies of the training labels used for the transitions.
- **Transitions per 1-min step,** per outer fold, counted from **all labelled minutes of the training folds, before
  caps and drops**, not from official examples only:
  - ExtraSensory: every minute with a class in `cls`, including dropped `SLEEPING` and capped still, not-carried
    and label-implausible minutes. The excluded user is left out, as everywhere.
  - SHL User 1: `cls`, or for mixed minutes their `majority_cls` (the minutes where transitions happen).
  - Pairs of consecutive labelled minutes in the same session, 1 min apart (|Δts − 60| ≤ 30 s); +1 pseudo-count,
    row-normalised.
- A gap of n min uses Aⁿ. A gap over 10 min, or a new session, restarts the chain from the initial distribution.
- The chain runs over all usable minutes of the session, and is scored on official examples.
- No HMM settings are tuned.

### Final configuration (approved 2026-10-29 at stop-point 2: B + HMM forward-backward)

- The best row is rerun with seeds 1 and 2. Development scores: the seed-0 value with its CI, plus the mean and
  range over the 3 seeds.
- Also reported on development data:
  - still recall split by the `quiet` flag, descriptive only. Quiet makes up 55% of ExtraSensory dev still and 41%
    of SHL User 1 still;
  - per-class recall for car, bus and train with user-bootstrap CIs (`bootstrap_users_recall`). ExtraSensory train
    has 3 dev users (folds 0, 2 and 3), so its CI will be wide.

### Test (stop-point 3, once) — exactly what is scored

*Fixed 2026-10-29, after stop-point 2 was approved and before any test user was read. Code: `test.py`.*

**Final configuration:** B + HMM forward-backward.
- 580 features: A's 116 + 464 sub-window statistics; no gravity, no context, no low-pass.

**Test models.** One per seed (0, 1, 2), each trained on **all official examples of all 47 ExtraSensory dev users +
SHL User 1** (all 5 folds; SHL weight 1). Balanced class weights, 02's LightGBM settings, no early stopping.
- **Number of trees per seed** = the median of that seed's 5 early-stopping counts from the cross-validation
  (`models/B_seed<k>_folds.json`), × 8 classes. The test model sees 5/5 of the dev data, while the counts were
  found on 3/5. The count is not rescaled.
- **HMM:** transitions (per 1-min step, +1 pseudo-count) and initial distribution from **all labelled dev minutes**
  of ExtraSensory dev and SHL User 1, before caps and drops, as in F. Forward-backward, restart after 10 min or at a
  new session.
- **Minutes:** the chain runs over every usable minute of the test users' sessions (`--part test` feature tables).
  Scoring is on official examples only.

**Scored.** Each seed is scored separately, one MLflow run per seed (`test_seed<k>`).
- **Headline** = the mean over the 3 seeds. All three values are reported too, each with its own user-bootstrap CI
  (1,000 resamples; resamples lacking a class are redrawn, and the redraw rate is reported).

1. **12 ExtraSensory test users** (`extrasensory_test_v2`), **overall, iPhone and Android**:
   - family and 8-class macro-F1 with CIs and their class lists (overall and Android: 7 classes / 5 families;
     iPhone: 6 / 4, since there is no train);
   - per-class recall with CIs; confusion matrices;
   - rail (family) and train (class) recall **before and after the HMM**, with CIs;
   - car, bus and train recall with CIs.
2. **Inference time:** single window (features + model) and per window over whole sessions (features + model +
   forward-backward). 200 test sessions with ≥ 3 usable minutes, single thread, after warm-up, median and p95.
   **Model size** per seed.
3. **SHL Users 2–3 rail check** (`shl_user23_test_v2`), reported separately:
   - family and 8-class macro-F1, per-class recall, and rail / train / subway recall before and after the HMM;
   - no CI: 2 users.
4. **Motorbike check set**, last. It holds the windowed minutes of `reason = check:MOTORBIKE`, all from user
   33A85C34, a reserved test user.
   - Reported as a histogram of predicted classes, before and after the HMM.
   - Descriptive only, labelled as reserved-test data from one user. Motorbike is never trained, and its label is
     not scored.

`test.py` refuses to run a second time once `test_seed*` runs exist.

### Reported metrics (every run)

- Pooled family and 8-class macro-F1 with their class lists and user-bootstrap CIs.
- Per-class recall with CIs, and confusion matrices (8-class and family; figures in MLflow).
- Per-fold spread.
- SHL User 1 scores, with rail recall (same-person day split).
- The paired difference CI against the best row so far.
- **Inference time**, single thread, after warm-up, median and p95. Every row reports both numbers:
  - **Single window:** features of one 15 s window + model (`tmd.evaluate.inference_latency`, with a new `shape`
    argument).
  - **Pipeline:** whole sessions as the pipeline runs them: features of every usable minute, context aggregation,
    model, HMM smoothing.
    - Time per session ÷ the number of windows in it.
    - Median and p95 over 200 ExtraSensory dev sessions of fold 0 with ≥ 3 usable minutes (seed 0), fold-0 model,
      after one warm-up pass.
- **Model size:** the mean of the 5 fold models (`model_size_bytes`). The HMM adds 8 × 8 + 8 numbers.
- **MLflow inputs:**
  - `extrasensory_dev_v2` and `shl_user1_dev`, context `training`, `oof_eval=True`, `window_s=15`, `stride_s=60`;
  - at stop-point 3 also `extrasensory_test_v2` and `shl_user23_test_v2`, context `evaluation`;
  - label rules `bench-v1`.

### Code and files

- **Shared code** (`src/tmd`):
  - `acc_features.py`: the accelerometer-only feature steps (A–C, E), versioned `acc-v1`. Later models reuse it as
    the reference.
  - `hmm.py`: new `step_s` and `pi` arguments and `forward_backward`. Defaults unchanged.
  - `evaluate.py`: new `shape` argument of `inference_latency` (default unchanged), and `bootstrap_users_diff`.
  - **Tests** (`tests/`): the defaults of `tmd.hmm` and `inference_latency` reproduce the current outputs, against
    golden values saved before the change. A further test covers the axis invariance of C.
- **This folder:** `context.py` and the scripts. Context moves to `src/tmd` only if it is adopted.
- **Feature cache:** `<data>/processed/05_acc-baseline_waiis-bench/`, named by parameters, e.g.
  `es_dev_25hz_w15s_acc-v1_sub5s_grav_lp-none.parquet`.
- Scripts print a short summary; full logs go to `logs/`.

### Estimated run time

- Features: about 0.2 ms per window, batched (measured). ~330k usable minutes take ~1–2 min per variant.
- LightGBM: ~15–25 min per row (5 folds × inner fit + refit, 8 threads).
- Rows: A (×2, + 2 SHL-weight reruns if needed), B, C, D, possibly D′, E. F needs no retraining. Plus checks and
  latency (~15 min). About 2.5–3.5 h.
- Final seeds 1–2: ~45 min. Test: ~15 min.
- **Total ≈ 3.5–4.5 h of compute.**

### Stop-points

1. ~~Plan.~~ Approved 2026-10-29 with changes.
2. ~~Ablation results and proposed final configuration.~~ Approved 2026-10-29 with changes (bootstrap fix, clean
   rerun, rail diagnostics, test protocol).
3. ~~Final configuration on the test users and the SHL rail check; Results, draft Conclusion, STATUS.md.~~ Done
   2026-10-30; Conclusion awaiting approval.

## Results

### Stop-point 1 (2026-10-29)

- The logging example in `CLAUDE.md` (line 51) and the `src/tmd/tracking.py` docstring (line 11) used the legacy
  split name `extrasensory_dev`. Both now use `extrasensory_dev_v2`; committed separately (`5852704`).
- Checked for the approved changes:
  - **Class weights:** 02's `class_weights` is balanced (w_c = N / (8 · n_c)), so the LightGBM probabilities are
    used as HMM emissions as they are.
  - **Macro-F1:** it has averaged over the classes present in the true labels since `1f52628`, so earlier numbers
    are not affected.
  - **Feature count:** features v1 has 29 statistics per channel. A is therefore 116 features (03's accelerometer
    set), not 132 as first proposed.
  - **Motorbike:** the 83 windowed motorbike minutes all belong to test user 33A85C34.

### Stop-point 2: ablation on development data (2026-10-29)

**Commands** (in this folder; full output in `logs/`):

```
uv run pytest -q ../../tests                                   # 35 passed (golden HMM / latency defaults, axis invariance)
uv run python make_features.py                                 # 4 tables, ~2 min
uv run python cv.py --name A
uv run python cv.py --name A_esonly --shl-w 0 --vs A
uv run python cv.py --name B --sub --vs A
uv run python cv.py --name C --sub --grav --vs B
uv run python cv.py --name D --sub --ctx plain --vs B   && uv run python checks.py dataset --row D
uv run python cv.py --name Dm --sub --ctx masked --vs B && uv run python checks.py dataset --row Dm
uv run python cv.py --name E --sub --lp 10 --vs B       && uv run python checks.py platform --row B
uv run python cv.py --name F_{fb,viterbi,forward} --from-row B --hmm {fb,viterbi,forward} --vs B
uv run python cv.py --name X_Dm_{fb,viterbi} --from-row Dm --hmm {fb,viterbi} --vs B   # read-out, outside the chain
uv run python compare.py --rows F_fb:F_viterbi F_fb:F_forward X_Dm_fb:F_fb Dm:D --detail F_fb
uv run python tables.py                                        # the tables below, from MLflow
```

- **Feature tables:** `<data>/processed/05_acc-baseline_waiis-bench/{extrasensory,shl_user1}_dev_25hz_w15s_acc-v1_lp-{none,10hz}.parquet`.
  - ExtraSensory: 291,155 usable minutes (118,986 official). SHL User 1: 36,806 (23,076 official).
  - 601 features, no NaN.
- **Runs:** all seed 0, one MLflow run per row and check. All are **dirty** at commit `5852704`: the experiment code
  was not yet committed.
- **Class lists:** ExtraSensory macro-F1 is over 5 families and 7 classes (no subway); SHL over 5 and 8.

**Ablation (pooled out-of-fold, seed 0).**

> **CIs recomputed at stop-point 3.** All CIs in this section come from the fixed bootstrap: a resample that lacks
> a class of the full set is redrawn, and the redraw rate on ExtraSensory dev is 4.6%. They come from the rescored
> runs `<row>_seed0_ci-redraw` (clean, commit `bb503f1`).
> - The stop-point-2 CIs had inflated upper bounds (e.g. F_fb 0.615 [0.567–0.745]); they stay in MLflow under
>   commit `5852704` and are superseded.
> - Point estimates and every adopt/reject decision are unchanged.

- Δ = paired user-bootstrap difference against the `vs` row (1,000 resamples).
- SHL is a same-person day split, with no CI.
- Trees = the mean over the 5 fold models (rounds × 8 classes).
- Single window = features + model for one 15 s window. Pipeline = whole sessions ÷ windows (200 fold-0
  ExtraSensory sessions).
- Latency and size are copied from the original runs.
- Generated by `tables.py` (from MLflow, or `--csv` from `results/runs.csv`).

| row | vs | ES family F1 [CI] | Δ family vs [paired CI] | ES 8-class F1 [CI] | SHL family | Δ SHL | SHL 8-class | SHL train / subway recall | trees (mean) | single window p50 / p95 ms | pipeline p50 / p95 ms | size MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_esonly | A | 0.452 [0.408–0.478] | -0.031 [-0.043, -0.020] | 0.366 [0.323–0.400] | 0.620 | -0.157 | 0.485 | 0.14 / 0.00 | 123 | 2.0 / 2.1 | 0.24 / 0.75 | 0.4 |
| A | – | 0.483 [0.439–0.513] | – | 0.411 [0.347–0.474] | 0.776 | – | 0.734 | 0.60 / 0.58 | 371 | 2.1 / 2.3 | 0.25 / 0.77 | 1.3 |
| B | A | 0.496 [0.451–0.526] | +0.013 [+0.008, +0.018] | 0.427 [0.357–0.495] | 0.790 | +0.014 | 0.755 | 0.66 / 0.60 | 365 | 10.8 / 11.2 | 1.06 / 3.83 | 1.3 |
| C | B | 0.498 [0.454–0.528] | +0.002 [-0.001, +0.005] | 0.430 [0.361–0.499] | 0.794 | +0.004 | 0.759 | 0.67 / 0.60 | 363 | 11.9 / 12.1 | 1.19 / 4.26 | 1.3 |
| D | B | 0.556 [0.510–0.584] | +0.060 [+0.049, +0.070] | 0.479 [0.406–0.548] | 0.928 | +0.138 | 0.870 | 0.80 / 0.70 | 384 | 11.0 / 12.0 | 1.12 / 4.01 | 1.4 |
| Dm | B | 0.560 [0.515–0.591] | +0.064 [+0.052, +0.078] | 0.485 [0.411–0.555] | 0.927 | +0.137 | 0.867 | 0.79 / 0.69 | 379 | 11.0 / 12.1 | 1.09 / 4.00 | 1.4 |
| E | B | 0.490 [0.444–0.521] | -0.006 [-0.010, -0.002] | 0.423 [0.354–0.490] | 0.779 | -0.011 | 0.752 | 0.66 / 0.60 | 352 | 11.3 / 12.0 | 1.10 / 3.99 | 1.3 |
| F_forward | B | 0.606 [0.562–0.636] | +0.111 [+0.092, +0.132] | 0.540 [0.459–0.621] | 0.923 | +0.133 | 0.900 | 0.85 / 0.80 | 365 | 10.8 / 13.2 | 1.09 / 3.94 | 1.3 |
| F_fb | B | 0.615 [0.567–0.653] | +0.120 [+0.096, +0.144] | 0.552 [0.467–0.638] | 0.951 | +0.161 | 0.936 | 0.94 / 0.87 | 365 | 11.1 / 12.3 | 1.19 / 4.21 | 1.3 |
| F_viterbi | B | 0.612 [0.562–0.648] | +0.116 [+0.094, +0.140] | 0.552 [0.466–0.636] | 0.948 | +0.157 | 0.933 | 0.94 / 0.87 | 365 | 10.8 / 11.1 | 1.10 / 4.00 | 1.3 |
| X_Dm_fb | B | 0.615 [0.571–0.648] | +0.119 [+0.101, +0.139] | 0.551 [0.468–0.628] | 0.959 | +0.168 | 0.938 | 0.91 / 0.86 | 379 | 10.7 / 11.0 | 1.12 / 4.03 | 1.4 |
| X_Dm_viterbi | B | 0.614 [0.569–0.648] | +0.119 [+0.099, +0.140] | 0.553 [0.470–0.630] | 0.957 | +0.167 | 0.939 | 0.91 / 0.86 | 379 | 10.7 / 11.6 | 1.12 / 4.00 | 1.4 |

| row | ES family F1 per fold (min–max) | ES family F1 iPhone / Android | ES recall still walk run bike car bus train | ES rail recall | CI redraw rate (family / class) |
|---|---|---|---|---|---|
| A_esonly | 0.406–0.601 | 0.474 / 0.412 | 0.74 0.65 0.24 0.66 0.51 0.37 0.05 | 0.02 | 0.046 / 0.046 |
| A | 0.433–0.639 | 0.503 / 0.446 | 0.75 0.69 0.25 0.67 0.52 0.22 0.18 | 0.29 | 0.046 / 0.046 |
| B | 0.440–0.662 | 0.518 / 0.456 | 0.77 0.70 0.25 0.68 0.54 0.28 0.16 | 0.24 | 0.046 / 0.046 |
| C | 0.448–0.660 | 0.518 / 0.460 | 0.77 0.70 0.25 0.68 0.55 0.27 0.20 | 0.28 | 0.046 / 0.046 |
| D | 0.504–0.732 | 0.575 / 0.517 | 0.84 0.74 0.27 0.74 0.61 0.32 0.09 | 0.13 | 0.046 / 0.046 |
| Dm | 0.507–0.730 | 0.576 / 0.523 | 0.84 0.74 0.28 0.75 0.61 0.32 0.18 | 0.26 | 0.046 / 0.046 |
| E | 0.439–0.655 | 0.510 / 0.453 | 0.77 0.69 0.25 0.68 0.54 0.27 0.19 | 0.28 | 0.046 / 0.046 |
| F_forward | 0.538–0.832 | 0.618 / 0.572 | 0.89 0.68 0.25 0.75 0.69 0.28 0.08 | 0.08 | 0.046 / 0.046 |
| F_fb | 0.545–0.851 | 0.623 / 0.584 | 0.88 0.75 0.26 0.76 0.73 0.34 0.06 | 0.08 | 0.046 / 0.046 |
| F_viterbi | 0.546–0.852 | 0.622 / 0.580 | 0.88 0.74 0.26 0.75 0.72 0.34 0.06 | 0.07 | 0.046 / 0.046 |
| X_Dm_fb | 0.559–0.829 | 0.625 / 0.587 | 0.89 0.74 0.29 0.78 0.71 0.35 0.08 | 0.08 | 0.046 / 0.046 |
| X_Dm_viterbi | 0.558–0.829 | 0.624 / 0.586 | 0.89 0.74 0.29 0.77 0.71 0.37 0.08 | 0.08 | 0.046 / 0.046 |

**Decisions under the rule.**
- **SHL User 1 (diagnostic):** without it, ExtraSensory family F1 is lower by −0.031 [−0.043, −0.020]. SHL helps,
  so there is no down-weighting (weight 1).
- **B adopted:** the CI excludes 0; SHL +0.014.
- **C not adopted:** the CI contains 0, and C is not cheaper.
- **D not adopted:** the check fails (below). D′ is not adopted either, for the same reason.
- **E not adopted:** −0.006, with a CI below 0; SHL −0.011.
- **F adopted:** all three decoders, with CIs far above 0.
  - Forward-backward vs Viterbi: +0.003 [−0.001, +0.007] family; 8-class −0.000 [−0.004, +0.003].
  - Forward-backward vs forward: +0.009 [−0.006, +0.023] family; SHL +0.028.
  - These comparisons come from `compare.py`, recomputed with the fixed bootstrap (`logs/compare_seed0_redraw.log`).

**Check D (dataset leak), out of fold on the paired folds, official rows.**

| row | AUC context alone (160) | AUC gap pattern alone (6) | AUC own minute (40) | pass? | family-from-gaps macro-F1 (majority) |
|---|---|---|---|---|---|
| D | 0.981 | 0.559 | 0.947 | **no** (0.981 > 0.947 + 0.02) | 0.111 (0.175) |
| D′ | 0.976 | 0.431 | 0.947 | **no** (0.976 > 0.967) | 0.111 (0.175) |

- **The gap pattern passes in both** (≤ 0.60), and gaps carry no class (below the majority baseline).
- What fails is the context **values**: 160 columns that summarise 6 neighbouring minutes separate SHL from
  ExtraSensory better than one minute's 40 features do.
  - Likely cause, untested: averaging over minutes sharpens the device / position signature (one SHL phone at the
    hips) rather than leaking the context length.
- D′ (approved fallback) changes only the gap pattern, so it could not fix this.
- **The real WAIIS gap pattern is unknown**; D′'s ExtraSensory-like gaps are an assumption.

**Check E (platform, ExtraSensory dev, out of fold by user, B's 580 features), AUC iPhone vs Android.**

| | still | walk | weighted mean |
|---|---|---|---|
| no low-pass | 0.836 | 0.510 | 0.796 |
| 10 Hz low-pass | 0.824 | 0.497 | 0.783 |

- The low-pass lowers the AUC by 0.013, so the check passes; E is still not adopted on F1.
- Platform is near chance during walking but clearly visible when still.

**Read-out: context after the HMM** (X_Dm_*, not part of the ablation chain, so a deviation from the Plan).
- D′ + forward-backward vs B + forward-backward: ExtraSensory family −0.000 [−0.011, +0.012]; SHL +0.007.
- Once the HMM is on, the context adds nothing on ExtraSensory. Its +0.06 alone is what the HMM also captures.
- D′ vs D: +0.004 [−0.002, +0.013].

**Proposed final configuration: B + HMM forward-backward** (F_fb).
- Features: A's 116 + the 464 sub-window statistics = 580, no low-pass, no gravity, no context.
- LightGBM, ExtraSensory dev + SHL User 1 at weight 1.
- HMM forward-backward over every usable minute of the session.
- Forward-backward over Viterbi: equal F1 (CI includes 0), and it gives per-minute posteriors. The on-demand
  labelling decision (2026-10-28) needs those to pick the least certain segments.
- Development: ExtraSensory family 0.615 [0.567–0.653], 8-class 0.552 [0.467–0.638]; SHL User 1 family 0.951,
  8-class 0.936 (same-person day split).
- Latency: single window 11.1 / 12.3 ms (p50 / p95), pipeline 1.19 / 4.21 ms per window. Model 1.3 MB.

**Final-configuration read-outs, seed 0 preview** (3 seeds: stop-point 3 below).
- **Still recall by `quiet`** (descriptive):
  - ExtraSensory: quiet 0.996 (n = 50,845), not quiet 0.736 (n = 41,885);
  - SHL User 1: quiet 0.888 (1,387), not quiet 0.699 (1,979).
- **ExtraSensory car / bus / train recall [user-bootstrap CI]:**
  - car 0.729 [0.628–0.811];
  - bus 0.340 [0.173–0.477];
  - train 0.058 [0.000–0.113].
- **Where they go:**
  - car → still 1,321, walk 818 (of 8,028);
  - bus → walk 354, car 312, still 180 (of 1,354);
  - train → car 85, still 73, walk 19, train 11 (of 190).

**Deviations and details** (file:line):
- The X_Dm_fb and X_Dm_viterbi rows are an extra read-out, outside the ablation chain.
- `pipeline.py:158`: the initial distribution is the class frequencies + 1 count per class, so classes with no
  training minutes don't get log(0).
- `pipeline.py:157`: `tol_s=30.5` implements "|Δts − 60| ≤ 30 s" on integer timestamps (the comparison is strict).
- `pipeline.py:69`: class weights W / (8 · W_c), as in 02's `class_weights`; every present class has equal total
  weight.
- `cv.py` `latency`:
  - the single-window number for context rows uses random neighbour features (precomputed for their own minutes);
  - the pipeline number uses 200 fold-0 ExtraSensory sessions with ≥ 3 usable minutes (seed 0) and the fold-0
    model.
  - The pipeline is ~10× cheaper per window than the single window because features and model run in one batch per
    session.
- **Early stopping stops early again:** 24–75 rounds per fold (A_esonly: 8–25), against the 1,000 allowed. Reported
  as planned; protocol unchanged.

**Observations and open questions for stop-point 2.** *Answered 2026-10-29: final configuration approved; bootstrap
fixed (1); rail loss accepted for 05, diagnostics below (3); no further context work in 05 (4).*
1. **The HMM destroys ExtraSensory rail.**
   - Rail recall falls from 0.24 (B) to 0.08 (F_fb), while SHL train rises from 0.66 to 0.94.
   - ExtraSensory train is 190 minutes from 3 users, and its minutes go to car and still.
   - Transitions are pooled: SHL User 1 contributes its rail sequences, ExtraSensory its 250k mostly still minutes.
   - Family F1 still rises, because rail is one of 5 families and the others improve more. The test has train from
     only 2 Android users.
   - **Accept, or look at it before the test?**
2. **The bootstrap CIs are skewed upward** (e.g. 0.615 [0.567–0.745]). In about 4.5% of resamples no train user is
   drawn, so (44 / 47)^47 ≈ 0.045. Rail then drops out of the macro average, and those resamples score higher.
   - This is a consequence of "macro-F1 over present classes".
   - Paired differences aren't affected the same way, since both rows lose rail together.
   - Keep as is, with this note?
3. **Check D:** under the fixed criterion context is out, but the failure comes from the values, not the gap pattern
   (see above). Given the read-out (no ExtraSensory gain after the HMM), I propose no further work on context in 05.

### Stop-point 3: final configuration and test (2026-10-29/30)

**Commands** (in this folder; full output in `logs/`):

```
uv run pytest -q ../../tests                                   # 39 passed
uv run python rescore.py                                       # 12 runs <row>_seed0_ci-redraw (fixed bootstrap)
uv run python cv.py --name B --sub --vs A                      # seed 0 again, clean (reproduction)
uv run python cv.py --name F_fb --from-row B --hmm fb --vs B
uv run python cv.py --name B --sub --seed {1,2} && uv run python cv.py --name F_fb --from-row B --hmm fb --vs B --seed {1,2}
uv run python rail.py --seed {0,1,2}                           # logs/rail_seed<k>.log
uv run python compare.py --rows F_fb:F_viterbi F_fb:F_forward X_Dm_fb:F_fb Dm:D --detail F_fb   # + --seed 1, 2
uv run python make_features.py --part test --sources extrasensory shl_user23
uv run python test.py                                          # 3 runs test_seed<k>, scored once
uv run python ../../src/tmd/tracking.py export 05_acc-baseline_waiis-bench   # results/runs.csv (54 runs)
uv run python tables.py --csv                                  # every table below and above
```

**1. Bootstrap fix** (`tmd.evaluate.user_resamples`, commit `9a1c04f`).
- A resample that lacks a class present in the full evaluation set's true labels is redrawn. The redraw rate is
  returned through `stats=` and logged as `*_redraw_rate`.
- `bootstrap_users`, `bootstrap_users_recall` and `bootstrap_users_diff` all use it. With no class missing, the
  draws are those of the old code.
- Test: `tests/test_bootstrap_redraw.py`.
- Redraw rates:
  - ExtraSensory dev: 4.6% (train in 3 of 47 users; (44 / 47)^47 = 0.046 expected);
  - test: 10.6–10.8% overall (train in 2 of 12 users), 16.5–18.4% Android, 0.3% iPhone family and 17.5% iPhone
    8-class.
- All dev CIs in Results were recomputed (stop-point 2 tables, rescored runs). Upper bounds fell, e.g. F_fb
  0.615 [0.567–0.745] → [0.567–0.653]. **No adopt/reject decision changed.**

**2. Reproduction at a clean commit.**
- The final configuration, seed 0, was rerun at `bb503f1`:
  - B's and F_fb's out-of-fold probabilities are **bit-identical** to the dirty runs (`5852704`);
  - the 5 fold models are identical tree for tree;
  - all 78 non-latency metrics agree, except the bootstrap CI bounds, which the fix changes by design.
- A first rerun at `9a1c04f` gave the same identical probabilities. It was still tagged dirty, because the scripts
  write their own logs into the tracked `logs/` folder before the run starts.
  - Fix (commit `0ac09b7`): `tmd.tracking.is_dirty` ignores `experiments/*/logs/**`; test
    `tests/test_tracking_dirty.py`; `CLAUDE.md` line 53.
  - The 18 dirty `9a1c04f` runs stay in MLflow and in `runs.csv`, superseded by the clean `bb503f1` runs.

**3. Final configuration on development data, 3 seeds** (clean, `bb503f1`).

| row | seed | ES family F1 [CI] | ES 8-class F1 [CI] | ES car / bus / train recall [CI] | ES rail recall | SHL family | SHL 8-class | SHL train / subway | trees (mean) | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| B | 0 | 0.496 [0.451–0.526] | 0.427 [0.357–0.495] | 0.54 [0.46–0.62] / 0.28 [0.20–0.33] / 0.16 [0.00–0.23] | 0.24 | 0.790 | 0.755 | 0.66 / 0.60 | 365 | bb503f1 |
| B | 1 | 0.493 [0.448–0.524] | 0.426 [0.358–0.495] | 0.54 [0.46–0.63] / 0.26 [0.20–0.32] / 0.18 [0.00–0.23] | 0.25 | 0.793 | 0.754 | 0.66 / 0.59 | 390 | bb503f1 |
| B | 2 | 0.491 [0.446–0.523] | 0.424 [0.355–0.491] | 0.54 [0.46–0.63] / 0.26 [0.19–0.32] / 0.16 [0.00–0.23] | 0.25 | 0.792 | 0.753 | 0.65 / 0.60 | 363 | bb503f1 |
| B | mean (range) | 0.493 (0.491–0.496) | 0.425 (0.424–0.427) | | | 0.792 | 0.754 | | | |
| F_fb | 0 | 0.615 [0.567–0.653] | 0.552 [0.467–0.638] | 0.73 [0.63–0.81] / 0.34 [0.17–0.48] / 0.06 [0.00–0.11] | 0.08 | 0.951 | 0.936 | 0.94 / 0.87 | 365 | bb503f1 |
| F_fb | 1 | 0.616 [0.567–0.653] | 0.551 [0.465–0.637] | 0.72 [0.62–0.81] / 0.33 [0.18–0.46] / 0.06 [0.00–0.12] | 0.08 | 0.950 | 0.936 | 0.95 / 0.87 | 390 | bb503f1 |
| F_fb | 2 | 0.615 [0.566–0.653] | 0.550 [0.467–0.634] | 0.72 [0.62–0.81] / 0.32 [0.17–0.45] / 0.06 [0.00–0.12] | 0.08 | 0.943 | 0.931 | 0.95 / 0.88 | 363 | bb503f1 |
| F_fb | mean (range) | 0.615 (0.615–0.616) | 0.551 (0.550–0.552) | | | 0.948 | 0.934 | | | |

- Headline (dev): ExtraSensory family **0.615** (range 0.615–0.616), 8-class 0.551; SHL User 1 family 0.948, a
  same-person day split.
- **Still recall by `quiet`** (descriptive, seeds 0 / 1 / 2):
  - ExtraSensory: quiet 0.996 / 0.994 / 0.996 (n = 50,845); not quiet 0.736 / 0.735 / 0.736 (n = 41,885);
  - SHL User 1: quiet 0.888 / 0.846 / 0.831 (n = 1,387); not quiet 0.699 / 0.710 / 0.660 (n = 1,979).
- **Car / bus / train recall** [user-bootstrap CI]: in the table (seed 0: car 0.73 [0.63–0.81], bus 0.34
  [0.17–0.48], train 0.06 [0.00–0.11]).
- **Where they go** (seed 0): car → still 1,321, walk 818 of 8,028; bus → walk 354, car 312 of 1,354.

**4. Rail diagnostics** (descriptive, no tuning; `rail.py`, `logs/rail_seed<k>.log`).

ExtraSensory dev train minutes, before (B) and after (F_fb) the HMM, seed 0 (seeds 1–2 within ±0.03):

| user | platform | fold | train minutes | rail recall before | rail recall after | mean P(rail) before |
|---|---|---|---|---|---|---|
| 3600D531 | iPhone | 0 | 69 | 0.13 | 0.00 | 0.19 |
| A76A5AF5 | iPhone | 3 | 24 | 0.13 | 0.00 | 0.12 |
| B7F9D634 | Android | 2 | 97 | 0.35 | 0.15 | 0.33 |
| all | | | 190 | 0.24 | 0.08 | 0.26 |

- **Mean P(rail) before smoothing:** 0.26 on train minutes, against 0.09 on all other minutes. The rail signal is
  present but rarely the largest.
- **Train minutes after the HMM:**
  - before: still 51, walk 15, car 56, bus 19, train 31, subway 14;
  - after: still 73, walk 19, car 85, train 11, subway 2.
- **False rail elsewhere:** non-train minutes predicted as rail fall from 6,899 to 282. The HMM removes most false
  rail too, which is where the family F1 gain on rail-free minutes comes from.
- **Test train minutes** (seed 0, `logs/rail_test_seed0.log`): the HMM does **not** lose rail there.
  - 1DBB0F6F (15 minutes): 0.07 → 0.07;
  - 33A85C34 (37 minutes): 0.51 → 0.54; its mean P(rail) before smoothing is 0.49.
- **Accepted for 05** (approved at stop-point 2); the trade-off is in the Conclusion.

**5. Test** (scored once, `test.py`, clean `a62fbd9`; protocol in the Plan). Each test model has 46–48 rounds (the
median of its seed's CV counts) × 8 classes.

Class lists:
- ExtraSensory overall and Android: 7 classes / 5 families;
- iPhone: 6 classes (still, walk, run, bike, car, bus) / 4 families (no rail);
- SHL Users 2–3: 8 / 5.

CIs come from user bootstraps over the 12 test users, with redraw rates as above.

| test (B + HMM fb) | mean of 3 seeds | seed 0 [CI] | seed 1 [CI] | seed 2 [CI] |
|---|---|---|---|---|
| ES all family macro-F1 | **0.698** | 0.704 [0.631–0.746] | 0.698 [0.628–0.748] | 0.693 [0.623–0.745] |
| ES all family macro-F1, before HMM | **0.520** | 0.521 [0.462–0.546] | 0.518 [0.455–0.544] | 0.522 [0.462–0.548] |
| ES all class macro-F1 | **0.666** | 0.676 [0.518–0.724] | 0.667 [0.513–0.726] | 0.657 [0.500–0.711] |
| ES all class macro-F1, before HMM | **0.494** | 0.494 [0.380–0.524] | 0.492 [0.378–0.526] | 0.496 [0.380–0.529] |
| ES iphone family macro-F1 | **0.801** | 0.803 [0.759–0.841] | 0.801 [0.758–0.836] | 0.800 [0.757–0.839] |
| ES iphone family macro-F1, before HMM | **0.661** | 0.663 [0.576–0.701] | 0.656 [0.565–0.697] | 0.664 [0.581–0.706] |
| ES iphone class macro-F1 | **0.762** | 0.763 [0.593–0.817] | 0.774 [0.610–0.816] | 0.748 [0.566–0.809] |
| ES iphone class macro-F1, before HMM | **0.555** | 0.555 [0.412–0.612] | 0.553 [0.407–0.618] | 0.557 [0.410–0.621] |
| ES android family macro-F1 | **0.690** | 0.693 [0.627–0.730] | 0.694 [0.614–0.736] | 0.682 [0.596–0.730] |
| ES android family macro-F1, before HMM | **0.490** | 0.490 [0.430–0.522] | 0.489 [0.420–0.526] | 0.489 [0.423–0.528] |
| ES android class macro-F1 | **0.665** | 0.670 [0.511–0.711] | 0.667 [0.509–0.711] | 0.658 [0.504–0.696] |
| ES android class macro-F1, before HMM | **0.492** | 0.493 [0.353–0.535] | 0.490 [0.348–0.535] | 0.494 [0.348–0.536] |
| ES all rail recall, before HMM | **0.391** | 0.385 [0.067–0.514] | 0.404 [0.133–0.514] | 0.385 [0.067–0.514] |
| ES all rail recall, after HMM | **0.404** | 0.404 [0.067–0.541] | 0.404 [0.067–0.541] | 0.404 [0.067–0.541] |
| ES all train recall, before HMM | **0.346** | 0.327 [0.067–0.432] | 0.365 [0.133–0.459] | 0.346 [0.133–0.432] |
| ES all train recall, after HMM | **0.404** | 0.404 [0.067–0.541] | 0.404 [0.067–0.541] | 0.404 [0.067–0.541] |
| ES android rail recall, before HMM | **0.391** | 0.385 [0.067–0.514] | 0.404 [0.133–0.514] | 0.385 [0.067–0.514] |
| ES android rail recall, after HMM | **0.404** | 0.404 [0.067–0.541] | 0.404 [0.067–0.541] | 0.404 [0.067–0.541] |
| ES android train recall, before HMM | **0.346** | 0.327 [0.067–0.432] | 0.365 [0.133–0.459] | 0.346 [0.133–0.432] |
| ES android train recall, after HMM | **0.404** | 0.404 [0.067–0.541] | 0.404 [0.067–0.541] | 0.404 [0.067–0.541] |
| ES all recall still | **0.904** | 0.906 [0.866–0.939] | 0.905 [0.864–0.939] | 0.901 [0.854–0.935] |
| ES all recall walk | **0.715** | 0.709 [0.659–0.779] | 0.721 [0.671–0.786] | 0.715 [0.659–0.787] |
| ES all recall run | **0.708** | 0.704 [0.000–0.861] | 0.716 [0.000–0.881] | 0.704 [0.000–0.861] |
| ES all recall bike | **0.831** | 0.828 [0.690–0.915] | 0.833 [0.710–0.913] | 0.831 [0.696–0.914] |
| ES all recall car | **0.751** | 0.758 [0.678–0.840] | 0.749 [0.671–0.830] | 0.746 [0.671–0.826] |
| ES all recall bus | **0.218** | 0.240 [0.193–0.418] | 0.216 [0.154–0.444] | 0.199 [0.120–0.441] |
| ES all recall train | **0.404** | 0.404 [0.067–0.541] | 0.404 [0.067–0.541] | 0.404 [0.067–0.541] |

| SHL Users 2–3 (no CI) | mean | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| family macro-F1 | 0.932 | 0.933 | 0.927 | 0.935 |
| 8-class macro-F1 | 0.795 | 0.797 | 0.790 | 0.798 |
| family macro-F1, before HMM | 0.789 | 0.793 | 0.792 | 0.781 |
| 8-class macro-F1, before HMM | 0.639 | 0.646 | 0.634 | 0.636 |
| recall still | 0.901 | 0.899 | 0.891 | 0.912 |
| recall walk | 0.995 | 0.995 | 0.995 | 0.995 |
| recall run | 0.267 | 0.244 | 0.311 | 0.244 |
| recall bike | 0.965 | 0.965 | 0.965 | 0.965 |
| recall car | 0.982 | 0.988 | 0.973 | 0.985 |
| recall bus | 0.920 | 0.933 | 0.893 | 0.933 |
| recall train | 0.741 | 0.742 | 0.739 | 0.742 |
| recall subway | 0.534 | 0.570 | 0.475 | 0.556 |
| rail recall, before / after HMM | | 0.72 / 0.85 | 0.74 / 0.85 | 0.73 / 0.85 |

| seed | trees | model MB | single window p50 / p95 ms | pipeline p50 / p95 ms | CI redraw rate ES all family / iPhone family |
|---|---|---|---|---|---|
| 0 | 376 | 1.36 | 11.5 / 13.1 | 0.94 / 4.53 | 0.106 / 0.003 |
| 1 | 368 | 1.33 | 13.7 / 18.2 | 1.23 / 5.04 | 0.106 / 0.003 |
| 2 | 384 | 1.39 | 13.8 / 18.7 | 1.32 / 5.44 | 0.106 / 0.003 |

- **Headline:** ExtraSensory test family macro-F1 **0.698** (seeds 0.704 / 0.698 / 0.693), 8-class **0.666**.
  - iPhone 0.801 (over 4 families), Android 0.690.
  - Before the HMM: 0.520 family, 0.494 8-class. The HMM adds +0.18 family on test, against +0.12 on dev.
- **The test scores are higher than dev** (0.698 vs 0.615 family). Different users; not investigated. iPhone test
  has no rail family, and test still recall is 0.90 (dev 0.88).
- **Rail and train on test:**
  - rail 0.39 → 0.40 and train 0.35 → 0.40 through the HMM; after the HMM, train recall is the same in all seeds
    (21 of 52 minutes);
  - CI [0.07–0.54]: 2 Android users only.
- **Bus on test is weak:** 0.22 [0.12–0.44] mean (dev 0.33).
- **SHL Users 2–3 rail check:** family 0.932, 8-class 0.795 (mean). Rail recall rises 0.73 → 0.85 through the HMM;
  train 0.74, subway 0.53.
- **Inference time and size:** single window 11.5–13.8 ms p50 (p95 13–19 ms); pipeline 0.9–1.3 ms per window
  (p95 4.5–5.4 ms); model 1.33–1.39 MB. Seeds 1–2 were slower than seed 0 on the same code; machine load, not
  investigated.

**6. Motorbike check set** (reserved-test data from one user, 33A85C34; descriptive only; scored after every
choice was frozen). 83 windowed motorbike minutes, predicted classes after the HMM (before):

| seed | still | walk | bike | car | bus | other |
|---|---|---|---|---|---|---|
| 0 | 7 (9) | 19 (15) | 0 (1) | 55 (32) | 2 (26) | 0 |
| 1 | 7 (9) | 19 (16) | 0 (2) | 45 (28) | 12 (28) | 0 |
| 2 | 7 (9) | 20 (16) | 0 (2) | 24 (26) | 32 (31) | 0 |

- Road family 56–57 of 83 (67–69%) in every seed. Car vs bus depends on the seed. Walk is 19–20 (23–24%).

**7. Notes.**
- **Early stopping stops early:**
  - CV: 24–75 rounds per fold out of 1,000 allowed (ExtraSensory only: 8–25);
  - final configuration: 45.4–48.8 mean rounds per seed;
  - test models: 46–48 rounds.
  - Reported as planned; the protocol didn't change.
- **Dirty runs:**
  - all 15 ablation and check runs of stop-point 2 are logged dirty at `5852704` (code not yet committed);
  - the 6 seed reruns and the 12 rescorings at `9a1c04f` are dirty only through their own log files (see 2).
  - Clean: the 12 rescored rows and the 6 final-configuration runs at `bb503f1`, and the 3 test runs at `a62fbd9`.
  - The Results tables use only clean runs, except the D / E check numbers, which come from dirty runs; their code
    is unchanged since.
- **Deviations** (in addition to stop-point 2's):
  - `rescore.py` copies latency, size and rounds from the original runs instead of measuring them again;
  - the `tmd.tracking` dirty rule changed (see 2);
  - `cv.py` `latency` takes its session pool and HMM parameters as arguments (needed for the test). Dev behaviour
    is unchanged.

## Conclusion

**Draft — not approved.** Written from Results and `results/runs.csv`.

**Answer.** An accelerometer-only LightGBM with forward-backward HMM smoothing reaches, on `waiis_bench_v1`:

| | family macro-F1 | 8-class macro-F1 |
|---|---|---|
| ExtraSensory test, 12 users (mean of 3 seeds) | **0.698** (0.693–0.704; seed-0 CI 0.631–0.746) | **0.666** (seed-0 CI 0.518–0.724) |
| — iPhone (4 families, 6 classes) / Android | 0.801 / 0.690 | 0.762 / 0.665 |
| ExtraSensory dev, pooled out-of-fold (3 seeds) | 0.615 (seed-0 CI 0.567–0.653) | 0.551 |
| SHL Users 2–3, rail check | 0.932 | 0.795 |

**This is the reference** every later model has to beat, on the same test users and protocol:
- **Model:** 580 features (`tmd.acc_features`: features v1 without gyroscope on the 15 s window + mean / std /
  min / max over 5 s sub-windows); LightGBM on ExtraSensory dev + SHL User 1, balanced class weights; HMM
  forward-backward per session over all usable minutes, with transitions from all labelled training minutes.
- **Cost:** 1.3–1.4 MB, 11–14 ms per single window, ~1 ms per window over whole sessions.

**What helped and what didn't** (ExtraSensory dev family F1, paired CI):
- **The HMM is by far the largest gain:** +0.120 [+0.096, +0.144] on dev, +0.18 on test.
  - Forward-backward ≈ Viterbi; forward-backward is kept for its per-minute posteriors (on-demand labelling).
- **Training with SHL User 1** helps ExtraSensory: removing it costs −0.031 [−0.043, −0.020]. It also gives the only
  metro data.
- **Sub-window statistics:** +0.013 [+0.008, +0.018].
- **Gravity / tilt features:** +0.002 (n.s.). **10 Hz low-pass:** −0.006 [−0.010, −0.002]; it lowers platform
  detectability only slightly (AUC 0.796 → 0.783).
- **Context over neighbouring minutes:** +0.060 alone, but its values identify the dataset (AUC 0.981 vs 0.947
  for the minute's own features), and after the HMM it adds nothing (−0.000 [−0.011, +0.012]). Not adopted.

**Trade-off: the HMM and ExtraSensory rail.**
- On dev the HMM cuts ExtraSensory rail recall from 0.24 to 0.08. ExtraSensory train minutes carry a weak rail
  signal (mean P(rail) 0.26), and the smoothing assigns their segments to car or still.
- In exchange it removes almost all false rail elsewhere (6,899 → 282 minutes) and lifts every other family.
- On the test users rail is not lost (0.39 → 0.40), and on SHL Users 2–3 it rises (0.73 → 0.85).
- Accepted for 05. Rail evidence on ExtraSensory is thin (3 dev users, 2 test users, Android only on test), and GPS
  fusion is expected to address rail.

**Weak spots for the next models.**
- Bus: test 0.22, dev 0.33; on dev it goes to walk and car.
- ExtraSensory train.
- Not-quiet still: 0.74 vs 0.996 quiet.
- Run: test CI from 0.
- Car vs bus on motorbike minutes flips with the seed.

**Limits.**
- One fixed set of LightGBM settings; early stopping stops at ~25–75 rounds.
- The SHL User 1 scores are a same-person day split.
- 11 of the 12 test users were dev users in experiment 02.
- Test train comes from 2 Android users.
- Test scores are higher than dev, for reasons not investigated.
- The real WAIIS gap pattern between fragments is unknown.
