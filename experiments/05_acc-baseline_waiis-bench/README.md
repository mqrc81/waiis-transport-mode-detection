# 05 Accelerometer-only baseline on waiis_bench_v1 (LightGBM + HMM)

## Question

How well does an accelerometer-only LightGBM + HMM baseline perform on `waiis_bench_v1`, and which pre-processing
steps improve it? The result is the reference every later model (InceptionTime + GPS, LIMU-BERT) has to beat.

## Plan

*Approved 2026-10-29 with changes (stop-point 1); the changes are worked into this Plan.*

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

### Final configuration (after stop-point 2)

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
2. Ablation results on development data and the proposed final configuration. Wait for OK before touching the test
   users.
3. Final configuration (3 seeds) scored once on the 12 test users and once on the SHL Users 2–3 rail check. Then
   the motorbike read-out, Results, a draft Conclusion, and an update of STATUS.md.

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
- Δ = paired user-bootstrap difference against the `vs` row (1,000 resamples).
- SHL is a same-person day split, with no CI.
- Trees = the mean over the 5 fold models (rounds × 8 classes).
- Single window = features + model for one 15 s window. Pipeline = whole sessions ÷ windows (200 fold-0
  ExtraSensory sessions).

| row | vs | ES family F1 [CI] | Δ family vs [paired CI] | ES 8-class F1 [CI] | SHL family | Δ SHL | SHL 8-class | SHL train / subway recall | trees (mean) | single window p50 / p95 ms | pipeline p50 / p95 ms | size MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_esonly | A | 0.452 [0.409–0.552] | -0.031 [-0.044, -0.021] | 0.366 [0.323–0.419] | 0.620 | -0.157 | 0.485 | 0.14 / 0.00 | 123 | 2.0 / 2.1 | 0.24 / 0.75 | 0.4 |
| A | – | 0.483 [0.439–0.591] | – | 0.411 [0.347–0.483] | 0.776 | – | 0.734 | 0.60 / 0.58 | 371 | 2.1 / 2.3 | 0.25 / 0.77 | 1.3 |
| **B** | A | 0.496 [0.451–0.607] | **+0.013 [+0.008, +0.019]** | 0.427 [0.358–0.503] | 0.790 | +0.014 | 0.755 | 0.66 / 0.60 | 365 | 10.8 / 11.2 | 1.06 / 3.83 | 1.3 |
| C | B | 0.498 [0.454–0.608] | +0.002 [-0.001, +0.005] | 0.430 [0.361–0.506] | 0.794 | +0.004 | 0.759 | 0.67 / 0.60 | 363 | 11.9 / 12.1 | 1.19 / 4.26 | 1.3 |
| D | B | 0.556 [0.511–0.677] | +0.060 [+0.049, +0.073] | 0.479 [0.406–0.555] | 0.928 | +0.138 | 0.870 | 0.80 / 0.70 | 384 | 11.0 / 12.0 | 1.12 / 4.01 | 1.4 |
| Dm (D′) | B | 0.560 [0.515–0.678] | +0.064 [+0.052, +0.079] | 0.485 [0.411–0.564] | 0.927 | +0.137 | 0.867 | 0.79 / 0.69 | 379 | 11.0 / 12.1 | 1.09 / 4.00 | 1.4 |
| E | B | 0.490 [0.444–0.600] | -0.006 [-0.010, -0.002] | 0.423 [0.354–0.498] | 0.779 | -0.011 | 0.752 | 0.66 / 0.60 | 352 | 11.3 / 12.0 | 1.10 / 3.99 | 1.3 |
| F_forward | B | 0.606 [0.563–0.736] | +0.111 [+0.092, +0.135] | 0.540 [0.459–0.627] | 0.923 | +0.133 | 0.900 | 0.85 / 0.80 | 365 | 10.8 / 13.2 | 1.09 / 3.94 | 1.3 |
| **F_fb** | B | **0.615 [0.567–0.745]** | **+0.120 [+0.096, +0.147]** | 0.552 [0.468–0.646] | 0.951 | +0.161 | 0.936 | 0.94 / 0.87 | 365 | 11.1 / 12.3 | 1.19 / 4.21 | 1.3 |
| F_viterbi | B | 0.612 [0.563–0.745] | +0.116 [+0.094, +0.143] | 0.552 [0.467–0.645] | 0.948 | +0.157 | 0.933 | 0.94 / 0.87 | 365 | 10.8 / 11.1 | 1.10 / 4.00 | 1.3 |
| X_Dm_fb | B | 0.615 [0.571–0.748] | +0.119 [+0.101, +0.145] | 0.551 [0.469–0.636] | 0.959 | +0.168 | 0.938 | 0.91 / 0.86 | 379 | 10.7 / 11.0 | 1.12 / 4.03 | 1.4 |
| X_Dm_viterbi | B | 0.614 [0.569–0.747] | +0.119 [+0.099, +0.145] | 0.553 [0.472–0.638] | 0.957 | +0.167 | 0.939 | 0.91 / 0.86 | 379 | 10.7 / 11.6 | 1.12 / 4.00 | 1.4 |

| row | ES family F1 per fold (min–max) | ES family F1 iPhone / Android | ES per-class recall: still walk run bike car bus train | ES rail recall |
|---|---|---|---|---|
| A_esonly | 0.406–0.601 | 0.474 / 0.412 | 0.74 0.65 0.24 0.66 0.51 0.37 0.05 | 0.02 |
| A | 0.433–0.639 | 0.503 / 0.446 | 0.75 0.69 0.25 0.67 0.52 0.22 0.18 | 0.29 |
| B | 0.440–0.662 | 0.518 / 0.456 | 0.77 0.70 0.25 0.68 0.54 0.28 0.16 | 0.24 |
| C | 0.448–0.660 | 0.518 / 0.460 | 0.77 0.70 0.25 0.68 0.55 0.27 0.20 | 0.28 |
| D | 0.504–0.732 | 0.575 / 0.517 | 0.84 0.74 0.27 0.74 0.61 0.32 0.09 | 0.13 |
| Dm (D′) | 0.507–0.730 | 0.576 / 0.523 | 0.84 0.74 0.28 0.75 0.61 0.32 0.18 | 0.26 |
| E | 0.439–0.655 | 0.510 / 0.453 | 0.77 0.69 0.25 0.68 0.54 0.27 0.19 | 0.28 |
| F_forward | 0.538–0.832 | 0.618 / 0.572 | 0.89 0.68 0.25 0.75 0.69 0.28 0.08 | 0.08 |
| F_fb | 0.545–0.851 | 0.623 / 0.584 | 0.88 0.75 0.26 0.76 0.73 0.34 0.06 | 0.08 |
| F_viterbi | 0.546–0.852 | 0.622 / 0.580 | 0.88 0.74 0.26 0.75 0.72 0.34 0.06 | 0.07 |
| X_Dm_fb | 0.559–0.829 | 0.625 / 0.587 | 0.89 0.74 0.29 0.78 0.71 0.35 0.08 | 0.08 |
| X_Dm_viterbi | 0.558–0.829 | 0.624 / 0.586 | 0.89 0.74 0.29 0.77 0.71 0.37 0.08 | 0.08 |

**Decisions under the rule.**
- **SHL User 1 (diagnostic):** without it, ExtraSensory family F1 is lower by −0.031 [−0.044, −0.021]. SHL helps,
  so there is no down-weighting (weight 1).
- **B adopted:** the CI excludes 0; SHL +0.014.
- **C not adopted:** the CI contains 0, and C is not cheaper.
- **D not adopted:** the check fails (below). D′ is not adopted either, for the same reason.
- **E not adopted:** −0.006, with a CI below 0; SHL −0.011.
- **F adopted:** all three decoders, with CIs far above 0.
  - Forward-backward vs Viterbi: +0.003 [−0.001, +0.007] family; 8-class −0.000 [−0.004, +0.003].
  - Forward-backward vs forward: +0.009 [−0.006, +0.022] family; SHL +0.028.

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
- Development: ExtraSensory family 0.615 [0.567–0.745], 8-class 0.552 [0.468–0.646]; SHL User 1 family 0.951,
  8-class 0.936 (same-person day split).
- Latency: single window 11.1 / 12.3 ms (p50 / p95), pipeline 1.19 / 4.21 ms per window. Model 1.3 MB.

**Final-configuration read-outs, seed 0 preview** (the 3-seed versions come with the final runs).
- **Still recall by `quiet`** (descriptive):
  - ExtraSensory: quiet 0.996 (n = 50,845), not quiet 0.736 (n = 41,885);
  - SHL User 1: quiet 0.888 (1,387), not quiet 0.699 (1,979).
- **ExtraSensory car / bus / train recall [user-bootstrap CI]:**
  - car 0.729 [0.627–0.811];
  - bus 0.340 [0.172–0.477];
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

**Observations and open questions for stop-point 2.**
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

## Conclusion

*Not yet.*
