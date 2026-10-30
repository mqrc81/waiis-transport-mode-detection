# 07 User label curve on waiis_bench_v1 (06's final configuration)

## Question

How does 06's final configuration improve with the number of labelled users? Where does the curve flatten, and which
classes are still gaining at the end? The answer decides whether more labelled people or a better model is the next
priority, and roughly how many WAIIS labellers per mode to plan for.

## Plan

*Approved 2026-10-30 with changes (stop-point 1); the changes are worked into this Plan.*

### Scope

- **Model:** 06's final configuration, settings unchanged:
  - acc model: 05's B (580 features, `tmd.acc_features`), LightGBM lr 0.1;
  - GPS model: gps-v2 (13 features, `tmd.gps_features`), LightGBM lr 0.02, trained on GPS-covered official minutes;
  - balanced class weights for both (`tmd.lgbm`), source weight 1 for SHL;
  - product fusion on GPS-covered minutes, acc model alone elsewhere (`pipe.product` of 06);
  - HMM forward-backward per session (`tmd.minutes`), transitions and initial distribution from the labelled minutes
    of the training users (before caps and drops).
- **Development data only.** The 12 ExtraSensory test users and SHL Users 2–3 are not read. No test stop-point.
- **Primary score:** pooled out-of-fold ExtraSensory dev 8-class macro-F1 over 7 classes (no subway); secondary:
  family macro-F1 over 5 families.
- **Two curves:**
  - **main:** SHL User 1 always in training (the public base); only the number of ExtraSensory users varies;
  - **no-SHL:** the same draws without SHL User 1 (both models and the HMM).

### Draw scheme

- Outer dev folds k = 0–4 (`experiments/04_frozen-benchmark_es-shl/splits/es_users_v2.csv`; 9, 10, 10, 10, 8 users).
- **Pool of fold k:** the ExtraSensory dev users of the other 4 folds: 38, 37, 37, 37, 39 users.
- **n ∈ {1, 2, 3, 5, 10, 20, all}**, and on the main curve **n = 0** (SHL User 1 only) as its first point.
- **Draw d (seeds 0–4):** one random ordering of each fold's pool (`numpy.random.default_rng([d, k]).permutation`
  of the sorted pool); n takes the first n users, so the training sets are nested within a draw.
  - The orderings are this experiment's split file, `splits/draws.json` (written by `draws.py`; it refuses to
    change once written). It gets an entry in `tracking.SPLITS`.
  - At **n = 0 and n = all**, one run each (draw 0, LightGBM seed 0).
- **SHL User 1 (main curve):** the SHL days of folds ≠ k, as in 06's paired folds (fold k's SHL days are not used,
  though SHL isn't scored here).
- Both models and the HMM transitions use the same n users (+ SHL User 1). The GPS model uses their GPS-covered
  official minutes.
- **LightGBM seed = draw seed** (bagging, feature fraction).

### Number of trees

- **Fixed at the final 06 models' rounds** (the median over 06's three test-model seeds):
  - acc model: **47** rounds at lr 0.1 (05's `test_B_seed{0,1,2}`: 47, 46, 48);
  - GPS model: **138** rounds at lr 0.02 (`test_G_v2_seed{0,1,2}`: 139, 136, 138).
- No early stopping, so no hidden labelled validation set.
- At n = all this is 06's H1_v2 with fixed rounds instead of per-fold early stopping. Its score is compared with
  H1_v2 seed 0 (0.613) as a sanity check.

**Early-stopping check (main curve, n = 2 and n = 10; reported, not adopted):**
- validation set: users n + 1 … n + 5 of the same ordering (5 extra users; the GPS model on their covered minutes);
- 06's early-stopping settings: patience 30 / at most 1,000 rounds (acc, lr 0.1), 150 / 10,000 (GPS, lr 0.02);
- the early-stopped models are scored directly (no refit: training data are the same n users);
- reported: rounds per model, and the paired difference per draw in pooled 8-class and family F1 against the
  fixed-rounds run with the same (n, draw), median and range over the 5 draws.

### Scoring

- For each (curve, n, draw), fold k's users are scored with that fold's models; the out-of-fold predictions are
  pooled over the 5 folds (as in 05/06). Per-fold scores only as spread.
- One MLflow run per (curve, n, draw): main 32, no-SHL 31, early-stopping check 10 = **73 runs**.
- **User-bootstrap 95% CIs** (`tmd.evaluate.bootstrap_users_recall`, redraws) **only at n = 0 and n = all**;
  elsewhere the spread over the 5 draws is the uncertainty (a deviation from CLAUDE.md, approved; see Results).
- Per run: 8-class and family macro-F1, before and after the HMM; per-class recall and F1; per-fold F1 min / max; iPhone and Android F1 (descriptive); covered / uncovered split;
  confusion matrices; model sizes; rounds; training minutes per class and iPhone share of the drawn users; classes
  absent from the training users.
- **Inference time** (single window and per window over 200 fold-0 sessions, as in 06): once per n, draw 0, fold-0
  models (main curve). Model size per run.

### Report

- **Per n:** 8-class (primary) and family F1, median and range over draws; with and without SHL.
- **Per n:** recall per class (all 7; still, walk, car, bus and train first), median and range.
- **Per n:** labelled minutes per class, platform mix, and the classes absent from the training users (Results,
  stop-point 1 has the dry run).
- **Gain per doubling** (fixed before running):
  - local gain between consecutive grid points: g_i = (F1(n_{i+1}) − F1(n_i)) / log2(n_{i+1} / n_i), computed per
    draw (the draws are nested, so the differences are paired), then the median over draws. For n = all,
    n = the fold's pool size, averaged over folds (37.6);
  - **the curve flattens at the first n_i whose median g_i < 0.01** (8-class). If no step is below 0.01, it has not
    flattened within the dev pool;
  - n = 0 has no log2 n: it is shown on the figure but is not part of the gain per doubling or the slope;
  - also the least-squares slope of F1 against log2 n over the grid points n ≥ 1, per draw, median and range;
  - **a class is still gaining at the end** if its median recall gain per doubling over the last step (20 → all) is
    ≥ 0.01.
- **One figure, two panels:** (left) 8-class F1 against n (log axis), with and without SHL User 1, median and range;
  (right) per-class recall against n, main curve.
- **Limits** (in the Conclusion): rail comes almost entirely from SHL User 1 (2–3 ExtraSensory train users in a fold's
  pool), so the rail curve can't be measured; ExtraSensory users are not WAIIS employees; a fold's pool is 37–39
  users, so the curve stops at ~38.

### Number of fits and run time

| part | runs | LightGBM fits |
|---|---|---|
| main curve: n = 0, 6 n × 5 draws, all; × 5 folds × 2 models | 32 | 320 |
| no-SHL curve: same | 31 | 310 |
| early-stopping check: 2 n × 5 draws × 5 folds × 2 models | 10 | 100 |
| **total** | **73** | **730** |

Estimate, from 06 / 05 fit times (acc ~12 s per fit on all dev data at ~50 rounds, GPS ~5 s; fits shrink with n):
- training: ~25 min per curve, ~10 min for the early-stopping check;
- evaluation (HMM over all dev minutes, user-bootstrap CIs) ~1 min per run, ~70 min; latency ~5 min;
- **total ~2–2.5 h**, in the background, with `tmd.progress` (units: model fits + one evaluation per run).

### Stop-points

1. ~~Plan~~: approved 2026-10-30 with changes.
2. Results, the figure, a draft Conclusion; update STATUS.md.

## Results

### Stop-point 1: dry run (2026-10-30)

```
uv run python draws.py        # splits/draws.json, results/coverage_dry_run.csv; logs/draws.log
```

No training. Training rows are official ExtraSensory dev minutes of the n drawn users (SHL User 1 is added on the main
curve: 23,076 official minutes over 5 day folds, all 8 classes). Each cell: median [min–max] over 5 draws × 5 folds
(n = all: 5 folds).

**Official minutes per class among the n users:**

| n | iPhone share | still | walk | run | bike | car | bus | train |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 [0–1] | 1,620 [167–3,819] | 300 [2–707] | 0 [0–283] | 0 [0–456] | 115 [0–1,011] | 0 [0–307] | 0 [0–97] |
| 2 | 0.50 [0–1] | 3,227 | 504 | 2 [0–283] | 2 [0–469] | 360 [0–1,421] | 35 [0–307] | 0 [0–97] |
| 3 | 0.67 [0.33–1] | 5,532 | 712 | 2 [0–296] | 99 [0–584] | 479 [157–1,463] | 43 [0–307] | 0 [0–97] |
| 5 | 0.60 [0.20–1] | 10,579 | 1,350 | 8 [1–297] | 120 [0–1,101] | 889 [233–1,687] | 97 [2–531] | 0 [0–97] |
| 10 | 0.60 [0.20–0.90] | 19,904 | 2,682 | 30 [5–386] | 516 [121–1,337] | 2,083 [661–3,053] | 241 [91–609] | 0 [0–166] |
| 20 | 0.60 [0.40–0.80] | 38,677 | 5,366 | 142 [22–433] | 1,297 [478–2,165] | 3,690 [2,707–4,776] | 606 [316–926] | 93 [0–190] |
| all | 0.59 [0.59–0.61] | 75,647 | 10,611 | 434 [175–444] | 2,518 [2,226–2,776] | 6,434 [6,221–6,623] | 1,071 [937–1,174] | 166 [93–190] |

**Classes absent from the n users** (number of the 25 draw × fold training sets; n = all: of 5):

| n | run | bike | car | bus | train |
|---|---|---|---|---|---|
| 1 | 13 | 16 | 5 | 14 | 23 |
| 2 | 5 | 9 | 2 | 7 | 21 |
| 3 | 5 | 4 | 0 | 5 | 20 |
| 5 | 0 | 1 | 0 | 0 | 18 |
| 10 | 0 | 0 | 0 | 0 | 14 |
| 20 | 0 | 0 | 0 | 0 | 2 |
| all | 0 | 0 | 0 | 0 | 0 |

Still and walk are never absent. Subway never occurs in ExtraSensory.

**Users with the class among the n**, median [min–max]: at n = 10, run 4 [1–7], bike 4 [1–6], car 8 [5–10], bus 4
[2–7], train 0 [0–2]; at n = all, run 13, bike 14, car 30, bus 17, **train 2–3**.
GPS-covered minutes (the GPS model's ExtraSensory rows) are 71–96% of these counts per class (at n = all: run 0.71,
still and walk 0.80, bus 0.85, car 0.90, bike 0.92, train 0.95; `results/coverage_dry_run.csv`).

**What this means for the curve.**
- Below n = 5, the minority classes are often missing (run, bike, bus in roughly a third to two thirds of the
  training sets at n ≤ 2). The low end of the curve mixes "no examples" with "few examples"; the absent-class counts
  are reported with each point.
- On the main curve, SHL User 1 supplies every class at every n, so the no-SHL curve at small n shows mostly the
  missing classes.
- ExtraSensory train is absent from most training sets up to n = 10 and comes from 2–3 users at n = all: the
  ExtraSensory rail curve can't be measured, as expected.

**Approved 2026-10-30, with changes** (worked into the Plan):
1. n = 0 (SHL User 1 only) is the main curve's first point.
2. Fixed trees: the median of 06's three seeds (acc 47, GPS 138).
3. The Plan, `splits/draws.json` and the code are committed before the first run, so all runs are clean.
4. User-bootstrap CIs only at n = 0 and n = all.

**Deviation from CLAUDE.md (approved 2026-10-30):** CLAUDE.md asks for user-bootstrap 95% CIs with every
cross-validation result. Here they are computed only at n = 0 and n = all; for n = 1 … 20 the spread over the 5 draws
(median and range) is the reported uncertainty. Code: `curve.py`, `CI_AT`.

**Code** (committed before the first run):
- `curve.py`: one MLflow run per (curve, n, draw); out-of-fold probabilities per run in
  `<data>/processed/07_user-label-curve_waiis-bench/oof/<run>.npz`.
- `src/tmd/fusion.py`: 06's frame loader, product fusion and latency, moved to shared code for this experiment
  (06's `pipe.py` keeps its own copy, unchanged).
- `tracking.SPLITS`: `extrasensory_dev_v2_draws07` → `splits/draws.json`.

**Smoke test** (`uv run python curve.py --smoke`; fold 0 only, no MLflow):
- main n = 0: fold-0 8-class 0.396; no-SHL n = 1, draw 0: 0.217; early-stopping check n = 2, draw 0: 0.514
  (acc 19 rounds, GPS 87).
- **LightGBM stops on its own below the fixed rounds when a training set is tiny:** no-SHL n = 1, draw 3, fold 0
  (one user, 167 still + 2 walk minutes) ends at 43 acc / 1 GPS rounds, because no split meets
  min_child_samples = 50. This is 06's settings applied to very little data, not a change; the rounds actually
  trained are logged per run (`acc_rounds_*`, `gps_rounds_*`).

### Stop-point 2: runs (in progress)

**Launch** (`d55d057`): `curve.py --curve main`, then `noshl`, then `escheck`, chained; output in
`logs/stage_curve.out`, details in `logs/curve_<curve>.log`.
- The main curve finished (32 runs, clean at `d55d057`).
- **The no-SHL curve stopped at its third run** (`noshl_n1_d2`): LightGBM refused an empty GPS training set.
  - Three dev users have no GPS-covered official minutes at all: 4E98F91F and 61359772 (Android), CDA3BBF7 (iPhone).
  - CDA3BBF7 is the only user of no-SHL, n = 1, draw 2, fold 0, so the GPS model had no rows. No other training set
    in the draws is affected (on the main curve SHL User 1 always provides GPS rows).
- **Deviation from the Plan (`curve.py`, `point()`):** when a fold has no GPS-covered training minutes there is no
  GPS model, and that fold uses the acc model alone. This is 06's fallback where there is no GPS, applied to a
  missing model. Logged per run as `folds_without_gps_model`.
- The code change only touches that case, so the runs before it (main, `noshl_n1_d0`, `noshl_n1_d1`) stand, at
  `d55d057`. The rest of the chain was resumed at the fix's commit; runs already in MLflow are skipped.
