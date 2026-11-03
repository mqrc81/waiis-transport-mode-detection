# 10 GPS + acc baseline on waiis_bench_v2

## Question

How does 06's configuration perform on `waiis_bench_v2`, and what do the new sources (NOR-TMD, Collecty, tram as a
ninth class) add without creating a dataset shortcut? The result becomes the reference model on v2 and the next
model for leg suggestions.

## Plan

*From the brief of 2026-11-03. Stop-point 1 approved 2026-11-03 with additions (Results, "Stop-point 1 answers").*

**Model.** 06's final configuration (acc LightGBM, gps-v2 GPS LightGBM, product fusion with acc fallback, HMM
forward-backward) and 06's settings, including inner early stopping. Only the training data and the class set change.
- GPS model: unchanged sources (ExtraSensory + SHL User 1, the only ones with GPS). It has no tram. Fusion rule for
  classes the GPS model lacks: stop-point 1.
- HMM transitions from all labelled training minutes of the sources in each row.

**Ablation** (seed 0; one MLflow run per row; each data row builds on the best so far)
- **A0.** 06 retrained on v1's sources (ExtraSensory + SHL User 1) with v2's labels, 8 classes. The reference on v2.
  Its NOR-TMD and Collecty dev scores are transfer scores: those sources aren't in training.
- **A1.** + NOR-TMD (9 classes, tram included).
- **A2.** + Collecty.
- **S.** SHL User 1 weight 0.5 instead of 1.
- **P.** + SHL User 1's other positions (opt-in rows, one random position per minute).

**Rules** (fixed before running)
- Data rows (A1, A2): adopt if ExtraSensory dev six-class doesn't drop by more than 0.01, ExtraSensory still and walk
  precision don't drop by more than 0.01, and NOR-TMD dev (bus, car, rail) improves by ≥ 0.01. For A2, NOR-TMD dev
  must not drop by more than 0.01 instead.
- Other rows (S, P): adopt if ExtraSensory dev six-class gains ≥ 0.01 and NOR-TMD dev (bus, car, rail) doesn't drop
  by more than 0.01.
- Every difference with its paired user-bootstrap CI.

**Reported for every row**
- Six-class, 9-class and family on ExtraSensory dev; NOR-TMD dev, Collecty dev and SHL day folds separately (per
  class, with the class list).
- Dataset shortcut:
  - (a) per-class precision on ExtraSensory dev, A0 vs later rows, especially still and walk against vehicle classes;
  - (b) NOR-TMD transfer: A0's NOR-TMD dev score next to A1's out-of-fold score;
  - (c) per-source recall per class;
  - (d) source identification within class from the model's output probabilities (AUC), next to 09's feature-level
    baseline.
- Tram predicted on ExtraSensory and SHL minutes (count and share; there is no tram there).
- Recall with CIs for bus, car, train, subway and tram.
- Inference time and model sizes.

**Stop-points**
1. Plan: the fusion rule for classes the GPS model lacks, the exact rows, run time and free disk space. **Wait for
   OK.**
2. Ablation on development data and the proposed final configuration. **Wait for OK before touching any test user.**
3. Write and commit the test protocol. Then score the final configuration and A0 (3 seeds each) once on:
   - ExtraSensory test (overall and per platform);
   - NOR-TMD test (bus, subway, train, tram, car);
   - the SHL Users 2–3 rail check per position.

   Then Results, draft Conclusion, update STATUS.md.

## Results

### Stop-point 1: plan details (2026-11-03)

#### Fusion rule for tram (the class the GPS model lacks)

**Proposal: a vehicle-neutral GPS factor.** On GPS-covered minutes the GPS model's tram factor is the mean of its
four vehicle probabilities:

  p_gps(tram) := mean(p_gps(car), p_gps(bus), p_gps(train), p_gps(subway)),

then p ∝ p_acc · p_gps over the 9 classes, renormalised (06's product, 1e-9 floor). Uncovered minutes keep p_acc, as
in 06.
- GPS then says whether the minute looks like a vehicle, and nothing about which vehicle is tram. The acc model alone
  decides tram against the other vehicles.
- **Why not a uniform factor (1/8):** the GPS model is trained with balanced class weights, so its outputs act as
  likelihoods. A constant 1/8 would *raise* tram relative to the vehicles whenever GPS is sure the minute is still or
  walk (vehicle probabilities ≈ 0.01), which creates false tram on exactly the ExtraSensory still / walk minutes the
  rules protect.
- **Why not a bus proxy:** tram moves like a bus in GPS, but copying bus ties tram to one class's errors; the mean
  over the vehicles is the assumption-free version of "a vehicle".
- **Limit, unavoidable on this benchmark:** tram exists only in NOR-TMD and Collecty, which have no GPS. On dev the
  rule therefore only acts where tram is never true (ExtraSensory, SHL): it can be checked for false tram (the
  tram-on-ExtraSensory/SHL count), never for tram recall under GPS. That waits for WAIIS data.
- **Sensitivity (descriptive, not a choice):** for the final row, false tram on ExtraSensory / SHL and six-class
  ExtraSensory dev are also reported with the uniform factor and the bus proxy, recombined from the saved
  probabilities (no retraining).

A0 is 8-class (no tram anywhere in its training data); tram gets probability 0 when it is scored 9-class.

#### The rows exactly

Common to all rows: v2's labels (`cls_fine`, 9 classes; the run flag applied), 05's 580 acc features recomputed
from v2, 06's gps-v2 features, lr 0.1 with inner early stopping as in 06 (`tmd.models.lgbm.paired_cv`: rounds from
early stopping on fold (k + 1) mod 5, refit on the 4 training folds), balanced class weights over the source-weighted
rows, seed 0. Dev folds: ExtraSensory by user, SHL User 1 by day, NOR-TMD and Collecty by user (09's fold files). The
early-stopping fold contains every training source of the row.

| Row | acc model trained on | GPS model trained on | Classes | HMM transitions from | Builds on |
|---|---|---|---|---|---|
| A0 | ES + SHL U1 | ES + SHL U1 (covered minutes) | 8 | labelled minutes of ES + SHL U1 | – |
| A1 | + NOR-TMD | unchanged (= A0's) | 9 | + NOR-TMD | A0 |
| A2 | + Collecty | unchanged | 9 | + Collecty | best of A0 / A1 |
| S | best, SHL U1 weight 0.5 | SHL U1 weight 0.5 (retrained) | as best | as best (counts, not weighted) | best so far |
| P | best, SHL U1 minutes of the 3 four-phone days use the position in `position_draw` | unchanged (GPS is per minute) | as best | as best | best so far |

- **NOR-TMD and Collecty dev in A0** (transfer): each outer fold k's model scores the NOR-TMD / Collecty dev users of
  fold k, so every dev user gets a score from a model that never saw that source.
- **P:** the four-phone days have 5,964 other-position windows. Per minute the frozen `position_draw` picks one of
  hips / hand / torso / bag, so SHL's row count and weight stay unchanged. SHL day folds are still *scored* on hips,
  as in every other row.
- **Not a row, descriptive:** 06 as-is (v1 models' saved out-of-fold probabilities), rescored on v2's labels, as 09
  asked. It shows how much of A0 − 06 is the label change and how much retraining.

#### Scores

- **Primary:** six-class macro-F1 (still, foot, bike, car, bus, rail) on ExtraSensory dev, pooled out of fold, with
  user-bootstrap 95% CIs (`tmd.evaluate`). ExtraSensory dev has no subway or tram, so the six-class rail is train.
- **NOR-TMD dev guard (bus, car, rail):** NOR-TMD dev minutes labelled bus, car, train, subway or tram, six-class
  predictions (any of the 6), macro-F1 averaged over bus, car and rail. Bike is reported separately, not in the guard.
- **Precision rule (data rows):** ExtraSensory dev six-class precision of still and of foot (walk + run) — the brief
  says "walk"; foot is what the six-class score has, and walk precision (9-class) is reported next to it.
- Paired user-bootstrap CI for every difference: ExtraSensory over its users, NOR-TMD over its installations.
- SHL User 1 (day folds) and Collecty (14 users, ≤ 5 per class except car): no CI.
- (d) source identification: 09's protocol (`checks.py`, `_group_auc`: StratifiedGroupKFold by user / SHL day,
  LightGBM, ≤ 2,000 windows per source, classes bus, car, train, subway, tram), with the model's out-of-fold
  log-probabilities as input instead of the 580 features. Two inputs:
  - the **acc model's raw probabilities** (comparable with 09's feature-level baseline);
  - the **final output** (fused, after the HMM). GPS coverage alone separates ExtraSensory / SHL from the sources
    without GPS, so this AUC is reported with that caveat.
  The function moves to `src/tmd/evaluate.py` (second user).

#### Shared code (src/tmd), v1 kept bit-identical

- `tmd.models.lgbm`, `tmd.models.minutes`: number of classes as a parameter (default 8); fine labels (`y_fine`) for
  the HMM.
- `tmd.models.fusion`: the product with a missing GPS class (rule above), and a v2 frame loader for all four sources.
- `tmd.evaluate`: six-class / 9-class scoring helpers, the source-identification AUC.
- **Port check before the ablation:** 06's H1_v2 seed 0 rerun through the changed code on v1 must give bit-identical
  out-of-fold probabilities, and 05's acc features recomputed from v2 must equal 05's tables for ExtraSensory and
  SHL User 1 (09 showed the sign conversion leaves them unchanged).

#### Run time (estimate, from 05 / 06's logs)

| Step | Estimate |
|---|---|
| acc features from v2 (ES, SHL U1 + other positions, NOR-TMD, Collecty; dev and test parts) | 10 min |
| gps-v2 features (ES, SHL; check against 06's tables) | 2 min |
| port check | 10 min |
| per row: acc CV (10 fits) + GPS CV (only A0, S) + HMM + evaluation and bootstraps + source AUC | 12–18 min |
| ablation, 5 rows + 06 rescored + tram sensitivity | 1.5–2 h |
| final configuration and A0, seeds 1–2 on dev (the test models' rounds come from them, as in 06) | 1 h |
| test (3 seeds × 2 configurations; ES test, NOR-TMD test, SHL Users 2–3 per position) and latency | 45 min |

All long jobs run in the background with `tmd.progress` and logs under `logs/`.

#### Disk

22 GB free on the data volume (460 GB, 96% used). New files: acc feature tables ~1.5–2 GB (05's took 2.8 GB with
its low-pass variants, which 10 doesn't need), out-of-fold probabilities and models < 0.5 GB. Enough, but the volume
is nearly full.

#### Open questions for OK

1. The tram rule above (vehicle mean), with the uniform factor and bus proxy as descriptive sensitivity only.
2. S halves SHL User 1's weight in **both** models (06 trained both with SHL User 1); the HMM's counts stay
   unweighted.
3. The precision rule uses six-class foot for "walk" (walk 9-class reported alongside).
4. Seeds 1–2 on dev for the final configuration and A0 after stop-point 2 (needed for the test models' rounds).

### Stop-point 1 answers (2026-11-03, approved)

1. **Tram rule: the vehicle mean, approved.** The uniform factor and the bus proxy are descriptive only and **can't
   replace the rule at stop-point 2**, whatever they score.
2. **S** halves SHL User 1's weight in both models; the HMM counts stay unweighted.
3. **Precision rule** on six-class **foot** precision (walk + run); 9-class walk precision reported next to it,
   descriptive only.
4. **Seeds 1–2 on dev** for the final configuration and A0: yes.
5. **Unseen positions (every row, descriptive):** SHL User 1's held-out days among the 3 four-phone days, scored on
   their hand, torso and bag windows, per position (out of fold: each day by the model of its day fold). A0 gives the
   baseline for unseen positions. P's rule is unchanged. In P the other positions of the training days are in
   training, so P's numbers there are positions seen on other days, not unseen positions.
6. **Mode changes (every row):** per source, the count of labelled transitions (consecutive labelled minutes of
   different classes within a session, the HMM's own counting), and the diagonal of the HMM transition matrix.
7. **06's seed spread on dev** (six-class, ExtraSensory, from its existing runs) as context for the 0.01 thresholds.
8. **(d) on the final output** also on minutes without GPS only.
9. **Training weight per class (every row):** each source's share of the training weight per class (source weight ×
   balanced class weight, summed over the training rows of the 5 outer folds).

### Setup and port check (2026-11-03)

**Shared code changed** (default 8 classes, so v1 results are unchanged; checked below):
- `src/tmd/models/hmm.py`: the number of states from the transition matrix (`transition_matrix(..., n=8)`).
- `src/tmd/models/lgbm.py`: `class_weights` / `dataset` take `n`; `paired_cv` reads `num_class` from the params.
- `src/tmd/models/minutes.py`: `labelled_minutes(..., fine=True)` (9 classes from `cls_fine`; excluded foot labels
  and flagged run users' minutes left out, as v2 makes them unlabelled); `hmm_params(lab, n)`; `smooth_cv` takes the
  class count from the probabilities.
- `src/tmd/labels.py`: tram → rail family.
- `src/tmd/evaluate.py`: levels `fine` and `six`; `six_pred`; `user_draws` (the draws of `user_resamples`, which now
  uses it); `cm_metrics`, `user_cms`, `bootstrap_cms` (user bootstrap on per-user confusion matrices: same draws and
  redraw rule, checked equal to the `scores` path on random data); `identification_auc` (copied from 09's
  `checks.py`, which is left as it is).
- `src/tmd/models/fusion.py`: `gps_fine` (the tram rule) and `product(..., tram_rule)`.

**Feature tables** (`make_features.py`; 70 s; `<data>/processed/10_gps-acc-baseline_waiis-bench-v2/`, 1.1 GB):

| Source | Usable dev minutes | Official | GPS-covered |
|---|---|---|---|
| ExtraSensory | 291,155 | 118,678 | 0.810 |
| SHL User 1 (hips) | 36,806 | 23,076 | 0.696 |
| NOR-TMD | 12,968 | 10,957 | – |
| Collecty | 13,424 | 3,383 | – |
| SHL User 1 other positions (hand / torso / bag, 3 days) | 5,960 | 2,979 | 0.593 |

- Dev = users in a fold (`fold >= 0`). 1,067 NOR-TMD and 2 Collecty usable minutes of non-reserved users have no fold
  in 09's fold file and are left out (no official examples).
- 4 implausible other-position windows are dropped (U3).

**Port check** (`port_check.py`, `logs/port_check.log`): all bit-identical.
- 10's acc and gps-v2 features from v2 = 05's / 06's tables from v1 on all 291,155 ExtraSensory and 36,806 SHL
  rows (max abs diff 0).
- Through the changed shared code: 05's B seed 0 (before and after the HMM) and 06's H1_v2 seed 0 (fused before and
  after the HMM, GPS model retrained) reproduce the saved out-of-fold probabilities exactly.

**Definitions used in scoring** (fixed before the first row):
- Six-class prediction = the 9-class argmax merged (09's rescoring of the logged confusion matrices, so v1's
  reference numbers compare). Family = argmax of the summed probabilities (06's family).
- 9-class macro-F1 over the classes present in the evaluation set (ExtraSensory dev: no subway, no tram).
- The identification check (d) reruns 09's sampling with 09's seed, so it uses 09's windows (bus 5,138, car 5,230,
  train 2,912, subway 3,312, tram 1,186; checked equal) and puts the out-of-fold log-probabilities where 09 had the
  580 features. With random probabilities it gives AUC ≈ 0.5.
- Code: `make_features.py`, `pipe.py`, `run.py` (one row = one MLflow run), `rescore06.py` (R06).

**Found while testing (item 6, labelled minutes of all dev users):**

| Source | Labelled 1-min steps | Mode changes |
|---|---|---|
| ExtraSensory | 228,560 | 3,157 |
| SHL User 1 | 22,283 | 173 |
| NOR-TMD | 22,637 | **0** |
| Collecty | 9,546 | 6 |

- NOR-TMD has no labelled mode change: each session holds one labelled mode, and the minutes between modes are
  excluded foot labels or gaps. Adding NOR-TMD raises the HMM's self-transition counts and adds none between
  vehicles.

### Ablation (in progress, seed 0)

*Running notes; the stop-point 2 tables come from MLflow once every row has run.*

Commands (logs in `logs/`):
```
uv run python run.py --row A0 --sources extrasensory shl_user1 --n 8
uv run python run.py --row A1 --sources extrasensory shl_user1 nor_tmd --gps-from A0 --vs A0
uv run python rescore06.py
uv run python run.py --row A2 --sources extrasensory shl_user1 collecty --gps-from A0 --vs A0
```
- A0's first run crashed after its MLflow run had finished (a `json.dumps` of an int64 in the log line after it;
  `run.py:82`, fixed with `default=float`). Its MLflow run and out-of-fold file are complete, so it wasn't rerun.
- `rescore06.py` failed once on the duplicated (source, user, ts) keys of the other-position rows (fixed as in
  `pipe.identification`) and was rerun.

**A0** (ExtraSensory + SHL User 1, 8 classes; the reference): ExtraSensory dev six-class 0.647, 9-class 0.672,
family 0.687; NOR-TMD guard (transfer) 0.324 (bus recall 0.37, subway 0.04, tram 0); Collecty six-class 0.639;
SHL User 1 0.949; unseen positions six-class hand 0.811, torso 0.923, bag 0.949. No tram predicted (8 classes).

**A1 (+ NOR-TMD): not adopted** under the rule.

| Rule (vs A0) | Difference [paired user-bootstrap 95% CI] | Limit | Pass |
|---|---|---|---|
| ExtraSensory dev six-class | **−0.020** [−0.037, +0.002] (0.647 → 0.627) | ≥ −0.01 | **no** |
| still precision (six-class) | +0.000 [−0.000, +0.001] | ≥ −0.01 | yes |
| foot precision (six-class) | −0.009 [−0.015, −0.004] | ≥ −0.01 | yes |
| (walk precision, 9-class; descriptive) | −0.009 [−0.014, −0.004] | – | – |
| NOR-TMD dev guard (bus, car, rail) | **+0.349** [+0.270, +0.435] (0.324 → 0.673) | ≥ +0.01 | yes |

- **Where the ExtraSensory drop comes from: false rail on still minutes.** Still → rail goes from 16 to 317 minutes
  (subway 245, tram 54, train 18), so rail F1 falls from 0.278 to 0.191, about −0.015 of the −0.020. Bus F1 0.465
  → 0.439 is most of the rest. ExtraSensory dev rail is 190 minutes from 3 users, so a few hundred false minutes
  weigh heavily.
  - 88% of the false-rail minutes have no GPS (so the GPS model can't overrule them), 40% are `quiet` (still phone),
    247 of 317 are Android, 142 come from one user (14 users in all).
  - Reading: from NOR-TMD's metro the acc model learns "very smooth with a little vibration → subway"; without GPS
    nothing contradicts it. This is the dataset shortcut item (a) is about, in rail precision rather than still
    precision (still has 92,700 minutes, so its precision barely moves).
- NOR-TMD per class (A0 → A1): bus 0.37 → 0.87, subway 0.04 → 0.77, tram 0 → 0.35, train 0.24 → 0.29, car 0.64 →
  0.45.
- Tram predicted where there is none: ExtraSensory 0.04% of usable minutes, SHL User 1 hips 0, other positions 0.4%.
- SHL User 1 0.949 → 0.955; unseen positions: hand +0.056, torso +0.028, bag +0.009.
- **Source identification (d), acc-model probabilities vs 09's features:** bus 0.68 / 0.82, car 0.65 / 0.82, train
  0.62 / 0.84, subway 0.81 / 0.96, tram 0.33 / 0.52 (A0: 0.67, 0.62, 0.64, 0.87, 0.27). AUCs below 0.5 (tram) come
  from group cross-validation with 4–17 groups per source (anti-correlated folds), not from information; read them
  as ≈ 0.5.
- **Next under the Plan:** A2 builds on the best so far, A0: ExtraSensory + SHL User 1 + Collecty (9 classes,
  Collecty has tram); A2's rule compares with A0.

**R06 (06 as-is, rescored on v2's labels; `results/tables/rescore06.csv`):** ExtraSensory dev six-class 0.643 /
0.641 / 0.640 (seeds 0–2; range 0.003; v1 labels 0.643 / 0.642 / 0.640). The run flag barely changes 06's score.
**06's seed spread (0.003) is a third of the 0.01 thresholds.** A0 − R06 (seed 0) = +0.004: retraining on v2's
labels changes little.

**A2 (A0 + Collecty): adopted** under the rule, narrowly.

| Rule (vs A0) | Difference [95% CI] | Limit | Pass |
|---|---|---|---|
| ExtraSensory dev six-class | −0.008 [−0.024, +0.009] (0.647 → 0.639) | ≥ −0.01 | yes |
| still precision | +0.001 [−0.000, +0.001] | ≥ −0.01 | yes |
| foot precision | −0.0096 [−0.013, −0.006] | ≥ −0.01 | yes (0.0004 inside) |
| (walk precision, 9-class) | −0.009 [−0.013, −0.006] | – | – |
| NOR-TMD dev guard | +0.032 [+0.006, +0.062] (0.324 → 0.356) | ≥ −0.01 | yes |

- Collecty dev six-class 0.639 → 0.672 (its own users, now out of fold); Collecty tram recall 0 (153 minutes,
  4 users); 9-class 0.672 → 0.677; family 0.687 → 0.687.
- ExtraSensory rail precision 0.271 → 0.219 (rail recall 0.28 → 0.37): false rail again, smaller than in A1.
- No tram predicted on ExtraSensory or SHL.
- SHL User 1 0.949 → 0.941; unseen positions: hand +0.028, torso +0.031, bag +0.013.
- **Foot precision falls by about 0.01 in both A1 and A2**, two different added sources.

**S (A2 with SHL User 1 weight 0.5, both models): not adopted.**
`uv run python run.py --row S --sources extrasensory shl_user1 collecty --shl-weight 0.5 --vs A0 A2`

| Rule (vs A2) | Difference [95% CI] | Limit | Pass |
|---|---|---|---|
| ExtraSensory dev six-class | −0.004 [−0.012, +0.009] (0.639 → 0.635) | ≥ +0.01 | **no** |
| NOR-TMD dev guard | +0.009 [−0.003, +0.021] | ≥ −0.01 | yes |

- Descriptive: still precision +0.003 [+0.002, +0.004], foot precision −0.010 [−0.017, −0.005], rail precision
  0.219 → 0.189; family −0.007; SHL User 1 0.941 → 0.918; unseen positions: hand +0.005, torso −0.010, bag −0.027.

**P (A2 + SHL User 1's drawn positions): not adopted.**
`uv run python run.py --row P --sources extrasensory shl_user1 collecty --positions --gps-from A0 --vs A0 A2`
- vs A2: ExtraSensory dev six-class −0.001 [−0.008, +0.010] (needs ≥ +0.01); NOR-TMD guard +0.003 [−0.008, +0.014].

### Stop-point 2: ablation on development data (2026-11-03)

Full tables: `results/tables/ablation.md` (`uv run python tables.py`, from MLflow; seed 0). Tram sensitivity:
`uv run python tram_sensitivity.py --row A2 --sources extrasensory shl_user1 collecty` and the same for A1
(descriptive; MLflow runs `<row>_tram-<rule>_seed0`). All runs are tagged dirty (uncommitted experiment code).

**Scores** (paired user-bootstrap 95% CIs; ExtraSensory over users, NOR-TMD over installations)

| Row | ES dev six-class [CI] | vs A0 | ES 9-class | ES family | NOR-TMD guard [CI] | vs A0 | Collecty six | SHL U1 six | Rule |
|---|---|---|---|---|---|---|---|---|---|
| R06 (06 as-is) | 0.643 [0.582, 0.687] | – | 0.675 | 0.682 | – | – | – | 0.957 | – |
| **A0** | **0.647** [0.582, 0.687] | – | 0.672 | 0.687 | 0.324 [0.255, 0.396] | – | 0.639 | 0.949 | reference |
| A1 (+ NOR-TMD) | 0.627 [0.571, 0.670] | −0.020 [−0.037, +0.002] | 0.681 | 0.669 | **0.673** [0.578, 0.769] | +0.349 [+0.270, +0.435] | 0.627 | 0.955 | not adopted (ES −0.020) |
| **A2** (A0 + Collecty) | 0.639 [0.579, 0.677] | −0.008 [−0.024, +0.008] | 0.677 | 0.687 | 0.356 [0.286, 0.425] | +0.032 [+0.006, +0.062] | 0.672 | 0.941 | **adopted** |
| S (A2, SHL 0.5) | 0.635 [0.577, 0.677] | −0.011 [−0.028, +0.008] | 0.677 | 0.680 | 0.365 [0.291, 0.438] | +0.041 [+0.014, +0.073] | 0.669 | 0.918 | not adopted (vs A2 −0.004) |
| P (A2 + positions) | 0.638 [0.579, 0.678] | −0.008 [−0.024, +0.008] | 0.676 | 0.681 | 0.359 [0.289, 0.430] | +0.035 [+0.013, +0.062] | 0.643 | 0.944 | not adopted (vs A2 −0.001) |

**(a) ExtraSensory dev six-class precision** (difference vs A0 [paired CI])

| Row | still | foot | bike | car | bus | rail |
|---|---|---|---|---|---|---|
| A0 | 0.942 | 0.592 | 0.917 | 0.737 | 0.516 | 0.271 |
| A1 | +0.000 [−0.000, +0.001] | −0.009 [−0.015, −0.004] | −0.003 | +0.015 | **−0.086** [−0.153, −0.022] | **−0.144** [−0.319, +0.000] |
| A2 | +0.001 [−0.000, +0.001] | −0.010 [−0.013, −0.006] | −0.004 | +0.011 | **−0.077** [−0.147, −0.019] | −0.052 [−0.183, +0.059] |
| S | +0.003 [+0.002, +0.005] | −0.020 [−0.026, −0.014] | −0.009 | +0.004 | −0.100 [−0.168, −0.032] | −0.082 |
| P | +0.000 | −0.011 [−0.016, −0.006] | −0.002 | +0.007 | −0.063 [−0.124, −0.015] | −0.066 |

- Still precision never moves (92,700 still minutes); walk (9-class) precision 0.586 → 0.577–0.578 in A1 / A2.
- **Every added source costs bus precision (−0.06 to −0.10, CIs exclude 0) and foot precision (−0.01)**; rail
  precision falls too (A1: false subway on still minutes, see A1 above).

**(b) NOR-TMD transfer:** A0 (never saw NOR-TMD) 0.324 vs A1 out of fold 0.673. Per class (9-class recall, A0 →
A1): bus 0.37 → 0.87, subway 0.04 → 0.77, tram 0.00 → 0.35, train 0.24 → 0.29, car 0.64 → 0.45, bike 0.88 →
0.88. **Without NOR-TMD in training (A0, A2, S, P), NOR-TMD subway and tram stay at ≤ 0.07 and bus at ~0.35**:
SHL's London subway and ExtraSensory's buses don't transfer to Oslo.

**(c) Recall per source** (9-class; `ablation.md` has every row): within the training sources, SHL User 1 is 0.77–1.00
everywhere; ExtraSensory bus 0.41–0.46, train 0.22–0.33. Collecty tram is 0.00 in every row (A1: 0.07), so tram
learnt from Collecty's 4 users (A2) doesn't recognise Collecty's own held-out tram users.

**(d) Source identification within class** (AUC on 09's windows; acc-model probabilities / final output / final on
minutes without GPS)

| Class | 09 features | A0 | A1 | A2 |
|---|---|---|---|---|
| bus | 0.824 | 0.67 / 0.85 / 0.79 | 0.68 / 0.80 / 0.66 | 0.68 / 0.86 / 0.80 |
| car | 0.818 | 0.62 / 0.70 / 0.68 | 0.65 / 0.69 / 0.66 | 0.64 / 0.74 / 0.63 |
| train | 0.835 | 0.64 / 0.76 / – | 0.62 / 0.76 / – | 0.63 / 0.75 / – |
| subway | 0.963 | 0.87 / 0.98 / 0.98 | 0.81 / 0.92 / 0.93 | 0.87 / 0.99 / 0.99 |
| tram | 0.521 | 0.27 / 0.58 / 0.58 | 0.33 / 0.67 / 0.67 | 0.39 / 0.67 / 0.67 |

- The acc model's probabilities identify the source less than its features do (bus 0.68 vs 0.82, subway 0.81–0.87
  vs 0.96), and adding NOR-TMD to training (A1) **lowers** identification of bus and subway from the final output.
  The output carries less source information when the source is in training, not more.
- The final output's AUC is higher than the acc model's: GPS coverage and the HMM differ by source (ExtraSensory /
  SHL have GPS, the new sources don't). On minutes without GPS it is lower for A1 (bus 0.66) than A0 (0.79).
- "Train, no GPS": too few ExtraSensory groups without GPS for the cross-validation (NaN). AUCs below 0.5 (tram,
  acc) come from group cross-validation with 4–17 groups per source; read them as ≈ 0.5.

**Tram predicted where there is none:** A1 130 ExtraSensory minutes (0.04% of usable) and 24 SHL other-position
minutes (0.4%); A0, A2, S, P none.
- Tram rule sensitivity (descriptive; can't replace the rule): A1 uniform 149, bus proxy 151 ExtraSensory minutes
  (bus proxy also 18 on SHL), against 130 with the vehicle mean; ExtraSensory six-class −0.000 / −0.002, NOR guard
  unchanged. A2: no difference (it predicts almost no tram).

**Recall with CIs** (9-class): ExtraSensory dev bus 0.41–0.46 [CIs ~0.30–0.59], car 0.68–0.71, train 0.22 (A0) →
0.31–0.33 (later rows) with CIs from 0.00 (3 users); NOR-TMD bus, car, train, subway, tram in (b). All CIs in
`ablation.md`.

**Unseen positions** (SHL User 1 held-out days, six-class): A0 hand 0.811, torso 0.923, bag 0.949. Every later
row is higher on hand (+0.02 to +0.06; A1 0.867) and torso (+0.02 to +0.03). P, which trains on the other positions
of other days, isn't better than A2 there (hand 0.829 vs 0.839).

**HMM:** NOR-TMD adds 22,637 labelled steps and **no** mode change; Collecty 9,546 and 6. Diagonal (A0 → A1): bus
0.977 → 0.990, subway 0.991 → 0.996; walk 0.920 unchanged. A1's HMM keeps bus and subway longer once entered.

**Training weight per class** (acc model; source shares; `ablation.md`): in A1, NOR-TMD carries 55% of bus, 58% of
subway, 12% of train, 100% of tram; in A2, Collecty carries 18% of car, 7% of train, 100% of tram. SHL User 1 still
carries 83–88% of train in every row (S: 79%).

**Cost:** 11.6–12.6 ms for one window (p95 ≤ 13.1), 1.3 ms per window over whole sessions (p95 ~4.5); acc model
0.8–1.5 MB, GPS model 3.8–4.1 MB.

#### Found at stop-point 2: tram drives the inner early stopping

Rounds of the acc model per outer fold (early stopping on fold (k + 1) mod 5):

| Row | fold 0 | 1 | 2 | 3 | 4 | Tram in the early-stopping fold |
|---|---|---|---|---|---|---|
| A0 | 59 | 40 | 39 | 55 | 67 | none (8 classes) |
| A1 | 33 | 46 | 42 | 26 | 36 | NOR-TMD tram in every fold |
| A2 | 20 | 19 | **45** | 25 | 18 | Collecty tram in folds 1, 2, 4, 0 (60, 8, 60, 25 minutes); **none in fold 3** (outer fold 2) |

- With balanced class weights, tram (153 Collecty minutes, 4 users, in A2) weighs as much in the early-stopping loss
  as still. Its validation loss turns up early, so the acc model stops at ~20 rounds instead of ~50 wherever the
  early-stopping fold holds tram. The one A2 fold without tram stops at 45.
- So **A1 / A2 / S / P vs A0 mixes two changes: the added data and a model trained half as long.** The rules were
  applied as fixed, but A1's −0.020 and A2's −0.008 can't be attributed to the data alone.
- This is 06's early stopping used as specified; 06 never had a class this small.

#### Proposed final configuration

**Under the rules as fixed: A2** (ExtraSensory + SHL User 1 + Collecty, 9 classes, tram rule = vehicle mean, SHL
weight 1, hips only; GPS model = A0's; HMM from ExtraSensory + SHL User 1 + Collecty).

**Recommendation: don't fix it yet.** The rule outcome rests on a comparison confounded by early stopping, and A2
differs from A0 by noise-sized amounts (ES −0.008, CI across 0; NOR-TMD +0.032) while costing bus precision. Before
the test, I propose one pre-registered follow-up on dev (seed 0, rules unchanged, ~30 min):
- **E1 / E2 = A1 / A2 with the acc model's rounds set by early stopping on the six-class-merged classes' loss**,
  or, simpler, **with A0's rounds per fold**. Either takes tram out of the stopping decision; the second needs no
  new code beyond passing the rounds.
- Then reapply the data-row rules to E1 / E2 (vs A0) and choose between A0, A2 and E1 / E2.

**Open questions**
1. Accept A2 as final now (the rules as fixed), or run E1 / E2 first? If E1 / E2: rounds from A0 per fold, or
   early stopping on a loss without tram?
2. A1's false subway on still minutes without GPS is the dataset shortcut item (a) is about. If E1 still fails on
   it, is NOR-TMD a follow-up experiment (e.g. a lower NOR-TMD weight, or NOR-TMD rail only where GPS exists),
   rather than part of 10?
3. Collecty's tram doesn't transfer even to Collecty's own held-out users (recall 0.00). Keep tram as a ninth class in
   the final model if NOR-TMD isn't in it?

### Stop-point 2 decisions (2026-11-03): E1 / E2, pre-registered before running

*A2 is not fixed. These rows and rules are fixed before running; they are deviations from the Plan's ablation
(recorded here).*

1. **E1 = A1** (A0 + NOR-TMD, 9 classes) **with tram excluded from the inner early-stopping loss** (tram windows
   removed from the stopping fold) and **HMM transitions from ExtraSensory + SHL User 1 only** (A0's matrix).
2. **E2 = best of A0 / E1 + Collecty.**
   - E1 not adopted: 8 classes; Collecty's tram minutes are unlabelled usable minutes (no tram class).
   - E1 adopted: 9 classes; the same stopping and HMM rules as E1.
3. **Rules:** the data-row rules of A1 / A2, each against the row it builds on (E1 vs A0 as A1; E2 vs its base as
   A2, with the NOR-TMD guard "no drop over 0.01"). **If neither passes, the final configuration is A0.** No further
   variants before the test.
4. Reported: rounds per fold of E1 / E2 next to A0's; E1's ExtraSensory still → rail minutes by quiet / not and
   with / without GPS (descriptive).
5. Descriptive, no retraining: **A1 with A0's transition matrix** (ExtraSensory six-class, NOR-TMD guard), the HMM's
   share of A1's drop.
6. **S and P are not rerun on the new base.** Both were clearly short of their +0.01 gain rule on A2 (−0.004, −0.001;
   CIs from −0.012 / −0.008), and the decision bars further variants before the test.
7. The experiment code and the `src/tmd` changes are committed before running (no push).
8. Then stop again with the proposed final configuration, before seeds 1–2 and before any test user.

### E1 / E2 (2026-11-03)

**Code for E1 / E2** (committed at `ff07ed5` before running, with every earlier `src/tmd` change):
- `src/tmd/models/lgbm.py`: `paired_cv(..., stop_ok=None)`, rows allowed in the early-stopping fold (default
  unchanged).
- `pipe.py`: `Row.hmm_sources` and `Row.stop_no_tram`; `labelled(sources, n)` (n = 8 leaves tram minutes out, so they
  are unlabelled for the HMM); with 8 classes, tram rows are never trained on (`acc_trainable`).
- `run.py`: `--hmm-sources`, `--stop-no-tram`. `hmm_swap.py` (item 5), `false_rail.py` (item 4).
- **A0's transition matrix at 9 states:** labelled minutes of ExtraSensory + SHL User 1 with 9 states. They hold no
  tram, so tram's row and column are the +1 pseudo-counts only: tram is rarely entered and left at once.

**Port check passed again** on the committed code (`logs/port_check.log`, rerun 2026-11-03 after the `stop_ok`
change): 10's feature tables equal 05's / 06's, and 05's B and 06's H1_v2 (seed 0) reproduce their saved
out-of-fold probabilities bit-identically, before and after the HMM.

Commands:
```
uv run python run.py --row E1 --sources extrasensory shl_user1 nor_tmd --hmm-sources extrasensory shl_user1 \
    --stop-no-tram --gps-from A0 --vs A0 A1
uv run python hmm_swap.py
uv run python run.py --row E2 --sources extrasensory shl_user1 collecty --n 8 --gps-from A0 --vs A0
uv run python false_rail.py A0 A1 E1 E2
```

**E1 (A1 with tram out of early stopping, A0's HMM): not adopted.**

| Rule (vs A0) | Difference [95% CI] | Limit | Pass |
|---|---|---|---|
| ExtraSensory dev six-class | **−0.013** [−0.032, +0.011] (0.647 → 0.634) | ≥ −0.01 | **no** |
| still precision | −0.001 [−0.001, +0.000] | ≥ −0.01 | yes |
| foot precision | −0.003 [−0.007, +0.001] | ≥ −0.01 | yes |
| (walk precision, 9-class) | −0.003 [−0.007, +0.001] | – | – |
| NOR-TMD dev guard | +0.349 [+0.264, +0.439] (0.324 → 0.673) | ≥ +0.01 | yes |

- **Rounds per fold:** E1 85, 60, 94, 77, 105, against A0 59, 40, 39, 55, 67 and A1 33, 46, 42, 26, 36. With tram out
  of the stopping fold the acc model trains longer than A0 (more data), not shorter.
- vs A1: ExtraSensory six-class +0.007 [−0.001, +0.017], NOR-TMD guard +0.000. Longer training recovers about a third
  of A1's drop; foot precision recovers fully (−0.009 → −0.003).
- What remains is bus and rail precision: bus 0.516 → 0.432 (−0.084 [−0.152, −0.018]), rail 0.271 → 0.172
  (−0.100 [−0.281, +0.061]).
- ExtraSensory 9-class +0.011 [−0.009, +0.028], family −0.010; SHL User 1 0.949 → 0.963; Collecty (transfer) 0.639
  → 0.661.
- NOR-TMD per class (9-class recall): bus 0.89, subway 0.79, train 0.28, car 0.52, bike 0.85, **tram 0.00** (A1:
  0.35). With A0's matrix, tram's transitions are pseudo-counts, so the HMM almost never enters tram (expected from
  the pre-registered HMM rule). No tram predicted on ExtraSensory or SHL.

**A1 with A0's transition matrix (descriptive, no retraining; MLflow `A1_hmmA0_seed0`):** ExtraSensory six-class
0.628, vs A1 +0.001 [−0.003, +0.006], vs A0 −0.019 [−0.039, +0.003]; NOR-TMD guard 0.654 (vs A1 −0.019
[−0.037, −0.003]); rail precision 0.144. **The HMM's share of A1's ExtraSensory drop is ~0.** NOR-TMD's HMM
helps NOR-TMD a little.

**E1 not adopted, so E2 = A0 + Collecty, 8 classes**, Collecty's tram minutes unlabelled usable minutes. Early
stopping as A0 (there is no tram class). HMM transitions from the row's sources (ExtraSensory + SHL User 1 +
Collecty, tram minutes unlabelled): the Plan's default, since the E1 HMM rule applies to E2 only if E1 is adopted.
Rule vs A0 as A2's.

**E2 (A0 + Collecty, 8 classes): adopted.**

| Rule (vs A0) | Difference [95% CI] | Limit | Pass |
|---|---|---|---|
| ExtraSensory dev six-class | +0.003 [−0.003, +0.008] (0.647 → 0.650) | ≥ −0.01 | yes |
| still precision | +0.000 [−0.000, +0.001] | ≥ −0.01 | yes |
| foot precision | −0.001 [−0.004, +0.003] | ≥ −0.01 | yes |
| (walk precision, 9-class) | −0.001 [−0.004, +0.003] | – | – |
| NOR-TMD dev guard | +0.019 [−0.005, +0.044] (0.324 → 0.343) | ≥ −0.01 | yes |

- Rounds per fold: 60, 37, 46, 57, 71 (A0 59, 40, 39, 55, 67).
- **No precision cost on ExtraSensory:** bus +0.006 [−0.021, +0.031], rail +0.010 [−0.026, +0.049]. A2's bus
  (−0.077) and foot (−0.010) losses are gone without the tram class: they came from tram's early stopping, not from
  Collecty's data.
- 9-class +0.005 [−0.002, +0.011], family +0.002 [−0.003, +0.006]; Collecty dev 0.639 → **0.706**; SHL User 1
  0.953; unseen positions as A0 (hand 0.816, torso 0.929, bag 0.949).
- NOR-TMD (transfer, unchanged in kind): bus 0.35, car 0.80, train 0.29, subway 0.03, tram 0.
- Costs: one window 11.6 ms (p95 11.9), 1.28 ms per window over sessions; acc model 1.6 MB, GPS model 4.1 MB.

**False rail on ExtraSensory dev still minutes** (`results/tables/false_rail.csv`; after the HMM)

| Row | quiet, GPS | quiet, no GPS | not quiet, GPS | not quiet, no GPS | all (users) |
|---|---|---|---|---|---|
| still minutes | 40,892 | 9,953 | 34,581 | 7,304 | 92,730 |
| A0 | 0 | 0 | 3 | 13 | 16 (6) |
| A1 | 0 | **127** (all subway) | 37 | **153** (104 subway, 48 tram) | 317 (14) |
| E1 | 0 | **115** (all subway) | 14 | 56 | 185 (9) |
| E2 | 0 | 0 | 3 | 24 | 27 (8) |

- With NOR-TMD, a quiet still phone without GPS becomes subway. Longer training (E1) removes most of the
  not-quiet cases but not the quiet ones. Where there is GPS, the fusion vetoes them (0 in every row).
- This is the dataset shortcut: NOR-TMD's metro minutes (Oslo, phones on the person) look like ExtraSensory's
  phones lying still. E1's 185 false-rail minutes come from 9 users.

**(d) with E1 / E2** (acc / final / final without GPS): bus E1 0.65 / 0.79 / 0.66, E2 0.66 / 0.84 / 0.80; subway
E1 0.81 / 0.91 / 0.90, E2 0.86 / 0.98 / 0.98; car, train, tram within ±0.06 of A0. As before, training on a source
lowers its identifiability from the output (E1 bus, subway); it doesn't raise it.

**HMM diagonal:** E1's tram 0.111 (pseudo-counts only, as stated above); E2 has no tram state. Collecty with tram
unlabelled: 4 mode changes in 9,151 steps.

### Stop-point 2 (second): proposed final configuration (2026-11-03)

**Proposed final configuration: E2**, by the pre-registered rules (E1 not adopted; E2 adopted vs A0).
- **acc model:** 05's 580 features, LightGBM lr 0.1, inner early stopping as 06, balanced class weights, **8 classes**;
  trained on ExtraSensory dev + SHL User 1 (hips, weight 1) + Collecty, Collecty's tram minutes unlabelled.
- **GPS model:** gps-v2, LightGBM lr 0.02 (06's), ExtraSensory + SHL User 1, covered minutes (= A0's).
- **Fusion:** product on GPS-covered minutes, acc model elsewhere. No tram class, so no tram rule.
- **HMM:** forward-backward, transitions from the labelled minutes of ExtraSensory + SHL User 1 + Collecty (tram
  unlabelled).
- **Reference:** A0 (06 retrained on v2's labels, ExtraSensory + SHL User 1).

**What E2 is and isn't:**
- Six-class dev +0.003 over A0 (CI across 0), Collecty +0.067, no precision cost: a safe addition, not an
  improvement on ExtraSensory.
- **No tram class and no NOR-TMD.** On the NOR-TMD test, tram is unreachable and subway / bus are transfer scores
  (dev: subway 0.03, bus 0.35). The NOR-TMD test of E2 and A0 measures transfer from ExtraSensory + SHL (+ Collecty)
  to Oslo, not a model trained for it.
- **NOR-TMD's gains (+0.35 on its dev) come with a shortcut:** quiet still minutes without GPS become subway on
  ExtraSensory. It is the main open question for the next experiment (e.g. NOR-TMD rail only where GPS exists, a
  lower NOR-TMD weight, or a quiet-phone feature check).

**Next, after OK:** seeds 1–2 on dev for E2 and A0. Then stop-point 3: the test protocol written into this README
and committed, before any test user is read.

**Open questions**
1. OK for E2 as the final configuration and A0 as the reference on the test?
2. On the NOR-TMD test, report tram (unreachable for both) as recall 0 in the per-class table, and the guard (bus,
   car, rail) as the headline?
3. E1 as a descriptive third model on the NOR-TMD test only (it is the only model trained on NOR-TMD), or not at all?

### Stop-point 2 approved (2026-11-03)

1. **Final configuration E2; reference A0.** Seeds 1–2 on dev for both; the test models' rounds come from those runs,
   as in 06.
2. **NOR-TMD test headline:** the dev guard's bus / car / rail macro-F1 (tram merged into rail). Per class: 9-class
   recall for bus, subway, train, tram and car with the support (users, windows); tram recall 0 where the model has
   no tram class; and which classes NOR-TMD's tram minutes are predicted as.
3. **E1: a descriptive third model** on the NOR-TMD test and the SHL Users 2–3 rail check per position only, 3 seeds
   (dev seeds 1–2 for the rounds). **Not on the ExtraSensory test**: that reading waits for the model a NOR-TMD
   follow-up chooses on dev. E1's results can't change the final configuration. E1's models are saved, to run beside
   E2 on labelled WAIIS days later.
4. **Before the test, descriptive, no retraining** (saved out-of-fold probabilities): acc-only + HMM (no GPS fusion)
   for A0, A1, A2, E1 and E2 on ExtraSensory dev (six-class, 9-class, per-class recall and precision), SHL User 1 day
   folds and NOR-TMD dev; the fusion gain (fused − acc-only) with a paired user-bootstrap CI, on all minutes and on
   GPS-covered minutes only, per row; whether the gain shrinks as sources are added; how much of the walk → still
   shift comes from fusion.
5. Then the test protocol in this README, committed with the E1 / E2 results and tables (no push) before any test
   user is read; then scored once, and stop-point 3 as in the brief.

### Dev seeds 1–2 (2026-11-03)

`run.py` with `--seed 1` / `--seed 2` for A0, E2 (`--gps-from A0`, the same seed's GPS model) and E1; log
`logs/stage_seeds.out`.

| Row | ES dev six-class, seeds 0 / 1 / 2 | mean | E2 − A0 per seed [paired CI] | NOR-TMD guard, seeds 0 / 1 / 2 |
|---|---|---|---|---|
| A0 | 0.647 / 0.650 / 0.653 | 0.650 | – | 0.324 / 0.320 / 0.335 |
| E2 | 0.650 / 0.651 / 0.648 | 0.650 | +0.003 [−0.003, +0.008] / +0.000 [−0.007, +0.008] / −0.006 [−0.016, +0.004] | 0.343 / 0.339 / 0.341 |
| E1 | 0.634 / 0.635 / 0.636 | 0.635 | (E1 − A0: −0.013 / −0.015 / −0.018) | 0.673 / 0.681 / 0.677 |

- **Over 3 seeds E2 and A0 are tied on ExtraSensory dev** (mean 0.650 each); E2's seed-0 lead (+0.003) was within
  the seed spread. E2 stays the final configuration (fixed at stop-point 2); what it adds over A0 is Collecty (dev
  0.706 / 0.692 / 0.709 vs A0 0.639 / 0.652 / 0.653) and +0.02 on the NOR-TMD guard, at no ExtraSensory cost.
- E1's ExtraSensory gap to A0 is stable across seeds (−0.013 to −0.018).

**Rounds** (median of the 5 out-of-fold early-stopping counts; the test models use these):

| Model | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| A0 acc | 55 | 62 | 56 |
| E2 acc | 57 | 60 | 55 |
| E1 acc | 85 | 88 | 87 |
| GPS (A0's, shared by all three) | 147 | 149 | 144 |

### Fusion gain: acc-only + HMM vs fused (descriptive, seed 0, no retraining)

`uv run python fusion_gain.py` (MLflow `<row>_acconly_seed0`; `results/tables/fusion_gain.csv`). Each row's acc
model with the row's own HMM, no GPS fusion, against the row's fused output; gain = fused − acc-only, paired
user-bootstrap 95% CI.

| Row | ES six-class acc-only | gain, all minutes | gain, GPS-covered minutes | ES 9-class gain | walk → still, acc-only / fused (of 13,109 walk minutes) |
|---|---|---|---|---|---|
| A0 | 0.578 | +0.069 [+0.024, +0.100] | +0.082 [+0.032, +0.118] | +0.058 [+0.026, +0.086] | 2,645 / 3,352 |
| A1 | 0.554 | +0.072 [+0.035, +0.113] | +0.105 [+0.045, +0.147] | +0.073 [+0.030, +0.115] | 2,608 / 3,337 |
| A2 | 0.563 | +0.076 [+0.026, +0.118] | +0.100 [+0.037, +0.144] | +0.073 [+0.028, +0.107] | 2,572 / 3,332 |
| E1 | 0.569 | +0.065 [+0.025, +0.107] | +0.091 [+0.031, +0.131] | +0.069 [+0.028, +0.113] | 2,785 / 3,397 |
| E2 | **0.589** | +0.061 [+0.020, +0.097] | +0.073 [+0.021, +0.117] | +0.054 [+0.020, +0.081] | 2,660 / 3,346 |

- **The gain doesn't shrink as sources are added; it follows the acc model.** It is largest where the acc model is
  weakest (A1, A2: acc-only 0.55–0.56, covered gain +0.10) and smallest for E2 (best acc-only, 0.589; +0.061). Every
  CI excludes 0.
- **Fusion is what keeps rail and bus precision up.** Acc-only rail precision is 0.013–0.073 in every row (fused
  0.13–0.28), bus 0.31–0.51 (fused 0.42–0.52). A1's acc-only bus precision (0.309) is the lowest: the NOR-TMD shortcut is in the acc
  model, and GPS hides part of it on covered minutes.
- NOR-TMD dev has no GPS, so its gain is 0 by construction; SHL User 1 −0.003 to +0.014.
- **Walk → still:** fused 3,332–3,397 walk minutes go to still, acc-only + HMM 2,572–2,785. **About 600–760 minutes
  (5–6% of walk; ~20% of the fused total) come from fusion**; the other ~80% are already in the acc model + HMM.
  The shift is the same in every row (06's walk → still weak spot, unchanged by the new sources).
- Acc-only per-class recall and precision for every row: MLflow `<row>_acconly_seed0`.

### Test protocol (fixed 2026-11-03, before any test user is read)

*Committed before `make_features.py --part test` runs. Nothing in it changes after the test is read. Code: `test.py`
(refuses to run once any `test_*` run exists).*

**Models, per seed 0, 1, 2** (one MLflow run per model and seed, `test_<model>_seed<k>`):

| Model | Role | acc model trained on (all dev users, all folds) | Classes | acc rounds (seeds 0 / 1 / 2) | HMM transitions from |
|---|---|---|---|---|---|
| **E2** | **final configuration** | ExtraSensory + SHL User 1 (hips) + Collecty, Collecty's tram minutes unlabelled | 8 | 57 / 60 / 55 | labelled dev minutes of ExtraSensory + SHL User 1 + Collecty (tram unlabelled) |
| A0 | reference | ExtraSensory + SHL User 1 (hips) | 8 | 55 / 62 / 56 | ExtraSensory + SHL User 1 |
| E1 | descriptive only | ExtraSensory + SHL User 1 + NOR-TMD | 9 | 85 / 88 / 87 | ExtraSensory + SHL User 1 (9 states) |

- Common to all: 05's 580 acc features, LightGBM lr 0.1, balanced class weights, source weight 1; GPS model gps-v2,
  lr 0.02, trained on the GPS-covered official dev minutes of ExtraSensory + SHL User 1, rounds 147 / 149 / 144 (A0's
  GPS model, shared); product fusion on GPS-covered minutes, acc model elsewhere (E1: vehicle-mean tram factor); HMM
  forward-backward over every usable minute of the test sessions.
- Rounds = the median of the seed's 5 out-of-fold early-stopping counts (as 06). No early stopping on the test.
- Saved: `<data>/processed/10_.../models/test_<model>_seed<k>.txt`, `test_gps_seed<k>.txt`,
  `test_<model>_seed<k>_hmm.npz` (E1's for the WAIIS days later).
- **E1 is descriptive and can't change the final configuration.** It is not scored on the ExtraSensory test.

**Test sets** (reserved set `splits/reserved_test_v3.json`; features by `make_features.py --part test`):
1. **ExtraSensory test, 12 users** (E2, A0): overall, iPhone and Android. Six-class (primary) over still, foot,
   bike, car, bus, rail; 9-class and family over the classes present (expected from 09: still, walk, run, bike,
   car, bus, train; train 50 windows / 2 Android users, run 74 / 3 users).
2. **NOR-TMD test, 14 installations** (E2, A0, E1). Headline: the dev guard's bus / car / rail macro-F1 (six-class
   predictions; tram merged into rail). 9-class recall for bus, subway, train, tram and car with the support (users,
   windows); tram recall 0 where the model has no tram class; the classes NOR-TMD's tram minutes are predicted as.
3. **SHL Users 2–3 rail check** (E2, A0, E1), per position (hips, hand, torso, bag): six-class, 9-class and family
   macro-F1, per-class recall (train and subway also before the HMM). No CI (2 users).

**Metrics and CIs**
- Macro-F1 over the classes present; per-class recall and precision; confusion matrices (MLflow).
- 95% CIs by user bootstrap (1,000 resamples, redrawn when a class is missing): ExtraSensory over users, NOR-TMD
  over installations; per seed.
- Paired differences E2 − A0 (and E1 − A0 on NOR-TMD) with paired bootstrap CIs, per seed, on the same minutes.
- **Headline = the mean of the 3 seeds**; each seed's value with its CI is reported too.
- Also: ExtraSensory six-class split by minutes with / without GPS per platform; walk → still minutes; tram predicted
  on ExtraSensory and SHL test minutes.
- Inference time (E2, A0; acc + GPS features, both models, product; single window and per window over 200 test
  sessions with the HMM; single thread, after warm-up, median and p95) and model sizes.

**After reading:** no retraining, no change of model, rounds, rules or class lists. Results, a draft Conclusion and
STATUS.md follow from these runs and the export.

### Stop-point 3: test (2026-11-03, scored once)

Commands (protocol committed at `315422a` before the first one; log `logs/stage_test.out`, `logs/test.log`):
```
uv run python make_features.py --part test
uv run python test.py
uv run python src/tmd/tracking.py export 10_gps-acc-baseline_waiis-bench-v2      # results/runs.csv
uv run python tables_test.py                                                     # results/tables/test.md
```
- Test features: ExtraSensory 65,991 usable minutes (26,382 official; GPS 0.814), SHL Users 2–3 hips 3,015 (2,365),
  other positions 9,047 (7,095), NOR-TMD 3,896 (3,221).
- **No deviation from the protocol.** The 9 test runs are clean at `315422a`. Earlier runs: A0–P, R06 and the tram
  runs are tagged dirty at `29934ba` (experiment code not yet committed); E1, E2 and `A1_hmmA0` seed 0 clean at
  `ff07ed5`; the dev seeds 1–2 and the `_acconly` runs dirty at `ff07ed5` (`fusion_gain.py`, `test.py` and the
  `pipe.load(part)` change uncommitted; none of them changes the dev runs' code path).
- `tables_test.py` shows a seed in which no tram minute went to a class as 0 (MLflow has no metric for it).
- Support: ExtraSensory test still 19,284 windows / 12 users, walk 3,682 / 11, run 74 / 3, bike 876 / 6, car 2,130 / 10,
  bus 286 / 7, train 50 / 2 (Android). NOR-TMD test bus 1,908 / 12, subway 738 / 5, tram 289 / 5, train 128 / 5,
  car 158 / 6.

Full tables: `results/tables/test.md` (each seed with its CI). Mean of 3 seeds:

**ExtraSensory test, 12 users**

| | E2 (final) | A0 (reference) | E2 − A0, per seed [paired CI] |
|---|---|---|---|
| six-class, all | **0.685** | 0.667 | +0.027 [+0.006, +0.037]; +0.006 [−0.021, +0.058]; +0.022 [+0.001, +0.030] |
| — iPhone | 0.756 | 0.757 | −0.002 [−0.014, +0.009]; −0.002 [−0.008, +0.001]; −0.002 [−0.022, +0.004] |
| — Android | 0.656 | 0.634 | +0.032 [+0.005, +0.044]; +0.009 [−0.020, +0.060]; +0.027 [+0.014, +0.035] |
| 9-class, all | 0.727 | 0.711 | +0.013; +0.016; +0.017 |
| family, all | 0.757 | 0.743 | +0.025; +0.003; +0.013 |

- E2's six-class per seed: 0.682 [0.605, 0.718], 0.685 [0.617, 0.719], 0.688 [0.611, 0.719]. A0's: 0.655, 0.679, 0.666;
  **A0's test seed spread (0.024) is as large as E2's mean gain (+0.018)**. E2's spread is 0.006.
- **Where E2's gain sits: Android rail and bus.** Android six-class F1, A0 → E2 (seed mean): rail 0.456 → 0.517,
  bus 0.228 → 0.276, bike +0.012, car +0.011, still and foot +0.00. Rail is ExtraSensory train: 50 windows from 2
  users. Rail precision A0 0.38 / 0.70 / 0.48, E2 0.59 / 0.62 / 0.64: E2 is the steadier model there.
- 9-class recall (E2 / A0): still 0.95 / 0.95, walk 0.65 / 0.65, run 0.79 / 0.80, bike 0.86 / 0.85, car 0.77 / 0.77,
  **bus 0.29 / 0.25**, train 0.45 / 0.42 (CIs in `test.md`; train and run CIs reach 0).
- **Walk → still: 989 / 990 of 3,682 walk minutes (27%)**, unchanged from 06's weak spot (06: 981).
- iPhone: no difference anywhere (as 06: no gain on iPhone test users); on iPhone minutes without GPS both score
  0.462 six-class.
- No tram can be predicted (8 classes).
- Comparison with 06 (v1 labels, rescored by 09): 06's test six-class 0.686, iPhone 0.752, Android 0.654. A0 is 06
  retrained on v2's labels (run flag: test run 81 → 74); its 0.667 is within its own seed spread of 06's number.

**NOR-TMD test, 14 installations** (bus / car / rail macro-F1, the dev guard)

| | E2 | A0 | E1 (descriptive) |
|---|---|---|---|
| bus / car / rail | 0.469 | 0.463 | **0.778** |
| − A0, per seed [paired CI] | −0.007 [−0.039, +0.024]; +0.017 [−0.023, +0.049]; +0.007 [−0.019, +0.029] | – | +0.307 [+0.123, +0.501]; +0.327 [+0.150, +0.516]; +0.310 [+0.128, +0.497] |
| bus recall | 0.39 | 0.42 | 0.82 |
| subway | 0.11 | 0.17 | 0.97 |
| train | 0.52 | 0.35 | 0.22 |
| tram | 0 (no class) | 0 (no class) | 0 |
| car | 0.66 | 0.66 | 0.58 |

- **NOR-TMD's tram minutes (289) are predicted as** train 170, still 55, subway 33, car 22 (E2); train 147, still 67,
  subway 54 (A0); **subway 262** (E1). E1 has a tram class but its HMM (A0's transitions) almost never enters it
  (the pre-registered E1 rule); its acc + GPS output calls Oslo trams subway.
- E1's dev → test: guard 0.677 → 0.778, subway 0.79 → 0.97, bus 0.89 → 0.82. E1 transfers to unseen NOR-TMD
  installations; E2 and A0 don't (dev 0.343 / 0.324 → test 0.469 / 0.463).

**SHL Users 2–3 rail check** (six-class; mean of 3 seeds; no CI)

| Position | E2 | A0 | E1 | subway recall, E2 / A0 / E1 |
|---|---|---|---|---|
| hips | 0.945 | 0.942 | 0.960 | 0.62 / 0.59 / 0.71 |
| hand | 0.852 | 0.837 | 0.877 | 0.08 / 0.08 / 0.33 |
| torso | 0.889 | 0.880 | 0.901 | 0.18 / 0.23 / **0.98** |
| bag | 0.931 | 0.938 | 0.947 | 0.46 / 0.51 / 0.62 |

- Train recall is 0.82–0.99 everywhere; **subway is the weak class off the hips** for E2 / A0 (hand 0.08, torso
  0.18), and the HMM lowers it on the hand (before the HMM 0.20). E1 (NOR-TMD's metro in training) lifts subway at
  every position.
- 06 (v1, rescored): 0.944 hips. E2 0.945.

**Cost** (E2 / A0): 12.0 ms per single window (p95 13.5 / 13.3), 1.1 ms per window over whole sessions (p95 4.5);
acc model 1.65 / 1.66 MB, GPS model 4.09 MB; E1's acc model 2.78 MB.

### Descriptive check before approval: what lies behind E1's false rail (2026-11-03)

*Dev only, no retraining; E1 seed 0 after the HMM. `uv run python diag_quiet.py` (`logs/diag_quiet.log`,
`results/tables/diag_quiet_*.csv`). No decision or rule changes.*

**1. Quiet windows among labelled vehicle windows** (official dev windows; per-user spread over users with ≥ 5
windows)

| Source | bus | car | train | subway | tram |
|---|---|---|---|---|---|
| ExtraSensory | 5.5% (21 users; median 3.7%, max 30%) | 8.5% (38; median 3.6%, IQR 0.5–11%, max 54%) | 11.6% (3; 0 / 0 / 23%) | – | – |
| SHL User 1 | 2.4% | 1.4% | 6.7% | 6.0% | – |
| NOR-TMD | **0.7%** (62; median 0, max 12%) | 4.0% (11; max 16%) | 1.7% (15; max 12%) | **3.3%** (25; median 0, max 10%) | 2.0% (17; max 11%) |
| Collecty | 0.4% | 2.9% | 2.8% | – | 2.0% |

- **NOR-TMD labels fewer quiet vehicle windows than ExtraSensory or SHL, not more.** Its subway has 3.3% quiet
  windows (108), against SHL's 6.0% and ExtraSensory's own vehicles 5.5–11.6%. A label convention that calls quiet
  windows "vehicle" is not what NOR-TMD adds.

**2. E1's 185 false-rail still minutes**

| | quiet | not quiet, no GPS | not quiet, GPS |
|---|---|---|---|
| minutes | 115 (of 50,845 quiet still) | 56 (of 7,304) | 14 |
| users | **1** (4E98F91F, Android) | 6 (55% Android; top user 15) | 3 (one iPhone user 11) |
| sessions / contiguous runs | 9 / 44 (median length 1, max 22) | 14 / 24 (24 minutes in runs ≥ 5) | 3 / 4 |
| predicted | subway 115 | not quiet together: subway 61, train 9 | |
| phone position (ExtraSensory tags; not quiet together) | in hand 54%, on table 30% (all quiet still: 5%, 17%) | in hand 46%, on table 9% (all not-quiet still: 13%, 9%) | |
| context tags | computer work 26%, browsing 21%, sitting 31%, home 24%, location 0% | sitting 80%, talking 20%; not quiet together: eating 11%, home 34% | talking 93% |

- **The quiet part is one device.** User 4E98F91F (Android, fold 3) has no location data at all (GPS coverage 0)
  and 1,118 still minutes. E1 calls 115 of its 334 quiet minutes subway (A1: 126). **The other 26 users' 9,619
  quiet still minutes without GPS give 0 false rail** (A1: 1); A0 and E2 give 0 on all of them, 4E98F91F
  included.
- The not-quiet minutes are scattered: mostly single minutes, in many sessions, while sitting, talking or eating.

**3. Vehicle labels near the 56 not-quiet false-rail minutes without GPS** (same user; class car / bus / train, or the
original ON_A_TRAIN / ON_A_BUS / IN_A_CAR / DRIVE tags; ExtraSensory has no trolley or tram label)

| | within ±10 min | within ±30 min | same day (UTC) | own vehicle tag |
|---|---|---|---|---|
| 56 false rail, not quiet, no GPS | **0%** | 1.8% (1 minute; a car label) | 73% | 0 |
| all 7,304 not-quiet still minutes without GPS | 2.4% | 7.4% | 56% | 0 |
| (14 false rail, not quiet, with GPS) | 43% | 43% | 100% | 0 |

- **No sign of rides:** no vehicle label within 10 minutes of any of the 56, fewer than for still minutes in general,
  and none tagged as a vehicle themselves. They are still minutes the model misreads, not unlabelled rides.
- The 14 not-quiet minutes with GPS sit next to labelled rides (43% within 10 min): ride boundaries.

**4. E1's false bus on ExtraSensory dev** (six-class bus predicted on another class; `diag_quiet_false_bus.csv`)

| Row | predicted bus | false bus | with GPS | from still / foot / bike / car / rail |
|---|---|---|---|---|
| A0 | 1,104 | 534 | 467 (87.5%) | 172 / 148 / 8 / 193 / 13 |
| E1 | 1,383 | 785 | 702 (89.4%) | 245 / 173 / 0 / 354 / 13 |

- E1's bus precision drop (−0.084) is +251 false-bus minutes, mostly car (+161) and still (+73). 89% have GPS (all
  ExtraSensory official minutes: 82%): unlike the false rail, fusion doesn't stop them.

**Reading.** E1's false rail is not a quiet-window label convention in NOR-TMD (1), and not unlabelled rides (3). It
is **concentrated in one Android device without location data** (4E98F91F, 115 of 185 minutes) **plus scattered
sitting minutes** (56 minutes, 6 users); **the mechanism is not shown.** Where GPS exists, fusion overrules it. E1's
false bus is a separate cost, on minutes with GPS (4). Stop-point 2's wording ("quiet still phones without GPS become
subway") over-generalised from one user; the Conclusion and the DECISIONS.md draft now say what is shown here.

## Conclusion

**Draft — not approved.**

**Answer.** On `waiis_bench_v2`, 06's configuration retrained on v2's labels (A0) is the reference, and **E2 — A0 +
Collecty, 8 classes — is the final configuration**. NOR-TMD is not in it: with it, ExtraSensory dev six-class falls
past the rule's limit, through false rail and false bus. Tram is not a class.

| Mean of 3 seeds | E2 (final) | A0 (reference) | E2 − A0 |
|---|---|---|---|
| ExtraSensory test, six-class (12 users) | **0.685** | 0.667 | +0.018 (seed CIs from −0.021 to +0.058) |
| — iPhone / Android | 0.756 / **0.656** | 0.757 / 0.634 | −0.002 / +0.023 |
| ExtraSensory dev, six-class (pooled OOF) | 0.650 | 0.650 | −0.001 |
| NOR-TMD test, bus / car / rail (14 installations) | 0.469 | 0.463 | +0.005 |
| SHL Users 2–3 rail check, hips six-class | 0.945 | 0.942 | +0.003 |

- **E2 is a safe addition, not a proven improvement.** On dev it ties A0 (0.650 each over 3 seeds). On the test it
  gains +0.018, all on Android (rail and bus, 2 and 7 users), and A0's own test seed spread is 0.024. Collecty's
  users gain +0.07 on dev at no ExtraSensory cost.
- **NOR-TMD (A1 / E1):** +0.35 on its dev users and +0.31 on its 14 unseen test installations (bus / car / rail
  0.778; subway 0.97), and better subway at every SHL position. But ExtraSensory dev six-class falls 0.013–0.020 in
  every variant and seed (A1, E1). Seeds don't resample users: E1's paired user-bootstrap CI is −0.013 [−0.032,
  +0.011], and one user carries 62% of its false rail. By the rules it is past the limit, so it isn't adopted; the
  rule outcome stands.
  - **False rail** on ExtraSensory still minutes (E1 185, A0 16), 171 of them without GPS. From the check before
    approval: not a label convention (NOR-TMD has fewer quiet vehicle windows than ExtraSensory or SHL: subway 3.3%,
    bus 0.7%) and not unlabelled rides (no vehicle label within 10 minutes of any of the 56 not-quiet ones without
    GPS). It is **concentrated in one Android device without location data (4E98F91F, 115 of 185 minutes) plus
    scattered sitting minutes; the mechanism is not shown.** The other 26 users' 9,619 quiet still minutes without
    GPS give 0. Where GPS exists, fusion overrules it.
  - **False bus:** E1's ExtraSensory bus precision falls by 0.084 [−0.152, −0.018]: 785 false-bus minutes against
    A0's 534, mostly from car and still, and 89% of them with GPS, so fusion doesn't stop them.
- **Tram:** Collecty's 4 tram users don't generalise (recall 0 on its own held-out users), and with balanced class
  weights the tiny tram class cut the acc model's early stopping in half (A2: 18–25 rounds vs ~50), which is what
  cost A2 bus and foot precision. Without NOR-TMD, tram isn't learnable on v2.
- **What carried over from 06:** GPS fusion is worth +0.06 to +0.08 six-class on dev in every row (+0.07 to +0.10 on
  GPS-covered minutes) and is what keeps rail and bus precision up. Walk → still is unchanged (27% of test walk
  minutes; ~20% of the shift comes from fusion, ~80% from the acc model + HMM). No gain on iPhone. SHL weight 0.5 and
  SHL's other positions add nothing.
- **Dataset shortcut checks:** within-class source identification from the model's output is lower than from the
  features (bus 0.65–0.68 from the acc model's probabilities vs 0.82 from its features), and training on a source
  lowers it. What costs ExtraSensory with NOR-TMD is across classes: false rail on still minutes without GPS
  (concentrated in one Android device without location data plus scattered sitting minutes; mechanism not shown)
  and false bus on minutes with GPS.

**Limits.**
- One ExtraSensory study (2015 phones) for still / walk and the primary score; the test gain rests on 2 Android train
  users and 7 bus users.
- NOR-TMD test is Android only; its train (5 users) and car (6 users) cells are thin.
- The rules fixed at stop-point 1 chose on seed 0; the dev tie between E2 and A0 showed up only with seeds 1–2.
- E1's tram class was switched off by its pre-registered HMM rule, so tram recall with NOR-TMD under a tram-aware HMM
  wasn't measured on the test.
- WAIIS's GPS cadence (~20 s) and phones are not in the benchmark.
- The false-rail failure needs a resting phone with no location at all (4E98F91F has none). How often that happens
  in WAIIS production (the app records while moving; GPS is missing mainly in tunnels) is unknown.

**Next.**
- **A NOR-TMD follow-up on dev:** keep NOR-TMD's vehicle gains without the false rail and false bus on ExtraSensory
  (e.g. a lower NOR-TMD weight, or still minutes of many devices as hard negatives), with a tram-aware HMM;
  per-user false-rail counts as a check (one user carried 62% of E1's); a comparison of 4E98F91F's resting signal
  with NOR-TMD's (subway and other classes). Only then the ExtraSensory test for it.
- **Leg suggestions:** E2 is the model for `predict_legs`; E1's saved models run beside it on the labelled WAIIS
  days.
