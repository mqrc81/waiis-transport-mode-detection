# Status

*As of 2026-10-30. Project facts and rules: [CLAUDE.md](CLAUDE.md). Decisions: [DECISIONS.md](DECISIONS.md).*

## 1. Data

**`waiis_bench_v1`** is the frozen benchmark every model reads (`tmd.bench`), built at commit `e51ea79`.
- Contents, counts and rules: [experiment 04's README](experiments/04_frozen-benchmark_es-shl/README.md).
- Data decisions: [decision register](experiments/04_frozen-benchmark_es-shl/decision_register.md).
- Splits: [splits/README.md](splits/README.md).

## 2. Current results

**Reference model: experiment 06**, accelerometer + GPS fusion
([README](experiments/06_gps-acc-baseline_waiis-bench/README.md)). Conclusion **drafted, not yet approved**.
- It replaces **05** (accelerometer only; [README](experiments/05_acc-baseline_waiis-bench/README.md), approved
  2026-10-30), which stays the accelerometer-only baseline.

| `waiis_bench_v1` | 8-class macro-F1 (06 / 05) | family macro-F1 (06 / 05) |
|---|---|---|
| ExtraSensory test, 12 users (mean of 3 seeds) | **0.707** / 0.666 | **0.747** / 0.698 |
| — iPhone / Android | 0.764 / 0.682 (05: 0.762 / 0.665) | 0.847 / 0.708 (05: 0.801 / 0.690) |
| ExtraSensory dev, pooled out-of-fold | 0.613 / 0.551 | 0.681 / 0.615 |
| SHL Users 2–3, rail check | 0.847 / 0.795 | 0.944 / 0.932 |

- **Model (06):**
  - acc model: 05's LightGBM (580 features, `tmd.acc_features`);
  - GPS model: LightGBM on 13 GPS features from one fix per minute (`tmd.gps_features`, gps-v2: speeds derived
    from positions, acceleration, stops, heading change), trained with SHL User 1;
  - fusion: product of the two models' probabilities, with the acc model alone where there is no GPS;
  - smoothing: HMM forward-backward per session.
  - Cost: 1.4 + 3.8 MB; ~12 ms per single window, ~1 ms per window over whole sessions.
- **Adopted despite a failed platform check** (a recorded deviation): GPS features identify iPhone vs Android
  within a class. A cross-platform transfer test showed the gain doesn't depend on it.
  - A DECISIONS.md entry (draft) makes this transfer test the criterion for GPS features.
- **What mattered:**
  - the HMM (05: +0.12 family);
  - GPS late fusion (06: +0.06 8-class on dev, +0.04 on test);
  - SHL User 1 in training, for both models.
  - Not adopted: stacking, early fusion, a lower learning rate, gravity, low-pass, context features.
- **Weak spots:** bus (test 0.25), ExtraSensory train / rail (2 Android test users), walk (test recall falls with
  GPS, 0.715 → 0.650), no 8-class gain on iPhone test users.
- **Evaluation:** user-bootstrap CIs redraw resamples that lack a class of the evaluation set
  (`tmd.evaluate.user_resamples`).

## 3. Superseded experiments

- **01 UniMTS few-shot on SHL.**
  - Outdated: one SHL user, a day split, acc + gyro comparisons.
  - Still holds: pretrained UniMTS beats LightGBM only at k ≤ 2 and is weak on vehicles; not adopted.
- **02 family baseline on SHL + ExtraSensory.**
  - Outdated: acc + gyro features, the old splits and labels, numbers not on the benchmark.
  - Still holds:
    - HMM smoothing (Viterbi) is the cheapest gain (+0.09–0.11 family F1);
    - SHL alone doesn't transfer to ExtraSensory, so training data matter more than the model;
    - context features built from the context length leak the dataset;
    - car vs bus and train vs subway stay weak without GPS.
- **03 label curve on SHL User 1.**
  - Outdated: a same-person day split, acc + gyro.
  - Still holds: the curve hadn't flattened by k = 1000, and the remaining gain is in the vehicle classes.

## 4. Next experiment

Not decided yet. Candidates named in 05 and 06: InceptionTime + GPS fusion, and LIMU-BERT pre-trained from scratch
(accelerometer only). Each has to beat 06 on the same test users and protocol.
