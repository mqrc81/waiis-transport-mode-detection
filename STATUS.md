# Status

*As of 2026-10-30. Project facts and rules: [CLAUDE.md](CLAUDE.md). Decisions: [DECISIONS.md](DECISIONS.md).*

## 1. Data

**`waiis_bench_v1`** is the frozen benchmark every model reads (`tmd.bench`), built at commit `e51ea79`.
- Contents, counts and rules: [experiment 04's README](experiments/04_frozen-benchmark_es-shl/README.md).
- Data decisions: [decision register](experiments/04_frozen-benchmark_es-shl/decision_register.md).
- Splits: [splits/README.md](splits/README.md).

## 2. Current results

**Reference model: experiment 05**, an accelerometer-only LightGBM + HMM baseline
([README](experiments/05_acc-baseline_waiis-bench/README.md)). Conclusion approved 2026-10-30.

| `waiis_bench_v1` | family macro-F1 | 8-class macro-F1 |
|---|---|---|
| ExtraSensory test, 12 users (mean of 3 seeds) | 0.698 (iPhone 0.801, Android 0.690) | 0.666 |
| ExtraSensory dev, pooled out-of-fold | 0.615 | 0.551 |
| SHL Users 2–3, rail check | 0.932 | 0.795 |

- **Model:**
  - 580 features (`tmd.acc_features`: features v1 without gyroscope on the 15 s window + 5 s sub-window
    statistics);
  - LightGBM on ExtraSensory dev + SHL User 1;
  - HMM forward-backward per session.
- **Cost:** 1.3 MB; 11–14 ms per single window, ~1 ms per window over whole sessions.
- **What mattered:**
  - the HMM was the largest gain (+0.12 dev, +0.18 test);
  - SHL User 1 in training helped ExtraSensory (+0.03);
  - sub-window statistics gave +0.01;
  - gravity features, a 10 Hz low-pass and context over neighbouring minutes were not adopted. Context values
    identify the dataset.
- **Weak spots:** bus (test 0.22), ExtraSensory train / rail (the HMM cuts dev rail recall 0.24 → 0.08; not on
  test), not-quiet still.
- **Evaluation change:** user-bootstrap CIs now redraw resamples that lack a class of the evaluation set
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

**06 GPS + accelerometer baseline** (planned 2026-10-30): GPS features fused with 05. Other candidates named in 05's Question: InceptionTime + GPS fusion, and LIMU-BERT pre-trained from
scratch (accelerometer only). Each has to beat 05 on the same test users and protocol.
