# Status

*As of 2026-10-29. Project facts and rules: [CLAUDE.md](CLAUDE.md). Decisions: [DECISIONS.md](DECISIONS.md).*

## 1. Data

**`waiis_bench_v1`** is the frozen benchmark every model reads (`tmd.bench`), built at commit `e51ea79`.
- Contents, counts and rules: [experiment 04's README](experiments/04_frozen-benchmark_es-shl/README.md).
- Data decisions: [decision register](experiments/04_frozen-benchmark_es-shl/decision_register.md).
- Splits: [splits/README.md](splits/README.md).

## 2. Current results

No model has been evaluated on `waiis_bench_v1` yet, so there is no best model.

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

An accelerometer-only baseline on `waiis_bench_v1`: LightGBM on 02's features plus HMM smoothing.
- Cross-validated on the dev folds; final evaluation on the reserved test users.
- Reported as `CLAUDE.md`'s evaluation conventions require.
