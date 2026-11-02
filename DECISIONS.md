# Decisions

## 2026-10-23: UniMTS is not adopted as the production model

**Decision.** We will not use pretrained UniMTS (zero-shot, linear probe or fine-tuned) as the production
transport-mode model.

**Why.**
- **Not better than LightGBM once there are a few labels.** On the same k-shot windows, the best pretrained variant
  (linear probe) behaves like this against LightGBM (|acc| features, 20 Hz):
  - Set B (still / walk-or-run / vehicle): below from k=5 (−0.07 macro-F1 at k=5, −0.04 at k=10).
  - Set C (8 classes): ahead at k=5 (+0.04) and level at k=10 (−0.00).

  The probe's advantage at k ≤ 2 (+0.06 to +0.08) comes from the motion classes. On vehicles it is weaker than
  LightGBM. For example, in set C at k=10 the probe recalls car 0.30, bus 0.29, train 0.24 and subway 0.18.

**Limits of the evidence.** There is one user (SHL User 1, hips), the split is by day, and there are 5 seeds per k.
The paired standard deviations are often as large as the differences against LightGBM. Inference time per window
was not measured.

**Evidence.** `experiments/01_unimts-fewshot_shl/` (README Results and Conclusion, and `results/shl_fewshot.md`).

## 2026-10-23: Bologna TMD is a minor dataset, not a main training or test set

**Decision.** We will not build the model or its evaluation mainly on the Bologna TMD dataset (Carpineti et al.
2018). At most it is an extra held-out check, or a small addition to training. It is never the only evidence for a
decision.

**Why.**
- **Little usable data.** Of about 35 h of recordings (16 users), only a small part gives usable 10 s windows. A
  window is usable when its real rate is at least 10 Hz and it has no gap over 0.5 s.

  | Track | Windows | Users per class | Car windows |
  |---|---|---|---|
  | Acc + gyro | 1021 (~2.8 h) | 1 | 33 |
  | Acc only | 2829 (~7.9 h) | 3–6 | 643 |
- **No split by user is possible with the gyroscope.** In the acc + gyro track, U12 provides 97% of the windows, and
  every class comes from a single user. U14–U16 contribute nothing to either track.
- **Recording varies a lot between users.** The sample rate goes from about 1 Hz (U15) to about 200–250 Hz (U11,
  U12). The paper's maximum of 20 Hz does not hold for U11 and U12. U1–U3 and U13–U15 record only when the value
  changes, at a median of 1–6 Hz, so resampling them to 20 Hz would mostly interpolate. Four users have no gyroscope
  (U4, U6, U8, U16), and U10's gyroscope runs at about 0.6 Hz.
- **Small format problems.** 9 files are empty. 4 U12 files are length-prefixed (Java `writeUTF`); they decode
  without guessing but were not in the paper's totals.

**Limits of the evidence.** The timestamps are assumed to be in ms: confirmed by walking cadence for U5, U6, U10
and U12, and not testable for the low-rate users. This is a quality check, not an experiment. Its numbers are not
in MLflow.

## 2026-10-25: The model runs on an edge server, not on the phone

**Decision.**
- The phone only records and uploads; the transport-mode model runs on an edge server.
- Model size is not a hard limit; it is only reported.
- Reported metrics: macro-F1, confusion matrix with per-class recall, inference time per window, and model size.

**Why.** The WAIIS app only records and uploads the sensor data, so the model doesn't have to fit on the phone. The
earlier 2 MB on-phone ceiling no longer applies.

**Evidence.** 2026-10-25. `CLAUDE.md` (Project, Evaluation conventions). Experiment 02's LightGBM (6.5 MB on
average) was over the old ceiling (`experiments/02_family-baseline_shl-es/README.md`, step 2).

## 2026-10-28: Accelerometer only; final test on the 12 ExtraSensory test users

**Decision.**
- The model uses the **accelerometer only**, no gyroscope.
- The **final test is the 12 ExtraSensory test users** (`splits/reserved_test.json`).
  Replaced the same day; see the next entry.
- **SHL Users 2–3 are a separate rail check only**, not the final test.
- **Prediction runs in batches** over uploaded recordings, so smoothing may use the whole sequence (e.g. Viterbi),
  not only past windows.

**Why.** Changed direction: the WAIIS app does not collect the gyroscope, and same-person day splits
(SHL User 1) are considered optimistic. The label curve will be redone on unseen ExtraSensory users.

**Evidence.** Experiment 03 (`experiments/03_label-curve_shl/`)
was closed at stop-point 2 because of it; on its one seed the gyroscope added about + 0.01 macro-F1 at k ≥ 50 (not
tested across seeds).

## 2026-10-28: ExtraSensory test users re-picked; the benchmark is waiis_bench_v1

**Decision.**
- Every model reads **`waiis_bench_v1`** (experiment 04).
- The reserved ExtraSensory test set is **`splits/reserved_test_v2.json`**: 12 users re-picked in experiment 04.
  It replaces the 2026-10-25 set, kept unchanged as `experiments/02_family-baseline_shl-es/splits/reserved_test.json`.
- The dev folds are `experiments/04_frozen-benchmark_es-shl/splits/es_users_v2.csv`.
- SHL Users 2–3 (the rail check) are unchanged.
- All data decisions behind waiis_bench_v1 (label rules, exclusions, caps, plausibility, windowing, GPS) are in
  experiments/04_frozen-benchmark_es-shl/decision_register.md.
- ExtraSensory test results are always reported overall and per platform. Per-class results come with
  user-bootstrap 95% CIs (`tmd.evaluate.bootstrap_users_recall`). There is no separate "without location"
  breakdown: it is effectively Android-only.

**Why.**
- The first pick (2026-10-25) used counts that no longer apply: labels v1, a gyroscope required, and no
  accelerometer plausibility check. It had 2 run users, 1 train user, and 95% location coverage against 83%
  overall.
- On `waiis_bench_v1` the new test set has ≥ 5 examples from run 4, bike 6, bus 6 and train 2 users. iPhone share
  is 62.5% (dev 61.7%) and location coverage 85.6% (dev 82.7%).

**Limits of the evidence.**
- 11 of the 12 new test users were dev users in experiment 02, used for training and model choices there. The new
  test set is therefore not untouched by earlier decisions.
- Test train is Android only: 52 examples from 2 users. Test run has 81 examples.

**Evidence.** Approved on 2026-10-28.
`experiments/04_frozen-benchmark_es-shl/README.md` and `decision_register.md`.

## 2026-10-28: GPS is part of the model

**Decision.**
- GPS features, including OpenStreetMap way types, are part of the model: an accelerometer model fused with a
  GPS-feature model.
- This replaces the internship plan's policy of keeping GPS as a separate validation stage
  (`experiments/02_family-baseline_shl-es/docs/brief_block1_experiments.md`, Goal: "GPS never enters the sensor
  model's features").

**Why.**
- A change of project direction (2026-10-28).
- GPS speed separates still from vehicles, and helps tell rail from road.
- WAIIS sends GPS with every upload.

**Evidence.** Direction change, 2026-10-28. The GPS tables of `waiis_bench_v1`
(`experiments/04_frozen-benchmark_es-shl/README.md`, Contents).

## 2026-10-28: No pre-trained LIMU-BERT or LIMU-BERT-X weights

**Decision.**
- The public LIMU-BERT and LIMU-BERT-X weights are not used.
- If LIMU-BERT is used, it is pre-trained from scratch with `feature_num = 3` (accelerometer only) on unlabelled
  benchmark data.
- Pre-training data comes from `tmd.bench.pretraining_minutes`, which excludes the reserved test users.

**Why.** The public weights expect accelerometer + gyroscope input, and the WAIIS app records no gyroscope.

**Evidence.**
- `limu_bert_x.pt`: its input layer is 72 × 6 (checked 2026-10-28; the file is not on this machine).
- LIMU-BERT's public config (`../ext/LIMU-BERT-Public/config/limu_bert.json`, `base_v1`): `feature_num` 6,
  `hidden` 72.

## 2026-10-28: Labelling is on demand

**Decision.**
- The model is trained on the labelled data, then predicts new trips in batches, with smoothing.
- The least certain segments go to the labelling tool, and the model is retrained.
- Confident predictions may be used as extra training labels.
- The test set is always labelled by people.

**Why.** Labels are expensive and unlabelled data is plentiful, so labelling effort goes where the model is least
certain. Human labels on the test set keep the evaluation independent of the model's own predictions.

**Evidence.** 2026-10-28. Batch prediction with whole-sequence smoothing: entry 2026-10-28 "Accelerometer
only".

## 2026-10-30: Platform effects of GPS features are judged by a cross-platform transfer test

**Decision.**
- For GPS features, platform effects are judged by a **cross-platform transfer test**: train the GPS model without
  one platform, score that platform, and compare the fused model's gain there with its gain in the normal folds
  (paired user-bootstrap CIs).
- The within-class platform AUC from the GPS features is reported **for context only**, not as a pass / fail check.

**Why.**
- GPS recording differs by platform: ExtraSensory Android positions update less often. So GPS features identify the
  platform within a class (walk AUC 0.746 for gps-v1, 0.730 for gps-v2, against 0.510 from the accelerometer
  features), even when the model doesn't rely on it.
- In experiment 06 the fusion's gain carried over completely to a platform the GPS model never saw: transfer − normal
  8-class differences were +0.002, +0.000, +0.007 and −0.002, all CIs containing 0.
- What matters is whether the gain carries across platforms, and the AUC doesn't measure that.

**Limits of the evidence.** One study (ExtraSensory, two platforms, 2015 phones and app); the transfer test doesn't
establish transfer to WAIIS phones or the WAIIS app's GPS recording. One seed for the transfer test.

**Evidence.** `experiments/06_gps-acc-baseline_waiis-bench/README.md` (Results: stop-point 2 follow-up, and the
decision after it).

## 2026-10-30: WAIIS labelling starts with a few users per mode and platform, then targets bus and train

**Draft — not approved.**

**Decision.**
- 06 is WAIIS's starting point (n = 0). The first WAIIS labels go to **a few users per mode and per platform**
  (iPhone, Android).
- After that, labelling **targets bus and train** rather than more users in general. A better model, or a public base
  with rail and bus, is weighed against further labelling.
- Public base data (SHL User 1) stays in training for both models.
- Run is not used to judge models until the dev / test difference in run recall is understood (a question for the
  next benchmark version).

**Why.**
- **The domain step:** on top of SHL User 1, 3 ExtraSensory users raise 8-class dev F1 from 0.452 to 0.586 (+0.13);
  the remaining ~35 add +0.03 (0.613). Past n = 3 the gain is about +0.01 per doubling (flattens at n = 3 by the
  pre-set rule; paired gain from n = 20 to all +0.004 to +0.013 per draw).
- **Classes still gaining at the end:** bus most (F1 +0.027 per doubling on the last step), then walk and bike;
  still, car and run have stopped.
- **SHL User 1:** without it, ~38 ExtraSensory users reach 0.544. At n = all, 59% of that gap is train, which only SHL
  provides. On the 6 non-rail classes the gap is +0.033.
- **Run:** dev recall 0.26 at every n, unchanged by 13 ExtraSensory run users, against 0.74 on the test users (06).

**Limits of the evidence.** ExtraSensory users (2015 phones and app), not WAIIS employees; development data only, a
pool of ~38 users; 5 draws; no paired bootstrap of the gains; rail can't be measured (2–3 ExtraSensory train users).

**Evidence.** `experiments/07_user-label-curve_waiis-bench/README.md` (Results, stop-point 2 and review;
Conclusion).

## 2026-11-02: Candidate data for waiis_bench_v2: NOR-TMD (Android), Collecty; not MobilitApp

**Decision.**
- `waiis_bench_v2` adds **NOR-TMD, Android recordings only**, for training, and as a second test set for vehicles
  and rail (~20% of its installations, stratified with a minimum of users per class; picked once, before any v2
  model reads it).
- `waiis_bench_v2` adds **Collecty** for training, with the approved clock repair, dedup and a per-user cap per class.
- **MobilitApp is not used.**
- v1's 8 classes stay; tram is the only candidate for a new class and waits on which modes WAIIS needs.
- **Every reader converts iOS accelerometers to the Android axis convention and m/s²** (× −9.80665).
- A within-class dataset-identification check (e.g. bus: ExtraSensory vs NOR-TMD) is required before pooling.
- Excluded foot labels (NOR-TMD outside / inside, Collecty Walk) stay in sessions as unlabelled minutes for context
  and the HMM, never as labels.

**Why.**
- NOR-TMD gives 12,988 clean v1 windows (219 h) in v1's weak classes: bus 70 users, subway 27, train 17, car 13,
  bike 6; clean timing, no duplicates, top-user share ≤ 0.48. Its iOS data are 5 Hz (214 h not clean).
- Collecty: 9,026 clean v1 windows, but ≤ 5 users per class except car, top-user shares 0.75–0.98.
- MobilitApp: gravity removed, 56% of files without sample times, ~69 h instead of the 298 h stated.
- Both usable sets bring only vehicles plus bike / run: a possible dataset shortcut.
- iOS accelerometers report with the opposite sign to Android, in NOR-TMD and ExtraSensory (still minutes: iPhone
  mean z −9.4, Android +9.1 m/s²); v1's features are sign-invariant, raw-axis models are not.

**Limits of the evidence.** A quality check, no models; no candidate has GPS; users are app installations, not
persons; whether NOR-TMD improves the ExtraSensory test users is not measured.

**Evidence.** `experiments/08_candidate-datasets-qc/README.md` (Results, stop-point 2; Conclusion); commercial use
of NOR-TMD and Collecty: `LICENSES.md`.
