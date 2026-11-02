# 09 Frozen benchmark v2: ExtraSensory + SHL + NOR-TMD + Collecty

## Question

Build **`waiis_bench_v2`**, frozen like v1 (experiment 04), from v1 plus the datasets approved in 08: NOR-TMD
(Android) and Collecty. It adds tram as a ninth fine class, a six-class primary score, a second reserved test set
(NOR-TMD, vehicles and rail), and SHL's other phone positions as opt-in rows. v1 stays readable and unchanged, so
05–07 remain reproducible. No models are trained here.

*(A data build, not a model experiment: no MLflow runs.)*

## Plan

*From the brief of 2026-11-02. Approved at stop-point 1 with changes (answers below, in Results).*

**Contents**
- ExtraSensory: v1's dev folds and test users, unchanged. iPhone windows converted to the Android axis convention
  (08's reader rule).
- SHL User 1: training. SHL Users 2–3: rail check, unchanged, never in training (the only rail test with GPS).
- SHL other positions: with a `position` column, excluded by default (`tmd.bench` defaults stay hips only).
  - User 1: for later training experiments, one random position per minute (so SHL's weight doesn't grow).
  - Users 2–3: in the reserved rail check, as a per-position breakdown.
- NOR-TMD, Android only (iOS flagged, 5 Hz): training, plus reserved test users.
- Collecty: training only.
- GPS tables: unchanged (ExtraSensory and SHL only).
- `tmd.bench` takes a version argument; the default stays `waiis_bench_v1` until v2 is approved.

**Labels and scores**
- 9 fine classes, which models train on: still, walk, run, bike, car, bus, train, subway, tram.
- **Primary score: 6 classes merged at scoring time**: still, foot (walk + run), bike, car, bus, rail (train +
  subway + tram).
- Secondary: 9-class (over the classes present in the evaluation set) and family.
- Labels that conflict with v1's classes (NOR-TMD `outside` / `inside`, Collecty `Walk`) are never labels. They stay
  in the sessions as unlabelled usable minutes, for context and the HMM.

**Implementation:** extend v1's pipeline, don't fork it (details: stop-point 1, "How v2 extends 04").
- One build path; one definition of a usable window.
- Flags, not deletions.
- The schema only grows.
- v1's files, paths and schema don't change, and a test proves it.

**Data rules**
- 08's reader rules: every iOS source in m/s² and the Android axis convention (ExtraSensory iPhone too).
- v1's windowing, plausibility rule (U3) and thresholds, X3 and still cap (X4) for all datasets; Collecty on acc +
  gravity. Report any dataset where a rule behaves differently.
- A per-user cap per class for the new datasets.
- `pretraining_minutes` includes the new datasets' unlabelled minutes and excludes all reserved test users.

**Splits**
- `splits/reserved_test_v3.json`: ExtraSensory's 12 test users, SHL Users 2–3 (all positions), and NOR-TMD test
  users with ≥ 5 held-out users for each of bus, subway, train and tram. Prefer users without car or bike. NOR-TMD
  test scores only those four classes.
- Dev folds: ExtraSensory's 5 folds and SHL's day folds unchanged. NOR-TMD and Collecty dev users are dealt into the
  same 5 folds by user, balancing users per class.

**Checks** (README)
- Counts per class × dataset × platform (× position for SHL); users per class; top-user shares before and after
  caps.
- Dataset identification within class from 05's acc features: bus (ExtraSensory vs NOR-TMD), car (all three), train
  (SHL vs NOR-TMD vs Collecty). Descriptive: the baseline for later shortcut checks.
- Platform identification within class (ExtraSensory): the sign conversion must leave 05's features unchanged.

**Comparability:** rescore 05's and 06's final configurations on the six-class set from their logged confusion
matrices (ExtraSensory dev pooled out-of-fold and test; SHL rail check). No retraining. These become v1's reference
numbers in the new primary score.

**Stop-points**
1. Plan: how v2 extends 04, the schema diff, the SHL positions inventory, the cap rule, the NOR-TMD test pick, the
   fold assignment, a dry-run count table. **Wait for OK.**
2. Build at a clean commit with manifests; a decision register like 04's; checks and rescored 05 / 06 in Results.
   Draft a DECISIONS.md entry (every model reads `waiis_bench_v2`; the six-class primary score) and update
   STATUS.md. **Wait for review.**

## Results

### Stop-point 1: plan details and dry run (2026-11-02)

Read first: 04's README, `decision_register.md`, `build.py`; `src/tmd/bench.py`, `resample.py`, `splits.py`;
08's `src/tmd/candidates.py` and readers. Dry run: `uv run python dry_run.py` (reads 08's window tables and v1;
writes `results/tables/dry_run_*.csv`, `splits/proposal_*.csv`). Nothing is built yet.

#### How v2 extends 04

v1's window rules are not in `tmd.bench` (a reader only). They live in **04's `build.py`**:
- per source: `es_user` and `shl_rec`;
- the flags: `stats_columns`, `official_flags`, and the constants `ACC_PLAUSIBLE`, `QUIET_STD`,
  `LABEL_QUIET_CLASSES`;
- the resampling in `tmd.resample`.

08 has a second copy of the same ideas, in `tmd.candidates`. Proposal:

1. **One shared package, `src/tmd/benchbuild/`:**
   - `rules.py`: **the single usable-window definition**:
     - a 15 s window centred in the minute's recording (snippet, or clock minute for streams);
     - no gap > 0.5 s, kept as the longest clean stretch as in v1 (C2);
     - real rate ≥ 25 Hz (from 08);
     - single label (H2);
     - U3, X3, `quiet`;
     - resampling as in v1: interpolate a gap-free stretch to 100 Hz, then `decimate_100_to_25`. This replaces
       08's ±1 s variant (08 deviation 1).
   - `extrasensory.py`, `shl.py`: 04's `es_user` / `shl_rec` moved out of `build.py`, unchanged in behaviour.
   - `nor_tmd.py`, `collecty.py`: 08's trip logic (clock repair, pieces, `time_overlap`), calling `rules.py`.
   - `write.py`: Parquet schemas, `official`, the manifest.
2. **Drivers:**
   - 04's `build.py` becomes a thin driver for v1 (`--version v1`; rate rule off, as v1 had none).
   - 09's `build.py` drives v2.
   - `tmd.candidates` becomes a re-export of `benchbuild.rules`. 08's scripts keep running, but a rerun would
     shift 08's numbers slightly through the resampling change; 08's README gets a note.
3. **Gate before any v2 build:** the refactored code rebuilds v1 into a scratch root, and all 23 data files must
   equal `results/manifests/waiis_bench_v1.json` (sha256). This proves "extend, not fork" and that v1 doesn't
   move.
4. **Tests** (`tests/test_bench_v1_frozen.py`):
   - `tmd.bench.verify("waiis_bench_v1")` is empty;
   - 06's dev out-of-fold probabilities, recomputed through `tmd.bench` with version v1 (H1_v2, seed 0), equal
     the saved `processed/06_gps-acc-baseline_waiis-bench/oof/H1_v2_seed0.npz` exactly. Runtime to be measured;
     if it is long, the test is marked slow.
5. **`tmd.bench`:**
   - `VERSIONS = (v1, v2)`, with `SOURCES` per version (`nor_tmd`, `collecty` in v2 only);
   - the reserved set per version (v2 reads `reserved_test_v3.json`; `tmd.splits.RESERVED_VERSION` changes only
     when v2 is approved);
   - `examples()` applies v2's default filters (hips only, caps, NOR-TMD Android, no excluded labels), each
     undoable by a flag;
   - `pretraining_minutes` covers all v2 sources and excludes every reserved user;
   - the class sets in `tmd.labels`: `CLASSES` (8, v1) stays, plus `FINE_CLASSES` (9) and `SCORE6`.

#### Schema diff (v2 against v1; nothing renamed or removed)

New sources: `nor_tmd/`, `collecty/` with `labels`, `sessions`, `accelerometer`, `stream`. ExtraSensory and SHL keep
their tables. The GPS tables are copied byte-identical from v1 (same sha256, listed in the manifest).

| Table | New column | Type | v1 rows / default | Meaning |
|---|---|---|---|---|
| all `labels`, `accelerometer` | `dataset` | str | `extrasensory` / `shl` | source dataset |
| all `labels`, `accelerometer` | `source_platform` | str | `ios` (iPhone) / `android` | OS of the recording phone |
| all `accelerometer`, ES `snippets` | `sign_converted` | bool | ES iPhone: **true**, else false | iOS → Android axes applied (× −1 after g → m/s²) |
| all `labels` | `cls_fine` | str | = `cls` | 9 fine classes (tram added); `cls` keeps v1's 8 (tram → '') |
| all `labels` | `y_fine` | int8 | = `y` | SHL coding 1–8, tram = 9 |
| all `labels` | `label_raw` | str | ES / SHL source label | the dataset's own label string |
| all `labels` | `label_excluded` | bool | false | conflicting label (NOR-TMD outside / inside, Collecty Walk): unlabelled usable minute |
| all `labels` | `rate_below_25hz` | bool | from `raw_rate_hz` | real accelerometer rate < 25 Hz over the window |
| all `labels` | `user_capped` | bool | false | dropped by the per-user cap per class (new datasets) |
| all `labels` | `duplicate_dropped`, `dup_group` | bool, int | false, −1 | Collecty byte-identical files |
| all `labels` | `clock_piece`, `time_overlap`, `clock_wraps` | int, bool, int | 0, false, 0 | Collecty clock repair |
| SHL `labels`, `accelerometer` | `position` (exists in SHL labels) | str | `hips` | phone position; hand / torso / bag rows added |
| SHL `labels` | `position_default`, `position_draw` | bool | true, true for hips | default = hips; draw = the one random position per User 1 minute |
| new `labels` | `trip`, `session` | str | – | NOR-TMD journey_trip, Collecty trajectory (+ piece) |
| all `labels` | `reason` (exists in ES) | str | SHL: '' | exclusion reason, added where missing |
| `acc_status` (exists) | new values | str | – | `rate_below_25hz`, `gap_over_0.5s`, `mixed_label`, `edge`, `time_overlap`, `no_events` |

- `official` (v2) = v1's definition, plus: `cls_fine` set and not `label_excluded`, not `rate_below_25hz`, not
  `user_capped`, not `duplicate_dropped`, not `time_overlap`, and `position_default`.
- Each of these filters is a column, so `examples(..., include=…)` can undo it.

#### SHL positions inventory (shl-dataset.org, checked 2026-11-02)

| | Positions | Days | Labels | Format | Size (zipped) | On disk? |
|---|---|---|---|---|---|---|
| User 1, 4phones (preview) | hand, torso, bag (+ hips) | 3 days not duplicated (v1's `accelerometer_other_positions`, 5,964 windows) | yes | raw | – | **yes** |
| User 1, challenge 2019/2020 train | bag, hips, torso (2019), hand (2020) | 59 | yes | 5 s frames, 100 Hz | 5.6–6.0 GB per position | no |
| User 1, challenge 2023 train | all 4 | 59 | yes | **raw continuous streams** (acc, gyr, mag, GPS) | **18.0 GB** | no |
| User 1, challenge 2024–26 train | all 4 | 59 | yes | acc / gyr / mag | ~3.0–3.3 GB per position | no |
| User 1, challenge 2019 test | hand | 20 | yes | 5 s frames, shuffled + permutations | 1.75 GB | no |
| Users 2–3, 4phones (preview) | all 4 | the reserved days | yes | raw | – | **yes** |
| Users 2–3, challenge 2020 test | 1 position, not named | ~40 (mixed, partial days) | yes | 5 s frames | 1.74 GB | no |
| Users 2–3, challenge 2023–26 test | 1 or 3 | 28 | **no** ("TBD") | – | 1–2 GB | no |

- **Users 2–3 other positions:** everything labelled for the reserved days is on disk (no download). The rail check
  gets per-position rows (≈ 3 × the 3,015 hips windows; exact count at stop-point 2).
- **User 1 other positions:** on disk only for the 3 preview days. The 59 challenge days need a download over 5 GB.
  My preference is the 2023 training set (18.0 GB; raw streams, closest to our pipeline; 45 GB free). **Your call.**
  Without it, v2 carries User 1's 3 preview days only, as v1 does.
- **Licence:** the SHL download page says "non-profit and research purposes" only, other uses by licence from the
  originators (as in `LICENSES.md`). This applies to v1 already. If WAIIS models are commercial, SHL (User 1 in
  training, Users 2–3 as the rail check) needs that licence. I'm flagging it, not deciding it.

#### Per-user cap per class (new datasets): proposal R2

**Rule R2:** within each (dataset, class), over dev users only:
- each user's windows are capped at the largest cap that keeps the top user's share ≤ 0.25, but never below 60
  windows (1 h);
- where the top user already holds ≤ 0.25, no cap;
- the windows kept are a seeded round-robin over the user's trips (as `tmd.sampling`), seed 0;
- the rest get `user_capped`;
- test users are never capped.

Alternatives compared (`results/tables/cap_options.csv`):
- 2 × the median user (R1) cuts NOR-TMD bus from 7,081 to 4,270, although its top share is only 0.06;
- a fixed 240 windows (R3) leaves Collecty's top users at 0.36–0.59.

| Dataset | Class | Dev users | Windows | Cap | Kept | Top user before → after | Users capped |
|---|---|---|---|---|---|---|---|
| Collecty | bike | 2 | 562 | 60 | 69 | 0.98 → 0.87 | 1 |
| Collecty | bus | 5 | 1,705 | 63 | 253 | 0.75 → 0.25 | 2 |
| Collecty | car | 13 | 3,899 | 651 | 2,604 | 0.43 → 0.25 | 2 |
| Collecty | run | 1 | 196 | 60 | 60 | 1.00 → 1.00 | 1 |
| Collecty | train | 5 | 2,664 | 63 | 252 | 0.84 → 0.25 | 3 |
| Collecty | tram | 4 | 433 | 60 | 173 | 0.62 → 0.35 | 2 |
| NOR-TMD | bike | 7 | 592 | 101 | 405 | 0.48 → 0.25 | 2 |
| NOR-TMD | car | 16 | 804 | 173 | 692 | 0.30 → 0.25 | 2 |
| NOR-TMD | train | 15 | 490 | 116 | 466 | 0.29 → 0.25 | 1 |
| NOR-TMD | bus / subway / tram | 66 / 24 / 17 | 6,345 / 3,360 / 1,144 | none | all | 0.07 / 0.18 / 0.19 | 0 |

- Collecty keeps 3,411 of 9,459 windows (36%). Bike and run (1–2 users) can't reach 0.25; there the 60-window
  floor only limits volume.
- NOR-TMD keeps 12,412 of 12,735 dev windows (97%).

#### NOR-TMD test pick (proposal: 12 installations)

- **Method:** `tmd.splits.pick_test_groups`, seed 0, 200,000 draws, as for v2's ExtraSensory test users:
  - candidates: the 71 of 91 Android installations with no car or bike window;
  - constraint: ≥ 5 held-out users with ≥ 5 clean windows for each of bus, subway, train, tram;
  - among valid subsets, the one whose class shares are closest to n / 91 wins.
- **n = 12** (13%) is the smallest size that meets the constraint (feasible from 10; at 10, 0.15% of random subsets
  pass). Train and tram hit their floor of 5 at every size, because only 13 / 16 candidate users have ≥ 5 windows
  of them. Larger n only adds bus users:

| n | bus | subway | train | tram | car / bike |
|---|---|---|---|---|---|
| **12 (proposed)** | **7 / 63** users, 736 / 6,345 windows | **5 / 22**, 548 / 3,360 | **5 / 12**, 113 / 490 | **5 / 14**, 130 / 1,144 | 0 / 13, 0 / 6 |
| 14 | 11 / 59, 1,024 / 6,057 | 6 / 21 | 5 / 12 | 5 / 14 | 0 |
| 18 | 12 / 58, 1,412 / 5,669 | 6 / 21 | 5 / 12 | 5 / 14 | 0 |

(held out / left for training; users with ≥ 5 windows; windows before caps.)

- The proposed users are in `splits/proposal_nor_tmd_test.csv`. They become `reserved_test_v3.json` only after OK.
- **Thin:** held-out train is 113 windows from 5 users.
- **Installations, not persons:** 152 installations for 101 participants, so one person may sit on both sides. The
  data have no person key.

#### Fold assignment (proposal)

- ExtraSensory and SHL folds are unchanged.
- NOR-TMD's 79 dev installations and Collecty's 14 users are dealt separately into the same 5 folds
  (`tmd.splits.balanced_folds`, seed 0). The matrix balanced is users with ≥ 5 clean windows per class, so folds
  balance *users* per class (`splits/proposal_folds_new.csv`).

| Users with ≥ 5 windows per fold | bike | bus | car | subway | train | tram | run |
|---|---|---|---|---|---|---|---|
| NOR-TMD folds 0–4 | 1 / 1 / 2 / 1 / 1 | 14 / 13 / 14 / 11 / 11 | 3 / 3 / 2 / 2 / 3 | 6 / 4 / 4 / 3 / 5 | 2 / 3 / 2 / 3 / 2 | 3 / 3 / 3 / 2 / 3 | – |
| Collecty folds 0–4 | 0 / 0 / 0 / 1 / 1 | 1 each | 3 / 3 / 2 / 3 / 2 | – | 1 each | 1 / 1 / 1 / 0 / 1 | fold 3 only |

#### Dry-run counts (clean windows, after the proposed caps; `dry_run_counts.csv` has users and platform)

| Dataset | Split | still | walk | run | bike | car | bus | train | subway | tram |
|---|---|---|---|---|---|---|---|---|---|---|
| ExtraSensory | dev | 92,730 | 13,109 | 463 | 3,118 | 8,028 | 1,348 | 190 | – | – |
| ExtraSensory | test | 19,286 | 3,682 | 81 | 876 | 2,130 | 287 | 52 | – | – |
| SHL User 1 (hips) | dev | 3,366 | 3,011 | 1,030 | 2,842 | 4,058 | 3,028 | 3,353 | 2,388 | – |
| SHL Users 2–3 (hips) | rail check | 487 | 427 | 45 | 199 | 339 | 150 | 360 | 358 | – |
| NOR-TMD (Android) | dev | – | – | – | 405 | 692 | 6,345 | 466 | 3,360 | 1,144 |
| NOR-TMD (Android) | test | – | – | – | – | – | 736 | 113 | 548 | 130 |
| Collecty | dev | – | – | 60 | 69 | 2,604 | 253 | 252 | – | 173 |

- **Six-class view** (dev, all sources): rail = 190 + 3,353 + 2,388 + 466 + 3,360 + 1,144 + 252 + 173 = 11,326
  windows, against 5,931 in v1 (ExtraSensory + SHL User 1). Bus: 10,974 against 4,376.
- **The rate rule on ExtraSensory:** the shared window definition adds "real rate ≥ 25 Hz", which v1 lacked. It
  flags 6 of 145,380 official windows, all from reserved test user 1DBB0F6F (Android): 3 still, 1 bus, and **2 of
  its 15 train windows** (test train 52 → 50). Options: apply it to ExtraSensory too (one definition; my
  preference), or exempt ExtraSensory to keep v1's test counts.

#### Comparability (stop-point 2): what is logged

- 05's and 06's final runs log their confusion matrices as data (`tables.json`: `cm_test_es_all_class`, per-platform,
  SHL, dev `cm_es_class`) with the class lists, so the six-class rescoring needs no retraining.
- The point estimates come from the merged confusion matrices. Bootstrap CIs need per-user predictions: 05 / 06's
  dev out-of-fold probabilities are saved (`processed/0{5,6}_…/oof/*.npz`), test predictions possibly not. I'll
  give CIs where the predictions exist and say where they don't.

#### Questions for OK

1. The `benchbuild` refactor with the v1 rebuild gate (stop-point 1, "How v2 extends 04").
2. Cap rule R2 (or R1 / R3, or other numbers).
3. NOR-TMD test: 12 installations (or 14 / 18).
4. Fold assignment as proposed.
5. The rate rule ≥ 25 Hz applied to ExtraSensory (−2 test train windows) or not.
6. SHL User 1 other positions: download the 2023 training set (18.0 GB), another variant, or none.
7. SHL licence (non-profit / research only) for WAIIS.
8. Not in the brief: 07 asked for a check of ExtraSensory run labels in the next benchmark version (dev run
   recall 0.26 at every n against 0.74 on test; 04's open question on `label_implausible` removing 60% of iPhone
   run). Include it in 09's checks (descriptive, no relabelling), or leave it for later?

### Stop-point 1 answers (2026-11-02) and follow-up

1. **Refactor approved**, as a package `tmd.bench` (not a separate package): the reader API stays importable
   unchanged, the build code and the per-dataset readers move inside it. Sub-packages elsewhere only where the
   grouping is obvious. Every old import path keeps working, so 05–08 run unchanged. The reorganisation is its own
   commit, before any v2 code, gated by the v1 rebuild hashes and 06's dev out-of-fold probabilities.
2. **Cap R2 approved** for NOR-TMD and Collecty. **SHL User 1 stays uncapped.** Its weight (1 vs down-weighted)
   belongs in the first model experiment on v2, as an ablation. SHL joins the dataset-identification check.
3. **NOR-TMD test:** car users allowed: ≥ 3 held-out car users (≥ 5 clean windows each), still no bike users, and car
   scored on the NOR-TMD test. Show the table before fixing it. Check whether any field links installations to
   persons: if yes, linked installations stay on the same side; if not, it is a limitation.
4. **Folds approved.**
5. **Rate rule:** one definition everywhere, real rate **≥ 24 Hz** (tolerance for nominal-25 Hz data with jitter).
   Report the ExtraSensory windows flagged at 24 and 25 Hz and the test train count at 24.
6. **SHL User 1's other positions:** said to be on disk in `release/User1` for all days. Include them as opt-in rows
   (position column; default hips only; one random position per minute when used). First quote 04's register (S3,
   H6). *Checked: not on disk; see 6 below.*
7. **Licences:** no action; SHL's non-commercial terms stay open in `LICENSES.md`.
8. **Run labels:** a descriptive check in 09: run windows per user (dev, test); motion energy and step cadence of
   ExtraSensory dev run vs test run vs SHL run; where 05 / 06's out-of-fold predictions put dev run minutes, per user.
   If some users' run looks like another activity, propose a flag (not a deletion) at stop-point 2.

#### 1. Reorganisation (done; revised after review)

**Moves** (first commit `223a5aa`; revised on review, second commit below):

| Old module | New module | Old path |
|---|---|---|
| `src/tmd/bench.py` (reader API) | `src/tmd/bench/__init__.py` (same content) | `tmd.bench`, unchanged |
| `experiments/04_…/build.py` (v1 build code) | `src/tmd/bench/build.py` | 04's `build.py` is a thin driver calling it (same CLI) |

- **Revised:** the per-dataset readers (`tmd.extrasensory`, `tmd.shl`, `tmd.nor_tmd`, `tmd.collecty`) and 08's
  window rules (`tmd.candidates`) were first moved into `tmd.bench.sources` / `tmd.bench.rules`, with alias modules
  at the old paths. On review that meant two files per reader. Inside `tmd.bench` and with old scripts unchanged,
  one file per reader is not possible, so they **moved back to `src/tmd/`** (byte-identical to before the
  reorganisation), and `tmd.bench.build` imports them from there (answer of 2026-11-02).
- **Other sub-packages:** none. The obvious candidate (`features`, `acc_features`, `gps_features` → a `features`
  package) has the same problem: it needs alias files or edits to 02–08's imports. `tmd.gps` and `tmd.resample` are
  used by both model and build code.
- `tmd.bench.build` gets optional `--tables-dir` / `--log-dir` (defaults: 04's folders).
- `tmd.paths` reads `$TMD_MLFLOW_DB` / `$TMD_MLARTIFACTS` (defaults as before), mirroring `$TMD_DATA`, so the gate's
  chain logs to a scratch MLflow store.

**Gate** (`gate_v1.py`, full output in `logs/gate_v1.log`). It passed after the first commit and again, in full,
after the revision:

```
TMD_DATA=<scratch>/data uv run python experiments/04_frozen-benchmark_es-shl/build.py --workers 4 \
    --tables-dir <scratch>/tables --log-dir <scratch>/logs        # v1 rebuilt from the moved code (raw = symlink)
uv run python gate_v1.py --scratch <scratch>                     # checks; the 05 -> 06 chain against the rebuild
```

1. `verify(waiis_bench_v1)`: intact.
2. **Rebuild:** all 23 data files have the reference sha256. The 32 count tables the build writes equal 04's
   committed tables byte for byte. `decision_details.csv` comes from 04's `register_numbers.py`, so it isn't
   compared.
3. **06's chain:** 05's features → B → F_fb; 06's gps-v1 and gps-v2 features → A → G_v2 → H1_v2, recomputed from
   the rebuild with a scratch MLflow store (seed 0, the recorded commands without `--vs`):
   - the 5 feature tables are identical;
   - the out-of-fold `p` / `raw` of B, F_fb, A, G_v2 and **H1_v2 are identical to the saved files, bit for bit**;
   - the only difference is in A's `info` metadata: the saved file predates the `gps_version` key 06's code added
     later.
   - The real MLflow store is untouched (05: 54 runs, 06: 28 runs). The six 05 / 06 logs the chain rewrites are
     restored from git each time.
4. `pytest tests`: 65 passed. The new `tests/test_bench_v1_frozen.py` checks `verify(v1)` and the sha256 of the 5
   out-of-fold files the gate matched; the recomputation itself stays in `gate_v1.py`.
- **Deviation:** the first chain run missed 06's plain `make_gps.py` (gps-v1, which row A reads). It was added, and
  that run continued from the step (`--start 3`). The re-gate after the revision ran the whole chain from step 0.
- **The scratch v1 rebuild** (6.7 GB, in the session scratchpad) is kept until the gate has passed after the v2
  changes. Reminder at stop-point 2: delete it.

#### 3. NOR-TMD test pick with car (fixed 2026-11-02: device-model pick, 8 models, 14 installations)

- **Can installations be linked to persons?** No field does it reliably (`results/tables/nor_tmd_installations.csv`):
  - there is no person key and no area or city field;
  - each installation has exactly one device model;
  - recording periods overlap heavily (2021-03-25 – 2022-02-22, mostly Sep–Oct 2021).
  - The only weak link is the **device model**: 27 models are shared by 85 of the 108 Android installations (DN2103
    by 9). There are 105 same-model pairs where one ends before the other starts, many on consecutive days. That
    fits reinstalls, but equally several participants with the same popular phone.
  - **Conservative option:** keep all installations of a device model on the same side. This rules out one person
    on both sides as long as nobody changed phones.
- **Rerun:** `pick_test_groups`, seed 0, 200,000 draws.
  - Constraints: ≥ 5 held-out users (≥ 5 clean windows) for bus, subway, train, tram, and **≥ 3 for car**; no
    bike users.
  - Score: class shares (bike, bus, car, subway, train, tram) closest to the test fraction.
  - Cells are held out / left for training, as users with ≥ 5 windows, then windows (before caps).

| Pick | Installations | bus | subway | train | tram | car | bike |
|---|---|---|---|---|---|---|---|
| per installation, n = 12 | 12 | 9 / 61 u, 1,038 / 6,043 w | 5 / 22 u, 640 / 3,268 w | 5 / 12 u, 89 / 514 w | 5 / 14 u, 222 / 1,052 w | 3 / 10 u, 102 / 702 w | 0 / 6 |
| per installation, n = 14 | 14 | 10 / 60 u, 1,267 / 5,814 w | 5 / 22 u, 617 / 3,291 w | 5 / 12 u, 88 / 515 w | 5 / 14 u, 257 / 1,017 w | 3 / 10 u, 138 / 666 w | 0 / 6 |
| per installation, n = 16 | 16 | 10 / 60 u, 1,107 / 5,974 w | 7 / 20 u, 741 / 3,167 w | 5 / 12 u, 112 / 491 w | 5 / 14 u, 212 / 1,062 w | 3 / 10 u, 114 / 690 w | 0 / 6 |
| **by device model, 8 models (proposed)** | **14** | **11 / 59 u, 1,735 / 5,346 w** | **5 / 22 u, 649 / 3,259 w** | **5 / 12 u, 110 / 493 w** | **5 / 14 u, 245 / 1,029 w** | **4 / 9 u, 143 / 661 w** | 0 / 6 |
| by device model, 10 models | 20 | 17 / 53 u, 1,849 / 5,232 w | 5 / 22 u, 755 / 3,153 w | 5 / 12 u, 118 / 485 w | 5 / 14 u, 284 / 990 w | 4 / 9 u, 143 / 661 w | 0 / 6 |
| by device model, 12 models | 21 | 17 / 53 u, 1,661 / 5,420 w | 5 / 22 u, 1,001 / 2,907 w | 5 / 12 u, 160 / 443 w | 6 / 13 u, 334 / 940 w | 3 / 10 u, 250 / 554 w | 0 / 6 |

- **Proposal:** by device model, 8 models = 14 installations. That is the same size as per installation n = 14,
  with the linkage guard, and gives 4 held-out car users instead of 3.
- Car training users drop from 13 to 9 (661 windows before caps). Train and tram stay at their floor of 5 at
  every size.

#### 5. Rate rule tolerance: ≥ 24 Hz (approved 2026-11-02, everywhere)

| ExtraSensory threshold | Windows flagged | of which official | Official by class | Split | Test train |
|---|---|---|---|---|---|
| < 25 Hz | 15 (1 user) | 6 | still 3, train 2, bus 1 | all test (1DBB0F6F) | 52 → 50 |
| **< 24 Hz** | **12 (1 user)** | **5** | **train 2, still 2, bus 1** | all test (1DBB0F6F) | **52 → 50** |

- Both train windows are at 20–21 Hz, so the tolerance doesn't save them.
- The cause (from v1's own columns): ExtraSensory snippets are a fixed 800 samples. This phone usually ran at ~50
  Hz (16 s snippets), but in these windows the 800 samples span 32–40 s, so the phone delivered samples slowly.
  v1's only guard was the 40 s span limit (C1), which these windows just miss.
- 02–05 mention 1DBB0F6F only as a test / train user (and the user with the most snippets under 15 s, 57%, at ~80 Hz
  at other times), never with a rate exception.

#### 6. SHL User 1's other positions: not on disk (answer: option a)

04's decision register on SHL positions:
- **S3** | SHL hips: User 1 training source, Users 2–3 rail check | metro and most train data | 25,441 examples |
  – | b: one phone, one position |
- **H6** | Hips stream, other positions, 1 Hz track stored | pre-training and position studies | 59.8 M samples | yes
  (extras) | – |
- *Dropped, not flagged:* "SHL hand, torso and bag phones, except User 1's three 4phones days."

**Checked on disk (2026-11-02):** `<data root>/raw/shl/release/User1` (there is no `data/shl`) has 91 day folders
with **only `Hips_*` files** (`Hips_Motion.txt` in 90), plus labels and video files. No `Hand_*`, `Torso_*` or
`Bag_*` file and no archive is anywhere under `raw/shl` except `User1_4phones` (5 day folders, 3 not duplicated in
`release/User1`), `User2_4phones` and `User3_4phones`. So hand / torso / bag for User 1 exist only for the 3 preview
days, the ones v1 already stores. **Waiting on a decision:**
- (a) v2 carries those 3 days as opt-in rows (position column, `position_draw`), as v1 did, but in the labels
  schema;
- (b) download User 1's 59 challenge days (2023 training set, 18.0 GB, the only raw-stream variant), then (a) for
  all of them.
Users 2–3's other positions are on disk for their reserved days and go into the rail check either way.

### Answers after the follow-up (2026-11-02)

1. **Readers:** back to `src/tmd/`, one file each (section 1 above; re-gated, commit `51cf3f3`). No other
   sub-packages.
2. **Reserved set fixed: `splits/reserved_test_v3.json`** (`pick_test.py`; refuses to overwrite):
   - ExtraSensory's 12 test users and SHL Users 2–3 (all days and positions), copied unchanged from
     `reserved_test_v2.json`;
   - NOR-TMD: the device-model pick, 8 models (Nokia 8.1, ONEPLUS A6003, SM-A505FN, SM-A725F, SM-G970F, SM-N981B,
     XQ-AS52, moto g 5G plus), 14 installations. Those are all the installations of those models, of any OS
     (checked), so the guard is complete.
   - **NOR-TMD test scores bus, subway, train, tram and car.**
   - **Linkage guard:** the data have no person field. Installations of one device model are kept on one side, which
     rules out one person on both sides unless someone changed phones (a limitation, not checkable).
   - **Thin:** held-out train is **110 windows from 5 users**. Held-out car is 143 windows from 4 users.
   - Counts at the pick were from 08's window tables (≥ 25 Hz, 08's resampling); v2's counts differ slightly
     (≥ 24 Hz, v1's resampling).
3. **Rate ≥ 24 Hz everywhere.** In ExtraSensory it flags 12 windows (5 official), all from test user 1DBB0F6F;
   **test train = 50**.
4. **SHL positions: option (a).**
   - User 1's other positions are the 3 preview days, as opt-in rows. The default stays hips only; one random
     position per minute when used.
   - Users 2–3's other positions go into the rail check as a per-position breakdown.
   - Added to the open decisions for v2: if the first model on v2 shows a clear rail-recall drop off the hip in
     that breakdown, check whether the 2023 training set's days overlap the 91 hip days, and consider it for a
     v2.1.
5. **Scratch v1 rebuild:** kept until the gate has passed after the v2 changes; reminder at stop-point 2.
- **Dev folds** (`splits/folds_nor_tmd_collecty.csv`, `folds.py`): the approved method, rerun for the final NOR-TMD
  test set (77 NOR-TMD dev installations, 14 Collecty users; `results/tables/folds_users_per_class.csv`).
  - NOR-TMD: 11–12 bus users per fold, subway 3–6, train 2–3, tram 2–4, car 1–2, bike 1–2.
  - Collecty: bus, train, 1 each; car 2–3; tram in 4 folds; bike in folds 3–4; run in fold 3 only.
  - Installations with no clean labelled window get fold −1 in the build (never scored).

### Stop-point 2: build, checks, rescoring (2026-11-02)

**Commands** (from this folder; logs in `logs/`, tables in `results/tables/`):

```
uv run python pick_test.py                 # splits/reserved_test_v3.json (once)
uv run python folds.py                     # splits/folds_nor_tmd_collecty.csv (once)
uv run python build.py --workers 3         # <data>/processed/waiis_bench_v2/ (~4.5 min), results/tables/bench_v2/
uv run python checks.py counts|identify|platform|rescore|run   # results/tables/checks/
uv run python gate_v1.py --scratch <dir>   # v1 unchanged (README, stop-point 1 follow-up)
uv run pytest tests                        # 86 passed
```

**Build**
- **`waiis_bench_v2`:** `<data>/processed/waiis_bench_v2/`, 32 Parquet files, 5.63 GB.
  - Built at commit **`c0956d6`, clean** (`git_dirty: false`).
  - Reference manifest: `results/manifests/waiis_bench_v2.json`; `tmd.bench.verify("waiis_bench_v2")` is empty.
- **Reproducible:** a second build (first one from `4167a7d`, discarded because the manifest was marked dirty by my
  stdout redirect) gave all 32 data files with the same sha256.
- **15 files are byte-identical to v1:** all GPS / location tables, the sessions tables, SHL's streams. GPS is
  unchanged as planned.
- **v1 unchanged:** the gate passed again after every v2 code change (v1 rebuild hash-identical, 06's dev
  out-of-fold probabilities bit-identical); `verify(v1)` is intact.
- **Rules and their effects:** `decision_register.md` (V1–V6, A1–A3, R1–R4, K1–K2, L1–L4, C1–C2, P1–P4, F1–F2).

**Official examples** (`checks/counts_class_source_platform.csv`; SHL hips only, the default)

| Source | Split | Platform | still | walk | run | bike | car | bus | train | subway | tram | all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ExtraSensory | dev | Android | 36,340 | 4,852 | 394 | 1,464 | 1,831 | 550 | 97 | – | – | 45,528 |
| ExtraSensory | dev | iOS | 56,390 | 8,257 | 69 | 1,654 | 6,197 | 798 | 93 | – | – | 73,458 |
| ExtraSensory | test | Android | 6,243 | 2,310 | 43 | 230 | 765 | 239 | **50** | – | – | 9,880 |
| ExtraSensory | test | iOS | 13,041 | 1,372 | 38 | 646 | 1,365 | 47 | – | – | – | 16,509 |
| SHL User 1 | dev | Android | 3,366 | 3,011 | 1,030 | 2,842 | 4,058 | 3,028 | 3,353 | 2,388 | – | 23,076 |
| SHL Users 2–3 | rail check | Android | 487 | 427 | 45 | 199 | 339 | 150 | 360 | 358 | – | 2,365 |
| NOR-TMD | dev | Android | – | – | – | 405 | 405 | 5,356 | 470 | 3,288 | 1,033 | 10,957 |
| NOR-TMD | **test** | Android | – | – | – | – | 158 | 1,908 | 128 | 738 | 289 | 3,221 |
| Collecty | dev | Android | – | – | 60 | 69 | 2,596 | 253 | 252 | – | 153 | 3,383 |

- ExtraSensory is v1 minus the 5 windows under 24 Hz (all test user 1DBB0F6F: still 2, bus 1, train 2).
- **Six-class view, dev** (all sources; v1 = ExtraSensory + SHL User 1 in parentheses): rail 11,127 (5,931: SHL
  5,741, NOR-TMD 4,791, Collecty 405, ExtraSensory 190), bus 9,985 (4,376), car 15,087 (12,086), bike 6,434 (5,960),
  foot 17,673 (17,613), still 96,096 (96,096).
- **Users per class and top-user share** (`checks/users_per_class.csv`):
  - NOR-TMD bus 74 users (top 0.06), subway 30 (0.15), tram 22 (0.16), train 20 (0.20), car 17 (0.18), bike 7
    (0.25);
  - Collecty (after caps): car 13 (0.25), bus 5 (0.25), train 5 (0.25), tram 4 (0.39), bike 2 (0.87), run 1;
  - ExtraSensory and SHL as in v1.
- **Caps (R2)** (`checks/caps.csv`):
  - NOR-TMD: bike 592 → 405, car 661 → 405, train 493 → 470; bus / subway / tram uncapped;
  - Collecty: car 3,894 → 2,596, bus 1,701 → 253, train 2,643 → 252, tram 411 → 153, bike 562 → 69, run 196 → 60.
  - NOR-TMD car loses the most: 4 of its car users went to the test set, and the 11 left are more concentrated
    (top share 0.36 before the cap).
- **SHL positions** (`checks/shl_positions.csv`):
  - User 1's other positions: 1,988 windows each for hand, torso and bag (3 preview days); `position_draw` keeps
    hips on 35,354 of 36,825 hips minutes.
  - Users 2–3: hand, torso, bag and hips have the same per-class counts (2,365 per position, walk 428 vs 427), so
    the rail check gets a per-position breakdown with equal support.
- **Unlabelled usable minutes from excluded labels:** NOR-TMD outside 1,636 and inside 401, Collecty Walk 3,529.
  Also unlabelled: NOR-TMD boat 3,715 and e-scooter 198, Collecty e-scooter 388.

**Plausibility rules across datasets** (`checks/flag_rates.csv`, % of windows with a class)
- **U3** (median |acc| outside 7–20 m/s²): ≤ 0.81% everywhere except SHL User 1 run (1.6%) and Collecty run
  (0.5%); 0% in every other NOR-TMD and Collecty class. It behaves the same everywhere.
- **X3 / `quiet`** behaves differently by dataset:
  - in ExtraSensory, `quiet` is 9–27% of moving-class windows (walk 15%, run 27%, bike 10%; X3 removes these),
    versus ≤ 0.2% for walk / bike in SHL and NOR-TMD;
  - in the vehicles, `quiet` is 1–8% everywhere (NOR-TMD car 7.5%, ExtraSensory car 8.2%).
  - Reading: in-the-wild phone placement (bag, table) in ExtraSensory, not a rule defect.

**Dataset identification within class** (`checks/dataset_identification.csv`; 05's 580 features, LightGBM,
cross-validated by group; groups = users, SHL User 1's recording days; dev users only; up to 2,000 windows per
source; descriptive, the baseline for later shortcut checks)

| Class | Sources | Windows | Groups | Macro AUC | Balanced accuracy (chance) |
|---|---|---|---|---|---|
| bus | ExtraSensory / NOR-TMD / Collecty / SHL | 1,348 / 1,537 / 253 / 2,000 | 21 / 62 / 5 / 40 days | 0.824 | 0.543 (0.25) |
| car | ExtraSensory / NOR-TMD / Collecty / SHL | 1,769 / 405 / 1,056 / 2,000 | 38 / 11 / 13 / 27 days | 0.818 | 0.519 (0.25) |
| train | SHL / NOR-TMD / Collecty / ExtraSensory | 2,000 / 470 / 252 / 190 | 27 days / 15 / 5 / 3 | 0.835 | 0.450 (0.25) |
| subway | SHL / NOR-TMD | 2,000 / 1,312 | 16 days / 25 | **0.963** | 0.891 (0.50) |
| tram | NOR-TMD / Collecty | 1,033 / 153 | 17 / 4 | 0.521 | 0.505 (0.50) |

- **Every vehicle class except tram is clearly identifiable by source from the accelerometer features alone**
  (AUC 0.82–0.96). Subway is the extreme: SHL's one phone in London against NOR-TMD's Oslo metro.
- SHL is grouped by day, so its numbers also contain "one person" (identifiable across days); the other sources are
  grouped by user.

**Platform identification within class, ExtraSensory** (`checks/platform_identification.csv`; dev, up to 2,000
windows per platform)

| Class | Windows | iPhone share | 05's features v1 = v2 | Platform AUC v1 | Platform AUC v2 |
|---|---|---|---|---|---|
| still | 3,983 | 0.50 | identical | 0.819 | 0.819 |
| walk | 3,395 | 0.58 | identical | 0.636 | 0.636 |
| car | 3,567 | 0.49 | identical | 0.607 | 0.607 |
| bike | 3,118 | 0.53 | identical | 0.425 | 0.425 |

- **The sign conversion leaves 05's features exactly unchanged**, so every v1 result and the platform AUCs stay
  the same.
- Still is identifiable by platform (0.82) with or without the conversion: iPhone vs Android still windows differ
  beyond the axis sign (noise floor, rest position).

**v1's reference numbers in the six-class score** (`checks/rescore_six_class.csv`, `rescore_per_seed.csv`,
`rescore_dev_oof_per_seed.csv`; no retraining; mean of seeds 0–2)

| Set | 05: six-class | 05: 8-class (check) | 06: six-class | 06: 8-class (check) |
|---|---|---|---|---|
| ExtraSensory dev, pooled OOF | **0.575** [0.515, 0.619] | 0.551 | **0.642** [0.583, 0.687] | 0.613 |
| SHL User 1, day folds | 0.946 | 0.934 | 0.955 | 0.949 |
| ExtraSensory test, all | **0.642** | 0.667 | **0.686** | 0.707 |
| — iPhone / Android | 0.751 / 0.642 | 0.762 / 0.665 | 0.752 / 0.654 | 0.764 / 0.682 |
| SHL Users 2–3, rail check | 0.920 | 0.795 | 0.944 | 0.847 |

- The 8-class column reproduces STATUS.md's numbers exactly, which validates the merging.
- **Deviation:** 06's dev runs logged no confusion matrix as data (only figures). Dev scores therefore come from
  the saved out-of-fold predictions (`processed/0{5,6}_…/oof/*.npz`) for both models, with a user-bootstrap 95% CI
  (seed 0).
  - For 05 both routes exist: 0.578 from the logged matrices vs 0.575 from the predictions. The logged matrices
    cover only the classes present (no subway), so windows *predicted* as subway drop out of them. Merged into
    rail, they count.
  - The test numbers come from the logged matrices (test predictions weren't saved), so they share that small gap.
- Six-class is lower than 8-class on test: run (high test recall) merges into foot with walk, and the macro
  average loses that class.

**Run labels** (item 8; `checks/run_energy_cadence.csv`, `run_per_user.csv`; v1's labels)

| Set | Windows | Users | std \|acc\| p25 / p50 / p75 [m/s²] | Cadence p25 / p50 / p75 [Hz] |
|---|---|---|---|---|
| ExtraSensory dev run | 463 | 17 | 0.36 / 2.23 / 6.37 | 1.67 / 2.33 / 2.73 |
| ExtraSensory test run | 81 | 5 | 5.09 / 6.43 / 11.64 | 2.47 / 2.53 / 2.87 |
| SHL User 1 run | 1,030 | 1 | 9.52 / 10.70 / 12.17 | 2.60 / 2.67 / 2.67 |

- **One dev user, 797D145F (Android), holds 283 of the 463 dev run windows (61%), and they look like walking or
  standing:**
  - median std |acc| 0.95 m/s², cadence 1.87 Hz (dev walk median 1.72 m/s², 1.93 Hz);
  - 05 / 06 predict walk 63–64% and still 32–36% for them, never run.
- Smaller dev users look similar: B9724848 (13 windows, 0.99 m/s²), A76A5AF5 (8, 1.09), 0A986513, 40E170A7.
- Users whose run looks like running (std 7–12 m/s², 2.7–2.8 Hz) are recalled at 79–100%: 74B86067 (88 windows),
  5119D0F8, 59818CD2, 4FC32141, F50235E0.
- The test run windows come mostly from two such users (098A72A5, 0BFC35E2). **This explains 07's dev run recall
  of 0.26 against 0.74 on test.**
- **Flag proposal (not applied):** `run_low_energy`: a run window whose std |acc| is below ExtraSensory dev walk's
  median (1.73 m/s², i.e. it moves less than a typical walk window). A flag in `labels`, not a deletion; whether
  `official` excludes it is the decision. Options (thresholds from reference distributions, not fitted to the
  suspect users):

| Rule | Dev run flagged | Users | Test run flagged | Users | SHL run flagged | 797D145F |
|---|---|---|---|---|---|---|
| **std < ExtraSensory dev walk p50 (1.73) (proposed)** | 202 / 463 | 9 | 10 / 81 | 4 | 0 / 1,030 | 175 / 283 |
| std < walk p75 (3.10) | 278 / 463 | 15 | 17 / 81 | 5 | 0 | 226 / 283 |
| std < walk p50 and cadence < SHL run p5 (2.53 Hz) | 128 / 463 | 7 | 4 / 81 | 3 | 0 | 106 / 283 |
| std < SHL run p1 (6.84) | 360 / 463 | 16 | 53 / 81 | 5 | – | – |

**Deviations from the plan** (all recorded here and in the register)
1. **SHL other positions** are separate `accelerometer_other_positions` tables (as v1's User 1 table), not rows in
   `labels`, so the `(user, ts)` key stays unique. The labels table carries `position_default` / `position_draw`.
2. **Duplicate user-minutes** are resolved by keeping one row per user-minute (R3): NOR-TMD 40, Collecty 6,383
   (mostly sawtooth clock pieces). The dropped rows are not flags. The keeper rule (an ok window first, then the
   earlier trip) also makes Collecty's within-user duplicate files disappear with their twin's rows (checked: 0
   rows left from the 11 dropped copies).
3. **Rescoring of dev** from saved predictions instead of logged matrices (above).
4. **Window rate for ExtraSensory** is the snippet's real rate (v1's `raw_rate_hz`, over the hole-trimmed snippet,
   ≥ 15 s), not a rate over the 15 s window. ExtraSensory's raw timestamps are not stored; the snippet is one
   recording of ~16–40 s.
5. **`stretch_too_short`:** a window that passes the gap rule but whose stretch's 25 Hz grid ends before 375
   samples (NOR-TMD 4, Collecty 18), never official.
6. **The first v2 build** (commit `4167a7d`) is discarded: its manifest was marked dirty by my stdout redirect, and
   its tables went to `results/tables/build/`, which `.gitignore` matches. Rebuilt at `c0956d6` (same data files).

**Open decisions for v2 and the first model on it**
1. **Run flag:** adopt `run_low_energy` (proposed rule above) as a flag, and whether `official` excludes it.
2. **SHL User 1's weight** (1 vs down-weighted) as an ablation in the first model on v2 (approved). It holds 79% of
   dev train windows and 66% of run.
3. **Dataset shortcut:** the within-class identification baseline is high (AUC 0.82–0.96). The first v2 model
   should report whether its predictions depend on the source, e.g. per-source recall, and the identification AUC
   of its own representation.
4. **SHL other positions:** if the first model on v2 shows a clear rail-recall drop off the hip in Users 2–3's
   per-position breakdown, check whether the 2023 training set's days overlap the 91 hip days, and consider it for
   a v2.1 (item 6).
5. **Thin test cells:** NOR-TMD test train 128 windows from 5 users, car 158 from 4; ExtraSensory test train 50
   from 2 users. Per-class results need user-bootstrap CIs.
6. **Licence:** SHL's non-commercial terms stay open (`LICENSES.md`).
7. **Scratch v1 rebuild:** the gate has passed after the v2 changes. The scratch copy (6.7 GB, plus the gate's
   recomputed 05 / 06 files, in the session scratchpad) and the discarded first build's outputs can be deleted.
   **Your OK, then I delete them.**

## Conclusion

**Draft — not approved.**

**`waiis_bench_v2` is built and frozen:** `<data>/processed/waiis_bench_v2/`, 32 Parquet files, 5.63 GB, from clean
commit `c0956d6`.
- Reference manifest: `results/manifests/waiis_bench_v2.json`; a second build reproduced every file.
- Read it with `tmd.bench` and `version="waiis_bench_v2"`; the default stays v1 until v2 is approved.
- Every rule is in `decision_register.md`.

| Source | Official examples | Role |
|---|---|---|
| `extrasensory` | 145,375 (59 users) | dev folds + 12 reserved test users (as v1; iPhone in the Android axis convention; 5 windows under 24 Hz flagged) |
| `shl_user1` | 23,076 (+ 5,964 other-position windows, opt-in) | training (uncapped) |
| `shl_user23` | 2,365 (+ 9,049 other-position windows) | rail check, per position |
| `nor_tmd` | 14,178 (91 installations): dev 10,957, test 3,221 | training + second reserved test (bus, subway, train, tram, car) |
| `collecty` | 3,383 (14 users), after caps | training |

- **What v2 adds:** dev rail 11,127 windows against v1's 5,931 (NOR-TMD 4,791), bus 9,985 against 4,376, tram as a
  ninth class, and a vehicle / rail test set from 14 unseen installations.
- **What it doesn't add:** still, walk, iOS or GPS from the new sources. Those remain ExtraSensory + SHL only.
- **v1 is unchanged and reproducible:** after every change, v1 rebuilds hash-identical and 06's dev out-of-fold
  probabilities are bit-identical. The iOS sign conversion leaves 05's features unchanged.
- **v1's reference numbers in the six-class primary score:** 05 test 0.642 (dev 0.575), 06 test 0.686 (dev 0.642),
  rail check 0.920 / 0.944.
- **Risks for the first model on v2:**
  - **dataset shortcut:** source identification within class reaches AUC 0.82–0.96 (tram 0.52);
  - SHL User 1 carries 79% of dev train and 66% of run (its weight: an ablation);
  - thin test cells (NOR-TMD train 128 windows / 5 users, car 158 / 4; ExtraSensory train 50 / 2).
- **Run labels:** 07's dev-vs-test gap comes mostly from one ExtraSensory dev user whose "run" moves like walking;
  `run_low_energy` is proposed as a flag.
- **Next:** decide the run flag; then the first model on v2 with SHL's weight as an ablation and a per-source /
  shortcut report.
