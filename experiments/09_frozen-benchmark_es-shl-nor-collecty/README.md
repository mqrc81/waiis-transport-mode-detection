# 09 Frozen benchmark v2: ExtraSensory + SHL + NOR-TMD + Collecty

## Question

Build **`waiis_bench_v2`**, frozen like v1 (experiment 04), from v1 plus the datasets approved in 08: NOR-TMD
(Android) and Collecty. It adds tram as a ninth fine class, a six-class primary score, a second reserved test set
(NOR-TMD, vehicles and rail), and SHL's other phone positions as opt-in rows. v1 stays readable and unchanged, so
05–07 remain reproducible. No models are trained here.

*(A data build, not a model experiment: no MLflow runs.)*

## Plan

*From the brief of 2026-11-02. Not yet approved: stop-point 1 below.*

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

#### 3. NOR-TMD test pick with car (to fix after review)

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

#### 5. Rate rule tolerance: ≥ 24 Hz

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

#### 6. SHL User 1's other positions: not on disk

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

## Conclusion

*Not written yet.*
