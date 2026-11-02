# Decision register: `waiis_bench_v2`

`waiis_bench_v2` = `waiis_bench_v1` (04's decision register applies unchanged unless a row below says otherwise) plus
NOR-TMD (Android) and Collecty, 9 fine classes, a six-class primary score and reserved set v3. Built at commit
`6d51dc2` (clean; with the run flag L5); 32 Parquet files, 5.63 GB; manifest `results/manifests/waiis_bench_v2.json`.

- Code: `tmd.bench.build` (v1 section unchanged in effect, v2 section), window rules `tmd.candidates`.
- Count tables: `results/tables/bench_v2/` (build), `results/tables/checks/` (checks).
- Risk: a = leakage or shortcut, b = gap to WAIIS data, c = test-set bias.

## Sources

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| V1 | ExtraSensory and SHL through v1's code, then v2's columns | one build path; v1 unchanged | 15 of 32 files byte-identical to v1 (all GPS, sessions, SHL streams) | – | – |
| V2 | **NOR-TMD, Android only**, for training and a reserved test | bus / subway / train / tram from many users (08) | 14,178 official, 91 installations | `source_platform` | b: Oslo / Bodø, 2021 |
| V3 | NOR-TMD iOS kept as rows, never official: 5 Hz | rate rule (R1) | 13,258 iOS minutes, 0 windows | `rate_below_24hz`, `acc_status` | – |
| V4 | **Collecty, training only** | car, some train / bus (08) | 3,383 official, 14 users | – | b: 1–5 users per class except car |
| V5 | SHL other positions as opt-in rows | position diversity without more SHL weight | User 1: 5,964 windows (3 days), Users 2–3: 9,049 | `position_default`, `position_draw`, separate tables | b: one phone model |
| V6 | MobilitApp not used | gravity removed, untimed files (08) | – | – | – |

## Units & axes

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| A1 | **iOS to the Android axis convention and m/s² (× −9.80665)**, ExtraSensory iPhone included | one convention; raw-axis models would see the sign as a platform tell | 220,961 ExtraSensory iPhone minutes; NOR-TMD iOS | `sign_converted` (undo: × −1) | – |
| A2 | Collecty: total acceleration = linear + gravity | gravity included, as v1 | all Collecty windows | – | – |
| A3 | 05's features unchanged by A1 (checked) | v1 results stay valid | features identical on 3,000–4,000 dev windows per class | – | – |

## Windows

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| R1 | **Real rate ≥ 24 Hz over the window, everywhere** (tolerance for nominal 25 Hz) | no upsampled windows | ES: 12 windows, 5 official (all test user 1DBB0F6F; test train 52 → 50); NOR-TMD Android 3,100 minutes; iOS all | `rate_below_24hz` | c: one ES test user |
| R2 | One window definition (`tmd.candidates`): centred in the clock minute, single label, no gap > 0.5 s, v1's per-stretch resampling, U3, X3 | same rule for every new source | NOR-TMD 17,931 / Collecty 13,427 usable windows | `acc_status` | – |
| R3 | Only minutes with ≥ 1 event are rows; one row per user-minute (an ok window wins, then the earlier trip) | broken clocks; trips sharing a minute | NOR-TMD −40, Collecty −6,383 duplicate minutes (clock pieces) | – | – |
| R4 | U3 and X3 for all datasets, v1's thresholds | v1's plausibility | U3 ≈ 0 everywhere (≤ 1.6%); X3 bites in ExtraSensory only (walk 15%, run 27%, bike 10%), ≤ 0.2% elsewhere | `acc_implausible`, `label_implausible` | b: in-the-wild carrying |

## Collecty clocks and duplicates

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| K1 | 12 h wraps undone; pieces at other backward steps; overlapping windows never clean (approved) | 12-hour clock without AM / PM; sawtooth clocks | 36 files with wraps; 45 `time_overlap` minutes left after R3 | `clock_wraps`, `clock_piece`, `time_overlap` | b: time of day unreliable |
| K2 | Byte-identical files: one copy within a user, both across users (approved) | duplicated uploads | 15 files; 196 flagged rows (2 cross-user pairs) | `duplicate_dropped`, `dup_group` | – |

## Labels

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| L1 | **9 fine classes** (tram = 9); `cls` keeps v1's 8 | tram ≥ 10 users in ≥ 2 datasets | tram official: NOR-TMD 1,322 (dev 1,033, test 289), Collecty 153 | `cls_fine`, `y_fine` | – |
| L2 | **Six-class primary score** (still, foot, bike, car, bus, rail), merged at scoring time | rail and foot separations are what WAIIS needs first | `tmd.labels.SCORE6` | – | – |
| L3 | Conflicting labels never labels, kept as unlabelled usable minutes | still inside walk / outside | NOR-TMD inside 401, outside 1,636 usable minutes; Collecty Walk 3,529 | `label_excluded`, `reason` | b: no still / walk from new sources |
| L5 | **Run users who move like walking are never run labels** (`run_user_flag`, approved 2026-11-02): a user whose median window std \|acc\| over labelled run windows is below ExtraSensory dev walk's median (1.712 m/s²) gets all run minutes flagged; out of `official`, kept as unlabelled usable minutes; every source and split | 07's dev / test run gap: one dev user's "run" moved like walking | 7 ExtraSensory users (5 dev, 2 test); official run dev 463 → 155, test 81 → 74; no SHL or Collecty user | `run_user_flagged` | c: test run 3 users |
| L4 | Modes outside v1 (boat, e-scooter) unlabelled | no class (08) | NOR-TMD boat 3,715, e-scooter 198 minutes; Collecty e-scooter 388 | `reason` | b: production has them |

## Caps

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| C1 | **Per-user cap per class (R2)**: top user ≤ 25%, never below 60 windows; dev users; seeded round-robin over trips (approved) | one user shouldn't dominate a class | NOR-TMD: bike 592 → 405, car 661 → 405, train 493 → 470; Collecty: car 3,894 → 2,596, bus 1,701 → 253, train 2,643 → 252, tram 411 → 153, bike 562 → 69, run 196 → 60 | `user_capped` | b: Collecty bike / run still 1–2 users |
| C2 | SHL User 1 uncapped (approved) | its weight is a model choice (ablation in the first v2 model) | 23,076 official | – | a: 79% of dev train windows |

## Splits

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| P1 | **Reserved set v3** = v2's 12 ExtraSensory users + SHL Users 2–3 + 14 NOR-TMD installations (8 device models) | a second test for vehicles and rail | NOR-TMD test 3,221 official: bus 1,908, subway 738, tram 289, car 158, train 128 | `split` | c: train 5 users, 128 windows |
| P2 | Device model as the linkage guard | no person field; installations ≠ persons | all 14 installations of the 8 models held out | – | a: a person who changed phones |
| P3 | NOR-TMD and Collecty dev users in the 5 dev folds, balancing users per class (approved) | evaluate on unseen users | 77 + 14 users with labelled windows; others fold −1 | `fold` | – |
| P4 | Pre-training accessors exclude all reserved users of v3 | as v1 | tested (`tests/test_bench_reserved.py`) | n/a | – |

## Format

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| F1 | Schema only grows; v1 untouched; `tmd.bench` defaults to v1 until v2 is approved | 05–07 reproducible | v1 verify intact; 06's dev OOF bit-identical after all changes | – | – |
| F2 | New sources get `stream` tables (25 Hz, gap-free stretches with rate ≥ 24 Hz) | pre-training | NOR-TMD Android, Collecty (dedup copies excluded) | `bench.stream` | – |

## Dropped, not flagged

- NOR-TMD's orientation sensors, activity recognition and every non-accelerometer sensor (gyroscope, magnetometer,
  pressure stay in the converted events, not in the benchmark).
- Collecty's gyroscope and magnetometer events.
- Windows of empty minutes (R3) and the duplicate user-minutes R3 resolves.
