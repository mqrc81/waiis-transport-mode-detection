# Decision register: `waiis_bench_v2`

`waiis_bench_v2` is the frozen ExtraSensory + SHL + NOR-TMD + Collecty transport-mode benchmark (accelerometer at
25 Hz, labels, GPS from ExtraSensory and SHL only), built at commit `6d51dc2`. 04's register applies to ExtraSensory
and SHL unless a row below says otherwise.

- Full numbers: `results/tables/bench_v2/` (build) and `results/tables/checks/` (checks); manifest
  `results/manifests/waiis_bench_v2.json`.
- Risk: a = leakage or shortcut, b = gap to WAIIS data, c = test-set bias.

## Sources

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| V1 | ExtraSensory and SHL through v1's build code, then v2's columns | one build path; v1 unchanged | 15 of 32 files byte-identical to v1 (GPS, sessions, SHL streams) | – | – |
| V2 | **NOR-TMD, Android only**: training + reserved test | bus, subway, train, tram from many users | 14,178 examples, 91 installations | `source_platform` | b: Oslo / Bodø, 2021 |
| V3 | NOR-TMD iOS stored, never official | 5 Hz (R1) | 13,258 minutes, 0 windows | `rate_below_24hz` | – |
| V4 | **Collecty: training only** | car, some train / bus | 3,383 examples, 14 users | – | b: ≤ 5 users per class except car |
| V5 | SHL other positions as opt-in rows, in separate tables | position diversity, `(user, ts)` key stays unique | User 1: 5,964 windows (3 preview days); Users 2–3: 9,049 (rail check per position) | `position_default`, `position_draw` | b: one phone model |
| V6 | MobilitApp not used | gravity removed, untimed files (08) | – | – | – |

## Units & windows

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| A1 | **iOS to the Android axis convention and m/s² (× −9.80665)**, ExtraSensory iPhone included | one convention for raw-axis models | 220,961 ExtraSensory minutes + NOR-TMD iOS; 05's features unchanged | `sign_converted` | – |
| A2 | Collecty acceleration = linear + gravity | gravity included, as v1 | all Collecty windows | – | – |
| R1 | **Real rate ≥ 24 Hz**, everywhere (ExtraSensory: the snippet's rate) | no upsampled windows; jitter tolerance | ES 5 official windows (test user 1DBB0F6F; test train 52 → 50); NOR-TMD Android 3,100 minutes | `rate_below_24hz` | c: one ES test user |
| R2 | **One window definition** (`tmd.candidates`): centred in the clock minute, single label, no gap > 0.5 s, v1's per-stretch resampling | same rule for every new source | 17,931 / 13,427 usable windows (NOR-TMD / Collecty); 22 `stretch_too_short` | `acc_status` | – |
| R3 | One row per user-minute with ≥ 1 event (an ok window first, then the earlier trip) | trips and clock pieces share minutes | NOR-TMD −40, Collecty −6,383 duplicate minutes | – | – |
| R4 | U3 and X3 for all datasets, v1's thresholds | v1's plausibility | U3 ≤ 1.6% everywhere; X3 only bites in ExtraSensory (walk 15%, run 27%, bike 10%) | `acc_implausible`, `label_implausible` | b: in-the-wild carrying |

## Collecty clocks & duplicates

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| K1 | 12 h wraps undone; pieces at other backward steps; overlapping windows never official | 12-hour clock, sawtooth clocks | 36 files with wraps; 45 `time_overlap` minutes | `clock_wraps`, `clock_piece`, `time_overlap` | b: time of day unreliable |
| K2 | Identical files: one copy within a user, both across users | duplicated uploads | 15 files dropped (2 cross-user pairs, 196 rows flagged) | `duplicate_dropped` | – |

## Labels & scoring

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| L1 | **9 fine classes** (tram = 9); `cls` keeps v1's 8 | tram: ≥ 10 users in ≥ 2 datasets | tram 1,322 (NOR-TMD) + 153 (Collecty) | `cls_fine`, `y_fine` | – |
| L2 | **Six-class primary score** (still, foot, bike, car, bus, rail), merged at scoring time | separations WAIIS needs first | `tmd.labels.SCORE6` | – | – |
| L3 | Conflicting labels are unlabelled usable minutes, never labels | still inside walk / outside | NOR-TMD outside 1,636, inside 401; Collecty Walk 3,529 | `label_excluded` | b: no still / walk from new sources |
| L4 | Modes outside v1 unlabelled | no class | NOR-TMD boat 3,715, e-scooter 198; Collecty e-scooter 388 minutes | `reason` | b: production has them |
| L5 | **Run users who move like walking have no run labels** (median std \|acc\| < ES dev walk median, 1.712 m/s²; all sources and splits) | 07's dev / test run gap | 7 ES users; official run dev 463 → 155, test 81 → 74 | `run_user_flagged` | c: test run 3 users |
| S1 | **Primary: six-class macro-F1 on ExtraSensory dev, pooled out of fold**; guard: NOR-TMD dev (bus, car, rail) must not drop | dev is where models are chosen | v1 refs (v1 labels): 05 0.575, 06 0.642 | – | – |
| S2 | Always reported separately: Collecty dev, SHL day folds, NOR-TMD test (bus, subway, train, tram, car), SHL rail check per position; ES test overall and per platform | no source hides in a pooled score | – | – | a: source shortcut (identification AUC 0.82–0.96) |

## Caps & splits

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| C1 | **Per-user cap per class**: top user ≤ 25%, ≥ 60 windows; dev users; seeded round-robin over trips | one user shouldn't dominate a class | NOR-TMD bike / car / train capped; Collecty bus 1,701 → 253, train 2,643 → 252 | `user_capped` | b: Collecty bike / run 1–2 users |
| C2 | SHL User 1 uncapped | its weight is a model ablation | 79% of dev train windows | – | a: one person |
| P1 | **Reserved set v3** = v2's 12 ES users + SHL Users 2–3 + 14 NOR-TMD installations (8 whole device models) | a second test for vehicles and rail | NOR-TMD test 3,221: bus 1,908, subway 738, tram 289, car 158, train 128 | `split` | c: train 5 users |
| P2 | Device model as the person-linkage guard | no person field | all installations of the 8 models held out | – | a: a person who changed phones |
| P3 | New dev users in the 5 dev folds, balancing users per class; ES and SHL folds unchanged | unseen-user evaluation | 77 NOR-TMD + 14 Collecty users | `fold` | – |
| P4 | Pre-training accessors exclude every v3 reserved user | test data unseen in pre-training | tested | n/a | – |

## Format

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| F1 | Schema only grows; v1's files untouched; 04–08 pin `waiis_bench_v1` | 05–08 reproducible | v1 rebuild hash-identical, 06's dev OOF bit-identical after every change | – | – |
| F2 | `stream` tables for the new sources (gap-free stretches, ≥ 24 Hz) | pre-training | NOR-TMD Android, Collecty (dedup copies excluded) | `bench.stream` | – |

## Dropped, not flagged

- NOR-TMD's and Collecty's non-accelerometer sensors (they remain in the converted events).
- Empty minutes and the duplicate user-minutes R3 resolves.

## History

- 08's window rules (≥ 25 Hz, window ± 1 s resampling) were replaced by R1 / R2 (≥ 24 Hz, v1's resampling).
- The NOR-TMD test pick (12 installations, no car) was replaced by the device-model pick with car (P1, P2).
- First built at `c0956d6`; rebuilt at `6d51dc2` with L5 (only the labels tables changed). A build from `4167a7d` was
  discarded (manifest marked dirty by an output redirect).
