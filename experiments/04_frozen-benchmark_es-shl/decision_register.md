# Decision register: `waiis_bench_v1`

`waiis_bench_v1` is the frozen ExtraSensory + SHL transport-mode benchmark (accelerometer at 25 Hz, labels, GPS),
built at commit `e51ea79`.

- Full numbers per class, user and platform, code locations and commits:
  [`results/tables/decision_details.csv`](results/tables/decision_details.csv), keyed by ID
  (`register_numbers.py`).
- Risk: a = leakage or shortcut, b = gap to WAIIS data, c = test-set bias.

## Sources

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| S1 | ExtraSensory: labels, original labels, raw accelerometer, absolute location | in-the-wild phone data with user labels | 377,346 min, 60 users | – | b: 2015 students, one city |
| S2 | Accelerometer only; gyroscope, magnetometer, audio unused | WAIIS app records no gyroscope | +5,343 examples (3 users) | – | – |
| S3 | SHL hips: User 1 training source, Users 2–3 rail check | metro and most train data | 25,441 examples | – | b: one phone, one position |
| S4 | SHL User 1 duplicated days kept once | same recordings in two folders | −3 days | – | – |

## Units & resampling

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| U1 | Units per snippet: median \|acc\| < 3 means g, × 9.81 | iPhone in g, Android in m/s² | 222,630 snippets from g | yes (`unit`) | b: WAIIS units unknown |
| U2 | Each snippet to 25 Hz: interpolate to 100 Hz, FIR 1:4 | production rate, never across snippets | 360,064 windows | no | a: platform-specific filtering path |
| U3 | Window median \|acc\| must lie within 7–20 m/s² | catch unit and processing errors | −19 windows | yes (`acc_median`) | – |

## Cleaning

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| C1 | No snippet for missing files or span > 40 s | no data or broken clock | −394 min | no | – |
| C2 | Holes > 0.5 s: keep the longest clean stretch | never interpolate across holes | +399 examples | yes (`hole_trimmed`) | b: production holes unknown |
| C3 | Backwards timestamps: sort, deduplicate, store, never official | clock faults; repair unverified | −1,185 snippets | yes (`time_repaired`) | a: Android only |

## Windowing

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| W1 | One 15 s window (375 × 3) per snippet, centred | one fixed shape for every model | 360,064 windows | – | b: WAIIS fragment length unknown |
| W2 | Snippets under 15 s stored, never official | 80–100 Hz Android phones give 8–10 s | −8,112 included min | yes (`short`) | c: concentrated in 6 users |
| W3 | Every snippet stored in full at 25 Hz | allows other window lengths later | 376,952 snippets | yes (`snippets`) | – |

## Labels

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| L1 | One class per minute by priority; NaN = not reported | overlapping self-reported labels | 221,285 labelled min | yes (source columns) | – |
| L2 | Train from the original `ON_A_TRAIN` label | cleaned labels dropped it | 344 min | yes (source columns) | c: 2 test users, Android only |
| L3 | "Passenger" with bus or train counts as that vehicle, else car | passenger doesn't name the vehicle | 475 bus min | yes (source columns) | – |
| L4 | All label sources kept, incl. retrospective | most labels are given afterwards | 81% retrospective | yes (`label_source`) | b: coarse label timing |
| L5 | Unlabelled minutes kept with their windows | self-supervised pre-training | 66,796 windows | yes (`reason`) | – |

## Exclusions & caps

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| X1 | Drop `SLEEPING` still minutes | phone on nightstand, adds no signal | −83,055 min | yes (`reason`) | b: night recording unknown |
| X2 | Drop walk / run / bike when the phone isn't carried | label describes the person, not the phone | −1,078 min | yes (`not_carried`) | – |
| X3 | Drop quiet walk / run / bike windows | phone not moving with the person | −3,183 windows | yes (`label_implausible`) | c: iPhone test run 38 left |
| X4 | Cap resting still at 20% per user | resting is real but shouldn't dominate | −62,499 min | yes (`capped`) | a: unknown positions often resting |
| X5 | Drop minutes with two moving classes | contradictory labels | −18 min | yes (`reason`) | – |
| X6 | Motorbike kept as a labelled check, never trained | one user only | 86 min | yes (`motorbike`) | b: relevance to WAIIS unknown |
| X7 | Drop plane, boat, elevator, treadmill, skateboard and similar | not one of our classes | −1,471 min | yes (`reason`) | b: production will contain these |
| X8 | Exclude user BEF6C611 entirely | inconsistent accelerometer processing | −1,835 min | yes (`user_excluded`) | – |

## Splits

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| P1 | 12 reserved test users, balanced by class, platform, location | supervisor-approved re-pick | 26,394 test examples | yes (`split`) | c: 11 were 02 dev users |
| P2 | Dev: 5 folds by user, balanced by platform | evaluate on unseen users | 118,986 dev examples | yes (`fold`) | a: train in 3 folds only |
| P3 | Pre-training accessors never return reserved users | test data unseen in pre-training | 12 + 2 users | n/a | – |
| P4 | SHL: User 1 day folds; Users 2–3 rail check only | unseen-person rail evaluation | 23,076 / 2,365 examples | – | a: User 1 is one person |

## GPS (ExtraSensory)

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| G1 | Lat/lon from the absolute-location files, never imputed | no raw track; NaN is ambiguous | 81% of minutes | – | a: Android 33% missing, iPhone 7% |
| G2 | `location_missing_reason` per minute | separate "no update" from "no fix" | 71,598 no_updates | yes | b: still looks like no update |
| G3 | speed = `max_speed`, accuracy = best; otherwise −1 | production format | 63% speed −1 | yes (`location_features`) | a: missing speed signals slow |
| G4 | Bearing from the previous fix, −1 when speed is −1 | phones omit bearing without speed | 66% bearing −1 | yes (`bearing_from_fixes`) | b: 60 s baseline, not instantaneous |
| G5 | One fix per minute, time = minute timestamp | fix time not published | median 60 s between fixes | – | b: sparser than production |
| G6 | Session = minutes at most 2 min apart | contiguous recording batches | 8,729 sessions | yes (`ts`) | b: WAIIS batching unknown |

## SHL

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| H1 | One 15 s window per clock minute, centred | mimic the app's fragments | 39,840 windows | – | b: no gaps between fragments |
| H2 | Label only when one coarse label fills the window | no transitions inside examples | −14,401 min | yes (`majority_cls`) | a: ExtraSensory has no mixed minutes |
| H3 | GPS thinned to the WAIIS cadence by speed (10 / 20 / 180 s) | match production fix rate | 80,892 fixes (User 1) | yes (`gps_1hz`) | b: fix trigger assumed |
| H4 | Second GPS variant: one fix per minute | avoid fix interval as dataset shortcut | 23,645 fixes (User 1) | – | a: use it for joint training |
| H5 | Speed and bearing blanked, more often when slow | production misses speed mostly when slow | 23% speed −1 | yes (`gps_unblanked`) | b: curve shape assumed |
| H6 | Hips stream, other positions, 1 Hz track stored | pre-training and position studies | 59.8 M samples | yes (extras) | – |

## Format

| ID | Decision | Why | Effect | Flag/reversible? | Risk |
|---|---|---|---|---|---|
| F1 | Parquet per source, user + minute key, sha256 manifest | frozen, verifiable, rebuildable | 23 files, 4.8 GB | – | – |
| F2 | Windows as float32 m/s², gravity included | lossless for 25 Hz data | 399,904 windows | – | b: WAIIS gravity unknown |
| F3 | `official` column = the example set; y = 0 outside it | one definition for every model | 145,380 examples | yes | a: `cls` alone includes excluded minutes |
| F4 | All source label columns kept | labels can be re-derived | 167 columns | – | a: never use them as features |

## Dropped, not flagged

- 290 ExtraSensory minutes without a raw accelerometer file, and 104 snippets with a span over 40 s.
- Raw ExtraSensory sample timestamps; only the 25 Hz resampled signal is stored.
- Gyroscope, magnetometer, audio, watch and phone-state data (they remain in the raw files).
- SHL hand, torso and bag phones, except User 1's three 4phones days.
- SHL windows of minutes at a recording edge (125) or across a NaN gap (27).
- SHL stream stretches shorter than 15 s.
- SHL pressure, light, cell, Wi-Fi and satellite data.

## Open questions for WAIIS

1. When does the app record, and how long is one fragment?
2. Is sampling exactly 25 Hz, or are there jitter, holes or clock faults?
3. Which acceleration units does it send, and is gravity included?
4. What is the iOS / Android split, and which device models?
5. How do users carry the phone, and how often does it rest?
6. Which modes matter: motorbike, scooter, plane, tram, taxi?
7. What triggers a GPS fix, and how often is speed or bearing missing per mode?
8. Who labels the production data, and at what granularity?
9. What is one upload: a trip, a day?

## History

- Labels v1 (`7f435c5`) were replaced by v2 with the `SLEEPING` drop, the not-carried drop and the still cap (`cb8a024`).
- Snippets with holes over 0.5 s were dropped (`1f52628`); the longest clean stretch is now kept (`b1e2a9c`).
- Snippets with backwards timestamps were dropped (`1f52628`); they are now repaired, stored and never official (`b1e2a9c`).
- Snippets under 15 s were dropped (`a711d65`); they are now stored and never official (`b1e2a9c`).
- SHL speed was blanked uniformly at 23% (`a711d65`), with bearing (`9ef9424`); blanking is now driven by speed (`b1e2a9c`).
- ExtraSensory bearing was kept on fixes without speed (`a711d65`); it is now −1 there (`30550d9`).
- The plausibility bound was 7–13 m/s² (`b1e2a9c`); it is now 7–20 m/s², plus the BEF6C611 exclusion (`cf1085c`).
- Test users picked on 2026-10-25, gyroscope required (`7f435c5`), were replaced by the re-pick (picked at `1d03c9d`,
  adopted at `aef0a4c`; method and reproduction in `splits/README.md`).
- Built during development as es_bench_v4 (commit cf1085c); es_bench_v1–v3 were intermediate builds, never used by any model, deleted.
