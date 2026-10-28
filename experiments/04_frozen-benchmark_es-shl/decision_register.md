# Decision register: `es_bench_v1` and `es_bench_v2`

Every decision that affects which examples, labels, values or splits are in `es_bench_v1`, from experiment 02 through 04.

**`es_bench_v2`** (2026-10-28, step 3) keeps every v1 decision except where an entry carries a **v2:** note.
- Its new decisions are in section 11 (V1–V10).
- v1 stays on disk unchanged; its numbers below are v1's.
- v2 numbers are from `results/tables_v2/` and `logs/build_v2.out`.

**Basis.** Numbers from `register_numbers.py` (output `logs/register_numbers.out`).
- Data: `es_bench_v1` as built from commit `30550d9` (`manifest.json`, clean tree), plus 02's raw-snippet audit
  `<data>/processed/es_minutes_audit.parquet`. Numbers were computed on 2026-10-28.
- Code references are file:line at commit `3822830`.
- "Brief" is `experiments/02_family-baseline_shl-es/docs/brief_block1_experiments.md`; "02" and "04" are those
  experiments' `README.md`.

**Terms.**
- "Example": an included minute that has a 15 s window (`tmd.bench.examples`). ExtraSensory has 149,681.
- "Included": `included_v2`.

**Fields.** Each decision gives:
- **Decided**: what was decided, and where (experiment, step, commit).
- **Code**: where it is implemented.
- **Why**: the reason given at the time.
- **Effect**: computed from the data now.
- **Reversible**: whether the stored flags undo it without a rebuild.
- **Risk**:
  - (a) leakage or a shortcut between datasets or classes;
  - (b) a gap against WAIIS-like data;
  - (c) bias in the test set.

---

## 1. Sources and downloads

**S1. ExtraSensory files used.**
- **Decided:** 02 step 1 (`7f435c5`, 2026-10-24/25), plus 04 for absolute location (`a711d65`). Files:
  - cleaned features + labels CSV (labels, position, `label_source`, `location:*`);
  - original labels;
  - `raw_acc`;
  - absolute location;
  - `cv_5_folds` (platform only).
- **Code:** `src/tmd/extrasensory.py:53-87`, `:186-191`.
- **Why:** brief lines 40-41. The cleaned CSV is used only for labels; motion features computed at native rate are
  not used. Cleaning removed `ON_A_TRAIN`, so the original labels are needed.
- **Effect:** 60 users and 377,346 minutes, with one row per minute in every source. Sha256 of all zips in
  `LICENSES.md`.
- **Reversible:** yes (raw files stay).
- **Risk:** (b) the dataset states no licence. Commercial use is unclear (LICENSES.md).

**S2. Accelerometer only.**
- **Decided:** DECISIONS.md 2026-10-28 (`9ae5d9c`), 04 brief. `proc_gyro`, `raw_magnet`, audio, watch and the
  phone-state features are not in the benchmark.
- **Code:** `experiments/04_frozen-benchmark_es-shl/build.py:96-125`.
- **Why:** the WAIIS app records no gyroscope.
- **Effect:** the 3 gyroscope-less users return, with 5,442 examples:

  | user | examples | fold |
  |---|---|---|
  | 61359772 | 3,200 | 1 |
  | CCAF77F0 | 1,302 | 3 |
  | F50235E0 | 940 | 2 |
- **Reversible:** n/a (raw files stay).
- **Risk:** none for WAIIS.

**S3. SHL files used.**
- **v2:** hand, torso and bag windows of User 1's 4phones days are stored, never official (V8).
- **Decided:** 02 step 1 (`7f435c5`), 04 brief.
  - Hips only (`Hips_Motion.txt`, acc columns), `Label.txt` coarse label and `Hips_Location.txt`.
  - User 1: `release/User1` without 220617, 260617 and 270617 (byte-identical copies of the `User1_4phones` hips
    data; 02 audit `shl_duplicates.csv`), plus `User1_4phones`.
  - 160517 has no Motion or Label file and is skipped.
- **Code:** `build.py:200-212`, `src/tmd/shl.py`.
- **Why:** the hips phone is closest to a pocket; duplicates would count twice.
- **Effect:**
  - User 1: 90 recordings, 36,968 minutes, 23,099 examples.
  - Users 2–3: 6 recordings, 3,024 minutes, 2,366 examples.
  - Hand, torso and bag are not used.
- **Reversible:** n/a (raw files stay).
- **Risk:** (b) one phone model (Huawei Mate 9) and one carrying position.

---

## 2. Unit detection and resampling

**U1. Units per snippet: median |acc| < 3 → g → × 9.80665.**
- **v2:** windows with a median |acc| outside 7–13 m/s² after conversion are flagged `acc_implausible` and left out of the official set (V1). That removes 1,192 of BEF6C611's 1,544 windows.
- **Decided:** 02 step 1 (`7f435c5`), brief line 48.
- **Code:** `src/tmd/resample.py:39-43`.
- **Why:** 02's audit found iPhone in g and Android in m/s², except BEF6C611 (Android), which is in g "for 1,713 of
  3,451 minutes". So the rule is per snippet.
- **Effect:** all 220,681 iPhone windows are treated as g. Of the Android windows, 1,352 are g (all BEF6C611) and
  137,207 m/s².
  - **The rule is ambiguous for 987 snippets** with a raw median |acc| between 1.5 and 6. 786 of them have a window
    and 439 are examples (still 295, car 83, walk 40, run 21); 920 of the 987 are BEF6C611.
  - **BEF6C611** (dev, fold 0, 1,518 examples) is not a clean g/m/s² mix. Its raw median |acc| per snippet spreads
    over every range:

    | raw median \|acc\| | examples |
    |---|---|
    | < 0.5 | 427 |
    | 0.5–1.5 | 72 |
    | 1.5–3 | 224 |
    | 3–6 | 162 |
    | 6–8 | 171 |
    | ≥ 8 | 462 |

    Only 462 examples show gravity near 9.8 m/s² after conversion. The rest look like gravity-removed or rescaled
    data.
  - 296 examples have a raw median ≥ 13 m/s² (217 of them BEF6C611).
- **Reversible:** partly. `accelerometer.unit` tells which windows were multiplied, so the conversion can be undone,
  but a different rule needs a rebuild.
- **Risk:**
  - (a) one dev user's still minutes have a magnitude unlike any other user's, and it trains models on a wrong
    gravity level;
  - (b) unknown whether WAIIS sends acc with gravity, and in which unit.

**U2. Resampling to 25 Hz inside each snippet.**
- **Decided:** brief lines 54 and 58; 02 step 2 (`1f52628`). The snippet is linearly interpolated onto a 100 Hz
  grid, then `resample_poly(1, 4)` (Kaiser FIR, `padtype="line"`). Never across snippets.
- **Code:** `src/tmd/resample.py:20-22`, `:67-77`.
- **Why:** common 25 Hz rule; the snippets are irregular (≈ 34 / 50 / 80–100 Hz).
- **Effect:** raw rate per window:

  | | median | p1 | min |
  |---|---|---|---|
  | iPhone | 34.1 Hz | 30.9 Hz | 27.6 Hz |
  | Android | 49.9 Hz | 40.4 Hz | 20.1 Hz |

  - 440 windows come from snippets below 30 Hz. Below 25 Hz, the 12.5 Hz band is filled by interpolation.
  - **1,005 Android windows** (examples: still 180, car 90, walk 24, bike 12, run 1, train 1) start or end less
    than 10 samples (0.4 s) from the snippet edge, inside the filter's edge transient.
- **Reversible:** no (rebuild from raw).
- **Risk:**
  - (a) the platforms reach 25 Hz by different paths: 34 Hz → 100 Hz linear interpolation (attenuates the top
    band) against 50 Hz → 100 Hz. The spectrum near 8–12.5 Hz may differ by platform;
  - (b) WAIIS records at 25 Hz natively, with the phone's own (unknown) filtering.

**U3. SHL resampling per NaN-free stretch.**
- **Decided:** as 02's `make_features.py:52-75` (02 step 2, `1f52628`); 04 `a711d65`. Each NaN-free 100 Hz
  stretch is decimated 1:4 as a whole, then windows are cut from it.
- **Code:** `build.py:224-228`.
- **Why:** SHL is already on an exact 100 Hz grid. Same filter as ExtraSensory, never across NaN gaps.
- **Effect:** 25 minutes (User 1) and 2 (Users 2–3) have a window that touches a NaN gap (`acc_status = nan_gap`).
- **Reversible:** no.
- **Risk:** (a) SHL has no rate jitter and needs no interpolation, unlike ExtraSensory. This is a dataset
  difference a model could pick up.

---

## 3. Snippet cleaning (ExtraSensory)

All four rules are checked in this order; the first failure counts (`src/tmd/resample.py:25-36`; minute row kept,
`acc_status` set, **no window stored**). They come from 02, `1f52628`.

**C1. Time not increasing** (any dt ≤ 0, including repeated timestamps).
- **v2:** repaired instead of dropped (sort, drop repeated timestamps, re-check), flagged `time_repaired`, stored in `snippets`, never official (V6).
- **Decided:** 02 decision 4, 2026-10-25; brief line 53.
- **Why:** broken device clock.
- **Effect:**
  - All minutes: 1,187, all Android, in 17 users (481F4DD2 234, BEF6C611 217, 99B204C0 160).
  - Examples lost: 562 (still 344, walk 162, car 40, bike 6, bus 6, run 3, train 1).
- **Reversible:** no.
- **Risk:**
  - (b) WAIIS may have the same clock faults; sorting or splitting would keep the data;
  - (a) the rule hits Android only.

**C2. Span > 40 s.**
- **Decided:** 02 decision 4 (2026-10-25).
- **Why:** broken clock (an 800-sample snippet should last ≤ 24 s).
- **Effect:**
  - All minutes: 102 (Android 66, iPhone 36), in 25 users.
  - Examples lost: 62 (still 47, train 5, bike 3, walk 3, car 4).
- **Reversible:** no.
- **Risk:** small.

**C3. Hole > 0.5 s inside a snippet.**
- **v2:** approved by Marc (2026-10-28). The longest clean stretch is kept, official if ≥ 15 s (`hole_trimmed`, V5).
- **Decided:** 02 step 2 (`1f52628`). **Never explicitly approved**: 02 lists it under "Extra ExtraSensory snippet
  drops, besides the approved ones" (02, Deviations step 2). 04 kept it as "02's cleaning".
- **Why:** interpolating across the hole would invent data.
- **Effect:**
  - All minutes: 1,946 (Android 1,710, iPhone 236), in 52 users (BEF6C611 357, 481F4DD2 301, 99B204C0 173).
  - Examples lost: 970 (still 716, walk 129, car 74, bike 36, bus 10, train 3, run 2).
- **Reversible:** no.
- **Risk:**
  - (b) production fragments with a hole are dropped here; the longest clean stretch could have been kept;
  - (a) mostly Android.

**C4. NaN or missing file.**
- **Decided:** brief line 50.
- **Effect:** 290 minutes have no `raw_acc` file (Android 282, iPhone 8; 24 users, CCAF77F0 77), 9 of them
  included. No snippet contains NaN.
- **Reversible:** no (no data).
- **Risk:** none.

---

## 4. Windowing (15 s, centre)

**W1. One 15 s window (375 × 3) per snippet; snippets shorter than 15 s are dropped.**
- **v2:** short snippets are stored with their real length (`short`, in `snippets`), never official (V6).
- **Decided:** 04 brief; confirmed by Marc after stop-point 1 (04 Decisions 1, 2026-10-28; `a711d65`).
- **Code:** `build.py:110-116`, `src/tmd/resample.py:80-85`.
- **Why:** one fixed shape for every model, like the fragments the app sends.
- **Effect:**
  - All minutes: 14,581 snippets, all Android; 2,696 of them unlabelled.
  - Examples lost: 7,502:

    | | still | walk | car | bike | bus | train | run |
    |---|---|---|---|---|---|---|---|
    | lost | 4,562 | 1,478 | 714 | 457 | 151 | 91 | 49 |
    | share of included | 3.8% | 7.0% | 6.4% | 9.4% | 8.4% | **26.6%** | 6.4% |

  - **97% come from 6 Android users with 80–100 Hz snippets.** Their examples lost and splits:

    | user | split | lost |
    |---|---|---|
    | 81536B0A | dev, fold 3 | 1,972 |
    | 1DBB0F6F | dev, fold 3 | 1,951 (incl. 91 train) |
    | CCAF77F0 | dev, fold 3 | 1,337 |
    | F50235E0 | dev, fold 2 | 1,037 |
    | 27E04243 | dev, fold 3 | 600 |
    | 96A358A0 | test | 396 (incl. all 63 bus) |
  - By split: dev loses 7,084, test 418 (still 329, bus 63, walk 26).
- **Reversible:** no; the short snippets are not stored.
- **Risk:**
  - (c) test bus drops from 6 to 5 users;
  - (a) dev fold 3 holds 4 of the 6 users and is thin (run 9, car 516, train 15);
  - (b) whether WAIIS fragments are ≥ 15 s is unknown.

**W2. The window is cut from the centre of the resampled snippet.**
- **v2:** every snippet is stored in full at 25 Hz (`snippets`); the official window is its centre crop (`window_start`) (V6).
- **Decided:** 04 brief (`a711d65`).
- **Code:** `src/tmd/resample.py:80-85`; offset stored as `accelerometer.start_offset_s`.
- **Why:** avoids filter edges and start-up effects.
- **Effect:** discarded per window, as mean (min–max): iPhone 8.8 s (5.7–13.9) of a ~23 s snippet; Android 1.3 s
  (0–24.8).
- **Reversible:** no.
- **Risk:**
  - (a) iPhone windows come from the middle of a long snippet, Android windows span almost the whole snippet, so
    edge effects are platform-dependent (see U2);
  - (b) 37% of every iPhone snippet is discarded.

---

## 5. Label mapping v2 (per class)

Implemented in `src/tmd/extrasensory.py:27-131` (`minute_classes`).
- v1: 02 step 1 (`7f435c5`). v2: 02 step 2b (`cb8a024`), approved 2026-10-26 (`02e321a`, brief lines 63-81).
- A NaN label means "not reported" (`extrasensory.py:103`, brief line 55).
- Position flags are cleaned OR original (`:129`).
- Priority: the first class whose label is set wins (`:27-35`).

**L1. train ← original `ON_A_TRAIN`.**
- **Why:** the cleaned CSV dropped the label (brief line 41); San Diego mixes commuter rail and a light-rail
  trolley, so trust it only at family level (brief line 66).
- **Effect:** 344 minutes in 5 users, 342 included, 242 examples. On 338 of the 342 the cleaned CSV says `SITTING`.
  Users: 1DBB0F6F 115 (15 examples), B7F9D634 97, 3600D531 69 (test), 33A85C34 37, A76A5AF5 24.
- **Reversible:** yes (source columns are stored).
- **Risk:**
  - (c) 1 test user;
  - (a) train vs metro is not separable in ExtraSensory.

**L2. bus ← `ON_A_BUS`; passenger with bus or train → bus or train.**
- **Decided:** 02 decision 1, 2026-10-25.
- **Why:** "passenger" does not name the vehicle.
- **Effect:** 475 bus minutes (3 users) are bus instead of conflicts; 1,793 included.
- **Reversible:** yes.
- **Risk:** low.

**L3. car ← `IN_A_CAR`, `I'M_THE_DRIVER`, or `PASSENGER` without bus or train.**
- **Effect:** 11,166 included: `IN_A_CAR` 6,082, driver 7,889, passenger 2,051 (overlapping). 839 minutes are car
  through "passenger" alone. All 86 `MOTORBIKE` minutes were car and are now check minutes (X5).
- **Reversible:** yes.
- **Risk:** (b) a passenger-only minute could be a taxi or shuttle; WAIIS's mode list is unknown.

**L4. bike ← `BICYCLING`.**
- **Effect:** 4,878 included.
- **Risk:** (b) e-bikes and scooters are unknown; `SKATEBOARDING` is dropped (X6).

**L5. run ← `FIX_running`; walk ← `FIX_walking`, `STAIRS_*`.**
- **Effect:** run 762 included; walk 21,241 (`FIX_walking` 21,098; stairs 747, of which 143 are stairs only).
  `STROLLING` (original, 806 minutes) is not dropped: 574 walk, 193 car, 39 still.
- **Reversible:** yes.
- **Risk:** low.

**L6. still ← `SITTING`, `OR_standing`, `LYING_DOWN`.**
- **Effect:** 118,604 included after X1–X3 (`SITTING` 74,029, `OR_standing` 28,872, `LYING_DOWN` 15,703;
  overlapping).
- **Risk:** see X3.

**L7. A moving class beats still; two or more moving classes = conflict.**
- **Decided:** brief line 73.
- **Effect:** see X4.

**L8. Every `label_source` is kept** (history interface, active feedback, notifications).
- **Effect:** history interface (2) is 54–81% of each included class (train 54%, walk 81%).
- **Reversible:** yes (`label_source` is stored).
- **Risk:** (b) labels given afterwards may be coarse in time; WAIIS label quality is unknown.

**L9. Unlabelled minutes are kept.**
- **Effect:** 70,353 minutes (Android 27,833, iPhone 42,520), 66,711 with a window, for self-supervised
  pre-training.
- **Reversible:** n/a.
- **Risk:** (c) the test users' unlabelled minutes are in the same table, and must never enter pre-training
  (reserved-test rule).

---

## 6. Exclusions and caps

All are **flags**: the minute and its window stay, and `included_v2` is false. The `cls` column keeps the class the
rules would have given; `y` is 0 (`build.py:82`).

**X1. `SLEEPING` still minutes → `drop:SLEEPING`.**
- **Decided:** v2 (02 step 2b, approved 2026-10-26).
- **Code:** `extrasensory.py:120`.
- **Why:** the phone is on the nightstand; 31% of still.
- **Effect:** **83,055** still minutes (Android 30,360, iPhone 52,695), 53 users (9759096F 3,714, 83CF687B 3,470,
  CCAF77F0 3,433).
- **Reversible:** yes (`reason`).
- **Risk:** (b) if the app records at night, the night-time resting phone is missing from evaluation.

**X2. Walk / run / bike with the phone not carried → `drop:not_carried`.**
- **Decided:** v2. "Not carried" = cleaned `PHONE_ON_TABLE` or original `PHONE_AWAY_FROM_ME`.
- **Code:** `extrasensory.py:50`, `:121`.
- **Why:** the label describes the person, not the phone.
- **Effect:** 1,078 minutes (walk 932, run 146, bike 0; Android 281, iPhone 797), 27 users (2C32C23E 279,
  481F4DD2 155, 9DC38D04 89).
- **Reversible:** yes (`not_carried`).
- **Risk:** none for WAIIS (the phone would show still).

**X3. Resting still capped at 20% of each user's still minutes.**
- **Decided:** v2 (Marc, 2026-10-26: 20% cap, in training and in evaluation).
- **Code:** `extrasensory.py:136-155`; called over the `ok` minutes of all 60 users, seed 0 (`build.py:77`).
- **Why:** resting phones are real still states, but should not dominate still (44% before v2).
- **Effect:** **62,499** still minutes capped (Android 30,435, iPhone 32,064).
  - 19 of 60 users lose more than half their still minutes (665514DE 83%, 61976C24 77%, B7F9D634 70%).
  - Among still examples, resting is 14.1%, a carried position (pocket / hand / bag) 24.8%, and **no position
    reported 62%**.
- **Reversible:** yes (`capped`, `resting`).
- **Risk:**
  - (a) the cap only sees reported positions. 02 found 49% of no-position still windows as quiet as a resting
    phone, so the true resting share is higher than 14%;
  - (c) resting share is 16.0% in test against 13.8% in dev;
  - (a) the random picks depend on the order and set of users (one generator over sorted users), so a different
    user set changes which minutes are capped.

**X4. Conflicts dropped.**
- **Decided:** brief line 73.
- **Effect:** 18 minutes: run + walk 15, train + walk 2, bus + car 1. 3 users (CF722AA9 15).
- **Reversible:** yes (`moving` column).
- **Risk:** none.

**X5. `MOTORBIKE` → `check:MOTORBIKE`.**
- **Decided:** 02 decision 2, 2026-10-25; brief line 76.
- **Code:** `extrasensory.py:45`, `:122-123`. It overrides every other reason.
- **Why:** one user, so a 9th class would learn that person; kept as a labelled check.
- **Effect:** 86 minutes that were car, all 33A85C34 (Android, dev fold 4).
- **Reversible:** yes (`motorbike`).
- **Risk:** (b) whether motorbikes matter to WAIIS is open.

**X6. Other transport and exercise labels (original) → drop.**
- **Decided:** 02 audit, 2026-10-25 (brief lines 73-75). `ELEVATOR` is from the brief.
- **Code:** `extrasensory.py:42-44`, `:116-117`. Only the **original** label is checked.
- **Effect** (the class the minute would have had):

  | label | minutes | would have been | users |
  |---|---|---|---|
  | `ON_A_PLANE` | 630 | still | 4 (9759096F 307) |
  | `ELLIPTICAL_MACHINE` | 233 | still 197, run 36 | 2 |
  | `ELEVATOR` | 200 | walk 117, still 83 | 12 |
  | `TREADMILL` | 164 | run 131, walk 33 | 2 |
  | `SKATEBOARDING` | 131 | still 59, bike 56, none 16 | 3 |
  | `STATIONARY_BIKE` | 86 | bike | 2 |
  | `ON_A_BOAT` | 18 | still | 1 |
  | `WHEELCHAIR` | 9 | still | 1 |

  `RIDING_AN_ANIMAL` and `USING_CRUTCHES` have 0 minutes. No cleaned-only `ELEVATOR` minute exists.
- **Reversible:** yes (`reason`).
- **Risk:** (b) plane, boat and skateboard or scooter trips will appear in production. They are not in any class
  here, so models have never seen them.

---

## 7. Splits and test-user selection

**P1. 12 reserved ExtraSensory test users.**
- **v2:** unchanged in `es_bench_v2`. A re-pick is **proposed, not in use** (V10).
- **Decided:** 02 step 1 (`7f435c5`), approved 2026-10-25, frozen in `splits/reserved_test.json` (`6012aa7`).
  Final test by DECISIONS.md 2026-10-28.
- **Code:** `experiments/02_family-baseline_shl-es/propose_splits.py:42-62`; `src/tmd/splits.py:49-70`; checked
  in `build.py:83-86`.
- **Rule:** among 200,000 random 12-subsets (seed 0) that pass the constraints, the one whose class shares are
  closest to 12/60 wins. Constraints:
  - each of bus, car and bike has ≥ 4 users with ≥ 5 minutes (`:53`);
  - 6–7 iPhone users (`:49`, `:55`);
  - ≥ 1 train user in test and ≥ 1 in dev (`:57`);
  - dev keeps ≥ 50% of every class (`:59`);
  - **candidates are only the 57 users with gyroscope minutes** (`:47`, `:62`).
- **Why:** brief lines 94-97; enough users per class; platform balance.
- **Effect.**
  - Counts at pick time: labels v1 with the passenger rule, no `SLEEPING` drop, no cap, and acc + gyro files
    present without the resampler's cleaning.
  - Test share then vs now:

    | | still | walk | run | bike | car | bus | train |
    |---|---|---|---|---|---|---|---|
    | at pick time | 20.8% | 18.4% | 17.7% | 19.5% | 17.6% | 19.9% | 20.2% |
    | now (examples) | 16.6% | 18.5% | 17.4% | 20.5% | 18.4% | 16.5% | 28.5% |

  - Test users with ≥ 5 examples now: still 12, walk 11, car 10, bus 5, bike 4, run 2, train 1.
  - The test set has 7 iPhone and 5 Android users; its examples are 66% iPhone (dev 60%).
  - Location coverage on examples is **95.0% in test against 80.4% in dev**.
  - The 3 gyroscope-less users (all Android, 2 of them among the 6 high-rate users) could not be picked. Of the 6
    high-rate users, only 96A358A0 is in test.
- **Reversible:** a new split is possible from the `user` column, but the test set is frozen by decision.
- **Risk:**
  - (c) the test set over-represents iPhone and users with GPS, so GPS models will look better on test than on dev;
  - (c) the selection was optimised on counts that no longer apply (v1, gyroscope present);
  - (c) run (2 users) and train (1 user) can't be tested on unseen users in any meaningful way.

**P2. ExtraSensory dev folds.**
- **Decided:** 02 step 1 (`7f435c5`): 5 folds by user, dealt round-robin within platform, 20,000 draws, closest to
  1/5 per class.
- **Code:** `propose_splits.py:64`; `splits.py:13-46`.
- **Why:** brief line 89.
- **Effect:**
  - A class must be in every fold only if ≥ 5 users have it, so fold 0 has no train.
  - The gyroscope-less users were already placed (folds 1, 3 and 2), so nobody moved in 04.
  - Fold 3 is thin now: run 9, car 516, train 15 (W1).
- **Reversible:** yes.
- **Risk:** (a) noisy per-fold metrics for run and train.

**P3. SHL splits.**
- **Decided:** 02 step 1.
  - User 1: 5 folds of whole calendar days (`shl_days.csv`, `split = dev`).
  - Users 2–3: `rail_check`, fold −1. DECISIONS.md 2026-10-28 makes them a rail check only, never used for decisions.
- **Code:** `build.py:200-212`.
- **Why:** day folds avoid same-day leakage.
- **Reversible:** yes.
- **Risk:** (a) User 1 is a single person; day folds are optimistic (DECISIONS.md 2026-10-28).

---

## 8. GPS derivation and missing values (ExtraSensory)

**G1. Lat/lon from the absolute-location files.**
- **Decided:** 04 brief.
- **Code:** `extrasensory.py:186-191`.
- **Why:** the website's definition: NaN if there was no update, or all updates were worse than 200 m. Otherwise the
  latest of the best-accuracy updates in the ~20 s recording.
- **Effect:** 305,747 of 377,346 minutes (81.0%) have coordinates. Coverage by class and platform (examples):

  | | still | walk | run | bike | car | bus | train |
  |---|---|---|---|---|---|---|---|
  | iPhone | 92.4% | 95.9% | 100% | 99.5% | 95.8% | 96.1% | 100% |
  | Android | 67.6% | 56.4% | 67.9% | 85.0% | 66.3% | 58.6% | 79.9% |
- **Reversible:** n/a. Nothing is imputed (Marc, 2026-10-28).
- **Risk:**
  - (a) "has a fix" separates the platforms (Android 32% missing, iPhone 8%);
  - (c) test 95% against dev 80% (P1).

**G2. `location_missing_reason`.**
- **Decided:** Marc, 2026-10-28.
- **Code:** `extrasensory.py:194-206`.
- **Effect:** `no_updates` 71,598, `withheld` 1, `poor_accuracy` 0, `unknown` 0.
  - `poor_accuracy` cannot be detected: the CSV features never exceed 200 m accuracy (max 199.3 m).
  - The home masking from the paper is not visible.
  - The no-update share among labelled minutes ranges from 6% (bike) to 22% (bus). Still (17%) is not an outlier in
    ExtraSensory.
- **Reversible:** yes.
- **Risk:** (b) a still phone may produce no update at all, which mixes "no movement" with "no fix".

**G3. `speed` = `location:max_speed`, NaN → −1.**
- **Decided:** 04 brief.
- **Code:** `build.py:181`.
- **Why:** the production format uses −1. Units are m/s; no negative value exists in the CSV.
- **Effect:** speed is −1 on 67.5% of fixes. On examples:

  | | still | walk | run | bike | car | bus | train |
  |---|---|---|---|---|---|---|---|
  | Android | 91.4% | 61.4% | 58.1% | 16.6% | 15.3% | 36.7% | 42.9% |
  | iPhone | 66.9% | 35.1% | 39.8% | 11.4% | 9.0% | 10.3% | 8.6% |

  3 minutes have `max_speed` > 45 m/s (1 over 60). Outliers are not removed.
- **Reversible:** yes (`location_features` keeps the raw NaN).
- **Risk:**
  - (a) **missing speed is strongly class-dependent** (still 67–91% against 9–17% in car), so a −1 speed is
    itself a still feature;
  - (b) production is ~23% missing overall, and its per-class rate is unknown.

**G4. `accuracy` = `location:best_horizontal_accuracy`, else −1.**
- **Code:** `build.py:182`.
- **Effect:** median 10 m (iPhone) and 12.7 m (Android), against ~8 m in production.
- **Reversible:** yes.
- **Risk:** (b) small.

**G5. `bearing` = bearing from the previous fix, −1 where speed is −1.**
- **Decided:** Marc, 2026-10-28 (`30550d9`). `bearing_from_fixes` is kept and never blanked; −1 for the first fix
  of a session or a move under 1 m.
- **Code:** `build.py:183-189`; `gps.py:50-55`.
- **Why:** phones record no bearing on a fix without speed.
- **Effect:** `bearing` −1 on 71.0%, `bearing_from_fixes` on 29.1%.
- **Reversible:** yes.
- **Risk:**
  - (a) inherits G3's class dependence;
  - (b) `bearing_from_fixes` spans ~60 s, while a phone bearing is instantaneous.

**G6. Fix `time` = minute timestamp; one fix per minute.**
- **Code:** `build.py:179`.
- **Why:** the fix's own time is not published.
- **Effect:** the fix interval has a median of 60 s, against 10–20 s in production while moving.
- **Reversible:** n/a.
- **Risk:** (a), (b) sparser than production; dataset-specific cadence (see H4).

**G7. Sessions = gap > 2 min between minutes (not between fixes).**
- **Decided:** 04 brief and Marc, 2026-10-28.
- **Code:** `gps.py:12`, `:34-36`; `build.py:92`.
- **Effect:** 8,729 ExtraSensory sessions: median 4 minutes, 25% are a single minute, max 7,875. Consecutive-fix
  values are computed within a session only.
- **Reversible:** yes (`ts` is stored).
- **Risk:** (b) WAIIS batches may define sessions differently.

---

## 9. SHL windows and GPS thinning

**H1. One 15 s window per clock minute, centred (`ts` + 22.5 s … + 37.5 s).**
- **v2:** the continuous 25 Hz hips stream is stored as well (`stream`, V7).
- **Decided:** 04 brief and stop-point 1 (Marc, 2026-10-28).
- **Code:** `build.py:224-253`.
- **Why:** mimic the fragments.
- **Effect:**
  - User 1: 36,825 windows of 36,968 minutes; 118 `edge` (the window doesn't fit in the recording) and 25
    `nan_gap`, both without a window. **45 s of every minute are discarded.**
  - Users 2–3: 3,015 windows; 7 `edge`, 2 `nan_gap`.
- **Reversible:** no.
- **Risk:**
  - (a) ExtraSensory windows sit at the centre of a snippet that starts at `ts`, SHL windows at `ts` + 22.5 s;
    harmless unless a model uses time;
  - (b) SHL has no fragment-to-fragment gaps.

**H2. SHL label = one coarse label over all 1,500 samples of the window.**
- **Code:** `build.py:232-243`, `:332`.
- **Effect:**
  - User 1: `ok` 23,100 (23,099 with a window); `null` 13,056; `mixed` 694.
  - Users 2–3: `ok` 2,366; `null` 586; `mixed` 65.
  - Mixed windows with a majority ≥ 80%: 219 (User 1) and 38 (Users 2–3). Their majority class is null 349, walk 86,
    still 59, bike 56, …
- **Reversible:** yes (`majority_cls`, `majority_share`, and the window is stored).
- **Risk:** (a) ExtraSensory has no "mixed" minutes (one label per minute), so transitions exist only in SHL.

**H3. SHL platform `android`, position `hips`.**
- **Code:** `build.py:331`.
- **Reversible:** n/a.
- **Risk:** (a) these columns are constant for SHL, so a model must not use them as features.

**H4. Two SHL GPS variants.**
- **v2:** the 1 Hz track (`gps_1hz`) and the unblanked speed and bearing (`gps_unblanked`) are stored (V9).
- **Decided:** Marc, 2026-10-28 (`a711d65`).
- **Code:** `build.py:259-309`; `gps.py:58-95`.
- **Why:** a single cadence would let a model tell the datasets apart (like `ctx_n` in 02).
- **The two variants:**
  - `gps_waiis_cadence`: the 1 Hz track thinned by the speed over ~10 s at the candidate fix: ≥ 3 m/s every 10 s,
    ≥ 0.5 m/s every 20 s, otherwise every 180 s.
  - `gps_per_minute`: the best-accuracy fix inside the window, latest on ties.
  - In both, speed and bearing come from the 1 Hz track over ±1.5 s, never across a gap > 5 s; bearing is −1 under
    1 m of movement.
- **Effect:**
  - User 1: 80,892 of 1,377,605 fixes kept (5.9%); 23,645 per-minute fixes.
  - Users 2–3: 5,550 of 101,516 kept; 1,749 per-minute fixes.
  - Fix interval (WAIIS cadence), median per class: car, bus, bike, train and subway 10 s; walk and run 20 s; still
    71 s (User 1).
- **Reversible:** no (the 1 Hz track is not stored).
- **Risk:**
  - (b) the cadence model is our assumption (the real trigger is unknown);
  - (b) SHL speed comes from positions, not from the GPS Doppler speed as on a phone;
  - (a) models that train on both datasets must use `gps_per_minute` (stated in the README).

**H5. Speed and bearing blanked on 23% of kept SHL fixes.**
- **v2:** replaced. Blanking now depends on the 1 Hz speed at the fix (V2).
- **Decided:** 04 brief; bearing added by Marc, 2026-10-28 (`9ef9424`).
- **Code:** `build.py:301-304` (`rng(0)`, separately per table).
- **Effect:**
  - `gps_waiis_cadence`: 18,551 blanked (User 1) and 1,295 (Users 2–3); 987 and 64 fixes are −1 naturally (no
    neighbour within ±1.5 s).
  - `gps_per_minute`: 5,340 and 395 blanked.
- **Reversible:** no. `speed_blanked` says which fixes, but the values are overwritten.
- **Risk:**
  - (a) the blanking is uniform over classes in SHL, while in ExtraSensory missing speed is strongly
    class-dependent (G3). The two datasets disagree on what −1 means;
  - (b) production's per-class rate is unknown.

**H6. SHL per-minute location coverage.**
- **Effect** (examples with a fix):

  | | still | walk | run | bike | car | bus | train | subway |
  |---|---|---|---|---|---|---|---|---|
  | User 1 | 39.4% | 85.9% | 99.8% | 99.3% | 83.2% | 89.7% | 49.0% | 30.6% |
  | Users 2–3 | 46.4% | 86.0% | 100% | 89.9% | 91.7% | 80.0% | 38.1% | 13.4% |
- **Reversible:** n/a (natural gaps are kept as gaps).
- **Risk:** (a) **"no fix" points to still and rail in SHL**, but not in ExtraSensory (G2).

---

## 10. Format

**F1. Parquet (zstd) tables per source, join key `user, ts`, version folder `es_bench_v1/`, manifest with sha256.**
- **v2:** folder `es_bench_v2/`; the reader takes `version=` (default v2).
- **Decided:** 04 brief, `a711d65`.
- **Code:** `build.py:47-69`, `:448-475`; reader `src/tmd/bench.py`.
- **Effect:** 15 files, 1.77 GB; rebuilds reproduce the hashes.
- **Reversible:** n/a.
- **Risk:** none.

**F2. Windows as float32 m/s², gravity included, 3 fixed-size lists (`x`, `y`, `z`).**
- **Decided:** Marc (float32, every minute), 2026-10-28.
- **Effect:** none beyond float32 rounding.
- **Risk:** (b) depends on whether WAIIS sends gravity (U1).

**F3. `y` = 0 for every minute that is not included; `cls` keeps the rule's class.**
- **Code:** `build.py:82`, `:334`.
- **Effect:** 62,499 capped still minutes have `cls = still`, `y = 0`.
- **Reversible:** yes.
- **Risk:** (a) a reader that uses `cls` without the inclusion flag gets uncapped still, `SLEEPING` minutes and
  dropped labels.

**F4. All source label columns are kept** (51 `label:*`, 116 `orig:*`), together with `reason`, `position`,
`resting` and `moving`.
- **Reversible:** this is what makes the label decisions reversible.
- **Risk:** (a) a model must not read them as features.

**F5. Times differ between the datasets.**
- ExtraSensory `ts` is the recording start (CSV minute); fix `time` = `ts`.
- SHL `ts` is the clock minute; fix `time` is the real time.
- **Risk:** (a) `time − ts` is 0 in ExtraSensory and non-zero in SHL, so a model must not use it.

---

## 11. es_bench_v2 (2026-10-28, step 3)

These decisions are by Marc, 2026-10-28.
- Code is at commit `b1e2a9c`, built from that commit. The manifest is `git_dirty` only because of uncommitted
  doc edits outside 04.
- v1's 359,240 windows are byte-identical in v2.
- Official ExtraSensory examples: 149,681 (v1) → **148,701** (+405 hole-trimmed, −1,385 implausible among the
  included windows).

**V1. `acc_implausible`: window median |acc| outside 7–13 m/s² after unit conversion; out of the official set.**
- **Code:** `build.py:50`, `:101-108`; `resample.py:126-129`; `bench.py:87-99`.
- **Why:** BEF6C611's units are not a clean g / m/s² mix (U1).
- **Effect.**
  - **ExtraSensory**, included windows: 1,408 flagged:

    | | still | walk | car | run | bus | train |
    |---|---|---|---|---|---|---|
    | flagged | 1,020 | 182 | 109 | 95 | 1 | 1 |

    - Below 7 m/s²: 721 windows, **all BEF6C611**.
    - Above 13 m/s²: 687 windows, of which BEF6C611 471 and other users 216 (run 95, walk 108, still 6, car 5,
      bus 1, train 1).
    - **BEF6C611 accounts for 1,192** (of its 1,544 windows). Next come 5119D0F8 with 56 (walk 45) and the test
      users 74B86067 (44, all run) and 098A72A5 (22, all run).
  - **SHL** (all above 13 m/s²):
    - User 1: 1,124 windows, of which run 566 of 1,047, walk 377 of 3,015 and bike 44 among labelled windows.
    - Users 2–3: 0.
- **Reversible:** yes. `acc_median` and the window are stored; `examples()` applies the flag.
- **Risk:**
  - (a) **the upper bound removes genuine running and brisk walking.** Median |acc| for SHL run is 13.2 m/s² and
    for ExtraSensory run p90 is 13.3 m/s². ExtraSensory test run drops from 123 to 57 examples, SHL User 1 run from
    1,047 to 481. The flag is then class-dependent.
  - (c) the same loss hits the test set.
  - Alternative (open question): keep the 7 m/s² lower bound, and handle BEF6C611 at user level or raise the upper
    bound (e.g. 20 m/s²). That keeps the run windows; the 471 BEF6C611 windows above 13 m/s² would then need the
    user-level rule.

**V2. SHL speed and bearing blanking driven by the 1 Hz speed at the fix.**
- **Rule:** p = 0.95 · exp(−v / v0). v0 = **1.954 m/s**, calibrated once on SHL User 1's 80,892 WAIIS-cadence
  fixes so that 23% of them have no speed. Fixes already without speed count. The same curve is used for every
  table, `rng(0)`. The label is never used.
- **Code:** `gps.py:97-117`; `build.py:368-414`.
- **Why:** in production (and in ExtraSensory) speed is missing mostly when slow.
- **Effect:** speed missing per class, against ExtraSensory examples:

  | | still | walk | run | bike | car | bus | train | subway | all |
  |---|---|---|---|---|---|---|---|---|---|
  | SHL U1 `gps_waiis_cadence` | 73.6% | 48.3% | 28.1% | 13.4% | 5.3% | 21.5% | 9.0% | 14.4% | **23.0%** |
  | SHL U1 `gps_per_minute` | 87.4% | 48.2% | 28.6% | 17.9% | 11.5% | 39.9% | 17.8% | 31.1% | 43.3% |
  | ES, iPhone | 66.9% | 35.3% | 46.3% | 11.4% | 9.0% | 10.2% | 8.6% | – | – |
  | ES, Android | 91.6% | 61.6% | 68.4% | 16.8% | 16.0% | 37.1% | 43.2% | – | – |
- **Reversible:** yes (`gps_unblanked`, V9).
- **Risk:**
  - (a) missing speed is now class-dependent in both datasets, so it is a still / walk feature in both;
  - (b) the curve's shape (exponential, p0 = 0.95) is our assumption. Production's per-class rate is unknown.

**V3. Pre-training accessors exclude the reserved test users by default.**
- **Code:** `bench.py:68-84`, `:102-145`. `tests/test_bench_reserved.py` fails if a reserved user reaches any
  accessor in `PRETRAINING_ACCESSORS` (18 tests pass on the built data).
- **Why:** reserved test users never enter pre-training (`splits/README.md`).
- **Effect:**
  - `examples()` still includes test users (column `split`), because evaluation needs them.
  - The reserved set is `splits/reserved_test.json` (the v1 pick) until V10 is approved.
- **Reversible:** `include_reserved=True` (evaluation only).
- **Risk:** (a) `read()` is unfiltered by design.

**V4. `quiet`: std of |acc| over the window < 0.03 m/s². Descriptive, never used for inclusion.**
- **Code:** `build.py:51`, `:101-108`.
- **Why:** 0.03 m/s² catches 78% of still windows on a table, and stays above the sensor noise (iPhone ≈ 0.008,
  Android ≈ 0.010; v1 windows).
- **Effect:** share of quiet examples:
  - ExtraSensory by class (iPhone / Android): still 55% / 54%; walk 12% / 15%; **run 70% / 0.3%**; bike 12% / 6%;
    car 6% / 14%; bus 6% / 7%; train 0% / 15%.
  - ExtraSensory by position: on a table 64–71%, in a pocket 20–26%, none reported 46–50%.
  - SHL: still 41–53%; moving classes ≤ 8%.
- **Reversible:** n/a. The 20% cap is unchanged.
- **Risk:** (a) quiet windows in moving classes are probably label noise. The phone was not carried (e.g. 70% of
  iPhone run, 161 of 231 examples).

**V5. Holed snippets: the longest clean stretch is kept** (C3 approved).
- **Code:** `resample.py:88-123`.
- **Effect:** 824 snippets (Android 610, iPhone 214).
  - 428 included and 405 official examples recovered: still 307, walk 46, car 27, bike 16, bus 9.
  - By split (included): dev 340, test 88.
  - Top users: 4E98F91F 75, 481F4DD2 57 (test), 99B204C0 47.
- **Reversible:** flag `hole_trimmed`.
- **Risk:** (b) small. The window comes from the longest stretch, not the centre of the original snippet.

**V6. ExtraSensory snippets stored in full at 25 Hz (`snippets`), incl. short and time-repaired ones; never
official.**
- **Code:** `build.py:137-171`; `resample.py:88-123`.
- **Effect:**
  - 376,952 snippets, 2.06 GB. Missing: 290 without a file and 104 with a span > 40 s (102 as in v1, plus 2
    time-repaired snippets whose span is over 40 s after sorting).
  - **Short:** 15,853 (median 9.3 s, 2.7–15.0 s), including 150 that are also time-repaired. Included short
    minutes: still 4,982, walk 1,587, car 764, bike 478, bus 153, train 95, run 53.
  - **Time-repaired:** 1,185.
- **Reversible:** n/a (adds data).
- **Risk:** (a) a model trained on short snippets would need padding or another window length; not comparable to
  the official set.

**V7. SHL continuous 25 Hz hips stream (`stream`).**
- **Content:** one row per sample: `t_ms`, `x`, `y`, `z`, coarse label, stretch number. Only NaN-free stretches of
  ≥ 15 s.
- **Code:** `build.py:329-339`.
- **Effect:** User 1 55.3 M samples (614 h, 791 MB); Users 2–3 4.5 M (63 MB). Each official window equals a slice of
  the stream (checked).
- **Risk:** (a) the stream has no fragment gaps, unlike the app's recordings.

**V8. SHL User 1 hand, torso and bag windows on the 4phones days (`accelerometer_other_positions`).**
- **Content:** 5,964 windows (36 MB), with label status and flags. Never official.
- **Risk:** (b) more positions; (a) only 3 days.

**V9. SHL 1 Hz GPS track (`gps_1hz`, incl. altitude) and the speed and bearing before blanking
(`gps_unblanked`).**
- **Effect:** `gps_1hz` 1,377,605 + 101,516 rows (68 MB); `gps_unblanked` 104,537 + 7,299.
- **Risk:** (a) the true speed must not be used as a feature next to blanked tracks.

**V10. Re-pick of the 12 ExtraSensory test users: proposed, NOT in use.**
- **Code:** `propose_test_v2.py`.
- **Writes:** `splits/reserved_test_v2.json` and `splits/es_users_v2.csv` in this folder. The old files are
  untouched.
- **Rule** (Marc):
  - eligible = all users except those with > 10% implausible windows (only BEF6C611, 79%);
  - 6–7 iPhone users;
  - iPhone share and location coverage of the test examples within ±3 points of the figures for all users;
  - users with ≥ 5 examples: bus ≥ 5, bike ≥ 4, run ≥ 3, train 1–2, still / walk / car ≥ 5;
  - among the 9,972 subsets that pass (of 3 M draws, seed 0), class shares closest to 12/59.
- **Effect:** old against new (comparison in step 3 of the README):
  - run users 2 → 4, bike 4 → 6, bus 5 → 6, train 1 → 2;
  - location coverage 94.9% → 85.4% (all users 82.9%);
  - iPhone share of examples 66.4% → 62.1% (61.6%).
  - Only 098A72A5 is in both.
  - Dev folds are rebuilt; 2 folds have no train (3 dev train users).
- **Risk:**
  - (c) **the 11 new test users were dev users in 02**, used for training and model choices there (R2, step 3,
    the ExtraSensory-only model). The old test users were never used;
  - (a) the run counts depend on V1's upper bound.

## Where data is dropped rather than flagged

*(v1. In v2, items 1, 5 and 6 are stored, apart from 290 missing files and 104 snippets with a span > 40 s. Item 2
is partly stored: full snippets are kept, but the raw sample timestamps are not. Item 4 is partly stored: the
continuous hips stream and the User 1 hand, torso and bag windows of the 4phones days are kept. See V6–V9.)*

1. **ExtraSensory snippets that fail cleaning or the 15 s rule.** The minute row stays with `acc_status`, but no
   window is stored. That is 18,106 minutes: under 15 s 14,581; hole 1,946; time 1,187; no file 290; span 102.
2. **Signal outside the 15 s window:** ~8.8 s of every iPhone snippet and ~1.3 s of every Android snippet. The raw
   sample timestamps are not stored either; the irregular sampling is replaced by interpolation.
3. **Sensors not in the benchmark:** gyroscope, magnetometer, audio, watch and phone-state features (they remain
   in the raw files).
4. **SHL signal:** 45 s of every minute; the hand, torso and bag phones; all non-acc sensors; the windows of `edge`
   (125) and `nan_gap` (27) minutes.
5. **SHL 1 Hz GPS track:** only the thinned and per-minute fixes are stored. Altitude, satellite, cell and Wi-Fi
   data are not used.
6. **SHL blanked speed and bearing:** overwritten with −1 (`speed_blanked` marks them, the values are lost).
7. **SHL User 1 duplicates:** 220617, 260617 and 270617 from `release/User1`, kept once from `User1_4phones` (no
   information lost).

Labels, exclusions, caps, location and splits are all flags; nothing there is deleted.

## Open questions that need facts about WAIIS

1. **When does the app record?** Continuously, duty-cycled or per trip? How long is a fragment (≥ 15 s?) and how
   long is the gap? Does it record at night or while the phone lies still? This bears on W1, X1 and X3.
2. **Sampling.** Is it exactly 25 Hz, and with how much jitter? Are there holes or backwards timestamps? Does the
   phone low-pass before decimating? (U2, C1–C3)
3. **Units and gravity.** Raw acceleration with gravity, or linear acceleration? m/s² or g? (U1, BEF6C611)
4. **Platform and device mix** (iOS against Android, models). It sets how much the platform effects in U2, W2 and
   G1 matter, and whether a 7/5 iPhone/Android test set is representative.
5. **Carrying positions** and how often the phone rests (table, car mount) while the user moves. (X2, X3)
6. **Which modes matter.** Motorbike, e-scooter or skateboard, plane, boat, tram or light rail against train
   against metro, taxi or ride-share against car, driver against passenger. (L3, X5, X6)
7. **GPS.**
   - What triggers a fix (time or distance filter, activity), and what the cadence is when still.
   - Whether fixes disappear indoors or underground.
   - Speed from Doppler or not; whether min, max or only the instantaneous speed is sent.
   - The missing-speed rate per mode.
   - Bearing availability.
   - Accuracy filtering.
   - Privacy masking near home.
   - How often location permission is off.
   (G1–G7, H4–H6)
8. **Labels in WAIIS.** Who labels (the user, afterwards?), at what granularity (minute, trip leg), and how
   transitions are handled. (L8, H2)
9. **Batching.** What is one upload (a trip, a day)? It defines sessions and the context available for smoothing.
   (G7)
