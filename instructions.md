# Instructions: a golden transport-mode dataset in waiis-mob-activity (from `waiis_bench_v2`)

*Written 2026-11-05 from transport-mode-detection (TMD: experiments 04, 08, 09, 10, `LICENSES.md`, `src/tmd/readers`,
`src/tmd/bench/build.py`) and a read of waiis-mob-activity's code (WMA). Paths starting `src/wma`, `configs/`, `docs/`
are WMA's; `experiments/`, `splits/`, `src/tmd` are TMD's.*

## 1. Goal

**One golden dataset to start fresh in WMA**, built the WMA way, with the sources and label rules of TMD's frozen
benchmark `waiis_bench_v2`. It is used to train several label-suggestion models and compare them on the same splits
and windows. It is **not** a reproduction of TMD's numbers: WMA's own window policy, resampling and dev folds apply,
TMD's rules are stored as flags or strata (never deletions), and caps are a later sampling choice.

| WMA dataset (proposed id) | Content | Role |
|---|---|---|
| `extrasensory_golden_v1` | ExtraSensory, 60 users (BEF6C611 flagged), accelerometer + published location | train / dev; 12 reserved test users |
| `shl_golden_v1` | SHL User 1 (all hips days; hand / torso / bag on the 3 preview days) and Users 2–3 (preview, all 4 positions), accelerometer + 1 Hz location | User 1 train / dev; Users 2–3 reserved (rail check) |
| `nor_tmd_v1` | NOR-TMD, all installations (iOS flagged: ~5 Hz) | train / dev; 14 reserved installations |
| `collecty_v1` | Collecty, 14 users (Android) | train / dev |
| **`golden_v1`** (composite, `wma dataset merge`) | the four above | the golden dataset: E1's training pool (ExtraSensory + SHL User 1 + NOR-TMD, 9 classes) plus Collecty, i.e. all of `waiis_bench_v2` |

- Reserved test people are **in** the datasets and kept out by the **split**: the reserved panel v3 for every source
  (section 6). WMA draws new dev folds.
- In TMD, E1 is the model trained on ExtraSensory + SHL User 1 + NOR-TMD with 9 fine classes (tram included);
  experiments 11–12 showed its limits on WAIIS days (metro read as walking or still), which is why the next model
  adds Collecty, then labelled WAIIS days. Appendix A has E1's details for a later port; it is not part of the build.

## 2. Before anything: licences and decisions

WMA wants a sign-off per dataset (ADR-008 / ADR-051 style) and `privacy_tier` and `license` in each config.

| Dataset | Licence | Commercial use | Credit (cite) |
|---|---|---|---|
| ExtraSensory | none stated on the page; WMA: `citation-required`, models `research_only` (ADR-051) | ask UCSD first | Vaizman, Ellis, Lanckriet, IEEE Pervasive Computing 16(4), 2017 |
| SHL | **non-profit and research only** (`<raw>/shl/LICENSE.txt`); WMA: `research-only` (ADR-008) | needs a licence from the originators (daniel.roggen@ieee.org) | Gjoreski et al., IEEE Access 2018 (+ Ciliberto et al., SenSys 2017); use the full name "The University of Sussex-Huawei Locomotion and Transportation Dataset" once |
| NOR-TMD | **CC BY 4.0** (Kaggle page) | verified as permitted 2026-11-02 | Skretting, Grønli, Mukkamala, J. Intell. Transp. Syst. 30(4), 2025, doi:10.1080/15472450.2025.2515567; dataset: Skretting & Grønli (2025), Kaggle |
| Collecty | Kaggle says "Unknown"; article CC BY 4.0 | verified as permitted 2026-11-02 | Erdelić, Erdelić, Carić, Data in Brief 50, 109481, 2023, doi:10.1016/j.dib.2023.109481 |

**The composite inherits the most restrictive terms** (SHL research-only; ExtraSensory ask-first). WMA's merge does
not check licence compatibility.

**Decisions, as ADRs:**
1. **Axis convention: decide before any build** (WMA has none). Raw-axis models (InceptionTime, HART) see
   ExtraSensory's iPhones with the opposite sign to every Android source unless one convention is fixed. TMD's
   choice (09 rule A1): everything in **m/s² and the Android axis convention**; iOS data × −1 after unit
   conversion; ExtraSensory iPhone g → m/s² is × −9.80665. Only ExtraSensory's iPhones are affected here (SHL,
   NOR-TMD Android and Collecty are Android); NOR-TMD iOS is flagged and unused. Feature models on |acc| don't care.
   Record the conversion per session (`sign_converted` in strata or sessions) so it can be undone.
2. **Model vocabulary.** TMD's 9 fine classes map one-to-one onto WMA `taxonomy.yaml` `classes` (still, walking,
   running, bike, car, bus, train, subway, tram); nothing to reorder. TMD's primary score merges them at scoring
   time into six (still, foot = walking + running, bike, car, bus, rail = train + subway + tram).
3. **Conflicting labels are not labels** (NOR-TMD `inside` / `outside`, Collecty `Walk`: they mix still and
   walking). Modes outside the 9 classes (boat, e-scooter) are not labels either. Their time stays in the sessions
   as unlabelled time, with the source label kept as a stratum (section 4).

## 3. What to download

There is no downloader in WMA: put files by hand under `<WMA_DATA_ROOT>/raw/<source>/` and record exact bytes and
sha256 in the dataset config (`_verified_archive` checks bytes, sha256, ZIP readability and an optional
`<file>.complete` marker). TMD's NOR-TMD and Collecty zips were deleted after its build, so **those two must be
downloaded again**; their sha256 were never recorded, so record them now.

### 3.1 ExtraSensory → `raw/extrasensory/`

From `http://extrasensory.ucsd.edu/data/` (sha256 checked by TMD 2026-10-28):

| File | Bytes | sha256 | Needed |
|---|---|---|---|
| `primary_data_files/ExtraSensory.per_uuid_features_labels.zip` | 225,374,973 | `412d307bdf54a60dff81a705c2f25833889a46dca64a184e3b588ad99c69cb56` | yes (labels) |
| `additional_data_files/ExtraSensory.per_uuid_original_labels.zip` | 992,570 | `7c017f75cbf3ca6d157a6ecfd8aacb1173493d9864772ab648ba27ac81373e79` | yes (`ON_A_TRAIN`, passenger rules) |
| `additional_data_files/ExtraSensory.per_uuid_absolute_location.zip` | 2,303,359 | `09d06c2bf1f69d306b2e529d50d5fe2c8c5d56a832ddd3b64c8ee3e4908b750e` | yes (raw location) |
| `raw_measurements/ExtraSensory.raw_measurements.raw_acc.zip` | 6,475,399,720 | `61a487d027db28a3dc6ae5ea96eeda01b158dbba37b9d3e6439508b5cf323c01` | yes |
| `cv5Folds.zip` | 10,803 | `66183635452d8c1b250b570fba3000296bf29e4d54eda6af270d2b4f5a3b9ee6` | yes (platform per user) |

Not needed: `proc_gyro` (9.3 GB), `raw_magnet` (4.1 GB); the WAIIS app records no gyroscope. WMA's
`extrasensory_v1.yaml` (and its location overlay, `extrasensory_v2`) already list these archives; reuse the entries.

### 3.2 SHL → `raw/shl/`

From shl-dataset.org, the **preview release** (TMD's layout, `src/tmd/readers/shl.py`):

    release/User1/<rec>/Hips_Motion.txt, Hips_Location.txt, Label.txt   91 day folders (rec DDMMYY or mDDMMYY)
    release/User1_4phones/<rec>/{Hand,Hips,Torso,Bag}_Motion.txt, ...    5 days (3 not duplicated in User1/)
    release/User2_4phones/, release/User3_4phones/                        3 days each, all 4 positions
    LICENSE.txt, doc/doc_dataset.pdf

- Motion: 100 Hz grid; col 1 time ms, 2–4 acc m/s². Label.txt col 2 = coarse label 0 Null, 1 Still, 2 Walk, 3 Run,
  4 Bike, 5 Car, 6 Bus, 7 Train, 8 Subway.
- WMA already has `shl_preview_v2.yaml`, `shl_user1hips_v3.yaml` and `source_mappings.shl_preview`; reuse them and
  check coverage against the list above.
- Not used: the challenge releases (User 1's 59 days with all positions; 18 GB for the 2023 raw streams).

### 3.3 NOR-TMD → `raw/nor_tmd/nor-tmd.zip`

    kaggle datasets download -d scholarone/nor-tmd -p <WMA_DATA_ROOT>/raw/nor_tmd/

- Kaggle `scholarone/nor-tmd`, **version 1** (updated 2025-01-12). Zip 7,359,659,643 bytes; one member
  `nor_tmd.csv`, 73,611,929,573 bytes. Keep it zipped and stream it. The Kaggle CLI fetches the latest version:
  check it is still v1 with these sizes.

### 3.4 Collecty → `raw/collecty/collecty.zip`

    kaggle datasets download -d merdelic/dataset-for-multimodal-transport-analytics -p <WMA_DATA_ROOT>/raw/collecty/

- Kaggle `merdelic/dataset-for-multimodal-transport-analytics`, **version 4** (updated 2023-07-15,
  doi:10.34740/kaggle/dsv/4976711). Zip 5,904,835,122 bytes; 454 trajectory files
  `Collecty data/archive/<user>_<trajectory>.txt` plus a consent form, 24,864,805,329 bytes unzipped. Keep it zipped.

Disk: about 15 GB of zips.

## 4. Adapters and build configs

One adapter, typed config class and dispatch branch per new source (WMA has no plugin registry). Each adapter emits
WMA's canonical tables (`sessions`, `acc` with `t_ms, ax, ay, az, gap`, `labels` half-open, `strata`, `gps`), m/s²,
phone frame, gravity included, native timestamps, the axis convention of decision 1. New public datasets use
`identity_mode: public_namespaced` and a `person_namespace` (ADR-071), `expected_users` and an inventory count;
`converter_version >= 3` needs `gps_sanity`.

**General rules for every adapter:**
- **Flags, never deletions.** TMD's exclusion and plausibility rules are stored as strata (time spans with
  columns) or label metadata, and never applied in the dataset. Which of them define training and evaluation sets is
  a later choice (selection or sampling), so models can be compared with or without them.
- **Keep unlabelled time.** Sessions cover all recorded accelerometer time, not only labelled spans. WMA windows over
  time with no label row get label `unknown` (purity 0; `majority_label`, `src/wma/datasets/domain/windows.py:121`)
  and stay in the session; `unknown` is `trainable: false` in `taxonomy.yaml`. That keeps context for an HMM or other
  sequence models. So: never emit a label row for conflicting or out-of-vocabulary labels (decision 3); put the
  source label in a stratum instead.
- **Caps are not applied.** TMD's per-user cap per class (09 C1) and resting-still cap (04 X4) are later sampling
  choices; the adapter may store what they need (`resting`, trip id), nothing more.
- **GPS raw only.** Keep the sources' own location data: SHL's 1 Hz `Hips_Location.txt`, ExtraSensory's published
  per-minute absolute location. No thinning, no blanking, no derived speed in the dataset. NOR-TMD and Collecty
  publish no GPS, so a GPS model can train on ExtraSensory and SHL only.

### 4.1 ExtraSensory (`src/wma/datasets/adapters/extrasensory.py`; new `label_rule_version`, e.g. `es_golden_v1`)

WMA's `es_v1` already detects units per burst (median |acc| < 3 → g; TMD U1), the 15 s minimum span, 0.5 s holes, the
7–20 m/s² plausibility, and refuses walking / running when the phone is on a table or away. TMD's label rules (04
register; v2 additions from 09):

| TMD rule | What | Store as |
|---|---|---|
| L1 | one class per minute by priority; NaN = not reported | label rule |
| L2 | train from the original `ON_A_TRAIN` label (needs the original-labels zip) | label rule |
| L3 | "passenger" + bus / train → that vehicle, else car | label rule |
| L4 | retrospective labels kept | `strata.retrospective` (exists) |
| X1 | `SLEEPING` still minutes | `strata.sleeping` (exists) |
| X2 | walk / run / bike when the phone isn't carried | `strata.placement` (exists) |
| X4 | resting still (cap later) | stratum `resting` |
| X5, X7 | two moving classes in a minute; plane, boat, elevator, treadmill, … | no label row; source labels in a stratum |
| X6 | motorbike | diagnostic, as `es_v1` |
| X8 | user BEF6C611 (inconsistent accelerometer processing) | flag on the subject (stratum or `parameters.skipped`), not dropped |
| L5 (09) | 7 run users who move like walking (median std \|acc\| of their run windows < 1.712 m/s²) | flag on their run labels (`run_user_flagged`; the user list from TMD `bench.read("extrasensory","labels")`) |
| A1 (09) | iPhone axes (decision 1) | `acc_canonicalization` + `sign_converted` |
| X3, U3, R1 | quiet walk / run / bike; median \|acc\| outside 7–20 m/s²; real rate < 24 Hz | window-level quality flags: compute per window at window time (or per minute as strata), never drop |

Platform per user from `cv5Folds.zip`.

### 4.2 SHL (existing adapter)

- Users User1–3, namespace `shl` (`shl.User1`, …). User 1's duplicated days kept once (TMD S4).
- Positions: hips for every User 1 day; hand / torso / bag on the 3 preview days not duplicated
  (`device_position`); Users 2–3 all 4 positions. Default training position is a later choice (TMD: hips).
- Labels: coarse 1–8 via `source_mappings.shl_preview`; 0 Null → no label row (unlabelled time).
- GPS: the 1 Hz hips track as is.

### 4.3 NOR-TMD (new adapter `nor_tmd.py`; port of TMD `src/tmd/readers/nor_tmd.py` and the trip logic in `tmd.candidates`)

- **Input:** one long CSV in the zip, one row per sensor event: `installationId, journeyNumber, tripNumber,
  manufacturer, model, timestamp, transportType, typeInteger, typeString, deviceLocation, value_0..5,
  activity_recognition, OS`. Rows are not sorted by time; timestamps have no time zone (section 6.3).
- **Subjects:** `installationId`, namespace `nor_tmd`. No person key exists (152 installations ≈ 101 participants);
  store `manufacturer` and `model` per session: the reserved split groups by device model.
- **Sessions:** one per (installation, journey, trip); keep `OS` and `deviceLocation`.
- **Sensors:** accelerometer only (`value_0..2`, m/s², gravity included). Drop rotation vectors, quaternions,
  attitude, activity recognition; gravity, gyroscope, magnetometer, pressure not needed. **NOR-TMD publishes no
  GPS** (TMD 08).
- **iOS installations:** ~5 Hz; keep and flag (`rate_below_24hz` stratum), never trainable by default.
- **Labels** (`transportType`, lower-cased): bicycle → bike, bus → bus, car → car, metro → subway, train → train,
  **tram → tram**. `inside`, `outside` (conflict), boat, e-scooter: no label row; source label as a stratum
  (`e_scooter` is in WMA's human vocabulary, not a model class). No still or walking labels come from NOR-TMD.
- Overlapping trips: one row per user-minute; the earlier trip wins where both have data (TMD R3); flag overlaps.
- Version the label mapping (`label_rule_version: nor_tmd_v1`), or add string mappings to `taxonomy.yaml`
  `source_mappings` (today integer codes only).

### 4.4 Collecty (new adapter `collecty.py`; port of TMD `src/tmd/readers/collecty.py`)

- **Subjects:** the `<user>` number of the file name, namespace `collecty`; sessions = trajectory files (+ clock
  piece).
- **Samples:** a 10 ms timer copies the latest value of each sensor; `lChangeTime<sensor>` is the sensor's last real
  event. **The real samples are the distinct `lChangeTimeacc` values** (first row of each); the rest are held
  values. **Acceleration = `acc*` (linear) + `grav*`** (gravity), to match the other sources (TMD A2).
- **Clock repair (TMD K1):** undo 12-hour wraps (36 files); split pieces at other backward steps; flag minutes where
  pieces overlap (`time_overlap`).
- **Duplicates (TMD K2):** byte-identical files: one copy within a user; across users keep both and flag (15 files,
  2 cross-user pairs). WMA's merge dedups by `content_hash`: check it doesn't silently drop the cross-user pair.
- **Labels:** Run → running, Bike → bike, Car → car, Bus → bus, Train → train, **Tramway → tram**. **Walk** (includes
  standing), E scooter, Unknown: no label row; source label as a stratum.
- **No GPS:** the app recorded it, but it is not published (TMD 08).

### 4.5 Configs and build

`configs/datasets/<id>.yaml` per source, as `extrasensory_v1.yaml`: `dataset_id`, `source`, `identity_mode:
public_namespaced`, `person_namespace`, `privacy_tier: public`, `license` (section 2), `raw_subdir`, `expected_users`,
`expected_records`, archives with bytes + sha256, `acc_hz`, `gap_threshold_s`, rule versions, `gps_sanity`.

    uv run wma dataset build --config configs/datasets/nor_tmd_v1.yaml
    uv run wma dataset show nor_tmd_v1

Then the composite `configs/datasets/golden_v1.yaml` (`CompositeDatasetConfig`):

    dataset_id: golden_v1
    source: composite
    source_datasets: [extrasensory_golden_v1, shl_golden_v1, nor_tmd_v1, collecty_v1]
    privacy_tier: public
    license: research-only          # most restrictive component (SHL)
    converter_version: 1

    uv run wma dataset merge --config configs/datasets/golden_v1.yaml

## 5. Windows

WMA's own window policy, e.g. `configs/windows/w15_h15_25hz.yaml` (15 s, 25 Hz, polyphase, `min_purity` 0.8),
windows from the session start with a fixed hop (`plan_windows`, `src/wma/datasets/domain/windows.py:91`). No
alignment to clock minutes or ExtraSensory snippets.
- Unlabelled time gives `unknown` windows that stay in their sessions (section 4); a partly labelled window below
  `min_purity` is `mixed` (also not trainable).
- ExtraSensory sessions are ~20 s bursts per minute: a 15 s window with a 15 s hop gives about one window per burst;
  check the yield in the count check.
- The window-level quality flags (X3, U3, R1) are computed per window and stored with the window record or a
  selection document, not applied.

## 6. Splits

### 6.1 Reserved panel v3, for every source

The test people are fixed (TMD `splits/reserved_test_v3.json`):

| Source | Reserved | Rule |
|---|---|---|
| ExtraSensory | 12 test users | as in `reserved_test_v3.json` |
| SHL | Users 2 and 3, all days and positions | whole persons |
| NOR-TMD | 14 installations = **all installations of 8 device models**: Nokia 8.1, ONEPLUS A6003, SM-A505FN, SM-A725F, SM-G970F, SM-N981B, XQ-AS52, moto g 5G plus | the device model is the person-linkage guard (no person key): every installation of a reserved model is reserved, of any OS |
| Collecty | none | training / dev only |

WMA draws **new dev folds** over everyone else. TMD's dev fold files are not carried over.

### 6.2 Code needed: `--reserved-people` is ExtraSensory-only today

Checked in the code: the canonical person split `make_stratified_person_folds`
(`src/wma/datasets/application/splits.py:748`; CLI `src/wma/entrypoints/cli.py:1626–1650`) cannot take this
panel:
- the reserved file's sha256 must equal the constant `EXTRASENSORY_RESERVED_SHA256` (l.766);
- reserved people are read only from `payload["extrasensory"]`, key `uuid` (l.775);
- `--dev-people` is mandatory and pinned too (`EXTRASENSORY_DEVELOPMENT_SHA256`, l.768);
- every subject of the dataset must be declared in those files (l.787), so a composite fails;
- `--folds` must be 5 and `--balance-by` exactly `class_minutes,platform` (l.759, l.762).
The other path, `make_rotating_person_folds` (l.874), has no reserved option (one person per fold).

**Change (proposed):** a generic reserved-panel split, e.g. `wma split create golden_v1 --by-person --folds 5
--reserved-panel configs/splits/golden_reserved_v3.json --balance-by class_minutes --seed <s>`:
1. **Panel file format:** `{"description", "subjects": [namespaced subject ids], "groups": [{"key":
   "nor_tmd.model", "values": [the 8 models]}]}`. Subjects are namespaced as the datasets store them
   (`extrasensory.<uuid>`, `shl.User2`, `nor_tmd.<installationId>`). Groups expand to every subject with that
   session attribute, so new installations of a reserved model are reserved automatically.
2. **Pinning:** record the file's sha256 in the split document (as the ExtraSensory path does), instead of
   comparing against a constant.
3. **Checks:** every panel subject exists in the dataset (or is listed as unavailable); no subject of a reserved
   group appears in a dev fold; the panel per source is printed by `wma split check`.
4. **Dev folds:** drawn over all non-reserved subjects of all sources (the existing greedy balancer, generalised to
   balance class minutes per source; platform where a source has it), no `--dev-people` needed.
5. **Outputs:** `<id>_f0..f4` (dev folds) and `<id>_reserved` (train = dev folds, test = panel), as today.
6. Keep the ExtraSensory path unchanged for its existing comparisons.

### 6.3 Time zone per source (for `trip_group` and `calendar_date`)

WMA groups by `trip_group` (person + local date, ADR-029). Today the ExtraSensory adapter takes the **UTC** date of
the session start (`extrasensory.py:379`), and SHL the recording folder's date (`build_dataset.py:337`).

| Source | Timestamps | Time zone for the date | Note |
|---|---|---|---|
| ExtraSensory | UTC epoch | **America/Los_Angeles** (UC San Diego, 2015–16) | UTC splits Californian evenings (after 16:00–17:00 local) into the next day; use the local date |
| SHL | UTC epoch ms; folders named by local day | **Europe/London** (Brighton) | the folder date is already local |
| NOR-TMD | **no time zone** | **Europe/Oslo** (Oslo / Bodø, 2021–22) | used as given in TMD; check the hour-of-day histogram (commute peaks near 07–09 and 15–17 if local) before deciding whether they are local or UTC |
| Collecty | **no time zone**, 12-hour clock wraps | **Europe/Zagreb** (Croatia) | time of day is unreliable after wraps (TMD K1); a trajectory's date from its first clock piece |

Store the zone per source in the dataset config and the decision in an ADR.

## 7. Count check

After building, a count per source × class: labelled minutes (and windows, from WMA's window cache) against TMD's
counts below. **Explain any gap above ~5 %** (expected sources: WMA's window policy, unlabelled vs `mixed` at label
edges, TMD's one-row-per-minute grid). This is a sanity check, not a reproduction target.

TMD reference: **labelled minutes before flags and caps** (from `waiis_bench_v2` labels: ExtraSensory without
BEF6C611, SHL hips only, NOR-TMD Android only; in brackets, minutes that also have a usable 15 s window; users):

| Source | still | walk | run | bike | car | bus | train | subway | tram |
|---|---|---|---|---|---|---|---|---|---|
| ExtraSensory (59 users) | 263,649 (252,949; 59) | 22,206 (20,424; 59) | 1,090 (1,035; 26) | 5,020 (4,518; 25) | 11,039 (10,214; 49) | 1,794 (1,627; 31) | 344 (242; 5) | – | – |
| SHL User 1 (hips) | 3,425 | 3,101 | 1,061 | 2,900 | 4,087 | 3,069 | 3,381 | 2,421 | – |
| SHL Users 2–3 (hips) | 495 | 441 | 46 | 199 | 342 | 153 | 363 | 361 | – |
| NOR-TMD (Android) | – | – | – | 621 (593; 7) | 834 (819; 17) | 8,172 (7,264; 74) | 663 (621; 21) | 4,647 (4,026; 30) | 1,405 (1,322; 22) |
| Collecty | – | – | 201 (197; 2) | 585 (562; 2) | 4,161 (3,894; 13) | 1,785 (1,737; 5) | 2,797 (2,709; 5) | – | 452 (439; 4) |

Unlabelled usable time from excluded labels (should exist as `unknown` windows, with the source label as a stratum):
NOR-TMD outside 1,636 and inside 401 minutes, boat 3,715, e-scooter 198; Collecty Walk 3,529, e-scooter 388.

Also check: no reserved subject in any dev fold; every installation of the 8 device models reserved; the ExtraSensory
iPhone sessions converted per decision 1.

## 8. Training and comparing models

For each model (WMA registry: `inceptiontime_acc`, `hart_acc`, `lightgbm_acc_spectral`, `lightgbm_gps`,
`fusion_late`, …), one experiment YAML on **the same** `dataset_id`, `split_id` and `window_config`; the choices of
flags and caps for training (and the evaluation set) go in each experiment's `sampling` / selection, identical for
the models compared:

    uv run wma split check <split_id>
    uv run wma model train --config configs/experiments/<task>-golden_v1-<model>-v1.yaml
    uv run wma model compare <run_a> <run_b> --stage after

Suggested reporting (TMD's convention): six-class macro-F1 (still, foot, bike, car, bus, rail) pooled out of fold
over the dev folds, with person-bootstrap CIs; per source separately (no source hidden in a pooled score); the
reserved panel once per chosen model, overall and per source / platform; confusion matrix, per-class recall,
inference time per window, model size.

## 9. Open points

1. Licence for commercial use of models trained on SHL / ExtraSensory (section 2).
2. Axis convention ADR (decision 1), before any build.
3. String label mappings (NOR-TMD, Collecty) in `taxonomy.yaml`, or versioned code mappings.
4. The generic reserved-panel split (section 6.2).
5. Time zones for NOR-TMD and Collecty (section 6.3).
6. WMA's `AGENTS.md` says never add `Co-Authored-By` to commits there.

## Appendix A. For a later E1 port (not part of the build)

TMD's E1 (experiment 10), as saved in `<TMD data>/processed/10_gps-acc-baseline_waiis-bench-v2/models/`:
- **Acc model:** LightGBM, 9 classes, on 580 |acc| features (`tmd.features.acc.features(w, sub=True, grav=False)`)
  of a 15 s, 25 Hz window; `objective multiclass`, learning rate 0.1, 31 leaves, `min_child_samples` 50, feature and
  bagging fraction 0.8 (bagging every iteration), `lambda_l2` 1.0, up to 1,000 rounds, patience 30. **Balanced class
  weights** on top of source weights (every present class gets the same total weight). Trained on ExtraSensory +
  SHL User 1 (hips) + NOR-TMD official rows.
- **Early stopping:** paired folds; the rounds for outer fold k come from early stopping on fold (k + 1) mod 5
  (fit on the other 3), then a refit on all 4. The stopping loss is LightGBM's multiclass log loss with **tram rows
  removed from the stopping fold** (so tram doesn't decide the rounds). A six-class-merged stopping loss was proposed
  in 10 but not what E1 used. Final models: rounds = median of the seed's 5 out-of-fold counts, no early stopping.
- **GPS model:** LightGBM, 8 classes (no tram), on 13 gps-v2 features from one fix per minute (±150 s window; speeds
  derived from positions only; stop threshold 1.0 m/s), learning rate 0.02, up to 10,000 rounds, patience 150.
  **Trained on ExtraSensory + SHL User 1 only** (NOR-TMD and Collecty publish no GPS).
- **Fusion:** product p ∝ p_acc · p_gps (both floored at 1e-9, renormalised) on GPS-covered minutes (≥ 1 pair speed in
  the window); acc model alone elsewhere. The GPS model has no tram column: its **tram factor = the mean of its four
  vehicle probabilities** (car, bus, train, subway), so GPS says "a vehicle" and the acc model decides tram.
- **HMM:** forward-backward per session; 1-minute steps (60 s, timestamps matched within ±30.5 s); missing minutes
  bridged with Aᵏ; the chain restarts after a gap > 10 min or a new session. Transitions per 1-minute step from the
  labelled minutes of ExtraSensory + SHL User 1 (+1 pseudo-count), 9 states (tram's row and column are the
  pseudo-counts only); initial distribution = class frequencies + 1.
- **Seeds:** 0, 1, 2; seed 0 is the model, seeds 1–2 the spread.
- **Output in WMA's vocabulary:** the 9 probabilities in WMA's class order; **motorcycle = 0** (and `unknown`,
  `mixed` = 0); no class is reordered.
