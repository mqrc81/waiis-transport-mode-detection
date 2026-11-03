# WAIIS app data: two own days, descriptive (2026-11-03)

A descriptive look at two days of the project's own WAIIS app recordings (2026-10-30 and 2026-10-31, one person,
unlabelled). Not part of 08's Plan: no models, no MLflow runs. Kept as a reference for a later benchmark or baseline
that adds GPS-only candidate datasets: it shows what WAIIS GPS and accelerometer data look like in production.

- **Data:** `<data>/waiis/m-waiis-days-2026-10-30_31/` (outside git; precise locations, keep private). Per UTC day: `gps.parquet`,
  `imu.parquet`, stay / trip segments, OpenStreetMap context and model predictions. Only GPS and accelerometer were
  used. The package's own README describes the files.
- **Method:** states from derived speed over a 60 s baseline: displacement to the first fix ≥ 60 s ahead (else
  behind), at most 180 s away. Still < 0.5 m/s, walk 0.5–2.5, vehicle > 4. Accelerometer still blocks: 5 s inside
  one burst with std |acc| < 0.05 m/s².
- **Scripts** (here; `uv run python <script>`, data path as optional argument): `describe.py` (all GPS and
  accelerometer tables), `gaps.py` (gaps by speed across, coarse fixes), `imu_timing.py` (step and drift detail,
  GPS–accelerometer overlap). They print summaries only, never coordinates.

## GPS (388 + 364 fixes)

**Format:**
- **Columns:** `t_ms`, `lat`, `lon`, `accuracy_m`, `speed_mps`, `speed_accuracy_mps`, `heading_deg`, `altitude_m`,
  `source`, `activity`, `activity_confidence`, `is_moving`, `event`.
- **Always empty:** `speed_accuracy_mps`, `activity_confidence` and `event`.
- **Units:** metres, m/s, degrees.
- **Time:** UTC epoch milliseconds, always whole seconds, sorted, no duplicates. Local time is Europe/Madrid, UTC+1
  on these dates (summer time ended 25 Oct). Folders are UTC days.
- **Platform:** not recorded (only `source = "waiis"`). The phone was probably an iPhone 16 (reported afterwards,
  not certain).
- **The phone's own activity flag is useless:** 93% of fixes say `in_vehicle` with `is_moving = true`, although the
  median speed is 0.8 m/s.

**States:** still 329 fixes, walk 284, vehicle 86, between walk and vehicle (2.5–4 m/s) 5, no baseline possible 48.

**Interval between consecutive fixes [s]:**
- The app records on a fixed 20 s cadence: 79% of intervals are 20–30 s, and the median is 20 s overall and in
  every state.

| State | p50 | p75 | p95 | Share ≤ 20 s |
|---|---|---|---|---|
| still | 20 | 23 | 65 | 0.66 |
| walk | 20 | 20 | 30 | 0.87 |
| vehicle | 20 | 39 | 185 | 0.53 |

- Vehicle has the longest tail: GPS drops out on some vehicle legs.

**Gaps over 60 s: 75.** The speed before and after a gap is mostly unknown, because the fixes next to a gap are
isolated. The speed across the gap, from the displacement between its two end fixes, separates them better:

| Speed across the gap | Gaps | Median length | Total |
|---|---|---|---|
| < 0.5 m/s, stationary | 37 | 18 min | 25.2 h |
| 0.5–2 m/s | 6 | 4 min | 0.5 h |
| ≥ 2 m/s, moving | 32 | 5 min | 2.4 h |

- **Stationary gaps are real still periods.** Overnight and long stays show as isolated fixes every 10–20 min,
  about 6–19 m accuracy, with no speed reported.
- **Moving gaps are vehicle legs without good GPS.** The speed across is 4–12 m/s, and 22 of the 32 end at a coarse
  fix (accuracy > 500 m) about every 4–5 min.
  - They cluster in five legs, 15–30 min each.
  - The pattern fits underground rail, but that's an inference: there are no labels.
- **Only 4 gaps have a vehicle state on one side and still on the other**, so short vehicle stops (traffic lights,
  stations) don't show up as gaps. They stay on the 20 s cadence.

**Reported speed:**
- Missing for 29% of fixes: still 31%, walk 12%, vehicle 40%, and every coarse fix.
- Where both exist, it agrees well with derived speed. Against the speed between consecutive fixes (≤ 30 s apart):
  correlation 0.97, median difference −0.05 m/s.
- Medians, reported vs 60 s derived: still 0.2 vs 0.1, walk 1.0 vs 1.0, vehicle 19.2 vs 18.8 m/s. The maximum
  reported is 26 m/s.

**Accuracy:**

| State | p50 | p95 | Share > 50 m |
|---|---|---|---|
| still | 14.2 m | 47 m | 5% |
| walk | 14.2 m | 40 m | 3% |
| vehicle | 13.3 m | 1,314 m | 31% |
| all | 14.2 m | 120 m | 9.3% |

- In a vehicle, accuracy is either very good or very poor: p25 is 3.9 m, p75 87 m.
- **33% of fixes report exactly 14.25 m.** That looks like a default value, not a real estimate.
- The coarse values are rounded: 1,414 m appears 7 times, and 2,000 m twice (the largest value, probably a cap).

## Accelerometer (2.3 h and 2.0 h)

- **Unit:** m/s², phone frame.
- **Gravity is included:** median |acc| over still 5 s blocks is 9.73 m/s² (p5–p95 9.69–9.84). That's about 0.8%
  below 9.81, a small scale offset. The median over all samples is 9.78.
- **Rate:** the real rate is 24.9 Hz in every burst, which passes v2's ≥ 24 Hz rule.
  - Timestamps are integer ms, and 99.7–99.8% of steps are 40 or 41 ms.
  - Because the true step is slightly over 40 ms, samples drift from a 40 ms grid over a burst: median 0.7–0.8 s,
    p99 2.7–4 s.
  - A few hundred steps per day fall outside that: under 39 ms (156 / 108) or 42–200 ms (198 / 133). The largest
    step inside a burst is 0.9 s.
- **Bursts:** 30 and 26. The package README says 29 and 25 because the first burst of each day has no `gap` flag.
  - Burst length: median 4–5 min, capped near 300 s, a few around 600 s.
  - The pause between bursts is usually 2–5 s, so recording is nearly continuous while moving.
  - Recording ran 16:09–22:57 UTC on the 30th and 12:14–20:28 on the 31st.
- **GPS fixes with accelerometer data within 30 s:** walk 96%, still 98% (still periods inside trips), vehicle 98%
  on the 30th but only 57% on the 31st.
- **Saturation (±65.5):** 6 samples per day.
- **Axis convention:** not settled by the data alone; if the phone was the iPhone 16, it looks like **iOS's
  convention scaled to m/s², not converted to Android's**. Only a few still blocks exist, since the app records only
  while moving.

  | Day | Lying flat and still | Other still blocks |
  |---|---|---|
  | 30th | 6 blocks, z = +9.77 | 21 blocks, y ≈ −9.8 |
  | 31st | 26 blocks, z = −9.78 | 55 blocks y ≈ −9.8; 7 blocks x positive |

  - Face up, Android gives z = +9.8 and raw iOS gives z negative. Face down reverses both. Without knowing which way
    the phone faced, the sign alone can't tell the two apart.
  - Upright (top up), Android gives y = +9.8 and raw iOS y negative. y ≈ −9.8 is the most common still orientation
    (76 of 116 blocks), and z is negative in most flat blocks (26 of 32). Both fit raw iOS signs with an upright or
    face-up phone; on Android they would need the phone mostly upside down and face down.
  - So with an iPhone, the app multiplies by +9.81 (median |acc| still 9.73 m/s²) and keeps iOS's signs. v2's rule
    A1 (iOS × −9.80665, to the Android convention) would then apply to WAIIS iPhone data as `× −1`.
  - To confirm: a 30 s recording with the phone face-up on a table (expected z ≈ −9.8 if iOS's signs are kept),
    and the phone model stored with the data.

## For a GPS-only benchmark or baseline

- **Cadence:** production GPS is one fix per ~20 s while recording, denser than the benchmarks' one fix per minute.
  Candidate GPS-only datasets with 1 Hz trajectories should be subsampled to ~20 s (and to 60 s for comparison).
- **Gaps:** about half of long gaps are moving. On some vehicle legs, likely underground rail, there is only a coarse
  fix (> 500 m) every ~5 min with no reported speed. A GPS-only model gets almost nothing there, so a benchmark
  should mimic these dropouts, not only clean trajectories.
- **Fields:** don't rely on `speed_accuracy_mps`, `activity` or `is_moving`. Reported speed is missing for 29% of
  fixes (40% in vehicles); derived speed matches it where both exist. An accuracy of exactly 14.25 m is not a real
  estimate.
- **Accelerometer:** 24.9 Hz in ~5 min bursts with gravity fits the 15 s windows. The iOS-to-Android sign rule needs
  the platform, which the data don't record; on the probable iPhone 16 the signs look like iOS's (see above).
