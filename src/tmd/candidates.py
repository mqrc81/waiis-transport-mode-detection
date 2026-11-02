"""Clean-window rules for event-stream datasets (experiment 08; waiis_bench_v2's window definition, experiment 09).

One definition for every dataset, following waiis_bench_v1 where it has a rule (04 decision register):
- per clock minute one 15 s window centred in the minute (as for SHL in 04's build.py);
- **single label** over the window (every event in it has the same label, and the trip covers the whole window);
- **real accelerometer rate >= MIN_RATE_HZ** over the window: distinct event times in it / 15 s. 24 Hz since
  experiment 09 (tolerance for nominal-25 Hz data with jitter); 08's tables were built at 25 Hz;
- **no gap > 0.5 s**, including window start -> first event and last event -> window end;
- resampled as in v1 (04, SHL): every gap-free stretch (no gap > 0.5 s) is interpolated onto a 100 Hz grid from its
  first event and decimated once (`decimate_100_to_25`); a window is the 375 samples from the first 25 Hz sample at
  or after its start (`stretches25`). 08's tables used a window +- 1 s grid instead (08 deviation 1);
- only windows with at least one accelerometer event are rows (empty minutes and tiles are not listed);
- flags as in 04: `acc_implausible` (window median |acc| outside 7-20 m/s2, U3), `quiet` (std |acc| < 0.03 m/s2),
  `label_implausible` (quiet walk / run / bike, X3). `clean` = all rules met and neither flag set.
Gyroscope "by the same rule": rate >= MIN_RATE_HZ and no gap > 0.5 s over the same window. Magnetometer and
barometer: at least one event in the window.

Clean hours (beside the minute grid): every trip is tiled into back-to-back 15 s windows from its first event;
clean hours = clean tiles x 15 s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tmd.resample import DOWN, FS_IN, decimate_100_to_25

WIN_S = 15.0
FS = 25
WIN = int(WIN_S * FS)          # 375
MIN_RATE_HZ = 24.0
RATE_STATUS = f"rate_below_{MIN_RATE_HZ:g}hz"
MAX_GAP_S = 0.5
ACC_PLAUSIBLE = (7.0, 20.0)    # m/s2, window median |acc| (04: U3)
QUIET_STD = 0.03               # m/s2 (04)
LABEL_QUIET_CLASSES = ["walk", "run", "bike"]   # 04: X3


def coverage(t: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """Rate [Hz] of distinct event times in [lo, hi) and the largest gap [s] inside it, edges included.
    t: sorted event times [s]."""
    i, j = np.searchsorted(t, [lo, hi])
    tt = np.unique(t[i:j])
    if len(tt) == 0:
        return 0.0, hi - lo
    gaps = np.diff(np.concatenate([[lo], tt, [hi]]))
    return len(tt) / (hi - lo), float(gaps.max())


def stretches25(t: np.ndarray, xyz: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The gap-free stretches (no gap > MAX_GAP_S, at least one window long) of sorted, distinct event times t [s],
    each interpolated onto a 100 Hz grid from its first event and decimated to 25 Hz: [(t25 [s], x25 (n, 3))]."""
    cut = np.concatenate([[0], np.flatnonzero(np.diff(t) > MAX_GAP_S) + 1, [len(t)]])
    out = []
    for lo, hi in zip(cut[:-1], cut[1:]):
        if t[hi - 1] - t[lo] < WIN_S:
            continue
        grid = t[lo] + np.arange(int((t[hi - 1] - t[lo]) * FS_IN) + 1) / FS_IN
        x = np.column_stack([np.interp(grid, t[lo:hi], xyz[lo:hi, k]) for k in range(3)])
        out.append((grid[::DOWN], decimate_100_to_25(x)))
    return out


def window_from(stretches: list, lo: float) -> tuple[np.ndarray | None, float]:
    """The (375, 3) window from the first 25 Hz sample at or after lo, inside one stretch that starts at most
    MAX_GAP_S after lo (the gap rule's edge tolerance); (None, nan) if none."""
    for t25, x25 in stretches:
        if t25[0] <= lo + MAX_GAP_S and t25[-1] >= lo:
            s = int(np.searchsorted(t25, lo - 1e-9))
            if s + WIN <= len(x25):
                return x25[s:s + WIN], float(t25[s])
    return None, np.nan


def distinct(t: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Sort by time, keep the first of repeated times. Returns t, xyz, n repeated, n out of order."""
    back = int((np.diff(t) < 0).sum())
    o = np.argsort(t, kind="stable")
    t, xyz = t[o], xyz[o]
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], xyz[keep], int((~keep).sum()), back


def trip_windows(t_acc: np.ndarray, acc: np.ndarray, label_t: np.ndarray, label: np.ndarray,
                 t_gyro: np.ndarray | None = None, t_mag: np.ndarray | None = None,
                 t_press: np.ndarray | None = None, grid: str = "minute") -> tuple[pd.DataFrame, np.ndarray]:
    """Windows of one trip. t_* [s] (any order), acc (n, 3) total acceleration m/s2, label per label_t event
    (label_t sorted; '' = no class). grid 'minute': one window centred in each clock minute the trip touches;
    'tile': back-to-back windows from the first event.

    Returns (rows: one per window with ts / t_start, label, rate_hz, max_gap_s, gyro_*, has_mag, has_press,
    acc_median, acc_std, status, flags; windows (n_ok, 375, 3) for the rows with status 'ok', in row order)."""
    t, xyz, _, _ = distinct(np.asarray(t_acc, float), np.asarray(acc, float))
    tg = np.unique(t_gyro) if t_gyro is not None and len(t_gyro) else None
    tm = np.unique(t_mag) if t_mag is not None and len(t_mag) else None
    tp = np.unique(t_press) if t_press is not None and len(t_press) else None
    if len(t) == 0:
        return pd.DataFrame(), np.zeros((0, WIN, 3), np.float32)
    t0, t1 = t[0], t[-1]
    if grid == "minute":
        starts = np.arange(np.floor(t0 / 60), np.floor(t1 / 60) + 1) * 60 + 30 - WIN_S / 2
    else:
        starts = t0 + np.arange(int((t1 - t0) // WIN_S)) * WIN_S
    # only windows that contain at least one event (a broken clock can make a trip span years)
    starts = starts[np.searchsorted(t, starts + WIN_S) > np.searchsorted(t, starts)]
    st25 = stretches25(t, xyz)
    rows, wins = [], []
    for lo in starts:
        hi = lo + WIN_S
        r = {"ts": int(np.floor((lo + WIN_S / 2) / 60) * 60), "t_start": float(lo)}
        i, j = np.searchsorted(label_t, [lo, hi])
        labs = np.unique(label[i:j])
        r["label"] = labs[0] if len(labs) == 1 else ("" if len(labs) == 0 else "|".join(map(str, labs)))
        r["rate_hz"], r["max_gap_s"] = coverage(t, lo, hi)
        if tg is not None:
            r["gyro_rate_hz"], r["gyro_max_gap_s"] = coverage(tg, lo, hi)
        else:
            r["gyro_rate_hz"], r["gyro_max_gap_s"] = 0.0, WIN_S
        r["has_gyro"] = r["gyro_rate_hz"] >= MIN_RATE_HZ and r["gyro_max_gap_s"] <= MAX_GAP_S
        r["has_mag"] = tm is not None and np.searchsorted(tm, hi) > np.searchsorted(tm, lo)
        r["has_press"] = tp is not None and np.searchsorted(tp, hi) > np.searchsorted(tp, lo)
        if lo < t0 or hi > t1:
            r["status"] = "edge"
        elif len(labs) != 1:
            r["status"] = "mixed_label" if len(labs) > 1 else "no_events"
        elif r["rate_hz"] < MIN_RATE_HZ:
            r["status"] = RATE_STATUS
        elif r["max_gap_s"] > MAX_GAP_S:
            r["status"] = "gap_over_0.5s"
        elif (w := window_from(st25, lo))[0] is None:
            r["status"] = "stretch_too_short"         # gap-free, but the 25 Hz grid ends before 375 samples
        else:
            w, r["t_start_25hz"] = w
            r["status"] = "ok"
            n = np.linalg.norm(w, axis=1)
            r["acc_median"], r["acc_std"] = float(np.median(n)), float(n.std())
            m = w.mean(0)
            k = int(np.abs(m).argmax())
            r["dom_axis"] = ("-" if m[k] < 0 else "+") + "xyz"[k]      # gravity direction in device axes
            wins.append(w)
        rows.append(r)
    df = pd.DataFrame(rows)
    if "acc_median" not in df:
        df["acc_median"] = df["acc_std"] = np.nan
    return df, (np.stack(wins).astype(np.float32) if wins else np.zeros((0, WIN, 3), np.float32))


def flag(df: pd.DataFrame, cls_col: str = "cls") -> pd.DataFrame:
    """04's flags and `clean`, in place; cls_col holds the v1 class ('' = none)."""
    ok = df.status.eq("ok")
    df["acc_implausible"] = ok & ~df.acc_median.between(*ACC_PLAUSIBLE)
    df["quiet"] = ok & (df.acc_std < QUIET_STD)
    df["label_implausible"] = df.quiet & df[cls_col].isin(LABEL_QUIET_CLASSES)
    df["clean"] = ok & ~df.acc_implausible & ~df.label_implausible
    return df


def timing(t_ms: np.ndarray) -> dict:
    """Timing of one sensor in one trip (check 3), from event times [ms] in file order.

    rate = distinct times / span; intervals over the sorted distinct times; `regular` = share of intervals within
    10% of the median (high = fixed-rate, low = event-driven / jittery); `rate_first5`, `rate_after5` = rate in the
    first 5 min and after (throttling shows as a lower later rate)."""
    t = np.asarray(t_ms, np.int64)
    if len(t) < 2:
        return {"events": len(t)}
    back = int((np.diff(t) < 0).sum())
    s = np.sort(t)
    rep = int((np.diff(s) == 0).sum())
    u = np.unique(s)
    dt = np.diff(u).astype(float)
    span = (u[-1] - u[0]) / 1e3
    med = float(np.median(dt)) if len(dt) else np.nan
    cut = u[0] + 300_000
    early, late = u[u < cut], u[u >= cut]
    return {"events": len(t), "span_s": span, "rate_hz": (len(u) - 1) / span if span > 0 else np.nan,
            "dt_median_ms": med, "dt_p5_ms": float(np.percentile(dt, 5)) if len(dt) else np.nan,
            "dt_p95_ms": float(np.percentile(dt, 95)) if len(dt) else np.nan,
            "repeated": rep, "backwards": back, "gaps_over_0.5s": int((dt > 500).sum()),
            "max_gap_s": float(dt.max() / 1e3) if len(dt) else np.nan,
            "regular": float(np.mean(np.abs(dt - med) <= 0.1 * med)) if len(dt) else np.nan,
            "rate_first5": (len(early) - 1) / ((early[-1] - early[0]) / 1e3) if len(early) > 1 and early[-1] > early[0] else np.nan,
            "rate_after5": (len(late) - 1) / ((late[-1] - late[0]) / 1e3) if len(late) > 1 and late[-1] > late[0] else np.nan}


def legs(t_ms: np.ndarray, label: np.ndarray, ordered: bool = False) -> pd.DataFrame:
    """Runs of one label in a trip's accelerometer events: label, start / end [ms], duration [s]. Events are sorted
    by time unless `ordered` (file order is the truth, e.g. Collecty's broken clocks)."""
    o = np.arange(len(t_ms)) if ordered else np.argsort(t_ms, kind="stable")
    t, lab = np.asarray(t_ms)[o], np.asarray(label)[o]
    if len(t) == 0:
        return pd.DataFrame(columns=["label", "t_start_ms", "t_end_ms", "dur_s"])
    cut = np.concatenate([[0], np.flatnonzero(lab[1:] != lab[:-1]) + 1, [len(t)]])
    return pd.DataFrame({"label": lab[cut[:-1]], "t_start_ms": t[cut[:-1]], "t_end_ms": t[cut[1:] - 1],
                         "dur_s": (t[cut[1:] - 1] - t[cut[:-1]]) / 1e3})


def clipping(acc: np.ndarray) -> dict:
    """Max |acc| and, per axis, max |value| and the share of events within 0.1% of it (a plateau = clipping)."""
    a = np.abs(np.asarray(acc, float))
    out = {"acc_norm_max": float(np.linalg.norm(acc, axis=1).max()) if len(a) else np.nan}
    for k, c in enumerate("xyz"):
        mx = a[:, k].max() if len(a) else np.nan
        out[f"{c}_absmax"] = float(mx)
        out[f"{c}_at_max"] = int((a[:, k] >= 0.999 * mx).sum()) if len(a) else 0
    return out


def qc_trip(t_acc_ms, acc, label, t_gyro_ms=None, t_mag_ms=None, t_press_ms=None) -> dict:
    """Everything the checks need from one trip. acc: (n, 3) total acceleration m/s2; label: raw label per
    accelerometer event. Returns minutes (rows), wins (n_ok, 375, 3), tiles (rows), timing (one row per sensor),
    legs, clipping."""
    t_acc_ms = np.asarray(t_acc_ms, np.int64)
    o = np.argsort(t_acc_ms, kind="stable")
    ta, xyz, lab = t_acc_ms[o] / 1e3, np.asarray(acc, float)[o], np.asarray(label).astype(str)[o]
    sec = lambda t: None if t is None else np.asarray(t, np.int64) / 1e3
    args = dict(t_gyro=sec(t_gyro_ms), t_mag=sec(t_mag_ms), t_press=sec(t_press_ms))
    minutes, wins = trip_windows(ta, xyz, ta, lab, grid="minute", **args)
    tiles, _ = trip_windows(ta, xyz, ta, lab, grid="tile", **args)
    tim = [{"sensor": "acc", **timing(t_acc_ms)}]
    for name, t in (("gyro", t_gyro_ms), ("mag", t_mag_ms), ("press", t_press_ms)):
        if t is not None and len(t):
            tim.append({"sensor": name, **timing(np.asarray(t, np.int64))})
    return {"minutes": minutes, "wins": wins, "tiles": tiles, "timing": pd.DataFrame(tim),
            "legs": legs(t_acc_ms, label), "clipping": clipping(acc)}
