"""Resampling to the production rate: anti-aliasing low-pass, then decimation (brief: Common rules).

Both datasets go through a uniform 100 Hz grid, then `scipy.signal.resample_poly(x, 1, 4)` (Kaiser-windowed FIR
low-pass at the new Nyquist + decimation). SHL is already on a 100 Hz grid. ExtraSensory snippets (≈34-100 Hz,
irregular) are linearly interpolated onto one first, within the snippet only.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

FS_IN = 100
FS = 25                       # production rate
DOWN = FS_IN // FS
G = 9.80665
MAX_SPAN_S = 40.0             # ExtraSensory: longer snippets have broken clocks (02 audit)
MAX_DT_S = 0.5                # ExtraSensory: a bigger hole inside a snippet drops it


def decimate_100_to_25(x: np.ndarray) -> np.ndarray:
    """(n, c) at 100 Hz -> (ceil(n/4), c) at 25 Hz; output k is aligned with input 4k."""
    return resample_poly(x, 1, DOWN, axis=0, padtype="line").astype(np.float32)


def es_snippet_problem(a: np.ndarray) -> str:
    """'' if the (n, 4) [t s, x, y, z] snippet is usable, else the first reason it is not (approved 2026-10-25)."""
    dt = np.diff(a[:, 0])
    if (dt <= 0).any():
        return "time_not_increasing"
    if a[-1, 0] - a[0, 0] > MAX_SPAN_S:
        return "span_over_40s"
    if dt.max() > MAX_DT_S:
        return "hole_over_0.5s"
    if np.isnan(a[:, 1:]).any():
        return "nan"
    return ""


def to_ms2(acc_xyz: np.ndarray) -> tuple[np.ndarray, str]:
    """ExtraSensory acc units are decided per snippet: median |acc| < 3 means g (02 audit). Returns m/s2 and the unit."""
    if np.median(np.linalg.norm(acc_xyz, axis=1)) < 3:
        return acc_xyz * G, "g"
    return acc_xyz, "m/s2"


def es_snippet_to_25hz(acc: np.ndarray, gyro: np.ndarray) -> tuple[np.ndarray | None, str]:
    """acc, gyro: (n, 4) [t s, x, y, z] as read by `extrasensory.read_snippet`.

    Returns ((m, 6) float32 [acc m/s2, gyro rad/s] at 25 Hz, 'ok') or (None, reason). Snippets with time going
    backwards or standing still, a span over 40 s, or a hole over 0.5 s are dropped (approved 2026-10-25).
    Acc units are decided per snippet: median |acc| < 3 means g.
    """
    for a in (acc, gyro):
        bad = es_snippet_problem(a)
        if bad:
            return None, bad
    t0, t1 = max(acc[0, 0], gyro[0, 0]), min(acc[-1, 0], gyro[-1, 0])
    if t1 - t0 < 5.0:
        return None, "overlap_under_5s"
    a = to_ms2(acc[:, 1:])[0]
    grid = t0 + np.arange(int((t1 - t0) * FS_IN) + 1) / FS_IN
    x = np.column_stack([np.interp(grid, acc[:, 0], a[:, i]) for i in range(3)]
                        + [np.interp(grid, gyro[:, 0], gyro[:, 1 + i]) for i in range(3)])
    return decimate_100_to_25(x), "ok"


def es_acc_to_25hz(acc: np.ndarray) -> tuple[np.ndarray | None, str, str]:
    """Accelerometer only (experiment 04): (n, 4) [t s, x, y, z] -> ((m, 3) float32 m/s2 at 25 Hz, unit, 'ok') or
    (None, unit, reason). Same cleaning and resampling as `es_snippet_to_25hz`, without the gyroscope."""
    bad = es_snippet_problem(acc)
    if bad:
        return None, "", bad
    a, unit = to_ms2(acc[:, 1:])
    t0, t1 = acc[0, 0], acc[-1, 0]
    grid = t0 + np.arange(int((t1 - t0) * FS_IN) + 1) / FS_IN
    x = np.column_stack([np.interp(grid, acc[:, 0], a[:, i]) for i in range(3)])
    return decimate_100_to_25(x), unit, "ok"


def centre(x: np.ndarray, n: int) -> tuple[np.ndarray | None, int]:
    """The n rows at the centre of x and their start index, or (None, -1) if x is shorter than n."""
    if len(x) < n:
        return None, -1
    s = (len(x) - n) // 2
    return x[s:s + n], s


def es_acc_repair_to_25hz(acc: np.ndarray) -> tuple[np.ndarray | None, str, dict]:
    """Experiment 04 (waiis_bench_v1): like `es_acc_to_25hz`, but repairs instead of dropping where possible.

    - Time not increasing: sort by time, drop repeated timestamps (first kept), then re-run the checks;
      flag `time_repaired`.
    - Holes > 0.5 s: keep the longest hole-free stretch (by duration); flag `hole_trimmed`.
    - Span > 40 s or NaN: dropped (None).
    Units are decided on the whole snippet (`to_ms2`). Returns ((m, 3) float32 m/s2 at 25 Hz or None, unit,
    {'status': 'ok' | 'span_over_40s' | 'nan' | 'too_few_samples', 'time_repaired', 'hole_trimmed', 'n_raw',
    'span_s' (of the resampled part), 'raw_rate_hz'}). For a snippet that `es_acc_to_25hz` accepts, the output is
    identical."""
    info = {"status": "ok", "time_repaired": False, "hole_trimmed": False}
    a = acc
    if (np.diff(a[:, 0]) <= 0).any():
        a = a[np.argsort(a[:, 0], kind="stable")]
        a = a[np.concatenate([[True], np.diff(a[:, 0]) > 0])]
        info["time_repaired"] = True
    if a[-1, 0] - a[0, 0] > MAX_SPAN_S:
        return None, "", {**info, "status": "span_over_40s"}
    if np.isnan(a[:, 1:]).any():
        return None, "", {**info, "status": "nan"}
    xyz, unit = to_ms2(a[:, 1:])
    t = a[:, 0]
    dt = np.diff(t)
    if (dt > MAX_DT_S).any():
        cut = np.concatenate([[0], np.flatnonzero(dt > MAX_DT_S) + 1, [len(t)]])
        lo, hi = max(zip(cut[:-1], cut[1:]), key=lambda s: t[s[1] - 1] - t[s[0]])
        t, xyz = t[lo:hi], xyz[lo:hi]
        info["hole_trimmed"] = True
    if len(t) < 2:
        return None, unit, {**info, "status": "too_few_samples"}
    span = t[-1] - t[0]
    info.update(n_raw=len(t), span_s=float(span), raw_rate_hz=(len(t) - 1) / span)
    grid = t[0] + np.arange(int(span * FS_IN) + 1) / FS_IN
    x = np.column_stack([np.interp(grid, t, xyz[:, i]) for i in range(3)])
    return decimate_100_to_25(x), unit, info


def window_stats(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and standard deviation of |acc| per window; w is (n, m, 3)."""
    n = np.linalg.norm(w, axis=-1)
    return np.median(n, axis=-1), n.std(axis=-1)
