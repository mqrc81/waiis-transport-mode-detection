"""GPS tracks in the benchmark format (experiment 04): sessions, distances, bearings and cadence thinning.

Track columns: time [s], lat, lon, speed [m/s], accuracy [m], bearing [deg, 0 = north, clockwise]. Unknown speed,
accuracy or bearing is -1, as the phone reports it; lat / lon are never imputed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

R_EARTH = 6_371_008.8          # m, mean radius
SESSION_GAP_S = 120            # a gap > 2 min between minutes starts a new session
# WAIIS production cadence (Marc, 2026-10-28): a fix every ~10 s while moving, ~20 s walking, minutes apart still.
CADENCE = ((3.0, 10.0), (0.5, 20.0), (0.0, 180.0))   # (speed >= m/s, interval s), first match wins
SPEED_MISSING_SHARE = 0.23     # production: ~23% of fixes have no speed


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial bearing from point 1 to point 2, [0, 360)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360


def session_ids(t: np.ndarray, gap_s: float = SESSION_GAP_S) -> np.ndarray:
    """0, 1, ... per sorted time array; a new session wherever consecutive times are more than gap_s apart."""
    return np.concatenate([[0], np.cumsum(np.diff(t) > gap_s)]).astype(np.int32)


def add_consecutive(track: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Distance [m], time [s] and speed [m/s] from the previous fix of the same group (e.g. user + session);
    NaN for the first fix of a group. `track` must be sorted by `by` + time."""
    prev = track.groupby(by, sort=False)[["time", "lat", "lon"]].shift(1)
    d = haversine_m(prev.lat.to_numpy(), prev.lon.to_numpy(), track.lat.to_numpy(), track.lon.to_numpy())
    dt = track.time.to_numpy() - prev.time.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(dt > 0, d / dt, np.nan)
    return track.assign(dist_prev_m=d, dt_prev_s=dt, speed_prev_mps=v)


def bearing_from_previous(track: pd.DataFrame, by: list[str], min_dist_m: float = 1.0) -> np.ndarray:
    """Bearing from the previous fix of the group to this one; -1 for the first fix or a move under min_dist_m."""
    prev = track.groupby(by, sort=False)[["lat", "lon"]].shift(1)
    b = bearing_deg(prev.lat.to_numpy(), prev.lon.to_numpy(), track.lat.to_numpy(), track.lon.to_numpy())
    d = haversine_m(prev.lat.to_numpy(), prev.lon.to_numpy(), track.lat.to_numpy(), track.lon.to_numpy())
    return np.where(np.isnan(prev.lat.to_numpy()) | ~(d >= min_dist_m), -1.0, b)


def local_speed_bearing(t: np.ndarray, lat: np.ndarray, lon: np.ndarray, half_s: float,
                        max_dt_s: float = 5.0, min_dist_m: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Speed [m/s] and bearing at every fix of a dense (~1 Hz) track, from the first and last fix within
    [t - half_s, t + half_s]. Fixes across a gap > max_dt_s are not used: the span is cut at the gap. -1 where only
    one fix is available (speed and bearing) or the move is under min_dist_m (bearing)."""
    seg = session_ids(t, max_dt_s)
    lo = np.searchsorted(t, t - half_s, side="left")
    hi = np.searchsorted(t, t + half_s, side="right") - 1
    idx = np.arange(len(t))
    # stay inside the fix's own gap-free segment
    seg_start = np.searchsorted(seg, seg, side="left")
    seg_end = np.searchsorted(seg, seg, side="right") - 1
    lo, hi = np.maximum(lo, seg_start), np.minimum(hi, seg_end)
    d = haversine_m(lat[lo], lon[lo], lat[hi], lon[hi])
    dt = t[hi] - t[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(dt > 0, d / dt, -1.0)
    b = np.where((dt > 0) & (d >= min_dist_m), bearing_deg(lat[lo], lon[lo], lat[hi], lon[hi]), -1.0)
    assert (lo <= idx).all() and (idx <= hi).all()
    return v, b


def cadence_interval(speed: np.ndarray, cadence=CADENCE) -> np.ndarray:
    out = np.full(len(speed), cadence[-1][1])
    for v_min, dt in reversed(cadence):
        out[speed >= v_min] = dt
    return out


def thin(t: np.ndarray, interval: np.ndarray) -> np.ndarray:
    """Indices of kept fixes: the first fix, then each fix at least interval[i] seconds after the last kept one
    (interval taken at the candidate fix). Natural gaps stay gaps: nothing is added where there is no fix."""
    keep, last = [], -np.inf
    for i in range(len(t)):
        if t[i] - last >= interval[i]:
            keep.append(i)
            last = t[i]
    return np.array(keep, dtype=np.int64)
