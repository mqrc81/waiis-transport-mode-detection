"""GPS motion features per minute from one fix per minute (experiment 06, version FEATURES_VERSION).

Inputs: a `gps_per_minute` table (tmd.bench). For a minute t, the window holds the fixes of the same (user, session)
with ts in [t - 150, t + 150] s (minutes t-2 .. t+2). Only positions and speeds derived from them are used:
    pair speed  v_i = dist_prev_m / dt_prev_s for a fix whose previous fix (same session) is <= 150 s earlier
    displacement bearing from the previous fix, when the displacement is >= 20 m; turns between consecutive valid
    bearings (skipping stops); path = displacements with both fixes in the window
No reported speed, accuracy, fix count, interval or missingness column. Missing values are NaN.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tmd.gps import bearing_deg, haversine_m

FEATURES_VERSION = "gps-v1"
HALF_WINDOW_S = 150
MAX_PAIR_DT_S = 150
STOP_MPS, MOVING_MPS = 0.5, 2.0
MIN_BEARING_M = 20.0
NAMES = ["v_p10", "v_p50", "v_p90", "v_max", "v_mean", "v_now", "v_next",
         "acc_mean", "acc_max", "stop_frac", "n_stops", "turn_mean", "turn_max",
         "path_m", "net_m", "straightness"]


def prepare(gps: pd.DataFrame) -> pd.DataFrame:
    """Sorted fixes with pair speed and displacement bearing (from the previous fix of the same session)."""
    g = gps.sort_values(["user", "session", "time"]).reset_index(drop=True)
    same = (g.user.eq(g.user.shift()) & g.session.eq(g.session.shift())).to_numpy()
    plat, plon = g.lat.shift().to_numpy(), g.lon.shift().to_numpy()
    dt = g.time.diff().to_numpy()
    pair = same & (dt > 0) & (dt <= MAX_PAIR_DT_S)
    dist = np.where(same, haversine_m(plat, plon, g.lat.to_numpy(), g.lon.to_numpy()), np.nan)
    g["pair_v"] = np.where(pair, dist / np.where(pair, dt, 1.0), np.nan)
    g["pair_dist"] = np.where(pair, dist, np.nan)
    g["brg"] = np.where(pair & (dist >= MIN_BEARING_M), bearing_deg(plat, plon, g.lat.to_numpy(), g.lon.to_numpy()),
                        np.nan)
    return g[["user", "session", "ts", "time", "lat", "lon", "pair_v", "pair_dist", "brg"]]


def _turn(b: np.ndarray) -> np.ndarray:
    d = np.abs(np.diff(b)) % 360
    return np.minimum(d, 360 - d)


def window_features(ts: np.ndarray, f_ts: np.ndarray, f_time: np.ndarray, v: np.ndarray, dist: np.ndarray,
                    brg: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Features of one minute t from the fixes of its session (arrays sorted by time). Returns (16,)."""
    lo, hi = np.searchsorted(f_ts, ts - HALF_WINDOW_S, "left"), np.searchsorted(f_ts, ts + HALF_WINDOW_S, "right")
    out = np.full(len(NAMES), np.nan)
    if hi <= lo:
        return out
    vv, tt = v[lo:hi], f_time[lo:hi]
    ok = ~np.isnan(vv)
    if ok.any():
        s = vv[ok]
        out[0:5] = [*np.percentile(s, [10, 50, 90]), s.max(), s.mean()]
        out[9] = (s < STOP_MPS).mean()
        cons = ok[1:] & ok[:-1]                         # consecutive pairs in the window
        if cons.any():
            dv = np.abs(np.diff(vv))[cons] / np.diff(tt)[cons]
            out[7:9] = [dv.mean(), dv.max()]
            out[10] = float(((vv[:-1] >= MOVING_MPS) & (vv[1:] < STOP_MPS))[cons].sum())
        else:
            out[10] = 0.0
        d = dist[lo + 1:hi]                             # pairs with both ends in the window
        out[13] = np.nansum(d) if (~np.isnan(d)).any() else np.nan
    for j, off in ((5, 0), (6, 60)):                    # pair ending at minute t, and at t + 1
        k = np.flatnonzero(np.abs(f_ts[lo:hi] - (ts + off)) <= 30)
        if len(k) and not np.isnan(vv[k[0]]):
            out[j] = vv[k[0]]
    b = brg[lo:hi]
    b = b[~np.isnan(b)]                                 # consecutive valid bearings (a stop has none)
    if len(b) >= 2:
        t_ = _turn(b)
        out[11:13] = [t_.mean(), t_.max()]
    if hi - lo >= 2:
        out[14] = haversine_m(lat[lo], lon[lo], lat[hi - 1], lon[hi - 1])
        if out[13] == out[13] and out[13] >= 1.0:
            out[15] = min(out[14] / out[13], 1.0)
    return out


def features(minutes: pd.DataFrame, gps: pd.DataFrame) -> np.ndarray:
    """minutes: user, session, ts (any order) -> (N, 16) float32 in minutes' order."""
    g = prepare(gps)
    out = np.full((len(minutes), len(NAMES)), np.nan, np.float32)
    groups = {k: d for k, d in g.groupby(["user", "session"], sort=False)}
    m = minutes.reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for (u, s), idx in m.groupby(["user", "session"], sort=False).indices.items():
            d = groups.get((u, s))
            if d is None:
                continue
            arr = [d[c].to_numpy() for c in ("ts", "time", "pair_v", "pair_dist", "brg", "lat", "lon")]
            for i in idx:
                out[i] = window_features(m.ts.iat[i], *arr)
    return out


def covered(f: np.ndarray) -> np.ndarray:
    """A minute is GPS-covered when its window has at least one pair speed (v_p50 not NaN)."""
    return ~np.isnan(f[:, NAMES.index("v_p50")])
