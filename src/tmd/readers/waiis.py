"""WAIIS app days (raw export of the WAIIS package; experiment 11): one folder per user and UTC day,
`<data>/raw/waiis/user=<key>/date=<YYYY-MM-DD>/` with `gps.parquet`, `imu.parquet`, `_manifest.json`, segments, OSM
context and the package's own predictions.

- GPS: `t_ms` (UTC epoch ms), `lat`, `lon`, `accuracy_m`, `speed_mps` (missing for ~26% of fixes), `heading_deg`, ...;
  `read_gps` adds `t` [s]. Fix states from the speed over a 60 s baseline (`fix_states`; 08's descriptive definition).
- Accelerometer: `t_ms`, `ax`, `ay`, `az` in m/s2 with gravity, ~24.9 Hz in ~5 min bursts while moving. `read_acc`
  returns times [s] and xyz **in the Android axis convention** (v2's rule A1), using the device table.
- Platform and phone model are not in the data: `<data>/processed/waiis/metadata/waiis_devices.csv` (`device`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import candidates
from tmd.gps import haversine_m
from tmd.paths import PROCESSED, RAW

ROOT = RAW / "waiis"
DEVICES = PROCESSED / "waiis" / "metadata" / "waiis_devices.csv"   # outside git (moved 2026-11-05)
SESSION_JOIN_S = 120.0          # bursts whose pause is <= this join into one session (11, approved 2026-11-04)
COARSE_M = 500.0                # a fix with accuracy above this counts as missing (11)
STATE_BINS = [-np.inf, 0.5, 2.5, 4.0, np.inf]
STATE_NAMES = ["still", "walk", "2.5-4", "vehicle"]


@dataclass(frozen=True)
class Day:
    user: str
    date: str
    path: Path


def days(root: Path = ROOT) -> list[Day]:
    out = []
    for d in sorted(root.glob("user=*/date=*")):
        out.append(Day(d.parent.name.split("=", 1)[1], d.name.split("=", 1)[1], d))
    return out


# ---------------------------------------------------------------- device table

def devices(path: Path = DEVICES) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def device(user: str, date: str, table: pd.DataFrame | None = None) -> pd.Series:
    """The device row covering (user, UTC date); fails on none or several."""
    t = devices() if table is None else table
    t = t[t.user_key == user]
    ok = (t.valid_from <= date) & ((t.valid_to == "") | (date <= t.valid_to))
    hit = t[ok]
    if len(hit) != 1:
        raise LookupError(f"{len(hit)} device rows for user {user} on {date} ({DEVICES.name})")
    return hit.iloc[0]


def axis_sign(dev: pd.Series) -> float:
    """Factor that brings the app's axes to the Android convention (v2 rule A1; units are already m/s2)."""
    conv = dev.axis_convention
    if conv == "ios":
        return -1.0
    if conv == "android":
        return 1.0
    raise ValueError(f"axis convention {conv!r} for user {dev.user_key}: not usable")


# ---------------------------------------------------------------- reading

def read_gps(day: Day) -> pd.DataFrame:
    g = pd.read_parquet(day.path / "gps.parquet").sort_values("t_ms").reset_index(drop=True)
    g["t"] = g.t_ms / 1000.0
    return g


def read_imu(day: Day) -> pd.DataFrame:
    """The raw accelerometer table (app axes), sorted by time."""
    return pd.read_parquet(day.path / "imu.parquet").sort_values("t_ms").reset_index(drop=True)


def read_acc(day: Day, dev: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Times [s] and xyz (n, 3) m/s2 in the Android axis convention."""
    a = read_imu(day)
    dev = device(day.user, day.date) if dev is None else dev
    return a.t_ms.to_numpy() / 1e3, a[["ax", "ay", "az"]].to_numpy(float) * axis_sign(dev)


# ---------------------------------------------------------------- sessions and minute windows

def session_ids(t: np.ndarray, join_s: float = SESSION_JOIN_S) -> np.ndarray:
    """Session id per sorted sample time: a pause > join_s between samples starts a new session."""
    return np.concatenate([[0], np.cumsum(np.diff(t) > join_s)]).astype(np.int32)


def minute_windows(t: np.ndarray, xyz: np.ndarray, join_s: float = SESSION_JOIN_S
                   ) -> tuple[pd.DataFrame, np.ndarray]:
    """waiis_bench_v2's minute windows over the whole day (`tmd.candidates.trip_windows`, label-free), with the
    session of each window. Returns (rows with status 'ok', windows (n, 375, 3))."""
    w, x = candidates.trip_windows(t, xyz, t, np.full(len(t), "u"))
    w = w[w.status == "ok"].reset_index(drop=True)
    sess = session_ids(t, join_s)
    w["session"] = sess[np.clip(np.searchsorted(t, w.t_start.to_numpy()), 0, len(t) - 1)]
    return w, x


# ---------------------------------------------------------------- GPS states (descriptive)

def baseline_speed(t: np.ndarray, lat: np.ndarray, lon: np.ndarray, forward: bool = True, base: float = 60.0,
                   max_span: float = 180.0) -> np.ndarray:
    """Displacement speed to the first fix >= base s ahead (or behind); NaN if that fix is > max_span s away."""
    n = len(t)
    out = np.full(n, np.nan)
    for i in range(n):
        if forward:
            j = np.searchsorted(t, t[i] + base)
            if j >= n:
                continue
        else:
            j = np.searchsorted(t, t[i] - base, side="right") - 1
            if j < 0:
                continue
        dt = abs(t[j] - t[i])
        if dt <= max_span:
            out[i] = haversine_m(lat[i], lon[i], lat[j], lon[j]) / dt
    return out


def state(v) -> pd.Series:
    return pd.cut(v, STATE_BINS, right=False, labels=STATE_NAMES)


def fix_states(g: pd.DataFrame) -> pd.DataFrame:
    """Adds v_fwd, v_bwd, v60 (60 s baseline ahead, else behind), state, dt_next, v_next (consecutive fixes)."""
    g = g.copy()
    t, lat, lon = g.t.to_numpy(), g.lat.to_numpy(), g.lon.to_numpy()
    g["dt_next"] = g.t.shift(-1) - g.t
    g["v_fwd"] = baseline_speed(t, lat, lon, True)
    g["v_bwd"] = baseline_speed(t, lat, lon, False)
    g["v60"] = g.v_fwd.fillna(g.v_bwd)
    g["state"] = state(g.v60).astype(object).fillna("unknown")
    g["v_next"] = haversine_m(lat, lon, g.lat.shift(-1).to_numpy(), g.lon.shift(-1).to_numpy()) / g.dt_next
    return g


def speed_across(a: pd.Series, b: pd.Series) -> float:
    """Displacement speed [m/s] between two fixes (rows with t, lat, lon)."""
    return float(haversine_m(a.lat, a.lon, b.lat, b.lon) / (b.t - a.t))
