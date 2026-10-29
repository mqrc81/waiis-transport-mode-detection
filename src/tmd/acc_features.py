"""Accelerometer-only window features for 15 s fragments (experiment 05, version FEATURES_VERSION).

Blocks (each a list of names; `features` returns them in this order):
    BASE   features v1 (tmd.features) without the gyroscope channel, on the whole window: 4 channels x 29 = 116
    SUB    the same 116 on the 5 sub-windows of 5 s / 2.5 s stride (tmd.windows), then mean / std / min / max = 464
    GRAV   gravity as the window's mean direction g = mean(a) / |mean(a)|: orientation ratio, tilt of the 1 s moving
           mean and of the sub-window directions against g, projections on the fixed g = 21

Every feature is an angle, a norm or a dot product between vectors in the same phone frame, so it is unchanged by
any sign flip or permutation of the axes (iOS vs Android conventions); tests/test_acc_features.py checks this.
`lowpass_hz` filters the 3 axes (4th-order Butterworth, zero-phase) before everything.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from tmd.features import _SOS_GRAV, BANDS, PER_CHANNEL, _channel_features
from tmd.resample import FS
from tmd.windows import WIN, starts

FEATURES_VERSION = "acc-v1"
CHANNELS = ["acc_norm", "acc_vert", "acc_hor", "jerk_norm"]
BASE = [f"{c}__{f}" for c in CHANNELS for f in PER_CHANNEL]
SUB_AGG = ("mean", "std", "min", "max")
SUB = [f"sub_{a}__{n}" for a in SUB_AGG for n in BASE]
_BANDLOG = [f"bandlog_{lo:g}_{min(hi, FS / 2):g}" for lo, hi in BANDS]
_FIX_STATS = ["std", "iqr"] + _BANDLOG
GRAV = (["grav_ratio", "tilt_mean", "tilt_std", "tilt_p95", "tilt_max", "subtilt_max", "subtilt_std",
         "subtilt_step_mean", "subtilt_step_max"] + [f"{c}__{f}" for c in ("vfix", "hfix") for f in _FIX_STATS])
# Context base set (experiment 05, step D): 10 whole-window features per channel.
CONTEXT_BASE = [f"{c}__{f}" for c in CHANNELS
                for f in ["std", "iqr", "p50", "acf_peak", "dom_freq", "spec_entropy"] + _BANDLOG]


def lowpass(acc: np.ndarray, hz: float) -> np.ndarray:
    sos = butter(4, hz, fs=FS, output="sos")
    return sosfiltfilt(sos, acc, axis=1)


def channels(acc: np.ndarray) -> np.ndarray:
    """acc (N, T, 3) m/s2 -> (N, T, 4): |a|, a . g_hat, horizontal |a|, |da/dt| (as tmd.features.channels)."""
    g = sosfiltfilt(_SOS_GRAV, acc, axis=1)
    g_hat = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-6)
    vert = (acc * g_hat).sum(-1)
    hor = np.linalg.norm(acc - vert[..., None] * g_hat, axis=-1)
    jerk = np.linalg.norm(np.gradient(acc, 1 / FS, axis=1), axis=-1)
    return np.stack([np.linalg.norm(acc, axis=-1), vert, hor, jerk], -1)


def base_features(acc: np.ndarray) -> np.ndarray:
    ch = channels(acc)
    return np.concatenate([_channel_features(ch[..., i]) for i in range(ch.shape[-1])], 1)


def sub_features(acc: np.ndarray) -> np.ndarray:
    f = np.stack([base_features(acc[:, s:s + WIN]) for s in starts(acc.shape[1])], 1)     # (N, 5, 116)
    return np.concatenate([np.nanmean(f, 1), np.nanstd(f, 1), np.nanmin(f, 1), np.nanmax(f, 1)], 1)


def _angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle in degrees between vectors along the last axis."""
    c = (u * v).sum(-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def gravity_features(acc: np.ndarray) -> np.ndarray:
    m = acc.mean(1)                                                                  # (N, 3)
    g_hat = m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)
    ratio = np.linalg.norm(m, axis=-1) / np.maximum(np.linalg.norm(acc, axis=-1).mean(1), 1e-12)
    tilt = _angle(uniform_filter1d(acc, FS, axis=1, mode="nearest"), g_hat[:, None])  # (N, T)
    d = np.stack([acc[:, s:s + WIN].mean(1) for s in starts(acc.shape[1])], 1)       # (N, 5, 3)
    sub = _angle(d, g_hat[:, None])
    step = _angle(d[:, 1:], d[:, :-1])
    v = (acc * g_hat[:, None]).sum(-1)
    h = np.linalg.norm(acc - v[..., None] * g_hat[:, None], axis=-1)
    pick = [PER_CHANNEL.index(f) for f in _FIX_STATS]
    return np.column_stack([ratio, tilt.mean(1), tilt.std(1), np.percentile(tilt, 95, axis=1), tilt.max(1),
                            sub.max(1), sub.std(1), step.mean(1), step.max(1),
                            _channel_features(v)[:, pick], _channel_features(h)[:, pick]])


def names(sub: bool = True, grav: bool = True) -> list[str]:
    return BASE + (SUB if sub else []) + (GRAV if grav else [])


def features(acc: np.ndarray, sub: bool = True, grav: bool = True, lowpass_hz: float | None = None) -> np.ndarray:
    """acc (N, T, 3) m/s2 at 25 Hz -> (N, len(names(sub, grav))) float32."""
    if len(acc) == 0:
        return np.zeros((0, len(names(sub, grav))), np.float32)
    acc = acc.astype(np.float64)
    if lowpass_hz is not None:
        acc = lowpass(acc, lowpass_hz)
    with warnings.catch_warnings():       # skew/kurtosis of near-constant windows, all-NaN sub-window stats
        warnings.simplefilter("ignore", RuntimeWarning)
        f = [base_features(acc)]
        if sub:
            f.append(sub_features(acc))
        if grav:
            f.append(gravity_features(acc))
    return np.concatenate(f, 1).astype(np.float32)
