"""Hand-crafted, orientation-independent window features (experiment 02, version FEATURES_VERSION).

Channels per window (all independent of how the phone is held):
    acc_norm   |a|
    acc_vert   a . g_hat         (g = gravity, low-pass 0.3 Hz of a, estimated inside the window only)
    acc_hor    |a - (a . g_hat) g_hat|
    gyro_norm  |w|
    jerk_norm  |da/dt|           (from the SHL 2024 reference, arXiv 2407.11048: SMV of the first gradient)

Per channel: distribution stats, crossings, autocorrelation peak, Hjorth parameters, and Welch-PSD band energies
(0-1, 1-3, 3-6, 6-12.5 Hz), dominant frequency, spectral centroid and entropy. Gravity is estimated per window so a
window's features depend on that window alone, as on the phone.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis, skew

from tmd.resample import FS

FEATURES_VERSION = "v1"
CHANNELS = ["acc_norm", "acc_vert", "acc_hor", "gyro_norm", "jerk_norm"]
BANDS = [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, FS / 2 + 1e-9)]
_SOS_GRAV = butter(2, 0.3, fs=FS, output="sos")
ACF_LAGS = (int(0.2 * FS), int(2.0 * FS))       # gait / pedalling periods 0.2-2 s
PER_CHANNEL = (["mean", "std", "min", "max", "p05", "p25", "p50", "p75", "p95", "iqr", "skew", "kurt",
                "mad_diff", "mcr", "acf_peak", "acf_lag_s", "hjorth_mob", "hjorth_comp"]
               + [f"bandlog_{lo:g}_{min(hi, FS / 2):g}" for lo, hi in BANDS]
               + [f"bandrel_{lo:g}_{min(hi, FS / 2):g}" for lo, hi in BANDS]
               + ["dom_freq", "spec_centroid", "spec_entropy"])
NAMES = [f"{c}__{f}" for c in CHANNELS for f in PER_CHANNEL]


def channels(w: np.ndarray) -> np.ndarray:
    """w: (N, T, 6) acc m/s2 + gyro rad/s at 25 Hz -> (N, T, 5) orientation-independent channels."""
    acc, gyro = w[..., :3].astype(np.float64), w[..., 3:6].astype(np.float64)
    g = sosfiltfilt(_SOS_GRAV, acc, axis=1)
    g_hat = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-6)
    vert = (acc * g_hat).sum(-1)
    hor = np.linalg.norm(acc - vert[..., None] * g_hat, axis=-1)
    jerk = np.linalg.norm(np.gradient(acc, 1 / FS, axis=1), axis=-1)
    return np.stack([np.linalg.norm(acc, axis=-1), vert, hor, np.linalg.norm(gyro, axis=-1), jerk], -1)


def _channel_features(x: np.ndarray) -> np.ndarray:
    """x: (N, T) -> (N, len(PER_CHANNEL))."""
    n, t = x.shape
    mean, std = x.mean(1), x.std(1)
    q = np.percentile(x, [5, 25, 50, 75, 95], axis=1)
    xc = x - mean[:, None]
    safe = np.where(std > 1e-9, std, 1.0)
    d1 = np.diff(x, axis=1)
    d2 = np.diff(d1, axis=1)
    mcr = (np.diff(np.signbit(xc), axis=1)).sum(1) / (t / FS)                 # mean crossings per second
    # autocorrelation via FFT, normalised to lag 0
    f = np.fft.rfft(xc, n=2 * t, axis=1)
    acf = np.fft.irfft(f * np.conj(f), axis=1)[:, :t]
    acf = acf / np.where(acf[:, :1] > 1e-12, acf[:, :1], 1.0)
    lo, hi = ACF_LAGS
    seg = acf[:, lo:hi + 1]
    acf_peak, acf_lag = seg.max(1), (seg.argmax(1) + lo) / FS
    var0, var1, var2 = x.var(1), d1.var(1), d2.var(1)
    mob = np.sqrt(var1 / np.maximum(var0, 1e-12))
    comp = np.sqrt(var2 / np.maximum(var1, 1e-12)) / np.maximum(mob, 1e-12)
    fr, p = welch(x, fs=FS, nperseg=64, noverlap=32, axis=1, detrend="constant")
    df = fr[1] - fr[0]
    band = np.stack([p[:, (fr >= a) & (fr < b)].sum(1) * df for a, b in BANDS], 1)
    tot = np.maximum(p[:, 1:].sum(1) * df, 1e-12)                               # without DC
    p_ac = p[:, 1:]
    pn = p_ac / np.maximum(p_ac.sum(1, keepdims=True), 1e-12)
    ent = -(pn * np.log(np.maximum(pn, 1e-12))).sum(1) / np.log(pn.shape[1])
    return np.column_stack([
        mean, std, x.min(1), x.max(1), *q, q[3] - q[1], skew(x, axis=1) * (std > 1e-9),
        kurtosis(x, axis=1) * (std > 1e-9), np.abs(d1).mean(1), mcr, acf_peak, acf_lag, mob, comp,
        np.log10(band + 1e-9), band / tot[:, None],
        fr[1:][p_ac.argmax(1)], (p_ac * fr[1:]).sum(1) / np.maximum(p_ac.sum(1), 1e-12), ent,
    ])


def window_features(w: np.ndarray) -> np.ndarray:
    """w: (N, T, 6) at 25 Hz -> (N, len(NAMES)) float32."""
    if len(w) == 0:
        return np.zeros((0, len(NAMES)), np.float32)
    ch = channels(w)
    with warnings.catch_warnings():       # skew/kurtosis of near-constant windows; zeroed via std > 1e-9
        warnings.simplefilter("ignore", RuntimeWarning)
        f = [_channel_features(ch[..., i]) for i in range(ch.shape[-1])]
    return np.concatenate(f, 1).astype(np.float32)
