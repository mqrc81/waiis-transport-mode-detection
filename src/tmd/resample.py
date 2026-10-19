"""Resampling of irregularly sampled sensor streams onto a uniform target-rate grid."""

from fractions import Fraction
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import resample_poly

_EPS = 1e-9


class Segment(NamedTuple):
    """A gap-free stretch of signal on a uniform time grid.

    Attributes
    ----------
    t : np.ndarray
        Shape (M,), seconds, spaced exactly ``1 / target_hz`` and aligned to multiples of it.
    x : np.ndarray
        Shape (M, C), float64.
    """

    t: np.ndarray
    x: np.ndarray


def resample(
        t: ArrayLike,
        x: ArrayLike,
        target_hz: float = 25.0,
        max_gap_s: float = 1.0,
) -> list[Segment]:
    """Resample an irregularly sampled multichannel signal to a uniform rate.

    The input is sorted by time and duplicate timestamps are dropped (the first occurrence
    wins). It is then split wherever consecutive samples are more than ``max_gap_s``
    apart. Each segment is linearly interpolated onto a uniform grid at its own native rate,
    then converted to ``target_hz`` with a polyphase anti-aliasing FIR
    (`scipy.signal.resample_poly`). Index skipping is never used.

    Parameters
    ----------
    t : array_like
        Shape (N,), timestamps in seconds, in any order.
    x : array_like
        Shape (N, C), samples.
    target_hz : float, default 25.0
        Output sampling rate.
    max_gap_s : float, default 1.0
        Gaps strictly longer than this split the series. Shorter gaps are interpolated.

    Returns
    -------
    list of Segment
        Uniform segments in time order. Output timestamps are integer multiples of
        ``1 / target_hz``, so streams resampled separately can be joined exactly on ``t``.
        Segments with fewer than 2 output samples are dropped.

    Raises
    ------
    ValueError
        If ``x`` is not 2-D, the lengths of ``t`` and ``x`` differ, or ``target_hz <= 0``.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"t must be 1-D, got shape {t.shape}")
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (N, C), got shape {x.shape}")
    if len(t) != len(x):
        raise ValueError(f"t and x lengths differ: {len(t)} vs {len(x)}")
    if target_hz <= 0:
        raise ValueError(f"target_hz must be positive, got {target_hz}")

    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, x = t[keep], x[keep]

    breaks = np.flatnonzero(np.diff(t) > max_gap_s) + 1
    segments = []
    for ts, xs in zip(np.split(t, breaks), np.split(x, breaks)):
        seg = _resample_segment(ts, xs, target_hz)
        if seg is not None:
            segments.append(seg)
    return segments


def _resample_segment(t: np.ndarray, x: np.ndarray, target_hz: float) -> Segment | None:
    if len(t) < 2:
        return None
    fs_in = max(1, round(1.0 / np.median(np.diff(t))))

    # Start the native grid on a multiple of 1/target_hz so the output lands on k / target_hz.
    k0 = int(np.ceil(t[0] * target_hz - _EPS))
    t_start = k0 / target_hz
    n_in = int(np.floor((t[-1] - t_start) * fs_in + _EPS)) + 1
    if n_in < 2:
        return None
    grid = t_start + np.arange(n_in) / fs_in
    x_grid = np.column_stack([np.interp(grid, t, x[:, c]) for c in range(x.shape[1])])

    ratio = Fraction(target_hz / fs_in).limit_denominator(1000)
    y = resample_poly(
        x_grid, ratio.numerator, ratio.denominator, axis=0, padtype="line"
    )

    t_out = (k0 + np.arange(len(y))) / target_hz
    inside = t_out <= grid[-1] + _EPS
    if inside.sum() < 2:
        return None
    return Segment(t_out[inside], y[inside])
