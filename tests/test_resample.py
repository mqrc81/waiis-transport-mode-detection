import numpy as np
import pytest

from tmd import config
from tmd.resample import Segment, resample

TARGET_HZ = 25.0


def _uniform_t(fs: float, dur: float, t0: float = 0.0) -> np.ndarray:
    return t0 + np.arange(round(dur * fs)) / fs


def _interior(seg: Segment, trim_s: float = 1.0) -> Segment:
    """Drop `trim_s` at each end so filter edge transients don't enter the checks."""
    m = (seg.t >= seg.t[0] + trim_s) & (seg.t <= seg.t[-1] - trim_s)
    return Segment(seg.t[m], seg.x[m])


def _assert_uniform(seg: Segment) -> None:
    np.testing.assert_allclose(np.diff(seg.t), 1.0 / TARGET_HZ, atol=1e-9)


def _sine_2ch(t: np.ndarray, f: float) -> np.ndarray:
    return np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)])


def _ramp(t: np.ndarray) -> np.ndarray:
    return (0.5 * t + 1.0)[:, None]


@pytest.mark.parametrize("fs_in", [20, 50, 100, 200])
def test_amplitude_preserved_2hz(fs_in):
    t = _uniform_t(fs_in, 20.0)
    segs = resample(t, _sine_2ch(t, 2.0))

    assert len(segs) == 1
    seg = segs[0]
    _assert_uniform(seg)
    assert seg.x.shape == (len(seg.t), 2)
    inner = _interior(seg)
    # Comparing against the true signal at output times checks amplitude and time alignment.
    assert np.max(np.abs(inner.x - _sine_2ch(inner.t, 2.0))) < 0.02


def _amplitude_near(y: np.ndarray, fs: float, f: float, tol: float = 0.5) -> float:
    spec = 2 * np.abs(np.fft.rfft(y)) / len(y)
    freqs = np.fft.rfftfreq(len(y), 1 / fs)
    return float(spec[np.abs(freqs - f) <= tol].max())


def test_20hz_attenuated_no_alias():
    t = _uniform_t(100, 20.0)
    x = np.sin(2 * np.pi * 20.0 * t + 0.3)[:, None]

    # Sanity: naive index skipping folds 20 Hz onto 5 Hz at full amplitude.
    assert _amplitude_near(x[::4, 0], TARGET_HZ, 5.0) > 0.8

    segs = resample(t, x)
    assert len(segs) == 1
    inner = _interior(segs[0])
    assert np.sqrt(np.mean(inner.x**2)) < 0.05
    assert _amplitude_near(inner.x[:, 0], TARGET_HZ, 5.0) < 0.05


def test_jitter_20pct():
    rng = np.random.default_rng(config.SEED)
    dt = 0.01 * (1 + rng.uniform(-0.2, 0.2, size=2000))
    t = np.cumsum(dt)
    segs = resample(t, _sine_2ch(t, 2.0))

    assert len(segs) == 1
    _assert_uniform(segs[0])
    inner = _interior(segs[0])
    assert np.max(np.abs(inner.x - _sine_2ch(inner.t, 2.0))) < 0.03


def test_duplicate_timestamps():
    rng = np.random.default_rng(config.SEED)
    t = _uniform_t(100, 20.0)
    x = _sine_2ch(t, 2.0)
    dup = rng.choice(len(t), size=len(t) // 20, replace=False)
    # Duplicates come after the originals; the first occurrence must win.
    t_dup = np.concatenate([t, t[dup]])
    x_dup = np.concatenate([x, np.full((len(dup), 2), 999.0)])

    clean = resample(t, x)
    dirty = resample(t_dup, x_dup)

    assert len(dirty) == len(clean) == 1
    np.testing.assert_allclose(dirty[0].t, clean[0].t)
    np.testing.assert_allclose(dirty[0].x, clean[0].x)
    assert np.max(np.abs(dirty[0].x)) < 2.0


def test_unsorted_input():
    rng = np.random.default_rng(config.SEED)
    t = _uniform_t(100, 20.0)
    x = _sine_2ch(t, 2.0)
    perm = rng.permutation(len(t))

    ref = resample(t, x)
    shuffled = resample(t[perm], x[perm])

    assert len(shuffled) == len(ref) == 1
    np.testing.assert_allclose(shuffled[0].t, ref[0].t)
    np.testing.assert_allclose(shuffled[0].x, ref[0].x)


def test_short_gap_interpolated():
    t = _uniform_t(100, 20.0)
    keep = (t < 8.0) | (t >= 8.5)
    t = t[keep]
    segs = resample(t, _ramp(t), max_gap_s=1.0)

    assert len(segs) == 1
    seg = segs[0]
    _assert_uniform(seg)
    assert seg.t[0] <= 0.05 and seg.t[-1] >= 19.9
    assert np.any((seg.t > 8.0) & (seg.t < 8.5))
    # Linear interpolation is exact on a ramp, so any error comes from the gap handling.
    inner = _interior(seg)
    assert np.max(np.abs(inner.x - _ramp(inner.t))) < 1e-3


def test_long_gap_splits():
    t = _uniform_t(100, 20.0)
    keep = (t < 8.0) | (t >= 11.0)
    t = t[keep]
    segs = resample(t, _ramp(t), max_gap_s=1.0)

    assert len(segs) == 2
    first, second = segs
    assert first.t[-1] <= 8.0
    assert second.t[0] >= 11.0
    for seg in segs:
        _assert_uniform(seg)
        assert not np.any((seg.t > 8.0) & (seg.t < 11.0))


def test_grid_aligned_to_target_multiples():
    t = _uniform_t(100, 10.0, t0=3.013)
    segs = resample(t, _sine_2ch(t, 2.0))

    assert len(segs) == 1
    k = segs[0].t * TARGET_HZ
    np.testing.assert_allclose(k, np.round(k), atol=1e-9)
    assert segs[0].t[0] >= t[0]


def test_invalid_input_raises():
    t = _uniform_t(100, 2.0)
    x = _sine_2ch(t, 2.0)
    with pytest.raises(ValueError):
        resample(t[:-1], x)
    with pytest.raises(ValueError):
        resample(t, x[:, 0])
    with pytest.raises(ValueError):
        resample(t, x, target_hz=0)
