"""tmd.acc_features: block sizes, agreement with features v1, and invariance to axis sign and order."""
import itertools

import numpy as np
import pytest

from tmd import acc_features as af
from tmd.features import NAMES, window_features


def _windows(n=6, t=375, seed=0):
    rng = np.random.default_rng(seed)
    tt = np.arange(t) / 25
    w = rng.normal(0, 0.5, (n, t, 3))
    w += np.stack([np.sin(2 * np.pi * (1 + i) * tt) for i in range(3)], -1)[None] * rng.uniform(0, 2, (n, 1, 3))
    g = rng.normal(size=(n, 3))
    w += 9.81 * (g / np.linalg.norm(g, axis=1, keepdims=True))[:, None]
    return w


def test_block_sizes():
    assert (len(af.BASE), len(af.SUB), len(af.GRAV), len(af.CONTEXT_BASE)) == (116, 464, 21, 40)
    assert af.features(_windows()).shape == (6, 601)
    assert set(af.CONTEXT_BASE) <= set(af.BASE)


def test_base_equals_features_v1_without_gyro():
    w = _windows()
    v1 = window_features(np.concatenate([w, np.zeros_like(w)], -1))
    keep = [i for i, n in enumerate(NAMES) if not n.startswith("gyro_norm")]
    assert [NAMES[i] for i in keep] == af.BASE
    np.testing.assert_allclose(af.features(w, sub=False, grav=False), v1[:, keep], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("lp", [None, 10.0])
def test_invariant_to_axis_sign_and_order(lp):
    w = _windows()
    ref = af.features(w, lowpass_hz=lp).astype(np.float64)
    for perm in itertools.permutations(range(3)):
        for sign in itertools.product([1, -1], repeat=3):
            f = af.features(w[..., list(perm)] * np.array(sign), lowpass_hz=lp)
            np.testing.assert_allclose(f, ref, rtol=1e-3, atol=1e-3, err_msg=f"perm {perm} sign {sign}")
