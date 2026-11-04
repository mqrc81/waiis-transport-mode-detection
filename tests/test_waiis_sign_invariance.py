"""E1's and E2's inputs on a WAIIS day don't depend on the axis-sign rule: one day read with iOS signs converted
(x -1, the rule) and as recorded gives identical minute windows (up to sign), identical 580 acc features and
identical acc-model probabilities. Skipped where the WAIIS data or 10's saved models are not on this machine."""
import numpy as np
import pytest

from tmd.features import acc as af
from tmd.paths import PROCESSED
from tmd.readers import waiis

MODELS = PROCESSED / "10_gps-acc-baseline_waiis-bench-v2" / "models"
N = 40                                   # minute windows used (keeps the test fast)


@pytest.fixture(scope="module")
def two_readings():
    days = waiis.days()
    if not days:
        pytest.skip("no WAIIS days on this machine")
    day = days[0]
    dev = waiis.device(day.user, day.date)
    out = {}
    for conv in ("ios", "android"):
        d = dev.copy()
        d["axis_convention"] = conv
        t, xyz = waiis.read_acc(day, d)
        w, x = waiis.minute_windows(t, xyz)
        out[conv] = (w.head(N), x[:N])
    return out


def test_windows_flip_exactly(two_readings):
    (wi, xi), (wa, xa) = two_readings["ios"], two_readings["android"]
    assert len(wi) == len(wa) > 0
    assert (wi.ts.to_numpy() == wa.ts.to_numpy()).all()
    assert np.array_equal(xi, -xa)


def test_features_identical(two_readings):
    fi = af.features(two_readings["ios"][1], sub=True, grav=False)
    fa = af.features(two_readings["android"][1], sub=True, grav=False)
    assert fi.shape[1] == 580
    np.testing.assert_allclose(fi, fa, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("model", ["E2", "E1"])
def test_acc_model_probabilities_identical(two_readings, model):
    lgb = pytest.importorskip("lightgbm")
    path = MODELS / f"test_{model}_seed0.txt"
    if not path.exists():
        pytest.skip(f"{path.name} not on this machine")
    b = lgb.Booster(model_file=str(path))
    pi = b.predict(af.features(two_readings["ios"][1], sub=True, grav=False))
    pa = b.predict(af.features(two_readings["android"][1], sub=True, grav=False))
    np.testing.assert_allclose(pi, pa, rtol=1e-9, atol=1e-12)
