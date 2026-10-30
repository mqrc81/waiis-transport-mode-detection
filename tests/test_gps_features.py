"""tmd.gps_features on synthetic one-fix-per-minute tracks."""
import numpy as np
import pandas as pd

from tmd import gps_features as gf

LAT0, M_PER_DEG = 50.0, 111_195.0


def _track(steps_m, start_ts=0, session=0, user="u"):
    """Fixes one minute apart; steps_m: list of (north_m, east_m) moves between consecutive fixes."""
    lat, lon = [LAT0], [0.0]
    for n, e in steps_m:
        lat.append(lat[-1] + n / M_PER_DEG)
        lon.append(lon[-1] + e / (M_PER_DEG * np.cos(np.radians(LAT0))))
    t = start_ts + 60 * np.arange(len(lat))
    return pd.DataFrame({"user": user, "session": session, "ts": t, "time": t.astype(float), "lat": lat, "lon": lon})


def _f(gps, ts):
    m = pd.DataFrame({"user": ["u"], "session": [0], "ts": [ts]})
    return dict(zip(gf.NAMES, gf.features(m, gps)[0]))


def test_constant_speed_straight():
    f = _f(_track([(600, 0)] * 6), ts=180)              # 10 m/s north
    assert abs(f["v_p50"] - 10) < 0.05 and abs(f["v_max"] - 10) < 0.05 and abs(f["v_now"] - 10) < 0.05
    assert f["stop_frac"] == 0 and f["n_stops"] == 0 and f["acc_max"] < 1e-3
    assert f["turn_max"] < 0.1 and abs(f["straightness"] - 1) < 1e-3


def test_stop_and_turn():
    f = _f(_track([(600, 0), (600, 0), (0, 0), (0, 600), (0, 600)]), ts=120)
    assert f["n_stops"] == 1 and 0 < f["stop_frac"] < 1
    assert abs(f["turn_max"] - 90) < 0.5                # north, then east (the stop has no bearing)
    assert f["straightness"] < 1


def test_no_fixes_in_window_is_nan_and_not_covered():
    gps = _track([(600, 0)] * 3)
    m = pd.DataFrame({"user": ["u"], "session": [0], "ts": [3600]})
    f = gf.features(m, gps)
    assert np.isnan(f).all() and not gf.covered(f)[0]


def test_v2_uniform_stop_threshold_and_names():
    """0.8 m/s (48 m per minute) is stopped in gps-v2 (< 1 m/s) but not in gps-v1 (< 0.5 m/s); v2 drops the three
    distance features and keeps the first 13 identical except the stop features."""
    gps = _track([(600, 0), (48, 0), (48, 0), (600, 0), (48, 0)])
    m = pd.DataFrame({"user": ["u"], "session": [0], "ts": [180]})
    f1 = dict(zip(gf.names("gps-v1"), gf.features(m, gps, "gps-v1")[0]))
    f2 = dict(zip(gf.names("gps-v2"), gf.features(m, gps, "gps-v2")[0]))
    assert len(f2) == 13 and "path_m" not in f2 and gf.names("gps-v1") == gf.NAMES
    assert f1["stop_frac"] == 0 and f2["stop_frac"] > 0
    assert f1["n_stops"] == 0 and f2["n_stops"] == 2          # 10 -> 0.8 twice; v1 needs < 0.5
    for k in f2:
        if k not in ("stop_frac", "n_stops"):
            assert f1[k] == f2[k] or (np.isnan(f1[k]) and np.isnan(f2[k])), k


def test_v2_empty_window_has_v2_length():
    gps = _track([(600, 0)] * 3)
    m = pd.DataFrame({"user": ["u"], "session": [0], "ts": [3600]})
    assert gf.features(m, gps, "gps-v2").shape == (1, 13)
