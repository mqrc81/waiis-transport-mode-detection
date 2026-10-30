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
