"""Experiment 12: GPS gaps, moving-gap rule, rail boost, GPS-only gap legs."""
import numpy as np
import pandas as pd

from tmd import gps as tgps
from tmd import legs as L

LAT0, LON0 = 51.5, -0.1
M_PER_DEG_LAT = 111_195.0


def track(times, north_m):
    return pd.DataFrame({"user": "u", "time": np.asarray(times, float),
                         "lat": LAT0 + np.asarray(north_m, float) / M_PER_DEG_LAT, "lon": LON0})


def test_gaps_speed_across():
    # fixes every 20 s, then a 600 s gap over 6 km (10 m/s), then a 50 s gap (below min)
    t = [0, 20, 40, 640, 690]
    tr = track(t, [0, 0, 0, 6000, 6000])
    g = tgps.gaps(tr, ["user"], min_gap_s=60)
    assert len(g) == 1
    r = g.iloc[0]
    assert (r.t0, r.t1, r["dt"]) == (40, 640, 600)
    assert abs(r.dist_m - 6000) < 5 and abs(r.v_across - 10) < 0.01


def test_gaps_never_cross_groups():
    a, b = track([0, 20], [0, 0]), track([1000, 1020], [9000, 9000]).assign(user="v")
    assert len(tgps.gaps(pd.concat([a, b]), ["user"])) == 0
    assert len(tgps.gaps(pd.concat([a, b.assign(user="u")]), ["user"])) == 1


def test_minute_gaps():
    g = tgps.gaps(track([0, 20, 40, 640], [0, 0, 0, 6000]), ["user"])
    m = pd.DataFrame({"user": ["u", "u", "u", "w"], "ts": [0, 60, 600, 60]})   # centres 30, 90, 630, 90
    mg = tgps.minute_gaps(m, g, ["user"])
    assert mg.gap.tolist() == [-1, 0, 0, -1]
    assert np.allclose(mg.gap_v.to_numpy()[1:3], 10, atol=0.01)
    mg0 = tgps.minute_gaps(m[m.user == "u"], tgps.gaps(track([0, 20, 40, 640], [0, 0, 0, 6000]), []), [])
    assert mg0.gap.tolist() == [-1, 0, 0]


def test_gap_rule_moving():
    r = L.GapRule(g_min_s=300, v_lo=2, v_hi=25)
    assert r.moving([300, 299, 600, 600, 600], [2, 10, 1.9, 25, 24.9]).tolist() == [True, False, False, False, True]


def test_boost_rail():
    raw = np.array([[0.5, 0.3, 0, 0, 0, 0, 0.1, 0.1], [0.5, 0.3, 0, 0, 0, 0, 0.1, 0.1]])
    assert L.boost_rail(raw, np.array([True, False]), 1) is raw
    b = L.boost_rail(raw, np.array([True, False]), 4)
    assert np.allclose(b[1], raw[1]) and np.allclose(b.sum(1), 1)
    assert np.allclose(b[0], np.array([0.5, 0.3, 0, 0, 0, 0, 0.4, 0.4]) / 1.6)
    raw9 = np.column_stack([raw, [0.0, 0.0]])
    raw9[:, 8] = 0.1                                       # tram
    raw9 /= raw9.sum(1, keepdims=True)
    b9 = L.boost_rail(raw9, np.array([True, True]), 2)
    tr = raw9[:, 6:8].sum(1)
    assert np.allclose(b9[:, 6:8].sum(1), 2 * tr / (1 + tr))           # train + subway doubled, renormalised
    assert np.allclose(b9[:, 8], raw9[:, 8] / (1 + tr))                 # tram only renormalised


def test_gps_only_gap_leg():
    # one model leg 0-600 s; then fixes walking slowly to 900 s, a 20 min gap over 12 km (10 m/s), fixes to 2400 s
    legs = pd.DataFrame({"start": [0.0], "end": [600.0]})
    t = list(range(600, 901, 20)) + list(range(2100, 2401, 20))
    north = [1.0 * (x - 600) for x in t[:16]] + [300 + 12000 + (x - 2100) for x in t[16:]]
    fx = track(t, north).rename(columns={"time": "t"})
    off = L.gps_only_legs(legs, fx, 0, 86400)
    on = L.gps_only_legs(legs, fx, 0, 86400, L.GapRule())
    assert len(off) == 1 and off.source.iat[0] == "rule:gps_only_moving"
    gl = on[on.source == "rule:gps_gap_moving"]
    assert len(gl) == 1
    r = gl.iloc[0]
    assert (r.start, r.end, r["mode"], r.hint) == (900, 2100, L.UNKNOWN, "underground_rail")
    f = r.flag_reasons[0]
    assert f["code"] == "gps_gap_moving" and f["gap_minutes"] == 20.0 and abs(f["speed_across_mps"] - 10) < 0.05
    assert (f["fix_before_ms"], f["fix_after_ms"]) == (900_000, 2_100_000)
    rest = on[on.source != "rule:gps_gap_moving"]
    # the stretch around the gap: 600-900 s (exactly 5 min, walking 1 m/s) and 2100-2400 s, both 11's moving legs
    assert rest[["start", "end"]].to_numpy().tolist() == [[600, 900], [2100, 2400]]
    assert (rest.source == "rule:gps_only_moving").all()


def test_predict_legs_gap_reason():
    n = 6
    p = np.zeros((n, 9))
    p[:, 7] = 1.0
    lg = L.predict_legs(np.zeros(n, int), np.arange(n) * 60, p, np.zeros(n, bool), L.RAW,
                        np.array([0, 1, 1, 0, 0, 0], bool))
    codes = [f["code"] for f in lg.flag_reasons.iat[0]]
    assert "gps_gap_moving" in codes and lg.attrs["minutes"].in_gap.sum() == 2
    lg0 = L.predict_legs(np.zeros(n, int), np.arange(n) * 60, p, np.zeros(n, bool), L.RAW)
    assert "gps_gap_moving" not in [f["code"] for f in lg0.flag_reasons.iat[0]]
