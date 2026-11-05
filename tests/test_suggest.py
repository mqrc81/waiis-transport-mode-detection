"""tmd.legs.predict_legs / gps_only_legs and tmd.legs_adapter on synthetic minutes (no WAIIS data)."""
import json

import numpy as np
import pandas as pd

from tmd import legs as L
from tmd import legs_adapter as A


def _p(fine_ids, p=0.9):
    """(N, 9) fine probabilities with p on the given fine class (1-based ids)."""
    out = np.full((len(fine_ids), 9), (1 - p) / 8)
    out[np.arange(len(fine_ids)), np.asarray(fine_ids) - 1] = p
    return out


def test_predict_legs_fields_and_hint():
    walk, subway, tram = 2, 8, 9
    ids = [walk] * 4 + [subway] * 6 + [tram] * 2
    ts = np.arange(len(ids)) * 60
    cov = np.array([True] * 4 + [False] * 8)
    lg = L.predict_legs(np.zeros(len(ids), int), ts, _p(ids), cov, L.Rules(c=0.5))
    assert list(lg.top1) == [L.MODES.index("walking"), L.MODES.index("rail")]   # subway + tram stay one rail leg
    rail = lg.iloc[1]
    assert rail.hint == "metro" and rail.coverage == "acc"
    assert {f["code"] for f in rail.flag_reasons} >= {"no_gps"}
    assert lg.iloc[0].coverage == "acc+gps"


def test_low_confidence_keeps_top2():
    p = np.full((5, 9), 0.0)
    p[:, 4], p[:, 5] = 0.45, 0.55                    # car 0.45, bus 0.55
    lg = L.predict_legs(np.zeros(5, int), np.arange(5) * 60, p, np.ones(5, bool), L.Rules(c=0.6))
    r = lg.iloc[0]
    assert r["mode"] == L.UNKNOWN and r.top1 == L.MODES.index("bus") and r.top2 == L.MODES.index("car")
    assert r.flag_reasons[0]["code"] == "low_confidence"


def _fixes(t, lat, lon):
    return pd.DataFrame({"t": np.asarray(t, float), "lat": lat, "lon": lon, "accuracy_m": 10.0})


def test_gps_only_still_and_moving():
    legs = pd.DataFrame({"start": [1000.0, 5000.0], "end": [2000.0, 6000.0]})
    # still stretch 2000-5000 (fixes 2 m apart); moving stretch after 6000 (fixes ~1 km apart in 600 s)
    f = _fixes([0, 900, 2500, 4500, 6500, 7100], [40.0, 40.0, 40.0, 40.00002, 40.0, 40.009], [-3.7] * 6)
    go = L.gps_only_legs(legs, f, 0, 86400)
    # 0-1000 s before the first leg (2 fixes, 0 m apart) and 2000-5000 s: still; after 6000 s: moving
    assert list(go.source) == ["rule:gps_only_still", "rule:gps_only_still", "rule:gps_only_moving"]
    mv = go.iloc[2]
    assert mv["mode"] == L.UNKNOWN and mv.flag_reasons[0]["code"] == "gps_only_moving" and mv.flag_reasons[0]["speed_across_mps"] > 1
    assert len(L.gps_only_legs(legs.iloc[:0], f.iloc[:1], 0, 86400)) == 0     # one fix: no leg


def test_adapter_round_trip(tmp_path):
    ids = [2] * 3 + [5] * 4
    lg = L.predict_legs(np.zeros(7, int), np.arange(7) * 60, _p(ids), np.ones(7, bool), L.Rules(c=0.5))
    doc = A.document(lg, "u", "2026-11-03", "E2", "10:test_E2_seed0", "legs-v1 (11)", {"gps_rule": "C1"}, "abc")
    A.write(doc, tmp_path / "x.json")
    d = json.loads((tmp_path / "x.json").read_text())
    assert [l["mode"] for l in d["legs"]] == ["walking", "car"] and len(d["minutes"]) == 7
    assert d["legs"][1]["top2"][0]["mode"] == "car" and d["legs"][0]["start_ms"] == 0
    assert set(d["minutes"][0]["p"]) == set(L.MODES)
    t = A.read_legs(tmp_path / "x.json")
    assert t.end.iloc[-1] == 7 * 60
