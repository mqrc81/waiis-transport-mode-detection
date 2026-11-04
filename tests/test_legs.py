"""tmd.legs: top-level mapping, leg rules and metrics on small hand-made timelines."""
import numpy as np

from tmd import legs as L

S, W, R, B, C, BUS, RAIL = range(7)


def _q(modes, p=0.9):
    q = np.full((len(modes), 7), (1 - p) / 6)
    q[np.arange(len(modes)), modes] = p
    return q


def _timeline(modes, seq=None, ts=None):
    n = len(modes)
    seq = np.zeros(n, int) if seq is None else np.asarray(seq)
    ts = np.arange(n) * 60 if ts is None else np.asarray(ts)
    return seq, ts, _q(np.asarray(modes))


def test_top_probs_and_mapping():
    p = np.zeros((1, 9))
    p[0, [6, 7, 8]] = [0.2, 0.3, 0.1]          # train, subway, tram
    p[0, 1] = 0.4                               # walk
    q = L.top_probs(p)
    assert np.isclose(q[0, RAIL], 0.6) and np.isclose(q[0, W], 0.4) and np.isclose(q.sum(), 1)
    assert list(L.fine_to_top(np.array([1, 2, 3, 7, 8, 9, 0]))) == [S, W, R, RAIL, RAIL, RAIL, -1]


def test_raw_runs_and_gaps():
    seq, ts, q = _timeline([C] * 3 + [S] * 2 + [C] * 3)
    lg = L.build(seq, ts, q)
    assert list(lg.top1) == [C, S, C]
    # a data gap of 12 missing minutes splits one mode; 10 joins (g = 10)
    ts2 = np.array([0, 60, 60 * 13, 60 * 14])
    assert len(L.build(np.zeros(4, int), ts2, _q(np.array([C] * 4)))) == 2
    ts3 = np.array([0, 60, 60 * 12, 60 * 13])
    assert len(L.build(np.zeros(4, int), ts3, _q(np.array([C] * 4)))) == 1


def test_still_absorbed_in_vehicle_not_in_walk():
    seq, ts, q = _timeline([C] * 5 + [S] * 2 + [C] * 5 + [W] * 4 + [S] * 2 + [W] * 4)
    lg = L.build(seq, ts, q, L.Rules(s_veh=2, s_other=0))
    assert list(lg.top1) == [C, W, S, W]
    lg = L.build(seq, ts, q, L.Rules(s_veh=2, s_other=2))
    assert list(lg.top1) == [C, W]


def test_min_length_merges_into_more_likely_neighbour():
    seq, ts, q = _timeline([W] * 5 + [BUS] + [C] * 5)
    q[5, C], q[5, W] = 0.3, 0.05               # the short bus minute looks more like car than walk
    lg = L.build(seq, ts, q, L.Rules(l_min=2))
    assert list(lg.top1) == [W, C] and lg.n_minutes.tolist() == [5, 6]
    seq, ts, q = _timeline([C] * 5 + [BUS] + [C] * 5)
    assert list(L.build(seq, ts, q, L.Rules(l_min=2)).top1) == [C]


def test_confidence_threshold_sets_unknown_keeps_top2():
    seq, ts, q = _timeline([C] * 4)
    q[:, C], q[:, BUS] = 0.5, 0.4
    lg = L.build(seq, ts, q, L.Rules(c=0.6))
    assert lg["mode"].tolist() == [L.UNKNOWN] and lg.top1.tolist() == [C] and lg.top2.tolist() == [BUS]


def test_perfect_prediction_scores():
    modes = [S] * 3 + [W] * 4 + [BUS] * 10 + [W] * 2
    seq, ts, q = _timeline(modes)
    lg = L.build(seq, ts, q)
    tl = L.truth_legs(seq, ts, np.array(modes))
    s = L.score(lg, tl, len(modes), q, np.zeros(len(modes), int)).metrics
    assert s["leg_f1"] == 1 and s["edits"] == 0 and s["time_acc"] == 1 and s["boundary_median_min"] == 0
    assert s["range_top1"] == 1 and s["wrong_unflagged_time"] == 0


def test_edits_counted():
    truth = np.array([W] * 5 + [BUS] * 10 + [W] * 5)
    pred = truth.copy()
    pred[5:8] = W                               # start of the bus leg 3 min late
    pred[15:17] = BUS                           # end 2 min late
    pred[10:12] = C                             # a car blip inside the bus leg
    seq, ts, q = _timeline(pred)
    lg = L.build(seq, ts, q)
    tl = L.truth_legs(seq, ts, truth)
    s = L.score(lg, tl, len(truth), q, np.zeros(len(truth), int)).metrics
    # predicted W(0-8) BUS(8-10) C(10-12) BUS(12-17) W(17-20): the bus truth leg maps to BUS(12-17); BUS(8-10) and
    # C(10-12) also map to it -> 2 merges; both bus boundaries off by > 1 min -> 2 moves; no mode change
    assert (s["merges"], s["splits"], s["mode_changes"], s["boundary_moves"]) == (2, 0, 0, 2)
    # the bus leg isn't matched at IoU 0.5 (5 of 10 minutes): leg F1 2/3 of the 3 truth legs and 5 predicted legs
    assert s["leg_recall"] == 2 / 3 and s["leg_precision"] == 2 / 5


def test_split_needed():
    truth = np.array([W] * 5 + [S] * 5)
    seq, ts, q = _timeline(np.array([W] * 10))
    s = L.score(L.build(seq, ts, q), L.truth_legs(seq, ts, truth), 10, q, np.zeros(10, int)).metrics
    assert (s["merges"], s["splits"], s["mode_changes"], s["boundary_moves"]) == (0, 1, 1, 0)
