"""Leg suggestions from minute probabilities (experiment 11): the labelling tool's top level, leg rules, and leg
metrics against truth legs.

Minutes are one row per model minute (`ts` = clock-minute start [s]), sorted by sequence (session) and time. A leg
covers [ts of its first minute, ts of its last minute + 60 s) and the minutes between; minutes missing inside a leg
are a gap the leg spans.

Top level (the tool's tree; e_scooter and motorcycle are never predicted): still, walking, running, bike, car, bus,
rail. Fine classes map to it (train / subway / tram -> rail); a top-level minute probability is the sum of its fine
probabilities. `UNKNOWN` (-1) is a leg below the confidence threshold.

Rules (11's Plan, applied in this order; `Rules`):
1. still runs of <= s_veh min between two legs of one vehicle mode (car, bus, rail) are absorbed;
2. still runs of <= s_other min between two legs of one of walking, running, bike are absorbed; legs of one mode
   separated by a data gap of <= g min join (and only legs <= g min apart are neighbours at all);
3. legs shorter than l_min min merge, shortest first, into the neighbour whose mode has the higher mean
   probability over the short leg's minutes (both neighbours of one mode: all three join);
4. confidence = mean over the leg's minutes of its mode's probability; below c -> UNKNOWN (top-2 still listed).
Raw HMM legs = `Rules()` defaults: runs of the minute argmax, g = 10, nothing else.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tmd.labels import FINE_CLASSES

MODES = ["still", "walking", "running", "bike", "car", "bus", "rail"]
TOOL_TOP = ["still", "walking", "running", "bike", "e_scooter", "motorcycle", "car", "bus", "rail", "unknown"]
FINE_TO_TOP = {"still": "still", "walk": "walking", "run": "running", "bike": "bike", "car": "car", "bus": "bus",
               "train": "rail", "subway": "rail", "tram": "rail"}
RAIL_HINT = {"train": "train", "subway": "metro", "tram": "tram"}
UNKNOWN = -1
STILL = MODES.index("still")
VEHICLE = np.array([MODES.index(m) for m in ("car", "bus", "rail")])
OTHER_MOVING = np.array([MODES.index(m) for m in ("walking", "running", "bike")])
STEP = 60
_LUT = np.array([-1] + [MODES.index(FINE_TO_TOP[c]) for c in FINE_CLASSES])   # fine id (1..9) -> top id


def fine_to_top(y: np.ndarray) -> np.ndarray:
    """Fine class ids (1..9; 0 = none) -> top-level ids (0..6; -1 = none)."""
    return _LUT[np.asarray(y)]


def top_probs(p: np.ndarray) -> np.ndarray:
    """(N, 8 or 9) fine probabilities (FINE_CLASSES order) -> (N, 7) top-level probabilities."""
    q = np.zeros((len(p), len(MODES)))
    for j in range(p.shape[1]):
        q[:, _LUT[j + 1]] += p[:, j]
    return q


@dataclass(frozen=True)
class Rules:
    s_veh: int = 0
    s_other: int = 0
    g: int = 10
    l_min: int = 1
    c: float | None = None

    def name(self) -> str:
        c = "off" if self.c is None else f"{self.c:g}"
        return f"sveh{self.s_veh}_soth{self.s_other}_g{self.g}_lmin{self.l_min}_c{c}"


RAW = Rules()


# ---------------------------------------------------------------- building legs

def runs(seq: np.ndarray, ts: np.ndarray, mode: np.ndarray, g: int) -> tuple[np.ndarray, np.ndarray]:
    """[i0, i1) index ranges of maximal runs of one mode within a sequence, where consecutive minutes are at most
    g missing minutes apart. Arrays sorted by (seq, ts)."""
    n = len(ts)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    brk = np.ones(n, bool)
    brk[1:] = (seq[1:] != seq[:-1]) | (mode[1:] != mode[:-1]) | (np.diff(ts) > (g + 1) * STEP)
    i0 = np.flatnonzero(brk)
    return i0, np.append(i0[1:], n)


class _Legs:
    """Mutable doubly linked legs of one evaluation set (all sequences)."""

    def __init__(self, seq, ts, q_cum, i0, i1, mode, g):
        self.seq, self.ts, self.q_cum, self.g = seq, ts, q_cum, g
        self.i0, self.i1, self.mode = list(i0), list(i1), list(mode)
        n = len(self.i0)
        self.prev = [k - 1 for k in range(n)]
        self.next = [k + 1 if k + 1 < n else -1 for k in range(n)]
        self.alive = [True] * n

    def start(self, k):
        return self.ts[self.i0[k]]

    def end(self, k):
        return self.ts[self.i1[k] - 1] + STEP

    def span(self, k):
        return (self.end(k) - self.start(k)) / STEP

    def neighbour(self, k, j):
        """j (prev or next of k) if it is a neighbour: same sequence and at most g missing minutes apart."""
        if j < 0:
            return -1
        a, b = (j, k) if j < k else (k, j)
        if self.seq[self.i0[a]] != self.seq[self.i0[b]]:
            return -1
        gap = (self.start(b) - self.end(a)) / STEP
        return j if gap <= self.g else -1

    def mean_q(self, k, i0=None, i1=None):
        i0 = self.i0[k] if i0 is None else i0
        i1 = self.i1[k] if i1 is None else i1
        return (self.q_cum[i1] - self.q_cum[i0]) / (i1 - i0)

    def merge(self, a, b, mode):
        """Merge b into a (a before b, linked), result keeps index a."""
        self.i1[a] = self.i1[b]
        self.mode[a] = mode
        self.alive[b] = False
        nb = self.next[b]
        self.next[a] = nb
        if nb >= 0:
            self.prev[nb] = a

    def live(self):
        return [k for k in range(len(self.i0)) if self.alive[k]]


def build(seq: np.ndarray, ts: np.ndarray, q: np.ndarray, rules: Rules = RAW) -> pd.DataFrame:
    """Legs of one evaluation set. seq, ts sorted by (seq, ts); q (N, 7) top-level minute probabilities.
    Returns one row per leg: seq, i0, i1 (minute index range), start, end [s], span_min, n_minutes, mode (id; -1
    unknown), top1, p1 (= confidence), top2, p2."""
    seq, ts = np.asarray(seq), np.asarray(ts, np.int64)
    q_cum = np.vstack([np.zeros((1, q.shape[1])), np.cumsum(q, 0)])
    i0, i1 = runs(seq, ts, q.argmax(1), rules.g)
    L = _Legs(seq, ts, q_cum, i0, i1, q.argmax(1)[i0], rules.g)
    _absorb_still(L, rules)
    if rules.l_min > 1:
        _min_length(L, rules.l_min)
    return _table(L, rules)


def _absorb_still(L: _Legs, r: Rules) -> None:
    if r.s_veh <= 0 and r.s_other <= 0:
        return
    k = 0
    while k >= 0:
        if L.alive[k] and L.mode[k] == STILL:
            a, b = L.neighbour(k, L.prev[k]), L.neighbour(k, L.next[k])
            if a >= 0 and b >= 0 and L.mode[a] == L.mode[b]:
                m = L.mode[a]
                lim = r.s_veh if m in VEHICLE else r.s_other if m in OTHER_MOVING else -1
                if L.span(k) <= lim:
                    L.merge(a, k, m)
                    L.merge(a, b, m)
                    k = a                         # the merged leg may now absorb the next still run
                    continue
        k = L.next[k]


def _min_length(L: _Legs, l_min: int) -> None:
    heap = [(L.span(k), k) for k in L.live() if L.span(k) < l_min]
    heapq.heapify(heap)
    while heap:
        s, k = heapq.heappop(heap)
        if not L.alive[k] or L.span(k) != s or s >= l_min:
            continue
        a, b = L.neighbour(k, L.prev[k]), L.neighbour(k, L.next[k])
        if a < 0 and b < 0:
            continue                               # isolated: stays, flagged short
        mq = L.mean_q(k)
        if a >= 0 and b >= 0 and L.mode[a] == L.mode[b]:
            m = L.mode[a]
            L.merge(a, k, m)
            L.merge(a, b, m)
            tgt = a
        else:
            if a >= 0 and (b < 0 or mq[L.mode[a]] >= mq[L.mode[b]]):
                L.merge(a, k, L.mode[a])
                tgt = a
            else:
                L.merge(k, b, L.mode[b])
                tgt = k
        if L.span(tgt) < l_min:
            heapq.heappush(heap, (L.span(tgt), tgt))


def _table(L: _Legs, r: Rules) -> pd.DataFrame:
    ks = L.live()
    i0 = np.array([L.i0[k] for k in ks], np.int64)
    i1 = np.array([L.i1[k] for k in ks], np.int64)
    mode = np.array([L.mode[k] for k in ks], np.int64)
    mq = (L.q_cum[i1] - L.q_cum[i0]) / (i1 - i0)[:, None]
    order = np.argsort(-mq, 1)
    conf = mq[np.arange(len(ks)), mode]
    # top-2: the leg's mode first (it can differ from the mean argmax after merges), then the best other mode
    other = np.where(order[:, 0] == mode, order[:, 1], order[:, 0])
    out = pd.DataFrame({"seq": L.seq[i0], "i0": i0, "i1": i1, "start": L.ts[i0], "end": L.ts[i1 - 1] + STEP,
                        "n_minutes": i1 - i0, "top1": mode, "p1": conf, "top2": other,
                        "p2": mq[np.arange(len(ks)), other]})
    out["span_min"] = (out.end - out.start) / STEP
    out["mode"] = mode if r.c is None else np.where(conf < r.c, UNKNOWN, mode)
    return out


def minute_leg(legs: pd.DataFrame, n: int) -> np.ndarray:
    """Leg row index of every minute (legs cover all minutes)."""
    return np.repeat(np.arange(len(legs)), (legs.i1 - legs.i0).to_numpy())[:n]


# ---------------------------------------------------------------- metrics

def truth_legs(seq: np.ndarray, ts: np.ndarray, truth: np.ndarray, g: int = RAW.g) -> pd.DataFrame:
    """Truth legs from labelled minutes (truth >= 0), with the raw rules (runs, joins over <= g missing minutes;
    unlabelled minutes count as missing). i0 / i1 index the labelled subset `lab_idx` (returned as attribute)."""
    lab = np.flatnonzero(truth >= 0)
    s, t, m = seq[lab], ts[lab], truth[lab]
    i0, i1 = runs(s, t, m, g)
    out = pd.DataFrame({"seq": s[i0], "i0": i0, "i1": i1, "start": t[i0], "end": t[i1 - 1] + STEP, "mode": m[i0]})
    out["span_min"] = (out.end - out.start) / STEP
    out["n_minutes"] = i1 - i0
    nxt_adj = np.append((out.seq.to_numpy()[1:] == out.seq.to_numpy()[:-1]) &
                        (out.start.to_numpy()[1:] == out.end.to_numpy()[:-1]), False)
    out["end_internal"] = nxt_adj                       # the next truth leg starts exactly here
    out["start_internal"] = np.insert(nxt_adj[:-1], 0, False)
    out.attrs["lab_idx"] = lab
    return out


@dataclass
class LegScore:
    metrics: dict
    per_mode: pd.DataFrame
    matched: pd.DataFrame


def score(legs: pd.DataFrame, tl: pd.DataFrame, n_minutes: int, q: np.ndarray, day: np.ndarray,
          iou_min: float = 0.5, tol_s: float = 60.0, short_min: float = 5.0) -> LegScore:
    """Leg metrics of predicted legs against truth legs (11's definitions). q: (N, 7) top-level minute
    probabilities (range mode); day: a day key per minute (edits per day)."""
    lab = tl.attrs["lab_idx"]
    nl = len(lab)
    hours = nl / 60.0
    pid = minute_leg(legs, n_minutes)[lab]                 # predicted leg of each labelled minute
    tid = np.repeat(np.arange(len(tl)), tl.n_minutes.to_numpy())
    tmode = tl["mode"].to_numpy()[tid]
    pmode, ptop1 = legs["mode"].to_numpy(), legs.top1.to_numpy()
    m = {"hours": hours, "days": len(np.unique(day[lab])), "truth_legs": len(tl)}

    # overlaps
    key = pid.astype(np.int64) * (len(tl) + 1) + tid
    uk, cnt = np.unique(key, return_counts=True)
    pp, tt = uk // (len(tl) + 1), uk % (len(tl) + 1)
    psize = np.bincount(pid, minlength=len(legs))
    tsize = tl.n_minutes.to_numpy()
    iou = cnt / (psize[pp] + tsize[tt] - cnt)
    has_lab = psize > 0
    m["pred_legs"] = int(has_lab.sum())

    def match(modes):
        ok = (modes[pp] == tl["mode"].to_numpy()[tt]) & (iou >= iou_min)
        order = np.argsort(-iou[ok], kind="stable")
        P, T = pp[ok][order], tt[ok][order]
        used_p, used_t, pairs = set(), set(), []
        for a, b in zip(P, T):
            if a not in used_p and b not in used_t:
                used_p.add(a)
                used_t.add(b)
                pairs.append((a, b))
        return np.array(pairs, np.int64).reshape(-1, 2)

    for tag, modes in (("", pmode), ("_top1", ptop1)):
        mt = match(modes)
        prec = len(mt) / max(m["pred_legs"], 1)
        rec = len(mt) / max(len(tl), 1)
        m[f"leg_precision{tag}"], m[f"leg_recall{tag}"] = prec, rec
        m[f"leg_f1{tag}"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        m[f"time_acc{tag}"] = float((modes[pid] == tmode).mean())
        if tag == "":
            pairs = mt
    m["extra_legs_per_h"] = (m["pred_legs"] - len(pairs)) / hours
    m["missed_legs_per_h"] = (len(tl) - len(pairs)) / hours

    # boundary errors of matched legs (truth ends that meet another truth leg)
    P, T = pairs[:, 0], pairs[:, 1]
    st = tl.iloc[T]
    es = np.abs(legs.start.to_numpy()[P] - st.start.to_numpy()) / STEP
    ee = np.abs(legs.end.to_numpy()[P] - st.end.to_numpy()) / STEP
    berr = np.concatenate([es[st.start_internal.to_numpy()], ee[st.end_internal.to_numpy()]])
    bmode = np.concatenate([st["mode"].to_numpy()[st.start_internal.to_numpy()],
                            st["mode"].to_numpy()[st.end_internal.to_numpy()]])
    m["boundary_n"] = len(berr)
    m["boundary_median_min"] = float(np.median(berr)) if len(berr) else np.nan
    m["boundary_p90_min"] = float(np.percentile(berr, 90)) if len(berr) else np.nan

    # edits: each predicted leg -> truth leg it overlaps most; each truth leg -> predicted leg it overlaps most
    # (ties: the one with the same mode first, then the earlier)
    same = (pmode[pp] == tl["mode"].to_numpy()[tt]).astype(int)
    o = np.lexsort((-same, -cnt, pp))
    p2t = pd.Series(tt[o]).groupby(pp[o]).first()
    o = np.lexsort((-same, -cnt, tt))
    t2p = pd.Series(pp[o]).groupby(tt[o]).first().reindex(np.arange(len(tl))).to_numpy()
    merges = int(np.maximum(np.bincount(p2t.to_numpy(), minlength=len(tl)) - 1, 0).sum())
    splits = int(np.maximum(np.bincount(t2p, minlength=len(legs)) - 1, 0).sum())
    mode_changes = int((pmode[t2p] != tl["mode"].to_numpy()).sum())
    a = np.flatnonzero(tl.end_internal.to_numpy())
    b = a + 1
    sep = t2p[a] != t2p[b]
    bnd = tl.end.to_numpy()[a][sep]
    err = np.minimum(np.abs(legs.end.to_numpy()[t2p[a][sep]] - bnd), np.abs(legs.start.to_numpy()[t2p[b][sep]] - bnd))
    moves = int((err > tol_s).sum())
    edits = merges + splits + mode_changes + moves
    m.update({"merges": merges, "splits": splits, "mode_changes": mode_changes, "boundary_moves": moves,
              "edits": edits, "edits_per_h": edits / hours, "edits_per_day": edits / m["days"]})

    # wrong but unflagged; flagged share
    known = pmode[pid] != UNKNOWN
    m["wrong_unflagged_time"] = float((known & (pmode[pid] != tmode)).mean())
    maj = pd.Series(tmode).groupby(pid).agg(lambda s: s.value_counts().index[0])
    lm = pmode[maj.index.to_numpy()]
    m["wrong_unflagged_legs"] = int(((lm != UNKNOWN) & (lm != maj.to_numpy())).sum())
    m["flagged_leg_share"] = float((pmode[has_lab] == UNKNOWN).mean())
    m["flagged_time_share"] = float((~known).mean())
    # second mode shown: share of legs whose top-1 is wrong where the truth is the second mode, by threshold
    wrong1 = ptop1[maj.index.to_numpy()] != maj.to_numpy()
    sec_ok = legs.top2.to_numpy()[maj.index.to_numpy()] == maj.to_numpy()
    p2 = legs.p2.to_numpy()[maj.index.to_numpy()]
    m["top1_wrong_legs"] = int(wrong1.sum())
    for tau in (0.1, 0.2, 0.3):
        m[f"second_shows_truth_tau{tau:g}"] = float((sec_ok & (p2 >= tau))[wrong1].mean()) if wrong1.any() else np.nan
        m[f"second_shown_share_tau{tau:g}"] = float((p2 >= tau)[has_lab[maj.index.to_numpy()]].mean())

    # short truth legs
    short = tl.span_min.to_numpy() < short_min
    matched_t = np.zeros(len(tl), bool)
    matched_t[T] = True
    m["short_truth_legs"] = int(short.sum())
    m["short_leg_recall"] = float(matched_t[short].mean()) if short.any() else np.nan

    # range mode: each truth leg as a selected range
    ql = q[lab]
    qm = pd.DataFrame(ql).groupby(tid).mean().to_numpy()
    tm = tl["mode"].to_numpy()
    rk = np.argsort(-qm, 1)
    m["range_top1"] = float((rk[:, 0] == tm).mean())
    m["range_top2"] = float(((rk[:, 0] == tm) | (rk[:, 1] == tm)).mean())
    conf = qm.max(1)
    hit = rk[:, 0] == tm
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(hit[bins == i].mean() - conf[bins == i].mean()) * (bins == i).mean() for i in range(10)
              if (bins == i).any())
    m["range_ece"] = float(ece)
    oh = np.zeros_like(qm)
    oh[np.arange(len(tm)), tm] = 1
    m["range_brier"] = float(((qm - oh) ** 2).sum(1).mean())

    # per mode
    rows = []
    for j, name in enumerate(MODES):
        tj = tm == j
        pj = (pmode == j) & has_lab
        if not tj.any() and not pj.any():
            continue
        rows.append({"mode": name, "truth_legs": int(tj.sum()), "pred_legs": int(pj.sum()),
                     "leg_recall": matched_t[tj].mean() if tj.any() else np.nan,
                     "leg_precision": np.isin(np.flatnonzero(pj), P).mean() if pj.any() else np.nan,
                     "short_truth_legs": int((tj & short).sum()),
                     "short_leg_recall": matched_t[tj & short].mean() if (tj & short).any() else np.nan,
                     "boundary_median_min": np.median(berr[bmode == j]) if (bmode == j).any() else np.nan,
                     "boundary_p90_min": np.percentile(berr[bmode == j], 90) if (bmode == j).any() else np.nan,
                     "time_recall": float((pmode[pid][tmode == j] == j).mean()) if (tmode == j).any() else np.nan,
                     "range_top1": float(hit[tj].mean()) if tj.any() else np.nan})
    matched = pd.DataFrame({"pred": P, "truth": T})
    return LegScore(m, pd.DataFrame(rows), matched)


# ---------------------------------------------------------------- suggestions (11, stop-point 3)

SHORT_MIN = 3.0                 # flag `short` below this span [min]
NO_GPS_SHARE = 0.5              # coverage "acc+gps" from this share of GPS-covered minutes; vehicle legs below: no_gps
HINT_UNCERTAIN = 0.6            # rail hint share below this: rail_hint_uncertain
GPS_ONLY_MIN_S = 300.0          # GPS-only stretches shorter than this get no leg
GPS_ONLY_STILL_MPS = 0.5
FINE_RAIL = [FINE_CLASSES.index(c) for c in ("train", "subway", "tram")]


def predict_legs(seq: np.ndarray, ts: np.ndarray, p: np.ndarray, covered: np.ndarray, rules: Rules) -> pd.DataFrame:
    """Leg suggestions from model minutes (any order). p: (N, 8 or 9) fine minute probabilities after the HMM;
    covered: GPS-covered minutes. Returns `build`'s table plus sub-mode hint, coverage and flag_reasons;
    times in seconds. Minutes in the returned `minutes` attribute order: sorted by (seq, ts)."""
    order = np.lexsort((ts, seq))
    seq, ts, p, covered = np.asarray(seq)[order], np.asarray(ts, np.int64)[order], p[order], covered[order]
    p9 = p if p.shape[1] == 9 else np.column_stack([p, np.zeros(len(p))])
    q = top_probs(p9)
    lg = build(seq, ts, q, rules)
    cov_cum = np.concatenate([[0], np.cumsum(covered)])
    rail_cum = np.vstack([np.zeros((1, 3)), np.cumsum(p9[:, FINE_RAIL], 0)])
    n = (lg.i1 - lg.i0).to_numpy()
    lg["gps_covered_share"] = (cov_cum[lg.i1] - cov_cum[lg.i0]) / n
    lg["coverage"] = np.where(lg.gps_covered_share >= NO_GPS_SHARE, "acc+gps", "acc")
    rail = (rail_cum[lg.i1] - rail_cum[lg.i0]) / n[:, None]
    tot = rail.sum(1)
    k = rail.argmax(1)
    is_rail = lg.top1.to_numpy() == MODES.index("rail")
    lg["hint"] = np.where(is_rail, np.array(["train", "metro", "tram"])[k], "")
    lg["hint_share"] = np.where(is_rail & (tot > 0), rail.max(1) / np.maximum(tot, 1e-12), np.nan)
    flags = []
    for r in lg.itertuples():
        f = []
        if r.mode == UNKNOWN:
            f.append({"code": "low_confidence", "confidence": round(float(r.p1), 3), "threshold": rules.c})
        if r.span_min < SHORT_MIN:
            f.append({"code": "short", "minutes": float(r.span_min)})
        if r.top1 in VEHICLE and r.gps_covered_share < NO_GPS_SHARE:
            f.append({"code": "no_gps", "gps_covered_share": round(float(r.gps_covered_share), 3)})
        if r.hint and r.hint_share < HINT_UNCERTAIN:
            f.append({"code": "rail_hint_uncertain", "share": round(float(r.hint_share), 3)})
        flags.append(f)
    lg["flag_reasons"] = flags
    lg["source"] = "model"
    lg.attrs["minutes"] = pd.DataFrame({"seq": seq, "ts": ts, "covered": covered, **{f"p_{m}": q[:, j] for j, m in
                                                                                   enumerate(MODES)},
                                        **{f"pf_{c}": p9[:, j] for j, c in enumerate(FINE_CLASSES)}})
    return lg


def gps_only_legs(legs: pd.DataFrame, fixes: pd.DataFrame, day_lo: float, day_hi: float) -> pd.DataFrame:
    """Legs for stretches without model minutes (11, approved 2026-11-04). fixes: t [s], lat, lon, accuracy already
    <= 500 m. A stretch = the time between consecutive model legs (and before the first / after the last, bounded by
    the day's first / last fix), at least 5 min long, with >= 2 fixes inside. Speed across (first -> last fix) < 0.5
    m/s: a still leg (gps_only_still); else an unknown leg with the speed across (gps_only_moving). No fixes, no leg."""
    from tmd.gps import haversine_m
    f = fixes[(fixes.t >= day_lo) & (fixes.t < day_hi)].sort_values("t")
    if len(f) < 2:
        return pd.DataFrame()
    t, la, lo = f.t.to_numpy(), f.lat.to_numpy(), f.lon.to_numpy()
    s = np.sort(legs.start.to_numpy(float)) if len(legs) else np.zeros(0)
    e = np.sort(legs.end.to_numpy(float)) if len(legs) else np.zeros(0)
    bounds = [(t[0], s[0] if len(s) else t[-1])] + list(zip(e[:-1], s[1:])) + ([(e[-1], t[-1])] if len(e) else [])
    rows = []
    for a, b in bounds:
        if b - a < GPS_ONLY_MIN_S:
            continue
        inside = np.flatnonzero((t >= a) & (t <= b))
        if len(inside) < 2:
            continue
        i, j = inside[0], inside[-1]
        v = float(haversine_m(la[i], lo[i], la[j], lo[j]) / max(t[j] - t[i], 1.0))
        still = v < GPS_ONLY_STILL_MPS
        rows.append({"start": a, "end": b, "span_min": (b - a) / STEP, "n_minutes": 0,
                     "top1": STILL if still else UNKNOWN, "mode": STILL if still else UNKNOWN,
                     "p1": np.nan, "top2": UNKNOWN, "p2": np.nan, "gps_covered_share": 1.0, "coverage": "gps",
                     "hint": "", "hint_share": np.nan, "n_fixes": len(inside),
                     "flag_reasons": [{"code": "gps_only_still" if still else "gps_only_moving",
                                "speed_across_mps": round(v, 2), "fixes": int(len(inside))}],
                     "source": "rule:gps_only_still" if still else "rule:gps_only_moving"})
    return pd.DataFrame(rows)
