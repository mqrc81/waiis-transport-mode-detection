"""Stop-point 2, step 3 (optional in the Plan): do the 20 s fixes improve leg boundaries? Leg rules only, never model
input. Dev: SHL User 1 at 20 s (rule C1), each model's chosen leg rules, seed 0.

Refinement: a boundary between a slow leg (still, walking) and a fast one (bike, car, bus, rail), legs adjacent in
one session, moves within +-60 s of the minute boundary to the midpoint of the first consecutive-fix pair (fixes
<= 60 s apart, pair midpoint within +-60 s) whose derived speed crosses halfway between the two legs' median pair
speeds (pairs within 5 min of the boundary, inside each leg); in time order, from the slow side to the fast side.
No crossing or too few pairs: the minute boundary stays.

Truth at 1 s: SHL's per-sample labels (`stream.parquet`, 25 Hz). A minute-level truth boundary between modes m1 -> m2
gets the precise time (end of the last m1 sample run + start of the next m2 run) / 2, searched within +-120 s.

Margin fixed at stop-point 1: adopted only if the median boundary error of matched legs falls by >= 0.5 min AND
edits per hour by >= 5%, both against the 1 s truth; (a) and (b) are minute-level and unchanged by construction.

    uv run python refine.py           # log: logs/refine.log; MLflow one run per model
"""
import json
import logging

import mlflow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import dev
from gps_candidates import thinned_track
from tmd import bench
from tmd import legs as L
from tmd.gps import haversine_m
from tmd.labels import CLASSES
from tmd.paths import PROCESSED
from tmd.tracking import DataInput, run

log = logging.getLogger()
SLOW = {L.MODES.index("still"), L.MODES.index("walking")}
FAST = {L.MODES.index(m) for m in ("bike", "car", "bus", "rail")}
MOVE_S, PAIR_DT_S, LEG_S, TRUTH_S = 60.0, 60.0, 300.0, 120.0
MARGIN_MIN, MARGIN_EDITS = 0.5, 0.05
STREAM = PROCESSED / "waiis_bench_v2" / "shl_user1" / "stream.parquet"
TABLES = dev.HERE / "results" / "tables"
SHL_TOP = L.fine_to_top(np.arange(len(CLASSES) + 1))      # SHL code 0..8 -> top id (-1 = null)


def stream_runs() -> pd.DataFrame:
    """Runs of one top-level label per recording from the 25 Hz labels: rec, label (top id, -1 null), start, end [s]."""
    f = pq.ParquetFile(STREAM)
    parts = []
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=["rec", "t_ms", "label"]).to_pandas()
        lab = SHL_TOP[t.label.to_numpy()]
        rec = t.rec.to_numpy()
        new = np.ones(len(t), bool)
        new[1:] = (lab[1:] != lab[:-1]) | (rec[1:] != rec[:-1]) | (np.diff(t.t_ms.to_numpy()) > 1000)
        idx = np.flatnonzero(new)
        end = np.append(idx[1:], len(t)) - 1
        parts.append(pd.DataFrame({"rec": rec[idx], "label": lab[idx], "start": t.t_ms.to_numpy()[idx] / 1e3,
                                   "end": t.t_ms.to_numpy()[end] / 1e3 + 0.04}))
    r = pd.concat(parts, ignore_index=True).sort_values(["rec", "start"]).reset_index(drop=True)
    # join runs split by row groups
    same = (r.rec.to_numpy()[1:] == r.rec.to_numpy()[:-1]) & (r.label.to_numpy()[1:] == r.label.to_numpy()[:-1]) & \
        (r.start.to_numpy()[1:] - r.end.to_numpy()[:-1] < 1.0)
    grp = np.concatenate([[0], np.cumsum(~same)])
    return r.groupby(grp).agg(rec=("rec", "first"), label=("label", "first"), start=("start", "min"),
                              end=("end", "max")).reset_index(drop=True)


def precise_truth(tl: pd.DataFrame, rec_of_seq: dict, runs: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Truth legs with internal boundaries moved to their 1 s time where found. Returns (legs, found, internal)."""
    lab_idx = tl.attrs["lab_idx"]
    tl = tl.copy()
    tl.attrs = {}
    tl = tl.astype({"start": float, "end": float})
    tl.attrs["lab_idx"] = lab_idx
    by_rec = {k: g.reset_index(drop=True) for k, g in runs[runs.label >= 0].groupby("rec")}
    a = np.flatnonzero(tl.end_internal.to_numpy())
    found = 0
    for i in a:
        b, m1, m2 = tl.end.iat[i], tl["mode"].iat[i], tl["mode"].iat[i + 1]
        g = by_rec.get(rec_of_seq.get(tl.seq.iat[i]))
        if g is None:
            continue
        k = np.flatnonzero((g.label.to_numpy()[:-1] == m1) & (g.label.to_numpy()[1:] == m2) &
                           (np.abs(g.end.to_numpy()[:-1] - b) <= TRUTH_S))
        if len(k):
            j = k[np.argmin(np.abs(g.end.to_numpy()[k] - b))]
            t = (g.end.iat[j] + g.start.iat[j + 1]) / 2
            tl.iat[i, tl.columns.get_loc("end")] = t
            tl.iat[i + 1, tl.columns.get_loc("start")] = t
            found += 1
    return tl, found, len(a)


def refine(lg: pd.DataFrame, pairs: dict) -> tuple[pd.DataFrame, int, int]:
    """Moves slow/fast boundaries of adjacent legs (see module doc). pairs: seq -> (t_mid, v) arrays."""
    lg = lg.copy()
    start, end = lg.start.to_numpy(float), lg.end.to_numpy(float)
    top, seq = lg.top1.to_numpy(), lg.seq.to_numpy()
    cand = moved = 0
    for i in range(len(lg) - 1):
        if seq[i] != seq[i + 1] or end[i] != start[i + 1]:
            continue
        a, b = top[i], top[i + 1]
        if not ((a in SLOW and b in FAST) or (a in FAST and b in SLOW)):
            continue
        cand += 1
        if seq[i] not in pairs:
            continue
        tm, v = pairs[seq[i]]
        bnd = end[i]
        va = v[(tm >= max(start[i], bnd - LEG_S)) & (tm < bnd)]
        vb = v[(tm >= bnd) & (tm < min(end[i + 1], bnd + LEG_S))]
        if len(va) < 2 or len(vb) < 2:
            continue
        thr = (np.median(va) + np.median(vb)) / 2
        near = (tm >= bnd - MOVE_S) & (tm <= bnd + MOVE_S)
        tt, vv = tm[near], v[near]
        hit = np.flatnonzero(vv >= thr) if a in SLOW else np.flatnonzero(vv < thr)
        if not len(hit):
            continue
        nb = float(np.clip(tt[hit[0]], bnd - MOVE_S, bnd + MOVE_S))
        if nb != bnd:
            end[i] = start[i + 1] = nb
            moved += 1
    lg["start"], lg["end"] = start, end
    return lg, cand, moved


def pair_speeds(tr: pd.DataFrame, seq_of: dict) -> dict:
    out = {}
    for (u, s), g in tr.sort_values(["user", "session", "time"]).groupby(["user", "session"]):
        t, la, lo = g.time.to_numpy(), g.lat.to_numpy(), g.lon.to_numpy()
        dt = np.diff(t)
        ok = (dt > 0) & (dt <= PAIR_DT_S)
        v = haversine_m(la[:-1], lo[:-1], la[1:], lo[1:]) / np.where(dt > 0, dt, 1)
        k = seq_of.get((u, s))
        if k is not None:
            out[k] = ((t[:-1] + t[1:])[ok] / 2, v[ok])
    return out


def main():
    logging.basicConfig(filename=dev.HERE / "logs" / "refine.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    choice = json.loads((TABLES / "leg_choice.json").read_text())
    df = dev.frame("shl_user1", gps=False)
    d = dev.leg_data(df)
    from tmd.models import minutes
    seq_full = minutes.seq_ids(df)
    seq_of = {(u, s): k for u, s, k in zip(df.user, df.session, seq_full)}
    rec = bench.read("shl_user1", "labels", columns=["user", "ts", "rec"])
    r = df[["user", "ts"]].merge(rec, on=["user", "ts"], how="left").rec.to_numpy()
    rec_of_seq = dict(zip(seq_full, r))
    tr = thinned_track(df)
    pairs = pair_speeds(tr, seq_of)
    runs = stream_runs()
    log.info("stream runs %d; sequences with 20 s pairs %d", len(runs), len(pairs))
    out = {}
    for name, ch in choice.items():
        if ch is None:
            continue
        c = ch["chosen"]
        rules = L.Rules(int(c["s_veh"]), int(c["s_other"]), int(c["g"]), int(c["l_min"]),
                        None if c["c"] is None or c["c"] != c["c"] else float(c["c"]))
        p = np.load(dev.OUT / f"shl_user1_{name}_seed0_{ch['gps_rule']}_p.npy").astype(np.float64)
        q = L.top_probs(p[d.order])
        lg = L.build(d.seq, d.ts, q, rules)
        tl = L.truth_legs(d.seq, d.ts, d.truth)
        tlp, found, internal = precise_truth(tl, rec_of_seq, runs)
        tlp.attrs["lab_idx"] = tl.attrs["lab_idx"]
        lr, cand, moved = refine(lg, pairs)
        res = {}
        for tag, legs_, truth_ in (("minute_vs_minute", lg, tl), ("minute", lg, tlp), ("refined", lr, tlp)):
            res[tag] = L.score(legs_, truth_, len(q), q, d.day).metrics
        dm = res["minute"]["boundary_median_min"] - res["refined"]["boundary_median_min"]
        de = 1 - res["refined"]["edits_per_h"] / res["minute"]["edits_per_h"]
        adopt = bool(dm >= MARGIN_MIN and de >= MARGIN_EDITS)
        m = {"truth_boundaries_internal": internal, "truth_boundaries_precise": found,
             "slow_fast_boundaries": cand, "moved": moved, "median_gain_min": dm, "edits_gain_share": de}
        for tag, v in res.items():
            for k in ("boundary_median_min", "boundary_p90_min", "edits_per_h", "boundary_moves", "leg_f1"):
                m[f"{tag}_{k}"] = v[k]
        out[name] = {**m, "adopted": adopt, "rules": rules.name()}
        log.info("%s %s", name, json.dumps(out[name], default=float))
        prm = {"model": name, "dataset": "shl_user1 (hips, day folds)", "rate_hz": 25, "class_set": "top level",
               "seed": 0, "gps_rule_shl": ch["gps_rule"], "leg_rules": rules.name(), "step": "refine",
               "margin": f"median boundary error -{MARGIN_MIN} min and edits/h -{MARGIN_EDITS:.0%}",
               "adopted": str(adopt)}
        ds = [DataInput("shl_user1_dev", "evaluation", "bench-v2 + 25 Hz stream labels (1 s truth)",
                        f"acc-v1-B + gps-v2 ({ch['gps_rule']}, 20 s)", window_s=15, stride_s=60, oof_eval=True)]
        with run(dev.EXP, prm, run_name=f"refine_{name}", datasets=ds):
            mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v})
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items()})
    (TABLES / "refine.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
