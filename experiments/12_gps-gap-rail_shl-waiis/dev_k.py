"""Stop-point 2: R2's k per model on SHL User 1 dev (all positions), with the ExtraSensory guard; then the gap-setting
sensitivity at the chosen k. Models as saved (10's fold models and out-of-fold arrays; no training).

Minute probabilities, out of fold:
- SHL User 1, hips (all days) and hand / torso / bag (4phones days): saved acc probabilities x A0_gps fold models on
  C1 features from the 1 Hz track thinned to 20 s (11's approved input), product fusion, R2 boost, HMM per fold
  (other positions: one HMM sequence per position, as 10). Truth: SHL User 1's labelled minutes (fine -> top level);
  the other positions take the hips labels of the same minute.
- ExtraSensory dev: 10's saved raw fused probabilities (its per-minute GPS), R2 boost, HMM per fold.
Gaps (approved): good fixes (<= 500 m) of the 20 s SHL track per recording / ExtraSensory's per-minute fixes per
session; moving = length >= 5 min and speed across in [2, 25) m/s. Legs: 11's approved rules per model
(`tmd.suggest.SPECS`), via `tmd.legs.predict_legs` (rail hint as in 11).

Choice (approved), per model, SHL User 1 all positions pooled, seed 0: fewest edits per hour subject to (a)
ExtraSensory false rail time (mean of seeds 0-2) up by <= 0.02 pp, (b) SHL extra rail legs per hour up by <= 0.05,
(c) SHL false rail time on still, walking, car, bus each up by <= 0.5 pp; ties within 0.01 edits/h: smaller k; no
k > 1 better than k = 1: rule off.

    uv run python dev_k.py grid           # k in {1,2,4,8,16,32} x seeds 0-2 x E1, E2; log logs/dev_k.log
    uv run python dev_k.py sensitivity    # chosen k, gap settings g_min x v_lo x v_hi, seed 0; log logs/sensitivity.log
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from tmd import bench
from tmd import gps as tgps
from tmd import legs as L
from tmd import suggest as S
from tmd.features import gps as gf
from tmd.labels import FINE_CLASSES
from tmd.models import e10 as E
from tmd.models import fusion, minutes
from tmd.progress import Progress
from tmd.tracking import DataInput, run

HERE = Path(__file__).resolve().parent
EXP = HERE.name
TABLES = HERE / "results" / "tables"
log = logging.getLogger()
KS = (1, 2, 4, 8, 16, 32)
GAP = L.GapRule()                                     # approved: 5 min, [2, 25) m/s
A_PP, B_LEGS_H, C_PP, TIE = 0.0002, 0.05, 0.005, 0.01
FALSE_CLASSES = ("still", "walk", "car", "bus")
RAIL_FINE = {"train", "subway", "tram"}
NAMES = np.array(["none"] + FINE_CLASSES)


@dataclass
class Part:
    """One evaluation part, minutes sorted by (sequence, ts)."""
    name: str
    df: pd.DataFrame          # frame rows (frame order) used for smoothing
    order: np.ndarray
    seq: np.ndarray
    ts: np.ndarray
    fine: np.ndarray          # fine class name per sorted minute ("none" = unlabelled)
    day: np.ndarray
    covered: np.ndarray       # frame order
    gap_dt: np.ndarray        # frame order
    gap_v: np.ndarray


def make_part(name: str, df: pd.DataFrame, label_src: str, covered: np.ndarray, gap: pd.DataFrame) -> Part:
    lab = minutes.labelled_minutes(label_src, fine=True)[["user", "ts", "y8"]]
    y = df[["user", "ts"]].merge(lab, on=["user", "ts"], how="left", validate="m:1").y8.fillna(0).astype(int)
    seq = minutes.seq_ids(df)
    order = np.lexsort((df.ts.to_numpy(), seq))
    ud = df.user.astype(str) + "/" + pd.to_datetime(df.ts, unit="s").dt.date.astype(str)
    day = df.day.astype(str).where(df.day.astype(str) != "", ud).to_numpy() if "day" in df else ud.to_numpy()
    return Part(name, df, order, seq[order], df.ts.to_numpy(np.int64)[order], NAMES[y.to_numpy()][order],
                day[order], covered, gap.gap_dt.to_numpy(), gap.gap_v.to_numpy())


def load() -> dict:
    """Frames, GPS probabilities' inputs, gaps and HMM training minutes (everything but the seed-dependent parts)."""
    hips, other = E.frame("shl_user1"), E.frame("shl_user1_other", gps=False)
    es = E.frame("extrasensory")
    # SHL: C1 at 20 s for both frames (hips GPS for every position), gaps per recording
    tr = E.shl_thinned_track()
    c1 = tgps.nearest_per_minute(tr)
    x_hips = gf.features(hips[["user", "session", "ts"]], c1, E.GPS_VERSION)
    assert np.allclose(x_hips, np.load(E.PROCESSED / "11_leg-suggestions_waiis-app-days" / "shl_user1_gps_C1.npy"),
                       equal_nan=True)
    x_other = gf.features(other[["user", "session", "ts"]], c1, E.GPS_VERSION)
    g_shl = tgps.gaps(tr, ["rec"])
    rec = bench.read("shl_user1", "labels", columns=["user", "ts", "rec"])
    mg = {}
    for nm, df in (("hips", hips), ("other", other)):
        k = df[["user", "ts"]].merge(rec, on=["user", "ts"], how="left", validate="m:1")
        assert k.rec.notna().all()
        mg[nm] = tgps.minute_gaps(k, g_shl, ["rec"])
    other_seq = other.assign(session=other.session.astype(str) + "/" + other.position)   # one sequence per position
    # ExtraSensory: per-minute fixes <= 500 m, gaps per session
    fx = bench.read("extrasensory", "gps_per_minute", columns=["user", "session", "time", "lat", "lon", "accuracy"])
    g_es = tgps.gaps(fx[fx.accuracy <= E.COARSE_M], ["user", "session"])
    mg["es"] = tgps.minute_gaps(es[["user", "session", "ts"]], g_es, ["user", "session"])
    log.info("SHL gaps %d (moving %d), ExtraSensory gaps %d (moving %d)", len(g_shl),
             GAP.moving(g_shl.dt, g_shl.v_across).sum(), len(g_es), GAP.moving(g_es.dt, g_es.v_across).sum())
    parts = {"hips": make_part("hips", hips, "shl_user1", gf.covered(x_hips), mg["hips"]),
             "other": make_part("other", other_seq, "shl_user1", gf.covered(x_other), mg["other"]),
             "es": make_part("es", es, "extrasensory", es.covered.to_numpy(), mg["es"])}
    labs = {m: E.labelled(E.MODELS[m].hmm_sources, E.MODELS[m].n) for m in E.MODELS}
    return {"hips": hips, "other": other, "es": es, "x": {"hips": x_hips, "other": x_other}, "parts": parts,
            "labs": labs, "tracks": {"shl": g_shl, "es": g_es}}


def raw_probs(D: dict, mid: str, seed: int) -> dict:
    """Raw fused (before HMM) per part, frame order; checks the other positions and ExtraSensory reproduce 10."""
    out = {}
    gps_models = E.fold_models("A0_gps", seed)
    for nm in ("hips", "other"):
        df = D[nm]
        p_gps = E.predict_oof(gps_models, D["x"][nm], df.fold.to_numpy())
        out[nm] = fusion.product(E.saved(mid, seed, df, "acc"), p_gps, gf.covered(D["x"][nm]))
    out["es"] = E.saved(mid, seed, D["es"], "raw")
    return out


def smooth(D: dict, mid: str, part: Part, raw: np.ndarray) -> np.ndarray:
    return E.pad9(minutes.smooth_cv(part.df, raw, D["labs"][mid], "fb"))


def check_reproduce(D: dict, mid: str) -> None:
    """Seed 0: 10's saved p from 10's saved raw (other positions, per-position sequences; ExtraSensory)."""
    for nm in ("other", "es"):
        part = D["parts"][nm]
        src = D[nm]
        raw = E.saved(mid, 0, src, "raw")
        d = float(np.abs(smooth(D, mid, part, raw) - E.pad9(E.saved(mid, 0, src, "p"))).max())
        log.info("%s %s: HMM on 10's saved raw reproduces 10's saved p: max |diff| %.2e", mid, nm, d)
        assert d < 1e-6, (mid, nm, d)


def evaluate(parts: list[tuple[Part, np.ndarray, np.ndarray]], rules: L.Rules) -> dict:
    """Leg metrics over one or more parts pooled: (part, p9 frame order, in_gap frame order)."""
    seq, ts, p, cov, gap, fine, day = [], [], [], [], [], [], []
    off = 0
    for pt, p9, ig in parts:
        seq.append(pt.seq + off)
        off += pt.seq.max() + 1
        ts.append(pt.ts)
        p.append(p9[pt.order])
        cov.append(pt.covered[pt.order])
        gap.append(ig[pt.order])
        fine.append(pt.fine)
        day.append(pt.name + "/" + pt.day.astype(str))
    seq, ts, p, cov, gap = np.concatenate(seq), np.concatenate(ts), np.vstack(p), np.concatenate(cov), np.concatenate(gap)
    fine, day = np.concatenate(fine), np.concatenate(day)
    lg = L.predict_legs(seq, ts, p, cov, rules, gap)          # already sorted: order is the identity
    q = L.top_probs(p)
    truth = L.fine_to_top(_fine_id(fine))                     # unlabelled -> -1
    tl = L.truth_legs(seq, ts, truth)
    sc = L.score(lg, tl, len(q), q, day)
    m = {k: float(v) for k, v in sc.metrics.items() if np.isscalar(v)}
    rail = L.MODES.index("rail")
    pm = sc.per_mode.set_index("mode")
    m["rail_leg_recall"] = float(pm.leg_recall.get("rail", np.nan))
    m["rail_leg_precision"] = float(pm.leg_precision.get("rail", np.nan))
    m["rail_truth_legs"] = int(pm.truth_legs.get("rail", 0))
    # subway truth legs: rail truth legs whose labelled minutes are mostly subway
    lab = tl.attrs["lab_idx"]
    fl = fine[lab]
    sub_major = np.array([(fl[a:b] == "subway").mean() > 0.5 for a, b in zip(tl.i0, tl.i1)]) if len(tl) else \
        np.zeros(0, bool)
    is_sub = (tl["mode"].to_numpy() == rail) & sub_major
    matched_t = np.zeros(len(tl), bool)
    matched_t[sc.matched.truth.to_numpy()] = True
    t2p = dict(zip(sc.matched.truth, sc.matched.pred))
    m["subway_truth_legs"] = int(is_sub.sum())
    m["subway_leg_recall"] = float(matched_t[is_sub].mean()) if is_sub.any() else np.nan
    hints = [lg.hint.iat[t2p[i]] for i in np.flatnonzero(is_sub & matched_t)]
    m["subway_hint_metro"] = float(np.mean([h == "metro" for h in hints])) if hints else np.nan
    # predicted rail legs matched to a subway truth leg / predicted rail legs
    pr = lg["mode"].to_numpy() == rail
    m["pred_rail_legs"] = int(pr.sum())
    # extra rail legs: displayed rail, labelled minutes inside, no truth-rail minute inside
    pid = L.minute_leg(lg, len(q))
    tr_rail = np.zeros(len(lg), bool)
    has_lab = np.zeros(len(lg), bool)
    np.logical_or.at(has_lab, pid[lab], True)
    np.logical_or.at(tr_rail, pid[lab], truth[lab] == rail)
    m["extra_rail_legs_per_h"] = float((pr & has_lab & ~tr_rail).sum() / (len(lab) / 60))
    shown_rail = pr[pid]
    for c in FALSE_CLASSES:
        sel = fine == c
        m[f"false_rail_{c}"] = float(shown_rail[sel].mean()) if sel.any() else np.nan
    nonrail = (fine != "none") & ~np.isin(fine, list(RAIL_FINE))
    m["false_rail_nonrail"] = float(shown_rail[nonrail].mean())
    m["false_rail_min"] = int((shown_rail & nonrail).sum())
    m["in_gap_minutes"] = int(gap.sum())
    m["in_gap_subway_minutes"] = int((gap & (fine == "subway")).sum())
    m["legs_with_gap_reason"] = int(sum(any(f["code"] == "gps_gap_moving" for f in fr) for fr in lg.flag_reasons))
    return m


_FID = {c: i + 1 for i, c in enumerate(FINE_CLASSES)}


def _fine_id(fine: np.ndarray) -> np.ndarray:
    return np.array([_FID.get(c, 0) for c in fine])


def score_all(D: dict, mid: str, raws: dict, gap: L.GapRule) -> tuple[dict, dict]:
    """Metrics per part / position and pooled for one model, seed and gap rule."""
    rules = S.SPECS[mid].rules
    P = D["parts"]
    ig = {nm: gap.moving(P[nm].gap_dt, P[nm].gap_v) for nm in P}
    p = {nm: smooth(D, mid, P[nm], L.boost_rail(raws[nm], ig[nm], gap.k)) for nm in P}
    res = {"pooled": evaluate([(P["hips"], p["hips"], ig["hips"]), (P["other"], p["other"], ig["other"])], rules),
           "hips": evaluate([(P["hips"], p["hips"], ig["hips"])], rules), "es": evaluate([(P["es"], p["es"], ig["es"])], rules)}
    pos = D["other"].position.to_numpy()
    for ps in ("hand", "torso", "bag"):
        part = sub_part(P["other"], pos == ps, ps)
        res[ps] = evaluate([(part, p["other"][pos == ps], ig["other"][pos == ps])], rules)
    return res, p


def sub_part(pt: Part, mask: np.ndarray, name: str) -> Part:
    df = pt.df[mask].reset_index(drop=True)
    idx = np.flatnonzero(mask)
    inv = np.full(len(mask), -1)
    inv[idx] = np.arange(len(idx))
    keep = mask[pt.order]
    return Part(name, df, inv[pt.order[keep]], pt.seq[keep], pt.ts[keep], pt.fine[keep], pt.day[keep],
                pt.covered[mask], pt.gap_dt[mask], pt.gap_v[mask])


KEYS = ["edits_per_h", "time_acc", "leg_f1", "wrong_unflagged_time", "rail_leg_recall", "rail_leg_precision",
        "subway_truth_legs", "subway_leg_recall", "subway_hint_metro", "extra_rail_legs_per_h", "false_rail_still",
        "false_rail_walk", "false_rail_car", "false_rail_bus", "false_rail_nonrail", "false_rail_min",
        "in_gap_minutes", "in_gap_subway_minutes", "legs_with_gap_reason", "hours", "truth_legs", "pred_legs"]


def flat(res: dict) -> dict:
    return {f"{part}_{k}": v for part, m in res.items() for k, v in m.items() if k in KEYS}


def datasets(desc: str) -> list[DataInput]:
    kw = {"window_s": 15, "stride_s": 60, "oof_eval": True}
    return [DataInput("shl_user1_dev", "evaluation", "bench-v2 (9 fine classes) -> tool top level",
                      f"acc-v1-B + gps-v2 (C1, 20 s); {desc}", **kw),
            DataInput("extrasensory_dev_v2", "evaluation", "bench-v2 (9 fine classes) -> tool top level",
                      f"acc-v1-B + gps-v2 (per minute); {desc}", **kw)]


def params(mid: str, seed: int, gap: L.GapRule, step: str) -> dict:
    return {"model": mid, "model_version": f"10:{mid} fold models seed {seed}", "dataset":
            "shl_user1 (all positions) + extrasensory dev", "rate_hz": 25, "class_set": "top level", "seed": seed,
            "leg_rules": S.SPECS[mid].rules.name(), "gps_rule": "C1 (SHL 20 s); per minute (ExtraSensory)",
            "gap_g_min_s": gap.g_min_s, "gap_v_lo": gap.v_lo, "gap_v_hi": gap.v_hi, "k": gap.k,
            "boost_columns": "train, subway (not tram)", "gap_rule": gap.name(), "step": step}


def choose(t: pd.DataFrame) -> dict:
    """The approved choice on seed-0 SHL pooled metrics and the seed-mean ExtraSensory guard."""
    s0 = t[t.seed == 0].set_index("k")
    es = t.groupby("k").es_false_rail_nonrail.mean()
    base = s0.loc[1]
    ok = pd.Series(True, index=s0.index)
    ok &= es - es.loc[1] <= A_PP
    ok &= s0.pooled_extra_rail_legs_per_h - base.pooled_extra_rail_legs_per_h <= B_LEGS_H
    for c in FALSE_CLASSES:
        ok &= (s0[f"pooled_false_rail_{c}"] - base[f"pooled_false_rail_{c}"]).fillna(0) <= C_PP
    cand = s0[ok]
    best = cand.pooled_edits_per_h.min()
    k = int(cand[cand.pooled_edits_per_h <= best + TIE].index.min())
    better = (cand.pooled_edits_per_h < base.pooled_edits_per_h).drop(1, errors="ignore")
    if not better.any():
        k = 1
    return {"k": k, "feasible": [int(x) for x in cand.index], "best_edits_per_h": float(best),
            "base_edits_per_h": float(base.pooled_edits_per_h), "es_guard_delta_pp": float(100 * (es - es.loc[1])[k]),
            "rule_off": k == 1}


def grid() -> None:
    logging.basicConfig(filename=HERE / "logs" / "dev_k.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    D = load()
    prog = Progress(EXP, "dev_k", len(E.MODELS) * len(E.SEEDS) * len(KS), log)
    rows, choice = [], {}
    for mid in ("E1", "E2"):
        check_reproduce(D, mid)
        for seed in E.SEEDS:
            raws = raw_probs(D, mid, seed)
            for k in KS:
                gap = L.GapRule(GAP.g_min_s, GAP.v_lo, GAP.v_hi, float(k))
                t0 = time.perf_counter()
                res, _ = score_all(D, mid, raws, gap)
                met = flat(res)
                with run(EXP, params(mid, seed, gap, "dev_k"), run_name=f"dev_k_{mid}_k{k}_seed{seed}",
                         datasets=datasets(gap.name())):
                    mlflow.log_metrics({a: b for a, b in met.items() if b == b})
                rows.append({"model": mid, "seed": seed, "k": k, **met})
                log.info("%s seed %d k %d (%.0f s) %s", mid, seed, k, time.perf_counter() - t0,
                         json.dumps({a: round(b, 4) for a, b in met.items() if a.startswith(("pooled", "es"))}))
                prog.step(f"{mid} seed {seed} k {k}")
        t = pd.DataFrame([r for r in rows if r["model"] == mid])
        choice[mid] = choose(t)
        log.info("choice %s %s", mid, choice[mid])
    t = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "dev_k.csv", index=False)
    (TABLES / "dev_k_choice.json").write_text(json.dumps(choice, indent=1))
    show = ["model", "k", "pooled_edits_per_h", "pooled_time_acc", "pooled_subway_leg_recall",
            "pooled_rail_leg_precision", "pooled_extra_rail_legs_per_h", "pooled_false_rail_still",
            "pooled_false_rail_walk", "pooled_false_rail_car", "pooled_false_rail_bus", "es_false_rail_nonrail"]
    print(t[t.seed == 0][show].round(4).to_string(index=False))
    print(json.dumps(choice, indent=1))


def sensitivity() -> None:
    logging.basicConfig(filename=HERE / "logs" / "sensitivity.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    choice = json.loads((TABLES / "dev_k_choice.json").read_text())
    D = load()
    settings = [(g, lo, hi) for g in (3, 5, 10) for lo in (1, 2, 3) for hi in (15, 25)]
    prog = Progress(EXP, "sensitivity", len(E.MODELS) * len(settings), log)
    rows = []
    for mid in ("E1", "E2"):
        k = float(choice[mid]["k"]) if choice[mid]["k"] > 1 else float(max(KS))   # rule off: show the largest k
        raws = raw_probs(D, mid, 0)
        for g, lo, hi in settings:
            gap = L.GapRule(60.0 * g, float(lo), float(hi), k)
            res, _ = score_all(D, mid, raws, gap)
            met = flat(res)
            with run(EXP, params(mid, 0, gap, "sensitivity"), run_name=f"sens_{mid}_{gap.name()}",
                     datasets=datasets(gap.name())):
                mlflow.log_metrics({a: b for a, b in met.items() if b == b})
            rows.append({"model": mid, "k": k, "g_min": g, "v_lo": lo, "v_hi": hi, **met})
            prog.step(f"{mid} {gap.name()}")
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "sensitivity.csv", index=False)
    show = ["model", "k", "g_min", "v_lo", "v_hi", "pooled_edits_per_h", "pooled_subway_leg_recall",
            "pooled_extra_rail_legs_per_h", "pooled_false_rail_car", "pooled_false_rail_bus", "es_false_rail_nonrail"]
    print(t[show].round(4).to_string(index=False))


if __name__ == "__main__":
    {"grid": grid, "sensitivity": sensitivity}[sys.argv[1]]()
