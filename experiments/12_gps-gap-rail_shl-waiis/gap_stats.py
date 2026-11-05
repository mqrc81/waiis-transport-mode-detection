"""Stop-point 1: GPS gaps by class on SHL User 1 (dev) and ExtraSensory dev; false rail now (guard baseline).
Descriptive; no MLflow run (no model is fitted or scored here except the guard baseline from 10's saved arrays).

Track:
- SHL User 1: the 1 Hz hips track thinned to 20 s per recording, as 11 (C1's input); fixes with accuracy <= 500 m
  ("good"). SHL has no coarser fixes (max 400 m); accuracy per class is described on the 1 Hz track.
- ExtraSensory dev: `gps_per_minute` (at most one fix per recorded minute), accuracy <= 500 m; per session.
Gap: the time between two consecutive good fixes of a recording (SHL) / session (ExtraSensory) longer than
MIN_GAP_S. Speed across = great-circle distance last fix before -> first fix after / time between them.
Per labelled minute: in a gap if its centre (ts + 30) lies strictly inside one; the gap's length and speed across.
Per gap: its majority class (labelled minute centres inside it).

    uv run python gap_stats.py        # log: logs/gap_stats.log; tables: results/tables/gap_*.csv
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tmd import bench
from tmd import gps as tgps
from tmd.labels import FINE_CLASSES
from tmd.models import e10 as d11
from tmd.models import minutes

HERE = Path(__file__).resolve().parent

log = logging.getLogger()
TABLES = HERE / "results" / "tables"
CADENCE_S, COARSE_M = 20.0, 500.0
MIN_GAP_S = 60.0                     # > 3 missed 20 s fixes (SHL); ExtraSensory: >= 1 recorded minute without a fix
LEN_BINS = [60, 120, 300, 600, 1800, np.inf]
SPEED_BINS = [0, 0.5, 1, 2, 3, 5, 8, 15, 25, 40, np.inf]
NAMES = np.array(["none"] + FINE_CLASSES)


def gaps_of(track: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Gaps between consecutive good fixes per group: by..., t0, t1, dt, dist_m, v_across."""
    out = []
    for k, g in track.sort_values(by + ["time"]).groupby(by, sort=False):
        t, la, lo = g.time.to_numpy(), g.lat.to_numpy(), g.lon.to_numpy()
        dt = np.diff(t)
        i = np.flatnonzero(dt > MIN_GAP_S)
        if not len(i):
            continue
        dist = tgps.haversine_m(la[i], lo[i], la[i + 1], lo[i + 1])
        k = k if isinstance(k, tuple) else (k,)
        out.append(pd.DataFrame({**{b: v for b, v in zip(by, k)}, "t0": t[i], "t1": t[i + 1], "dt": dt[i],
                                 "dist_m": dist, "v_across": dist / dt[i]}))
    return pd.concat(out, ignore_index=True)


def assign(mins: pd.DataFrame, gaps: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per minute: gap id (-1 none), its dt and v_across."""
    mins = mins.copy()
    mins["gap"], mins["gap_dt"], mins["gap_v"] = -1, np.nan, np.nan
    c = mins.ts.to_numpy(float) + 30
    gi = gaps.reset_index(drop=True)
    for k, g in gi.groupby(by, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        sel = np.ones(len(mins), bool)
        for b, v in zip(by, k):
            sel &= mins[b].to_numpy() == v
        idx = np.flatnonzero(sel)
        t0, t1 = g.t0.to_numpy(), g.t1.to_numpy()
        j = np.searchsorted(t0, c[idx], side="right") - 1
        ok = (j >= 0) & (c[idx] < t1[np.maximum(j, 0)]) & (c[idx] > t0[np.maximum(j, 0)])
        rows = g.index.to_numpy()[j[ok]]
        mins.iloc[idx[ok], mins.columns.get_loc("gap")] = rows
        mins.iloc[idx[ok], mins.columns.get_loc("gap_dt")] = gi.dt.to_numpy()[rows]
        mins.iloc[idx[ok], mins.columns.get_loc("gap_v")] = gi.v_across.to_numpy()[rows]
    return mins


def per_class(mins: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per class: minutes, coverage, share of minutes in a gap >= 2 / 5 min, and in a gap >= 2 min with speed
    across >= 1, 2, 3, 5 m/s; speed-across quantiles of those minutes (gap >= 2 min)."""
    rows = []
    for c, g in mins.groupby("cls"):
        in2 = g.gap_dt >= 120
        r = {"set": name, "class": c, "minutes": len(g), "covered": g.has_fix.mean(),
             "in_gap_ge1min": (g.gap >= 0).mean(), "in_gap_ge2min": in2.mean(), "in_gap_ge5min": (g.gap_dt >= 300).mean()}
        for v in (1, 2, 3, 5):
            r[f"in_gap_ge2min_v_ge{v}"] = (in2 & (g.gap_v >= v)).mean()
        q = g.gap_v[in2]
        r.update({"v_across_p10": q.quantile(.1), "v_across_p50": q.median(), "v_across_p90": q.quantile(.9)})
        rows.append(r)
    return pd.DataFrame(rows)


def per_gap(gaps: pd.DataFrame, mins: pd.DataFrame, name: str) -> pd.DataFrame:
    m = mins[mins.gap >= 0]
    maj = m.groupby("gap").cls.agg(lambda s: s.value_counts().index[0])
    share = m.groupby("gap").cls.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    g = gaps.assign(cls=maj.reindex(gaps.index).fillna("unlabelled"), maj_share=share.reindex(gaps.index))
    g = g[g.dt >= 120]
    rows = []
    for c, x in g.groupby("cls"):
        rows.append({"set": name, "class": c, "gaps_ge2min": len(x), "gaps_ge5min": int((x.dt >= 300).sum()),
                     "gap_min_median": x.dt.median() / 60, "gap_min_p90": x.dt.quantile(.9) / 60,
                     "v_across_p10": x.v_across.quantile(.1), "v_across_p50": x.v_across.median(),
                     "v_across_p90": x.v_across.quantile(.9), "moving_ge2mps": int((x.v_across >= 2).sum()),
                     "majority_share_median": x.maj_share.median()})
    return pd.DataFrame(rows), g


def shl() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = bench.shl_gps_1hz("shl_user1").sort_values(["user", "rec", "time"]).reset_index(drop=True)
    keep = [g.index.to_numpy()[tgps.thin(g.time.to_numpy(), np.full(len(g), CADENCE_S))]
            for _, g in full.groupby(["user", "rec"], sort=False)]
    tr = full.loc[np.concatenate(keep)]
    tr = tr[tr.accuracy <= COARSE_M]
    lab = bench.read("shl_user1", "labels", columns=["user", "ts", "rec", "session"])
    lm = minutes.labelled_minutes("shl_user1", fine=True).merge(lab[["user", "ts", "rec"]], on=["user", "ts"])
    lm["cls"] = NAMES[lm.y8]
    lm["has_fix"] = lm.set_index(["rec", "ts"]).index.isin(tr.set_index(["rec", "ts"]).index)
    gaps = gaps_of(tr, ["rec"])
    lm = assign(lm, gaps, ["rec"])
    # accuracy of the 1 Hz fixes by class (minute label)
    acc = full.merge(lm[["rec", "ts", "cls"]], on=["rec", "ts"])
    a = acc.groupby("cls").accuracy.agg(p50="median", p90=lambda s: s.quantile(.9), max="max",
                                        share_gt100=lambda s: (s > 100).mean(), share_gt200=lambda s: (s > 200).mean())
    # 1 Hz fixes per labelled minute by class (does SHL log anything underground?)
    n1 = acc.groupby(["rec", "ts"]).size().rename("n1hz")
    lm = lm.merge(n1, left_on=["rec", "ts"], right_index=True, how="left").fillna({"n1hz": 0})
    a["minutes_with_any_1hz_fix"] = lm.groupby("cls").n1hz.agg(lambda s: (s > 0).mean())
    log.info("SHL User 1: 1 Hz %d fixes, thinned %d, labelled minutes %d, gaps > %g s: %d", len(full), len(tr),
             len(lm), MIN_GAP_S, len(gaps))
    return lm, gaps, a.reset_index()


def es() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    g = bench.read("extrasensory", "gps_per_minute")
    g = g[g.accuracy <= COARSE_M]
    lab = bench.read("extrasensory", "labels", columns=["user", "ts", "platform", "has_location",
                                                         "location_missing_reason"])
    lm = minutes.labelled_minutes("extrasensory", fine=True).merge(lab, on=["user", "ts"])
    lm["cls"] = NAMES[lm.y8]
    lm["has_fix"] = lm.set_index(["user", "ts"]).index.isin(g.set_index(["user", "ts"]).index)
    g = g[g.user.isin(lm.user.unique())]
    gaps = gaps_of(g, ["user", "session"])
    lm = assign(lm, gaps, ["user", "session"])
    raw = bench.read("extrasensory", "gps_per_minute", columns=["user", "ts", "accuracy"])
    raw = raw.merge(lm[["user", "ts", "cls", "platform"]], on=["user", "ts"])
    a = raw.groupby("cls").accuracy.agg(p50="median", p90=lambda s: s.quantile(.9), max="max",
                                        share_gt500=lambda s: (s > 500).mean())
    log.info("ExtraSensory dev: fixes %d (<= 500 m), labelled minutes %d, gaps: %d", len(g), len(lm), len(gaps))
    return lm, gaps, a.reset_index()


def guard_baseline() -> pd.DataFrame:
    """False rail now on ExtraSensory dev: minutes whose top-level argmax after the HMM (10's saved out-of-fold p,
    seeds 0-2) is rail, by truth class; and the same for SHL User 1 hips."""
    from tmd.legs import MODES, top_probs
    rows = []
    for src in ("extrasensory", "shl_user1"):
        df = d11.frame(src, gps=False)
        lm = minutes.labelled_minutes(src, fine=True)[["user", "ts", "y8"]]
        y = df[["user", "ts"]].merge(lm, on=["user", "ts"], how="left").y8.fillna(0).astype(int).to_numpy()
        cls = NAMES[y]
        for mid in ("E1", "E2"):
            for seed in d11.SEEDS:
                q = top_probs(d11.pad9(d11.saved(mid, seed, df, "p")))
                rail = q.argmax(1) == MODES.index("rail")
                lab = y > 0
                nonrail = lab & ~np.isin(cls, ["train", "subway", "tram"])
                r = {"source": src, "model": mid, "seed": seed, "labelled_min": int(lab.sum()),
                     "false_rail_min": int((rail & nonrail).sum()),
                     "false_rail_share_of_nonrail": (rail & nonrail).sum() / nonrail.sum()}
                for c in ("still", "walk", "car", "bus"):
                    r[f"false_rail_share_{c}"] = (rail & (cls == c)).sum() / max((cls == c).sum(), 1)
                rows.append(r)
    return pd.DataFrame(rows)


def r1_table(s_m: pd.DataFrame, e_m: pd.DataFrame) -> pd.DataFrame:
    """R1 read as GPS only (as if no minute had accelerometer data): minutes whose centre lies in a gap of at least
    g_min with speed across in [v_lo, v_hi), by class; rail / subway precision and subway recall (SHL User 1),
    non-rail minutes caught (ExtraSensory dev)."""
    rows = []
    for g_min in (3, 5, 10):
        for v_lo in (1, 2, 3):
            for v_hi in (15, 25):
                r = {"g_min": g_min, "v_lo": v_lo, "v_hi": v_hi}
                for name, m in (("shl", s_m), ("es", e_m)):
                    sel = (m.gap_dt >= 60 * g_min) & (m.gap_v >= v_lo) & (m.gap_v < v_hi)
                    c = m.cls[sel].value_counts()
                    n = int(sel.sum())
                    rail = int(c.reindex(["train", "subway", "tram"]).fillna(0).sum())
                    r[f"{name}_minutes"] = n
                    if name == "shl":
                        r["shl_rail_precision"] = rail / max(n, 1)
                        r["shl_subway_precision"] = c.get("subway", 0) / max(n, 1)
                        r["shl_subway_recall"] = c.get("subway", 0) / (m.cls == "subway").sum()
                        for k in ("bus", "car", "walk", "still"):
                            r[f"shl_{k}_min"] = int(c.get(k, 0))
                    else:
                        r["es_nonrail_share"] = (n - rail) / (~m.cls.isin(["train", "subway", "tram"])).sum()
                rows.append(r)
    return pd.DataFrame(rows)


def headroom(s_m: pd.DataFrame) -> pd.DataFrame:
    """What E1 / E2 (10's saved out-of-fold p, seed 0, top-level argmax after the HMM) say now on SHL User 1 minutes
    inside moving gaps (>= 3 min, 1 <= v < 15 m/s), by class and position."""
    from tmd.legs import MODES, top_probs
    rows = []
    for src in ("shl_user1", "shl_user1_other"):
        df = d11.frame(src, gps=False)
        pos = df.position.to_numpy() if "position" in df else np.full(len(df), "hips")
        m = df[["user", "ts"]].merge(s_m[["user", "ts", "cls", "gap_dt", "gap_v"]], on=["user", "ts"], how="left")
        mv_all = ((m.gap_dt >= 180) & (m.gap_v >= 1) & (m.gap_v < 15)).to_numpy()
        for mid in ("E1", "E2"):
            pred = np.array(MODES)[top_probs(d11.pad9(d11.saved(mid, 0, df, "p"))).argmax(1)]
            for p in np.unique(pos):
                for c in ("subway", "train", "bus", "car", "walk", "still"):
                    s = (m.cls == c).to_numpy() & (pos == p)
                    mv = s & mv_all
                    rows.append({"model": mid, "position": p, "class": c, "minutes": int(s.sum()),
                                 "rail_now": (pred[s] == "rail").mean() if s.any() else np.nan,
                                 "in_moving_gap": int(mv.sum()),
                                 "rail_now_in_gap": (pred[mv] == "rail").mean() if mv.any() else np.nan,
                                 "pred_in_gap": pd.Series(pred[mv]).value_counts().head(3).to_dict()})
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(filename=HERE / "logs" / "gap_stats.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    s_m, s_g, s_a = shl()
    e_m, e_g, e_a = es()
    pc = pd.concat([per_class(s_m, "shl_user1"), per_class(e_m, "extrasensory_dev")])
    pc.to_csv(TABLES / "gap_by_class.csv", index=False)
    (sg, sgg), (eg, _egg) = per_gap(s_g, s_m, "shl_user1"), per_gap(e_g, e_m, "extrasensory_dev")
    pg = pd.concat([sg, eg])
    pg.to_csv(TABLES / "gap_by_majority_class.csv", index=False)
    s_a.to_csv(TABLES / "gap_shl_accuracy_by_class.csv", index=False)
    e_a.to_csv(TABLES / "gap_es_accuracy_by_class.csv", index=False)
    # speed across x class (minutes in gaps >= 2 min)
    for name, m in (("shl_user1", s_m), ("extrasensory_dev", e_m)):
        x = m[m.gap_dt >= 120]
        h = pd.crosstab(x.cls, pd.cut(x.gap_v, SPEED_BINS, right=False))
        h.to_csv(TABLES / f"gap_speed_hist_{name}.csv")
        log.info("%s minutes in gaps >= 2 min by speed across:\n%s", name, h.to_string())
        print(f"\n{name}: minutes in gaps >= 2 min, by speed across [m/s]\n{h.to_string()}")
    # ExtraSensory: platform and missing reason of minutes without a fix
    pl = e_m.groupby(["platform"]).agg(minutes=("ts", "size"), covered=("has_fix", "mean"),
                                       in_gap_ge2min=("gap_dt", lambda s: (s >= 120).mean()))
    pl["in_gap_ge2min_v_ge3"] = e_m.assign(z=(e_m.gap_dt >= 120) & (e_m.gap_v >= 3)).groupby("platform").z.mean()
    pl.to_csv(TABLES / "gap_es_platform.csv")
    mr = e_m[~e_m.has_fix].location_missing_reason.value_counts()
    gb = guard_baseline()
    gb.to_csv(TABLES / "guard_baseline_false_rail.csv", index=False)
    print("\nper class (minutes):\n", pc.round(3).to_string(index=False))
    print("\nper gap >= 2 min (majority class):\n", pg.round(2).to_string(index=False))
    print("\nSHL 1 Hz accuracy by class:\n", s_a.round(3).to_string(index=False))
    print("\nES accuracy by class:\n", e_a.round(3).to_string(index=False))
    print("\nES by platform:\n", pl.round(3).to_string())
    print("\nES missing reason (labelled minutes without a good fix):\n", mr.to_string())
    print("\nguard baseline (minute argmax after HMM):\n",
          gb.groupby(["source", "model"]).mean(numeric_only=True).drop(columns="seed").round(4).to_string())
    r1 = r1_table(s_m, e_m)
    r1.to_csv(TABLES / "gap_r1_gps_only.csv", index=False)
    print("\nR1 read as GPS only:\n", r1.round(3).to_string(index=False))
    hr = headroom(s_m)
    hr.to_csv(TABLES / "gap_headroom_shl.csv", index=False)
    print("\nheadroom (seed 0):\n", hr[hr["class"].isin(["subway", "still", "car"])].round(3).to_string(index=False))
    # the long moving gaps of SHL road vehicles (tunnels?)
    rv = sgg[sgg.cls.isin(["car", "bus"]) & (sgg.v_across >= 2)]
    print("\nSHL car / bus gaps >= 2 min with speed across >= 2 m/s:\n",
          rv[["rec", "dt", "dist_m", "v_across", "cls", "maj_share"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
