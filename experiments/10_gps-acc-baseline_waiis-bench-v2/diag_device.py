"""Second descriptive check before the Conclusion is approved (dev only, no retraining, no decision change).

1. Is 4E98F91F's accelerometer abnormal? Its still windows against every other ExtraSensory Android dev user's, against
   BEF6C611 (excluded in 04, X8: |acc| level jumps between g, m/s2 and gravity-free scales) and against NOR-TMD's
   subway dev windows: real sampling rate, gravity level (median |acc|), noise floor (std of |acc|), quantisation step
   and timestamp jitter. Window statistics come from the benchmark tables (25 Hz, 15 s windows); quantisation, jitter
   and the raw noise floor from the raw recordings (up to RAW_N windows per user, quiet ones first).
   Also: 4E98F91F's quiet still minutes that E1 calls rail against its other quiet still minutes.
2. E1 vs A0 on ExtraSensory dev without 4E98F91F, seeds 0-2: six-class macro-F1 with the paired user-bootstrap CI,
   per-class F1 / precision / recall changes.

    uv run python diag_device.py        # results/tables/diag_device_*.csv; logs/diag_device.log
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import pipe
from tmd import bench
from tmd.evaluate import cm_metrics
from tmd.readers import extrasensory as es
from tmd.readers import nor_tmd
from tmd.resample import G, raw_window_stats as raw_stats

OUT = pipe.HERE / "results" / "tables"
DEV = "4E98F91F-4654-42EF-B908-A3389443F2E7"
BEF = "BEF6C611-50DA-4971-A040-87FB979F3FC1"
WIN_S = 15
RAW_N = 60
NOR_ACC = ["accelerometer", "CMAccelerometerData"]
log = logging.getLogger()


def pick(d: pd.DataFrame, rng) -> pd.DataFrame:
    """Up to RAW_N windows, quiet ones first."""
    q, nq = d[d.quiet], d[~d.quiet]
    a = q.sample(min(len(q), RAW_N), random_state=rng)
    return pd.concat([a, nq.sample(min(len(nq), RAW_N - len(a)), random_state=rng)])


def es_raw(user: str, w: pd.DataFrame, rng) -> pd.DataFrame:
    out = []
    for r in pick(w, rng).itertuples():
        a = es.read_snippet(es.snippet_path(user, int(r.ts)))
        if a is None or len(a) < 10:
            continue
        a = a[~np.isnan(a[:, 1:]).any(axis=1)]
        xyz = a[:, 1:] * (G if r.unit == "g" else 1.0)
        out.append({**raw_stats(a[:, 0], xyz), "quiet": r.quiet, "ts": r.ts})
    return pd.DataFrame(out)


def nor_raw(user: str, w: pd.DataFrame, rng) -> pd.DataFrame:
    d = pq.read_table(nor_tmd.OUT / "events" / f"user={user}", columns=["sensor", "t_ms", "x", "y", "z"],
                      filters=[("sensor", "in", NOR_ACC)]).to_pandas()
    ios = (d.sensor.astype(str) == "CMAccelerometerData").to_numpy()
    t = d.t_ms.to_numpy(np.int64)
    o = np.argsort(t, kind="stable")
    t, xyz = t[o] / 1e3, d[["x", "y", "z"]].to_numpy(float)[o] * np.where(ios[o], G, 1.0)[:, None]
    out = []
    for r in pick(w, rng).itertuples():
        i, j = np.searchsorted(t, [r.t_start_ms / 1e3, r.t_start_ms / 1e3 + WIN_S])
        if j - i >= 10:
            out.append({**raw_stats(t[i:j], xyz[i:j]), "quiet": r.quiet, "ts": r.ts})
    return pd.DataFrame(out)


def per_user(group: str, user: str, w: pd.DataFrame, raw: pd.DataFrame, platform: str = "", model: str = "") -> dict:
    q = w[w.quiet]
    rq = raw[raw.quiet] if len(raw) else raw
    return {"group": group, "user": user[:8], "platform": platform, "model": model, "windows": len(w),
            "quiet_share": w.quiet.mean(), "rate_hz": w.raw_rate_hz.median(),
            "acc_median": w.acc_median.median(), "acc_median_p5": w.acc_median.quantile(.05),
            "acc_median_p95": w.acc_median.quantile(.95), "implausible_share": w.acc_implausible.mean(),
            "g_unit_share": (w.unit == "g").mean() if "unit" in w else np.nan,
            "std_p10": w.acc_std.quantile(.1), "std_median": w.acc_std.median(),
            "std_quiet_median": q.acc_std.median() if len(q) else np.nan,
            "raw_n": len(raw), "raw_rate_hz": raw.raw_rate_hz.median() if len(raw) else np.nan,
            "dt_cv": raw.dt_cv.median() if len(raw) else np.nan,
            "quant_step": raw.quant_step.median() if len(raw) else np.nan,
            "distinct_share": raw.distinct_share.median() if len(raw) else np.nan,
            "raw_std_p10": raw.raw_std.quantile(.1) if len(raw) else np.nan,
            "raw_std_quiet_median": rq.raw_std.median() if len(rq) else np.nan}


def device_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(0)
    lab = bench.read("extrasensory", "labels", columns=["user", "ts", "platform", "split", "fold", "cls", "official",
                                                        "user_excluded"], version=pipe.BENCH)
    acc = bench.read("extrasensory", "accelerometer", columns=["user", "ts", "unit", "raw_rate_hz", "acc_median",
                                                               "acc_std", "acc_implausible", "quiet"], version=pipe.BENCH)
    e = lab.merge(acc, on=["user", "ts"], validate="1:1")
    e = e[(e.platform == "android") & (e.cls == "still") & (e.split != "test")]
    e = e[e.official | (e.user == BEF)]                 # BEF6C611 has no official windows: all its still windows
    rows, raws = [], []
    for u, w in e.groupby("user"):
        r = es_raw(u, w, rng)
        g = "4E98F91F" if u == DEV else "BEF6C611" if u == BEF else "ES Android, other"
        rows.append(per_user(g, u, w, r, "android"))
        raws.append(r.assign(user=u[:8], group=g))
    nl = bench.read("nor_tmd", "labels", columns=["user", "ts", "platform", "fold", "cls_fine", "official", "raw_rate_hz",
                                                  "acc_median", "acc_std", "acc_implausible", "quiet", "t_start_ms",
                                                  "model"], version=pipe.BENCH)
    nl = nl[nl.official & (nl.fold >= 0) & (nl.cls_fine == "subway")]
    for u, w in nl.groupby("user"):
        r = nor_raw(u, w, rng)
        rows.append(per_user("NOR-TMD subway", u, w, r, w.platform.iloc[0], w.model.mode().iloc[0]))
        raws.append(r.assign(user=u[:8], group="NOR-TMD subway"))
    return pd.DataFrame(rows), pd.concat(raws, ignore_index=True)


def summary(t: pd.DataFrame) -> pd.DataFrame:
    """Per group: 4E98F91F / BEF6C611 as they are; other groups as median [min, max] over users (>= 20 windows), and
    4E98F91F's rank among the ExtraSensory Android users."""
    cols = ["windows", "quiet_share", "rate_hz", "acc_median", "acc_median_p5", "acc_median_p95", "implausible_share",
            "g_unit_share", "std_p10", "std_median", "std_quiet_median", "raw_rate_hz", "dt_cv", "quant_step",
            "distinct_share", "raw_std_p10", "raw_std_quiet_median"]
    out = {}
    for g, d in t.groupby("group"):
        d = d[d.windows >= 20] if len(d) > 1 else d
        out[f"{g} (users {len(d)})"] = d[cols].median() if len(d) > 1 else d[cols].iloc[0]
        if len(d) > 1:
            out[f"{g} min"], out[f"{g} max"] = d[cols].min(), d[cols].max()
    s = pd.DataFrame(out)
    an = t[t.group.isin(["4E98F91F", "ES Android, other"]) & (t.windows >= 20)]
    s["4E98F91F rank among ES Android (1 = lowest)"] = [
        f"{int((an[c] < an.loc[an.group == '4E98F91F', c].iloc[0]).sum()) + 1} / {an[c].notna().sum()}" for c in cols]
    return s


def own_false_rail(raws: pd.DataFrame) -> pd.DataFrame:
    """4E98F91F's quiet still minutes: E1 (seed 0) rail vs not rail, window and raw statistics."""
    df = pipe.load()
    m = ((df.source == "extrasensory") & df.official & (df.user == DEV) & (df.y == 1)).to_numpy()
    f = pipe.load_probs("E1", 0, df)[m].argmax(1) + 1
    d = df.loc[m, ["ts", "quiet"]].assign(rail=np.isin(f, [7, 8, 9]))
    acc = bench.read("extrasensory", "accelerometer", columns=["user", "ts", "acc_median", "acc_std"],
                     version=pipe.BENCH)
    d = d.merge(acc[acc.user == DEV].drop(columns="user"), on="ts", how="left")
    d = d[d.quiet]
    rng = np.random.RandomState(1)
    rows = []
    for nm, g in d.groupby("rail"):
        r = es_raw(DEV, g.assign(unit="m/s2").sample(min(len(g), RAW_N), random_state=rng), rng)
        rows.append({"rail": nm, "minutes": len(g), "acc_median": g.acc_median.median(),
                     "acc_std_median": g.acc_std.median(), "raw_std_median": r.raw_std.median(),
                     "quant_step": r.quant_step.median(), "raw_rate_hz": r.raw_rate_hz.median(),
                     "dt_cv": r.dt_cv.median()})
    return pd.DataFrame(rows).set_index("rail")


def without_user() -> pd.DataFrame:
    df = pipe.load()
    sel = ((df.source == "extrasensory") & df.official).to_numpy()
    keep = sel & (df.user.to_numpy() != DEV)
    rows = []
    names = pipe.NAMES["six"]
    for seed in (0, 1, 2):
        pr = {r: pipe.preds(pipe.load_probs(r, seed, df))["six"] for r in ("A0", "E1")}
        y = pipe.truth(df.y.to_numpy())["six"]
        for part, mk in (("all users", sel), ("without 4E98F91F", keep)):
            m, tab = {}, {}
            pipe.block(m, tab, {}, "es", y[mk], pr["E1"][mk], "six", df.user.to_numpy()[mk],
                       {"A0": pr["A0"][mk]}, fig=False)
            row = {"seed": seed, "part": part, "E1": m["es_six_f1"], "A0": m["es_six_f1"] - m["es_six_f1_diff_A0"],
                   "diff": m["es_six_f1_diff_A0"], "diff_lo": m["es_six_f1_diff_A0_lo"],
                   "diff_hi": m["es_six_f1_diff_A0_hi"]}
            present = np.isin(np.arange(1, 7), np.unique(y[mk]))
            pts = {r: cm_metrics(tab["es_six_cm"] if r == "E1" else pipe._cm(y[mk], pr["A0"][mk], "six"), present)
                   for r in ("E1", "A0")}
            for i, c in enumerate(names):
                f1 = {}
                for r in ("E1", "A0"):
                    p, rc = pts[r]["precision"][i], pts[r]["recall"][i]
                    f1[r] = 2 * p * rc / (p + rc) if p + rc > 0 else 0.0
                row[f"f1_{c}"] = f1["E1"] - f1["A0"]
                row[f"precision_{c}"] = pts["E1"]["precision"][i] - pts["A0"]["precision"][i]
                row[f"recall_{c}"] = pts["E1"]["recall"][i] - pts["A0"]["recall"][i]
                if seed == 0 and f"es_six_precision_{c}_diff_A0_lo" in m:
                    row[f"precision_{c}_lo"] = m[f"es_six_precision_{c}_diff_A0_lo"]
                    row[f"precision_{c}_hi"] = m[f"es_six_precision_{c}_diff_A0_hi"]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(filename=pipe.HERE / "logs" / "diag_device.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    t, raws = device_table()
    t.round(4).to_csv(OUT / "diag_device_users.csv", index=False)
    s = summary(t)
    s.round(4).to_csv(OUT / "diag_device_summary.csv")
    print(s.round(4).to_string(), flush=True)
    log.info("\n%s\n%s", t.round(4).to_string(index=False), s.round(4).to_string())
    o = own_false_rail(raws)
    o.round(4).to_csv(OUT / "diag_device_4e98_rail.csv")
    print(o.round(4).to_string(), flush=True)
    log.info("\n%s", o.round(4).to_string())
    w = without_user()
    w.round(4).to_csv(OUT / "diag_device_without_4e98.csv", index=False)
    print(w.round(4).T.to_string(), flush=True)
    log.info("\n%s", w.round(4).T.to_string())


if __name__ == "__main__":
    main()
