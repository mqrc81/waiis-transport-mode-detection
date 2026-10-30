"""Final test of 06 (stop-point 3, scored once): H1 = 05's B acc model × gps-v2 GPS model (product on GPS-covered
minutes, acc fallback), then HMM forward-backward; seeds 0-2. A (05's final configuration) is scored alongside on the
same minutes, for the paired gain.

Per seed:
- acc model = 05's test model `test_B_seed<k>.txt` (all ExtraSensory dev + SHL User 1, rounds = median of 05's CV
  counts): exactly the final configuration's acc model, reused, not retrained;
- GPS model = gps-v2 LightGBM (lr 0.02, balanced weights) on all covered official dev minutes (ExtraSensory dev + SHL
  User 1), rounds = median of G_v2 seed k's 5 early-stopping counts;
- HMM transitions and initial distribution from all labelled dev minutes.
One MLflow run per seed: test_seed<k>.

    uv run python make_gps.py --part test --version gps-v2
    uv run python test.py
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import argparse
import json
import logging

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from tmd import acc_features as af
from tmd import bench, lgbm, minutes
from tmd import gps_features as gf
from tmd.evaluate import (
    bootstrap_users_diff,
    bootstrap_users_recall,
    class_list,
    family_proba,
    latency_ms,
    pipeline_latency,
    plot_cm,
    scores,
)
from tmd.labels import CLASSES, to_family
from tmd.progress import Progress
from tmd.tracking import TRACKING_URI, DataInput, run

import pipe

SEEDS = (0, 1, 2)
GPS_V = "gps-v2"
MODELS = pipe.OUT / "models"
log = logging.getLogger()


def preds(p):
    return {"class": p.argmax(1) + 1, "family": family_proba(p).argmax(1) + 1}


def block(m, figs, tab, pre, y8, users, p, ci, cov=None, base=None):
    """F1 and recall (user-bootstrap CIs if ci), covered / uncovered split, paired diff vs base (if given)."""
    for lvl, pr in preds(p).items():
        yy = y8 if lvl == "class" else to_family(y8)
        s = scores(yy, pr, lvl)
        m[f"{pre}_{lvl}_f1"] = s["macro_f1"]
        tab[f"{pre}_{lvl}_classes"] = class_list(yy, lvl)
        tab[f"cm_{pre}_{lvl}"] = s["cm"].tolist()
        for c, r in s["recall"].items():
            if r == r:
                m[f"{pre}_{lvl}_recall_{c}"] = r
        if ci:
            st = {}
            for c, (lo, hi) in bootstrap_users_recall(yy, pr, users, lvl, stats=st).items():
                key = f"{pre}_{lvl}_f1" if c == "macro_f1" else f"{pre}_{lvl}_recall_{c}"
                m[f"{key}_lo"], m[f"{key}_hi"] = lo, hi
            m[f"{pre}_{lvl}_ci_redraw_rate"] = st["redraw_rate"]
        if cov is not None:
            for part, mk in (("covered", cov), ("uncovered", ~cov)):
                if mk.any():
                    m[f"{pre}_{lvl}_f1_{part}"] = scores(yy[mk], pr[mk], lvl)["macro_f1"]
                    tab[f"{pre}_{lvl}_classes_{part}"] = class_list(yy[mk], lvl)
        if base is not None:
            bp = preds(base)[lvl]
            if ci:
                d = bootstrap_users_diff(yy, pr, bp, users, lvl)
                m[f"{pre}_{lvl}_gain_vs_A"], m[f"{pre}_{lvl}_gain_vs_A_lo"], m[f"{pre}_{lvl}_gain_vs_A_hi"] = \
                    d["diff"], d["lo"], d["hi"]
            else:
                m[f"{pre}_{lvl}_gain_vs_A"] = s["macro_f1"] - scores(yy, bp, lvl)["macro_f1"]
        if not pre.endswith(("_raw", "_A")):
            figs[f"cm_{pre}_{lvl}.png"] = plot_cm(s["cm"], lvl, f"{pre} {lvl}")


def coverage(te):
    d = te[te.official].copy()
    d["cls"] = d.y.map(lambda i: CLASSES[i - 1])
    d["group"] = np.where(d.source == "extrasensory", "ES " + d.platform.astype(str), "SHL Users 2-3")
    t = d.pivot_table(index="group", columns="cls", values="covered", aggfunc="mean")
    t = t[[c for c in CLASSES if c in t.columns]]
    t["all"] = d.groupby("group").covered.mean()
    return t


def test_pool(te, seed, n=200):
    """200 ExtraSensory test sessions with >= 3 usable minutes: windows and GPS fixes (latency only)."""
    es = te[te.source == "extrasensory"]
    cnt = es.groupby(["user", "session"]).size()
    pick = cnt[cnt >= 3].sample(min(n, int((cnt >= 3).sum())), random_state=seed).index
    keep = es.set_index(["user", "session"]).loc[pick].reset_index()[["user", "ts", "session"]]
    keys, w = bench.load_windows("extrasensory", keep)
    keys = keys.assign(source="extrasensory")
    gps = bench.read("extrasensory", "gps_per_minute")
    gps = gps[gps.user.isin(set(keep.user))]
    return [(keys.iloc[idx], w[idx], gps[(gps.user == u) & (gps.session == s)])
            for (u, s), idx in keys.groupby(["user", "session"], sort=False).indices.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="score again although test runs exist")
    a = ap.parse_args()
    logging.basicConfig(filename=pipe.HERE / "logs" / "test.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    mlflow.set_tracking_uri(TRACKING_URI)
    done = mlflow.search_runs(experiment_names=[pipe.EXP], filter_string="attributes.run_name LIKE 'test_seed%'")
    if len(done) and not a.force:
        raise SystemExit(f"test already scored ({len(done)} runs); the test is scored once")
    pipe.use_gps(GPS_V)
    dev, te = pipe.load("dev"), pipe.load("test")
    hmm_ap = minutes.hmm_params(pipe.labelled())
    cov_dev, cov_te = dev.covered.to_numpy(), te.covered.to_numpy()
    tr = dev.official.to_numpy() & cov_dev & (dev.fold.to_numpy() >= 0)
    xg_tr, y_tr = dev.loc[tr, pipe.GPS].to_numpy(np.float32), dev.y.to_numpy()[tr]
    xa_te, xg_te = te[pipe.ACC].to_numpy(np.float32), te[pipe.GPS].to_numpy(np.float32)
    es = ((te.source == "extrasensory") & te.official).to_numpy()
    shl = ((te.source == "shl_user23") & te.official).to_numpy()
    mb_keys = bench.read("extrasensory", "labels", columns=["user", "ts", "reason"])
    mb_keys = mb_keys[mb_keys.reason == "check:MOTORBIKE"]
    mb = te.set_index(["user", "ts"]).index.isin(mb_keys.set_index(["user", "ts"]).index) & \
        (te.source == "extrasensory").to_numpy()
    cov_tab = coverage(te)
    cov_tab.to_csv(pipe.HERE / "results" / "gps_coverage_test.csv")
    pool = None
    summary = {}
    for seed in SEEDS:
        prog = Progress(pipe.EXP, f"test_seed{seed}", 4, log)
        acc = lgb.Booster(model_file=str(pipe.P05 / "models" / f"test_B_seed{seed}.txt"))
        g_rounds = int(np.median([f["rounds"] for f in json.loads((MODELS / f"G_v2_seed{seed}_folds.json").read_text())]))
        prm, _, _ = pipe.params(0.02)
        gm = lgb.train({**prm, "seed": seed}, lgbm.dataset(xg_tr, y_tr, np.ones(len(y_tr))), g_rounds)
        gm.save_model(str(MODELS / f"test_G_v2_seed{seed}.txt"))
        prog.step("GPS model")
        pa_raw, pg = acc.predict(xa_te), gm.predict(xg_te)
        raw = pipe.product(pa_raw, pg, cov_te)
        p = minutes.smooth(te, raw, *hmm_ap, "fb")
        p_a = minutes.smooth(te, pa_raw, *hmm_ap, "fb")
        prog.step("predict, fuse, HMM")
        m, figs, tab = {}, {}, {}
        for scope, sel in (("all", es), ("iphone", es & (te.platform == "iphone").to_numpy()),
                           ("android", es & (te.platform == "android").to_numpy())):
            d = te[sel]
            y8, users, cv = d.y.to_numpy(), d.user.to_numpy(), cov_te[sel]
            block(m, figs, tab, f"test_es_{scope}", y8, users, p[sel], True, cv, p_a[sel])
            block(m, figs, tab, f"test_es_{scope}_raw", y8, users, raw[sel], True, cv)
            block(m, figs, tab, f"test_es_{scope}_A", y8, users, p_a[sel], True, cv)
        d = te[shl]
        block(m, figs, tab, "test_shl23", d.y.to_numpy(), d.user.to_numpy(), p[shl], False, cov_te[shl], p_a[shl])
        block(m, figs, tab, "test_shl23_raw", d.y.to_numpy(), d.user.to_numpy(), raw[shl], False, cov_te[shl])
        block(m, figs, tab, "test_shl23_A", d.y.to_numpy(), d.user.to_numpy(), p_a[shl], False, cov_te[shl])
        tab["motorbike"] = {"n": int(mb.sum()), "users": sorted({u[:8] for u in te.user[mb]}),
                            "covered": int(cov_te[mb].sum()),
                            "pred_after_hmm": dict(zip(CLASSES, np.bincount(p[mb].argmax(1), minlength=8).tolist())),
                            "pred_before_hmm": dict(zip(CLASSES, np.bincount(raw[mb].argmax(1), minlength=8).tolist())),
                            "pred_A": dict(zip(CLASSES, np.bincount(p_a[mb].argmax(1), minlength=8).tolist()))}
        tab["gps_coverage_test"] = json.loads(cov_tab.round(3).to_json())
        # latency: features (acc + gps-v2) + both models + product + (pipeline) HMM
        pool = pool or test_pool(te, 0)
        k0, w0, g0 = next((k, w, g) for k, w, g in pool if len(g) >= 3)
        t0 = int(k0.ts.iloc[len(k0) // 2])

        def one():
            fa = af.features(w0[:1], True, False)
            gg = gf.prepare(g0)
            fg = gf.window_features(t0, *[gg[c].to_numpy() for c in ("ts", "time", "pair_v", "pair_dist", "brg",
                                                                     "lat", "lon")], version=GPS_V)[None]
            return pipe.product(acc.predict(fa, num_threads=1), gm.predict(fg.astype(np.float32), num_threads=1),
                                gf.covered(fg))

        def sess(s):
            keys, w, gps = s
            fa, fg = af.features(w, True, False), gf.features(keys, gps, GPS_V)
            q = pipe.product(acc.predict(fa, num_threads=1), gm.predict(fg, num_threads=1), gf.covered(fg))
            return minutes.smooth(keys, q, *hmm_ap, "fb")

        m["lat_inference_ms_p50"], m["lat_inference_ms_p95"] = latency_ms(one)
        m.update(pipeline_latency(sess, pool, [len(s[0]) for s in pool]))
        m.update({"acc_model_bytes": len(acc.model_to_string()), "gps_model_bytes": len(gm.model_to_string()),
                  "acc_rounds": acc.num_trees() / 8, "gps_rounds": g_rounds})
        prog.step("latency")
        prm_ = {"model": "lightgbm-acc(05 B) x lightgbm-gps(gps-v2) product + hmm-fb", "row": "test", "seed": seed,
                "dataset": "es_dev+shl_u1", "rate_hz": 25, "class_set": "8class+family", "primary": "es 8-class macro-F1",
                "features": f"acc-v1_whole_sub5s+{GPS_V}_w2min", "gps_features": GPS_V, "fusion": "product, acc fallback",
                "acc_model": f"05 test_B_seed{seed}", "gps_rounds_rule": "median of G_v2 seed's 5 CV counts",
                "hmm_transitions": "all labelled dev minutes, before caps and drops", "smoothing": "fb",
                "bootstrap": "users, redraw resamples lacking a class", "deviation": "platform check overridden (README)",
                "test_es_iphone_class_classes": tab["test_es_iphone_class_classes"], "window_s": 15}
        kw = {"label_rules": "bench-v1", "features": prm_["features"], "rate_hz": 25, "window_s": 15, "stride_s": 60}
        ds = [DataInput("extrasensory_dev_v2", "training", **kw), DataInput("shl_user1_dev", "training", **kw),
              DataInput("extrasensory_test_v2", "evaluation", **kw), DataInput("shl_user23_test_v2", "evaluation", **kw)]
        with run(pipe.EXP, prm_, run_name=f"test_seed{seed}", datasets=ds):
            mlflow.log_metrics({k: float(v) for k, v in m.items() if v == v})
            for fn, fig in figs.items():
                mlflow.log_figure(fig, fn)
                plt.close(fig)
            mlflow.log_dict(pipe.dump(tab), "tables.json")
        prog.step("MLflow")
        log.info("seed %d %s %s", seed, json.dumps(m), json.dumps(tab["motorbike"]))
        summary[seed] = m
        print(seed, {k: round(m[k], 4) for k in ("test_es_all_class_f1", "test_es_all_class_gain_vs_A",
                                                 "test_es_all_family_f1", "test_es_iphone_class_f1",
                                                 "test_es_android_class_f1", "test_shl23_class_f1")}, flush=True)
    for k in ("test_es_all_class_f1", "test_es_all_class_gain_vs_A", "test_es_all_family_f1", "test_shl23_class_f1"):
        print(k, "mean", round(float(np.mean([summary[s][k] for s in SEEDS])), 4))
    print(cov_tab.round(2).to_string())


if __name__ == "__main__":
    main()
