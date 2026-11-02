"""06's final configuration as shared pieces (from experiment 06's pipe.py; 06 keeps its own copy): the dev / test
frame (05's acc features + 06's GPS tables), product fusion with acc fallback, and its latency.

Frame rows = 05's usable minutes (ExtraSensory + SHL), same order as 05's feature tables, with the 580 acc features
(05's B), the GPS features of the chosen version and `covered`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tmd import acc_features as af
from tmd import bench, minutes
from tmd import gps_features as gf
from tmd.evaluate import latency_ms, pipeline_latency
from tmd.paths import PROCESSED

P05 = PROCESSED / "05_acc-baseline_waiis-bench"
P06 = PROCESSED / "06_gps-acc-baseline_waiis-bench"
ACC = af.names(True, False)                     # 05's B: 580 features
KEYS = ["user", "ts", "session", "official", "y", "platform", "fold", "quiet"]
GPS_COLS = ("ts", "time", "pair_v", "pair_dist", "brg", "lat", "lon")


def load_frame(part: str = "dev", gps_version: str = "gps-v2") -> pd.DataFrame:
    """dev: ExtraSensory dev + SHL User 1; test: ExtraSensory test users + SHL Users 2-3."""
    srcs = ["extrasensory", "shl_user1"] if part == "dev" else ["extrasensory", "shl_user23"]
    gps = gf.names(gps_version)
    d = []
    for s in srcs:
        a = pd.read_parquet(P05 / f"{s}_{part}_25hz_w15s_acc-v1_lp-none.parquet", columns=KEYS + ACC)
        g = pd.read_parquet(P06 / f"{s}_{part}_{gps_version}_w2min.parquet", columns=["user", "ts"] + gps)
        assert (a.user.to_numpy() == g.user.to_numpy()).all() and (a.ts.to_numpy() == g.ts.to_numpy()).all()
        d.append(pd.concat([a, g[gps]], axis=1).assign(source=s))
    df = pd.concat(d, ignore_index=True)
    df["y"], df["fold"] = df.y.astype(int), df.fold.astype(int)
    df["covered"] = gf.covered(df[gps].to_numpy(np.float32))
    return df


def product(p_acc, p_gps, cov):
    """06's H1: p ∝ p_acc · p_gps on GPS-covered rows; p_acc elsewhere."""
    out = p_acc.astype(np.float64).copy()
    q = np.maximum(p_acc[cov], 1e-9) * np.maximum(p_gps[cov], 1e-9)
    out[cov] = q / q.sum(1, keepdims=True)
    return out


def session_pool(df: pd.DataFrame, seed: int = 0, n: int = 200, version: str = bench.VERSION) -> list:
    """n fold-0 ExtraSensory sessions with >= 3 usable minutes: (keys, windows, GPS fixes) per session. `version`:
    the benchmark (07 pins v1)."""
    es = df[(df.source == "extrasensory") & (df.fold == 0)]
    cnt = es.groupby(["user", "session"]).size()
    pick = cnt[cnt >= 3].sample(min(n, int((cnt >= 3).sum())), random_state=seed).index
    keep = es.set_index(["user", "session"]).loc[pick].reset_index()[["user", "ts", "session"]]
    keys, w = bench.load_windows("extrasensory", keep, version)
    keys = keys.assign(source="extrasensory")
    gps = bench.read("extrasensory", "gps_per_minute", version=version)
    gps = gps[gps.user.isin(set(keep.user))]
    return [(keys.iloc[idx], w[idx], gps[(gps.user == u) & (gps.session == s)])
            for (u, s), idx in keys.groupby(["user", "session"], sort=False).indices.items()]


def latency(pool: list, acc_model, gps_model, a: np.ndarray, pi: np.ndarray, gps_version: str = "gps-v2") -> dict:
    """End to end, single thread (set the env vars before importing numpy), after warm-up, median and p95 [ms]:
    one window (acc + GPS features, both models, product) and per window over whole sessions (+ HMM fb)."""
    k0, w0, g0 = next((k, w, g) for k, w, g in pool if len(g) >= 3)
    t0 = int(k0.ts.iloc[len(k0) // 2])

    def one():
        fa = af.features(w0[:1], True, False)
        g = gf.prepare(g0)
        fg = gf.window_features(t0, *[g[c].to_numpy() for c in GPS_COLS], version=gps_version)[None]
        fg = fg.astype(np.float32)
        return product(acc_model.predict(fa, num_threads=1), gps_model.predict(fg, num_threads=1), gf.covered(fg))

    def session(s):
        keys, w, gps = s
        fa, fg = af.features(w, True, False), gf.features(keys, gps, gps_version)
        q = product(acc_model.predict(fa, num_threads=1), gps_model.predict(fg, num_threads=1), gf.covered(fg))
        return minutes.smooth(keys, q, a, pi, "fb")

    out = {}
    out["lat_inference_ms_p50"], out["lat_inference_ms_p95"] = latency_ms(one)
    out.update(pipeline_latency(session, pool, [len(s[0]) for s in pool]))
    return out
