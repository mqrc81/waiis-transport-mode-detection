"""Step 2: 25 Hz windows (5 s, stride 2.5 s) and features v1 for the dev partitions only.

Usage: uv run python make_features.py [--workers 6] [--only shl|es]
Writes <data>/processed/{shl_u1_25hz_w5s_s2.5s_feat-v1, es_dev_25hz_w5s_s2.5s_feat-v1}.parquet (one row per window:
metadata + features) and results/features_counts.json. Test days/users (splits/) are never read.
"""
import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import extrasensory as es
from tmd import shl
from tmd.features import FEATURES_VERSION, NAMES, window_features
from tmd.paths import PROCESSED
from tmd.resample import DOWN, decimate_100_to_25, es_snippet_to_25hz
from tmd.splits import es_users, shl_days
from tmd.windows import WIN, cut, majority, starts

HERE = Path(__file__).resolve().parent
TAG = f"25hz_w5s_s2.5s_feat-{FEATURES_VERSION}"
OUT_SHL = PROCESSED / f"shl_u1_{TAG}.parquet"
OUT_ES = PROCESSED / f"es_dev_{TAG}.parquet"


def shl_jobs() -> list[tuple[shl.Recording, str, int]]:
    days = shl_days()
    fold = days[(days.user == "User1") & (days.partition == "dev")].set_index("day").fold.to_dict()
    four = shl.recordings("User1_4phones")
    four_days = {r.day for r in four}
    recs = [r for r in shl.recordings("User1") if r.day not in four_days] + four   # 02 audit: duplicates
    return [(r, p, fold[r.day]) for r in recs for p in r.positions]


def shl_one(rec: shl.Recording, pos: str, fold: int) -> tuple[pd.DataFrame, dict]:
    t, x = shl.read_motion(rec, pos, ("acc", "gyro"))
    _, y = shl.read_labels(rec)
    ok = ~np.isnan(x).any(1)
    d = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    parts, n_short = [], 0
    for lo, hi in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if hi - lo < WIN * DOWN:
            n_short += 1
            continue
        x25, y25, t25 = decimate_100_to_25(x[lo:hi]), y[lo:hi:DOWN], t[lo:hi:DOWN]
        s = starts(len(x25))
        maj, pure = majority(y25, s)
        keep = maj != 0
        s, maj, pure = s[keep], maj[keep], pure[keep]
        if len(s) == 0:
            continue
        f = window_features(cut(x25, s))
        meta = pd.DataFrame({"dataset": "shl", "user": rec.user, "day": rec.day, "rec": rec.rec, "position": pos,
                             "fold": fold, "t_start_ms": t25[s], "y": maj, "pure": pure})
        parts.append(pd.concat([meta, pd.DataFrame(f, columns=NAMES)], axis=1))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return df, {"stretches_under_5s": n_short}


def es_one(uuid: str, fold: int, platform: str) -> tuple[pd.DataFrame, dict]:
    m = es.minute_classes(es.read_cleaned_labels(uuid), es.read_original_labels(uuid))
    m = m[(m.reason == "ok") | m.reason.str.startswith("check:")]
    parts, cnt = [], Counter()
    for ts, row in m.iterrows():
        acc = es.read_snippet(es.snippet_path(uuid, int(ts), "acc"))
        gyro = es.read_snippet(es.snippet_path(uuid, int(ts), "gyro"))
        if acc is None or gyro is None:
            cnt["missing_acc_or_gyro"] += 1
            continue
        x25, why = es_snippet_to_25hz(acc, gyro)
        cnt[why] += 1
        if x25 is None:
            continue
        s = starts(len(x25))
        if len(s) == 0:
            cnt["no_window"] += 1
            continue
        f = window_features(cut(x25, s))
        meta = pd.DataFrame({"dataset": "es", "user": uuid, "ts": int(ts), "win": np.arange(len(s)),
                             "platform": platform, "position": row.position or "none",
                             "label_source": row.label_source, "fold": fold, "y": np.int8(row.y),
                             "reason": row.reason, "pure": True})
        parts.append(pd.concat([meta, pd.DataFrame(f, columns=NAMES)], axis=1))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return df, dict(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", choices=["shl", "es"])
    a = ap.parse_args()
    logging.basicConfig(filename=HERE / "logs" / "make_features.log", filemode="a", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    log = logging.getLogger()
    counts_path = HERE / "results" / "features_counts.json"
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}

    if a.only in (None, "shl"):
        jobs = shl_jobs()
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(shl_one, *zip(*jobs)))
        for (r, p, _), (df, c) in zip(jobs, res):
            log.info("shl %s %s %s: %d windows %s", r.session, r.rec, p, len(df), c)
        df = pd.concat([d for d, _ in res], ignore_index=True)
        df.to_parquet(OUT_SHL)
        counts["shl"] = {"recording_positions": len(jobs), "windows": len(df), "pure": int(df.pure.sum()),
                         "stretches_under_5s": sum(c["stretches_under_5s"] for _, c in res),
                         "windows_per_position": df.position.value_counts().to_dict()}
        print(f"SHL: {len(df)} windows ({df.pure.mean():.1%} pure) -> {OUT_SHL.name}")

    if a.only in (None, "es"):
        u = es_users()
        dev = u[u.partition == "dev"]
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(es_one, dev.uuid, dev.fold, dev.platform))
        tot = Counter()
        for uid, (df, c) in zip(dev.uuid, res):
            log.info("es %s: %d windows %s", uid, len(df), c)
            tot.update(c)
        df = pd.concat([d for d, _ in res], ignore_index=True)
        df.to_parquet(OUT_ES)
        counts["es_dev"] = {"users": len(dev), "windows": len(df), "minutes_by_outcome": dict(tot),
                            "windows_by_reason": df.reason.value_counts().to_dict()}
        print(f"ES dev: {len(df)} windows from {df.groupby(['user', 'ts']).ngroups} minutes; snippets {dict(tot)}")

    counts_path.write_text(json.dumps(counts, indent=1, default=int))


if __name__ == "__main__":
    main()
