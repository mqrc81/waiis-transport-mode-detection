"""Stop-point 1 dry run (README, Plan): counts for waiis_bench_v2 before anything is built.

    uv run python dry_run.py

Reads 08's window tables (<data>/processed/{nor_tmd,collecty}_windows_25hz_w15s/, built with 08's approved rules:
Collecty dedup and clock repair, NOR-TMD Android only) and the v1 benchmark (official examples). Writes
results/tables/dry_run_*.csv and the proposal files splits/proposal_*.csv (not yet the v2 split files).
Nothing here is the build; the numbers are what the plan predicts.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import bench, collecty
from tmd.paths import PROCESSED
from tmd.splits import balanced_folds, pick_test_groups

HERE = Path(__file__).resolve().parent
TABLES = HERE / "results" / "tables"
SEED = 0
FINE = ["still", "walk", "run", "bike", "car", "bus", "train", "subway", "tram"]
FINE_NOR = {"bicycle": "bike", "bus": "bus", "car": "car", "metro": "subway", "train": "train", "tram": "tram"}
FINE_COL = {"Run": "run", "Bike": "bike", "Car": "car", "Bus": "bus", "Train": "train", "Tramway": "tram"}
NOR_TEST_CLASSES = ["bus", "subway", "train", "tram"]
NOR_TEST_N = 12            # smallest size meeting >= 5 held-out users per test class (README, stop-point 1)
MIN_USER_WINDOWS = 5       # a user "has" a class with >= 5 windows of it
CAP_TOP_SHARE = 0.25       # rule R2: per (dataset, class) cap so the top user holds <= 25% ...
CAP_FLOOR = 60             # ... but never a cap below 60 windows (1 h)


def candidate(ds: str) -> pd.DataFrame:
    """Clean minute windows of a 08 window table with their fine class (9 classes)."""
    m = pd.read_parquet(PROCESSED / f"{ds}_windows_25hz_w15s" / "minutes.parquet")
    if ds == "nor_tmd":
        m = m[m.platform == "android"]
        m["fine"] = m.label.map(FINE_NOR).fillna("")
    else:
        d = collecty.duplicates()
        drop = {(str(u), str(k)) for u, k in d.loc[d["drop"], ["user", "traj"]].itertuples(index=False)}
        m = m[[(u, t.split(".")[0]) not in drop for u, t in zip(m.user.astype(str), m.trip.astype(str))]]
        m["fine"] = m.label.map(FINE_COL).fillna("")
    return m[m.clean & m.fine.ne("")].assign(dataset=ds)[["dataset", "user", "trip", "ts", "fine", "platform"]]


def cap_r2(counts: pd.Series) -> float:
    """Largest per-user cap with top-user share <= CAP_TOP_SHARE, floored at CAP_FLOOR; inf if it doesn't bind."""
    if counts.max() / counts.sum() <= CAP_TOP_SHARE:
        return np.inf
    for c in range(int(counts.max()), CAP_FLOOR - 1, -1):
        x = np.minimum(counts, c)
        if x.max() / x.sum() <= CAP_TOP_SHARE:
            return float(c)
    return float(CAP_FLOOR)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    w = pd.concat([candidate("nor_tmd"), candidate("collecty")], ignore_index=True)
    U = {ds: g.pivot_table(index="user", columns="fine", values="ts", aggfunc="size", fill_value=0)
         for ds, g in w.groupby("dataset")}

    # NOR-TMD test pick: >= 5 held-out users (>= 5 windows) per test class, from users without car or bike
    un = U["nor_tmd"]
    cand = un[(un.get("car", 0) == 0) & (un.get("bike", 0) == 0)].index
    valid = lambda sel: all((un.loc[sel, c] >= MIN_USER_WINDOWS).sum() >= 5 for c in NOR_TEST_CLASSES)
    test = pick_test_groups(un[NOR_TEST_CLASSES], NOR_TEST_N, valid, seed=SEED, n_iter=200000, candidates=cand)
    pd.DataFrame({"user": test}).to_csv(HERE / "splits" / "proposal_nor_tmd_test.csv", index=False)
    w["split"] = np.where(w.dataset.eq("nor_tmd") & w.user.isin(test), "test", "dev")

    # caps (R2) per (dataset, class), from dev users only; test users are never capped
    caps = []
    for (ds, c), g in w[w.split == "dev"].groupby(["dataset", "fine"]):
        s = g.groupby("user").size()
        cap = cap_r2(s)
        x = np.minimum(s, cap)
        caps.append({"dataset": ds, "class": c, "users": len(s), "windows": int(s.sum()),
                     "cap": "" if np.isinf(cap) else int(cap), "kept": int(x.sum()),
                     "top_share_before": round(s.max() / s.sum(), 2), "top_share_after": round(x.max() / x.sum(), 2),
                     "users_capped": int((s > cap).sum())})
    caps = pd.DataFrame(caps)
    caps.to_csv(TABLES / "dry_run_caps.csv", index=False)

    # folds: dev users of each new dataset over the same 5 folds, balancing users per class
    folds = []
    for ds in ("nor_tmd", "collecty"):
        dev = U[ds].drop(index=test, errors="ignore") if ds == "nor_tmd" else U[ds]
        has = (dev >= MIN_USER_WINDOWS).astype(int)
        has = has.loc[:, has.sum() > 0]
        f = balanced_folds(has, 5, seed=SEED, n_iter=20000)
        folds.append(pd.DataFrame({"dataset": ds, "user": f.index, "fold": f.to_numpy()}))
        per = has.groupby(f).sum()
        per.index.name = "fold"
        per.to_csv(TABLES / f"dry_run_folds_{ds}.csv")
    pd.concat(folds).to_csv(HERE / "splits" / "proposal_folds_new.csv", index=False)

    # count table: class x dataset x split (x platform), before / after caps
    capd = {(r["dataset"], r["class"]): (np.inf if r["cap"] == "" else r["cap"]) for _, r in caps.iterrows()}
    rows = []
    for (ds, sp, c), g in w.groupby(["dataset", "split", "fine"]):
        s = g.groupby("user").size()
        after = s if sp == "test" else np.minimum(s, capd.get((ds, c), np.inf))
        rows.append({"dataset": ds, "split": sp, "platform": "android", "class": c, "windows": len(g),
                     "after_caps": int(after.sum()), "users": len(s), "users_ge5": int((s >= 5).sum())})
    for src in ("extrasensory", "shl_user1", "shl_user23"):
        e = bench.examples(src)
        sp = e.split.replace({"rail_check": "test"})
        for (spl, plat, c), g in e.assign(split=sp).groupby(["split", "platform", "cls"]):
            s = g.groupby("user").size()
            rows.append({"dataset": src, "split": spl, "platform": "ios" if plat == "iphone" else plat, "class": c,
                         "windows": len(g), "after_caps": len(g), "users": len(s), "users_ge5": int((s >= 5).sum())})
    ct = pd.DataFrame(rows)
    ct.to_csv(TABLES / "dry_run_counts.csv", index=False)

    # ExtraSensory: windows a >= 25 Hz real-rate rule would remove from v1's official set (one window definition)
    acc = bench.read("extrasensory", "accelerometer", columns=["user", "ts", "raw_rate_hz"])
    off = bench.examples("extrasensory", ["platform", "cls"]).merge(acc, on=["user", "ts"])
    low = off[off.raw_rate_hz < 25]
    pd.DataFrame([{"official": len(off), "rate_below_25hz": len(low), "users": low.user.nunique(),
                   "by_class": low.cls.value_counts().to_dict(), "by_platform": low.platform.value_counts().to_dict(),
                   "raw_rate_p1": round(off.raw_rate_hz.quantile(.01), 1)}]).to_csv(TABLES / "dry_run_es_rate.csv",
                                                                                   index=False)
    piv = ct.pivot_table(index=["dataset", "split"], columns="class", values="after_caps", aggfunc="sum",
                         fill_value=0).reindex(columns=FINE, fill_value=0)
    print(piv.to_string())
    print(f"ES official windows below 25 Hz: {len(low)} of {len(off)}")


if __name__ == "__main__":
    main()
