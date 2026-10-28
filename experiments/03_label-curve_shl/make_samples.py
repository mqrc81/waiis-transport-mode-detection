"""Sampled training windows per (seed, class) for every k <= 1000 (tmd.sampling), and the hips window manifest.

Usage: uv run python make_samples.py
Writes splits/samples_k.csv (seed, class_id, class, rank, day, rec, t_start_ms; rank < 1000; the (k, seed) set is
rank < k) and <data>/processed/shl_u1_hips_25hz_w5s_s2.5s_windows.parquet (every hips window: day, rec, t_start_ms,
y, pure, partition, trip). Prints days and trips covered per k.
"""
from pathlib import Path

import pandas as pd

from tmd.labels import CLASSES
from tmd.paths import PROCESSED
from tmd.sampling import ranks, trips

HERE = Path(__file__).resolve().parent
TABLE = PROCESSED / "shl_u1_25hz_w5s_s2.5s_feat-v1_all.parquet"
MANIFEST = PROCESSED / "shl_u1_hips_25hz_w5s_s2.5s_windows.parquet"
SEEDS = range(5)
KS = [1, 2, 5, 10, 50, 200, 1000]


def main():
    w = pd.read_parquet(TABLE, columns=["day", "rec", "position", "t_start_ms", "y", "pure"],
                        filters=[("position", "==", "Hips")]).drop(columns="position")
    part = pd.read_csv(HERE / "splits" / "shl_u1_days.csv", dtype={"day": str}).set_index("day").partition
    w["partition"] = w.day.map(part).fillna("none")
    pure = w[w.pure & (w.y > 0)]
    w["trip"] = trips(pure).reindex(w.index).fillna(-1).astype(int)
    w.sort_values(["rec", "t_start_ms"]).reset_index(drop=True).to_parquet(MANIFEST)

    pool = w[(w.partition == "training") & w.pure & (w.y > 0)]
    rows, cover = [], []
    for s in SEEDS:
        r = ranks(pool, s, list(range(1, 9)))
        p = pool.assign(rank=r)
        p = p[p["rank"] < max(KS)]
        rows.append(p.assign(seed=s))
        for k in KS:
            sk = p[p["rank"] < k]
            g = sk.groupby("y").agg(n=("day", "size"), days=("day", "nunique"), trips=("trip", "nunique"))
            cover += [dict(seed=s, k=k, cls=CLASSES[y - 1], **g.loc[y].to_dict()) for y in g.index]
    out = pd.concat(rows)
    out = out.assign(class_id=out.y, **{"class": out.y.map(lambda y: CLASSES[y - 1])})
    out = out[["seed", "class_id", "class", "rank", "day", "rec", "t_start_ms"]].sort_values(["seed", "class_id", "rank"])
    out.to_csv(HERE / "splits" / "samples_k.csv", index=False)
    cov = pd.DataFrame(cover)
    cov.to_csv(HERE / "results" / "samples_coverage.csv", index=False)
    print(f"samples_k.csv: {len(out)} rows; manifest: {len(w)} hips windows -> {MANIFEST.name}")
    print("short classes (n < k):", cov[cov.n < cov.k][["seed", "k", "cls", "n"]].values.tolist() or "none")
    print("days covered (seed 0):\n" + cov[cov.seed == 0].pivot(index="k", columns="cls", values="days")[CLASSES].to_string())
    print("trips covered (seed 0):\n" + cov[cov.seed == 0].pivot(index="k", columns="cls", values="trips")[CLASSES].to_string())


if __name__ == "__main__":
    main()
