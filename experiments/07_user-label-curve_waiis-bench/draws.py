"""Experiment 07: the user draws (splits/draws.json) and their class coverage per n (dry run, no training).

For outer dev fold k, the pool is the ExtraSensory dev users of the other 4 folds. Draw d (seeds 0-4) is one random
ordering of each fold's pool; n takes its first n users (nested). n = all is the whole pool, one run.

    uv run python draws.py            # writes splits/draws.json (refuses to change an existing one) and
                                      # results/coverage_dry_run.csv; prints the summary per n
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tmd.features import gps as gf
from tmd.labels import CLASSES
from tmd.paths import PROCESSED

HERE = Path(__file__).resolve().parent
P05 = PROCESSED / "05_acc-baseline_waiis-bench"
P06 = PROCESSED / "06_gps-acc-baseline_waiis-bench"
DRAWS_FILE = HERE / "splits" / "draws.json"
K, DRAWS = 5, (0, 1, 2, 3, 4)
NS = (1, 2, 3, 5, 10, 20)               # + "all"
ES_EXTRA = 5                            # early-stopping check: users n+1 .. n+5 of the ordering
GPS_V = "gps-v2"


def es_minutes() -> pd.DataFrame:
    """Official ExtraSensory dev minutes (the training rows), with platform, fold and GPS coverage."""
    a = pd.read_parquet(P05 / "extrasensory_dev_25hz_w15s_acc-v1_lp-none.parquet",
                        columns=["user", "ts", "official", "y", "platform", "fold"])
    g = pd.read_parquet(P06 / f"extrasensory_dev_{GPS_V}_w2min.parquet", columns=["user", "ts"] + gf.names(GPS_V))
    assert (a.user.to_numpy() == g.user.to_numpy()).all() and (a.ts.to_numpy() == g.ts.to_numpy()).all()
    a["covered"] = gf.covered(g[gf.names(GPS_V)].to_numpy(np.float32))
    return a[a.official & (a.fold >= 0)].reset_index(drop=True)


def make_draws(users_fold: pd.Series) -> dict:
    """{draw: {fold: ordered pool}}; the ordering of draw d, fold k uses rng seed (d, k)."""
    out = {}
    for d in DRAWS:
        out[str(d)] = {}
        for k in range(K):
            pool = sorted(users_fold[users_fold != k].index)
            out[str(d)][str(k)] = [pool[i] for i in np.random.default_rng([d, k]).permutation(len(pool))]
    return out


def load_draws() -> dict:
    return json.loads(DRAWS_FILE.read_text())["draws"]


def train_users(draws: dict, d: int, k: int, n) -> list[str]:
    o = draws[str(d)][str(k)]
    return o if n == "all" else o[:n]


def main():
    argparse.ArgumentParser().parse_args()
    es = es_minutes()
    users = es.groupby("user").agg(fold=("fold", "first"), platform=("platform", "first"))
    if DRAWS_FILE.exists():
        draws = load_draws()
        assert draws == make_draws(users.fold), "splits/draws.json differs from the draw rule"
    else:
        DRAWS_FILE.parent.mkdir(exist_ok=True)
        DRAWS_FILE.write_text(json.dumps({
            "rule": "pool(k) = ExtraSensory dev users (official minutes) with fold != k, sorted; draw d orders it "
                    "with numpy default_rng([d, k]).permutation; n takes the first n",
            "fold_file": "experiments/04_frozen-benchmark_es-shl/splits/es_users_v2.csv",
            "draws": make_draws(users.fold)}, indent=1))
        draws = load_draws()
    mins = es.pivot_table(index="user", columns="y", values="ts", aggfunc="size", fill_value=0)
    mins = mins.reindex(columns=range(1, 9), fill_value=0)
    mins.columns = CLASSES
    covm = es[es.covered].pivot_table(index="user", columns="y", values="ts", aggfunc="size", fill_value=0)
    covm = covm.reindex(index=mins.index, columns=range(1, 9), fill_value=0)
    covm.columns = CLASSES
    rows = []
    for n in (*NS, "all"):
        for d in (DRAWS if n != "all" else (0,)):
            for k in range(K):
                u = train_users(draws, d, k, n)
                m, c = mins.loc[u].sum(), covm.loc[u].sum()
                rows.append({"n": str(n), "draw": d, "fold": k, "n_users": len(u),
                             "iphone_share": float((users.loc[u].platform == "iphone").mean()),
                             **{f"min_{c_}": int(m[c_]) for c_ in CLASSES},
                             **{f"gps_{c_}": int(c[c_]) for c_ in CLASSES},
                             **{f"users_{c_}": int((mins.loc[u, c_] > 0).sum()) for c_ in CLASSES}})
    t = pd.DataFrame(rows)
    (HERE / "results").mkdir(exist_ok=True)
    t.to_csv(HERE / "results" / "coverage_dry_run.csv", index=False)
    cls7 = [c for c in CLASSES if c != "subway"]
    print("users per fold (ES dev):", users.fold.value_counts().sort_index().to_dict(),
          "pool sizes:", [len(draws["0"][str(k)]) for k in range(K)])
    print("\nES official minutes per class over the n training users: median [min-max] over draws x folds")
    for n, g in t.groupby("n", sort=False):
        cells = [f"{c}={int(g[f'min_{c}'].median())} [{g[f'min_{c}'].min()}-{g[f'min_{c}'].max()}]" for c in cls7]
        print(f"n={n:>3} iphone {g.iphone_share.median():.2f} [{g.iphone_share.min():.2f}-{g.iphone_share.max():.2f}] "
              + " ".join(cells))
    print("\nclass absent from the n training users: number of (draw, fold) of", "25 (n=all: 5)")
    for n, g in t.groupby("n", sort=False):
        print(f"n={n:>3} " + " ".join(f"{c}={int((g[f'min_{c}'] == 0).sum())}" for c in cls7))
    print("\nGPS-covered official minutes (GPS model's ES rows), median over draws x folds")
    for n, g in t.groupby("n", sort=False):
        print(f"n={n:>3} " + " ".join(f"{c}={int(g[f'gps_{c}'].median())}" for c in cls7))
    print("\nusers with the class among the n, median [min-max]")
    for n, g in t.groupby("n", sort=False):
        print(f"n={n:>3} " + " ".join(f"{c}={int(g[f'users_{c}'].median())} [{g[f'users_{c}'].min()}-"
                                       f"{g[f'users_{c}'].max()}]" for c in cls7))


if __name__ == "__main__":
    main()
