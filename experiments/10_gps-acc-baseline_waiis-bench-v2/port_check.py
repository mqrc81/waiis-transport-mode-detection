"""Port check (stop-point 1): the n-class changes to tmd.models (lgbm, minutes, hmm) must leave v1 bit-identical, and
10's feature tables from waiis_bench_v2 must equal 05's / 06's on ExtraSensory and SHL User 1.

1. 05's B (seed 0) retrained: out-of-fold probabilities before (B) and after the HMM (F_fb) = 05's saved ones.
2. 06's final configuration (H1_v2, seed 0): GPS model G_v2 retrained (lr 0.02), product, HMM fb = 06's saved ones.
3. 05's 580 acc features and 06's gps-v2 features on the rows both tables have (v2 tables from make_features.py).

    uv run python port_check.py          # logs/port_check.log
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from make_features import acc_path, gps_path
from tmd.features import acc as af
from tmd.features import gps as gf
from tmd.models import fusion, lgbm, minutes
from tmd.paths import PROCESSED

V1 = "waiis_bench_v1"
HERE = Path(__file__).resolve().parent
P05, P06 = PROCESSED / "05_acc-baseline_waiis-bench", PROCESSED / "06_gps-acc-baseline_waiis-bench"
GPS_LR = {**lgbm.PARAMS, "learning_rate": 0.02}, 10000, 150      # 06's G_v2: lr 0.02, max rounds, patience


def same(name: str, p: np.ndarray, ref: np.ndarray) -> str:
    return (f"{name}: bit-identical = {np.array_equal(p, ref, equal_nan=True)}; "
            f"max abs diff {np.nanmax(np.abs(p - ref)):.3g}")


def models(lines: list[str]) -> None:
    df = fusion.load_frame("dev", "gps-v2")                 # v1: 05's acc tables + 06's gps-v2 tables
    lab = pd.concat([minutes.labelled_minutes(s, version=V1) for s in ("extrasensory", "shl_user1")],
                    ignore_index=True)
    y, fold, off = df.y.to_numpy(), df.fold.to_numpy(), df.official.to_numpy()
    cov = df.covered.to_numpy()
    acc = lgbm.paired_cv(df[fusion.ACC].to_numpy(np.float32), y, fold, off & (fold >= 0), np.ones(len(df)), seed=0,
                         name="B")
    lines.append(same("05 B", acc["p"], np.load(P05 / "oof" / "B_seed0.npz")["p"]))
    lines.append(same("05 F_fb", minutes.smooth_cv(df, acc["p"], lab, "fb"),
                      np.load(P05 / "oof" / "F_fb_seed0.npz")["p"]))
    prm, mr, pat = GPS_LR
    gps = lgbm.paired_cv(df[gf.names("gps-v2")].to_numpy(np.float32), y, fold, off & cov & (fold >= 0),
                         np.ones(len(df)), 0, prm, mr, pat, name="G_v2")
    z = np.load(P06 / "oof" / "H1_v2_seed0.npz", allow_pickle=True)
    assert (z["user"] == df.user.to_numpy(str)).all() and (z["ts"] == df.ts.to_numpy()).all()
    raw = fusion.product(acc["p"], gps["p"], cov)
    lines.append(same("06 H1_v2 raw", raw, z["raw"]))
    lines.append(same("06 H1_v2 fb", minutes.smooth_cv(df, raw, lab, "fb"), z["p"]))


def features(lines: list[str]) -> None:
    for s in ("extrasensory", "shl_user1"):
        a5 = pd.read_parquet(P05 / f"{s}_dev_25hz_w15s_acc-v1_lp-none.parquet", columns=["user", "ts"] + fusion.ACC)
        a10 = pd.read_parquet(acc_path(s, "dev"), columns=["user", "ts"] + fusion.ACC)
        g6 = pd.read_parquet(P06 / f"{s}_dev_gps-v2_w2min.parquet", columns=["user", "ts"] + gf.names("gps-v2"))
        g10 = pd.read_parquet(gps_path(s, "dev"), columns=["user", "ts"] + gf.names("gps-v2"))
        for nm, old, new, cols in (("acc", a5, a10, fusion.ACC), ("gps-v2", g6, g10, gf.names("gps-v2"))):
            m = old.merge(new, on=["user", "ts"], suffixes=("_o", "_n"), validate="1:1")
            o = m[[c + "_o" for c in cols]].to_numpy(np.float32)
            n = m[[c + "_n" for c in cols]].to_numpy(np.float32)
            lines.append(f"{s} {nm}: rows 05/06 {len(old)}, 10 {len(new)}, common {len(m)}; "
                         f"identical = {np.array_equal(o, n, equal_nan=True)}; "
                         f"max abs diff {np.nanmax(np.abs(o - n)):.3g}")


def main():
    lines: list[str] = []
    features(lines)
    print("\n".join(lines), flush=True)
    models(lines)
    text = "\n".join(lines)
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / "port_check.log").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
