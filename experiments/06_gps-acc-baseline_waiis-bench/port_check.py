"""Port check: tmd.lgbm + tmd.minutes (moved from 05's pipeline.py) retrain 05's B (seed 0) and must reproduce 05's
saved out-of-fold probabilities bit-identically, before (B) and after the HMM (F_fb).

    uv run python port_check.py          # logs/port_check.log
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tmd import acc_features as af
from tmd import lgbm, minutes
from tmd.paths import PROCESSED

HERE = Path(__file__).resolve().parent
P05 = PROCESSED / "05_acc-baseline_waiis-bench"


def main():
    cols = ["user", "ts", "session", "official", "y", "platform", "fold", "quiet"] + af.names(True, False)
    df = pd.concat([pd.read_parquet(P05 / f"{s}_dev_25hz_w15s_acc-v1_lp-none.parquet", columns=cols).assign(source=s)
                    for s in ("extrasensory", "shl_user1")], ignore_index=True)
    df["y"], df["fold"] = df.y.astype(int), df.fold.astype(int)
    x = df[af.names(True, False)].to_numpy(np.float32)
    trainable = df.official.to_numpy() & (df.fold.to_numpy() >= 0)
    r = lgbm.paired_cv(x, df.y.to_numpy(), df.fold.to_numpy(), trainable, np.ones(len(df)), seed=0, name="B")
    lab = pd.concat([minutes.labelled_minutes(s) for s in ("extrasensory", "shl_user1")], ignore_index=True)
    fb = minutes.smooth_cv(df, r["p"], lab, "fb")
    lines = []
    for name, p in (("B", r["p"]), ("F_fb", fb)):
        z = np.load(P05 / "oof" / f"{name}_seed0.npz")
        assert (z["user"] == df.user.to_numpy(str)).all() and (z["ts"] == df.ts.to_numpy()).all()
        same = np.array_equal(p, z["p"], equal_nan=True)
        lines.append(f"{name}: bit-identical to 05 = {same}; max abs diff {np.nanmax(np.abs(p - z['p'])):.3g}")
    lines.append(f"rounds per fold: {[x['rounds'] for x in r['rows']]}")
    text = "\n".join(lines)
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / "port_check.log").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
