"""gps-v2 rule input (descriptive): are ExtraSensory Android repeated positions real stops?

Android dev users, all labelled minutes (cls, before caps and drops) whose fix has a pair <= 150 s. Repeat = distance
exactly 0 to the previous fix. Real stop = a repeat in a still minute, or in a moving-class minute where an adjacent
pair (previous or next, same session) has derived speed < 1 m/s. Rule (Plan): >= 50% real stops -> keep repeats,
uniform 1 m/s stop threshold; otherwise repeats become missing.

    uv run python diag_repeats.py        # logs/diag_repeats.log
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tmd import bench
from tmd import gps_features as gf

HERE = Path(__file__).resolve().parent
STOP = 1.0


def main():
    lab = bench.read("extrasensory", "labels", columns=["user", "ts", "cls", "platform", "split"])
    lab = lab[(lab.split == "dev")]
    g = bench.read("extrasensory", "gps_per_minute")
    g = g[g.user.isin(set(lab.user))]
    f = gf.prepare(g).merge(lab, on=["user", "ts"], how="left")
    same = (f.user.eq(f.user.shift()) & f.session.eq(f.session.shift())).to_numpy()
    nxt_same = (f.user.eq(f.user.shift(-1)) & f.session.eq(f.session.shift(-1))).to_numpy()
    v = f.pair_v.to_numpy()
    prev_v = np.where(same, np.roll(v, 1), np.nan)
    next_v = np.where(nxt_same, np.roll(v, -1), np.nan)
    f["repeat"] = f.pair_dist.eq(0).to_numpy()
    f["at_stop"] = (prev_v < STOP) | (next_v < STOP)
    a = f[(f.platform == "android") & f.pair_v.notna() & f.cls.astype(str).ne("")]
    moving = a.cls != "still"
    rep = a[a.repeat]
    real = (rep.cls == "still") | rep.at_stop
    lines = [
        f"Android dev, labelled minutes with a pair: {len(a):,}; repeats: {len(rep):,} ({len(rep) / len(a):.1%})",
        f"repeats in still minutes: {(rep.cls == 'still').mean():.1%}; in moving-class minutes: {(rep.cls != 'still').mean():.1%}",
        "repeat share per class: " + str(a.groupby("cls").repeat.mean().round(3).to_dict()),
        f"repeats in moving minutes at a stop (adjacent pair < {STOP} m/s): "
        f"{rep[rep.cls != 'still'].at_stop.mean():.1%} of {int((rep.cls != 'still').sum())}",
        "  per class: " + str(rep[rep.cls != "still"].groupby("cls").at_stop.mean().round(3).to_dict()),
        f"moving minutes without a repeat, at a stop (reference): {a[moving & ~a.repeat].at_stop.mean():.1%}",
        f"REAL-STOP SHARE of Android repeats: {real.mean():.1%} -> "
        + ("keep repeats; uniform stop threshold 1.0 m/s" if real.mean() >= 0.5 else "repeats become missing"),
        "",
        "iPhone (reference): repeat share per class: "
        + str(f[(f.platform == "iphone") & f.pair_v.notna() & f.cls.astype(str).ne("")].groupby("cls").repeat.mean().round(3).to_dict()),
    ]
    text = "\n".join(lines)
    (HERE / "logs" / "diag_repeats.log").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
