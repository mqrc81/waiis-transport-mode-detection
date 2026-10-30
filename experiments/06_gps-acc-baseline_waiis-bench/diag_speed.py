"""Derived-speed diagnostic (descriptive, before any training): why is ExtraSensory's speed from consecutive fixes low?

Dev users only (reserved test users excluded), labelled minutes of the official set.
1. ratio derived / reported speed per minute where both exist, by dataset, platform and class;
2. fix age: the fix's own time against the minute's, and which times the derived speed uses;
3. repeated / near-repeated positions by platform and class, and what the next pair after a repeat looks like.

    uv run python diag_speed.py          # logs/diag_speed.log, results/diag_speed_*.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tmd import bench
from tmd.labels import CLASSES

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def minutes(src: str) -> pd.DataFrame:
    e = bench.examples(src)
    e = e[e.split == "dev"]
    g = bench.read(src, "gps_per_minute")
    m = e.merge(g, on=["user", "ts"], how="inner", suffixes=("", "_g"))
    m["cls"] = m.y.map(lambda i: CLASSES[i - 1])
    m["plat"] = m.platform if src == "extrasensory" else "shl"
    m["pair"] = m.dt_prev_s.notna() & (m.dt_prev_s <= 150)
    m["derived"] = np.where(m.pair, m.speed_prev_mps, np.nan)
    m["reported"] = np.where(m.speed >= 0, m.speed, np.nan)
    m["fix_age_s"] = m.time - m.ts
    return m


def main():
    OUT.mkdir(exist_ok=True)
    lines = []
    es, shl = minutes("extrasensory"), minutes("shl_user1")
    m = pd.concat([es, shl], ignore_index=True)
    order = {c: i for i, c in enumerate(CLASSES)}

    # 1. ratio derived / reported, both > 0.5 m/s so the ratio is meaningful
    b = m[(m.derived > 0) & (m.reported > 0.5)].copy()
    b["ratio"] = b.derived / b.reported
    r = b.groupby(["plat", "cls"]).agg(n=("ratio", "size"), ratio_p25=("ratio", lambda v: v.quantile(.25)),
                                       ratio_med=("ratio", "median"), ratio_p75=("ratio", lambda v: v.quantile(.75)),
                                       share_below_0_2=("ratio", lambda v: (v < 0.2).mean()),
                                       derived_med=("derived", "median"), reported_med=("reported", "median"))
    r = r.reset_index().sort_values(["plat", "cls"], key=lambda c: c.map(order) if c.name == "cls" else c)
    r.to_csv(OUT / "diag_speed_ratio.csv", index=False)
    lines += ["1. derived / reported speed per minute (both present, reported > 0.5 m/s)", r.round(2).to_string(index=False), ""]

    # 2. fix age and the times the derived speed uses
    lines += ["2. fix time - minute timestamp [s]"]
    for p, d in m.groupby("plat"):
        lines.append(f"  {p}: {d.fix_age_s.describe()[['min', '50%', 'max']].round(1).to_dict()}; "
                     f"dt_prev_s p10/50/90 {d.dt_prev_s.quantile([.1, .5, .9]).round(1).tolist()}")
    lines += ["  derived speed = dist_prev_m / dt_prev_s; dt_prev_s = difference of the fixes' `time`, which is the",
              "  minute timestamp in ExtraSensory (fix time not published, register G5) and the fix's own time in SHL.", ""]

    # 3. repeated / near-repeated positions, and the pair after a repeat
    p3 = m[m.pair].copy()
    p3["repeat"] = p3.dist_prev_m == 0
    p3["near"] = p3.dist_prev_m < 2
    t = p3.groupby(["plat", "cls"]).agg(n=("repeat", "size"), repeated=("repeat", "mean"), under_2m=("near", "mean"))
    t = t.reset_index().sort_values(["plat", "cls"], key=lambda c: c.map(order) if c.name == "cls" else c)
    t.to_csv(OUT / "diag_speed_repeats.csv", index=False)
    lines += ["3. repeated (distance 0) and near-repeated (< 2 m) positions, pairs <= 150 s", t.round(3).to_string(index=False), ""]

    # moving minutes: do repeats come in runs followed by a jump (cached fixes)?
    g = bench.read("extrasensory", "gps_per_minute").sort_values(["user", "session", "time"])
    g = g[~g.user.isin(bench.reserved_users("extrasensory"))]
    lab = bench.read("extrasensory", "labels", columns=["user", "ts", "cls", "platform"])
    g = g.merge(lab, on=["user", "ts"], how="left")
    g["rep"] = (g.dist_prev_m == 0) & (g.dt_prev_s <= 150)
    g["next_after_rep"] = g.groupby(["user", "session"]).rep.shift(1).fillna(False).astype(bool) & ~g.rep
    mov = g[g.cls.isin(["car", "bus", "train"]) & (g.dt_prev_s <= 150)]
    jump = mov.groupby(["platform", "next_after_rep"]).speed_prev_mps.median().unstack().round(2)
    lines += ["   ES car/bus/train: median derived speed of pairs right after a repeat (True) vs others (False)",
              jump.to_string(), ""]

    # within-minute movement (ExtraSensory location features) vs derived speed, car minutes
    lf = bench.read("extrasensory", "location_features",
                    columns=["user", "ts", "location:num_valid_updates", "location:diameter"])
    c = es[es.cls.isin(["car", "bus", "train", "walk", "bike"])].merge(lf, on=["user", "ts"], how="left")
    w = c.groupby(["platform", "cls"]).agg(updates_med=("location:num_valid_updates", "median"),
                                           diameter_med_m=("location:diameter", "median"),
                                           derived_med=("derived", "median"), reported_med=("reported", "median"))
    lines += ["4. ES within-minute location updates and diameter (max distance between the minute's updates) [m]",
              w.round(2).to_string(), ""]
    text = "\n".join(lines)
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / "diag_speed.log").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
