"""Accelerometer timing (steps, drift from a 40 ms grid) and GPS fixes near an accelerometer burst, by state.
Same data and states as describe.py.

    uv run python imu_timing.py [<data root of the WAIIS days>]
"""
import numpy as np
import pandas as pd

from describe import all_days, gps_day
from tmd.readers import waiis

G = pd.concat([gps_day(d) for d in all_days()], ignore_index=True)
for d in all_days():
    a = waiis.read_imu(d)
    dt = a.t_ms.diff()[~a.gap].iloc[1:]
    print(f"date={d.date}", "dt 40/41 share %.4f, 39 %.4f, <39 %d, 42-200 %d, >200 %d" % (dt.isin([40, 41]).mean(), (dt == 39).mean(), (dt < 39).sum(), ((dt > 41) & (dt <= 200)).sum(), (dt > 200).sum()))
    # per-sample offsets from a 40 ms grid within bursts: cumulative drift
    burst = a.gap.cumsum()
    off = a.groupby(burst).t_ms.transform(lambda t: (t - t.iloc[0]) - 40 * np.arange(len(t)))
    print("   drift from 40 ms grid within burst [ms]: p1 %.0f p50 %.0f p99 %.0f" % tuple(off.quantile([.01, .5, .99])))
    # short steps: where?
    pos = a.groupby(burst).cumcount()
    sh = a.index[(a.t_ms.diff() < 39) & ~a.gap]
    print("   short steps at sample index within burst: median", int(pos[sh].median()) if len(sh) else None, "n", len(sh))
    bb = a.groupby(burst).t_ms.agg(["min", "max"])
    g = G[G.day == d.date]
    inside = np.zeros(len(g), bool)
    for lo, hi in bb.itertuples(index=False):
        inside |= (g.t_ms >= lo - 30000).to_numpy() & (g.t_ms <= hi + 30000).to_numpy()
    print("   GPS fixes within 30 s of an IMU burst, by state:", g.assign(i=inside).groupby("state").i.mean().round(2).to_dict())
    print("   GPS (UTC) %s–%s, IMU %s–%s" % tuple(pd.to_datetime(x, unit="ms").strftime("%H:%M") for x in (g.t_ms.min(), g.t_ms.max(), a.t_ms.min(), a.t_ms.max())))
