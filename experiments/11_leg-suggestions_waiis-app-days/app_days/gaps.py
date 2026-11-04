"""GPS gaps > 60 s by the speed across the gap, and the coarse fixes (> 500 m). Same data and states as describe.py.

    uv run python gaps.py [<data root of the WAIIS days>]
"""
import numpy as np
import pandas as pd

from describe import all_days, gps_day, hav

G = pd.concat([gps_day(d) for d in all_days()], ignore_index=True)
print("fixes without a 60 s baseline:", G.v60.isna().sum())
r = []
for day, g in G.groupby("day"):
    g = g.reset_index(drop=True)
    for i in np.flatnonzero(g.dt_next > 60):
        a, b = g.iloc[i], g.iloc[i + 1]
        r.append(dict(len=a.dt_next, v=hav(a.lat, a.lon, b.lat, b.lon) / a.dt_next, coarse=max(a.accuracy_m, b.accuracy_m) > 500))
r = pd.DataFrame(r)
r["kind"] = pd.cut(r.v, [0, 0.5, 2, np.inf], right=False, labels=["stationary <0.5", "0.5-2", "moving >=2"])
print(r.groupby("kind", observed=True).agg(n=("len", "size"), len_med=("len", "median"), len_sum_h=("len", lambda s: s.sum() / 3600), coarse_end=("coarse", "sum")).round(1))
c = G[G.accuracy_m > 500]
print("fixes >500 m:", len(c), "speed null:", c.speed_mps.isna().mean(), "interval to next p50:", c.dt_next.median(), "values:", c.accuracy_m.round(0).value_counts().head(6).to_dict())
