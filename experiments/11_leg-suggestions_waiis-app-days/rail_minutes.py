"""Stop-point 3, descriptive: model minutes inside the diaries' rail legs (metro, tram), from the written output
files: argmax fine class and top level per minute, GPS coverage, mean top-level probabilities. Counts only.

    uv run python rail_minutes.py > logs/rail_minutes.log
"""
import json

import numpy as np
import pandas as pd

import review as R
from tmd import legs as L
from tmd.readers import waiis

rows = []
for day in waiis.days():
    d = R.diary(day.date)
    for mid in ("E2", "E1"):
        doc = json.loads((R.OUT / f"user={day.user}" / f"date={day.date}" / f"legs_{mid}.json").read_text())
        m = pd.DataFrame(doc["minutes"])
        if m.empty:
            continue
        t = m.t_ms.to_numpy() / 1000
        for r in d[d["mode"] == "rail"].itertuples():
            sel = (t + 60 > r.start) & (t < r.end)
            for i in np.flatnonzero(sel):
                pf = m.p_fine.iat[i]
                p = m.p.iat[i]
                rows.append({"date": day.date, "model": mid, "diary_id": r.diary_id, "sub": r.sub_mode,
                             "fine": max(pf, key=pf.get), "top": max(p, key=p.get), "p_rail": p["rail"],
                             "covered": m.coverage.iat[i] == "acc+gps"})
x = pd.DataFrame(rows)
n_rail = sum(int((R.diary(dd.date)["mode"] == "rail").sum()) for dd in waiis.days())
print(f"diary rail legs {n_rail}; model minutes inside them: {x.groupby('model').size().to_dict()}")
for (mid, sub), g in x.groupby(["model", "sub"]):
    print(f"\n{mid} {sub}: {len(g)} minutes in {g.diary_id.nunique()} legs; GPS-covered {g.covered.mean():.2f}; "
          f"mean p(rail) {g.p_rail.mean():.2f}")
    print("  argmax fine:", g.fine.value_counts().to_dict())
    print("  argmax top:", g.top.value_counts().to_dict())
    print("  by GPS coverage, argmax top:", {c: gg.top.value_counts().to_dict() for c, gg in g.groupby("covered")})
