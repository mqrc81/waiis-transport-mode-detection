"""Descriptive (stop-point 2 decisions, item 4): ExtraSensory dev still minutes predicted as rail (train, subway,
tram; after the HMM), split by quiet / not and with / without GPS, per row. Seed 0, from the saved probabilities.

    uv run python false_rail.py A0 A1 E1        # results/tables/false_rail.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import pipe


def main():
    rows = sys.argv[1:] or ["A0", "A1", "E1"]
    df = pipe.load()
    sel = ((df.source == "extrasensory") & df.official).to_numpy() & (df.y.to_numpy() == 1)
    d = df.loc[sel, ["quiet", "covered", "user"]].reset_index(drop=True)
    out = []
    for r in rows:
        f = pipe.load_probs(r, 0, df)[sel].argmax(1) + 1
        rail = np.isin(f, [7, 8, 9])
        for q in (True, False):
            for c in (True, False):
                mk = (d.quiet.to_numpy() == q) & (d.covered.to_numpy() == c)
                out.append({"row": r, "quiet": q, "gps": c, "still_minutes": int(mk.sum()),
                            "to_rail": int((rail & mk).sum()), "to_train": int(((f == 7) & mk).sum()),
                            "to_subway": int(((f == 8) & mk).sum()), "to_tram": int(((f == 9) & mk).sum())})
        out.append({"row": r, "quiet": "all", "gps": "all", "still_minutes": int(len(d)), "to_rail": int(rail.sum()),
                    "to_train": int((f == 7).sum()), "to_subway": int((f == 8).sum()), "to_tram": int((f == 9).sum()),
                    "users": int(d.user[rail].nunique())})
    t = pd.DataFrame(out)
    p = pipe.HERE / "results" / "tables" / "false_rail.csv"
    t.to_csv(p, index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
