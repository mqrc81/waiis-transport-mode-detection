"""Rail diagnostics on ExtraSensory dev, descriptive only (no tuning): rail recall per user before (B) and after the
HMM (F_fb), mean P(rail) on train minutes before smoothing, and where train minutes go.

    uv run python rail.py --seed 0
Output: logs/rail_seed<k>.log (and printed).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pipeline as pl

from tmd.evaluate import family_proba
from tmd.labels import CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--before", default="B")
    ap.add_argument("--after", default="F_fb")
    a = ap.parse_args()
    df = pl.load(pl.Cfg("keys"))
    pb, pa = pl.load_oof(a.before, a.seed, df), pl.load_oof(a.after, a.seed, df)
    es = ((df.source == "extrasensory") & df.official).to_numpy()
    d = df[es].reset_index(drop=True)
    pb, pa = pb[es], pa[es]
    fam_b, fam_a = family_proba(pb).argmax(1) + 1, family_proba(pa).argmax(1) + 1
    p_rail = pb[:, 6] + pb[:, 7]                     # train + subway, before smoothing
    tr = (d.y == 7).to_numpy()
    rows = []
    for u in sorted(d.user[tr].unique()):
        s = tr & (d.user == u).to_numpy()
        rows.append({"user": u[:8], "platform": d.platform[s].iloc[0], "fold": int(d.fold[s].iloc[0]),
                     "n_train": int(s.sum()), "rail_recall_before": float((fam_b[s] == 5).mean()),
                     "rail_recall_after": float((fam_a[s] == 5).mean()),
                     "mean_p_rail_before": float(p_rail[s].mean())})
    out = {"per_user": rows,
           "all": {"n_train": int(tr.sum()), "rail_recall_before": float((fam_b[tr] == 5).mean()),
                   "rail_recall_after": float((fam_a[tr] == 5).mean()),
                   "mean_p_rail_before_train": float(p_rail[tr].mean()),
                   "mean_p_rail_before_other": float(p_rail[~tr].mean()),
                   "rail_predicted_before_other": int((fam_b[~tr] == 5).sum()),
                   "rail_predicted_after_other": int((fam_a[~tr] == 5).sum())},
           "train_to_before": {CLASSES[i]: int(n) for i, n in enumerate(np.bincount(pb[tr].argmax(1), minlength=8))},
           "train_to_after": {CLASSES[i]: int(n) for i, n in enumerate(np.bincount(pa[tr].argmax(1), minlength=8))}}
    text = pd.DataFrame(rows).to_string(index=False) + "\n" + json.dumps({k: v for k, v in out.items()
                                                                         if k != "per_user"}, indent=1)
    (pl.HERE / "logs" / f"rail_seed{a.seed}.log").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
