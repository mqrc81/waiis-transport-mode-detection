"""Stop-point 2, descriptive: what E2 and E1 predict (top level, saved out-of-fold, seed 0) on ExtraSensory dev minutes
flagged `motorbike` (the tool's motorcycle; never a model class). No MLflow run (descriptive count).

    uv run python motorbike.py > logs/motorbike.log
"""
import numpy as np
import pandas as pd

import dev
from tmd import bench
from tmd import legs as L

df = dev.frame("extrasensory", gps=True)
lab = bench.read("extrasensory", "labels", columns=["user", "ts", "motorbike", "cls"])
mb = df[["user", "ts"]].merge(lab, on=["user", "ts"], how="left").motorbike.fillna(False).astype(bool).to_numpy()
print(f"ExtraSensory dev usable minutes flagged motorbike: {mb.sum()} from {df.user[mb].nunique()} users; "
      f"GPS-covered {df.covered[mb].mean():.2f}; their label cls: "
      f"{df[['user', 'ts']].merge(lab, on=['user', 'ts'], how='left').cls[mb].value_counts().to_dict()}")
if not mb.any():
    raise SystemExit("no dev minute is flagged motorbike: the flag occurs only in the reserved test set (not used)")
for name in dev.MODELS:
    q = L.top_probs(dev.saved(name, 0, df, "p")[mb])
    top = pd.Series(np.array(L.MODES)[q.argmax(1)]).value_counts(normalize=True).round(3).to_dict()
    print(f"{name}: top-level argmax shares {top}; mean probabilities "
          f"{dict(zip(L.MODES, q.mean(0).round(3)))}")
