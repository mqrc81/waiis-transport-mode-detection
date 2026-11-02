"""iOS sign check on v1 (README, stop-point 2 answers, item 4): mean z of ExtraSensory still windows per platform
(dev users only, 4,000 sampled per platform) and invariance of 05/06's 580 accelerometer features to the
iOS -> Android transform (all axes negated) and an axis swap, on 500 dev windows. No change to v1.

    uv run python sign_check.py      # -> results/tables/es_still_mean_z.csv
"""
import numpy as np, pandas as pd
from tmd import bench, acc_features as af
e = bench.examples("extrasensory")
e = e[e.split == "dev"]                                   # dev users only
still = e[e.cls == "still"]
rows = []
for plat, g in still.groupby("platform"):
    keep = g.sample(min(len(g), 4000), random_state=0)
    df, w = bench.load_windows("extrasensory", keep)
    m = w.mean(1)                                          # (n, 3) window mean, m/s2
    dom = np.abs(m).argmax(1)
    rows.append({"platform": plat, "users": keep.user.nunique(), "windows": len(w),
                 "mean_z": float(np.median(m[:, 2])), "mean_z_p5": float(np.percentile(m[:, 2], 5)),
                 "mean_z_p95": float(np.percentile(m[:, 2], 95)),
                 "share_z_dominant": float((dom == 2).mean()),
                 "share_z_dominant_positive": float(((dom == 2) & (m[:, 2] > 0)).mean()),
                 "share_z_dominant_negative": float(((dom == 2) & (m[:, 2] < 0)).mean())})
t = pd.DataFrame(rows).round(3)
t.to_csv("results/tables/es_still_mean_z.csv", index=False)
print(t.to_string(index=False))
# invariance of 05/06's 580 features on real windows (all axes negated = iOS -> Android direction)
keep = e.sample(500, random_state=1)
df, w = bench.load_windows("extrasensory", keep)
f0 = af.features(w, sub=True, grav=False); f1 = af.features(-w, sub=True, grav=False)
fp = af.features(w[:, :, [1, 0, 2]], sub=True, grav=False)
d = np.nanmax(np.abs(f0 - f1) / (np.abs(f0) + 1e-6)); dp = np.nanmax(np.abs(f0 - fp) / (np.abs(f0) + 1e-6))
print(f"580 features, {len(w)} dev windows: max relative change, all axes negated {d:.2e}; x/y swapped {dp:.2e}; "
      f"NaN pattern equal {np.array_equal(np.isnan(f0), np.isnan(f1))}")
