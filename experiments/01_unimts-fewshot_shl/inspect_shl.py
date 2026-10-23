"""Step 0: sanity numbers for one recording and pure-window counts for all cached ones.
Usage: python inspect_shl.py [recid_for_stats]   (default 010317)
Writes logs/window_counts.csv
"""
import sys, glob, os
import numpy as np
import pandas as pd
from shl_common import RAW, LOGS, CLASSES, load_raw, pure_windows, day_of

rec = sys.argv[1] if len(sys.argv) > 1 else '010317'
t, x, y = load_raw(rec)
dt = np.diff(t)
print(f'== {rec}: {len(t)} rows, {(t[-1]-t[0])/3.6e6:.2f} h')
print(f'dt [ms]: median {np.median(dt)}, p95 {np.percentile(dt, 95)}, min {dt.min()}, max {dt.max()}, '
      f'n(dt!=10) {(dt != 10).sum()}  -> rate {1000/np.median(dt):.1f} Hz')
nan = np.isnan(x)
print('NaN per channel (acc xyz, gyro xyz):', nan.sum(0).tolist(), f'rows with any NaN: {nan.any(1).sum()}')
v = ~nan.any(1)
acc, gyr = np.linalg.norm(x[v, :3], axis=1), np.linalg.norm(x[v, 3:], axis=1)
print(f'|acc| median {np.median(acc):.3f} m/s2 (p5 {np.percentile(acc,5):.2f}, p95 {np.percentile(acc,95):.2f})')
print(f'|gyro| median {np.median(gyr):.3f} rad/s (p95 {np.percentile(gyr,95):.2f})')
for c in range(1, 9):
    m = v & (y == c)
    if m.any():
        print(f'  {CLASSES[c]:7s} |acc| median {np.median(np.linalg.norm(x[m,:3],axis=1)):.2f}  '
              f'|gyro| median {np.median(np.linalg.norm(x[m,3:],axis=1)):.3f}')
print('label values:', dict(zip(*np.unique(y, return_counts=True))))

rows = []
for f in sorted(glob.glob(f'{RAW}/*.npz')):
    r = os.path.basename(f)[:-4]
    t, x, y = load_raw(r)
    w = pure_windows(t, x, y)
    cnt = np.bincount(y[w], minlength=9)
    rows.append({'rec': r, 'day': day_of(r), **{CLASSES[c]: int(cnt[c]) for c in range(1, 9)}})
df = pd.DataFrame(rows)
df.to_csv(f'{LOGS}/window_counts.csv', index=False)
pd.set_option('display.width', 200, 'display.max_rows', 200)
per_day = df.drop(columns='rec').groupby('day', sort=False).sum()
print(per_day.to_string())
print('TOTAL', per_day.sum().to_dict())
