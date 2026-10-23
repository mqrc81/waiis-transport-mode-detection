"""Step 4: draw k windows per class from fewshot_pool, per (set, k, seed). All methods use these files.
Usage: python make_kshot.py
Writes <data>/processed/shl_u1_hips_w10s/kshot/{set}/k{k}_s{seed}/: idx.npy (rows of fewshot_pool, same at 20 and 100 Hz),
X_hz20.npy, X_hz100.npy, y.npy, meta.csv
"""
import os
import numpy as np
import pandas as pd
from shl_common import OUT

SETS, KS, SEEDS = ['A', 'B', 'C'], [1, 2, 5, 10], range(5)

for s in SETS:
    y = np.load(f'{OUT}/hz20/{s}/y_fewshot_pool.npy')
    meta = pd.read_csv(f'{OUT}/hz20/{s}/meta_fewshot_pool.csv')
    X20 = np.load(f'{OUT}/hz20/{s}/X_fewshot_pool.npy', mmap_mode='r')
    X100 = np.load(f'{OUT}/hz100/{s}/X_fewshot_pool.npy', mmap_mode='r')
    for k in KS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = np.concatenate([np.sort(rng.choice(np.flatnonzero(y == c), k, replace=False))
                                  for c in range(y.max() + 1)])
            d = f'{OUT}/kshot/{s}/k{k}_s{seed}'
            os.makedirs(d, exist_ok=True)
            np.save(f'{d}/idx.npy', idx)
            np.save(f'{d}/X_hz20.npy', np.ascontiguousarray(X20[idx]))
            np.save(f'{d}/X_hz100.npy', np.ascontiguousarray(X100[idx]))
            np.save(f'{d}/y.npy', y[idx])
            meta.iloc[idx].to_csv(f'{d}/meta.csv', index=False)
            assert (np.bincount(y[idx]) == k).all()
    print(f'set {s}: wrote k {KS} x seeds {list(SEEDS)}')
