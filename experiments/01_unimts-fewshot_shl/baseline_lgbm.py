"""Step 5: LightGBM baseline on the same k-shot windows and test set as UniMTS.
Variants: {acc, accgyro} x {hz20, hz100}. Orientation-invariant channels |acc| (and |gyro|).
Per channel: mean, std, p10/p50/p90, zero crossings around the mean, Welch band energies
0-1/1-3/3-6/6-10 Hz, dominant frequency (same features at both rates).
Usage: experiments/shl/.env-lgbm/bin/python experiments/shl/baseline_lgbm.py
Writes experiments/shl/results/fewshot/runs/{set}_k{k}_s{seed}_lgbm_{chan}_{hz}.npz (y, pred), same format as UniMTS runs.
"""
import os
import numpy as np
import lightgbm as lgb
from scipy.signal import welch

D = '/Users/ms/data/shl_unimts'
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'fewshot', 'runs')
SETS, KS, SEEDS = ['B', 'C'], [1, 2, 5, 10], range(5)
RATES = {'hz20': 20, 'hz100': 100}
BANDS = [(0, 1), (1, 3), (3, 6), (6, 10)]
PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=1, min_data_in_bin=1,
              subsample=1.0, colsample_bytree=1.0, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1)


def features(X, fs, gyro):
    """X: (N, L, 6) acc xyz, gyro xyz -> (N, n_features)."""
    chans = [np.linalg.norm(X[:, :, :3], axis=-1)] + ([np.linalg.norm(X[:, :, 3:], axis=-1)] if gyro else [])
    out = []
    for c in chans:
        c = c.astype(np.float64)
        mu = c.mean(1, keepdims=True)
        zc = (np.diff(np.sign(c - mu), axis=1) != 0).sum(1)
        f, p = welch(c, fs=fs, nperseg=4 * fs, axis=1)          # 4 s segments, 0.25 Hz resolution
        df = f[1] - f[0]
        bands = [p[:, (f >= lo) & (f < hi)].sum(1) * df for lo, hi in BANDS]
        dom = f[1:][p[:, 1:].argmax(1)]                           # skip DC
        out += [mu[:, 0], c.std(1), *np.percentile(c, [10, 50, 90], axis=1), zc, *bands, dom]
    return np.stack(out, 1)


os.makedirs(RES, exist_ok=True)
for s in SETS:
    y_test = np.load(f'{D}/hz20/{s}/y_test.npy')
    for hz, fs in RATES.items():
        X_test = np.load(f'{D}/{hz}/{s}/X_test.npy')
        for chan, gyro in [('acc', False), ('accgyro', True)]:
            F_test = features(X_test, fs, gyro)
            for k in KS:
                for seed in SEEDS:
                    kd = f'{D}/kshot/{s}/k{k}_s{seed}'
                    F_tr, y_tr = features(np.load(f'{kd}/X_{hz}.npy'), fs, gyro), np.load(f'{kd}/y.npy')
                    clf = lgb.LGBMClassifier(random_state=seed, **PARAMS).fit(F_tr, y_tr)
                    pred = clf.predict(F_test)
                    np.savez(f'{RES}/{s}_k{k}_s{seed}_lgbm_{chan}_{hz}.npz', y=y_test, pred=pred, seed=seed)
            print(f'set {s} {hz} {chan}: done ({F_test.shape[1]} features)', flush=True)
