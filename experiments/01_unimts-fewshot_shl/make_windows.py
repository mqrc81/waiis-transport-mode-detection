"""Step 2: cut SHL hips acc+gyro into 10 s pure windows in UniMTS format, at 20 Hz and 100 Hz.

20 Hz: zero-phase FIR decimation (resample_poly, factor 5, padtype='line') of each continuous
signal stretch BEFORE windowing. 100 Hz: the same windows, same order, same meta.
Usage: python make_windows.py      (reads <data>/processed/shl_u1_hips_w10s/raw, run cache_raw.py first)
Writes <data>/processed/shl_u1_hips_w10s/{hz20,hz100}/{A,B,C}/ and logs/windows_log.txt
"""
import glob, json, os, sys
from collections import Counter
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from shl_common import RAW, OUT, LOGS, EXT, FS, WIN, CLASSES, day_of, load_raw, signal_stretches, pure_windows

POOL_DAYS = {'190517', '250517'}
TEST_CAP = 300          # windows per SHL class in the test split
SEED = 0
DEC = 5                 # 100 Hz -> 20 Hz

# SHL coarse label -> class index, and the two label wordings per class set
SETS = {
    'A': {'map': {1: 0, 2: 1, 3: 2},
          'w1': ['still', 'walk', 'run'],
          'w2': ['standing still', 'walking', 'jogging']},
    'B': {'map': {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2},
          'w1': ['still', 'walk or run', 'vehicle'],
          'w2': ['standing still', 'walking or running', 'riding in a vehicle']},
    'C': {'map': {c: c - 1 for c in range(1, 9)},
          'w1': ['still', 'walk', 'run', 'bike', 'car', 'bus', 'train', 'subway'],
          'w2': ['standing still', 'walking', 'jogging', 'cycling', 'driving a car',
                 'riding a bus', 'riding a train', 'riding the subway']},
}

log_lines = []


def log(*a):
    s = ' '.join(str(v) for v in a)
    print(s, flush=True)
    log_lines.append(s)


def grid_drops(t, x, y):
    """Diagnostic: non-overlapping 10 s grid from recording start, first failing rule per window."""
    out = Counter()
    nan = np.isnan(x).any(1)
    for g in range(0, len(t) - WIN + 1, WIN):
        sl = slice(g, g + WIN)
        if (y[sl] == 0).any():
            out['null'] += 1
        elif nan[sl].any():
            out['nan'] += 1
        elif (np.diff(t[sl]) != 1000 // FS).any():
            out['gap'] += 1
        elif (y[sl] != y[g]).any():
            out['mixed_label'] += 1
        else:
            out['pure'] += 1
    return out


def main():
    recs = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{RAW}/*.npz'))
    log(f'recordings: {len(recs)}; pool days: {sorted(POOL_DAYS)}; seed {SEED}; test cap {TEST_CAP}/class')

    # pass 1: window starts + labels for every recording (no signal copied yet)
    cand, drops = [], {'fewshot_pool': Counter(), 'test': Counter()}
    for r in recs:
        t, x, y = load_raw(r)
        split = 'fewshot_pool' if day_of(r) in POOL_DAYS else 'test'
        drops[split] += grid_drops(t, x, y)
        for s in pure_windows(t, x, y, align=DEC):
            cand.append((split, r, int(s), int(y[s]), int(t[s])))
    cand = pd.DataFrame(cand, columns=['split', 'rec', 'start_idx', 'shl_label', 'start_ms'])
    for sp in drops:
        log(f'[{sp}] diagnostic 10 s grid from recording start, first failing rule: {dict(drops[sp])}')
    tab = cand.groupby(['split', 'shl_label']).size().unstack(fill_value=0).rename(columns=CLASSES)
    log('pure windows extracted (run tiling, start snapped to 20 Hz grid):\n' + tab.to_string())

    # test cap, applied once on SHL labels so A/B/C share the same test windows
    rng = np.random.default_rng(SEED)
    keep = []
    for c in range(1, 9):
        i = np.flatnonzero((cand.split == 'test') & (cand.shl_label == c))
        keep.append(i if len(i) <= TEST_CAP else np.sort(rng.choice(i, TEST_CAP, replace=False)))
    sel = pd.concat([cand[cand.split == 'fewshot_pool'], cand.iloc[np.concatenate(keep)]])
    sel = sel.sort_values(['split', 'rec', 'start_idx']).reset_index(drop=True)
    tab = sel.groupby(['split', 'shl_label']).size().unstack(fill_value=0).rename(columns=CLASSES)
    log('final windows after test cap:\n' + tab.to_string())

    # pass 2: decimate each signal stretch, cut the selected windows at both rates
    X100 = np.zeros((len(sel), WIN, 6), np.float32)
    X20 = np.zeros((len(sel), WIN // DEC, 6), np.float32)
    interior = []
    for r, g in sel.groupby('rec'):
        t, x, y = load_raw(r)
        for lo, hi in signal_stretches(t, x):
            m = (g.start_idx >= lo) & (g.start_idx < hi)
            if not m.any():
                continue
            x20 = resample_poly(x[lo:hi].astype(np.float64), 1, DEC, axis=0, padtype='line')
            for row, s in zip(g.index[m], g.start_idx[m]):
                assert (s - lo) % DEC == 0 and s + WIN <= hi
                X100[row] = x[s:s + WIN]
                X20[row] = x20[(s - lo) // DEC:(s - lo) // DEC + WIN // DEC]
                assert (y[s:s + WIN] == y[s]).all() and y[s] != 0
    assert not np.isnan(X100).any() and not np.isnan(X20).any()

    # evidence: 20 Hz window vs decimating its own 100 Hz window (differs only near the window edges)
    d = np.abs(X20 - resample_poly(X100.astype(np.float64), 1, DEC, axis=1, padtype='line'))
    log(f'|X20 - decimate(X100 window)| mean: edges (first/last 10) {d[:, list(range(10)) + list(range(-10, 0))].mean():.4f}, '
        f'interior {d[:, 20:-20].mean():.5f}  (stretch-level decimation has no window-edge transient)')
    for sp in ['fewshot_pool', 'test']:
        m = (sel.split == sp).to_numpy()
        log(f'[{sp}] median |acc| 100 Hz {np.median(np.linalg.norm(X100[m, :, :3], axis=-1)):.3f}, '
            f'20 Hz {np.median(np.linalg.norm(X20[m, :, :3], axis=-1)):.3f} m/s2; '
            f'median |gyro| 20 Hz {np.median(np.linalg.norm(X20[m, :, 3:], axis=-1)):.3f} rad/s')

    sel['day'] = sel.rec.map(day_of)
    sel['start_utc'] = pd.to_datetime(sel.start_ms, unit='ms', utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    meta_cols = ['day', 'rec', 'start_idx', 'start_ms', 'start_utc', 'shl_label']
    for hz, X in [('hz20', X20), ('hz100', X100)]:
        for name, cs in SETS.items():
            d = f'{OUT}/{hz}/{name}'
            os.makedirs(d, exist_ok=True)
            for w in ['w1', 'w2']:
                with open(f'{d}/labels_{w}.json', 'w') as f:
                    json.dump({'label_dictionary': {str(i): [n] for i, n in enumerate(cs[w])}}, f, indent=4)
            for sp in ['fewshot_pool', 'test']:
                m = ((sel.split == sp) & sel.shl_label.isin(cs['map'])).to_numpy()
                np.save(f'{d}/X_{sp}.npy', X[m])
                np.save(f'{d}/y_{sp}.npy', sel.shl_label[m].map(cs['map']).to_numpy(np.int64))
                sel.loc[m, meta_cols].to_csv(f'{d}/meta_{sp}.csv', index=False)
                if hz == 'hz20':
                    cnt = np.bincount(sel.shl_label[m].map(cs['map']), minlength=len(cs['w1']))
                    log(f'set {name} {sp:12s} N={m.sum():5d} per class {dict(zip(cs["w1"], cnt.tolist()))}')

    # checks: same meta at both rates, shapes/dtypes, test cap, and what load_custom_data makes of it
    for name in SETS:
        for sp in ['fewshot_pool', 'test']:
            a, b = (pd.read_csv(f'{OUT}/{hz}/{name}/meta_{sp}.csv') for hz in ['hz20', 'hz100'])
            assert a.equals(b)
            x20, x100 = (np.load(f'{OUT}/{hz}/{name}/X_{sp}.npy') for hz in ['hz20', 'hz100'])
            y = np.load(f'{OUT}/hz20/{name}/y_{sp}.npy')
            assert x20.dtype == np.float32 and x20.shape[1:] == (200, 6) and x100.shape[1:] == (1000, 6)
            assert len(x20) == len(x100) == len(y) == len(a) and y.dtype == np.int64
    assert (sel[sel.split == 'test'].shl_label.value_counts() <= TEST_CAP).all()
    sys.path.insert(0, str(EXT))
    from data import load_custom_data
    inp, mask, lab, label_list, _ = load_custom_data(
        f'{OUT}/hz20/C/X_test.npy', f'{OUT}/hz20/C/y_test.npy', f'{OUT}/hz20/C/labels_w1.json', [5], 20)
    log(f'load_custom_data(hz20/C test, joint 5, rate 20): inputs {tuple(inp.shape)}, mask mean {mask.mean():.2f}, '
        f'labels {label_list}; resampling is identity: {np.allclose(inp[:, :, 30:36].numpy(), np.load(f"{OUT}/hz20/C/X_test.npy"), atol=1e-4)}')

    with open(f'{LOGS}/windows_log.txt', 'w') as f:
        f.write('\n'.join(log_lines) + '\n')


if __name__ == '__main__':
    main()
