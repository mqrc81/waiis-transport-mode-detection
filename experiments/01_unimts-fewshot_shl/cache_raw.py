"""Cache SHL User1 hips acc+gyro and coarse labels as one .npz per recording.

Columns (doc/doc_dataset.pdf, sec. 3.2 and 3.11):
  Hips_Motion.txt: 1 time [ms], 2-4 acc xyz [m/s2], 5-7 gyro xyz [rad/s]
  Label.txt:       1 time [ms], 2 coarse label (0 Null ... 8 Subway)
Usage: python cache_raw.py [recid ...]   (default: all recordings)
"""
import sys
import numpy as np
import pandas as pd
from shl_common import SHL_RAW as SRC, RAW as DST


def read(path, ncols):
    return pd.read_csv(path, sep=' ', header=None, usecols=range(ncols),
                       engine='c', dtype=np.float64).to_numpy()


def main(recids):
    DST.mkdir(parents=True, exist_ok=True)
    for rec in recids:
        out = DST / f'{rec}.npz'
        motion, label = SRC / rec / 'Hips_Motion.txt', SRC / rec / 'Label.txt'
        if out.exists():
            continue
        if not (motion.exists() and label.exists()):
            print(f'{rec}: missing Hips_Motion.txt or Label.txt, skipped', flush=True)
            continue
        m = read(motion, 7)
        lab = read(label, 2)
        assert len(m) == len(lab), (rec, len(m), len(lab))
        assert np.array_equal(m[:, 0], lab[:, 0]), f'{rec}: timestamps differ'
        np.savez(out, t=m[:, 0].astype(np.int64), x=m[:, 1:7].astype(np.float32),
                 y=lab[:, 1].astype(np.int8))
        print(f'{rec}: {len(m)} rows', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or sorted(p.name for p in SRC.iterdir() if p.is_dir()))
