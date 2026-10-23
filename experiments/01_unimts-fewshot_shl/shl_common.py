"""Shared helpers and paths for the SHL experiments."""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT.parent / 'data'                      # data root, see CLAUDE.local.md
EXT = PROJECT.parent / 'ext' / 'UniMTS'             # upstream clone + patches/UniMTS.patch
UNIMTS_HOME = DATA / 'checkpoints' / 'unimts'       # cwd for upstream scripts: checkpoint/, config.json, UniMTS_data
CKPT = UNIMTS_HOME / 'checkpoint' / 'UniMTS.pth'
SHL_RAW = DATA / 'raw' / 'shl' / 'release' / 'User1'
OUT = DATA / 'processed' / 'shl_u1_hips_w10s'       # {raw,hz20,hz100,kshot}/
RAW = OUT / 'raw'
RUNS = DATA / 'processed' / HERE.name               # result .npz files: zeroshot/, fewshot/
LOGS = HERE / 'logs'
FS = 100          # Hz, SHL resampled grid (doc 3.2)
WIN = 10 * FS     # 10 s windows
CLASSES = {0: 'Null', 1: 'Still', 2: 'Walk', 3: 'Run', 4: 'Bike',
           5: 'Car', 6: 'Bus', 7: 'Train', 8: 'Subway'}


def day_of(rec):
    """Calendar day of a recording: 'm010317' (morning) and '010317' are the same day."""
    return rec.lstrip('m')


def load_raw(rec):
    d = np.load(f'{RAW}/{rec}.npz')
    return d['t'], d['x'], d['y']


def _runs(brk):
    """(start, end) of runs given a boolean array that is True where a new run starts."""
    starts = np.flatnonzero(brk)
    return list(zip(starts, np.append(starts[1:], len(brk))))


def signal_stretches(t, x):
    """Maximal [lo, hi) stretches with no NaN in acc/gyro and a 10 ms step between samples."""
    ok = ~np.isnan(x).any(1)
    brk = np.ones(len(t), bool)
    brk[1:] = (np.diff(t) != 1000 // FS) | (ok[1:] != ok[:-1])
    return [(s, e) for s, e in _runs(brk) if ok[s]]


def label_runs(y, lo, hi):
    """Maximal [s, e) runs of one constant non-Null label inside [lo, hi)."""
    yy = y[lo:hi]
    brk = np.ones(len(yy), bool)
    brk[1:] = yy[1:] != yy[:-1]
    return [(lo + s, lo + e) for s, e in _runs(brk) if yy[s] != 0]


def pure_windows(t, x, y, win=WIN, align=1):
    """Start indices of non-overlapping pure windows (constant non-Null label, no NaN, no gap).

    Each label run inside a signal stretch is tiled from its start; the start is
    moved up to the next multiple of `align` samples from the stretch start
    (align=5 keeps windows on the 20 Hz decimation grid). The remainder is dropped.
    """
    idx = []
    for lo, hi in signal_stretches(t, x):
        for s, e in label_runs(y, lo, hi):
            s = lo + -(-(s - lo) // align) * align
            idx.append(np.arange(s, e - win + 1, win))
    return np.concatenate(idx) if idx else np.zeros(0, int)
