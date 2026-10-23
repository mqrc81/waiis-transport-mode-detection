"""Step 1: compare UniMTS's resampling (scipy.signal.resample, FFT) with decimate (FIR, zero-phase)
on real 10 s SHL windows from one recording. Usage: python check_resample.py [recid]"""
import sys
import numpy as np
from scipy.signal import resample, decimate, welch
from shl_common import load_raw, pure_windows, CLASSES, FS, WIN

rec = sys.argv[1] if len(sys.argv) > 1 else '010317'
t, x, y = load_raw(rec)
w = pure_windows(t, x, y)
for c in np.unique(y[w]):
    seg = np.stack([x[s:s + WIN] for s in w[y[w] == c]]).astype(np.float64)   # n,1000,6
    f, p = welch(seg[:, :, :3], fs=FS, nperseg=256, axis=1)
    frac = p[:, f > 10].sum(1) / p[:, f > 0].sum(1)                            # power above new Nyquist
    a = np.stack([resample(s, WIN // 5) for s in seg])                         # what load_custom_data does
    b = decimate(seg, 5, ftype='fir', zero_phase=True, axis=1)
    d = np.abs(a - b)
    edge, mid = d[:, :5].mean(), d[:, 20:-20].mean()
    print(f'{CLASSES[c]:6s} n={len(seg):4d}  acc power >10 Hz: {100*np.median(frac):.1f}% | '
          f'|fft-resample - FIR-decimate| mean: first/last 5 samples {edge:.3f}, interior {mid:.3f} m/s2 (acc+gyro mixed)')
