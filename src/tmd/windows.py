"""Windows of 5 s with a 2.5 s stride at 25 Hz (brief: Common rules).

2.5 s is 62.5 samples, so starts are floor(k * 62.5): strides alternate 62 / 63 samples and stay on the 2.5 s grid.
"""
from __future__ import annotations

import numpy as np

from tmd.resample import FS

WIN_S, STRIDE_S = 5.0, 2.5
WIN = int(WIN_S * FS)          # 125 samples


def starts(n: int) -> np.ndarray:
    """Window start indices for a contiguous stretch of n samples."""
    k = np.arange(int(max(n - WIN, -1) // (STRIDE_S * FS)) + 2)
    s = np.floor(k * STRIDE_S * FS).astype(int)
    return s[s + WIN <= n]


def cut(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(n, c) -> (len(s), WIN, c)."""
    return x[s[:, None] + np.arange(WIN)]


def majority(y: np.ndarray, s: np.ndarray, n_labels: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Majority label and purity (all samples share it) per window. y: (n,) int labels, 0 = Null."""
    yw = y[s[:, None] + np.arange(WIN)]
    counts = np.stack([(yw == c).sum(1) for c in range(n_labels)], 1)
    maj = counts.argmax(1)
    return maj.astype(np.int8), counts.max(1) == WIN
