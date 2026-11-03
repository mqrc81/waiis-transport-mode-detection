"""Class sets shared by all experiments: the 8 SHL classes and the on-phone family."""
from __future__ import annotations

import numpy as np

# SHL coarse label ids (doc_dataset.pdf, section 3.11). 0 = Null, dropped.
CLASSES = ["still", "walk", "run", "bike", "car", "bus", "train", "subway"]
CLASS_ID = {c: i + 1 for i, c in enumerate(CLASSES)}   # still=1 ... subway=8, as in SHL
NULL = 0

FAMILIES = ["still", "foot", "bike", "road", "rail"]
FAMILY_OF = {"still": "still", "walk": "foot", "run": "foot", "bike": "bike",
             "car": "road", "bus": "road", "train": "rail", "subway": "rail", "tram": "rail"}

# Lookup table: class id (0..9; 9 = tram, waiis_bench_v2) -> family id (1..5), 0 stays 0.
_FAM_LUT = np.array([0] + [FAMILIES.index(FAMILY_OF[c]) + 1 for c in CLASSES + ["tram"]], dtype=np.int8)


def to_family(y: np.ndarray) -> np.ndarray:
    """Map class ids (0..8, SHL coding; 9 = tram) to family ids (0..5, 0 = Null)."""
    return _FAM_LUT[np.asarray(y)]

# waiis_bench_v2 (experiment 09): 9 fine classes, which models train on (tram added, id 9), and the six-class
# primary score, merged at scoring time. CLASSES / CLASS_ID above stay v1's 8 classes.
FINE_CLASSES = CLASSES + ["tram"]
FINE_ID = {c: i + 1 for i, c in enumerate(FINE_CLASSES)}   # still=1 ... subway=8, tram=9
SCORE6 = ["still", "foot", "bike", "car", "bus", "rail"]
SCORE6_OF = {"still": "still", "walk": "foot", "run": "foot", "bike": "bike", "car": "car", "bus": "bus",
             "train": "rail", "subway": "rail", "tram": "rail"}
_SCORE6_LUT = np.array([0] + [SCORE6.index(SCORE6_OF[c]) + 1 for c in FINE_CLASSES], dtype=np.int8)


def to_score6(y: np.ndarray) -> np.ndarray:
    """Map fine class ids (0..9, FINE_ID) to six-class ids (0..6, SCORE6 order; 0 stays 0)."""
    return _SCORE6_LUT[np.asarray(y)]
