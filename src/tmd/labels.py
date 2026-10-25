"""Class sets shared by all experiments: the 8 SHL classes and the on-phone family."""
from __future__ import annotations

import numpy as np

# SHL coarse label ids (doc_dataset.pdf, section 3.11). 0 = Null, dropped.
CLASSES = ["still", "walk", "run", "bike", "car", "bus", "train", "subway"]
CLASS_ID = {c: i + 1 for i, c in enumerate(CLASSES)}   # still=1 ... subway=8, as in SHL
NULL = 0

FAMILIES = ["still", "foot", "bike", "road", "rail"]
FAMILY_OF = {"still": "still", "walk": "foot", "run": "foot", "bike": "bike",
             "car": "road", "bus": "road", "train": "rail", "subway": "rail"}

# Lookup table: class id (0..8) -> family id (1..5), 0 stays 0.
_FAM_LUT = np.array([0] + [FAMILIES.index(FAMILY_OF[c]) + 1 for c in CLASSES], dtype=np.int8)


def to_family(y: np.ndarray) -> np.ndarray:
    """Map class ids (0..8, SHL coding) to family ids (0..5, 0 = Null)."""
    return _FAM_LUT[np.asarray(y)]
