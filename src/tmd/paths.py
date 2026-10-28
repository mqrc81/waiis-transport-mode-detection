"""Project and data paths. The data root is outside git (see CLAUDE.local.md); override with $TMD_DATA."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]          # src/tmd/paths.py -> repo root
DATA = Path(os.getenv("TMD_DATA", PROJECT.parent / "data"))
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CHECKPOINTS = DATA / "checkpoints"
EXT = PROJECT.parent / "ext"

MLFLOW_DB = PROJECT / "mlflow.db"                 # git-ignored
MLARTIFACTS = PROJECT / "mlartifacts"             # git-ignored

SHL_RAW = RAW / "shl" / "release"
ES_RAW = RAW / "extrasensory"
