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

# git-ignored; $TMD_MLFLOW_DB / $TMD_MLARTIFACTS redirect them (e.g. a scratch store for a reproduction check)
MLFLOW_DB = Path(os.getenv("TMD_MLFLOW_DB", PROJECT / "mlflow.db"))
MLARTIFACTS = Path(os.getenv("TMD_MLARTIFACTS", PROJECT / "mlartifacts"))

SHL_RAW = RAW / "shl" / "release"
ES_RAW = RAW / "extrasensory"
