"""Experiment 11, dev pieces: moved to `tmd.models.e10` in 12 (2026-11-05; leg scores checked unchanged, 12's
README). This module keeps 11's folder constants and re-exports the moved helpers so 11's scripts run unchanged."""
from pathlib import Path

from tmd.models.e10 import *
from tmd.models.e10 import PROCESSED

HERE = Path(__file__).resolve().parent
EXP = HERE.name
OUT = PROCESSED / EXP
