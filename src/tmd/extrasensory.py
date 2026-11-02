"""Moved to `tmd.bench.sources.extrasensory` (experiment 09). This name stays importable: it is the same module object."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("tmd.bench.sources.extrasensory")
