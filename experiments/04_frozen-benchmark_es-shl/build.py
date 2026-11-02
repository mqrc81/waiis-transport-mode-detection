"""Build the frozen benchmark waiis_bench_v1 (04 README). The build code lives in `tmd.bench.build` (moved there in
experiment 09); this driver keeps 04's command:

    uv run python build.py [--workers 8] [--sources extrasensory shl_user1 shl_user23] [--tables-dir DIR] [--log-dir DIR]
"""
from tmd.bench.build import *  # noqa: F401,F403  (04's names stay importable from here)
from tmd.bench.build import main

if __name__ == "__main__":
    main()
