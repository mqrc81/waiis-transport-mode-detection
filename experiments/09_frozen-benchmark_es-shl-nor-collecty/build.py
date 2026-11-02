"""Build the frozen benchmark waiis_bench_v2 (09 README). One command rebuilds everything:

    uv run python build.py [--workers 4]

Code: tmd.bench.build. v1's code writes ExtraSensory and SHL into the v2 root, `extend_v1_sources_v2` adds v2's
columns and rules, `build_new` builds NOR-TMD and Collecty through tmd.candidates, and the manifest goes to
<data>/processed/waiis_bench_v2/manifest.json with a committed copy in results/manifests/. Count tables:
results/tables/bench_v2/; full log: logs/build.log; stdout gets a short summary. Build at a clean commit.
"""
import argparse
import logging
import shutil
import time
from pathlib import Path

from tmd.bench import build as b
from tmd.progress import Progress

HERE = Path(__file__).resolve().parent
EXP = HERE.name
TABLES = HERE / "results" / "tables" / "bench_v2"     # not "build/": the .gitignore template ignores it


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    commit, dirty = b.git()             # before the build rewrites its own results/ and logs/
    (HERE / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=HERE / "logs" / "build.log", filemode="w", level=logging.INFO,
                        format="%(message)s")
    b.configure_v2(TABLES)
    prog = Progress(EXP, "build_v2", total=6, log=b.log)
    t0 = time.time()
    v0 = b.calibrate_blanking(a.workers)
    b.build_es(a.workers)
    prog.step("extrasensory (v1 code)")
    for s in ("shl_user1", "shl_user23"):
        b.build_shl(s, a.workers, v0)
    prog.step("shl (v1 code, + Users 2-3 other positions)")
    b.extend_v1_sources_v2()
    prog.step("v2 columns, sign conversion, rate rule, position draw")
    for s in b.NEW_SOURCES:
        b.build_new(s, a.workers)
        prog.step(s)
    b.write_manifest_v2(commit, dirty, v0)
    (HERE / "results" / "manifests").mkdir(parents=True, exist_ok=True)
    shutil.copy(b.ROOT / "manifest.json", HERE / "results" / "manifests" / f"{b.V2}.json")
    prog.step("manifest")
    b.say(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
