"""Convert the zipped Kaggle datasets to Parquet in one streaming pass each (README, stop-point 1 answers, item 1).

    uv run python convert.py collecty [--workers 6]
    uv run python convert.py nor_tmd

Output: <data>/processed/<dataset>_events_*/ (see tmd.readers.collecty, tmd.readers.nor_tmd). Full log in logs/convert_<dataset>.log.
"""
import argparse
import logging
import shutil
import time
from pathlib import Path

from tmd.paths import DATA
from tmd.progress import Progress

HERE = Path(__file__).resolve().parent
EXP = HERE.name

log = logging.getLogger("convert")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["collecty", "nor_tmd"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    (HERE / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=HERE / "logs" / f"convert_{a.dataset}.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    t0 = time.time()
    if a.dataset == "collecty":
        from tmd.readers import collecty as c
        jobs = c.members()
        prog = Progress(EXP, "convert_collecty", total=len(jobs), log=log)
        files = c.convert(a.workers, progress=lambda r: prog.step(f"{r['user']}_{r['traj']}"))
        out = c.OUT
        summary = f"{len(files)} files, {files.rows.sum():,} rows, {files.acc_events.sum():,} acc events"
    else:
        from tmd.readers import nor_tmd as n
        prog = Progress(EXP, "convert_nor_tmd", total=n.ZIP_MEMBER_BYTES // n.CHUNK_BYTES + 1, log=log)
        summary = n.convert(progress=prog.step)
        out = n.OUT
    size = sum(p.stat().st_size for p in out.rglob("*.parquet"))
    free = shutil.disk_usage(DATA).free
    msg = f"{a.dataset}: {summary}; {size / 1e9:.2f} GB Parquet in {out}; {free / 1e9:.1f} GB free; {time.time() - t0:.0f} s"
    log.info(msg)
    print(msg)


if __name__ == "__main__":
    main()
