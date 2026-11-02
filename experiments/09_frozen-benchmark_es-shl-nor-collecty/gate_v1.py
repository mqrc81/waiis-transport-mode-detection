"""Gate for the tmd.bench reorganisation (README, stop-point 1 answers, item 1): v1 must not move.

    uv run python gate_v1.py --scratch DIR [--skip-chain]

DIR holds a scratch data root DIR/data (raw = a symlink to the real raw data) into which v1 was rebuilt with the
moved code:
    TMD_DATA=DIR/data uv run python experiments/04_frozen-benchmark_es-shl/build.py --workers 4 \\
        --tables-dir DIR/tables --log-dir DIR/logs
Checks, in order (all must pass):
1. `tmd.bench.verify()` on the real v1 is empty.
2. Every data file of the rebuild has the sha256 of `04/results/manifests/waiis_bench_v1.json`, and the rebuild's
   count tables equal 04's committed `results/tables/` byte for byte.
3. 06's chain to its final dev configuration, recomputed from the rebuilt v1 through the moved code with a scratch
   MLflow store (05: features, B, F_fb; 06: gps-v2 features, A, G_v2, H1_v2; seed 0, the recorded commands without
   `--vs`). Every intermediate and **H1_v2's out-of-fold probabilities** must equal the saved files exactly.
The chain rewrites 05's and 06's committed logs; restore them with git afterwards (listed at the end).
Full output: logs/gate_v1.log.
"""
import argparse
import filecmp
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tmd import bench
from tmd.paths import PROCESSED, PROJECT

HERE = Path(__file__).resolve().parent
E04, E05, E06 = (PROJECT / "experiments" / n for n in ("04_frozen-benchmark_es-shl", "05_acc-baseline_waiis-bench",
                                                      "06_gps-acc-baseline_waiis-bench"))
CHAIN = [  # (folder, command) as recorded in 05's and 06's READMEs, seed 0, without --vs
    (E05, ["make_features.py", "--workers", "4"]),
    (E05, ["cv.py", "--name", "B", "--sub"]),
    (E05, ["cv.py", "--name", "F_fb", "--from-row", "B", "--hmm", "fb"]),
    (E06, ["make_gps.py"]),                                   # gps-v1 (row A reads it), as in 06's README
    (E06, ["make_gps.py", "--version", "gps-v2"]),
    (E06, ["run.py", "--row", "A"]),
    (E06, ["run.py", "--row", "G", "--name", "G_v2", "--gps", "gps-v2", "--lr", "0.02", "--acc-row", "A"]),
    (E06, ["run.py", "--row", "H1", "--name", "H1_v2", "--gps", "gps-v2", "--acc-row", "A", "--gps-row", "G_v2"]),
]
OOF = [("05_acc-baseline_waiis-bench", "B_seed0"), ("05_acc-baseline_waiis-bench", "F_fb_seed0"),
       ("06_gps-acc-baseline_waiis-bench", "A_seed0"), ("06_gps-acc-baseline_waiis-bench", "G_v2_seed0"),
       ("06_gps-acc-baseline_waiis-bench", "H1_v2_seed0")]
FEATURES = [("05_acc-baseline_waiis-bench", "extrasensory_dev_25hz_w15s_acc-v1_lp-none.parquet"),
            ("06_gps-acc-baseline_waiis-bench", "extrasensory_dev_gps-v1_w2min.parquet"),
            ("05_acc-baseline_waiis-bench", "shl_user1_dev_25hz_w15s_acc-v1_lp-none.parquet"),
            ("06_gps-acc-baseline_waiis-bench", "extrasensory_dev_gps-v2_w2min.parquet"),
            ("06_gps-acc-baseline_waiis-bench", "shl_user1_dev_gps-v2_w2min.parquet")]
NOT_BUILD_TABLES = {"decision_details.csv"}    # written by 04's register_numbers.py, not by the build
LOG = []


def say(msg: str) -> None:
    LOG.append(msg)
    print(msg, flush=True)


def check_rebuild(scratch: Path) -> bool:
    ref = bench.reference_manifest("waiis_bench_v1")["files"]
    root = scratch / "data" / "processed" / "waiis_bench_v1"
    got = {str(p.relative_to(root)): bench.sha256(p) for p in sorted(root.rglob("*.parquet"))}
    bad = [f for f in ref if got.get(f) != ref[f]["sha256"]] + sorted(set(got) - set(ref))
    say(f"2a rebuild: {len(got)} data files, {len(ref)} in the reference; mismatched or extra: {bad or 'none'}")
    t_ref, t_new = E04 / "results" / "tables", scratch / "tables"
    committed = {p.name for p in t_ref.glob("*.csv")} - NOT_BUILD_TABLES
    names = sorted(committed)
    diff = [n for n in names if not (t_new / n).exists() or not filecmp.cmp(t_ref / n, t_new / n, shallow=False)]
    extra = sorted({p.name for p in t_new.glob("*.csv")} - committed)
    say(f"2b tables written by the build: {len(names)} committed, differing or missing: {diff or 'none'}, "
        f"extra in rebuild: {extra or 'none'} (not compared, written by other scripts: {sorted(NOT_BUILD_TABLES)})")
    return not bad and not diff and not extra


def run_chain(scratch: Path, start: int = 0) -> None:
    env = {**os.environ, "TMD_DATA": str(scratch / "data"), "TMD_MLFLOW_DB": str(scratch / "mlflow.db"),
           "TMD_MLARTIFACTS": str(scratch / "mlartifacts"), "MLFLOW_DISABLE_AGENT_HINT": "1"}
    for i, (folder, cmd) in enumerate(CHAIN):
        if i < start:
            say(f"3 chain [{i}] (done before, not rerun): {folder.name}: {' '.join(cmd)}")
            continue
        say(f"3 chain [{i}]: {folder.name}: {' '.join(cmd)}")
        r = subprocess.run([sys.executable, *cmd], cwd=folder, env=env, capture_output=True, text=True)
        (scratch / "logs" / f"chain_{folder.name[:2]}_{cmd[0][:-3]}_{'_'.join(cmd[1:3])}.out").write_text(
            r.stdout + r.stderr)
        if r.returncode:
            raise SystemExit(f"chain step failed ({r.returncode}): {' '.join(cmd)}; see {scratch / 'logs'}")


def check_chain(scratch: Path) -> bool:
    ok = True
    new_root = scratch / "data" / "processed"
    for exp, f in FEATURES:
        a, b = pd.read_parquet(PROCESSED / exp / f), pd.read_parquet(new_root / exp / f)
        same = a.shape == b.shape and a.equals(b)
        say(f"3a features {exp[:2]} {f}: {a.shape} vs {b.shape}, identical {same}")
        ok &= same
    for exp, name in OOF:
        a, b = np.load(PROCESSED / exp / "oof" / f"{name}.npz", allow_pickle=True), \
            np.load(new_root / exp / "oof" / f"{name}.npz", allow_pickle=True)
        for k in sorted(set(a.files) & set(b.files)):
            if a[k].dtype == object or a[k].dtype.kind in "US":
                same = np.array_equal(a[k], b[k])
            else:
                same = a[k].shape == b[k].shape and np.array_equal(a[k], b[k], equal_nan=True)
            if k in ("p", "raw") or not same:
                say(f"3b oof {exp[:2]} {name}[{k}]: identical {same}"
                    + ("" if same or a[k].dtype.kind not in "fc" else
                       f" (max abs diff {np.nanmax(np.abs(a[k] - b[k])):.3g})"))
            if k in ("p", "raw", "user", "ts"):
                ok &= same
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--skip-chain", action="store_true")
    ap.add_argument("--start", type=int, default=0, help="first chain step to run (earlier ones already ran)")
    a = ap.parse_args()
    v = bench.verify("waiis_bench_v1")
    say(f"1 verify(v1): {v or 'intact'}")
    passed = not v and check_rebuild(a.scratch)
    if not a.skip_chain:
        run_chain(a.scratch, a.start)
        passed &= check_chain(a.scratch)
    say(f"GATE {'PASSED' if passed else 'FAILED'}")
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / "gate_v1.log").write_text("\n".join(LOG) + "\n")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(E05), str(E06)], capture_output=True,
                           text=True, cwd=PROJECT).stdout.strip()
    if dirty:
        print("rewritten by the chain (restore with git checkout -- <files>):\n" + dirty)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
