"""Live progress of long runs: a small JSON file per run in .progress/ (git-ignored) plus one line in the run's log.

In a script, declare the total units up front (model fits, ablation rows, epochs, ...) and mark each one done:

    from tmd.progress import Progress

    prog = Progress("06_gps-acc-baseline_waiis-bench", "H2_seed0", total=36, log=log)
    ...
    prog.step("inner A k=0 j=2")

Each step atomically rewrites .progress/<experiment>__<run>.json (experiment, run, done, total, started, updated,
pid) and logs "progress 7/36 ...". Nothing here touches randomness or results.

Read it back:

    uv run python -m tmd.progress watch      # live tqdm bar, polls the newest file every 2 s
    .venv/bin/python src/tmd/progress.py line # one line for the Claude Code status line (stdlib only, fast)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

DIR = Path(__file__).resolve().parents[2] / ".progress"     # repo root (no tmd import: the status line stays fast)
RECENT_S = 30 * 60      # a finished or stopped run is shown this long after its last update, then nothing


class Progress:
    def __init__(self, experiment: str, run: str, total: int, log: logging.Logger | None = None):
        self.path = DIR / f"{experiment}__{run}.json"
        self.log = log or logging.getLogger()
        self.t0 = time.time()
        self.state = {"experiment": experiment, "run": run, "done": 0, "total": int(total),
                      "started": _iso(self.t0), "updated": _iso(self.t0), "pid": os.getpid()}
        self._write()

    def step(self, what: str = "", n: int = 1) -> None:
        """Mark n units done; `what` names the unit in the log line."""
        s = self.state
        s["done"] += n
        s["updated"] = _iso(time.time())
        self._write()
        self.log.info("progress %s/%s %s · %s", s["done"], s["total"], what, _timing(s, time.time()))

    def _write(self) -> None:
        DIR.mkdir(exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.state))
        os.replace(tmp, self.path)


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def _ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def _dur(sec: float) -> str:
    sec = int(max(sec, 0))
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m"
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def status(s: dict) -> str:
    """'done', 'stopped' (pid gone before done == total) or 'running'."""
    if s["done"] >= s["total"]:
        return "done"
    return "running" if _alive(s["pid"]) else "stopped"


def _timing(s: dict, now: float) -> str:
    """Elapsed, then ETA (mean time per completed unit × units remaining), 'done' or 'stopped'."""
    st = status(s)
    end = now if st == "running" else _ts(s["updated"])
    out = _dur(end - _ts(s["started"]))
    if st != "running":
        return f"{out} · {st}"
    if s["done"] > 0:
        per = (_ts(s["updated"]) - _ts(s["started"])) / s["done"]
        out += f" · ~{_dur(per * (s['total'] - s['done']))} left"
    return out


def newest() -> dict | None:
    """State of the most recently updated progress file, or None if there is nothing current."""
    try:
        files = [(f.stat().st_mtime, f) for f in DIR.glob("*.json")]
    except OSError:
        return None
    if not files:
        return None
    mtime, f = max(files)
    try:
        s = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    if status(s) != "running" and time.time() - mtime > RECENT_S:
        return None
    return s


def label(s: dict) -> str:
    return f"{s['experiment'].split('_', 1)[0]} · {s['run']}"


def line(s: dict, width: int = 10) -> str:
    """e.g. '06 · H2_seed0 ▓▓▓▓▓▓░░░░ 7/20 · 12m · ~22m left'."""
    fill = round(width * s["done"] / s["total"]) if s["total"] else width
    return f"{label(s)} {'▓' * fill}{'░' * (width - fill)} {s['done']}/{s['total']} · {_timing(s, time.time())}"


def watch(poll: float = 2.0) -> None:
    from tqdm import tqdm
    bar, key = None, None
    try:
        while True:
            s = newest()
            if s is not None:
                k = (s["experiment"], s["run"], s["started"])
                if k != key:
                    if bar is not None:
                        bar.close()
                    bar = tqdm(total=s["total"], desc=label(s), unit="", dynamic_ncols=True,
                               bar_format="{desc} {bar} {n}/{total} · {unit}")
                    key = k
                bar.total, bar.n = s["total"], s["done"]
                bar.unit = _timing(s, time.time())          # free text slot of bar_format (postfix adds ", ")
                bar.refresh()
            time.sleep(poll)
    except KeyboardInterrupt:
        pass
    finally:
        if bar is not None:
            bar.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "line":
        s = newest()
        if s is not None:
            print(line(s))
    elif cmd == "watch":
        watch()
    else:
        sys.exit("usage: python -m tmd.progress {watch,line}")
