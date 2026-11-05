"""Live progress of long runs and of the task that launched them, in .progress/ (git-ignored).

Runs
----
In a script, declare the total units up front (model fits, ablation rows, epochs, ...) and mark each one done:

    from tmd.progress import Progress

    prog = Progress("06_gps-acc-baseline_waiis-bench", "H2_seed0", total=36, log=log)
    ...
    prog.step("inner A k=0 j=2")

Each step atomically rewrites .progress/<experiment>__<run>.json (experiment, run, done, total, started, updated,
pid) and logs "progress 7/36 ...". Nothing here touches randomness or results.

Tasks
-----
A long task (ML runs, or more than about 5 min of work) starts with a plan: its items with minute estimates.

    uv run python -m tmd.progress plan "Exp 11: leg grid" 11:gps_candidates=10 11:leg_grid=15 11:readme=5
    uv run python -m tmd.progress done 11:readme      # an item that is not a Progress run
    uv run python -m tmd.progress end                 # close the task early (plan changed, run given up)
    uv run python -m tmd.progress watch               # board, in a separate terminal pane

`exp` is an experiment number (resolved against experiments/) or a folder name. `plan` replaces
.progress/_task.json; only these commands write it. An item is a Progress run when a run file with the same
experiment and run name started after the plan was declared; its state (running, done, or stopped when the pid
died before the last step) comes from that file. Other items are pending until marked `done`.

Overall % = Σ fraction·estimate / Σ estimate; time left = the running items' own ETA (or estimate minus elapsed)
plus the estimates of the items not started. The task is complete when every item is done, or after `end`.

`watch` clears the pane when a new plan appears, redraws every 2 s, draws the completed state once at the end
and then stops updating until the next plan. Without any plan it shows the most recently updated run.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / ".progress"
RECENT_S = 30 * 60      # without a plan, a finished or stopped run is shown this long after its last update


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
        _write_json(self.path, self.state)


def _write_json(path: Path, d: dict) -> None:
    DIR.mkdir(exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def _ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def _hm(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%H:%M")


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


def _eta_s(s: dict) -> float | None:
    """Mean time per completed unit × units remaining; None before the first unit."""
    if s["done"] <= 0:
        return None
    return (_ts(s["updated"]) - _ts(s["started"])) / s["done"] * (s["total"] - s["done"])


def _timing(s: dict, now: float) -> str:
    """Elapsed, then ETA, 'done' or 'stopped'."""
    st = status(s)
    out = _dur((now if st == "running" else _ts(s["updated"])) - _ts(s["started"]))
    if st != "running":
        return f"{out} · {st}"
    eta = _eta_s(s)
    return out + (f" · ~{_dur(eta)} left" if eta is not None else "")


def _bar(frac: float, width: int) -> str:
    fill = round(width * min(max(frac, 0.0), 1.0))
    return "▓" * fill + "░" * (width - fill)


def _num(experiment: str) -> str:
    return experiment.split("_", 1)[0]


def run_line(s: dict, width: int = 10) -> str:
    """e.g. '06 · H2_seed0 ▓▓▓▓▓▓░░░░ 7/20 · 12m · ~22m left'."""
    frac = s["done"] / s["total"] if s["total"] else 1.0
    return (f"{_num(s['experiment'])} · {s['run']} {_bar(frac, width)} {s['done']}/{s['total']} · "
            f"{_timing(s, time.time())}")


def newest() -> dict | None:
    """State of the most recently updated run file, or None if there is nothing current."""
    files = []
    for f in DIR.glob("*.json"):
        if not f.name.startswith(("_", ".")):
            try:
                files.append((f.stat().st_mtime, f))
            except OSError:
                pass
    if not files:
        return None
    mtime, f = max(files)
    s = _read_json(f)
    try:
        if s is None or (status(s) != "running" and time.time() - mtime > RECENT_S):
            return None
    except (KeyError, TypeError):
        return None
    return s


# ---------------------------------------------------------------- task

def _task_path() -> Path:
    return DIR / "_task.json"


def read_task() -> dict | None:
    t = _read_json(_task_path())
    if t is None or not isinstance(t.get("items"), list) or not isinstance(t.get("started"), str):
        return None
    return t


def _resolve_exp(exp: str) -> str:
    if exp.isdigit():
        hits = sorted(p.name for p in (ROOT / "experiments").glob(f"{int(exp):02d}_*") if p.is_dir())
        if len(hits) != 1:
            raise SystemExit(f"experiment {exp!r}: {'no folder' if not hits else 'ambiguous: ' + ', '.join(hits)}")
        return hits[0]
    return exp


def _parse_item(spec: str) -> dict:
    """'11:leg_grid=15' -> item; the estimate may be omitted."""
    key, _, est = spec.partition("=")
    exp, sep, run = key.partition(":")
    if not sep or not exp or not run:
        raise SystemExit(f"bad item {spec!r}: expected <exp>:<item>=<minutes>")
    try:
        est_min = float(est) if est else None
    except ValueError:
        raise SystemExit(f"bad estimate in {spec!r}") from None
    return {"experiment": _resolve_exp(exp), "run": run, "est_min": est_min, "finished": None}


def _views(t: dict, now: float) -> list[dict]:
    """Per item: state (pending/running/done/stopped), done, total, elapsed and eta (s), end (timestamp)."""
    out = []
    for it in t["items"]:
        v = {"it": it, "state": "pending", "done": 0, "total": 0, "elapsed": None, "eta": None, "end": None}
        r = _read_json(DIR / f"{it['experiment']}__{it['run']}.json")
        try:
            if r is not None and _ts(r["started"]) >= _ts(t["started"]):
                st = status(r)
                end = now if st == "running" else _ts(r["updated"])
                v.update(state=st, done=r["done"], total=r["total"], elapsed=end - _ts(r["started"]),
                         eta=_eta_s(r) if st == "running" else None, end=None if st == "running" else end)
        except (KeyError, TypeError, ValueError):
            pass
        if it.get("finished") and v["state"] != "running":
            v["state"] = "done"
            v["end"] = _ts(it["finished"])
        out.append(v)
    return out


def summary(t: dict, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    views = _views(t, now)

    def frac(v):
        if v["state"] == "done":
            return 1.0
        return min(v["done"] / v["total"], 1.0) if v["total"] else 0.0

    sum_est = sum(v["it"].get("est_min") or 0 for v in views)
    if sum_est:
        pct = sum(frac(v) * (v["it"].get("est_min") or 0) for v in views) / sum_est
    else:
        pct = sum(frac(v) for v in views) / len(views) if views else 1.0
    left = 0.0
    for v in views:
        e = (v["it"].get("est_min") or 0) * 60
        if v["state"] == "running":
            left += max(v["eta"] if v["eta"] is not None else e - v["elapsed"], 0)
        elif v["state"] != "done":
            left += e * (1 - frac(v))
    complete = bool(t.get("ended")) or all(v["state"] == "done" for v in views)
    done_at = None
    if complete:
        ends = [v["end"] for v in views if v["end"] is not None] + ([_ts(t["ended"])] if t.get("ended") else [])
        done_at = max(ends) if ends else _ts(t["started"])
    return {"views": views, "pct": pct, "left": left, "sum_est": sum_est, "complete": complete, "done_at": done_at}


def plan(title: str, specs: list[str]) -> dict:
    """Declare a new task (replaces the current one)."""
    t = {"title": title, "started": _iso(time.time()), "ended": None, "items": [_parse_item(s) for s in specs]}
    _write_json(_task_path(), t)
    return t


def mark_done(specs: list[str]) -> dict:
    t = read_task()
    if t is None:
        raise SystemExit("no task: declare one with `plan`")
    for spec in specs:
        k = _parse_item(spec)
        it = next((i for i in t["items"] if (i["experiment"], i["run"]) == (k["experiment"], k["run"])), None)
        if it is None:
            raise SystemExit(f"{spec} is not in the plan")
        it["finished"] = _iso(time.time())
    _write_json(_task_path(), t)
    return t


def end() -> dict:
    t = read_task()
    if t is None:
        raise SystemExit("no task")
    t["ended"] = t.get("ended") or _iso(time.time())
    _write_json(_task_path(), t)
    return t


# ---------------------------------------------------------------- board

COL = {"red": "\033[31m", "green": "\033[32m", "dim": "\033[2m", "off": "\033[0m"}


def board(now: float | None = None, color: bool = False, width: int = 79) -> str:
    now = time.time() if now is None else now

    def col(s, c):
        return f"{COL[c]}{s}{COL['off']}" if color else s

    t = read_task()
    if t is None:
        s = newest()
        return f" {run_line(s) if s is not None else col('no task planned', 'dim')}\n"
    sm = summary(t, now)
    started = _ts(t["started"])
    exps = ", ".join(dict.fromkeys(_num(it["experiment"]) for it in t["items"]))
    if sm["complete"]:
        est = f" (est {_dur(sm['sum_est'] * 60)})" if sm["sum_est"] else ""
        right = f"✓ Done {_hm(sm['done_at'])} · {_dur(sm['done_at'] - started)}{est}"
    else:
        right = f"started {_hm(started)} · {_dur(now - started)}"
    title = f" Exp {exps} · {t['title']}" if exps else f" {t['title']}"
    pad = " " * max(width - len(title) - len(right), 2)
    rule = " " + "─" * (width - 2)
    out = [title + pad + (col(right, "green") if sm["complete"] else right), rule]
    multi = "," in exps
    names = [f"{_num(v['it']['experiment'])} {v['it']['run']}" if multi else v["it"]["run"] for v in sm["views"]]
    nw = max([len(n) for n in names] + [12]) + 2
    for v, name in zip(sm["views"], names):
        it, st, name = v["it"], v["state"], name.ljust(nw)
        e = (f"est {it['est_min']:>3g}m" if it.get("est_min") else "est   —").ljust(11)
        if st == "done":
            out.append(f" ✓ {name}{e} " + (f"took {_dur(v['elapsed'])}" if v["elapsed"] is not None else "done"))
        elif st == "running":
            eta = f" · ~{_dur(v['eta'])} left" if v["eta"] is not None else ""
            out.append(f" ▶ {name}{e} {_bar(v['done'] / v['total'], 10)} {v['done']}/{v['total']} · "
                       f"{_dur(v['elapsed'])}{eta}")
        elif st == "stopped":
            out.append(col(f" ✗ {name}{e} stopped at {v['done']}/{v['total']}", "red"))
        else:
            out.append(f" · {name}{e}")
    out.append(rule)
    tail = "" if sm["complete"] else f"   ~{_dur(sm['left'])} left · ≈{_hm(now + sm['left'])}"
    out.append(f" Overall {_bar(sm['pct'], 15)} {round(100 * sm['pct'])}%{tail}")
    return "\n".join(out) + "\n"


def frame(state: dict, color: bool = False, width: int = 79) -> str | None:
    """Next screen for `watch`, or None to leave it as it is. `state` carries the task between calls."""
    t = read_task()
    key = t["started"] if t is not None else None
    if key != state.get("key"):
        state.update(key=key, frozen=False)
        prefix = "\033[H\033[2J"                       # new task: clear and start anew
    elif state.get("frozen"):
        return None                                    # completed: no more updates
    else:
        prefix = "\033[H"
    text = board(color=color, width=width)
    state["frozen"] = t is not None and summary(t)["complete"]
    return prefix + text + "\033[J"


def watch(poll: float = 2.0) -> None:
    import shutil
    tty = sys.stdout.isatty()
    state: dict = {"key": ""}
    try:
        while True:
            out = frame(state, color=tty, width=min(shutil.get_terminal_size((80, 24)).columns, 100) - 1)
            if out is not None:
                sys.stdout.write(out)
                sys.stdout.flush()
            time.sleep(poll)
    except KeyboardInterrupt:
        pass


def _print_plan(t: dict) -> None:
    for it in t["items"]:
        est = f"{it['est_min']:g} min" if it.get("est_min") else "—"
        print(f"  {_num(it['experiment'])}:{it['run']:<28} {est}")
    print(f"  total ≈ {sum(it.get('est_min') or 0 for it in t['items']):g} min")


if __name__ == "__main__":
    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "plan" and rest:
        _print_plan(plan(rest[0], rest[1:]))
    elif cmd == "done" and rest:
        mark_done(rest)
    elif cmd == "end":
        end()
    elif cmd == "watch":
        if "--once" in rest:
            sys.stdout.write(board())
        else:
            watch()
    else:
        sys.exit('usage: python -m tmd.progress {plan "<title>" exp:item=min ... | done exp:item ... | end | watch}')
