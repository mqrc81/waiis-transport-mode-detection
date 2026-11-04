"""Live progress of long runs and of the Claude task that launched them, in .progress/ (git-ignored).

Runs
----
In a script, declare the total units up front (model fits, ablation rows, epochs, ...) and mark each one done:

    from tmd.progress import Progress

    prog = Progress("06_gps-acc-baseline_waiis-bench", "H2_seed0", total=36, log=log)
    ...
    prog.step("inner A k=0 j=2")

Each step atomically rewrites .progress/<experiment>__<run>.json (experiment, run, done, total, started, updated,
pid) and logs "progress 7/36 ...". Nothing here touches randomness or results, and Progress sends no notifications.

Tasks
-----
A task is one request to Claude: it may launch several runs and ends when Claude stops and waits for input.
.progress/_task.json holds the current task: the turn state (working / waiting / permission, set by Claude Code
hooks), the plan (items with minute estimates, plan membership, skip marks) and whether "done" was notified.
Finished tasks with a plan are archived to .progress/_tasks/<started>.json (used by `history`).

- Item state is derived on read: the per-run file is the source of truth for done/total and pid; _task.json only
  stores plan membership, estimates, skip marks and when a run was linked.
- A run links itself to the task only while a task is active: _task.json exists and Claude is mid-turn (working or
  at a permission prompt), the turn state changed < 10 min ago, or a sentinel is watching background runs. Otherwise
  Progress leaves _task.json alone. Planned runs are marked running (and done on the last step); others are
  appended as unplanned.
- Every write of _task.json goes through _update_task: flock on .progress/_task.lock, read, modify, write a temp
  file, os.replace, all inside the lock (2 s timeout, then the update is skipped with a warning). Readers don't lock.
- Hooks (.claude/settings.local.json): UserPromptSubmit -> working (a finished task is archived and a new one
  started; a task with pending items continues); Stop -> waiting, notify "done" when nothing runs any more and the
  turn took >= 2 min or a run finished or stopped during it; while runs still go, start one detached sentinel that
  polls every 15 s, notifies each run that stops before its last step and the task's end, and exits when Claude
  works again or after 12 h; Notification(permission_prompt) -> permission; PostToolUse -> back to working after an
  approved prompt (its command greps _task.json for the exact text '"turn": "permission"' before starting Python).
  At most one "done" notification per turn end (`notified`, reset on each prompt); one "stopped" notification per
  stopped item.
- Limitation: `plan` runs from Claude's Bash tool, which has no session id; it attaches to whatever _task.json the
  last UserPromptSubmit wrote. That is fine with one Claude session per repo; several sessions are not supported.

CLI (uv run python -m tmd.progress ...)
---------------------------------------
    plan "<title>" 11:leg_grid=15 11:test=5   # declare/replace the plan (exp = number or folder name)
    plan add 11:extra=5 ...                    # add items;  plan skip 11:test [reason]
    history [pattern]                          # median minutes of past finished runs matching pattern
    watch [--once]                             # board for a separate terminal pane (redraws every 2 s)
    hook <event>                               # called by Claude Code hooks; never fails, prints nothing
    sentinel                                   # started by the Stop hook

The hooks and sentinel run this file directly with stdlib only (no tmd import), so they start fast.
TMD_PROGRESS_DIR overrides the directory and TMD_PROGRESS_POLL the sentinel poll interval (both for tests).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]                  # repo root (no tmd import: the hooks stay fast)
DIR = Path(os.environ.get("TMD_PROGRESS_DIR") or ROOT / ".progress")
RECENT_S = 30 * 60      # a finished or stopped run (or task) is shown this long after its last update, then nothing
ACTIVE_S = 10 * 60      # a run links to the task if the turn state changed less than this long ago
SHORT_TURN_S = 120      # turns shorter than this without a plan don't notify
LOCK_TIMEOUT_S = 2.0
SENTINEL_MAX_S = 12 * 3600
C_CLAMP = (0.5, 3.0)
TITLE = "Claude · tmd"


# ---------------------------------------------------------------- runs

class Progress:
    def __init__(self, experiment: str, run: str, total: int, log=None):
        import logging
        self.path = DIR / f"{experiment}__{run}.json"
        self.log = log or logging.getLogger()
        self.t0 = time.time()
        self.state = {"experiment": experiment, "run": run, "done": 0, "total": int(total),
                      "started": _iso(self.t0), "updated": _iso(self.t0), "pid": os.getpid()}
        self._write()
        self._task = _link_run(experiment, run, self.state["started"])    # task_started if linked, else None
        if self._task is not None and self.state["total"] <= 0:
            self._finish()

    def step(self, what: str = "", n: int = 1) -> None:
        """Mark n units done; `what` names the unit in the log line."""
        s = self.state
        was_done = s["done"] >= s["total"]
        s["done"] += n
        s["updated"] = _iso(time.time())
        self._write()
        self.log.info("progress %s/%s %s · %s", s["done"], s["total"], what, _timing(s, time.time()))
        if self._task is not None and not was_done and s["done"] >= s["total"]:
            self._finish()

    def _finish(self) -> None:
        key, started, task_started = (self.state["experiment"], self.state["run"]), self.state["started"], self._task

        def fn(t):
            if t is None or t.get("task_started") != task_started:
                return None
            it = _find(t, key)
            if it is None or it.get("started") != started:
                return None
            it.update(state="done", finished=_iso(time.time()))
            return t
        _update_task(fn)

    def _write(self) -> None:
        DIR.mkdir(exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.state))
        os.replace(tmp, self.path)


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


def _alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
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
    """Mean time per completed unit × units remaining, or None before the first unit."""
    if s["done"] <= 0:
        return None
    per = (_ts(s["updated"]) - _ts(s["started"])) / s["done"]
    return per * (s["total"] - s["done"])


def _timing(s: dict, now: float) -> str:
    """Elapsed, then ETA, 'done' or 'stopped'."""
    st = status(s)
    end = now if st == "running" else _ts(s["updated"])
    out = _dur(end - _ts(s["started"]))
    if st != "running":
        return f"{out} · {st}"
    eta = _eta_s(s)
    if eta is not None:
        out += f" · ~{_dur(eta)} left"
    return out


def _read_json(p: Path):
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _run_files():
    try:
        return [f for f in DIR.glob("*.json") if not f.name.startswith(("_", "."))]
    except OSError:
        return []


def newest() -> dict | None:
    """State of the most recently updated progress file, or None if there is nothing current."""
    files = []
    for f in _run_files():
        try:
            files.append((f.stat().st_mtime, f))
        except OSError:
            pass
    if not files:
        return None
    mtime, f = max(files)
    s = _read_json(f)
    if s is None:
        return None
    try:
        if status(s) != "running" and time.time() - mtime > RECENT_S:
            return None
    except (KeyError, TypeError):
        return None
    return s


def _num(experiment: str) -> str:
    return experiment.split("_", 1)[0]


def _bar(frac: float, width: int) -> str:
    fill = round(width * min(max(frac, 0.0), 1.0))
    return "▓" * fill + "░" * (width - fill)


def _run_row(s: dict, width: int = 10) -> str:
    """Single-run view, e.g. '06 · H2_seed0 ▓▓▓▓▓▓░░░░ 7/20 · 12m · ~22m left'."""
    frac = s["done"] / s["total"] if s["total"] else 1.0
    return (f"{_num(s['experiment'])} · {s['run']} {_bar(frac, width)} {s['done']}/{s['total']} · "
            f"{_timing(s, time.time())}")


# ---------------------------------------------------------------- task file

def _task_path() -> Path:
    return DIR / "_task.json"


def _read_task() -> dict | None:
    t = _read_json(_task_path())
    if t is None or not isinstance(t.get("items"), list):
        return None
    return t


def _log(msg: str, name: str = "_hooks.log") -> None:
    try:
        DIR.mkdir(exist_ok=True)
        with open(DIR / name, "a") as f:
            f.write(f"{_iso(time.time())} [{os.getpid()}] {msg}\n")
    except OSError:
        pass


def _update_task(fn):
    """Read _task.json, apply fn, write it back, all under an exclusive flock on _task.lock.

    fn gets the current task (None if missing or corrupt) and returns the task to write, or None to leave the file
    untouched. Returns the written task (or None). If the lock isn't free within LOCK_TIMEOUT_S the update is
    skipped with a warning, so a run never blocks on it.
    """
    import fcntl
    DIR.mkdir(exist_ok=True)
    with open(DIR / "_task.lock", "a+") as lk:
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    import logging
                    logging.getLogger("tmd.progress").warning("tmd.progress: _task.lock busy, update skipped")
                    _log("lock busy, update skipped")
                    return None
                time.sleep(0.002)
        try:
            new = fn(_read_task())
            if new is None:
                return None
            p = _task_path()
            tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(new, indent=1))
            os.replace(tmp, p)
            return new
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _new_task(session: str, now: float) -> dict:
    return {"session": session, "title": "", "turn": "working", "turn_since": _iso(now), "prompt_at": _iso(now),
            "task_started": _iso(now), "stop_at": None, "items": [], "notified": False, "sentinel_pid": None}


def _active(t: dict | None, now: float) -> bool:
    if t is None:
        return False
    if t.get("turn") in ("working", "permission") or t.get("sentinel_pid"):
        return True
    try:
        return now - _ts(t["turn_since"]) < ACTIVE_S
    except (KeyError, TypeError, ValueError):
        return False


def _find(t: dict, key) -> dict | None:
    for it in t["items"]:
        if (it.get("experiment"), it.get("run")) == key:
            return it
    return None


def _link_run(experiment: str, run: str, started: str) -> str | None:
    """Mark a planned item running or append an unplanned one, if a task is active. Returns task_started."""
    try:
        if not _active(_read_task(), time.time()):          # cheap unlocked check: no task, no lock, no write
            return None
        out = []

        def fn(t):
            if not _active(t, time.time()):
                return None
            it = _find(t, (experiment, run))
            if it is None:
                it = {"experiment": experiment, "run": run, "est_min": None, "planned": False}
                t["items"].append(it)
            it.update(state="running", started=started, finished=None)
            it.pop("reason", None)
            it.pop("crash_notified", None)
            out.append(t["task_started"])
            return t
        _update_task(fn)
        return out[0] if out else None
    except Exception as e:                                   # noqa: BLE001 — never break a run over the board
        import logging
        logging.getLogger("tmd.progress").warning("tmd.progress: task link failed: %r", e)
        return None


# ---------------------------------------------------------------- derived task view

def _view(it: dict, now: float) -> dict:
    """Derived state of one item: state, done, total, run (file dict), actual (s), eta (s), busy."""
    v = {"it": it, "state": it.get("state", "pending"), "done": 0, "total": 0, "run": None, "actual": None,
         "eta": None, "busy": False}
    if v["state"] == "skipped":
        return v
    if not it.get("started"):
        v["state"] = "pending"
        return v
    r = _read_json(DIR / f"{it['experiment']}__{it['run']}.json")
    try:
        if r is not None and _ts(r["started"]) >= _ts(it["started"]) - 1:
            v.update(run=r, done=r["done"], total=r["total"], state=status(r))
            end = now if v["state"] == "running" else _ts(r["updated"])
            v["actual"] = end - _ts(r["started"])
            if v["state"] == "running":
                v["eta"] = _eta_s(r)
                v["busy"] = True
            elif v["state"] == "done":
                # the script that ran it may start the next run of the plan: keep the task going while it lives
                v["busy"] = _alive(r.get("pid")) and now - _ts(r["updated"]) < RECENT_S
            return v
    except (KeyError, TypeError, ValueError):
        pass
    if v["state"] == "done" and it.get("finished"):
        v["actual"] = _ts(it["finished"]) - _ts(it["started"])
    elif v["state"] == "running":
        v["state"] = "stopped"                               # linked but its run file is gone
    return v


def summary(t: dict, now: float | None = None) -> dict:
    """Overall % (Σ fraction·est / Σ est over planned items with an estimate), calibration c and ETA."""
    now = time.time() if now is None else now
    views = [_view(it, now) for it in t["items"]]
    live = [v for v in views if v["state"] != "skipped"]
    planned = [v for v in live if v["it"].get("planned")]
    est = [v for v in planned if v["it"].get("est_min")]

    def frac(v):
        if v["state"] == "done":
            return 1.0
        if v["state"] in ("running", "stopped") and v["total"]:
            return min(v["done"] / v["total"], 1.0)
        return 0.0

    sum_est = sum(v["it"]["est_min"] for v in est)
    if sum_est > 0:
        pct = sum(frac(v) * v["it"]["est_min"] for v in est) / sum_est
    elif planned:
        pct = sum(frac(v) for v in planned) / len(planned)
    else:
        pct = None
    done_est = [v for v in est if v["state"] == "done" and v["actual"] is not None]
    c = None
    if done_est:
        c = sum(v["actual"] for v in done_est) / 60 / sum(v["it"]["est_min"] for v in done_est)
        c = min(max(c, C_CLAMP[0]), C_CLAMP[1])
    cc = 1.0 if c is None else c
    running = [v for v in live if v["state"] == "running"]
    pending = [v for v in live if v["state"] == "pending"]
    run_eta = []
    for v in running:
        if v["eta"] is not None:
            run_eta.append(v["eta"])
        elif v["it"].get("est_min"):
            run_eta.append(max(cc * v["it"]["est_min"] * 60 - (v["actual"] or 0), 0))
    eta = (max(run_eta) if run_eta else 0) + cc * 60 * sum(v["it"].get("est_min") or 0 for v in pending)
    has_eta = bool(running or pending)
    finished = not running and not pending and not any(v["busy"] for v in views)
    done_at = None
    if finished:
        ends = [_ts(t["stop_at"])] if t.get("stop_at") else []
        for v in live:
            if v["run"] is not None:
                ends.append(_ts(v["run"]["updated"]))
            elif v["it"].get("finished"):
                ends.append(_ts(v["it"]["finished"]))
        done_at = max(ends) if ends else None
    return {"views": views, "live": live, "planned": planned, "pct": pct, "c": c, "eta": eta if has_eta else None,
            "finish_at": now + eta if has_eta else None, "running": running, "pending": pending,
            "n_done": sum(v["state"] == "done" for v in live), "n_stopped": sum(v["state"] == "stopped" for v in live),
            "n_items": len(live), "sum_est": sum_est, "finished": finished, "done_at": done_at,
            "busy": any(v["busy"] for v in views)}


def _exps(t: dict) -> str:
    nums = []
    for it in t["items"]:
        n = _num(it["experiment"])
        if n not in nums:
            nums.append(n)
    return ", ".join(nums)


def _name(v: dict, multi: bool) -> str:
    it = v["it"]
    return f"{_num(it['experiment'])} {it['run']}" if multi else it["run"]


def _task_dur(t: dict, sm: dict) -> str:
    try:
        d = _dur((sm["done_at"] or time.time()) - _ts(t["task_started"]))
    except (KeyError, TypeError, ValueError):
        return ""
    return d + (f" (est {_dur(sm['sum_est'] * 60)})" if sm["sum_est"] else "")


def _turn_s(t: dict, now: float) -> float:
    """How long the current turn has run (since the last prompt); 0 if unknown."""
    try:
        return now - _ts(t.get("prompt_at") or "")
    except (TypeError, ValueError):
        return 0.0


def _changed_in_turn(t: dict, sm: dict) -> bool:
    """A run finished (last update since the prompt) or stopped (not yet reported at an earlier Stop) this turn."""
    try:
        p0 = _ts(t.get("prompt_at") or "")
    except (TypeError, ValueError):
        p0 = None
    for v in sm["views"]:
        if v["state"] == "stopped" and not v["it"].get("crash_notified"):
            return True
        if v["state"] == "done" and p0 is not None:
            end = v["run"]["updated"] if v["run"] is not None else v["it"].get("finished")
            try:
                if end and _ts(end) >= p0:
                    return True
            except (TypeError, ValueError):
                pass
    return False


def _done_message(t: dict, sm: dict, now: float) -> str:
    """Message for the end of a turn (or of the background runs) with nothing running any more."""
    if not t["items"]:
        return f"Claude done · {_dur(_turn_s(t, now))}"
    exp = f"Exp {_exps(t)}"
    stopped = f" · {sm['n_stopped']} stopped" if sm["n_stopped"] else ""
    if sm["pending"]:
        n = len(sm["pending"])
        return f"{exp} · Waiting at stop-point · {n} run{'s' * (n != 1)} left in plan{stopped}"
    mins = (sm["done_at"] or now) - _ts(t["task_started"])
    est = f" (est {sm['sum_est']:g})" if sm["sum_est"] else ""
    took = f"{round(mins / 60)} min" if mins >= 60 else _dur(mins)
    return f"{exp} done · {sm['n_done']}/{sm['n_items']} runs · {took}{est}{stopped}"


def _stop_message(v: dict) -> str:
    it = v["it"]
    return f"✗ {_num(it['experiment'])} {it['run']} stopped at {v['done']}/{v['total']}"


def notify(msg: str) -> None:
    _log(msg, "_notify.log")
    if sys.platform != "darwin":
        return
    import subprocess
    q = msg.replace("\\", "\\\\").replace('"', '\\"')
    try:
        subprocess.run(["osascript", "-e", f'display notification "{q}" with title "{TITLE}" sound name "Glass"'],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                       check=False)
    except (OSError, subprocess.SubprocessError) as e:
        _log(f"notify failed: {e!r}")


# ---------------------------------------------------------------- hooks and sentinel

def _archive(t: dict, now: float) -> None:
    sm = summary(t, now)
    snap = dict(t, archived=_iso(now), items=[dict(v["it"], state=v["state"], done=v["done"], total=v["total"],
                                                   actual_min=None if v["actual"] is None else round(v["actual"] / 60, 2))
                                              for v in sm["views"]])
    d = DIR / "_tasks"
    d.mkdir(exist_ok=True)
    name = t["task_started"].replace(":", "").replace("-", "")
    tmp = d / f".{name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(snap, indent=1))
    os.replace(tmp, d / f"{name}.json")


def _hook_prompt(data: dict, now: float) -> None:
    sid = str(data.get("session_id") or "")

    def fn(t):
        if t is not None:
            sm = summary(t, now)
            if sm["finished"] and (not t["items"] or t.get("stop_at")):
                if t["items"]:
                    _archive(t, now)
                t = None
        if t is None:
            t = _new_task(sid, now)
        t.update(session=sid or t.get("session", ""), turn="working", turn_since=_iso(now), prompt_at=_iso(now),
                 notified=False)
        return t
    _update_task(fn)


def _spawn_sentinel() -> int:
    import subprocess
    DIR.mkdir(exist_ok=True)
    with open(DIR / "_sentinel.log", "a") as out:
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "sentinel"], cwd=str(ROOT),
                             stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    return p.pid


def _hook_stop(data: dict, now: float) -> None:
    msgs = []

    def fn(t):
        if t is None:
            t = _new_task(str(data.get("session_id") or ""), now)
            t["prompt_at"] = None
        t.update(turn="waiting", turn_since=_iso(now), stop_at=_iso(now))
        sm = summary(t, now)
        worth = _turn_s(t, now) >= SHORT_TURN_S or _changed_in_turn(t, sm)    # before marking stops as seen
        for v in sm["views"]:                    # Claude saw these during its turn: no separate crash message
            if v["state"] == "stopped":
                v["it"]["crash_notified"] = True
        if sm["running"] or sm["busy"]:
            if not _alive(t.get("sentinel_pid")):
                t["sentinel_pid"] = _spawn_sentinel()
        elif worth and not t.get("notified"):
            t["notified"] = True
            msgs.append(_done_message(t, sm, now))
        return t
    _update_task(fn)
    for m in msgs:
        notify(m)


def _hook_notification(data: dict, now: float) -> None:
    if data.get("notification_type") != "permission_prompt":
        return
    msgs = []

    def fn(t):
        if t is None:
            return None
        t.update(turn="permission", turn_since=_iso(now))
        turn = _turn_s(t, now)
        if turn > SHORT_TURN_S:
            exp = f"Exp {_exps(t)} · " if t["items"] else ""
            msgs.append(f"⚠ {exp}Permission needed · turn running {_dur(turn)}")
        return t
    _update_task(fn)
    for m in msgs:
        notify(m)


def _hook_post_tool(data: dict, now: float) -> None:
    t = _read_task()
    if t is None or t.get("turn") != "permission":            # the common case: no lock, no write
        return

    def fn(t):
        if t is None or t.get("turn") != "permission":
            return None
        t.update(turn="working", turn_since=_iso(now))
        return t
    _update_task(fn)


HOOKS = {"UserPromptSubmit": _hook_prompt, "Stop": _hook_stop, "Notification": _hook_notification,
         "PostToolUse": _hook_post_tool}


def hook(event: str) -> None:
    """Entry point for Claude Code hooks: reads the hook JSON from stdin, never raises, prints nothing."""
    real_out = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")                     # noqa: SIM115 — closed with the process
        try:
            data = json.loads(sys.stdin.read() or "{}")
        except (ValueError, OSError, UnicodeDecodeError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        ev = event or str(data.get("hook_event_name") or "")
        f = HOOKS.get(ev)
        if f is not None:
            f(data, time.time())
    except BaseException as e:                                  # noqa: BLE001 — a hook must never fail loudly
        try:
            _log(f"hook {event} failed: {e!r}")
        except BaseException:                                   # noqa: BLE001, S110
            pass
    finally:
        sys.stdout = real_out


def _sentinel_tick(me: int, expired: bool, now: float) -> tuple[list[str], bool]:
    """One poll under the lock: (messages to send, exit?)."""
    msgs, out = [], {"exit": False}

    def fn(t):
        if t is None or t.get("sentinel_pid") != me:            # archived, replaced or superseded
            out["exit"] = True
            return None
        if t.get("turn") == "working" or expired:              # Claude is back (it reports) or too long
            out["exit"] = True
            t["sentinel_pid"] = None
            return t
        sm = summary(t, now)
        for v in sm["views"]:
            if v["state"] == "stopped" and not v["it"].get("crash_notified"):
                v["it"]["crash_notified"] = True
                msgs.append(_stop_message(v))
        if not sm["running"] and not sm["busy"]:
            if not t.get("notified"):
                t["notified"] = True
                msgs.append(_done_message(t, sm, now))
            t["sentinel_pid"] = None
            out["exit"] = True
        return t
    _update_task(fn)
    return msgs, out["exit"]


def sentinel(poll: float | None = None, max_s: float = SENTINEL_MAX_S) -> None:
    """Watch the task's background runs after Claude stopped: notify stops and the task's end, then exit."""
    poll = float(os.environ.get("TMD_PROGRESS_POLL") or 15) if poll is None else poll
    me, t0 = os.getpid(), time.time()
    _log("sentinel start", "_sentinel.log")
    while True:
        now = time.time()
        try:
            msgs, done = _sentinel_tick(me, now - t0 > max_s, now)
        except Exception as e:                                  # noqa: BLE001 — keep watching, log it
            _log(f"sentinel error: {e!r}", "_sentinel.log")
            msgs, done = [], now - t0 > max_s
        for m in msgs:
            notify(m)
        if done:
            _log("sentinel exit", "_sentinel.log")
            return
        time.sleep(poll)


# ---------------------------------------------------------------- plan CLI

def _resolve_exp(exp: str) -> str:
    if exp.isdigit():
        hits = sorted(p.name for p in (ROOT / "experiments").glob(f"{int(exp):02d}_*") if p.is_dir())
        if len(hits) != 1:
            raise SystemExit(f"experiment {exp!r}: {'no folder' if not hits else 'ambiguous: ' + ', '.join(hits)}")
        return hits[0]
    return exp


def _parse_item(spec: str, need_est: bool = True) -> dict:
    """'11:leg_grid=15' -> item. The estimate may be omitted (planned without estimate)."""
    key, _, est = spec.partition("=")
    exp, sep, run = key.partition(":")
    if not sep or not exp or not run:
        raise SystemExit(f"bad item {spec!r}: expected <exp>:<run>=<minutes>")
    try:
        est_min = float(est) if est else None
    except ValueError:
        raise SystemExit(f"bad estimate in {spec!r}") from None
    if est_min is not None and est_min.is_integer():
        est_min = int(est_min)
    return {"experiment": _resolve_exp(exp), "run": run, "est_min": est_min, "state": "pending", "started": None,
            "finished": None, "planned": True}


def _print_plan(t: dict) -> None:
    sm = summary(t)
    print(f"plan: {t.get('title') or '(untitled)'}")
    for v in sm["views"]:
        it = v["it"]
        est = f"{it['est_min']:g}m" if it.get("est_min") else "—"
        extra = f"  skipped: {it.get('reason', '')}" if v["state"] == "skipped" else ""
        print(f"  {_num(it['experiment'])}:{it['run']:<28} est {est:>5}  {v['state']}{extra}")
    print(f"  total est {sm['sum_est']:g}m")


def plan_cli(args: list[str]) -> None:
    now = time.time()
    if not args:
        raise SystemExit('usage: plan "<title>" exp:run=min ... | plan add exp:run=min ... | plan skip exp:run [reason]')
    if args[0] == "add":
        new = [_parse_item(a) for a in args[1:]]

        def fn(t):
            t = t or _new_task("", now)
            for n in new:
                it = _find(t, (n["experiment"], n["run"]))
                if it is None:
                    t["items"].append(n)
                else:
                    it.update(planned=True, est_min=n["est_min"])
                    if it.get("state") == "skipped":
                        it["state"] = "pending"
                        it.pop("reason", None)
            return t
    elif args[0] == "skip":
        if len(args) < 2:
            raise SystemExit("usage: plan skip exp:run [reason]")
        k = _parse_item(args[1])
        reason = " ".join(args[2:])

        def fn(t):
            it = None if t is None else _find(t, (k["experiment"], k["run"]))
            if it is None:
                raise SystemExit(f"{args[1]} is not in the plan")
            it.update(state="skipped", reason=reason)
            return t
    else:
        title, new = args[0], [_parse_item(a) for a in args[1:]]

        def fn(t):
            t = t or _new_task("", now)
            old = {(it["experiment"], it["run"]): it for it in t["items"]}
            items = []
            for n in new:                                  # keep what already ran for items still in the plan
                o = old.pop((n["experiment"], n["run"]), None)
                if o is not None and o.get("started"):
                    n.update(state=o.get("state", "running"), started=o["started"], finished=o.get("finished"))
                items.append(n)
            items += [o for o in old.values() if o.get("started")]   # runs that happened stay, as unplanned
            for o in items[len(new):]:
                o["planned"] = False
            t.update(title=title, items=items)
            return t
    t = _update_task(fn)
    if t is None:
        raise SystemExit("could not update .progress/_task.json (lock busy)")
    _print_plan(t)


def history(pattern: str = "") -> None:
    """Median minutes of past finished runs whose '<experiment>:<run>' contains pattern (fnmatch if it has *?[)."""
    import fnmatch
    from statistics import median

    def match(exp, run):
        s = f"{exp}:{run}"
        if any(ch in pattern for ch in "*?["):
            return fnmatch.fnmatch(s, pattern) or fnmatch.fnmatch(run, pattern)
        return pattern in s

    seen, runs = set(), {}
    for f in _run_files():
        s = _read_json(f)
        try:
            if s and s["done"] >= s["total"] and match(s["experiment"], s["run"]):
                seen.add((s["experiment"], s["run"], s["started"]))
                runs.setdefault((s["experiment"], s["run"]), []).append((_ts(s["updated"]) - _ts(s["started"])) / 60)
        except (KeyError, TypeError, ValueError):
            pass
    for f in sorted((DIR / "_tasks").glob("*.json")):
        t = _read_json(f) or {}
        for it in t.get("items", []):
            try:
                k = (it["experiment"], it["run"], it["started"])
                if it.get("state") == "done" and it.get("actual_min") is not None and k not in seen \
                        and match(it["experiment"], it["run"]):
                    seen.add(k)
                    runs.setdefault(k[:2], []).append(float(it["actual_min"]))
            except (KeyError, TypeError, ValueError):
                pass
    if not runs:
        print(f"no finished runs match {pattern!r}")
        return
    allm = []
    for (exp, run), ms in sorted(runs.items()):
        allm += ms
        print(f"{_num(exp)}:{run:<30} n={len(ms):<3} median {median(ms):6.1f} min")
    print(f"{'all matches':<33} n={len(allm):<3} median {median(allm):6.1f} min")


# ---------------------------------------------------------------- board

C = {"dim": "\033[2m", "yellow": "\033[33m", "red": "\033[31m", "green": "\033[32m", "bold": "\033[1m",
     "off": "\033[0m"}


def board(now: float | None = None, color: bool = False, width: int = 79) -> str:
    """The task board shown by `watch`."""
    now = time.time() if now is None else now

    def col(s, c):
        return f"{C[c]}{s}{C['off']}" if color else s

    t = _read_task()
    rule = " " + "─" * (width - 2)
    head_r = ""
    if t is not None:
        try:
            since = _ts(t["turn_since"])
        except (KeyError, TypeError, ValueError):
            since = now
        turn = t.get("turn")
        sm = summary(t, now) if t["items"] else None
        if turn == "waiting" and sm is not None and sm["finished"]:
            head_r = col(f"✓ Task done {_hm(sm['done_at'] or now)} · {_task_dur(t, sm)}", "green")
        elif turn == "waiting":
            head_r = col(f"⏸ Waiting for you since {_hm(since)} ({_dur(now - since)})", "yellow")
        elif turn == "permission":
            head_r = col(f"⚠ Permission prompt since {_hm(since)}", "red")
        else:
            try:
                ws = _ts(t.get("prompt_at") or t["turn_since"])
            except (KeyError, TypeError, ValueError):
                ws = since
            head_r = col(f"● Claude working · {_dur(now - ws)}", "dim")
    if t is None or not t["items"]:
        s = newest()
        body = _run_row(s) if s is not None else col("no run in progress", "dim")
        return "\n".join([f" {head_r}", rule, f" {body}", ""])

    multi = "," in _exps(t)
    title = f" Exp {_exps(t)} · {t.get('title') or '(untitled)'}"
    vis = len(head_r) - (len(C["dim"]) + len(C["off"]) if color else 0)
    out = [title + " " * max(width - len(title) - vis, 2) + head_r, rule]
    nw = max([len(_name(v, multi)) for v in sm["views"]] + [12]) + 2
    for v in sm["views"]:
        it, st, name = v["it"], v["state"], _name(v, multi).ljust(nw)
        est = f"est {it['est_min']:>3g}m" if it.get("est_min") else ("(unplanned)" if not it.get("planned") else "est   —")
        est = est.ljust(11)
        prog = ""
        if v["total"]:
            prog = f"{_bar(v['done'] / v['total'], 10)} {v['done']}/{v['total']}"
        if st == "skipped":
            out.append(col(f" – {name}skipped: {it.get('reason') or ''}", "dim"))
        elif st == "stopped":
            out.append(col(f" ✗ {name}{est} stopped at {v['done']}/{v['total']}", "red"))
        elif st == "done":
            took = f"took {_dur(v['actual'])}" if v["actual"] is not None else ""
            out.append(f" ✓ {name}{est} {took}")
        elif st == "running":
            r = v["run"]
            tm = ""
            if r is not None:
                tm = f" · {_dur(now - _ts(r['started']))}" + (f" · ~{_dur(v['eta'])} left" if v["eta"] is not None else "")
            out.append(f" ▶ {name}{est} {prog}{tm}")
        else:
            out.append(f" · {name}{est}")
    out.append(rule)
    if sm["pct"] is not None:
        tail = f"   ~{_dur(sm['eta'])} left · ≈{_hm(sm['finish_at'])}" if sm["eta"] is not None else ""
        cal = f"   (running ×{sm['c']:.1f} vs estimates)" if sm["c"] is not None else ""
        out.append(f" Overall {_bar(sm['pct'], 15)} {round(100 * sm['pct'])}%{tail}{cal}")
    return "\n".join(out) + "\n"


def watch(poll: float = 2.0, once: bool = False) -> None:
    import shutil
    tty = sys.stdout.isatty()
    try:
        while True:
            w = min(shutil.get_terminal_size((80, 24)).columns, 100) - 1
            try:
                text = board(color=tty, width=w)
            except Exception as e:                            # noqa: BLE001 — a bad file must not kill the pane
                text = f" (board error: {e!r})\n"
            if once:
                sys.stdout.write(text)
                return
            sys.stdout.write("\033[H\033[2J" + text)
            sys.stdout.flush()
            time.sleep(poll)
    except KeyboardInterrupt:
        pass


def main(argv: list[str]) -> None:
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])
    if cmd == "watch":
        watch(once="--once" in rest)
    elif cmd == "hook":
        hook(rest[0] if rest else "")
    elif cmd == "sentinel":
        sentinel()
    elif cmd == "plan":
        plan_cli(rest)
    elif cmd == "history":
        history(rest[0] if rest else "")
    else:
        sys.exit("usage: python -m tmd.progress {plan,history,watch,hook,sentinel}")


if __name__ == "__main__":
    main(sys.argv[1:])
