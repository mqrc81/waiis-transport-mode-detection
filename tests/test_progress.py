"""tmd.progress: task plan, overall % / ETA, hooks, sentinel, locking, board. No Claude Code needed.

osascript is mocked by a script on PATH that appends its arguments to a file; hooks run as subprocesses the way
Claude Code runs them, with TMD_PROGRESS_DIR pointing at a temporary directory.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tmd import progress as P

SRC = Path(P.__file__).resolve()
EXP = "11_leg-suggestions_waiis-app-days"


@pytest.fixture
def env(tmp_path, monkeypatch):
    d = tmp_path / "progress"
    d.mkdir()
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    calls = tmp_path / "osascript_calls"
    fake = bin_ / "osascript"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$2" >> "{calls}"\n')
    fake.chmod(0o755)
    monkeypatch.setattr(P, "DIR", d)
    monkeypatch.setenv("TMD_PROGRESS_DIR", str(d))
    monkeypatch.setenv("TMD_PROGRESS_POLL", "0.2")
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    procs = []

    class E:
        dir = d

        @staticmethod
        def notes():
            lines = calls.read_text().splitlines() if calls.exists() else []
            return [re.match(r'display notification "(.*)" with title', x).group(1) for x in lines]

        @staticmethod
        def hook(event, data=None, raw=None):
            stdin = raw if raw is not None else json.dumps(
                {"session_id": "s1", "cwd": str(d), "hook_event_name": event, **(data or {})})
            return subprocess.run([sys.executable, str(SRC), "hook", event], input=stdin, capture_output=True,
                                  text=True, timeout=20, check=False)

        @staticmethod
        def alive_pid():
            p = subprocess.Popen(["sleep", "120"])
            procs.append(p)
            return p

        @staticmethod
        def dead_pid():
            p = subprocess.Popen(["true"])
            p.wait()
            return p.pid

    yield E
    for p in procs:
        p.kill()
    t = P._read_task()
    if t and t.get("sentinel_pid") and P._alive(t["sentinel_pid"]):
        os.kill(t["sentinel_pid"], signal.SIGKILL)


def _iso(dt):
    return P._iso(time.time() + dt)


def write_run(d, run, done, total, pid, started=-600, updated=-60, exp=EXP):
    (d / f"{exp}__{run}.json").write_text(json.dumps(
        {"experiment": exp, "run": run, "done": done, "total": total, "started": _iso(started),
         "updated": _iso(updated), "pid": pid}))
    return _iso(started)


def item(run, est, state="pending", started=None, planned=True, **kw):
    return {"experiment": EXP, "run": run, "est_min": est, "state": state, "started": started, "finished": None,
            "planned": planned, **kw}


def write_task(d, items, turn="working", turn_since=0, prompt_at=None, **kw):
    t = P._new_task("s1", time.time())
    t.update(items=items, turn=turn, turn_since=_iso(turn_since), title="test",
             prompt_at=_iso(turn_since if prompt_at is None else prompt_at), task_started=_iso(-3600), **kw)
    (d / "_task.json").write_text(json.dumps(t))
    return t


def wait_for(cond, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------- % and ETA

def test_overall_pct_eta_and_calibration(env):
    d = env.dir
    live = env.alive_pid().pid
    s_done = write_run(d, "a", 10, 10, env.dead_pid(), started=-1800, updated=-1200)     # 10 min for est 5
    s_run = write_run(d, "b", 50, 100, live, started=-300, updated=0)                    # half, ~5 min left
    s_e = write_run(d, "e", 5, 10, live, started=-60, updated=0)       # ~1 min left: b sets the ETA
    t = write_task(d, [item("a", 5, "done", s_done), item("b", 10, "running", s_run), item("c", 20),
                       item("d", 30, "skipped", reason="no time"), item("e", None, "running", s_e, planned=False)])
    now = time.time()
    sm = P.summary(t, now)
    # (1·5 + 0.5·10 + 0·20) / 35; skipped d excluded; unplanned e excluded
    assert sm["pct"] == pytest.approx(10 / 35)
    assert sm["c"] == pytest.approx(2.0)                     # 10 actual / 5 est
    assert sm["eta"] == pytest.approx(300 + 2.0 * 20 * 60, abs=5)
    assert sm["finish_at"] == pytest.approx(now + sm["eta"])
    assert [v["state"] for v in sm["views"]] == ["done", "running", "pending", "skipped", "running"]
    assert not sm["finished"]


@pytest.mark.parametrize("actual_s,c", [(60, 0.5), (3600, 3.0)])
def test_calibration_clamped(env, actual_s, c):
    s = write_run(env.dir, "a", 1, 1, env.dead_pid(), started=-actual_s - 10, updated=-10)
    t = write_task(env.dir, [item("a", 10, "done", s), item("b", 10)])
    sm = P.summary(t)
    assert sm["c"] == c
    assert sm["eta"] == pytest.approx(c * 600)


def test_no_done_item_c_is_one(env):
    t = write_task(env.dir, [item("a", 10), item("b", 5)])
    sm = P.summary(t)
    assert sm["c"] is None and sm["eta"] == pytest.approx(900) and sm["pct"] == 0


# ---------------------------------------------------------------- Progress and the plan

def test_progress_marks_planned_and_appends_unplanned(env):
    write_task(env.dir, [item("planned_run", 5)])
    p = P.Progress(EXP, "planned_run", total=2)
    t = P._read_task()
    assert t["items"][0]["state"] == "running" and t["items"][0]["started"] == p.state["started"]
    p.step()
    assert P._read_task()["items"][0]["state"] == "running"
    p.step()
    it = P._read_task()["items"][0]
    assert it["state"] == "done" and it["finished"]
    P.Progress(EXP, "other", total=3)
    t = P._read_task()
    assert len(t["items"]) == 2
    assert t["items"][1] | {"started": None} == item("other", None, "running", None, planned=False)


@pytest.mark.parametrize("case", ["missing", "old_waiting"])
def test_no_active_task_leaves_task_file_alone(env, case):
    path = env.dir / "_task.json"
    if case == "old_waiting":
        write_task(env.dir, [item("x", 5)], turn="waiting", turn_since=-11 * 60)
        before = path.read_bytes()
    p = P.Progress(EXP, "x", total=1)
    p.step()
    if case == "missing":
        assert not path.exists()
    else:
        assert path.read_bytes() == before
    assert (env.dir / f"{EXP}__x.json").exists()           # the run file itself behaves as before


def test_recent_waiting_turn_still_links(env):
    write_task(env.dir, [item("x", 5)], turn="waiting", turn_since=-60)     # run launched just before Stop
    P.Progress(EXP, "x", total=1)
    assert P._read_task()["items"][0]["state"] == "running"


def _appender(dirname, k):
    os.environ["TMD_PROGRESS_DIR"] = dirname
    from tmd import progress as Q
    for i in range(50):
        Q.Progress(EXP, f"p{k}_{i}", total=1)


def test_locking_parallel_appends(env):
    write_task(env.dir, [])
    code = (f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); from test_progress import _appender; "
            "_appender(sys.argv[1], int(sys.argv[2]))")
    ps = [subprocess.Popen([sys.executable, "-c", code, str(env.dir), str(k)]) for k in range(4)]
    assert all(p.wait(60) == 0 for p in ps)
    names = {it["run"] for it in P._read_task()["items"]}
    assert len(names) == 200


# ---------------------------------------------------------------- hooks

def test_short_turn_without_items_does_not_notify(env):
    assert env.hook("UserPromptSubmit", {"prompt": "quick question"}).returncode == 0
    assert P._read_task()["turn"] == "working"
    r = env.hook("Stop")
    assert r.returncode == 0 and r.stdout == ""
    assert P._read_task()["turn"] == "waiting"
    assert env.notes() == []


def test_long_turn_without_items_notifies_once(env):
    env.hook("UserPromptSubmit")
    t = P._read_task()
    t["prompt_at"] = _iso(-300)
    (env.dir / "_task.json").write_text(json.dumps(t))
    env.hook("Stop")
    env.hook("Stop")
    assert len(env.notes()) == 1 and "Claude done" in env.notes()[0]


def test_all_done_one_notification(env):
    env.hook("UserPromptSubmit")
    t = P._read_task()
    dead = env.dead_pid()
    s1 = write_run(env.dir, "a", 3, 3, dead, updated=0)          # both finish during the turn
    s2 = write_run(env.dir, "b", 4, 4, dead, updated=0)
    t["items"] = [item("a", 5, "done", s1), item("b", 5, "done", s2)]
    (env.dir / "_task.json").write_text(json.dumps(t))
    env.hook("Stop")
    env.hook("Stop")
    notes = env.notes()
    assert len(notes) == 1 and "Exp 11 done · 2/2 runs" in notes[0] and "(est 10)" in notes[0]
    assert P._read_task()["sentinel_pid"] is None


def _stop_point(env, prompt_at, runs):
    """Task at a stop-point (b pending) whose run a ended `runs`; prompt `prompt_at` s ago; then Stop."""
    env.hook("UserPromptSubmit")
    t = P._read_task()
    t["prompt_at"] = _iso(prompt_at)
    if runs == "before":                                          # a finished in an earlier turn
        s = write_run(env.dir, "a", 3, 3, env.dead_pid(), started=-900, updated=-600)
    elif runs == "done":                                          # a finished during this turn
        s = write_run(env.dir, "a", 3, 3, env.dead_pid(), started=-60, updated=-5)
    else:                                                         # a stopped during this turn
        s = write_run(env.dir, "a", 1, 3, env.dead_pid(), started=-60, updated=-5)
    t["items"] = [item("a", 5, "done", s), item("b", 5)]
    (env.dir / "_task.json").write_text(json.dumps(t))
    env.hook("Stop")
    return env.notes()


@pytest.mark.parametrize("prompt_at,runs,n", [(-30, "before", 0), (-180, "before", 1), (-30, "done", 1),
                                              (-30, "stopped", 1)])
def test_stop_notifies_only_long_turns_or_run_changes(env, prompt_at, runs, n):
    notes = _stop_point(env, prompt_at, runs)
    assert len(notes) == n
    if n:
        assert "Waiting at stop-point · 1 run left in plan" in notes[0]
    if runs == "stopped":
        assert "1 stopped" in notes[0]
        env.hook("UserPromptSubmit", {"prompt": "what happened?"})    # same stop, short follow-up: quiet
        env.hook("Stop")
        assert len(env.notes()) == 1


def test_running_items_sentinel_then_one_notification(env):
    env.hook("UserPromptSubmit")
    t = P._read_task()
    worker = env.alive_pid()
    s = write_run(env.dir, "a", 1, 2, worker.pid)
    t["items"] = [item("a", 5, "running", s)]
    (env.dir / "_task.json").write_text(json.dumps(t))
    env.hook("Stop")
    assert env.notes() == []
    spid = P._read_task()["sentinel_pid"]
    assert spid and P._alive(spid)
    env.hook("Stop")                                          # a second Stop doesn't start a second sentinel
    assert P._read_task()["sentinel_pid"] == spid
    time.sleep(0.5)
    assert env.notes() == []
    write_run(env.dir, "a", 2, 2, worker.pid, started=-600, updated=0)   # same started second: rewrite keeps it
    t = P._read_task()
    t["items"][0]["started"] = json.loads((env.dir / f"{EXP}__a.json").read_text())["started"]
    (env.dir / "_task.json").write_text(json.dumps(t))
    worker.kill()
    worker.wait()
    assert wait_for(lambda: len(env.notes()) == 1)
    assert "done · 1/1 runs" in env.notes()[0]
    assert wait_for(lambda: not P._alive(spid))
    time.sleep(0.5)
    assert len(env.notes()) == 1


def test_sentinel_stopped_run_then_others_watched(env):
    env.hook("UserPromptSubmit")
    t = P._read_task()
    w1, w2 = env.alive_pid(), env.alive_pid()
    s1 = write_run(env.dir, "leg_grid", 120, 260, w1.pid)
    s2 = write_run(env.dir, "b", 1, 10, w2.pid)
    t["items"] = [item("leg_grid", 15, "running", s1), item("b", 5, "running", s2)]
    (env.dir / "_task.json").write_text(json.dumps(t))
    env.hook("Stop")
    spid = P._read_task()["sentinel_pid"]
    w1.kill()
    w1.wait()
    assert wait_for(lambda: len(env.notes()) >= 1, timeout=2)   # within one poll (0.2 s) plus slack
    assert env.notes() == ["✗ 11 leg_grid stopped at 120/260"]
    time.sleep(0.6)
    assert len(env.notes()) == 1 and P._alive(spid)            # still watching b
    write_run(env.dir, "b", 10, 10, w2.pid, updated=0)        # b finishes normally, then its script exits
    w2.kill()
    w2.wait()
    assert wait_for(lambda: len(env.notes()) == 2)
    assert "done · 1/2 runs" in env.notes()[1] and "1 stopped" in env.notes()[1]
    assert wait_for(lambda: not P._alive(spid))


def test_sentinel_exits_when_claude_works_again(env):
    env.hook("UserPromptSubmit")
    t = P._read_task()
    w = env.alive_pid()
    s = write_run(env.dir, "a", 1, 10, w.pid)
    t["items"] = [item("a", 5, "running", s)]
    (env.dir / "_task.json").write_text(json.dumps(t))
    env.hook("Stop")
    spid = P._read_task()["sentinel_pid"]
    env.hook("UserPromptSubmit", {"prompt": "how is it going?"})
    assert wait_for(lambda: not P._alive(spid), timeout=5)
    w.kill()
    time.sleep(0.5)
    assert env.notes() == []
    assert P._read_task()["items"][0]["run"] == "a"               # same task: a is still running


def test_prompt_after_stop_point_keeps_plan(env):
    env.hook("UserPromptSubmit")
    s = write_run(env.dir, "a", 1, 1, env.dead_pid(), updated=0)
    t = P._read_task()
    t["items"] = [item("a", 5, "done", s), item("b", 5)]
    (env.dir / "_task.json").write_text(json.dumps(t))
    env.hook("Stop")
    assert len(env.notes()) == 1 and "stop-point · 1 run left" in env.notes()[0]
    env.hook("UserPromptSubmit", {"prompt": "go on"})
    t2 = P._read_task()
    assert t2["task_started"] == t["task_started"] and len(t2["items"]) == 2 and t2["notified"] is False
    assert not (env.dir / "_tasks").exists()


def test_prompt_after_finished_task_archives(env):
    env.hook("UserPromptSubmit")
    s = write_run(env.dir, "a", 1, 1, env.dead_pid())
    t = P._read_task()
    t["items"] = [item("a", 5, "done", s)]
    (env.dir / "_task.json").write_text(json.dumps(t))
    env.hook("Stop")
    env.hook("UserPromptSubmit", {"prompt": "next thing"})
    t2 = P._read_task()
    assert t2["items"] == [] and t2["turn"] == "working"
    arch = list((env.dir / "_tasks").glob("*.json"))
    assert len(arch) == 1 and json.loads(arch[0].read_text())["items"][0]["actual_min"] is not None


def test_permission_prompt(env):
    env.hook("UserPromptSubmit")
    env.hook("Notification", {"notification_type": "permission_prompt", "message": "Claude needs your permission"})
    assert P._read_task()["turn"] == "permission" and env.notes() == []          # short turn: no notification
    assert '"turn": "permission"' in (env.dir / "_task.json").read_text()      # the PostToolUse shell guard
    env.hook("PostToolUse", {"tool_name": "Bash"})
    assert P._read_task()["turn"] == "working"
    t = P._read_task()
    t["prompt_at"] = _iso(-600)
    (env.dir / "_task.json").write_text(json.dumps(t))
    env.hook("Notification", {"notification_type": "permission_prompt"})
    assert len(env.notes()) == 1 and "Permission" in env.notes()[0]
    env.hook("Notification", {"notification_type": "idle_prompt"})
    assert P._read_task()["turn"] == "permission" and len(env.notes()) == 1


@pytest.mark.parametrize("raw", ["", "not json", "[1, 2]", "\x00\xff", '{"session_id": 5}'])
def test_hook_garbage_is_silent(env, raw):
    for ev in ("UserPromptSubmit", "Stop", "Notification", "PostToolUse", "Bogus"):
        r = env.hook(ev, raw=raw)
        assert r.returncode == 0 and r.stdout == "" and r.stderr == ""


def test_hook_with_corrupt_task_file_is_silent(env):
    (env.dir / "_task.json").write_text("{corrupt")
    for ev in ("UserPromptSubmit", "Stop"):
        r = env.hook(ev)
        assert r.returncode == 0 and r.stdout == ""


# ---------------------------------------------------------------- plan CLI

def test_plan_cli(env):
    run = lambda *a: subprocess.run([sys.executable, str(SRC), "plan", *a], capture_output=True, text=True, check=False)
    env.hook("UserPromptSubmit")
    r = run("Exp 11: grid", "11:leg_grid=15", "11:test=5")
    assert r.returncode == 0, r.stderr
    t = P._read_task()
    assert t["title"] == "Exp 11: grid" and [(i["experiment"], i["run"], i["est_min"]) for i in t["items"]] == \
        [(EXP, "leg_grid", 15), (EXP, "test", 5)]
    run("add", "11:extra=2.5")
    run("skip", "11:test", "not", "needed")
    t = P._read_task()
    assert t["items"][2]["est_min"] == 2.5
    assert t["items"][1]["state"] == "skipped" and t["items"][1]["reason"] == "not needed"
    assert run("skip", "11:nope").returncode != 0
    assert run("x", "99999:a=1").returncode != 0


# ---------------------------------------------------------------- board

def test_board_states(env):
    w = env.alive_pid()
    s = write_run(env.dir, "b", 207, 260, w.pid, updated=0)
    t = write_task(env.dir, [item("a", 10, "done", write_run(env.dir, "a", 1, 1, env.dead_pid())),
                             item("b", 15, "running", s), item("c", 5)])
    b = P.board()
    assert "Exp 11 · test" in b and "● Claude working" in b and "▶ b" in b and "207/260" in b and "· c" in b
    assert "Overall" in b and "(running ×" in b
    t["turn"] = "waiting"
    (env.dir / "_task.json").write_text(json.dumps(t))
    assert "⏸ Waiting for you since" in P.board()
    t["turn"] = "permission"
    (env.dir / "_task.json").write_text(json.dumps(t))
    assert "⚠ Permission prompt" in P.board()


def test_board_done_skipped_stopped(env):
    dead = env.dead_pid()
    write_task(env.dir, [item("a", 10, "done", write_run(env.dir, "a", 1, 1, dead)),
                         item("b", 5, "skipped", reason="later"),
                         item("c", 5, "running", write_run(env.dir, "c", 120, 260, dead))],
               turn="waiting", stop_at=_iso(-10))
    b = P.board()
    assert "✓ Task done" in b and "– b" in b and "skipped: later" in b and "✗ c" in b and "stopped at 120/260" in b


@pytest.mark.parametrize("content", [None, "{corrupt", '{"items": 3}'])
def test_board_without_plan_shows_single_run(env, content):
    w = env.alive_pid()
    write_run(env.dir, "solo", 3, 10, w.pid, updated=0)
    if content is not None:
        (env.dir / "_task.json").write_text(content)
    r = subprocess.run([sys.executable, str(SRC), "watch", "--once"], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "11 · solo ▓▓▓░░░░░░░ 3/10" in r.stdout
