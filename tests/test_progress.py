"""tmd.progress: plan, overall % and time left, board, and the watch screen (clear on a new task, freeze at the end)."""
import json
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
    monkeypatch.setattr(P, "DIR", tmp_path)
    procs = []

    class E:
        dir = tmp_path

        @staticmethod
        def alive_pid():
            p = subprocess.Popen(["sleep", "120"])
            procs.append(p)
            return p.pid

        @staticmethod
        def dead_pid():
            p = subprocess.Popen(["true"])
            p.wait()
            return p.pid

    yield E
    for p in procs:
        p.kill()


def write_run(d, run, done, total, pid, started=-300, updated=0):
    now = time.time()
    (d / f"{EXP}__{run}.json").write_text(json.dumps(
        {"experiment": EXP, "run": run, "done": done, "total": total, "started": P._iso(now + started),
         "updated": P._iso(now + updated), "pid": pid}))


def backdate(seconds):
    t = P.read_task()
    t["started"] = P._iso(P._ts(t["started"]) - seconds)
    P._write_json(P._task_path(), t)


def test_pct_and_time_left(env):
    P.plan("t", [f"{EXP}:a=5", f"{EXP}:b=10", f"{EXP}:c=20", f"{EXP}:notes=5"])
    backdate(3600)
    write_run(env.dir, "a", 10, 10, env.dead_pid(), started=-1800, updated=-1200)
    write_run(env.dir, "b", 50, 100, env.alive_pid(), started=-300, updated=0)      # half done, ~5 min left
    P.mark_done([f"{EXP}:notes"])
    sm = P.summary(P.read_task())
    assert [v["state"] for v in sm["views"]] == ["done", "running", "pending", "done"]
    assert sm["pct"] == pytest.approx((5 + 0.5 * 10 + 0 + 5) / 40)
    assert sm["left"] == pytest.approx(300 + 20 * 60, abs=2)
    assert not sm["complete"]


def test_running_without_eta_uses_estimate(env):
    P.plan("t", [f"{EXP}:a=10"])
    backdate(600)
    write_run(env.dir, "a", 0, 10, env.alive_pid(), started=-120)
    assert P.summary(P.read_task())["left"] == pytest.approx(480, abs=2)


def test_run_from_before_the_plan_does_not_count(env):
    write_run(env.dir, "a", 10, 10, env.dead_pid(), started=-900, updated=-600)
    P.plan("t", [f"{EXP}:a=5"])
    assert P.summary(P.read_task())["views"][0]["state"] == "pending"


def test_progress_does_not_touch_task(env):
    P.plan("t", [f"{EXP}:a=5"])
    before = P._task_path().read_bytes()
    p = P.Progress(EXP, "a", total=2)
    p.step()
    p.step()
    assert P._task_path().read_bytes() == before
    assert P.summary(P.read_task())["complete"]


def test_stopped_run_and_end(env):
    P.plan("t", [f"{EXP}:a=5", f"{EXP}:b=5"])
    backdate(600)
    write_run(env.dir, "a", 120, 260, env.dead_pid())
    sm = P.summary(P.read_task())
    assert sm["views"][0]["state"] == "stopped" and not sm["complete"]
    assert "✗ a" in P.board() and "stopped at 120/260" in P.board()
    P.end()
    assert P.summary(P.read_task())["complete"] and "✓ Done" in P.board()


def test_plan_parsing(env):
    t = P.plan("Exp 11: grid", ["11:leg_grid=15", "11:test=5", "11:notes"])
    assert [(i["experiment"], i["run"], i["est_min"]) for i in t["items"]] == \
        [(EXP, "leg_grid", 15), (EXP, "test", 5), (EXP, "notes", None)]
    for bad in (["99999:a=1"], ["no_colon=1"], ["11:a=x"]):
        with pytest.raises(SystemExit):
            P.plan("x", bad)
    with pytest.raises(SystemExit):
        P.mark_done(["11:nope"])
    r = subprocess.run([sys.executable, str(SRC), "plan"], capture_output=True, text=True, check=False)
    assert r.returncode != 0 and "usage" in r.stderr


def test_board_running_and_done(env):
    P.plan("leg grid", [f"{EXP}:a=10", f"{EXP}:b=15", f"{EXP}:c=5"])
    backdate(900)
    write_run(env.dir, "a", 1, 1, env.dead_pid(), started=-800, updated=-200)
    write_run(env.dir, "b", 207, 260, env.alive_pid(), started=-190)
    b = P.board()
    assert "Exp 11 · leg grid" in b and "started " in b
    assert "✓ a" in b and "took 10m" in b and "▶ b" in b and "207/260" in b and " · c" in b
    assert "left · ≈" in b and "Waiting" not in b
    write_run(env.dir, "b", 260, 260, env.dead_pid(), started=-190, updated=-10)
    P.mark_done([f"{EXP}:c"])
    b = P.board()
    assert "✓ Done" in b and "(est 30m)" in b and "100%" in b and "left" not in b


def test_watch_clears_on_new_task_and_freezes_when_complete(env):
    state = {"key": ""}
    assert P.frame(state).startswith("\033[H\033[2J")                  # first screen (no task): cleared
    P.plan("one", [f"{EXP}:a=1"])
    first = P.frame(state)
    assert first.startswith("\033[H\033[2J") and "one" in first          # new task: cleared, start anew
    again = P.frame(state)
    assert again.startswith("\033[H") and "\033[2J" not in again         # periodic redraw in place
    P.mark_done([f"{EXP}:a"])
    assert "✓ Done" in P.frame(state)                                   # completed state drawn once ...
    assert P.frame(state) is None and P.frame(state) is None            # ... then no more updates
    time.sleep(1.1)                                                     # plans are keyed by their start second
    P.plan("two", [f"{EXP}:b=1"])
    nxt = P.frame(state)
    assert nxt.startswith("\033[H\033[2J") and "two" in nxt and "one" not in nxt


@pytest.mark.parametrize("content", [None, "{corrupt", '{"items": 3}'])
def test_board_without_plan_shows_newest_run(env, content):
    write_run(env.dir, "solo", 3, 10, env.alive_pid())
    if content is not None:
        P._task_path().write_text(content)
    assert P.board().startswith(" 11 · solo ▓▓▓░░░░░░░ 3/10")
