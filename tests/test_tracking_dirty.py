"""tracking.is_dirty: changes under experiments/*/logs/ don't count; any other change does."""
import subprocess

from tmd import tracking


def _repo(tmp_path):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "experiments" / "05_x" / "logs").mkdir(parents=True)
    (tmp_path / "experiments" / "05_x" / "run.py").write_text("x = 1\n")
    (tmp_path / "experiments" / "05_x" / "logs" / "a.log").write_text("old\n")
    git("add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")


def test_logs_ignored_code_not(tmp_path, monkeypatch):
    _repo(tmp_path)
    monkeypatch.setattr(tracking, "PROJECT", tmp_path)
    assert not tracking.is_dirty()
    (tmp_path / "experiments" / "05_x" / "logs" / "a.log").write_text("new\n")       # modified log
    (tmp_path / "experiments" / "05_x" / "logs" / "b.log").write_text("new\n")       # untracked log
    assert not tracking.is_dirty()
    (tmp_path / "experiments" / "05_x" / "run.py").write_text("x = 2\n")
    assert tracking.is_dirty()
