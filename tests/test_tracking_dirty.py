"""tracking.is_dirty: logs, run outputs, daily logs and docs don't count; code, environment and 04's reference
manifests do."""
import subprocess

import pytest

from tmd import tracking

M04 = "experiments/04_frozen-benchmark_es-shl/results/manifests/waiis_bench_v1.json"
FILES = {"experiments/05_x/run.py": "x = 1\n", "experiments/05_x/logs/a.log": "old\n",
         "experiments/05_x/results/runs.csv": "a\n", "experiments/05_x/README.md": "# x\n", "log/2026-10-30.md": "-\n",
         "STATUS.md": "s\n", "pyproject.toml": "[project]\n", "uv.lock": "lock\n", "src/tmd/m.py": "y = 1\n",
         M04: "{}\n"}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    for f, c in FILES.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(c)
    git("add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    monkeypatch.setattr(tracking, "PROJECT", tmp_path)
    return tmp_path, git


@pytest.mark.parametrize("path", ["experiments/05_x/logs/a.log", "experiments/05_x/logs/new.log",
                                  "experiments/05_x/results/runs.csv", "experiments/06_y/results/new/cov.csv",
                                  "experiments/05_x/README.md", "STATUS.md", "log/2026-10-30.md", "log/2026-10-31.md"])
def test_ignored(repo, path):
    root, _ = repo
    assert not tracking.is_dirty()
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text("changed\n")
    assert not tracking.is_dirty(), path


@pytest.mark.parametrize("path", ["experiments/05_x/run.py", "experiments/05_x/new_script.py", "src/tmd/m.py",
                                  "pyproject.toml", "uv.lock", M04, "experiments/04_frozen-benchmark_es-shl/results/"
                                  "manifests/new.json"])
def test_counted(repo, path):
    root, _ = repo
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text("changed\n")
    assert tracking.is_dirty(), path


def test_rename_of_code_counts(repo):
    _, git = repo
    git("mv", "src/tmd/m.py", "src/tmd/m2.py")
    assert tracking.is_dirty()
    assert set(tracking.changed_paths()) >= {"src/tmd/m.py", "src/tmd/m2.py"}
