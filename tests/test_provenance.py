"""Tests for scorecard provenance.

The rule these enforce is the one AGENTS.md already states for `.md` results —
a number is only comparable when its origin is known — applied to the JSON
scorecards. The second rule is that collecting provenance must never cost a
completed scoring run, so every lookup degrades to an absent field instead of
raising.
"""

import json
import subprocess

import pytest

from eval import provenance as prov
from eval.provenance import git_provenance, machine_provenance, run_provenance


class _Completed:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def _fake_git(monkeypatch, replies):
    """Answer git by subcommand, so the porcelain path is exercised for real."""

    def fake_run(cmd, **kwargs):
        for key, stdout in replies.items():
            if key in cmd:
                return _Completed(stdout)
        return _Completed("", returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)


# --- the parsing bug this module shipped with ---------------------------------


def test_an_unstaged_path_keeps_its_first_character(monkeypatch):
    """`git status --porcelain` left-pads: an unstaged edit is `" M path"`.

    Stripping that leading space shifts the column and eats one character of
    every unstaged path — it reported `eval/benchmark.py` as `val/benchmark.py`,
    a file that does not exist, in a field whose whole job is to be exact.
    """
    _fake_git(
        monkeypatch,
        {
            "rev-parse": "a" * 40,
            "status": " M eval/benchmark.py\n?? eval/provenance.py\nM  staged.py\n",
        },
    )

    result = git_provenance()

    assert result["dirty_files"] == ["eval/benchmark.py", "eval/provenance.py", "staged.py"]
    assert result["dirty_file_count"] == 3
    assert result["dirty"] is True


def test_a_clean_tree_is_reported_clean(monkeypatch):
    _fake_git(monkeypatch, {"rev-parse": "b" * 40, "status": ""})

    result = git_provenance()

    assert result["dirty"] is False
    assert result["dirty_files"] == []


def test_the_dirty_file_list_is_capped_but_the_count_is_not(monkeypatch):
    """A record, not a diff — but a truncated list must not understate the count."""
    lines = "".join(f" M file{i:03d}.py\n" for i in range(60))
    _fake_git(monkeypatch, {"rev-parse": "c" * 40, "status": lines})

    result = git_provenance()

    assert len(result["dirty_files"]) == prov.MAX_DIRTY_FILES
    assert result["dirty_file_count"] == 60


# --- degradation --------------------------------------------------------------


def test_no_git_is_an_absent_field_not_a_crash(monkeypatch):
    def explode(*a, **kw):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", explode)

    assert git_provenance() == {"available": False}
    assert run_provenance()["git"] == {"available": False}


def test_a_non_repo_directory_does_not_claim_a_commit(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _Completed("", returncode=128))

    result = git_provenance()

    assert result["available"] is False
    assert "commit" not in result


def test_collection_survives_a_broken_machine_lookup(monkeypatch):
    """Losing a finished scoring run to a hostname lookup would be a poor trade."""
    monkeypatch.setattr(prov, "machine_provenance", lambda: (_ for _ in ()).throw(OSError))

    assert run_provenance()["machine"] == {}


# --- what actually gets recorded ---------------------------------------------


def test_the_machine_record_identifies_the_interpreter_not_just_the_host():
    """`.venv` is --system-site-packages, so the host alone does not pin the stack.

    Two runs on this same machine resolve different torch installs depending on
    which python launched them; only the executable path separates them.
    """
    record = machine_provenance()

    assert record["hostname"]
    assert record["python_executable"].endswith("python.exe") or record[
        "python_executable"
    ].endswith("python")
    assert set(record["graph_ops"]["packages"]) == {"torch_geometric", "torch_scatter"}


def test_the_run_record_names_its_command_and_time():
    record = run_provenance()

    assert record["recorded_at"].endswith("+00:00"), "timestamps are UTC"
    assert record["command"]["cwd"]
    assert isinstance(record["command"]["argv"], list)


def test_the_whole_record_is_json_safe():
    """It is written straight into the scorecard; a non-serializable field loses the run."""
    json.dumps(run_provenance())


def test_provenance_of_this_repo_reports_the_real_branch():
    """No monkeypatching: the collector must work against an actual repo."""
    result = git_provenance()

    if not result.get("available"):
        pytest.skip("not run from a git checkout")
    assert len(result["commit"]) == 40
    assert result["commit_short"] == result["commit"][:8]
