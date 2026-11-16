"""The SITL container must write into the run dir as the host user, or the cleanup cannot delete its files."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "sitl_runs", Path(__file__).resolve().parents[1] / "scripts" / "sitl_runs.py")
sitl_runs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sitl_runs)


def test_container_runs_as_host_user_on_posix(monkeypatch):
    monkeypatch.setattr(sitl_runs.os, "getuid", lambda: 1001, raising=False)
    monkeypatch.setattr(sitl_runs.os, "getgid", lambda: 1002, raising=False)
    assert sitl_runs.container_user() == ["--user", "1001:1002", "-e", "HOME=/runs"]


def test_container_falls_back_to_root_without_posix_ids(monkeypatch):
    monkeypatch.delattr(sitl_runs.os, "getuid", raising=False)
    assert sitl_runs.container_user() == ["--user", "root"]
