"""Data handling for uploaded logs: temp files, per-session memory, and no outbound transfer of raw logs."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from autonomy_eval import bisect, checks, privacy
from autonomy_eval.metrics import load, summarize

ROOT = Path(__file__).resolve().parent.parent
BIN = Path(__file__).parent / "data" / "rover-boot-a64bad1a.BIN"


# --- temporary files -------------------------------------------------------------------------

def test_temp_file_is_removed_after_success():
    seen = {}

    def reader(path):
        seen["path"] = path
        assert path.read_bytes() == b"abc"
        return "ok"

    assert privacy.read_in_temp("flight.bin", b"abc", reader) == "ok"
    assert not seen["path"].exists()
    assert not seen["path"].parent.exists()


def test_temp_file_is_removed_after_an_exception():
    seen = {}

    def reader(path):
        seen["path"] = path
        raise ValueError("corrupt log")

    with pytest.raises(ValueError, match="corrupt log"):
        privacy.read_in_temp("flight.bin", b"abc", reader)
    assert not seen["path"].exists()
    assert not seen["path"].parent.exists()


def test_temp_file_is_removed_after_a_real_read_failure():
    # An empty .bin makes the real reader fail after it has opened the file. On Windows the open file
    # used to block the folder's removal, so the upload stayed on disk.
    seen = {}

    def reader(path):
        seen["path"] = path
        return load(path)

    with pytest.raises(ValueError):
        privacy.read_in_temp("empty.bin", b"", reader)
    assert not seen["path"].exists()
    assert not seen["path"].parent.exists()


def test_upload_name_cannot_escape_the_temp_directory():
    seen = {}

    def reader(path):
        seen["path"] = path
        return None

    privacy.read_in_temp("../../outside.bin", b"x", reader)
    assert seen["path"].name == "outside.bin"
    assert ".." not in seen["path"].parts


# --- per-session memory ----------------------------------------------------------------------

def _counting_reader():
    calls = []

    def reader(path, *args):
        calls.append(args)
        return {"size": path.stat().st_size, "args": args}

    return reader, calls


def test_result_is_reused_within_one_session():
    reader, calls = _counting_reader()
    store = {}
    first = privacy.memo(store, reader, "a.bin", b"data")
    second = privacy.memo(store, reader, "a.bin", b"data")
    assert first == second and len(calls) == 1


def test_result_is_not_shared_between_sessions():
    reader, calls = _counting_reader()
    alice, bob = {}, {}
    privacy.memo(alice, reader, "a.bin", b"data")
    privacy.memo(bob, reader, "a.bin", b"data")
    assert len(calls) == 2
    assert privacy.MEMO_KEY in alice and privacy.MEMO_KEY in bob
    assert alice[privacy.MEMO_KEY] is not bob[privacy.MEMO_KEY]


def test_same_name_different_bytes_is_not_served_a_stale_result():
    reader, calls = _counting_reader()
    store = {}
    privacy.memo(store, reader, "a.bin", b"one")
    privacy.memo(store, reader, "a.bin", b"two")
    assert len(calls) == 2


def test_different_arguments_are_different_results():
    reader, calls = _counting_reader()
    store = {}
    privacy.memo(store, reader, "a.bin", b"data", "ATT", "Roll")
    privacy.memo(store, reader, "a.bin", b"data", "ATT", "Pitch")
    assert len(calls) == 2


def test_memory_is_bounded_per_session():
    reader, _ = _counting_reader()
    store = {}
    for i in range(privacy.MEMO_MAX_ENTRIES + 5):
        privacy.memo(store, reader, "a.bin", bytes([i]))
    assert len(store[privacy.MEMO_KEY]) == privacy.MEMO_MAX_ENTRIES


def test_wipe_forgets_everything_in_the_session():
    reader, calls = _counting_reader()
    store = {}
    privacy.memo(store, reader, "a.bin", b"data")
    privacy.wipe(store)
    assert privacy.MEMO_KEY not in store
    privacy.memo(store, reader, "a.bin", b"data")
    assert len(calls) == 2


def test_only_public_functions_use_a_process_wide_cache():
    """Static guard. Only the functions listed here may use a cache shared between visitors, and each
    must take no uploaded bytes. Any other cache decorator, call form or import alias fails the test."""
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    public_cached = {"load_history", "cached_sample", "_public_lookup"}
    cache_words = ("cache_data", "cache_resource", "lru_cache", "functools", "cache(")

    decorator_nodes = set()
    cached_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                decorator_nodes.update(ast.walk(dec))
                if any(w in ast.unparse(dec) for w in cache_words):
                    cached_names.add(node.name)
    assert cached_names == public_cached, f"unexpected cached functions: {cached_names ^ public_cached}"

    for node in ast.walk(tree):
        if node in decorator_nodes:
            continue
        text = ast.unparse(node) if isinstance(node, (ast.Call, ast.Name, ast.Attribute)) else ""
        assert not any(w in text for w in cache_words), f"cache used outside a decorator: {text}"


# --- no raw log leaves the process -----------------------------------------------------------

def test_reading_a_log_makes_no_network_calls(monkeypatch):
    """Blocks name resolution and connections at the socket layer, so any library that tries to reach
    a host fails here, not only urllib."""
    def refuse(*args, **kwargs):
        raise AssertionError("network access during local log analysis")

    monkeypatch.setattr("socket.getaddrinfo", refuse)
    monkeypatch.setattr("socket.create_connection", refuse)
    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with patch("anthropic.Anthropic", side_effect=AssertionError("AI call during local analysis")):
        summary = privacy.read_in_temp(BIN.name, BIN.read_bytes(), summarize)
    assert summary is not None


def test_hosted_switch_turns_uploads_off(monkeypatch):
    monkeypatch.delenv(privacy.HOSTED_ENV, raising=False)
    assert privacy.uploads_enabled()
    monkeypatch.setenv(privacy.HOSTED_ENV, "1")
    assert not privacy.uploads_enabled()


def test_memo_separates_readers_with_the_same_name_in_different_modules():
    def make(module):
        def reader(path):
            return module
        reader.__module__ = module
        reader.__qualname__ = "reader"
        return reader

    store = {}
    assert privacy.memo(store, make("a"), "x.bin", b"d") == "a"
    assert privacy.memo(store, make("b"), "x.bin", b"d") == "b"


def test_failed_lookup_is_not_kept_and_is_retried(monkeypatch):
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        raise bisect.urllib.error.URLError("rate limited")

    monkeypatch.setattr(bisect, "_get", flaky)
    store = {}
    first = bisect.lookup("xtrack_mean_m", "bf08027", "b2b1b3d", store)
    second = bisect.lookup("xtrack_mean_m", "bf08027", "b2b1b3d", store)
    assert first.error and second.error
    assert store == {}
    assert len(calls) == 2


def test_successful_lookup_is_kept_for_this_session_only(monkeypatch):
    calls = []

    def ok(url, *args, **kwargs):
        calls.append(url)
        if "/compare/" in url:
            return {"commits": []}
        raise AssertionError("no commit details expected for an empty range")

    monkeypatch.setattr(bisect, "_get", ok)
    alice, bob = {}, {}
    bisect.lookup("xtrack_mean_m", "bf08027", "b2b1b3d", alice)
    bisect.lookup("xtrack_mean_m", "bf08027", "b2b1b3d", alice)
    assert len(calls) == 1
    bisect.lookup("xtrack_mean_m", "bf08027", "b2b1b3d", bob)
    assert len(calls) == 2


def test_ai_verdict_is_off_unless_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv(bisect.AI_OPT_IN_ENV, raising=False)
    b = bisect.Bisection("xtrack_mean_m", "bf08027", "b2b1b3d", "https://x")
    with patch("anthropic.Anthropic") as Client:
        assert bisect.ai_verdict(b) is None
    Client.assert_not_called()


def test_ai_verdict_sends_only_the_finding_when_enabled(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv(bisect.AI_OPT_IN_ENV, "1")
    b = bisect.Bisection("xtrack_mean_m", "bf08027", "b2b1b3d", "https://x", total_commits=1,
                          suspects=[bisect.SuspectCommit("2222222b", "fix", ["libraries/AC_WPNav/x.cpp"])])
    reply = json.dumps({"verdict": "uncertain", "reason": "unclear"})
    with patch("anthropic.Anthropic") as Client:
        Client.return_value.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=reply)])
        bisect.ai_verdict(b)
    sent = Client.return_value.messages.create.call_args.kwargs["messages"][0]["content"]
    assert set(json.loads(sent)) == {"metric", "commits_in_range", "suspect_commits"}


# --- existing public functionality is kept ---------------------------------------------------

def test_sample_flights_are_still_fetched_once_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(checks, "SAMPLE_CACHE", tmp_path)
    (tmp_path / "sysid-candidate-rollgain.bin").write_bytes(b"cached")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("refetched")))
    assert checks.fetch_sample("sysid-candidate-rollgain.bin").read_bytes() == b"cached"
    assert set(checks.SAMPLES) == {"sysid", "nees", "step"}


def test_example_histories_still_load():
    files = sorted((ROOT / "examples").glob("*/*.json"))
    assert files, "examples folder is empty"
    assert all(load(f) is not None for f in files)
