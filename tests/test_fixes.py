"""Checks added after the review: the GitHub address guard, the command-line lookup failure path, and
the memo returning copies."""
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from autonomy_eval import bisect, cli, privacy


def test_lookup_refuses_any_address_other_than_github_api():
    with pytest.raises(ValueError, match="api.github.com"):
        bisect._get("file:///etc/passwd")
    with pytest.raises(ValueError, match="api.github.com"):
        bisect._get("https://example.com/x")


def test_command_line_lookup_failure_is_reported_not_raised(monkeypatch):
    def unreachable(url, *args, **kwargs):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(bisect, "_get", unreachable)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = SimpleNamespace(regressions=[SimpleNamespace(metric="xtrack_mean_m")],
                             candidate=SimpleNamespace(label="b2b1b3d"))
    baseline = [SimpleNamespace(label="bf08027")]
    with patch("anthropic.Anthropic") as Client:
        b = cli._bisect_check(result, baseline, True)
    assert b is not None and b.error
    assert "couldn't reach GitHub" in b.error
    Client.assert_not_called()


def test_memo_hands_out_copies_so_callers_cannot_change_the_stored_result():
    def reader(path):
        return {"values": [1, 2, 3]}

    store = {}
    first = privacy.memo(store, reader, "a.bin", b"data")
    first["values"].append(99)
    second = privacy.memo(store, reader, "a.bin", b"data")
    assert second == {"values": [1, 2, 3]}
    second["values"].clear()
    third = privacy.memo(store, reader, "a.bin", b"data")
    assert third == {"values": [1, 2, 3]}


def test_derived_signal_refuses_strings_and_long_expressions(tmp_path):
    from autonomy_eval import flightdata
    log = tmp_path / "x.bin"
    log.write_bytes(b"")
    for bad in ['"a" * 4000000000', "'x'", "1" * 300]:
        with pytest.raises(ValueError):
            flightdata.derived_signal(log, "BAT", bad)


@pytest.mark.parametrize("bad", ["Volt *", "abs(1, 2)", "sqrt()", "1 +", "2 ** 3"])
def test_bad_expressions_give_a_plain_value_error(tmp_path, bad):
    from autonomy_eval import flightdata
    log = tmp_path / "x.bin"
    log.write_bytes(b"")
    with pytest.raises(ValueError):
        flightdata.derived_signal(log, "BAT", bad)


def test_github_redirect_off_github_is_refused():
    handler = bisect._GitHubOnlyRedirects()
    with pytest.raises(urllib.error.URLError, match="refusing to follow a redirect"):
        handler.redirect_request(None, None, 302, "Found", {}, "https://evil.example/steal")


def test_connector_cuts_and_cleans_log_text():
    from mcp_server import server
    text = server._label("bad\x00text" + "x" * 300)
    assert "\x00" not in text
    assert len(text) == server.MAX_LABEL_CHARS


def test_connector_refuses_too_many_baseline_files(monkeypatch, tmp_path):
    from mcp_server import server
    monkeypatch.setenv(server.ENV_DIR, str(tmp_path))
    with pytest.raises(server.Refused, match="at most"):
        server.compare_log("x.bin", ["y.bin"] * (server.MAX_BASELINE_FILES + 1))


def test_bare_function_name_gives_a_plain_value_error(tmp_path, monkeypatch):
    import numpy as np
    from autonomy_eval import flightdata
    monkeypatch.setattr(flightdata, "message_columns", lambda path, msg: {"Volt": 0})
    monkeypatch.setattr(flightdata, "read_fields", lambda path, msg, names: (np.array([0.0]), {"Volt": np.array([1.0])}))
    log = tmp_path / "x.bin"
    log.write_bytes(b"")
    with pytest.raises(ValueError, match="could not be computed"):
        flightdata.derived_signal(log, "BAT", "abs")


def test_github_redirect_handler_is_really_used_by_the_lookup():
    assert any(isinstance(h, bisect._GitHubOnlyRedirects) for h in bisect._opener().handlers)


def test_redirect_to_odd_scheme_gives_url_error_not_index_error():
    with pytest.raises(urllib.error.URLError, match="refusing to follow a redirect"):
        bisect._GitHubOnlyRedirects().redirect_request(None, None, 302, "Found", {}, "ftp:abc")


def test_commit_id_with_trailing_newline_is_refused():
    assert bisect.localize("xtrack_mean_m", "bf08027\n", "b2b1b3d") is None


def test_connector_cleans_keys_of_summary_maps():
    from mcp_server import server
    out = server._clean({"bad\nkey": "x" * 300, "ok": 1.5})
    assert all("\n" not in k and len(k) <= server.MAX_LABEL_CHARS for k in out)
    assert out[server._label("ok")] == 1.5


def test_connector_gives_fixed_message_for_an_unreadable_file(monkeypatch, tmp_path):
    from mcp_server import server
    bad = tmp_path / "broken.json"
    bad.write_bytes(b"{not valid json at all")
    monkeypatch.setenv(server.ENV_DIR, str(tmp_path))
    with pytest.raises(server.Refused) as info:
        server.summarize_log(str(bad))
    assert str(info.value) == "this file could not be read as a flight log"


def test_memo_arrays_come_back_read_only_and_unchanged():
    import numpy as np

    def reader(path):
        return {"t": np.arange(5.0)}

    store = {}
    first = privacy.memo(store, reader, "a.bin", b"data")
    assert not first["t"].flags.writeable
    with pytest.raises(ValueError):
        first["t"][0] = 99.0
    again = privacy.memo(store, reader, "a.bin", b"data")
    assert again["t"][0] == 0.0
