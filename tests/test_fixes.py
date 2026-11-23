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


@pytest.fixture
def stub_log(monkeypatch):
    # Give the guard real field names, so the only thing that can fail is the expression itself.
    import numpy as np
    from autonomy_eval import flightdata
    monkeypatch.setattr(flightdata, "message_columns", lambda path, msg: {"Volt": 0, "Curr": 0})
    monkeypatch.setattr(flightdata, "read_fields",
                        lambda path, msg, names: (np.array([0.0]), {n: np.array([1.0]) for n in names}))


@pytest.mark.parametrize("bad, why", [
    ('"a" * 4000000000', "only numbers"),
    ("'x'", "only numbers"),
    ("1" * 300, "too long"),
])
def test_derived_signal_refuses_strings_and_long_expressions(tmp_path, stub_log, bad, why):
    from autonomy_eval import flightdata
    log = tmp_path / "x.bin"
    log.write_bytes(b"")
    with pytest.raises(ValueError, match=why):
        flightdata.derived_signal(log, "BAT", bad)


@pytest.mark.parametrize("bad, why", [
    ("Volt *", "not valid"),
    ("abs(1, 2)", "exactly one value"),
    ("sqrt()", "exactly one value"),
    ("1 +", "not valid"),
    ("2 ** 3", "not allowed"),
])
def test_bad_expressions_give_a_plain_value_error(tmp_path, stub_log, bad, why):
    from autonomy_eval import flightdata
    log = tmp_path / "x.bin"
    log.write_bytes(b"")
    with pytest.raises(ValueError, match=why):
        flightdata.derived_signal(log, "BAT", bad)


def test_array_field_gives_a_plain_value_error(monkeypatch, tmp_path):
    from autonomy_eval import flightdata

    class Row:
        TimeUS = 1000
        ISBD_SAMPLES = [1, 2, 3]  # an array-type field, not one number per row

    class FakeLog:
        rows = [Row()]

        def recv_match(self, type):
            return self.rows.pop() if self.rows else None

        def close(self):
            pass

    monkeypatch.setattr(flightdata, "_open", lambda path: FakeLog())
    with pytest.raises(ValueError, match="not a single number"):
        flightdata.read_fields(tmp_path / "x.bin", "ISBD", ["ISBD_SAMPLES"])


def test_cut_short_github_body_is_reported_not_raised(monkeypatch):
    import http.client
    monkeypatch.setattr(bisect, "_get", lambda url: (_ for _ in ()).throw(http.client.IncompleteRead(b"{")))
    b = bisect.localize("xtrack_mean_m", "bf08027", "b2b1b3d")
    assert b is not None and b.error.startswith("couldn't reach GitHub")


def test_parser_printing_goes_to_stderr_not_stdout(monkeypatch, capsys, tmp_path):
    from mcp_server import server
    monkeypatch.setenv(server.ENV_DIR, str(tmp_path))
    log = tmp_path / "bad.bin"
    log.write_bytes(b"x")

    def noisy_load(path):
        print("DFFormat: Unsupported format char")  # pymavlink prints parse problems like this
        raise ValueError("bad")

    monkeypatch.setattr(server, "load", noisy_load)
    with pytest.raises(server.Refused):
        server._read(str(log))
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout carries the MCP messages; nothing else may write there
    assert "Unsupported format char" in captured.err


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
    out = server._map({"bad\nkey": "x" * 300, "ok": 1.5})
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


def test_stored_arrays_cannot_be_made_writable_again():
    import numpy as np

    def reader(path):
        return {"t": np.arange(4.0)}

    store = {}
    got = privacy.memo(store, reader, "a.bin", b"data")
    with pytest.raises(ValueError):
        got["t"].setflags(write=True)
    with pytest.raises(ValueError):
        got["t"][0] = 7.0
    again = privacy.memo(store, reader, "a.bin", b"data")
    assert again["t"][0] == 0.0


def test_connector_refuses_nested_values_in_summaries():
    from mcp_server import server
    with pytest.raises(server.Refused, match="could not be read"):
        server._map({"quality": {"nested": 1}})
    assert server._map({"count": 3, "ok": True, "ratio": 0.5}) == {"count": 3, "ok": True, "ratio": 0.5}


def test_connector_keeps_distinct_long_names_distinct():
    from mcp_server import server
    used: set = set()
    base = "A" * 80
    first = server._unique(server._label(base + "L"), used)
    second = server._unique(server._label(base + "R"), used)
    assert first != second
    assert len(first) <= server.MAX_LABEL_CHARS and len(second) <= server.MAX_LABEL_CHARS


def test_connector_path_checks_refuse_escapes_and_non_logs(monkeypatch, tmp_path):
    from mcp_server import server
    (tmp_path / "notes.txt").write_bytes(b"x")
    monkeypatch.setenv(server.ENV_DIR, str(tmp_path))
    for bad in [str(tmp_path / ".." / "outside.bin"), str(tmp_path / "notes.txt"), str(tmp_path / "missing.bin")]:
        with pytest.raises(server.Refused):
            server._checked_file(bad)
    monkeypatch.setenv(server.ENV_DIR, "relative/logs")
    with pytest.raises(server.Refused):
        server._allowed_root()


class _CaptureOpener:
    """Records the request headers and fails the call, so no network is used."""

    def __init__(self, sink):
        self.sink = sink

    def open(self, req, timeout=None):
        self.sink.append(dict(req.headers))
        raise urllib.error.URLError("offline")


def test_github_token_is_sent_only_when_set(monkeypatch):
    url = "https://api.github.com/repos/x/y"
    sent = []
    monkeypatch.setattr(bisect, "_opener", lambda: _CaptureOpener(sent))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(urllib.error.URLError):
        bisect._get(url)
    monkeypatch.setenv("GITHUB_TOKEN", "TEST-TOKEN-1")
    with pytest.raises(urllib.error.URLError):
        bisect._get(url)
    assert "Authorization" not in sent[0]
    assert sent[1]["Authorization"] == "Bearer TEST-TOKEN-1"


@pytest.mark.parametrize("frame_class, mav_type", [
    (1, 2),    # quad -> QUADROTOR
    (7, 15),   # tri -> TRICOPTER
    (8, 3),    # single -> COAXIAL (as in AP_MotorsSingle.cpp)
    (9, 3),    # coax -> COAXIAL
    (12, 29),  # dodecahexa -> DODECAROTOR
    (14, 35),  # deca -> DECAROTOR
])
def test_copter_frame_classes_map_to_heartbeat_types(frame_class, mav_type):
    from autonomy_eval import dataflash
    (vt,) = dataflash._vehicle_type(0.0, "ArduCopter", frame_class)
    assert vt.mav_type == mav_type


def test_copter_tailsitter_and_scripted_frames_stay_unknown():
    from autonomy_eval import dataflash
    for frame_class in (0, 10, 15, 16, 17):
        assert list(dataflash._vehicle_type(0.0, "ArduCopter", frame_class)) == []


def test_real_boot_log_text_is_joined_before_severity():
    # MSG text is logged in chunks that share an ID. The reader must join them, so the keyword rules
    # see whole messages. This real log's long messages are the test.
    from autonomy_eval.dataflash import read_dataflash
    from autonomy_eval.telemetry import Text
    texts = [x.text for x in read_dataflash("tests/data/rover-boot-a64bad1a.BIN") if isinstance(x, Text)]
    assert "EKF3 IMU0 MAG0 initial yaw alignment complete" in texts
    assert not any(t.endswith("initial yaw alignment comp") for t in texts)  # no cut-off fragment
