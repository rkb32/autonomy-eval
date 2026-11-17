"""PX4 ULog reading. Sample-dependent tests skip when the public PX4 example log is not in the temp cache."""
import os
from pathlib import Path

import pytest

from autonomy_eval import metrics
from autonomy_eval import ulog as ulog_mod
from autonomy_eval.telemetry import Armed, Sample, Text, is_ulog

SAMPLE = Path(os.environ.get("TEMP", "")) / "autonomy-eval-samples" / "px4-quadrotor-example.ulg"
needs_sample = pytest.mark.skipif(not SAMPLE.exists(), reason="no PX4 sample log in the temp cache")


def test_ulog_header_detected(tmp_path):
    p = tmp_path / "x.ulg"
    p.write_bytes(b"ULog\x01\x12\x35" + b"\0" * 32)
    assert is_ulog(p)


def test_other_formats_are_not_ulog(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"\xa3\x95\x80" + b"\0" * 32)
    assert not is_ulog(p)


@needs_sample
def test_sample_summary_identifies_px4_and_format():
    s = metrics.summarize(SAMPLE)
    assert s.quality["format"] == "ulog"
    assert s.vehicle == "PX4"
    assert s.firmware == "1.10.0"
    assert s.mav_type == 2   # quadrotor


@needs_sample
def test_sample_armed_time_matches_takeoff_to_landing():
    s = metrics.summarize(SAMPLE)
    # Armed from the first record until disarm at ~117.2 s; takeoff and landing are logged inside that window.
    assert s.metrics["armed_time_s"] == pytest.approx(117.2, abs=0.5)
    assert "[commander] Takeoff detected" in s.events
    assert "[commander] Landing detected" in s.events


@needs_sample
def test_sample_battery_and_gps_metrics_are_plausible():
    s = metrics.summarize(SAMPLE)
    assert 14.0 < s.metrics["batt_min_v"] < 17.0       # 4S pack, the values the log holds
    assert s.metrics["gps_sats_min"] >= 6
    assert 0 < s.metrics["gps_hdop_p95"] < 5


@needs_sample
def test_sample_has_no_arducopter_only_metrics():
    s = metrics.summarize(SAMPLE)
    for name in ("xtrack_mean_m", "vibe_p95", "att_err_p95_deg", "ekf_vel_var_p95"):
        assert name not in s.metrics


def test_log_without_arming_records_is_flagged():
    s = metrics.summarize_records([Sample(1.0, "batt", 15.0), Sample(2.0, "batt", 14.9)], "x.ulg")
    assert s.quality.get("no_armed_state") == 1.0


def test_log_with_arming_records_is_not_flagged():
    s = metrics.summarize_records([Armed(1.0, True), Armed(5.0, False)], "x.ulg")
    assert "no_armed_state" not in s.quality


def test_no_armed_flag_survives_json_round_trip():
    s = metrics.summarize_records([Sample(1.0, "batt", 15.0)], "x.ulg")
    back = metrics.FlightSummary.from_json(s.to_json())
    assert back.quality.get("no_armed_state") == 1.0


class _Data:
    def __init__(self, name, multi_id, data):
        self.name, self.multi_id, self.data = name, multi_id, data


class _Msg:
    def __init__(self, timestamp, log_level, message):
        self.timestamp, self.log_level, self.message = timestamp, log_level, message


class _FakeULog:
    def __init__(self, data_list, tagged=None):
        self.data_list = data_list
        self.start_timestamp = 0
        self.logged_messages = []
        self.logged_messages_tagged = tagged or {}


def _read_fake(monkeypatch, fake):
    monkeypatch.setattr(ulog_mod, "ULog", lambda *a, **k: fake)
    return list(ulog_mod.read_ulog("x.ulg"))


def test_every_battery_instance_counts(monkeypatch):
    bat0 = _Data("battery_status", 0, {"timestamp": [1_000_000, 2_000_000], "connected": [True, True],
                                        "voltage_v": [15.0, 14.9]})
    bat1 = _Data("battery_status", 1, {"timestamp": [1_000_000, 2_000_000], "connected": [True, True],
                                        "voltage_v": [12.0, 11.8]})
    records = _read_fake(monkeypatch, _FakeULog([bat0, bat1]))
    batt = [r.value for r in records if isinstance(r, Sample) and r.name == "batt"]
    assert min(batt) == 11.8          # the second battery's reading is in the minimum


def test_connected_battery_with_unknown_voltage_is_ignored(monkeypatch):
    bat0 = _Data("battery_status", 0, {"timestamp": [1_000_000], "connected": [True], "voltage_v": [15.0]})
    bat1 = _Data("battery_status", 1, {"timestamp": [1_000_000], "connected": [True], "voltage_v": [0.0]})
    records = _read_fake(monkeypatch, _FakeULog([bat0, bat1]))
    batt = [r.value for r in records if isinstance(r, Sample) and r.name == "batt"]
    assert batt == [15.0]             # the 0 V reading is unknown, not a real minimum


def test_gps_without_a_3d_fix_is_skipped(monkeypatch):
    gps = _Data("vehicle_gps_position", 0, {"timestamp": [1_000_000, 2_000_000, 3_000_000],
                                             "fix_type": [0, 2, 3], "satellites_used": [0, 6, 10],
                                             "hdop": [99.0, 1.5, 1.0]})
    records = _read_fake(monkeypatch, _FakeULog([gps]))
    sats = [r.value for r in records if isinstance(r, Sample) and r.name == "sats"]
    assert sats == [10.0]             # only the 3D-fix reading remains


def test_tagged_messages_are_read_as_text(monkeypatch):
    tagged = {7: [_Msg(3_000_000, ord("4"), "tagged warning")]}
    records = _read_fake(monkeypatch, _FakeULog([], tagged=tagged))
    texts = [r for r in records if isinstance(r, Text)]
    assert [(r.severity, r.text) for r in texts] == [(4, "tagged warning")]
