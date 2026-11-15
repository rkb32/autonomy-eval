"""Flight timeline and first-seen times. The sample-dependent tests skip when the PX4 example log is not cached."""
import os
from pathlib import Path

import pytest

from autonomy_eval import metrics
from autonomy_eval.telemetry import Armed, Text

SAMPLE = Path(os.environ.get("TEMP", "")) / "autonomy-eval-samples" / "px4-quadrotor-example.ulg"
needs_sample = pytest.mark.skipif(not SAMPLE.exists(), reason="no PX4 sample log in the temp cache")


def test_band_is_min_and_max_of_the_baselines_where_two_or_more_have_data():
    baselines = [{0: 1.0, 1: 2.0}, {0: 3.0, 1: 4.0}, {0: 2.0}]
    line, band = metrics.timeline_band({0: 5.0}, baselines)
    assert line == [(0, 5.0)]
    assert band == [(0, 1.0, 3.0), (1, 2.0, 4.0)]     # second 1 has two logs, second 0 has three


def test_band_skips_seconds_seen_by_only_one_baseline():
    _, band = metrics.timeline_band({}, [{5: 1.0}, {6: 2.0}])
    assert band == []


def test_first_seen_time_is_relative_to_the_flight_start():
    records = [Armed(1.0, True), Text(3.0, 6, "hello"), Text(4.0, 6, "hello again"), Armed(10.0, False)]
    s = metrics.summarize_records(records, "x.ulg")
    assert s.event_first["hello"] == 2.0               # 3.0 s minus arming at 1.0 s
    assert "hello again" in s.event_first


@needs_sample
def test_sample_has_first_seen_times_inside_the_flight():
    s = metrics.summarize(SAMPLE)
    assert 0 <= s.event_first["[commander] Takeoff detected"] <= s.metrics["armed_time_s"]


@needs_sample
def test_sample_timeline_has_the_signals_it_should():
    series = metrics.timeline(SAMPLE)
    assert {"hdop", "batt", "sats"} <= set(series)
    assert all(0 <= sec <= 120 for sec in series["batt"])
