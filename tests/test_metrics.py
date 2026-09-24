import glob
from pathlib import Path

import pytest

from autonomy_eval.compare import scan
from autonomy_eval.metrics import FlightSummary, in_intervals, load, normalize_event, percentile

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
RAW = Path(__file__).resolve().parents[1] / "data" / "autotest"


@pytest.mark.parametrize("raw, normalized", [
    ("Reached waypoint #4", "Reached waypoint #N"),
    ("EKF3 IMU0 is using GPS", "EKF3 IMU0 is using GPS"),          # identifiers keep their digits
    ("GPS 1: probing for u-blox at 230400 baud", "GPS N: probing for u-blox at N baud"),
    ("03e9f14ee0eb4ea59d1ea6959ad8b352", "<hex>"),
    ("speed: 1.5 m/s", "speed: N m/s"),
])
def test_normalize_event(raw, normalized):
    assert normalize_event(raw) == normalized


def test_percentile_ignores_a_single_spike():
    assert percentile([0.4] * 99 + [9.0], 0.95) == 0.4


def test_arm_message_just_before_the_heartbeat_flips_counts_as_in_flight():
    flights = [(40.9, 123.3)]
    assert in_intervals(40.2, flights)       # heartbeats are 1 Hz; the armed flag lags
    assert not in_intervals(38.0, flights)


def test_summary_json_round_trip():
    s = FlightSummary("x.tlog", "ArduRover", "abc123", 11, {"armed_time_s": 1.0}, {"e": 1}, ["p"], {"e": 6}, {})
    assert FlightSummary.from_json(s.to_json()) == s


def _examples(kind):
    return [load(p) for p in sorted(glob.glob(str(EXAMPLES / kind / "*.json")))]


def test_real_sailboat_history_flags_only_the_extra_tack_build():
    comparisons = scan(_examples("ardurover-surface-boat"))
    assert [c.candidate.label for c in comparisons if c.failed] == ["bf080274"]
    odd = next(c for c in comparisons if c.candidate.label == "bf080274")
    assert {f.metric for f in odd.regressions} == {"xtrack_mean_m", "armed_time_s"}


def test_real_rover_history_has_no_false_regressions():
    assert not any(c.failed for c in scan(_examples("ardurover-ground-rover")))


@pytest.mark.skipif(not RAW.exists(), reason="raw logs not downloaded (scripts/fetch_autotest.py)")
def test_parser_still_reproduces_the_committed_summaries():
    for summary in EXAMPLES.glob("*/*.json"):
        expected = FlightSummary.from_json(summary.read_text())
        log = RAW / Path(expected.source).name
        if log.exists():
            got = load(log)
            assert (got.vehicle, got.firmware, got.mav_type) == (expected.vehicle, expected.firmware, expected.mav_type)
            assert got.metrics == expected.metrics and got.events == expected.events
            return
    pytest.skip("no raw log matching a committed summary")
