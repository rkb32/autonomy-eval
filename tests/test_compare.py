import pytest

from autonomy_eval.compare import Incomparable, compare, robust_stats, scan
from autonomy_eval.metrics import FlightSummary


def run(label, vehicle="ArduRover", mav_type=11, events=None, preflight=(), severity=None, **metrics):
    base = {"armed_time_s": 244.0, "xtrack_mean_m": 1.2, "waypoints_reached": 17, "vibe_p95": 0.45}
    return FlightSummary(
        source=f"{label}.tlog", vehicle=vehicle, firmware=label, mav_type=mav_type,
        metrics=base | metrics, events=events or {"Sailboat: Tacking": 7},
        preflight=list(preflight), severity=severity or {},
    )


def history(n=7, **kw):
    jitter = [0.0, 0.3, -0.4, 0.1, -0.2, 0.2, 0.0, 0.4, -0.1]
    return [run(f"b{i}", armed_time_s=244.0 + jitter[i], **kw) for i in range(n)]


def test_identical_run_is_clean():
    c = compare(run("new"), history())
    assert not c.findings and not c.failed


def test_slower_mission_is_a_regression():
    c = compare(run("new", armed_time_s=257.0), history())
    assert [(f.metric, f.verdict) for f in c.findings] == [("armed_time_s", "worse")]
    assert c.failed


def test_better_is_reported_but_does_not_fail():
    c = compare(run("new", xtrack_mean_m=0.6), history())
    assert [f.verdict for f in c.findings] == ["better"] and not c.failed


def test_an_outlier_in_the_baseline_does_not_hide_the_next_regression():
    baseline = history(6) + [run("bad", armed_time_s=300.0)]
    assert compare(run("new", armed_time_s=257.0), baseline).failed


def test_mad_collapses_on_a_bimodal_metric_but_iqr_catches_it():
    # 4 builds land on 5.30 and 3 on 5.64: MAD is 0, so without the IQR every 5.64 would be flagged
    values = [5.30, 5.30, 5.30, 5.30, 5.64, 5.64, 5.64]
    med, spread = robust_stats(values)
    assert med == 5.30 and (5.64 - med) / spread < 3.5


def test_statistically_unusual_but_negligible_change_is_ignored():
    # every baseline run has vibration 0.45 exactly; 0.49 is far outside that, and irrelevant
    c = compare(run("new", vibe_p95=0.49), [run(f"b{i}") for i in range(7)])
    assert not c.findings


def test_count_change_of_one_is_reported():
    c = compare(run("new", events={"Sailboat: Tacking": 8}), history())
    assert [(f.metric, f.value, f.median) for f in c.findings] == [("event: Sailboat: Tacking", 8, 7)]
    assert not c.failed                        # an extra tack is a change, not by itself a failure


def test_new_error_fails_but_new_info_does_not():
    err = compare(run("new", preflight=["EKF3 lane switch N"], severity={"EKF3 lane switch N": 3}), history())
    assert [m.text for m in err.new_alarms] == ["EKF3 lane switch N"] and err.failed
    info = compare(run("new", preflight=["hello"], severity={"hello": 6}), history())
    assert info.new_messages and not info.failed


def test_failed_prearm_check_is_never_an_alarm():
    c = compare(run("new", preflight=["PreArm: Mode not armable"], severity={"PreArm: Mode not armable": 2}), history())
    assert c.new_messages and not c.new_alarms and not c.failed


def test_message_near_arming_is_not_missing_just_because_it_changed_phase():
    baseline = history(events={"Sailboat: Tacking": 7, "EKF3 IMU0 is using GPS": 1})
    c = compare(run("new", preflight=["EKF3 IMU0 is using GPS"]), baseline)
    assert not c.missing_messages and not c.findings


def test_same_firmware_name_different_vehicle_is_refused():
    with pytest.raises(Incomparable, match="surface boat.*ground rover"):
        compare(run("new", mav_type=11), history(mav_type=10))


def test_too_few_baseline_runs_is_refused():
    with pytest.raises(Incomparable, match="at least 3"):
        compare(run("new"), history(2))


def test_scan_finds_the_odd_one_out():
    runs = history(7) + [run("odd", armed_time_s=257.0, events={"Sailboat: Tacking": 8})]
    failed = [c.candidate.label for c in scan(runs) if c.failed]
    assert failed == ["odd"]
