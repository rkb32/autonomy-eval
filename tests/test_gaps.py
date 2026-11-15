"""Gap fixes: DataFlash text severity (G2) and the opt-in simulation floor (G4)."""
from autonomy_eval import compare as cmp_mod
from autonomy_eval.dataflash import CRITICAL, ERROR, WARNING, text_severity
from autonomy_eval.metrics import FlightSummary


def _run(label: str, vibe: float) -> FlightSummary:
    return FlightSummary(source=f"{label}.bin", vehicle="ArduRover", mav_type=10, firmware=label,
                         metrics={"vibe_p95": vibe, "armed_time_s": 100.0})


# G2: severity from wording
def test_failsafe_and_ekf_failures_are_warnings():
    assert text_severity("Failsafe: GPS lost") == WARNING
    assert text_severity("EKF3 IMU0 is unhealthy") == WARNING


def test_crash_is_critical_and_error_wording_is_error():
    assert text_severity("Crash detected") == CRITICAL
    assert text_severity("Error in sensor init") == ERROR


def test_plain_text_gets_no_severity():
    assert text_severity("Hello from the rover") is None


# G4: sim floor
def test_default_floor_hides_a_mild_change_on_a_quiet_baseline():
    baseline = [_run("b1", 0.05), _run("b2", 0.06), _run("b3", 0.04), _run("b4", 0.05)]
    candidate = _run("c", 0.30)
    result = cmp_mod.compare(candidate, baseline)
    assert not any(f.metric == "vibe_p95" for f in result.findings)


def test_sim_floor_flags_the_same_change():
    baseline = [_run("b1", 0.05), _run("b2", 0.06), _run("b3", 0.04), _run("b4", 0.05)]
    candidate = _run("c", 0.30)
    result = cmp_mod.compare(candidate, baseline, sim=True)
    assert any(f.metric == "vibe_p95" for f in result.findings)


def test_identical_run_is_not_flagged_in_either_mode():
    baseline = [_run("b1", 0.05), _run("b2", 0.06), _run("b3", 0.04), _run("b4", 0.05)]
    candidate = _run("c", 0.05)
    for sim in (False, True):
        assert not any(f.metric == "vibe_p95" for f in cmp_mod.compare(candidate, baseline, sim=sim).findings)
