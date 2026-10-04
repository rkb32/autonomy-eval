import json

from autonomy_eval.compare import compare
from autonomy_eval.metrics import FlightSummary
from autonomy_eval.report import render, to_dict


def run(label, xtrack, **params):
    return FlightSummary(source=f"{label}.bin", vehicle="ArduCopter", firmware=label, mav_type=2,
                         metrics={"xtrack_mean_m": xtrack}, params=params)


BASELINE = [run(f"b{i}", x, ATC_RAT_YAW_FLTD=0.0, WPNAV_SPEED=500.0, ZZZ_OTHER=1.0)
            for i, x in enumerate([1.00, 1.02, 0.98, 1.01, 1.00])]


def test_regression_report_lists_parameters_changed_since_the_baseline_related_first():
    candidate = run("new", 2.5, ATC_RAT_YAW_FLTD=20.0, WPNAV_SPEED=500.0, ZZZ_OTHER=2.0)
    text = render(compare(candidate, BASELINE))
    assert "PARAMETERS CHANGED SINCE THE BASELINE (1 in a family that can affect the regression, 1 other)" in text
    assert text.index("ATC_RAT_YAW_FLTD: 0 -> 20   <- related") < text.index("ZZZ_OTHER: 1 -> 2")
    assert "WPNAV_SPEED" not in text.split("PARAMETERS CHANGED")[1]


def test_clean_run_shows_no_parameter_section_even_if_parameters_differ():
    candidate = run("new", 1.01, ATC_RAT_YAW_FLTD=20.0, WPNAV_SPEED=500.0, ZZZ_OTHER=1.0)
    assert "PARAMETERS CHANGED" not in render(compare(candidate, BASELINE))


def test_regression_without_logged_parameters_has_no_parameter_section():
    candidate = run("new", 2.5)
    text = render(compare(candidate, [run(f"b{i}", x) for i, x in enumerate([1.0, 1.02, 0.98, 1.01])]))
    assert "REGRESSIONS" in text and "PARAMETERS CHANGED" not in text


def test_json_report_carries_the_parameter_changes():
    candidate = run("new", 2.5, ATC_RAT_YAW_FLTD=20.0, WPNAV_SPEED=500.0, ZZZ_OTHER=1.0)
    d = json.loads(json.dumps(to_dict(compare(candidate, BASELINE))))
    assert d["param_changes"] == [{"name": "ATC_RAT_YAW_FLTD", "before": 0.0, "after": 20.0, "related": True}]
