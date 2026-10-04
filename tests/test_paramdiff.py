from autonomy_eval import paramdiff
from autonomy_eval.metrics import FlightSummary


def flight(**params):
    return FlightSummary(source="x.bin", vehicle="ArduCopter", params=params)


def test_unchanged_parameters_are_not_reported():
    a = flight(ATC_RAT_YAW_P=0.18, WPNAV_SPEED=500.0)
    assert paramdiff.diff(a, flight(ATC_RAT_YAW_P=0.18, WPNAV_SPEED=500.0)) == []


def test_float32_round_off_is_not_a_change():
    assert paramdiff.diff(flight(ATC_RAT_YAW_P=0.18), flight(ATC_RAT_YAW_P=0.180000007)) == []


def test_changed_added_and_removed_are_all_reported():
    changes = {c.name: c for c in paramdiff.diff(
        flight(WPNAV_SPEED=500.0, OLD_PARAM=1.0), flight(WPNAV_SPEED=800.0, NEW_PARAM=2.0))}
    assert (changes["WPNAV_SPEED"].before, changes["WPNAV_SPEED"].after) == (500.0, 800.0)
    assert changes["OLD_PARAM"].after is None
    assert changes["NEW_PARAM"].before is None


def test_parameters_in_the_regressed_metrics_family_come_first():
    before = flight(AAA_UNRELATED=1.0, INS_HNTCH_BW=40.0)
    after = flight(AAA_UNRELATED=2.0, INS_HNTCH_BW=20.0)
    changes = paramdiff.diff(before, after, metric="vibe_p95")
    assert [c.name for c in changes] == ["INS_HNTCH_BW", "AAA_UNRELATED"]
    assert [c.related for c in changes] == [True, False]


def test_no_metric_means_nothing_is_marked_related():
    assert not any(c.related for c in paramdiff.diff(flight(INS_X=1.0), flight(INS_X=2.0)))


def test_logs_without_parameters_are_not_comparable():
    assert not paramdiff.comparable(flight(), flight(A=1.0))
    assert paramdiff.comparable(flight(A=1.0), flight(A=2.0))


def test_baseline_diff_reports_what_the_candidate_changed_against_a_unanimous_baseline():
    base = [flight(ATC_RAT_YAW_FLTD=0.0, WPNAV_SPEED=500.0), flight(ATC_RAT_YAW_FLTD=0.0, WPNAV_SPEED=500.0)]
    changes = paramdiff.diff_baseline(flight(ATC_RAT_YAW_FLTD=20.0, WPNAV_SPEED=500.0), base, "xtrack_mean_m")
    assert [(c.name, c.before, c.after, c.related) for c in changes] == [("ATC_RAT_YAW_FLTD", 0.0, 20.0, True)]


def test_baseline_diff_ignores_parameters_the_baseline_runs_disagree_about():
    base = [flight(WPNAV_SPEED=500.0), flight(WPNAV_SPEED=900.0)]
    assert paramdiff.diff_baseline(flight(WPNAV_SPEED=700.0), base) == []


def test_baseline_diff_does_not_report_a_parameter_missing_from_some_baseline_runs_as_added():
    base = [flight(A=1.0, NEW_IN_LATER_FIRMWARE=2.0), flight(A=1.0)]
    assert paramdiff.diff_baseline(flight(A=1.0, NEW_IN_LATER_FIRMWARE=2.0), base) == []


def test_baseline_diff_reports_a_brand_new_and_a_removed_parameter():
    base = [flight(OLD=1.0), flight(OLD=1.0)]
    changes = {c.name: c for c in paramdiff.diff_baseline(flight(NEW=3.0), base)}
    assert changes["OLD"].after is None and changes["NEW"].before is None


def test_baseline_diff_is_empty_when_either_side_has_no_parameters():
    assert paramdiff.diff_baseline(flight(), [flight(A=1.0), flight(A=1.0)]) == []
    assert paramdiff.diff_baseline(flight(A=1.0), [flight(), flight()]) == []


def test_old_saved_summaries_without_params_still_load():
    s = FlightSummary.from_json('{"source": "old.json", "vehicle": "ArduRover", "metrics": {"vibe_p95": 3.0}}')
    assert s.params == {}
