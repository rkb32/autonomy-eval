import json
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from autonomy_eval import bisect
from autonomy_eval.metrics import FlightSummary


def build(sha, **metrics):
    return FlightSummary(source=f"{sha}.json", vehicle="ArduRover", firmware=sha, mav_type=11, metrics=metrics)


COMPARE_PAYLOAD = {
    "commits": [
        {"sha": "1111111aaaa", "commit": {"message": "waf: enable warnings as errors\n\nmore detail"}},
        {"sha": "2222222bbbb", "commit": {"message": "AC_WPNav: fix a sign error in the L1 gain"}},
    ]
}
DETAILS = {
    "1111111aaaa": {"files": [{"filename": "Tools/ardupilotwaf/boards.py"}]},
    "2222222bbbb": {"files": [{"filename": "libraries/AC_WPNav/AC_WPNav.cpp"}]},
}


def fake_get(url):
    if "compare" in url:
        return COMPARE_PAYLOAD
    sha = url.rsplit("/", 1)[-1]
    return DETAILS[sha]


def test_unmapped_metric_is_not_localized():
    assert bisect.localize("armed_time_s", "bf08027", "b2b1b3d") is None


def test_non_sha_label_is_not_localized():
    assert bisect.localize("xtrack_mean_m", "some-file.tlog", "b2b1b3d") is None


def test_localize_keeps_only_commits_touching_the_relevant_subsystem():
    with patch.object(bisect, "_get", side_effect=fake_get):
        b = bisect.localize("xtrack_mean_m", "bf08027", "b2b1b3d")
    assert b.total_commits == 2
    assert [s.sha for s in b.suspects] == ["2222222b"]
    assert b.suspects[0].files == ["libraries/AC_WPNav/AC_WPNav.cpp"]


def test_localize_reports_no_suspects_when_nothing_touches_the_subsystem():
    only_tooling = {"commits": [COMPARE_PAYLOAD["commits"][0]]}
    with patch.object(bisect, "_get", side_effect=lambda url: only_tooling if "compare" in url else DETAILS["1111111aaaa"]):
        b = bisect.localize("xtrack_mean_m", "bf08027", "b2b1b3d")
    assert b.total_commits == 1
    assert b.suspects == []
    assert b.error is None


def test_localize_reports_network_failure_without_raising():
    with patch.object(bisect, "_get", side_effect=urllib.error.URLError("no network")):
        b = bisect.localize("xtrack_mean_m", "bf08027", "b2b1b3d")
    assert b.error and "GitHub" in b.error


def test_localize_refuses_to_check_a_huge_range():
    huge = {"commits": [{"sha": f"{i:07x}", "commit": {"message": "m"}} for i in range(bisect.MAX_COMMITS + 1)]}
    with patch.object(bisect, "_get", return_value=huge) as get:
        b = bisect.localize("xtrack_mean_m", "bf08027", "b2b1b3d")
    assert b.error and str(bisect.MAX_COMMITS + 1) in b.error
    get.assert_called_once()  # never fetched individual commits


def test_neighbor_build_prefers_the_earlier_build():
    a, b, c = build("aaaaaaa"), build("bbbbbbb"), build("ccccccc")
    assert bisect.neighbor_build(b, [a, b, c]) == (a, b)


def test_neighbor_build_falls_back_to_the_next_build_when_first_in_history():
    # bf080274 in the real data is the oldest build in the set: there's nothing earlier to bisect
    # against, so the range checked is (candidate, next) instead of (previous, candidate).
    a, b = build("aaaaaaa"), build("bbbbbbb")
    assert bisect.neighbor_build(a, [a, b]) == (a, b)


def test_neighbor_build_none_for_a_lone_run_or_unknown_run():
    a = build("aaaaaaa")
    assert bisect.neighbor_build(a, [a]) is None
    assert bisect.neighbor_build(build("zzzzzzz"), [a]) is None


def test_ai_verdict_skipped_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    b = bisect.Bisection("xtrack_mean_m", "bf08027", "b2b1b3d", "https://x")
    assert bisect.ai_verdict(b) is None


def test_ai_verdict_skipped_when_localize_already_errored(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    b = bisect.Bisection("xtrack_mean_m", "bf08027", "b2b1b3d", "https://x", error="couldn't reach GitHub")
    assert bisect.ai_verdict(b) is None


def test_ai_verdict_parses_the_model_json_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    b = bisect.Bisection("xtrack_mean_m", "bf08027", "b2b1b3d", "https://x", total_commits=2,
                          suspects=[bisect.SuspectCommit("2222222b", "AC_WPNav: fix a sign error", ["libraries/AC_WPNav/AC_WPNav.cpp"])])
    reply = json.dumps({"verdict": "likely true positive", "reason": "touches the L1 gain directly"})
    fake_response = SimpleNamespace(content=[SimpleNamespace(type="text", text=reply)])
    with patch("anthropic.Anthropic") as Client:
        Client.return_value.messages.create.return_value = fake_response
        v = bisect.ai_verdict(b)
    assert v.label == "likely true positive"
    assert "L1 gain" in v.reason


def test_ai_verdict_fails_closed_on_bad_json(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    b = bisect.Bisection("xtrack_mean_m", "bf08027", "b2b1b3d", "https://x")
    fake_response = SimpleNamespace(content=[SimpleNamespace(type="text", text="not json")])
    with patch("anthropic.Anthropic") as Client:
        Client.return_value.messages.create.return_value = fake_response
        assert bisect.ai_verdict(b) is None


def test_for_comparison_needs_a_regression():
    from autonomy_eval.compare import Comparison
    c = Comparison(candidate=build("aaaaaaa"), baseline=[])
    assert bisect.for_comparison(c, [build("aaaaaaa")]) is None
