import os
from unittest.mock import patch

import anthropic

from autonomy_eval import bisect

OPT_IN = "AUTONOMY_EVAL_SEND_TO_AI"


def bisection():
    return bisect.Bisection(
        metric="xtrack_mean_m", base="bf08027", head="b2b1b3d",
        compare_url="https://example.invalid/compare", total_commits=1,
        suspects=[bisect.SuspectCommit("2222222b", "AC_WPNav: fix a sign error", ["libraries/AC_WPNav/AC_WPNav.cpp"])],
    )


def test_passed_key_goes_to_client_and_not_to_environ(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv(OPT_IN, "1")
    with patch("anthropic.Anthropic") as client_cls:
        bisect.ai_verdict(bisection(), api_key="TEST-KEY-1")
    client_cls.assert_called_once_with(api_key="TEST-KEY-1")
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_no_key_means_no_verdict_and_no_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv(OPT_IN, "1")
    with patch("anthropic.Anthropic") as client_cls:
        assert bisect.ai_verdict(bisection()) is None
    client_cls.assert_not_called()


def test_api_error_prints_only_the_class_name(monkeypatch, capsys):
    class Boom(Exception):
        pass

    secret = "TEST-SECRET-9f2c"
    monkeypatch.setattr(anthropic, "APIError", Boom)
    monkeypatch.setenv(OPT_IN, "1")
    with patch("anthropic.Anthropic") as client_cls:
        client_cls.return_value.messages.create.side_effect = Boom(f"request failed, body contained {secret}")
        bisect.ai_verdict(bisection(), api_key=secret)
    err = capsys.readouterr().err
    assert "Boom" in err
    assert secret not in err


def test_ai_enabled_needs_the_opt_in(monkeypatch):
    monkeypatch.setenv(OPT_IN, "1")
    assert bisect.ai_enabled(api_key="x") is True
    monkeypatch.delenv(OPT_IN, raising=False)
    assert bisect.ai_enabled(api_key="x") is False
