"""The stability, estimator and step checks as plain functions, for the web demo.

Same logic as `autonomy-eval sysid-check`, `nees-check`, `step-check` and `triage`: the CLI and the
web page both go through these builders, so they cannot disagree.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cli import _margin_flight, _missing_in_candidate, _nees_flight, _step_flight, triage_verdict
from .compare import Incomparable, Comparison, compare

CHECKS = {
    "sysid": ("Stability margins",
              "Gain and phase margin of the roll-rate loop, from a SystemID chirp. Fails if the margins "
              "shrink or cannot be measured."),
    "nees": ("Estimator consistency",
             "Is the EKF overconfident against the simulator's truth? Needs SITL logs that record truth."),
    "step": ("Step response",
             "Overshoot and settling of GUIDED position steps. Information only: it never fails a build."),
}
_BUILDERS = {"sysid": _margin_flight, "nees": _nees_flight, "step": _step_flight}
_NO_DATA = {"sysid": None, "nees": "no in-flight EKF data in", "step": "no position-controller steps found in"}


@dataclass
class CheckOutcome:
    comparison: Comparison | None = None
    error: str | None = None
    triage: tuple[str, list[str]] | None = None   # (label, items outside the baseline spread), if asked for


def run_check(kind: str, candidate: str, baseline: list[str], triage: bool = False) -> CheckOutcome:
    """Run one check on a candidate .bin against earlier .bin flights of the same mission."""
    try:
        flights = [_BUILDERS[kind](p) for p in [candidate, *baseline]]
    except ImportError as e:   # the stability check needs scipy (the [sysid] extra)
        return CheckOutcome(error=f"this check needs an extra package: {e.name}. Install it and reload.")
    if _NO_DATA[kind]:
        for f in flights:
            if not f.metrics:
                return CheckOutcome(error=f"{_NO_DATA[kind]} {f.source}")
    cand, *base = flights
    if kind == "nees":
        missing = _missing_in_candidate(cand, base)
        if missing:
            return CheckOutcome(error=f"cannot judge {cand.source}: missing {', '.join(missing)}")
    try:
        result = compare(cand, base)
    except Incomparable as e:
        return CheckOutcome(error=str(e))
    return CheckOutcome(comparison=result, triage=triage_verdict(result) if triage else None)
