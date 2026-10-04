"""What changed in the parameters between two flights, ranked by whether it could explain a regression.

After a firmware upgrade a vehicle sometimes flies worse for a reason nobody changed on purpose: a new
firmware ships new default values (this is how a loiter oscillation after a 4.4 -> 4.5 upgrade was
traced on the ArduPilot forum, by comparing the two parameter sets by hand). DataFlash (.bin) logs
record every parameter at boot (PARM), so two logs are enough to do that comparison automatically.

Like bisect.SUBSYSTEMS, the related-parameter table is a lead, not a diagnosis: a parameter in the
right family changing is a place to look, not proof of cause.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .metrics import FlightSummary

# Metric prefix -> parameter name prefixes that plausibly affect it.
RELATED: dict[str, tuple[str, ...]] = {
    "xtrack": ("WPNAV_", "PSC_", "ATC_", "NAVL1_", "WP_", "LOIT_", "CRUISE_"),
    "ekf": ("EK3_", "EK2_", "AHRS_", "GPS_"),
    "vibe": ("INS_", "MOT_"),
    "clip": ("INS_",),
    "gps": ("GPS_",),
    "hdop": ("GPS_",),
    "batt": ("BATT_", "MOT_BAT_"),
}


@dataclass
class ParamChange:
    name: str
    before: float | None    # None: the parameter did not exist in the earlier log
    after: float | None     # None: the parameter is gone in the later log
    related: bool           # in a family that can affect the metric that regressed


def is_noise(name: str) -> bool:
    """True for a parameter whose change is expected churn rather than something worth a human's eye.

    Every flight rewrites some parameters that say nothing about how the vehicle behaves: counters and
    timers it updates itself, calibration results, per-unit identifiers. Reporting those buries the one
    default that actually changed under dozens of lines that always differ.

    TODO(human): decide which parameters to treat as noise. Return True to hide a change.
    """
    return False


def _differs(a: float, b: float) -> bool:
    return not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)   # logged as float32: ignore round-off


def comparable(before: FlightSummary, after: FlightSummary) -> bool:
    """Parameters come only from .bin logs; a telemetry log or an older saved summary has none."""
    return bool(before.params) and bool(after.params)


def _changes(before: dict[str, float], after: dict[str, float], metric: str | None) -> list[ParamChange]:
    prefixes = next((p for key, p in RELATED.items() if metric and metric.startswith(key)), ())
    changes = []
    for name in sorted(before.keys() | after.keys()):
        a, b = before.get(name), after.get(name)
        if a is not None and b is not None and not _differs(a, b):
            continue
        if is_noise(name):
            continue
        changes.append(ParamChange(name, a, b, name.startswith(prefixes) if prefixes else False))
    return sorted(changes, key=lambda c: (not c.related, c.name))


def diff(before: FlightSummary, after: FlightSummary, metric: str | None = None) -> list[ParamChange]:
    """Changed, added and removed parameters, those related to `metric` first."""
    return _changes(before.params, after.params, metric)


def diff_baseline(candidate: FlightSummary, baseline: list[FlightSummary], metric: str | None = None) -> list[ParamChange]:
    """What the candidate flew with that every baseline run agreed on and it did not.

    A parameter the baseline runs disagree about says nothing about what changed since then (the
    operator was already varying it), so it is left out. Baseline runs with no parameters (telemetry
    logs, old saved summaries) are ignored; with none left, or none for the candidate, nothing is returned.
    """
    runs = [b.params for b in baseline if b.params]
    if not runs or not candidate.params:
        return []
    in_baseline = set().union(*runs)
    agreed: dict[str, float] = {}
    for name in in_baseline:
        values = [r.get(name) for r in runs]            # None: this baseline run lacked the parameter
        if None not in values and not any(_differs(values[0], v) for v in values[1:]):
            agreed[name] = values[0]
    undecided = in_baseline - agreed.keys()              # baseline runs disagreed (or some lacked it)
    ours = {n: v for n, v in candidate.params.items() if n not in undecided}
    return _changes(agreed, ours, metric)
