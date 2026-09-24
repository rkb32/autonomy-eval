"""Compare one flight against a baseline of earlier flights of the same vehicle and mission.

A metric is flagged when it sits more than Z_THRESHOLD robust standard deviations from the baseline
median. The spread is the largest of three estimates, each covering a failure of the others:

  MAD   ignores a bad run or two in the baseline (a mean/stdev would let one outlier widen the
        spread enough to hide the next regression), but collapses to zero once more than half
        the runs share one value, which is common in simulation.
  IQR   sees a second cluster of values when a metric is bimodal across builds (e.g. two possible
        scheduling orders), which MAD misses entirely.
  floor REL_FLOOR of the median, for metrics that are identical in every run; float jitter under
        a percent is ignored, a mission taking 5% longer or a count going 7 -> 8 is not.

Messages are compared two ways. Presence ignores flight phase, because a message near the arm or
disarm moment lands on either side of it from run to run. Counts are compared only for messages
the baseline has only ever seen in flight, for the same reason.

Being unusual is not the same as mattering: simulated runs are so repeatable that a 4% change in
vibration can be many standard deviations out and still be irrelevant to anyone. So a metric must
also move by at least MIN_EFFECT in absolute terms to be reported.

The build fails (for CI) on a worse metric or a warning/error message never seen before. A message
that disappears, or a new informational one, is reported but does not fail the build. Failed
pre-arm checks are never treated as alarms: ArduPilot logs them at CRITICAL and retries, and if
the vehicle never does arm, the lost flight already shows up as fewer missions and less armed time.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .metrics import PREFLIGHT_PREFIXES, WORSE_IF_HIGHER, WORSE_IF_LOWER, FlightSummary

Z_THRESHOLD = 3.5
REL_FLOOR = 0.01
ABS_FLOOR = 1e-3
MIN_BASELINE = 3
ALARMING = 4  # MAV_SEVERITY_WARNING; lower numbers are more severe

# Smallest change worth a human's attention. Counts (not listed) use 1: any change in how many
# waypoints, missions or errors there were matters. Vibration: ArduPilot considers levels under
# 30 m/s/s acceptable. EKF variances: GCS software warns at 0.5 and the EKF failsafe trips at 0.8.
MIN_EFFECT = {
    "armed_time_s": 2.0,
    "xtrack_mean_m": 0.1,
    "xtrack_p95_m": 0.25,
    "xtrack_max_m": 0.5,
    "vibe_p95": 1.0,
    "ekf_vel_var_p95": 0.05,
    "ekf_pos_horiz_var_p95": 0.05,
    "ekf_pos_vert_var_p95": 0.05,
    "ekf_compass_var_p95": 0.05,
    "gps_eph_p95_m": 0.25,
    "batt_min_v": 0.1,
}


MAV_TYPES = {1: "fixed wing", 2: "quadcopter", 4: "helicopter", 10: "ground rover", 11: "surface boat",
             12: "submarine", 13: "hexacopter", 14: "octocopter", 15: "tricopter", 20: "VTOL quadplane"}


class Incomparable(ValueError):
    pass


def vehicle_kind(vehicle: str | None, mav_type: int | None) -> str:
    return f"{vehicle} {MAV_TYPES.get(mav_type, f'MAV_TYPE {mav_type}')}"


@dataclass
class Finding:
    metric: str
    value: float
    median: float
    spread: float
    z: float
    verdict: str          # "worse" | "better" | "changed"

    @property
    def pct(self) -> float | None:
        return None if self.median == 0 else 100 * (self.value - self.median) / abs(self.median)


@dataclass
class Message:
    text: str
    severity: int

    @property
    def alarming(self) -> bool:
        return self.severity <= ALARMING and not self.text.startswith(PREFLIGHT_PREFIXES)


@dataclass
class Comparison:
    candidate: FlightSummary
    baseline: list[FlightSummary]
    findings: list[Finding] = field(default_factory=list)
    new_messages: list[Message] = field(default_factory=list)
    missing_messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict == "worse"]

    @property
    def new_alarms(self) -> list[Message]:
        return [m for m in self.new_messages if m.alarming]

    @property
    def failed(self) -> bool:
        return bool(self.regressions or self.new_alarms)

    @property
    def flag_count(self) -> int:
        return len(self.findings) + len(self.new_messages) + len(self.missing_messages)


def robust_stats(values: list[float]) -> tuple[float, float]:
    med = statistics.median(values)
    mad = statistics.median(abs(v - med) for v in values)
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return med, max(1.4826 * mad, (q3 - q1) / 1.349, REL_FLOOR * abs(med), ABS_FLOOR)


def _verdict(metric: str, delta: float) -> str:
    if metric in WORSE_IF_HIGHER:
        return "worse" if delta > 0 else "better"
    if metric in WORSE_IF_LOWER:
        return "worse" if delta < 0 else "better"
    return "changed"


def check_comparable(candidate: FlightSummary, baseline: list[FlightSummary]) -> list[str]:
    """Refuse to compare different vehicles; warn about things that weaken the comparison."""
    if len(baseline) < MIN_BASELINE:
        raise Incomparable(f"need at least {MIN_BASELINE} baseline runs, got {len(baseline)}")
    kinds = {(b.vehicle, b.mav_type) for b in baseline}
    if len(kinds) > 1:
        raise Incomparable(f"baseline mixes vehicles: {sorted(map(str, kinds))}")
    if (candidate.vehicle, candidate.mav_type) not in kinds:
        raise Incomparable(
            f"candidate is a {vehicle_kind(candidate.vehicle, candidate.mav_type)} but the baseline is a "
            f"{vehicle_kind(*next(iter(kinds)))}; comparing different vehicles is meaningless"
        )
    notes = []
    if len(baseline) < 5:
        notes.append(f"only {len(baseline)} baseline runs; spread estimates are rough")
    if candidate.firmware and candidate.firmware in {b.firmware for b in baseline}:
        notes.append(f"firmware {candidate.firmware} also appears in the baseline")
    return notes


def _seen(s: FlightSummary) -> set[str]:
    return set(s.events) | set(s.preflight)


def compare(candidate: FlightSummary, baseline: list[FlightSummary]) -> Comparison:
    result = Comparison(candidate, baseline, warnings=check_comparable(candidate, baseline))

    def flag(metric: str, value: float, history: list[float], verdict: str | None = None) -> None:
        med, spread = robust_stats(history)
        z = (value - med) / spread
        if abs(z) >= Z_THRESHOLD and abs(value - med) >= MIN_EFFECT.get(metric, 1):
            result.findings.append(Finding(metric, value, med, spread, round(z, 1), verdict or _verdict(metric, value - med)))

    for metric, value in candidate.metrics.items():
        history = [b.metrics[metric] for b in baseline if metric in b.metrics]
        if len(history) >= MIN_BASELINE:
            flag(metric, value, history)

    ever_preflight = set(candidate.preflight).union(*(b.preflight for b in baseline))
    flight_only = set().union(*(b.events for b in baseline)) - ever_preflight
    for event in sorted(flight_only):
        flag(f"event: {event}", candidate.events.get(event, 0), [b.events.get(event, 0) for b in baseline], "changed")

    seen_before = set().union(*(_seen(b) for b in baseline))
    result.new_messages = [
        Message(text, candidate.severity.get(text, 7))
        for text in sorted(_seen(candidate) - seen_before)
    ]
    result.new_messages.sort(key=lambda m: m.severity)
    result.missing_messages = sorted(set.intersection(*(_seen(b) for b in baseline)) - _seen(candidate))
    result.findings.sort(key=lambda f: (f.verdict != "worse", -abs(f.z)))
    return result


def scan(runs: list[FlightSummary]) -> list[Comparison]:
    """Leave-one-out over a history: which run is the odd one out against all the others?"""
    return [compare(run, runs[:i] + runs[i + 1:]) for i, run in enumerate(runs)]
