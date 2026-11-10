"""Step-response metrics from a position-controller flight (scripts/sitl_step.py).

The controller's desired position (PSCN/PSCE DPN/DPE) is the reference. The controller shapes the commanded
position into a ramp (its target moves over about 3 s, not in one sample), so a step is the moment the
reference leaves rest, and its size is the value it comes to rest at minus the value it left. Each step is
judged against the rest value, on each axis:

  overshoot_pct  how far past the final reference the vehicle went, as a % of the step size
  settle_s       time from the step until the vehicle stays inside a band around the reference
  ss_err_m       mean distance from the reference over the last SS_WINDOW_S of the step

The gate uses the worst step of each kind, so one bad step fails the flight.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path

HOLD_S = 25.0        # hold per step in scripts/sitl_step.py
STEP_MIN_M = 1.0     # a reference move this big is a step
MOVE_EPS_MPS = 0.2    # a reference speed above this is moving; slower plateau creep (about 0.01 m/s) is not
REST_EPS_MPS = 0.015  # walking back from the first moving sample, stop at this speed (above plateau creep)
BAND_FRAC = 0.05     # settled = within 5% of the step size ...
BAND_MIN_M = 0.1     # ... and never tighter than 10 cm
SS_WINDOW_S = 5.0    # steady-state error is measured over the last 5 s of the step


@dataclass
class StepResult:
    axis: str
    start_s: float
    size_m: float
    overshoot_pct: float
    settle_s: float
    ss_err_m: float
    settled: bool


def _step_starts(t: list[float], ref: list[float]) -> list[int]:
    """Sample indices where the reference starts moving after being at rest.

    Speed is taken in metres per second, so a slow ramp is seen as well as a jump. The start is walked back
    to the last sample that was at rest, so a slow start in an S-curve is not counted as a smaller step.
    """
    n = len(ref)
    if n < 3:
        return []
    speed = [abs(ref[i] - ref[i - 1]) / (t[i] - t[i - 1]) for i in range(1, n)]   # speed[i-1] is into sample i
    starts = []
    for i in range(1, n):
        moving_now = speed[i - 1] > MOVE_EPS_MPS
        moving_before = i >= 2 and speed[i - 2] > MOVE_EPS_MPS
        if moving_now and not moving_before:
            k = i
            while k >= 2 and speed[k - 2] > REST_EPS_MPS:      # still moving before k: the step began earlier
                k -= 1
            if not starts or k > starts[-1]:
                starts.append(k)
    return starts


def analyse_axis(axis: str, t: list[float], ref: list[float], actual: list[float],
                 hold_s: float = HOLD_S) -> list[StepResult]:
    """Find each step in the reference and measure its response. Inputs are equal-length, time-sorted."""
    changes = _step_starts(t, ref)
    results: list[StepResult] = []
    for n, k in enumerate(changes):
        stop = changes[n + 1] if n + 1 < len(changes) else len(t)
        end = k
        while end < stop and t[end] < t[k] + hold_s:
            end += 1
        window = range(k, end)
        if len(window) < 2:
            continue
        final = statistics.median(ref[i] for i in window if t[i] >= t[end - 1] - SS_WINDOW_S)
        prev = ref[k - 1]
        size = final - prev
        if abs(size) < STEP_MIN_M:
            continue
        d = math.copysign(1.0, size)
        size = abs(size)
        overshoot = max(0.0, max(d * (actual[i] - final) for i in window)) / size * 100
        band = max(BAND_FRAC * size, BAND_MIN_M)
        outside = [i for i in window if abs(actual[i] - final) > band]
        if not outside:
            settle, settled = 0.0, True
        elif outside[-1] == end - 1:
            settle, settled = t[end - 1] - t[k], False
        else:
            settle, settled = t[outside[-1] + 1] - t[k], True
        tail = [i for i in window if t[i] >= t[end - 1] - SS_WINDOW_S]
        ss = statistics.fmean(abs(actual[i] - final) for i in tail)
        results.append(StepResult(axis, t[k], size, round(overshoot, 2), round(settle, 2), round(ss, 3), settled))
    return results


def read_axes(path: str | Path) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """(time, desired, actual) per axis from PSCN/PSCE, read the same way as autonomy_eval.dataflash."""
    from pymavlink import DFReader

    series = {"N": ([], [], []), "E": ([], [], [])}
    log = DFReader.DFReader_binary(str(path), zero_time_base=True)
    try:
        while True:
            m = log.recv_match(type=["PSCN", "PSCE"])
            if m is None:
                break
            ax = "N" if m.get_type() == "PSCN" else "E"
            want, have = (m.DPN, m.PN) if ax == "N" else (m.DPE, m.PE)
            series[ax][0].append(m.TimeUS / 1e6)
            series[ax][1].append(want)
            series[ax][2].append(have)
    finally:
        log.close()      # an open handle keeps the file locked on Windows
    return series


def step_metrics(path: str | Path) -> dict[str, float]:
    """The gate's metrics for one flight. Empty if the flight has no position-controller steps."""
    results: list[StepResult] = []
    for ax, (t, ref, act) in read_axes(path).items():
        results += analyse_axis(ax, t, ref, act)
    if not results:
        return {}
    return {
        "step_overshoot_max_pct": round(max(r.overshoot_pct for r in results), 2),
        "step_settle_max_s": round(max(r.settle_s for r in results), 2),
        "step_ss_err_max_m": round(max(r.ss_err_m for r in results), 3),
    }
