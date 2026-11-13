"""Logging and timing health from one DataFlash (.bin) log.

A firmware change can make the flight worse without changing the tuning: the scheduler loop can start
overrunning, the CPU load can climb, log blocks can drop, or the IMU clock can skip. This module reads
those numbers from the log's own PM, DSF and IMU messages.

Units are as logged. PM.MaxT and PM.Load are read as-is; this module does not convert them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def gap_ratio(times_us) -> float | None:
    """Largest gap between consecutive samples, as a multiple of the median gap. None if too few samples."""
    t = np.sort(np.asarray(times_us, dtype=float))
    if t.size < 3:
        return None
    d = np.diff(t)
    d = d[d > 0]
    if d.size < 2:
        return None
    return float(d.max() / np.median(d))


def overrun_fraction(nl, nlon) -> float | None:
    """Share of scheduler loops that overran: sum(NLon) / sum(NL). None when no loops were counted."""
    total = float(sum(nl))
    if total <= 0:
        return None
    return float(sum(nlon)) / total


def timing_health(path: str | Path) -> dict | None:
    """Timing numbers for one flight, or None if the log has no scheduler (PM) messages."""
    from pymavlink import DFReader

    log = DFReader.DFReader_binary(str(path), zero_time_base=True)
    try:
        nl: list[float] = []
        nlon: list[float] = []
        max_t = 0.0
        max_load = 0.0
        have_pm = False
        dropped: int | None = None
        imu: dict[int, list[float]] = {}
        while True:
            m = log.recv_match(type=["PM", "DSF", "IMU"])
            if m is None:
                break
            kind = m.get_type()
            if kind == "PM":
                have_pm = True
                nl.append(float(getattr(m, "NL", 0)))
                nlon.append(float(getattr(m, "NLon", 0)))
                max_t = max(max_t, float(getattr(m, "MaxT", 0)))
                max_load = max(max_load, float(getattr(m, "Load", 0)))
            elif kind == "DSF":
                dropped = (dropped or 0) + int(getattr(m, "Dp", 0))
            else:
                imu.setdefault(int(getattr(m, "I", 0)), []).append(m.TimeUS)
    finally:
        log.close()
    if not have_pm:
        return None
    frac = overrun_fraction(nl, nlon)
    gaps = [g for g in (gap_ratio(ts) for ts in imu.values()) if g is not None]
    return {
        "overrun_pct": None if frac is None else 100.0 * frac,
        "max_loop_time": max_t,
        "max_cpu_load": max_load,
        "dropped_blocks": dropped,
        "worst_imu_gap": max(gaps) if gaps else None,
    }
