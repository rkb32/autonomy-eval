"""EKF innovation health: how far the EKF's predictions are from the sensors it fuses.

XKF3 logs the innovation (measurement minus prediction) for each fused quantity. Large innovations mean the
EKF's model disagrees with its sensors: a noisy or biased sensor, or a model that is no longer right. This
module summarizes the 95th percentile of the absolute innovation per axis, for core 0 only, after the
first SETTLE_S seconds of the log (filter convergence) and on the ground (altitude below AIR_MIN_M).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SETTLE_S = 30.0
AIR_MIN_M = 5.0
AXES = ("IVN", "IVE", "IVD", "IPN", "IPE", "IPD")
NAMES = {"IVN": "innov_vn_p95", "IVE": "innov_ve_p95", "IVD": "innov_vd_p95",
         "IPN": "innov_pn_p95", "IPE": "innov_pe_p95", "IPD": "innov_pd_p95"}


def p95_abs(values) -> float | None:
    v = np.abs(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 95)) if len(v) else None


def read_innovations(path: str | Path):
    """Innovation series for core 0, restricted to in-flight samples (altitude above AIR_MIN_M, after SETTLE_S)."""
    from pymavlink import mavutil

    t_alt, alt_d = [], []           # XKF1 down position, the altitude source for the mask
    t_inn, inn = [], {k: [] for k in AXES}
    m = mavutil.mavlink_connection(str(path))
    try:
        while True:
            x = m.recv_match(type=["XKF1", "XKF3"])
            if x is None:
                break
            if getattr(x, "C", 0) != 0:
                continue
            t = x.TimeUS / 1e6
            if x.get_type() == "XKF1":
                t_alt.append(t)
                alt_d.append(x.PD)
            else:
                t_inn.append(t)
                for k in AXES:
                    inn[k].append(getattr(x, k))
    finally:
        m.close()
    if not t_inn or not t_alt:
        return {k: np.asarray([]) for k in AXES}
    t_inn = np.asarray(t_inn)
    t_alt = np.asarray(t_alt)
    alt_d = np.asarray(alt_d, dtype=float)
    # the latest XKF1 altitude at or before each innovation sample
    idx = np.clip(np.searchsorted(t_alt, t_inn, side="right") - 1, 0, len(t_alt) - 1)
    keep = (alt_d[idx] <= -AIR_MIN_M) & (t_inn - t_alt[0] >= SETTLE_S)
    return {k: np.asarray(inn[k], dtype=float)[keep] for k in AXES}


def innovation_metrics(path: str | Path) -> dict[str, float]:
    """p95 absolute innovation per axis, in flight only. Empty if the log has no in-flight XKF3 data."""
    series = read_innovations(path)
    out = {}
    for k in AXES:
        v = p95_abs(series[k])
        if v is not None:
            out[NAMES[k]] = round(v, 4)
    return out
