"""Is the EKF honest about its own uncertainty? Normalized estimation error squared (NEES) against SITL truth.

For each position and velocity axis, the EKF's estimate (XKF1) minus the simulator's truth (SIM2), squared
and divided by the EKF's own variance for that state (XKV1/XKV2, the diagonal of its covariance):

  NEES = (estimate - truth)^2 / variance

A consistent filter averages about 1 per axis. Much above 1 means the filter is overconfident: its
variances are too small for the errors it actually makes. Only the diagonal of the covariance is logged,
so this is an approximation of the full-covariance NEES.

Samples are used only in flight: altitude above AIR_MIN_M, after the first SETTLE_S seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

AIR_MIN_M = 5.0
SETTLE_S = 30.0
# EKF3 state index of each axis (XKV1 holds states 0-11, XKV2 holds states 12-23)
AXES = {"pos_n": 7, "pos_e": 8, "pos_d": 9, "vel_n": 4, "vel_e": 5, "vel_d": 6}


@dataclass
class EkfLog:
    t_est: np.ndarray
    est: dict[str, np.ndarray]      # XKF1 PN, PE, PD, VN, VE, VD
    t_truth: np.ndarray
    truth: dict[str, np.ndarray]    # SIM2 PN, PE, PD, VN, VE, VD
    t_var: np.ndarray
    var: np.ndarray                 # (samples, 24) EKF state variances


def nees_series(e: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-sample NEES for one axis, with non-positive or non-finite variances dropped."""
    ok = np.isfinite(p) & (p > 0) & np.isfinite(e)
    return (e[ok] ** 2) / p[ok]


def read_ekf_log(path: str | Path) -> EkfLog:
    from pymavlink import mavutil

    m = mavutil.mavlink_connection(str(path))
    xkf1 = {k: [] for k in ("t", "PN", "PE", "PD", "VN", "VE", "VD")}
    sim2 = {k: [] for k in ("t", "PN", "PE", "PD", "VN", "VE", "VD")}
    xkv: dict[float, list[float]] = {}
    try:
        while True:
            x = m.recv_match(type=["XKF1", "SIM2", "XKV1", "XKV2"])
            if x is None:
                break
            kind = x.get_type()
            if getattr(x, "C", 0) != 0:          # core 0 only: XKV1/XKV2 are logged for core 0 alone
                continue
            if kind == "XKF1":
                xkf1["t"].append(x.TimeUS / 1e6)
                for k in ("PN", "PE", "PD", "VN", "VE", "VD"):
                    xkf1[k].append(getattr(x, k))
            elif kind == "SIM2":
                sim2["t"].append(x.TimeUS / 1e6)
                for k in ("PN", "PE", "PD", "VN", "VE", "VD"):
                    sim2[k].append(getattr(x, k))
            else:
                t = x.TimeUS / 1e6
                vals = xkv.setdefault(t, [np.nan] * 24)
                names = [f"V{i:02d}" for i in range(12)] if kind == "XKV1" else [f"V{i:02d}" for i in range(12, 24)]
                base = 0 if kind == "XKV1" else 12
                for j, name in enumerate(names):
                    vals[base + j] = getattr(x, name)
    finally:
        m.close()                                 # an open handle keeps the file locked on Windows
    t_var = np.asarray(sorted(xkv))
    var = np.asarray([xkv[t] for t in t_var], dtype=float)
    return EkfLog(
        t_est=np.asarray(xkf1["t"]),
        est={k: np.asarray(xkf1[k], dtype=float) for k in ("PN", "PE", "PD", "VN", "VE", "VD")},
        t_truth=np.asarray(sim2["t"]),
        truth={k: np.asarray(sim2[k], dtype=float) for k in ("PN", "PE", "PD", "VN", "VE", "VD")},
        t_var=t_var,
        var=var,
    )


def ekf_nees(log: EkfLog) -> dict[str, float]:
    """Mean NEES per axis over the in-flight part of the log. Empty if there is not enough data."""
    if len(log.t_est) < 10 or len(log.t_truth) < 10 or len(log.t_var) < 10:
        return {}
    t0 = log.t_est[0]
    air = (log.est["PD"] <= -AIR_MIN_M) & (log.t_est - t0 >= SETTLE_S)
    if air.sum() < 10:
        return {}
    t = log.t_est[air]
    # the latest variance sample at or before each estimate (XKV is logged at about 2 Hz)
    idx_var = np.clip(np.searchsorted(log.t_var, t, side="right") - 1, 0, len(log.t_var) - 1)
    out = {}
    for name, state in AXES.items():          # name like "pos_n": kind and axis letter
        kind, axis = name.split("_")
        field = ("P" if kind == "pos" else "V") + axis.upper()
        est = log.est[field][air]
        truth = np.interp(t, log.t_truth, log.truth[field])
        series = nees_series(est - truth, log.var[idx_var, state])
        if len(series):
            out[name] = round(float(np.mean(series)), 3)
    return out


def nees_from_file(path: str | Path) -> dict[str, float]:
    return ekf_nees(read_ekf_log(path))
