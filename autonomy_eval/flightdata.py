"""Readers for the views in the web demo, all from one DataFlash (.bin) log with pymavlink.

Each function reads only the message types it names. Values are as logged; units are noted where known.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from .dataflash import ERR_SUBSYSTEMS

# ArduCopter flight-mode numbers. Other vehicles use other tables, so unknown numbers print as "mode N".
COPTER_MODES = {0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED", 5: "LOITER", 6: "RTL",
                7: "CIRCLE", 9: "LAND", 16: "POSHOLD", 17: "BRAKE", 18: "THROW", 19: "AVOID_ADSB",
                20: "GUIDED_NOGPS", 21: "SMART_RTL", 22: "FLOWHOLD", 23: "FOLLOW", 24: "ZIGZAG",
                25: "SYSTEMID", 26: "AUTOROTATE", 27: "AUTO_RTL"}


def _open(path: str | Path):
    from pymavlink import DFReader

    return DFReader.DFReader_binary(str(path), zero_time_base=True)


def read_fields(path: str | Path, msg: str, fields: list[str]):
    """(time in seconds, {field: values}) for one message type, in log order. Missing values are NaN."""
    log = _open(path)
    t: list[float] = []
    cols: dict[str, list[float]] = {f: [] for f in fields}
    try:
        while True:
            m = log.recv_match(type=msg)
            if m is None:
                break
            t.append(m.TimeUS / 1e6)
            for f in fields:
                cols[f].append(float(getattr(m, f, np.nan)))
    finally:
        log.close()
    return np.asarray(t), {f: np.asarray(v) for f, v in cols.items()}


def message_columns(path: str | Path, msg: str) -> list[str]:
    """Field names of one message type in this log, or [] if the log has no such type."""
    log = _open(path)
    try:
        idx = log.name_to_id.get(msg)
        return list(log.formats[idx].columns) if idx is not None else []
    finally:
        log.close()


def last_time(path: str | Path) -> float:
    """Time of the last IMU sample, as the end of the log."""
    t, _ = read_fields(path, "IMU", ["I"])
    return float(t[-1]) if t.size else 0.0


# ---- map -------------------------------------------------------------------------------------------

def gps_track(path: str | Path):
    """(lat, lng) in degrees for every sample with a 3D fix or better (GPS Status >= 3)."""
    _, c = read_fields(path, "GPS", ["Status", "Lat", "Lng"])
    ok = (c["Status"] >= 3) & np.isfinite(c["Lat"]) & np.isfinite(c["Lng"]) & (c["Lat"] != 0)
    return c["Lat"][ok], c["Lng"][ok]


# ---- flight modes ----------------------------------------------------------------------------------

def mode_segments(path: str | Path):
    """[(start s, end s, mode number, name)] from the MODE messages, ending at the last IMU sample."""
    t, c = read_fields(path, "MODE", ["ModeNum"])
    if t.size == 0:
        return []
    end = max(last_time(path), float(t[-1]))
    out = []
    for i, (t0, num) in enumerate(zip(t, c["ModeNum"])):
        t1 = float(t[i + 1]) if i + 1 < t.size else end
        num = int(num)
        out.append((float(t0), t1, num, COPTER_MODES.get(num, f"mode {num}")))
    return out


# ---- actuator outputs ------------------------------------------------------------------------------

def actuator_outputs(path: str | Path, channels: list[str] | None = None):
    """(time, {channel: PWM values}) from RCOU, the output channels C1..C14 that the log records."""
    cols = message_columns(path, "RCOU")
    wanted = channels or [c for c in cols if c.startswith("C") and c[1:].isdigit()][:4]
    return read_fields(path, "RCOU", wanted)


# ---- firmware --------------------------------------------------------------------------------------

def firmware_info(path: str | Path) -> dict:
    """Firmware string from VER, the parameter count from PARM, and the log's length."""
    log = _open(path)
    version = None
    names: set[str] = set()
    try:
        while True:
            m = log.recv_match(type=["VER", "PARM"])
            if m is None:
                break
            if m.get_type() == "VER" and version is None:
                version = str(getattr(m, "FWS", "")).strip() or None
            else:
                names.add(str(m.Name))
    finally:
        log.close()
    return {"version": version, "parameters": len(names), "log_seconds": last_time(path)}


# ---- named fault checks ----------------------------------------------------------------------------

def _check(name: str, status: str, detail: str) -> dict:
    return {"check": name, "status": status, "detail": detail}


def fault_checks(path: str | Path) -> list[dict]:
    """Five named checks. Each says 'found', 'clear' or 'not in log', with the number behind it."""
    out = []

    # Motor failure: a motor pinned at the top of its range while the others spin, or ESC errors.
    cols = message_columns(path, "RCOU")
    chans = [c for c in cols if c.startswith("C") and c[1:].isdigit()][:4]
    if not chans:
        out.append(_check("Motor saturation", "not in log", "no RCOU messages"))
    else:
        _, c = read_fields(path, "RCOU", chans)
        top = np.nanmax(np.vstack([c[k] for k in chans]), axis=0)
        spinning = top > 1100
        frac = float((top[spinning] >= 1950).mean()) if spinning.any() else 0.0
        status = "found" if frac > 0.02 else "clear"
        out.append(_check("Motor saturation", status,
                          f"{100 * frac:.1f}% of spinning samples at or above 1950 us"))
    esc_t, esc = read_fields(path, "ESC", ["Err"]) if message_columns(path, "ESC") else (np.array([]), {})
    if esc_t.size:
        n_err = int(np.nansum(esc["Err"] != 0))
        out.append(_check("ESC errors", "found" if n_err else "clear", f"{n_err} ESC error record(s)"))
    else:
        out.append(_check("ESC errors", "not in log", "no ESC messages"))

    # GPS fix lost after it was first acquired.
    if message_columns(path, "GPS"):
        _, g = read_fields(path, "GPS", ["Status"])
        have = np.where(g["Status"] >= 3)[0]
        if have.size:
            lost = int(np.sum(g["Status"][have[0]:] < 3))
            out.append(_check("GPS fix lost", "found" if lost else "clear",
                              f"{lost} GPS sample(s) below a 3D fix after the first fix"))
        else:
            out.append(_check("GPS fix lost", "not in log", "no 3D fix in this log"))
    else:
        out.append(_check("GPS fix lost", "not in log", "no GPS messages"))

    # Battery brownout: voltage falls more than 15% below its starting value.
    if message_columns(path, "BAT"):
        _, b = read_fields(path, "BAT", ["Volt"])
        v = b["Volt"][np.isfinite(b["Volt"]) & (b["Volt"] > 0)]
        if v.size >= 10:
            start = float(np.median(v[:10]))
            drop = (start - float(v.min())) / start
            out.append(_check("Battery brownout", "found" if drop > 0.15 else "clear",
                              f"voltage fell {100 * drop:.0f}% from its start"))
        else:
            out.append(_check("Battery brownout", "not in log", "too few battery samples"))
    else:
        out.append(_check("Battery brownout", "not in log", "no BAT messages"))

    # EKF failure: any error record from an EKF subsystem.
    if message_columns(path, "ERR"):
        et, e = read_fields(path, "ERR", ["Subsys", "ECode"])
        ekf = {k for k, name in ERR_SUBSYSTEMS.items() if "EKF" in name}
        n = int(sum(1 for s, code in zip(e["Subsys"], e["ECode"]) if int(s) in ekf and code != 0))
        out.append(_check("EKF failure", "found" if n else "clear", f"{n} EKF error record(s)"))
    else:
        out.append(_check("EKF failure", "clear", "no ERR records in this log"))

    # RC link loss: link quality drops to zero after the link was up.
    if message_columns(path, "RSSI"):
        _, r = read_fields(path, "RSSI", ["RXLQ"])
        up = np.where(r["RXLQ"] > 0)[0]
        if up.size:
            lost = int(np.sum(r["RXLQ"][up[0]:] == 0))
            out.append(_check("RC link lost", "found" if lost else "clear", f"{lost} sample(s) with link quality 0"))
        else:
            out.append(_check("RC link lost", "not in log", "link never came up in this log"))
    else:
        out.append(_check("RC link lost", "not in log", "no RSSI messages"))
    return out


# ---- derived signals -------------------------------------------------------------------------------

_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Call)
_FUNCS = {"abs": np.abs, "sqrt": np.sqrt}


def derived_signal(path: str | Path, msg: str, expr: str):
    """(time, values) for an arithmetic expression over the fields of one message type, e.g. 'Volt * Curr'.

    Only numbers, field names, + - * /, unary minus, and abs() or sqrt() are allowed.
    Raises ValueError with a plain message otherwise.
    """
    tree = ast.parse(expr, mode="eval")
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"'{type(node).__name__}' is not allowed here")
        if isinstance(node, ast.Name):
            names.add(node.id)
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise ValueError("only abs() and sqrt() can be called")
    columns = message_columns(path, msg)
    if not columns:
        raise ValueError(f"the log has no {msg} messages")
    unknown = sorted(n for n in names if n not in columns and n not in _FUNCS)
    if unknown:
        raise ValueError(f"{msg} has no field {', '.join(unknown)}")
    t, cols = read_fields(path, msg, sorted(n for n in names if n not in _FUNCS))
    if t.size == 0:
        raise ValueError(f"{msg} has no records")
    env = {**cols, **_FUNCS}
    value = eval(compile(tree, "<expression>", "eval"), {"__builtins__": {}}, env)  # names checked above
    return t, np.broadcast_to(np.asarray(value, dtype=float), t.shape)


# ---- rate tracking and vibration -------------------------------------------------------------------

def rate_tracking(path: str | Path, axis: str = "R"):
    """(time, desired rate, measured rate) for one roll/pitch/yaw axis from RATE: 'R', 'P' or 'Y'."""
    desired = axis + "Des" if axis != "R" else "RDes"
    t, c = read_fields(path, "RATE", [desired, axis])
    return t, c[desired], c[axis]


def gyro_spectrum(path: str | Path, axis: str = "GyrX"):
    """(frequency Hz, power, sample rate Hz) for the first IMU instance. The top frequency is half the sample rate."""
    from scipy.signal import welch

    t, c = read_fields(path, "IMU", ["I", axis])
    if t.size < 64:
        raise ValueError("the log has too few IMU samples for a spectrum")
    inst = np.nanmin(c["I"])
    keep = c["I"] == inst
    x = c[axis][keep]
    tt = t[keep]
    dt = float(np.median(np.diff(tt)))
    if dt <= 0:
        raise ValueError("IMU timestamps do not advance")
    fs = 1.0 / dt
    f, p = welch(x - np.nanmean(x), fs=fs, nperseg=min(256, x.size))
    return f, p, fs
