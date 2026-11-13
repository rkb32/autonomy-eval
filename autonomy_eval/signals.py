"""Plot any logged signal from one DataFlash (.bin) log: list the message types and fields, then read one field.

The demo uses this for the "Plot a signal" view. Only numeric fields are offered; text fields are left out.
"""
from __future__ import annotations

from pathlib import Path

SKIP_FIELDS = {"TimeUS", "I"}     # the clock itself and the instance index are not signals
TEXT_FORMATS = {"N", "Z"}         # DataFlash format characters for char arrays (text)


def _open(path: str | Path):
    from pymavlink import DFReader

    return DFReader.DFReader_binary(str(path), zero_time_base=True)


def list_signals(path: str | Path) -> dict[str, list[str]]:
    """Message types that carry a clock and at least one numeric field, with those field names.

    Reads only the header, so it is fast even for large logs.
    """
    log = _open(path)
    try:
        out: dict[str, list[str]] = {}
        for name, idx in log.name_to_id.items():
            fmt = log.formats.get(idx)
            if fmt is None or "TimeUS" not in fmt.columns:
                continue
            chars = getattr(fmt, "format", "") or ""
            numeric = []
            for i, col in enumerate(fmt.columns):
                ch = chars[i] if i < len(chars) else "f"
                if col not in SKIP_FIELDS and ch not in TEXT_FORMATS:
                    numeric.append(col)
            if numeric:
                out[name] = numeric
        return dict(sorted(out.items()))
    finally:
        log.close()   # on Windows an open log keeps its temp file locked


def read_signal(path: str | Path, msg: str, field: str) -> tuple[list[float], list[float]]:
    """(time in seconds, value) for one field of one message type, in log order."""
    log = _open(path)
    t: list[float] = []
    v: list[float] = []
    try:
        while True:
            m = log.recv_match(type=msg)
            if m is None:
                break
            try:
                value = getattr(m, field)
            except AttributeError:
                continue
            t.append(m.TimeUS / 1e6)
            v.append(float(value))
    finally:
        log.close()
    return t, v
