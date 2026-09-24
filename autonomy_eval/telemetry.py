"""Read a flight log, in either format, as one stream of format-independent records.

ArduPilot records the same flight two ways: the MAVLink telemetry a ground station saves (.tlog)
and the autopilot's own onboard DataFlash log (.bin). They name everything differently; readers
here translate both into the records below, so metrics are computed identically for either.

The format is detected from the file's first bytes, not its extension.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union


@dataclass
class Text:
    t: float
    severity: int | None      # None: DataFlash MSG records carry no severity
    text: str


@dataclass
class Armed:
    t: float
    armed: bool


@dataclass
class Mode:
    t: float
    mode: int


@dataclass
class VehicleType:
    t: float
    mav_type: int


@dataclass
class Reboot:
    t: float


@dataclass
class Sample:
    """One reading of a named signal: xtrack, ekf_vel, ekf_ph, ekf_pv, ekf_mag, vibe, clip, sats, hdop, batt."""
    t: float
    name: str
    value: float


@dataclass
class Corrupt:
    t: float


Record = Union[Text, Armed, Mode, VehicleType, Reboot, Sample, Corrupt]


class Clock:
    """Boot-relative time restarts near zero when the autopilot reboots; stitch reboots into one timeline."""

    def __init__(self) -> None:
        self.offset = 0.0
        self.last: float | None = None
        self.reboots = 0

    def update(self, seconds: float) -> bool:
        """Advance to `seconds` since boot; True if that means the autopilot just rebooted."""
        rebooted = self.last is not None and seconds + 1.0 < self.last
        if rebooted:
            self.offset += self.last
            self.reboots += 1
        self.last = seconds
        return rebooted

    @property
    def now(self) -> float:
        return self.offset + (self.last or 0.0)


class TextAssembler:
    """Text over 50 chars is split into chunks sharing a non-zero id; id 0 means unchunked.
    MAVLink STATUSTEXT (id, chunk_seq) and DataFlash MSG (ID, Seq) use the same scheme."""

    def __init__(self) -> None:
        self.id: int | None = None
        self.parts: list[str] = []
        self.start: tuple[float, int | None] = (0.0, None)

    def feed(self, t: float, msg_id: int, severity: int | None, text: str) -> list[Text]:
        if msg_id == 0:
            return self.flush() + [Text(t, severity, text)]
        out = self.flush() if msg_id != self.id else []
        if not self.parts:
            self.id, self.start = msg_id, (t, severity)
        self.parts.append(text)
        return out

    def flush(self) -> list[Text]:
        if not self.parts:
            return []
        t, sev = self.start
        text = "".join(self.parts)
        self.id, self.parts = None, []
        return [Text(t, sev, text)]


def is_dataflash(path: str | Path) -> bool:
    with open(path, "rb") as f:
        return f.read(3) == b"\xa3\x95\x80"   # DataFlash message header + the FMT message id


def read(path: str | Path) -> Iterator[Record]:
    if is_dataflash(path):
        from .dataflash import read_dataflash
        return read_dataflash(str(path))
    from .tlog import read_tlog
    return read_tlog(str(path))
