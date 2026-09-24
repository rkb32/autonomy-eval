"""Read a MAVLink telemetry log (.tlog) as a clean stream of messages from the vehicle only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from pymavlink import mavutil

MAV_TYPE_GCS = 6
MAV_AUTOPILOT_INVALID = 8


@dataclass
class Text:
    t: float
    severity: int
    text: str


class Clock:
    """time_boot_ms restarts near zero when the autopilot reboots; stitch reboots into one timeline."""

    def __init__(self) -> None:
        self.offset = 0.0
        self.last: float | None = None
        self.reboots = 0

    def update(self, boot_ms: int) -> float:
        s = boot_ms / 1000
        if self.last is not None and s + 1.0 < self.last:
            self.offset += self.last
            self.reboots += 1
        self.last = s
        return self.now

    @property
    def now(self) -> float:
        return self.offset + (self.last or 0.0)


class TextAssembler:
    """MAVLink 2 splits STATUSTEXT over 50 chars into chunks sharing a non-zero id; id 0 means unchunked."""

    def __init__(self) -> None:
        self.id: int | None = None
        self.parts: list[str] = []
        self.start: tuple[float, int] = (0.0, 0)

    def feed(self, t: float, msg) -> list[Text]:
        msg_id = getattr(msg, "id", 0)
        if msg_id == 0:
            return self.flush() + [Text(t, msg.severity, msg.text)]
        out = self.flush() if msg_id != self.id else []
        if not self.parts:
            self.id, self.start = msg_id, (t, msg.severity)
        self.parts.append(msg.text)
        return out

    def flush(self) -> list[Text]:
        if not self.parts:
            return []
        t, sev = self.start
        text = "".join(self.parts)
        self.id, self.parts = None, []
        return [Text(t, sev, text)]


def _is_vehicle_heartbeat(msg) -> bool:
    return msg.type != MAV_TYPE_GCS and msg.autopilot != MAV_AUTOPILOT_INVALID


def read_vehicle(path: str, sysid: int | None = None) -> Iterator[tuple[float, object]]:
    """Yield (t, msg) for messages from the vehicle, on a reboot-aware clock.

    The vehicle is identified from its own HEARTBEAT, not assumed: autotest logs also carry a
    test harness (sysid 250) whose chatter would otherwise be mixed into the vehicle's data.
    Messages seen before the vehicle's first heartbeat are held and replayed once it is known.
    STATUSTEXT is yielded already reassembled, as Text.
    """
    conn = mavutil.mavlink_connection(path, robust_parsing=True)
    clock, texts = Clock(), TextAssembler()
    pending: list = []
    while True:
        msg = conn.recv_match()
        if msg is None:
            break
        mtype = msg.get_type()
        if mtype == "BAD_DATA":
            yield clock.now, msg
            continue
        if sysid is None:
            if mtype == "HEARTBEAT" and _is_vehicle_heartbeat(msg):
                sysid = msg.get_srcSystem()
                backlog, pending = pending, []
                for old in backlog:
                    yield from _emit(old, sysid, clock, texts)
            else:
                pending.append(msg)
                continue
        yield from _emit(msg, sysid, clock, texts)
    for text in texts.flush():
        yield text.t, text
    yield clock.now, _Reboots(clock.reboots)


def _emit(msg, sysid, clock, texts):
    if msg.get_srcSystem() != sysid:
        return
    boot_ms = getattr(msg, "time_boot_ms", None)
    if boot_ms is not None:
        clock.update(boot_ms)
    if msg.get_type() == "STATUSTEXT":
        for text in texts.feed(clock.now, msg):
            yield text.t, text
    else:
        yield clock.now, msg


@dataclass
class _Reboots:
    count: int

    def get_type(self) -> str:
        return "_REBOOTS"
