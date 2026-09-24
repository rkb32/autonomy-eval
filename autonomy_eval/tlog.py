"""MAVLink telemetry logs (.tlog), as saved by a ground station or a test harness."""
from __future__ import annotations

from typing import Iterator

from pymavlink import mavutil

from .telemetry import Armed, Clock, Corrupt, Mode, Reboot, Record, Sample, TextAssembler, VehicleType

MAV_TYPE_GCS = 6
MAV_AUTOPILOT_INVALID = 8
ARMED_FLAG = 128  # MAV_MODE_FLAG_SAFETY_ARMED


def _is_vehicle_heartbeat(msg) -> bool:
    return msg.type != MAV_TYPE_GCS and msg.autopilot != MAV_AUTOPILOT_INVALID


def read_tlog(path: str, sysid: int | None = None) -> Iterator[Record]:
    """Records from the vehicle only, on a reboot-aware clock.

    The vehicle is identified from its own HEARTBEAT, not assumed: test harness logs also carry a
    harness (sysid 250) whose chatter would otherwise be mixed into the vehicle's data. Messages
    seen before the vehicle's first heartbeat are held and replayed once it is known.
    """
    conn = mavutil.mavlink_connection(path, robust_parsing=True)
    clock, texts = Clock(), TextAssembler()
    pending: list = []
    compid = None
    while True:
        msg = conn.recv_match()
        if msg is None:
            break
        if msg.get_type() == "BAD_DATA":
            yield Corrupt(clock.now)
            continue
        if sysid is None:
            if msg.get_type() == "HEARTBEAT" and _is_vehicle_heartbeat(msg):
                sysid, compid = msg.get_srcSystem(), msg.get_srcComponent()
                backlog, pending = pending, []
                for old in backlog:
                    yield from _translate(old, sysid, compid, clock, texts)
            else:
                pending.append(msg)
                continue
        yield from _translate(msg, sysid, compid, clock, texts)
    yield from texts.flush()


def _translate(msg, sysid: int, compid: int | None, clock: Clock, texts: TextAssembler) -> Iterator[Record]:
    if msg.get_srcSystem() != sysid:
        return
    boot_ms = getattr(msg, "time_boot_ms", None)
    if boot_ms is not None and clock.update(boot_ms / 1000):
        yield Reboot(clock.now)
    t, kind = clock.now, msg.get_type()
    if kind == "STATUSTEXT":
        yield from texts.feed(t, getattr(msg, "id", 0), msg.severity, msg.text)
    elif kind == "HEARTBEAT":
        if compid is not None and msg.get_srcComponent() != compid:
            return                   # a companion computer or camera on the same system is not the autopilot
        yield VehicleType(t, msg.type)
        yield Armed(t, bool(msg.base_mode & ARMED_FLAG))
        yield Mode(t, msg.custom_mode)
    elif kind == "NAV_CONTROLLER_OUTPUT":
        yield Sample(t, "xtrack", abs(msg.xtrack_error))
    elif kind == "EKF_STATUS_REPORT":
        yield Sample(t, "ekf_vel", msg.velocity_variance)
        yield Sample(t, "ekf_ph", msg.pos_horiz_variance)
        yield Sample(t, "ekf_pv", msg.pos_vert_variance)
        yield Sample(t, "ekf_mag", msg.compass_variance)
    elif kind == "VIBRATION":
        yield Sample(t, "vibe", max(msg.vibration_x, msg.vibration_y, msg.vibration_z))
        yield Sample(t, "clip", msg.clipping_0 + msg.clipping_1 + msg.clipping_2)
    elif kind == "GPS_RAW_INT":
        yield Sample(t, "sats", msg.satellites_visible)
        if msg.eph != 65535:
            yield Sample(t, "hdop", msg.eph / 100)   # eph is HDOP x 100, not a distance
    elif kind == "SYS_STATUS" and msg.voltage_battery not in (0, 65535):
        yield Sample(t, "batt", msg.voltage_battery / 1000)
