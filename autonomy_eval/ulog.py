"""Read a PX4 ULog (.ulg) as the same records telemetry.py defines, so metrics.py works on it unchanged.

Only signals with a clear ArduPilot counterpart are read: arming, flight mode, battery voltage,
GPS satellites and HDOP, vehicle type, and logged messages. Signals PX4 logs under other names or
with other meanings (cross-track error, attitude error, vibration, EKF variances) are not read,
so those metrics are absent for PX4 logs and compare() skips them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from pyulog import ULog

from .telemetry import Armed, Mode, Record, Sample, Text, VehicleType

ARMING_STATE_ARMED = 2   # vehicle_status.arming_state, PX4 ARMING_STATE_ARMED
GPS_FIX_3D = 3           # vehicle_gps_position.fix_type; below this there is no usable position (PX4 FIX_TYPE_3D)
TOPICS = [
    "vehicle_status",
    "battery_status",
    "vehicle_gps_position",
]


def identity(path: str | Path) -> tuple[str, str | None]:
    """(vehicle, firmware) from the ULog header. PX4 release numbers pack major.minor.patch into 3 bytes."""
    info = ULog(str(path), parse_header_only=True).msg_info_dict
    release = int(info.get("ver_sw_release", 0))
    if release:
        firmware = f"{release >> 24 & 0xff}.{release >> 16 & 0xff}.{release >> 8 & 0xff}"
    else:
        firmware = str(info.get("ver_sw", ""))[:8] or None
    return "PX4", firmware


def _level(raw: int) -> int:
    """ULog stores the log level as an ASCII digit ('0'..'7'); MAV_SEVERITY uses the same numbers."""
    return raw - 48 if raw >= 48 else raw


def read_ulog(path: str | Path) -> Iterator[Record]:
    log = ULog(str(path), message_name_filter_list=TOPICS)
    t0 = int(log.start_timestamp)

    def sec(us) -> float:
        # Timestamps are numpy uint64; subtracting them unsigned wraps around, so convert to int first.
        return (int(us) - t0) / 1e6

    records: list[Record] = []
    by_name = {d.name: d for d in log.data_list if d.multi_id == 0}

    status = by_name.get("vehicle_status")
    if status is not None:
        d = status.data
        ts = d["timestamp"]
        if len(ts):
            records.append(VehicleType(sec(ts[0]), int(d["system_type"][0])))
        armed = None
        mode = None
        for i, us in enumerate(ts):
            t = sec(us)
            is_armed = int(d["arming_state"][i]) == ARMING_STATE_ARMED
            if is_armed != armed:
                records.append(Armed(t, is_armed))
                armed = is_armed
            nav = int(d["nav_state"][i])
            if nav != mode:
                records.append(Mode(t, nav))
                mode = nav

    # Every instance counts: a second battery or GPS unit is read too, and the minimum is taken over all of them.
    for battery in (d for d in log.data_list if d.name == "battery_status"):
        d = battery.data
        for i, us in enumerate(d["timestamp"]):
            if d["connected"][i]:
                records.append(Sample(sec(us), "batt", float(d["voltage_v"][i])))

    for gps in (d for d in log.data_list if d.name == "vehicle_gps_position"):
        d = gps.data
        for i, us in enumerate(d["timestamp"]):
            if int(d["fix_type"][i]) < GPS_FIX_3D:
                continue       # before a fix, satellite count and HDOP are not meaningful
            t = sec(us)
            records.append(Sample(t, "sats", float(d["satellites_used"][i])))
            records.append(Sample(t, "hdop", float(d["hdop"][i])))

    for m in log.logged_messages:
        records.append(Text(sec(m.timestamp), _level(m.log_level), m.message))

    records.sort(key=lambda r: r.t)
    yield from records
