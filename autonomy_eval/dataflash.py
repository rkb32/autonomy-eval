"""ArduPilot DataFlash logs (.bin), written by the autopilot itself, usually to its SD card.

Differences from telemetry that matter here:
  - Text (MSG) has no severity. Errors and failsafes are logged separately as ERR records
    (subsystem + code, code 0 meaning "cleared"), which are translated to ERROR-level text.
  - Text the autopilot *received* from other systems (a ground station, a test harness) is logged
    too, as "SRC=sysid/compid:text". Those are dropped: in a CI log they are most of the text, and
    none of it is the vehicle's own behavior. Telemetry is filtered the same way, by source system.
  - There is no heartbeat, so the vehicle type comes from the FRAME_CLASS parameter, whose meaning
    depends on the firmware, which comes from the version banner.
  - Signals are logged per sensor instance (IMU 0, 1, 2; EKF core 0, 1). Telemetry reports the
    primary instance, so instance 0 is used for signal values; clipping is summed over IMUs, as
    telemetry's VIBRATION message does.
"""
from __future__ import annotations

import re
from typing import Iterator

from pymavlink import DFReader

from .telemetry import Armed, Clock, Mode, Reboot, Record, Sample, Text, TextAssembler, VehicleType

ERROR = 3  # MAV_SEVERITY_ERROR
BANNER_RE = re.compile(r"^(ArduCopter|ArduPlane|ArduRover|ArduSub|Blimp|AntennaTracker) V")
RELAYED_RE = re.compile(r"^SRC=\d+/\d+:")

# FRAME_CLASS -> the MAV_TYPE the same firmware puts in its heartbeat. Frames not listed come out
# as unknown rather than guessed.
FRAME_TYPES = {
    "ArduRover": {1: 10, 2: 11, 3: 10},                            # rover, boat, balancebot
    "ArduCopter": {1: 2, 2: 13, 3: 14, 4: 14, 5: 13, 6: 4, 7: 15, 11: 4, 13: 4},  # quad hexa octa ... heli tri
}
DEFAULT_TYPES = {"ArduPlane": 1, "ArduSub": 12, "Blimp": 7}

# AP_Logger error subsystems (LogErrorSubsystem).
ERR_SUBSYSTEMS = {
    1: "main", 2: "radio", 3: "compass", 4: "optical flow", 5: "radio failsafe", 6: "battery failsafe",
    7: "GPS failsafe", 8: "GCS failsafe", 9: "fence failsafe", 10: "flight mode", 11: "GPS",
    12: "crash check", 13: "flip", 14: "autotune", 15: "parachute", 16: "EKF check",
    17: "EKF failsafe", 18: "barometer", 19: "CPU load watchdog", 20: "ADSB failsafe", 21: "terrain",
    22: "navigation", 23: "terrain failsafe", 24: "EKF primary change", 25: "thrust loss",
    26: "sensor failsafe", 27: "leak failsafe", 28: "pilot input", 29: "vibration failsafe",
    30: "internal error", 31: "dead-reckoning failsafe",
}

WANTED = ["MSG", "ERR", "ARM", "EV", "MODE", "PARM", "NTUN", "XKF4", "VIBE", "GPS", "BAT"]
EV_ARMED, EV_DISARMED = 10, 11


def read_dataflash(path: str) -> Iterator[Record]:
    log = DFReader.DFReader_binary(path, zero_time_base=True)
    clock, texts = Clock(), TextAssembler()
    firmware = frame_class = None
    have_arm_msg = False
    clip_by_imu: dict[int, int] = {}
    while True:
        m = log.recv_match(type=WANTED)
        if m is None:
            break
        if clock.update(m.TimeUS / 1e6):
            yield Reboot(clock.now)
        t, kind = clock.now, m.get_type()

        if kind == "MSG":
            banner = BANNER_RE.match(m.Message)
            if banner and firmware is None:
                firmware = banner.group(1)
                yield from _vehicle_type(t, firmware, frame_class)
            for text in texts.feed(t, getattr(m, "ID", 0), None, m.Message):
                if not RELAYED_RE.match(text.text):
                    yield text
        elif kind == "PARM" and m.Name == "FRAME_CLASS" and frame_class is None:
            frame_class = int(m.Value)
            if firmware:
                yield from _vehicle_type(t, firmware, frame_class)
        elif kind == "ERR" and m.ECode != 0:
            name = ERR_SUBSYSTEMS.get(m.Subsys, f"subsystem {m.Subsys}")
            yield Text(t, ERROR, f"ERR: {name} code {m.ECode}")
        elif kind == "ARM":
            have_arm_msg = True
            yield Armed(t, bool(m.ArmState))
        elif kind == "EV" and not have_arm_msg and m.Id in (EV_ARMED, EV_DISARMED):
            yield Armed(t, m.Id == EV_ARMED)          # firmware older than the ARM message
        elif kind == "MODE":
            yield Mode(t, m.ModeNum)
        elif kind == "NTUN" and hasattr(m, "XTrack"):
            yield Sample(t, "xtrack", abs(m.XTrack))
        elif kind == "XKF4" and m.C == 0:
            yield Sample(t, "ekf_vel", m.SV)
            yield Sample(t, "ekf_ph", m.SP)
            yield Sample(t, "ekf_pv", m.SH)
            yield Sample(t, "ekf_mag", m.SM)
        elif kind == "VIBE":
            imu = getattr(m, "IMU", 0)
            if imu == 0:
                yield Sample(t, "vibe", max(m.VibeX, m.VibeY, m.VibeZ))
            if hasattr(m, "Clip"):
                clip_by_imu[imu] = m.Clip
            else:                                     # older format: one record, three counters
                clip_by_imu = {0: m.Clip0, 1: m.Clip1, 2: m.Clip2}
            yield Sample(t, "clip", sum(clip_by_imu.values()))
        elif kind == "GPS" and getattr(m, "I", 0) == 0:
            yield Sample(t, "sats", m.NSats)
            yield Sample(t, "hdop", m.HDop)
        elif kind == "BAT" and getattr(m, "Inst", 0) == 0 and m.Volt > 0:
            yield Sample(t, "batt", m.Volt)
    yield from (x for x in texts.flush() if not RELAYED_RE.match(x.text))


def _vehicle_type(t: float, firmware: str, frame_class: int | None) -> Iterator[VehicleType]:
    mav_type = FRAME_TYPES.get(firmware, {}).get(frame_class) if frame_class is not None else None
    mav_type = mav_type or DEFAULT_TYPES.get(firmware)
    if mav_type is not None:
        yield VehicleType(t, mav_type)
