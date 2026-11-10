"""Fly one SystemID chirp on the roll rate loop in a running Copter SITL (tcp:127.0.0.1:5760).

ArduCopter only enters SYSTEMID for an attitude axis (SID_AXIS 7 = Rate Roll) while the flight mode has
manual throttle (ModeSystemId::init, ArduCopter/mode_systemid.cpp). So: take off in GUIDED, switch to
STABILIZE, hold altitude with RC throttle overrides (a simple P loop on altitude), switch to SYSTEMID
(mode 25) and wait for the chirp, then release the overrides and land. The DataFlash log carries the SIDD
records (target and measured roll rate) that margins.py reads.
"""
import argparse
import os
import sys
import time

os.environ["MAVLINK20"] = "1"

from pymavlink import mavutil

ALT = 20
GUIDED, STABILIZE, LAND, SYSTEMID_MODE = 4, 0, 9, 25
ROLL_RATE_AXIS = 7
CHIRP_WAIT_S = 120            # fade-in 15 + record 70 + fade-out 2, with margin
THROTTLE_MID = 1550           # starting throttle PWM for hover; the altitude P loop corrects it


def wait_msg(m, kinds, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(type=kinds, blocking=True, timeout=1)
        if msg is not None:
            return msg
    raise TimeoutError(f"no {kinds} within {timeout}s")


def heartbeat(m):
    m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)


def set_param(m, name, value):
    m.mav.param_set_send(m.target_system, m.target_component, name.encode(), float(value),
                         mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    t0 = time.time()
    while time.time() - t0 < 5:
        msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
        if msg is not None and msg.param_id == name and abs(msg.param_value - float(value)) < 1e-4:
            return
    raise RuntimeError(f"could not set {name}={value}")


def set_mode(m, mode):
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode, 0, 0, 0, 0, 0)


def throttle_for(alt_m):
    """P loop on altitude: more throttle when below ALT. Clamped to a safe PWM band."""
    return max(1300, min(1800, int(THROTTLE_MID + 20 * (ALT - alt_m))))


def override(m, throttle):
    """Roll, pitch and yaw centred; throttle from the altitude loop."""
    m.mav.rc_channels_override_send(m.target_system, m.target_component,
                                    1500, 1500, throttle, 1500, 0, 0, 0, 0)


def release(m):
    m.mav.rc_channels_override_send(m.target_system, m.target_component, 0, 0, 0, 0, 0, 0, 0, 0)


def hold_for(m, seconds, alt_box):
    t_end = time.time() + seconds
    while time.time() < t_end:
        heartbeat(m)
        msg = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=0.2)
        if msg is not None:
            alt_box[0] = msg.relative_alt / 1000.0
        override(m, throttle_for(alt_box[0]))


def run(flight_mode: int = STABILIZE) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="tcp:127.0.0.1:5760")
    ap.add_argument("--param", action="append", default=[], help="NAME=VALUE set before arming")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    m = mavutil.mavlink_connection(args.conn, source_system=255, source_component=190)
    wait_msg(m, "HEARTBEAT", 60)
    heartbeat(m)
    print("connected, sysid", m.target_system, flush=True)
    m.mav.request_data_stream_send(m.target_system, m.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)

    set_param(m, "SID_AXIS", ROLL_RATE_AXIS)
    for p in args.param:
        name, value = p.split("=")
        set_param(m, name, value)
        print("set", name, value, flush=True)

    t0 = time.time()
    while True:
        heartbeat(m)
        if time.time() - t0 > 180:
            raise TimeoutError("could not arm within 180s")
        set_mode(m, GUIDED)
        time.sleep(1)
        m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                0, 1, 0, 0, 0, 0, 0, 0)
        m.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        if hb is not None and hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            break
    print("armed", flush=True)
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                            0, 0, 0, 0, 0, 0, 0, ALT)

    alt = [0.0]
    t0 = time.time()
    while alt[0] < ALT - 2:                        # climb in GUIDED first
        heartbeat(m)
        msg = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
        if msg is not None:
            alt[0] = msg.relative_alt / 1000.0
        if time.time() - t0 > 90:
            raise TimeoutError(f"did not reach {ALT} m in 90 s (alt {alt[0]:.1f})")
    print("reached altitude", round(alt[0], 1), flush=True)

    set_mode(m, flight_mode)
    hold_for(m, 15, alt)                           # settle at altitude in STABILIZE
    print("STABILIZE altitude", round(alt[0], 1), flush=True)

    set_mode(m, SYSTEMID_MODE)
    print("SYSTEMID started", flush=True)
    hold_for(m, CHIRP_WAIT_S, alt)

    release(m)
    set_mode(m, LAND)
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        heartbeat(m)
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if hb is not None and not (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print("disarmed, flight done", flush=True)
            return 0
    print("timed out", flush=True)
    return 1


def main():
    return run(STABILIZE)


if __name__ == "__main__":
    sys.exit(main())
