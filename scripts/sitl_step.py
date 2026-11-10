"""Fly GUIDED position-step tests in a running Copter SITL (tcp:127.0.0.1:5760) and exit when done.

Takes off to ALT metres in GUIDED, then commands north/east position steps (SET_POSITION_TARGET_LOCAL_NED)
and holds each step long enough to settle. The DataFlash log of the flight is what the step-response
metrics are computed from. Optional --param NAME=VALUE is set before arming, as in sitl_fly.py.

UNTESTED: written without a running SITL (Docker was not running when this was written).
"""
import argparse
import os
import sys
import time

os.environ["MAVLINK20"] = "1"

from pymavlink import mavutil

ALT = 20
# (north_m, east_m, hold_s): step to the target, hold until settled. The last entry returns to origin.
STEPS = [(10, 0, 25), (0, 0, 25), (0, 10, 25), (0, 0, 25)]
# type_mask: ignore velocity, acceleration, yaw, yaw rate; use position only
POS_ONLY_MASK = 0b0000111111111000


def heartbeat(m):
    m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)


def wait_msg(m, kinds, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(type=kinds, blocking=True, timeout=1)
        if msg is not None:
            return msg
    raise TimeoutError(f"no {kinds} within {timeout}s")


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


def goto_local_ned(m, north, east, down):
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        POS_ONLY_MASK, north, east, down, 0, 0, 0, 0, 0, 0, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="tcp:127.0.0.1:5760")
    ap.add_argument("--param", action="append", default=[], help="NAME=VALUE set before arming")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    m = mavutil.mavlink_connection(args.conn, source_system=255, source_component=190)
    wait_msg(m, "HEARTBEAT", 60)
    heartbeat(m)
    print("connected, sysid", m.target_system, flush=True)

    for p in args.param:
        name, value = p.split("=")
        set_param(m, name, value)
        print("set", name, value, flush=True)

    # GUIDED, arm, take off to ALT
    t0 = time.time()
    while True:
        heartbeat(m)
        if time.time() - t0 > 180:
            raise TimeoutError("could not arm within 180s")
        set_mode(m, 4)  # GUIDED
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
    time.sleep(30)  # let takeoff complete before stepping

    # position steps; NED down is negative for altitude
    for north, east, hold in STEPS:
        print(f"step to N={north} E={east}", flush=True)
        t_end = time.time() + hold
        while time.time() < t_end:
            heartbeat(m)
            goto_local_ned(m, north, east, -ALT)
            time.sleep(0.5)

    # land and disarm
    set_mode(m, 9)  # LAND
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        heartbeat(m)
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if hb is not None and not (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print("disarmed, flight done", flush=True)
            return 0
    print("timed out", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
