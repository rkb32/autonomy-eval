"""Arm-only diagnostic for a running Copter SITL (tcp:127.0.0.1:5760). Flies nothing.

Sets the given parameters, then tries to arm in GUIDED for up to --seconds, printing every STATUSTEXT
message the vehicle sends (pre-arm failures say why arming is refused). Exit 0 if it armed, 1 if not.
Used to see why a fresh SITL won't arm when a fault is injected; see scripts/sitl_runs.py.
"""
import argparse
import os
import sys
import time

os.environ["MAVLINK20"] = "1"

from pymavlink import mavutil

from sitl_sysid import GUIDED, heartbeat, set_mode, set_param, wait_msg


def text_of(msg) -> str:
    return msg.text if isinstance(msg.text, str) else msg.text.decode(errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="tcp:127.0.0.1:5760")
    ap.add_argument("--param", action="append", default=[], help="NAME=VALUE set before arming")
    ap.add_argument("--seconds", type=int, default=180)
    args = ap.parse_args()

    m = mavutil.mavlink_connection(args.conn, source_system=255, source_component=190)
    wait_msg(m, "HEARTBEAT", 60)
    heartbeat(m)
    m.mav.request_data_stream_send(m.target_system, m.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
    for p in args.param:
        name, value = p.split("=")
        set_param(m, name, value)
        print("set", name, value, flush=True)

    set_mode(m, GUIDED)
    t0 = time.time()
    last_try = 0.0
    while time.time() - t0 < args.seconds:
        heartbeat(m)
        if time.time() - last_try > 5:
            last_try = time.time()
            set_mode(m, GUIDED)
            m.mav.command_long_send(m.target_system, m.target_component,
                                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        msg = m.recv_match(type=["STATUSTEXT", "HEARTBEAT"], blocking=True, timeout=1)
        if msg is None:
            continue
        if msg.get_type() == "STATUSTEXT":
            print(f"[{time.time() - t0:6.1f}s] {text_of(msg)}", flush=True)
        elif msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            print(f"ARMED after {time.time() - t0:.1f}s", flush=True)
            return 0
    print(f"NOT ARMED after {args.seconds}s", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
