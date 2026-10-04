"""Fly one fixed Copter mission in a running SITL (tcp:127.0.0.1:5760) and exit when it is disarmed.

Mission: take off to 20 m, fly a ~150 m square at 20 m, return and land. Optional --param NAME=VALUE
are set before arming (used to inject faults).
"""
import argparse
import math
import os
import sys
import time

os.environ["MAVLINK20"] = "1"          # the autopilot only replies to mission uploads over MAVLink 2

from pymavlink import mavutil

HOME = (-35.363261, 149.165230)
ALT = 20


def offset(lat, lon, north_m, east_m):
    return (lat + north_m / 111320.0, lon + east_m / (111320.0 * math.cos(math.radians(lat))))


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


def upload_mission(m, points):
    items = []
    # item 0 is home; then takeoff, waypoints, RTL
    items.append((mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, HOME[0], HOME[1], 0, mavutil.mavlink.MAV_FRAME_GLOBAL))
    items.append((mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, ALT, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT))
    for lat, lon in points:
        items.append((mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, lat, lon, ALT, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT))
    items.append((mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT))
    m.mav.mission_count_send(m.target_system, m.target_component, len(items), mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
    sent = set()
    while len(sent) < len(items):
        req = wait_msg(m, ["MISSION_REQUEST_INT", "MISSION_REQUEST"], 10)
        i = req.seq
        cmd, lat, lon, alt, frame = items[i]
        m.mav.mission_item_int_send(m.target_system, m.target_component, i, frame, cmd, 0, 1, 0, 0, 0, 0,
                                    int(lat * 1e7), int(lon * 1e7), alt, mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
        sent.add(i)
    ack = wait_msg(m, "MISSION_ACK", 10)
    if ack.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
        raise RuntimeError(f"mission rejected: {ack.type}")
    return len(items)


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
    m.mav.request_data_stream_send(m.target_system, m.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)

    for p in args.param:
        name, value = p.split("=")
        set_param(m, name, value)
        print("set", name, value, flush=True)

    sq = 150
    points = [offset(*HOME, sq, 0), offset(*HOME, sq, sq), offset(*HOME, 0, sq), offset(*HOME, 0, 0)]
    n = upload_mission(m, points)
    print("mission uploaded,", n, "items", flush=True)

    # wait until the EKF is happy, then arm in GUIDED and start AUTO
    t0 = time.time()
    while True:
        heartbeat(m)
        if time.time() - t0 > 180:
            raise TimeoutError("could not arm within 180s")
        m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, 0, 0, 0, 0, 0)   # GUIDED
        time.sleep(1)
        m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                                1, 0, 0, 0, 0, 0, 0)
        ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        if hb is not None and hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            break
    print("armed after", round(time.time() - t0), "s", flush=True)
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 3, 0, 0, 0, 0, 0)       # AUTO
    # In AUTO a takeoff item needs the throttle up; SITL accepts a mission start command.
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_MISSION_START, 0, 0, 0, 0, 0, 0, 0, 0)

    t0 = time.time()
    last_seq = -1
    while time.time() - t0 < args.timeout:
        heartbeat(m)
        msg = m.recv_match(type=["MISSION_ITEM_REACHED", "HEARTBEAT"], blocking=True, timeout=2)
        if msg is None:
            continue
        if msg.get_type() == "MISSION_ITEM_REACHED" and msg.seq != last_seq:
            last_seq = msg.seq
            print("reached item", msg.seq, flush=True)
        if msg.get_type() == "HEARTBEAT" and last_seq >= 1 and not (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print("disarmed, flight done", flush=True)
            return 0
    print("timed out", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
