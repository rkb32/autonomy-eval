"""Turn one log into a FlightSummary: who flew, on what firmware, and a flat set of comparable numbers.

Only what happens in flight is counted. Pre-flight messages (boot banners, calibration, pre-arm
checks) repeat a varying number of times depending on when recording started and on how fast the
operator or test harness tried to arm, so their counts are noise; only their presence is kept.

Continuous signals use the 95th percentile over the flight rather than the maximum, because a
single-sample peak is the noisiest statistic a signal has.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .telemetry import Armed, Corrupt, Mode, Reboot, Record, Sample, Text, VehicleType, is_dataflash, read

INFO, DEBUG = 6, 7  # MAV_SEVERITY levels; text with no severity is treated as INFO
GRACE_S = 1.0     # heartbeats are 1 Hz, so the armed flag can lag the arm message by up to a second
# ArduPilot names its pre-arm check failures; they are pre-flight by definition, whatever their timestamp.
PREFLIGHT_PREFIXES = ("PreArm:", "Arm:")
VERSION_RE = re.compile(r"^(Ardu\w+|Rover|Copter|Plane|Sub|Blimp) V(\S+)(?: \((\w+)\))?")

WORSE_IF_HIGHER = {
    "armed_time_s", "xtrack_max_m", "xtrack_mean_m", "xtrack_p95_m", "ekf_vel_var_p95",
    "ekf_pos_horiz_var_p95", "ekf_pos_vert_var_p95", "ekf_compass_var_p95", "vibe_p95",
    "clip_total", "gps_hdop_p95", "errors", "warnings", "reboots",
}
WORSE_IF_LOWER = {"missions_completed", "waypoints_reached", "gps_sats_min", "batt_min_v"}


@dataclass
class FlightSummary:
    source: str
    vehicle: str | None = None
    firmware: str | None = None
    mav_type: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    events: dict[str, int] = field(default_factory=dict)       # in-flight messages, counted
    preflight: list[str] = field(default_factory=list)         # pre-flight messages, presence only
    severity: dict[str, int] = field(default_factory=dict)     # most severe level seen per message kind
    quality: dict[str, float] = field(default_factory=dict)    # about the recording, not the vehicle

    @property
    def label(self) -> str:
        return self.firmware or Path(self.source).stem

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "FlightSummary":
        return cls(**json.loads(text))


def normalize_event(text: str) -> str:
    """'Reached waypoint #4' and 'Reached waypoint #7' are the same kind of event; 'EKF3' stays 'EKF3'."""
    text = re.sub(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{6,}\b", "<hex>", text)
    return re.sub(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", "N", text).strip()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def in_intervals(t: float, intervals: list[tuple[float, float]], grace: float = GRACE_S) -> bool:
    return any(start - grace <= t <= end + grace for start, end in intervals)


def summarize(path: str | Path) -> FlightSummary:
    s = summarize_records(read(path), str(path))
    s.quality["format"] = "dataflash" if is_dataflash(path) else "tlog"
    return s


def summarize_records(records: Iterable[Record], source: str) -> FlightSummary:
    s = FlightSummary(source=source)
    texts: list[Text] = []
    flights: list[tuple[float, float]] = []
    armed, armed_since, last_mode, mode_changes = False, 0.0, None, 0
    series: dict[str, list[float]] = {k: [] for k in ("xtrack", "ekf_vel", "ekf_ph", "ekf_pv", "ekf_mag", "vibe", "hdop")}
    sats: list[float] = []
    batt: list[float] = []
    clip = corrupt = reboots = 0
    end = 0.0

    for rec in records:
        t = rec.t
        end = max(end, t)
        if isinstance(rec, Text):
            m = VERSION_RE.match(rec.text)
            if m:
                s.vehicle, s.firmware = m.group(1), m.group(3) or s.firmware
            else:
                texts.append(rec)
        elif isinstance(rec, Corrupt):
            corrupt += 1
        elif isinstance(rec, Reboot):
            reboots += 1
            if armed:                          # a reboot ends the flight, whatever the last heartbeat said
                flights.append((armed_since, t))
                armed = False
        elif isinstance(rec, VehicleType):
            s.mav_type = rec.mav_type
        elif isinstance(rec, Armed):
            if rec.armed and not armed:
                armed_since = t
            elif armed and not rec.armed:
                flights.append((armed_since, t))
            armed = rec.armed
        elif isinstance(rec, Mode):
            if armed and last_mode is not None and rec.mode != last_mode:
                mode_changes += 1
            last_mode = rec.mode
        elif isinstance(rec, Sample) and armed:
            if rec.name == "clip":
                clip = max(clip, rec.value)
            elif rec.name == "sats":
                sats.append(rec.value)
            elif rec.name == "batt":
                batt.append(rec.value)
            else:
                series[rec.name].append(rec.value)
    if armed:
        flights.append((armed_since, end))

    events: Counter[str] = Counter()
    preflight: set[str] = set()
    errors = warnings = missions = waypoints = 0
    severity: dict[str, int] = {}
    for x in texts:
        key = normalize_event(x.text)
        level = INFO if x.severity is None else x.severity
        severity[key] = min(severity.get(key, DEBUG), level)
        if x.text.startswith(PREFLIGHT_PREFIXES) or not in_intervals(x.t, flights):
            preflight.add(key)
            continue
        events[key] += 1
        errors += level <= 3
        warnings += level == 4
        missions += x.text.startswith("Mission Complete")
        waypoints += x.text.startswith("Reached waypoint")

    xt = series["xtrack"]
    raw = {
        "armed_time_s": sum(b - a for a, b in flights),
        "flights": len(flights),
        "reboots": reboots,
        "mode_changes": mode_changes,
        "missions_completed": missions,
        "waypoints_reached": waypoints,
        "errors": errors,
        "warnings": warnings,
        "xtrack_mean_m": sum(xt) / len(xt) if xt else None,
        "xtrack_p95_m": percentile(xt, 0.95),
        "xtrack_max_m": max(xt) if xt else None,
        "ekf_vel_var_p95": percentile(series["ekf_vel"], 0.95),
        "ekf_pos_horiz_var_p95": percentile(series["ekf_ph"], 0.95),
        "ekf_pos_vert_var_p95": percentile(series["ekf_pv"], 0.95),
        "ekf_compass_var_p95": percentile(series["ekf_mag"], 0.95),
        "vibe_p95": percentile(series["vibe"], 0.95),
        "clip_total": clip,
        "gps_sats_min": min(sats) if sats else None,
        "gps_hdop_p95": percentile(series["hdop"], 0.95),
        "batt_min_v": min(batt) if batt else None,
    }
    s.metrics = {k: round(float(v), 4) for k, v in raw.items() if v is not None}
    s.events = dict(sorted(events.items()))
    s.preflight = sorted(preflight)
    s.severity = dict(sorted(severity.items()))
    s.quality = {"log_duration_s": round(end, 1), "corrupt_frames": corrupt}
    return s


def load(path: str | Path) -> FlightSummary:
    """A raw .tlog or .bin log, or a summary saved earlier with `autonomy-eval summarize -o` (so CI need not re-parse history)."""
    p = Path(path)
    if p.suffix == ".json":
        return FlightSummary.from_json(p.read_text())
    return summarize(p)
