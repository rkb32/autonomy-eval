"""Plain-language rendering of a Comparison, meant to be read by a person, and a JSON form for CI."""
from __future__ import annotations

from dataclasses import asdict

from .compare import Comparison, Finding

SEVERITY = ["EMERGENCY", "ALERT", "CRITICAL", "ERROR", "WARNING", "NOTICE", "INFO", "DEBUG"]

LABELS = {
    "armed_time_s": ("time armed (mission time)", "s"),
    "flights": ("separate armed flights", ""),
    "reboots": ("reboots", ""),
    "mode_changes": ("in-flight mode changes", ""),
    "missions_completed": ("missions completed", ""),
    "waypoints_reached": ("waypoints reached", ""),
    "errors": ("in-flight error messages", ""),
    "warnings": ("in-flight warning messages", ""),
    "xtrack_mean_m": ("mean cross-track error", "m"),
    "xtrack_p95_m": ("95th-percentile cross-track error", "m"),
    "xtrack_max_m": ("worst cross-track error", "m"),
    "ekf_vel_var_p95": ("EKF velocity variance (p95)", ""),
    "ekf_pos_horiz_var_p95": ("EKF horizontal position variance (p95)", ""),
    "ekf_pos_vert_var_p95": ("EKF vertical position variance (p95)", ""),
    "ekf_compass_var_p95": ("EKF compass variance (p95)", ""),
    "vibe_p95": ("vibration (p95)", "m/s/s"),
    "clip_total": ("accelerometer clipping events", ""),
    "gps_sats_min": ("fewest GPS satellites", ""),
    "gps_hdop_p95": ("GPS HDOP, dilution of precision (p95)", ""),
    "batt_min_v": ("lowest battery voltage", "V"),
}


def _num(x: float) -> str:
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.3g}" if abs(x) < 10 else f"{x:.1f}"


def _line(f: Finding) -> str:
    name, unit = LABELS.get(f.metric, (f.metric, ""))
    unit = f" {unit}" if unit else ""
    pct = f"  ({f.pct:+.1f}%)" if f.pct is not None else ""
    return f"  {name}: {_num(f.value)}{unit}, baseline {_num(f.median)}{unit}{pct}"


def render(c: Comparison) -> str:
    cand = c.candidate
    firmwares = ", ".join(b.label for b in c.baseline)
    out = [
        f"{cand.label}  ({cand.vehicle}, {cand.source})",
        f"compared against {len(c.baseline)} earlier runs: {firmwares}",
    ]
    out += [f"  note: {w}" for w in c.warnings]
    if cand.quality.get("corrupt_frames"):
        out.append(f"  note: {cand.quality['corrupt_frames']} corrupt telemetry frames skipped while reading")
    if not c.flag_count:
        out.append("\nno regressions: every metric is within the normal range of the baseline")
        return "\n".join(out)

    sections = [
        ("REGRESSIONS (worse than every-day variation):", [f for f in c.findings if f.verdict == "worse"]),
        ("IMPROVED:", [f for f in c.findings if f.verdict == "better"]),
        ("OTHER CHANGES:", [f for f in c.findings if f.verdict == "changed"]),
    ]
    for title, items in sections:
        if items:
            out.append("\n" + title)
            out += [_line(f) for f in items]
    alarms = c.new_alarms
    if alarms:
        out.append("\nNEW WARNINGS / ERRORS (never seen in any baseline run):")
        out += [f"  [{SEVERITY[m.severity]}] {m.text}" for m in alarms]
    info = [m for m in c.new_messages if not m.alarming]
    if info:
        out.append("\nNEW MESSAGES (informational, never seen in any baseline run):")
        out += [f"  {m.text}" for m in info]
    if c.missing_messages:
        out.append("\nMISSING MESSAGES (seen in every baseline run, absent here):")
        out += [f"  {e}" for e in c.missing_messages]
    return "\n".join(out)


def render_scan(comparisons: list[Comparison]) -> str:
    out = ["leave-one-out scan: each run compared against all the others\n"]
    width = max(len(c.candidate.label) for c in comparisons)
    for c in comparisons:
        status = "REGRESSED" if c.failed else ("changed" if c.flag_count else "ok")
        if c.findings:
            top = _line(c.findings[0]).strip()
        elif c.new_messages:
            top = f"new message: {c.new_messages[0].text}"
        elif c.missing_messages:
            top = f"missing message: {c.missing_messages[0]}"
        else:
            top = ""
        out.append(f"  {c.candidate.label:<{width}}  {status:<9}  {c.flag_count:>2} flag(s)  {top}")
    return "\n".join(out)


def to_dict(c: Comparison) -> dict:
    return {
        "candidate": c.candidate.label,
        "source": c.candidate.source,
        "baseline": [b.label for b in c.baseline],
        "failed": c.failed,
        "findings": [asdict(f) | {"pct": f.pct} for f in c.findings],
        "new_messages": [asdict(m) | {"level": SEVERITY[m.severity]} for m in c.new_messages],
        "missing_messages": c.missing_messages,
        "warnings": c.warnings,
    }
