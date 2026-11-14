"""Web demo: the CI result, a way to try your own logs, and the stability and estimator checks.

    streamlit run app.py
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from autonomy_eval import bisect as bisect_mod
from autonomy_eval.checks import CHECKS, CheckOutcome, run_check, run_sample
from autonomy_eval.compare import MIN_BASELINE, Comparison, Incomparable, scan, vehicle_kind
from autonomy_eval.metrics import FlightSummary, load
from autonomy_eval.report import LABELS, SEVERITY
from autonomy_eval import batch, flightdata
from autonomy_eval.health import timing_health
from autonomy_eval.signals import arm_time, list_signals, read_signal

EXAMPLES = Path(__file__).parent / "examples"
REPO = "https://github.com/rkb32/autonomy-eval"
HISTORIES = {
    "Sailboat - 8 daily builds": "ardurover-surface-boat",
    "Ground rover - 8 daily builds": "ardurover-ground-rover",
}
STATUS_COLORS = {"REGRESSED": "#E5484D", "changed": "#D9A441", "ok": "#8B929C"}
SAMPLES = [
    ("sysid", "Run the stability sample (about 300 MB; the first run is slow)"),
    ("nees", "Run the estimator sample"),
    ("step", "Run the step response sample"),
]

st.set_page_config(page_title="Autonomy-eval", layout="centered")

# Small visual layer: monospace numbers in tables, a quiet loading placeholder, and a keyboard hint.
STYLE = """
<style>
[data-testid="stTable"] table { font-size: 0.9rem; font-variant-numeric: tabular-nums; }
[data-testid="stTable"] td, [data-testid="stTable"] th { white-space: nowrap; }
[data-testid="stTable"] th { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.03em;
  text-transform: uppercase; opacity: 0.6; }
[data-testid="stTable"] td { text-align: right; }
[data-testid="stTable"] td:first-child, [data-testid="stTable"] th:first-child { text-align: left; }
button, [role="tab"] { transition: background-color .12s ease-out, border-color .12s ease-out, color .12s ease-out; }
.ae-skel { padding: 0.25rem 0 0.5rem; }
.ae-skel p { margin: 0 0 0.6rem; font-size: 0.9rem; opacity: 0.7; }
.ae-skel span { display: block; height: 0.9rem; margin: 0.55rem 0; border-radius: 4px;
  background: color-mix(in srgb, currentColor 10%, transparent); animation: ae-pulse 1.2s ease-in-out infinite; }
.ae-skel span:nth-of-type(2) { width: 80%; }
.ae-skel span:nth-of-type(3) { width: 55%; }
@keyframes ae-pulse { 50% { opacity: .45; } }
@media (prefers-reduced-motion: reduce) { .ae-skel span { animation: none; } }
.ae-keys { font-size: 0.85rem; opacity: 0.65; margin: 0 0 1rem; }
.ae-keys kbd { font-family: ui-monospace, monospace; border: 1px solid color-mix(in srgb, currentColor 25%, transparent);
  border-radius: 4px; padding: 0 0.3rem; }
.ae-wordmark { font-size: 1.4rem; font-weight: 600; letter-spacing: -0.01em; margin: 0.25rem 0 0.4rem; }
.ae-tagline { margin: 0 0 0.5rem; opacity: 0.85; max-width: 46rem; }
.ae-facts { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 0.8rem; opacity: 0.6;
  margin: 0 0 1.25rem; }
.ae-cards { display: flex; flex-wrap: wrap; gap: 2rem; margin: 0.5rem 0 1rem; }
.ae-card { flex: 1 1 8rem; border-top: 1px solid color-mix(in srgb, currentColor 16%, transparent);
  padding-top: 0.6rem; }
.ae-card-label { font-size: 0.8rem; opacity: 0.6; }
.ae-card-value { font-size: 1.75rem; font-weight: 600; line-height: 1.15; font-variant-numeric: tabular-nums; }
.ae-card-context { font-size: 0.8rem; opacity: 0.55; }
.ae-first { border-left: 3px solid #E5484D; padding: 0.1rem 0 0.1rem 0.75rem; margin: 0.9rem 0 1rem; }
.ae-first-amber { border-left-color: #D9A441; }
.ae-first-ok { border-left-color: #8B929C; }
</style>
"""
st.html(STYLE)

# Keys 1-3 switch tabs. The script runs in a hidden component and attaches one listener to the page.
KEYS_JS = """
<script>
(function () {
  const doc = window.parent.document;
  if (doc.__aeKeysBound) return;
  doc.__aeKeysBound = true;
  doc.addEventListener("keydown", function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target;
    if (t && t.closest && t.closest("input, textarea, select, [contenteditable='true'], [role='listbox'], [role='combobox']")) return;
    const n = parseInt(e.key, 10);
    const tabs = doc.querySelectorAll('[data-testid="stTabs"] [role="tab"]');
    if (n >= 1 && n <= tabs.length) { tabs[n - 1].click(); e.preventDefault(); }
  });
})();
</script>
"""
components.html(KEYS_JS, height=0)


def skeleton(text: str) -> str:
    return f'<div class="ae-skel" role="status"><p>{text}</p><span></span><span></span><span></span></div>'


def label(metric: str) -> str:
    if metric.startswith("event: "):
        return f'"{metric[7:]}" messages'
    return LABELS.get(metric, (metric, ""))[0]


def status(c: Comparison) -> str:
    return "REGRESSED" if c.failed else ("changed" if c.flag_count else "ok")


def build_date(s: FlightSummary) -> str:
    stamp = Path(s.source).stem.split("-")[-1] if "autotest" in s.source else ""
    if stamp.isdigit():
        return datetime.fromtimestamp(int(stamp) / 1e6, tz=timezone.utc).strftime("%b %d")
    return ""


@st.cache_data
def load_history(folder: str) -> list[FlightSummary]:
    return [load(p) for p in sorted((EXAMPLES / folder).glob("*.json"))]


def scan_table(results: list[Comparison]) -> pd.DataFrame:
    rows = []
    for c in results:
        top = ""
        if c.findings:
            f = c.findings[0]
            top = f"{label(f.metric)}: {f.value:g} vs {f.median:g}"
        elif c.new_messages:
            top = f"new message: {c.new_messages[0].text}"
        elif c.missing_messages:
            top = f"missing message: {c.missing_messages[0]}"
        rows.append({"build": c.candidate.label, "date": build_date(c.candidate), "status": status(c),
                     "flags": c.flag_count, "biggest change": top})
    df = pd.DataFrame(rows)
    return df if df["date"].any() else df.drop(columns="date")


def metric_chart(results: list[Comparison], metric: str) -> alt.Chart:
    rows = []
    for c in results:
        flagged = any(f.metric == metric for f in c.findings)
        rows.append({"build": c.candidate.label, "value": c.candidate.metrics.get(metric),
                     "flagged": "flagged on this metric" if flagged else "normal"})
    df = pd.DataFrame(rows).dropna()
    # Bars would start at zero and hide exactly the difference that matters (257 s vs 244 s), so dots
    # on a zoomed axis, with the median of all runs as the "normal" line.
    x = alt.X("build:N", sort=None, title="firmware build, oldest to newest", axis=alt.Axis(labelAngle=-45))
    y = alt.Y("value:Q", title=label(metric), scale=alt.Scale(zero=False, padding=12))
    line = alt.Chart(df).mark_line(color="#8B929C", strokeWidth=1).encode(x=x, y=y)
    dots = alt.Chart(df).mark_circle(size=140, opacity=1).encode(
        x=x, y=y,
        color=alt.Color("flagged:N", scale=alt.Scale(domain=["normal", "flagged on this metric"],
                                                     range=["#8B929C", "#E5484D"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["build", alt.Tooltip("value:Q", format=".4g"), "flagged"],
    )
    median = alt.Chart(pd.DataFrame({"median": [df["value"].median()]})).mark_rule(
        strokeDash=[4, 4], color="#8B929C").encode(y="median:Q")
    return (line + median + dots).properties(height=300)


@st.cache_data(show_spinner="Checking the ArduPilot commit history...")
def cached_bisection(metric: str, base: str, head: str):
    b = bisect_mod.localize(metric, base, head)
    if b is not None and not b.error:
        b.ai = bisect_mod.ai_verdict(b)
    return b


def show_bisection(c: Comparison, ordered_runs: list[FlightSummary]) -> None:
    pair = bisect_mod.neighbor_build(c.candidate, ordered_runs)
    if pair is None:
        return
    base, head = pair
    b = cached_bisection(c.regressions[0].metric, base.label, head.label)
    if b is None:
        return
    st.markdown(f"**Possible cause** ([{b.base}...{b.head}]({b.compare_url}))")
    if b.error:
        st.caption(f"Could not pin down the cause: {b.error}")
        return
    if not b.suspects:
        st.caption(f"None of the {b.total_commits} commits in this range touch code related to "
                   f"{label(c.regressions[0].metric)}, which points to noise rather than a firmware bug.")
    else:
        st.caption(f"{len(b.suspects)} of {b.total_commits} commits touch related code:")
        for s in b.suspects:
            st.markdown(f"- `{s.sha}` {s.message}: {', '.join(s.files)}")
    if b.ai:
        st.markdown(f"**AI read: {b.ai.label}.** {b.ai.reason}")


def findings_table(c: Comparison) -> pd.DataFrame:
    return pd.DataFrame([{
        "": {"worse": "worse", "better": "better", "changed": "changed"}[f.verdict],
        "what": label(f.metric),
        "this run": f.value,
        "normal (median)": f.median,
        "change": f"{f.pct:+.1f}%" if f.pct is not None else "",
    } for f in c.findings])


def show_detail(c: Comparison, ordered_runs: list[FlightSummary]) -> None:
    cand = c.candidate
    st.markdown(f"**{cand.label}** ({vehicle_kind(cand.vehicle, cand.mav_type)}), compared with "
                f"{len(c.baseline)} other runs: {', '.join(b.label for b in c.baseline)}")
    for w in c.warnings:
        st.caption(f"note: {w}")
    if not c.flag_count:
        st.success("Nothing regressed. Every metric sits within the normal range of the other runs.")
        return
    if c.findings:
        st.table(findings_table(c).style.hide(axis="index"))
    if c.regressions:
        show_bisection(c, ordered_runs)
    if c.new_alarms:
        st.error("These warnings or errors have never appeared in the other runs:\n\n" +
                 "\n".join(f"- [{SEVERITY[m.severity]}] {m.text}" for m in c.new_alarms))
    info = [m for m in c.new_messages if not m.alarming]
    if info:
        st.info("New messages (for information only):\n\n" + "\n".join(f"- {m.text}" for m in info))
    if c.missing_messages:
        st.info("Messages that appear in every other run but are missing here:\n\n" +
                "\n".join(f"- {m}" for m in c.missing_messages))
    if cand.source.startswith("http"):
        st.caption(f"Raw log: {cand.source}")


@st.cache_data(show_spinner=False)
def cached_sample(kind: str) -> CheckOutcome:
    """Each visitor shares the result: the slow check runs once per sample, not once per visit."""
    return run_sample(kind)


def in_temp_file(name: str, data: bytes, fn, *args):
    """Run a reader that needs a file path, on the uploaded bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / Path(name).name
        p.write_bytes(data)
        return fn(p, *args)


@st.cache_data(show_spinner="Reading the message list...")
def log_signals(name: str, data: bytes) -> dict[str, list[str]]:
    return in_temp_file(name, data, list_signals)


@st.cache_data(show_spinner="Reading the signal from the log...")
def log_signal(name: str, data: bytes, msg: str, field: str):
    return in_temp_file(name, data, read_signal, msg, field)


@st.cache_data(show_spinner=False)
def log_arm_time(name: str, data: bytes):
    return in_temp_file(name, data, arm_time)


@st.cache_data(show_spinner="Reading scheduler and IMU timing...")
def log_health(name: str, data: bytes):
    return in_temp_file(name, data, timing_health)


@st.fragment
def health_section(bins: list) -> None:
    """Scheduler, log and IMU timing for each uploaded .bin log."""
    st.subheader("Logging and timing health")
    st.caption("For each log: how often the scheduler overran, how long its loops took, the logged CPU load, "
               "dropped log blocks, and the worst gap in the IMU clock. A change here usually points at the "
               "firmware rather than the tuning.")
    rows = []
    for f in bins:
        h = log_health(f.name, f.getvalue())
        if h is None:
            continue
        rows.append({"log": f.name, "loops overrun (%)": h["overrun_pct"], "max loop time": h["max_loop_time"],
                     "max CPU load": h["max_cpu_load"], "dropped log blocks": h["dropped_blocks"],
                     "worst IMU gap (x median)": h["worst_imu_gap"]})
    if not rows:
        st.info("None of these logs has scheduler (PM) messages, so timing health cannot be read.")
        return
    table = pd.DataFrame(rows).set_index("log").round(2)
    st.table(table.style.format(na_rep="n/a"))
    st.caption("On the simulated (SITL) logs we checked, CPU load stayed at 0, so the table looks healthy there. "
               "Use logs from real hardware to see a real change in load.")


@st.cache_data(show_spinner="Reading the flight...")
def fd_call(name: str, data: bytes, fn_name: str, args: tuple = ()):
    """Run one flightdata reader on the uploaded bytes, cached per log and call."""
    return in_temp_file(name, data, getattr(flightdata, fn_name), *args)


def _thin(df: pd.DataFrame, limit: int = 4000) -> pd.DataFrame:
    step = max(1, len(df) // limit)
    return df.iloc[::step]


def _fmt(value, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def summary_markdown(runs: list[FlightSummary], bins: list, notes: list) -> str:
    """Markdown summary: the comparison per vehicle, the fault checks and timing per log, and the notes."""
    lines = ["# Autonomy-eval summary", ""]
    groups: dict[tuple, list[FlightSummary]] = {}
    for r in runs:
        groups.setdefault((r.vehicle, r.mav_type), []).append(r)
    for (vehicle, mav_type), group in groups.items():
        lines.append(f"## {vehicle_kind(vehicle, mav_type)}: {len(group)} log(s)")
        if len(group) <= MIN_BASELINE:
            lines += [f"Needs more than {MIN_BASELINE} logs to compare.", ""]
            continue
        try:
            results = scan(group)
        except Incomparable as e:
            lines += [f"Could not compare: {e}", ""]
            continue
        for c in results:
            lines.append(f"- {c.candidate.label}: {status(c)} ({c.flag_count} flagged)")
        lines.append("")
    lines += ["## Checks per log", ""]
    for f in bins:
        data = f.getvalue()
        lines.append(f"### {f.name}")
        for row in fd_call(f.name, data, "fault_checks"):
            lines.append(f"- {row['check']}: {row['status']} ({row['detail']})")
        h = log_health(f.name, data)
        if h is not None:
            lines.append(f"- Timing: overruns {_fmt(h['overrun_pct'], '.2f')}%, "
                         f"max loop {_fmt(h['max_loop_time'], '.0f')}, max CPU load {_fmt(h['max_cpu_load'], '.0f')}")
        lines.append("")
    if notes:
        lines += ["## Notes", ""]
        lines += [f"- {n['log']} {n['from (s)']:g} to {n['to (s)']:g} s: {n['note']}" for n in notes]
        lines.append("")
    return "\n".join(lines) + "\n"


@st.fragment
def views_section(bins: list) -> None:
    """Map, modes, outputs, firmware, fault checks, derived signals, rate tracking, spectrum, notes and summary."""
    st.subheader("More views")
    names = [f.name for f in bins]
    with st.expander("Load a saved view"):
        loaded = st.file_uploader("Choose a view file (JSON)", type=["json"], key="view-up")
        if loaded is not None and st.session_state.get("view-applied") != loaded.file_id:
            try:
                saved = json.loads(loaded.getvalue())
                if saved.get("log") in names:
                    st.session_state["views-log"] = saved["log"]
                st.session_state["views-msg"] = str(saved.get("msg", "BAT"))
                st.session_state["views-expr"] = str(saved.get("expr", "Volt * Curr"))
                st.session_state["notes"] = list(saved.get("notes", []))
                st.session_state["view-applied"] = loaded.file_id
                st.success("View loaded. Mode choices are not restored; pick them again if needed.")
            except (ValueError, AttributeError, TypeError):
                st.error("That file is not a saved view.")
    log_name = st.selectbox("Log for these views", names, key="views-log")
    data = next(f.getvalue() for f in bins if f.name == log_name)
    notes = st.session_state.setdefault("notes", [])

    with st.expander("Flight modes and the mode filter"):
        segs = fd_call(log_name, data, "mode_segments")
        if not segs:
            st.info("This log has no MODE messages.")
        else:
            modes = list(dict.fromkeys(s[3] for s in segs))
            keep = st.multiselect("Modes to keep in the rate chart", modes, default=modes, key="views-modes")
            band = pd.DataFrame([{"start": s[0], "end": s[1], "mode": s[3]} for s in segs])
            st.altair_chart(
                alt.Chart(band).mark_rect(opacity=0.5).encode(
                    x=alt.X("start:Q", title="time (s)"), x2="end:Q",
                    color=alt.Color("mode:N", legend=alt.Legend(title=None, orient="top"))).properties(height=60),
                width="stretch")

    with st.expander("Roll rate: desired against measured"):
        t, des, act = fd_call(log_name, data, "rate_tracking", ("R",))
        df = pd.DataFrame({"t": t, "desired": des, "measured": act})
        if segs and keep:
            mask = np.zeros(t.size, dtype=bool)
            for s in segs:
                if s[3] in keep:
                    mask |= (t >= s[0]) & (t < s[1])
            df = df[mask]
        if len(df):
            err = df["desired"].to_numpy() - df["measured"].to_numpy()
            st.metric("RMS tracking error for the modes kept (rad/s)", f"{float(np.sqrt(np.mean(err ** 2))):.4f}")
        long = _thin(df).melt("t", var_name="series", value_name="rate (rad/s)")
        st.altair_chart(
            alt.Chart(long).mark_line(strokeWidth=1.2).encode(
                x=alt.X("t:Q", title="time (s)"), y=alt.Y("rate (rad/s):Q", scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", legend=alt.Legend(title=None, orient="top"))).properties(height=260),
            width="stretch")
        st.caption("Only the modes chosen above are drawn. This shows how closely the roll rate follows its target; "
                   "it does not estimate PID gains.")

    with st.expander("Map of the track"):
        lat, lng = fd_call(log_name, data, "gps_track")
        if lat.size == 0:
            st.info("No 3D GPS fix in this log.")
        else:
            st.map(_thin(pd.DataFrame({"lat": lat, "lon": lng}), 2000), size=4)
            st.caption(f"{lat.size} GPS samples with a 3D fix or better.")

    with st.expander("Motor and servo outputs"):
        t, c = fd_call(log_name, data, "actuator_outputs")
        if not c:
            st.info("This log has no RCOU messages.")
        else:
            df = pd.DataFrame({"t": t, **c}).melt("t", var_name="channel", value_name="PWM (us)")
            st.altair_chart(
                alt.Chart(_thin(df)).mark_line(strokeWidth=1.2).encode(
                    x=alt.X("t:Q", title="time (s)"), y=alt.Y("PWM (us):Q", scale=alt.Scale(zero=False)),
                    color=alt.Color("channel:N", legend=alt.Legend(title=None, orient="top"))).properties(height=260),
                width="stretch")

    with st.expander("Firmware and parameters"):
        info = fd_call(log_name, data, "firmware_info")
        st.table(pd.DataFrame([{
            "firmware": info["version"] or "not in log",
            "parameters": info["parameters"],
            "log length (s)": round(info["log_seconds"], 1),
        }]).style.hide(axis="index"))

    with st.expander("Fault checks"):
        rows = fd_call(log_name, data, "fault_checks")
        table = pd.DataFrame(rows)
        st.table(table.style.map(
            lambda v: "color: #E5484D; font-weight: 600" if v == "found"
            else ("opacity: 0.55" if v == "not in log" else ""), subset=["status"]).hide(axis="index"))
        st.caption("Each check uses a simple threshold: for example, a motor counts as saturated when it sits at "
                   "1950 us or more on over 2% of spinning samples. The thresholds are starting points, not tuned.")

    with st.expander("Derived signal"):
        msg = st.text_input("Message type", "BAT", key="views-msg")
        expr = st.text_input("Expression, using that message's fields", "Volt * Curr", key="views-expr")
        try:
            t, v = fd_call(log_name, data, "derived_signal", (msg, expr))
            st.altair_chart(
                alt.Chart(_thin(pd.DataFrame({"t": t, "value": v}))).mark_line(strokeWidth=1.2).encode(
                    x=alt.X("t:Q", title="time (s)"), y=alt.Y("value:Q", title=expr, scale=alt.Scale(zero=False))
                ).properties(height=260),
                width="stretch")
        except ValueError as e:
            st.error(str(e))
        st.caption("Allowed: numbers, the message's field names, + - * /, unary minus, abs() and sqrt(). "
                   "Nothing else runs.")

    with st.expander("Vibration spectrum (gyro)"):
        picked = st.multiselect("Logs", names, default=[log_name], key="views-spec-logs")
        frames, rates = [], []
        for n in picked:
            d = next(f.getvalue() for f in bins if f.name == n)
            try:
                f, p, fs = in_temp_file(n, d, batch.batch_spectrum, 0)
                source = "sensor-rate batches"
            except ValueError:
                try:
                    f, p, fs = fd_call(n, d, "gyro_spectrum", ("GyrX",))
                    source = "IMU, 50 Hz"
                except ValueError as e:
                    st.warning(f"{n}: {e}")
                    continue
            rates.append(f"{n}: {fs:.0f} Hz, {source}")
            frames.append(pd.DataFrame({"frequency (Hz)": f, "power": p + 1e-20, "log": n}))
        if frames:
            st.altair_chart(
                alt.Chart(pd.concat(frames)).mark_line(strokeWidth=1.2).encode(
                    x=alt.X("frequency (Hz):Q"), y=alt.Y("power:Q", scale=alt.Scale(type="log"), title="power"),
                    color=alt.Color("log:N", legend=alt.Legend(title=None, orient="top"))).properties(height=260),
                width="stretch")
            st.caption("Sample rates: " + "; ".join(rates) + ". A spectrum reaches only half its sample rate. "
                       "Batch blocks alternate gyro and accel, so the gyro is sampled about half the time. "
                       "Logs without batches use the 50 Hz IMU stream, which shows up to 25 Hz.")

    with st.expander("Roll-rate tracking by gain"):
        picked_g = st.multiselect("Logs", names, default=[log_name], key="views-gain-logs")
        rows = []
        for n in picked_g:
            d = next(f.getvalue() for f in bins if f.name == n)
            try:
                s = fd_call(n, d, "roll_tracking_summary")
            except ValueError as e:
                st.warning(f"{n}: {e}")
                continue
            rows.append({"log": n, "ATC_RAT_RLL_P": s["gain"], "roll error RMS (deg/s)": round(s["rms"], 3),
                         "RATE samples": s["samples"]})
        if rows:
            st.table(pd.DataFrame(rows))
            st.caption("Lower RMS means the roll rate followed its command more closely. "
                       "One run per gain shows a direction, not a fitted optimum.")

    with st.expander("Notes on the timeline"):
        c1, c2, c3 = st.columns([1, 1, 3])
        start = c1.number_input("From (s)", min_value=0.0, value=0.0, step=1.0, key="note-start")
        end = c2.number_input("To (s)", min_value=0.0, value=0.0, step=1.0, key="note-end")
        text = c3.text_input("Note", key="note-text")
        if st.button("Add note", key="note-add"):
            if text and end > start:
                notes.append({"log": log_name, "from (s)": start, "to (s)": end, "note": text})
            else:
                st.warning("A note needs text and an end time after the start time.")
        if notes:
            st.table(pd.DataFrame(notes).style.hide(axis="index"))
        st.download_button("Download notes (JSON)", json.dumps(notes, indent=1), file_name="notes.json",
                           mime="application/json", key="notes-dl")

    with st.expander("Save this view, or download a summary"):
        view = {"log": log_name, "msg": st.session_state.get("views-msg", "BAT"),
                "expr": st.session_state.get("views-expr", "Volt * Curr"), "notes": notes}
        st.download_button("Download this view (JSON)", json.dumps(view, indent=1), file_name="view.json",
                           mime="application/json", key="view-dl")
        if st.session_state.get("uploaded_runs"):
            # Built only on request: it reads every uploaded log.
            if st.button("Prepare the summary", key="summary-prep"):
                st.session_state["summary_text"] = summary_markdown(st.session_state["uploaded_runs"], bins, notes)
            if "summary_text" in st.session_state:
                st.download_button("Download summary (Markdown)", st.session_state["summary_text"],
                                   file_name="summary.md", mime="text/markdown", key="summary-dl")
        else:
            st.caption("Analyze uploaded logs first to include the comparison in the summary.")


@st.fragment
def signal_section(bins: list) -> None:
    """Plot one logged field over time, for one or several logs. A fragment, so the pickers do not rerun the page."""
    st.subheader("Plot any signal")
    st.caption("Pick a message and a field to see it over time. Choose several logs to overlay them. "
               "Text fields are left out.")
    names = [f.name for f in bins]
    picked = st.multiselect("Logs", names, default=names[:2], key="sig-logs")
    if not picked:
        st.info("Pick at least one log.")
        return
    datas = {f.name: f.getvalue() for f in bins if f.name in picked}
    lists = [log_signals(n, d) for n, d in datas.items()]
    common = sorted(set.intersection(*(set(s) for s in lists)))
    if not common:
        st.info("These logs have no message type in common.")
        return
    msg = st.selectbox("Message", common, key="sig-msg")
    fields = [f for f in lists[0][msg] if all(f in s[msg] for s in lists)]
    field = st.selectbox("Field", fields, key="sig-field")
    align = st.toggle("Line up on arming", value=True, key="sig-align")

    frames, unaligned = [], []
    for n, d in datas.items():
        t, v = log_signal(n, d, msg, field)
        if not t:
            continue
        offset = 0.0
        if align:
            at = log_arm_time(n, d)
            if at is None:
                unaligned.append(n)
            else:
                offset = at
        step = max(1, len(t) // 4000)     # about 4000 points per log is plenty to see a signal
        frames.append(pd.DataFrame({"t": [x - offset for x in t[::step]], "value": v[::step], "log": n}))
    if not frames:
        st.info(f"{msg}.{field} has no samples in the chosen logs.")
        return
    df = pd.concat(frames, ignore_index=True)
    chart = alt.Chart(df).mark_line(strokeWidth=1.3).encode(
        x=alt.X("t:Q", title="time from arming (s)" if align else "time (s)"),
        y=alt.Y("value:Q", title=f"{msg}.{field}", scale=alt.Scale(zero=False)),
        color=alt.Color("log:N", legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("log:N", title="log"), alt.Tooltip("t:Q", title="time (s)", format=".1f"),
                 alt.Tooltip("value:Q", title=f"{msg}.{field}", format=".4g")],
    ).properties(height=300)
    st.altair_chart(chart, width="stretch")
    note = "Each line is thinned to at most 4000 points."
    if align and unaligned:
        note += " Logs with no arm record start from their first sample: " + ", ".join(unaligned) + "."
    st.caption(note)


def regression_text(outcome: CheckOutcome) -> str:
    """One verdict line for a failed check, with the triage folded in so the page never says two things."""
    if outcome.triage:
        verdict, items = outcome.triage
        if verdict == "LIKELY REAL":
            return f"This looks like a real regression: {', '.join(items)} fell outside the normal range."
        if verdict == "POSSIBLE":
            return (f"This may be noise. Only {items[0]} fell outside the normal range, "
                    "so rerun the flight before trusting it.")
        return "This is probably noise. The run as a whole matches the normal range of the earlier flights."
    return "The run fell outside the normal range of the earlier flights."


def show_check(outcome: CheckOutcome, kind: str) -> None:
    if outcome.error:
        st.error(outcome.error)
        return
    c = outcome.comparison
    st.markdown(f"**{c.candidate.label}**, compared with {len(c.baseline)} earlier flight(s)")
    if kind == "step":
        st.info("For information only: step metrics never fail a build.")
    elif c.failed:
        st.error(regression_text(outcome))
    else:
        st.success("Nothing regressed. The run sits within the normal range of the earlier flights.")
    if outcome.note:
        st.caption(outcome.note)
    if c.findings:
        st.table(findings_table(c).style.hide(axis="index"))
    for w in c.warnings:
        st.caption(f"note: {w}")


@st.fragment
def detail_section(results: list[Comparison], runs: list[FlightSummary], key: str,
                   default_metric: str | None, worst: str) -> None:
    """Chart and per-run detail. A fragment, so changing a selector does not rerun the whole page."""
    labels = [c.candidate.label for c in results]
    flagged_metrics = [f.metric for c in results for f in c.findings if not f.metric.startswith("event:")]
    metrics = sorted({m for r in runs for m in r.metrics})
    default = default_metric or (flagged_metrics[0] if flagged_metrics else "armed_time_s")
    with st.expander("Charts and details"):
        metric = st.selectbox("Pick a metric to compare across builds", metrics,
                              index=metrics.index(default) if default in metrics else 0,
                              format_func=label, key=f"{key}-metric")
        st.altair_chart(metric_chart(results, metric), width="stretch")
        choice = st.selectbox("Pick one build to look at closely", labels, index=labels.index(worst), key=f"{key}-detail")
        show_detail(next(c for c in results if c.candidate.label == choice), runs)


def stat_cards(regressed: int, changed: int, ok: int, total: int) -> None:
    """Three plain numbers with their context. Only a non-zero regressed count is coloured."""
    def card(label: str, value: int, color: str | None = None) -> str:
        style = f' style="color: {color}"' if color else ""
        return (f'<div class="ae-card"><div class="ae-card-label">{label}</div>'
                f'<div class="ae-card-value"{style}>{value}</div>'
                f'<div class="ae-card-context">of {total} builds</div></div>')
    st.html('<div class="ae-cards">'
            + card("Regressed", regressed, "#E5484D" if regressed else None)
            + card("Changed", changed) + card("OK", ok) + "</div>")


def results_section(runs: list[FlightSummary], key: str, default_metric: str | None = None) -> None:
    results = scan(runs)
    df = scan_table(results)
    counts = {s: sum(status(c) == s for c in results) for s in ("REGRESSED", "changed", "ok")}
    worst = next((c.candidate.label for c in results if c.failed), results[0].candidate.label)
    # Three numbers first, then one sentence. The table, chart and per-run detail below are for checking them.
    stat_cards(counts["REGRESSED"], counts["changed"], counts["ok"], len(results))
    if counts["REGRESSED"]:
        st.html(f'<p class="ae-first">{counts["REGRESSED"]} of {len(results)} builds regressed. '
                f'The first one is {worst}.</p>')
    elif counts["changed"]:
        st.html(f'<p class="ae-first ae-first-amber">Nothing regressed. {counts["changed"]} of {len(results)} '
                'builds changed on at least one metric.</p>')
    else:
        st.html(f'<p class="ae-first ae-first-ok">Nothing regressed across the {len(results)} builds.</p>')

    only_changed = st.toggle("Show only builds that changed", key=f"{key}-only")
    shown = df[df["status"] != "ok"] if only_changed else df
    if shown.empty:
        st.caption("No build changed. Turn the filter off to see them all.")
    else:
        st.table(shown.style.map(lambda v: f"color: {STATUS_COLORS.get(v, 'inherit')}; font-weight: 600",
                                 subset=["status"]).hide(axis="index"))
    with st.popover("What does normal mean?"):
        st.markdown(
            "\"Normal\" is the typical value across the other runs, one metric at a time (their median). "
            "A metric gets flagged when this run lands outside the range the other runs cover."
        )
    st.download_button("Download this table (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{key}-builds.csv", mime="text/csv", key=f"{key}-csv")
    detail_section(results, runs, key, default_metric, worst)


st.html(
    '<div class="ae-wordmark">Autonomy-eval</div>'
    '<p class="ae-tagline">Compares a flight log with earlier flights of the same vehicle, and flags anything '
    "outside their normal range. It shows whether a build flew worse than the ones before it.</p>"
    '<p class="ae-facts">ArduPilot CI history · 2 vehicles · 8 builds each</p>'
)
st.markdown(f"[Source on GitHub ↗]({REPO})")
st.html('<p class="ae-keys">Press 1, 2 or 3 to switch tabs</p>')

tab_ci, tab_upload, tab_checks = st.tabs(["Example result", "Check your logs", "Check one flight"])

with tab_ci:
    st.markdown(
        "ArduPilot's CI flies the same test missions on every firmware build and saves the telemetry. "
        "Each build here is compared with the other seven."
    )
    # A shared link can open a given history: ?history=ardurover-ground-rover
    linked = st.query_params.get("history")
    start = next((k for k, v in HISTORIES.items() if v == linked), next(iter(HISTORIES)))
    history = st.segmented_control("History", list(HISTORIES), default=start, key="ci-history",
                                   label_visibility="collapsed") or start
    folder = HISTORIES[history]
    st.query_params["history"] = folder
    results_section(load_history(folder), key=folder,
                    default_metric="armed_time_s" if folder == "ardurover-surface-boat" else None)
    if folder == "ardurover-surface-boat":
        st.markdown(
            "**What happened in build bf080274:** the boat tacked one extra time, which alone explains "
            "the 13 extra seconds and the worse tracking. The \"possible cause\" above was found automatically, "
            "not written by hand, and it agrees. The one commit it flags only moves a parameter table and does not "
            "change control logic. So the likely cause is that the simulator made a different tack decision. "
            "That is a flaky test, not a firmware bug."
        )
    else:
        st.markdown(
            "No build regressed. This history was held out while the rules were developed; the first "
            "run on it produced two false alarms, which were fixed "
            f"([before and after]({REPO}#honest-results))."
        )

with tab_upload:
    st.markdown(f"To compare runs, upload at least {MIN_BASELINE + 1} logs of the **same vehicle and mission**.")
    st.markdown(
        "- **Formats:** MAVLink telemetry (`.tlog`), ArduPilot DataFlash logs (`.bin`), or summaries saved by "
        "`autonomy-eval summarize -o` (`.json`).\n"
        "- **Oldest build first.** The possible cause of a regression comes from the commits between it and the "
        "build just before it.\n"
        "- **Private:** your files are read into memory for this session only and are never kept."
    )
    files = st.file_uploader("Flight logs", type=["tlog", "bin", "json"], accept_multiple_files=True)
    if not files:
        st.info("No logs yet? The Example result tab (press 1) shows the same comparison on real build history.")
    if files and st.button(f"Analyze {len(files)} log(s)", type="primary"):
        runs, failed = [], []
        progress = st.progress(0.0, text="Reading logs...")
        with tempfile.TemporaryDirectory() as tmp:
            for i, f in enumerate(files):
                path = Path(tmp) / Path(f.name).name
                path.write_bytes(f.getvalue())
                try:
                    s = load(path)
                except Exception as e:  # a corrupt or unsupported upload should not take the page down
                    failed.append(f"{f.name}: {e}")
                else:
                    if s.vehicle is None and s.mav_type is None:
                        failed.append(f"{f.name}: no autopilot found in it (not a flight log?)")
                    else:
                        s.source = f.name
                        runs.append(s)
                progress.progress((i + 1) / len(files), text=f"Read {f.name}")
        progress.empty()
        st.caption(f"Read {len(runs)} of {len(files)} files" + (f"; {len(failed)} could not be read." if failed else "."))
        for msg in failed:
            st.warning(f"Could not read {msg}")
        st.session_state["uploaded_runs"] = runs
    if not files:
        st.session_state.pop("uploaded_runs", None)   # cleared files must clear the old results too
    runs = st.session_state.get("uploaded_runs", [])
    groups: dict[tuple, list[FlightSummary]] = {}
    for r in runs:
        groups.setdefault((r.vehicle, r.mav_type), []).append(r)
    for (vehicle, mav_type), group in groups.items():
        st.subheader(f"{vehicle_kind(vehicle, mav_type)}: {len(group)} log(s)")
        if len(group) <= MIN_BASELINE:
            st.warning(f"Upload more than {MIN_BASELINE} logs of this vehicle to compare them.")
            continue
        results_section(group, key=f"upload-{vehicle}-{mav_type}")
    bins = [f for f in (files or []) if f.name.lower().endswith(".bin")]
    if bins:
        signal_section(bins)
        health_section(bins)
        views_section(bins)

with tab_checks:
    st.markdown(
        "These are the same stability and estimator checks the command line runs. Use your own flights or a sample. Each one reads "
        "raw DataFlash `.bin` logs and needs a flight made for it: a SystemID chirp for stability margins, "
        "a SITL run with ground truth for the estimator, or GUIDED position steps for the step response."
    )
    kind = st.segmented_control("Check", list(CHECKS), format_func=lambda k: CHECKS[k][0],
                                default=next(iter(CHECKS)), key="chk-kind") or next(iter(CHECKS))
    st.caption(CHECKS[kind][1])

    st.subheader("Try a sample flight")
    st.caption("These are simulated flights with a known fault. The first run downloads them; later runs reuse them.")
    for sample_kind, button_text in SAMPLES:
        if st.button(button_text, key=f"chk-sample-{sample_kind}"):
            wait = ("Downloading about 300 MB. This happens only on the first run..." if sample_kind == "sysid"
                    else "Downloading the sample flights. This happens only on the first run...")
            slot = st.empty()
            slot.html(skeleton(wait))
            try:
                st.session_state[f"chk-result-{sample_kind}"] = cached_sample(sample_kind)
                st.toast(f"The {CHECKS[sample_kind][0].lower()} sample is ready")
            except Exception as e:  # no network, or a bad download, should not take the page down
                st.session_state[f"chk-result-{sample_kind}"] = CheckOutcome(
                    error=f"The sample flights could not be loaded: {e}")
            slot.empty()
        if f"chk-result-{sample_kind}" in st.session_state:
            st.markdown(f"**Sample: {CHECKS[sample_kind][0]}**")
            show_check(st.session_state[f"chk-result-{sample_kind}"], sample_kind)

    st.subheader("Check your own flights")
    cand = st.file_uploader("New flight (.bin)", type=["bin"], accept_multiple_files=False, key="chk-candidate")
    base = st.file_uploader(f"Earlier flights of the same mission (.bin, at least {MIN_BASELINE})",
                            type=["bin"], accept_multiple_files=True, key="chk-baseline")
    want_triage = st.checkbox("Also judge whether a flagged result is real or just noise", value=True,
                              key="chk-triage")
    if cand is None:
        st.session_state.pop("chk-result-upload", None)
        st.caption("Upload a new flight to begin.")
    elif len(base) < MIN_BASELINE:
        st.caption(f"Upload at least {MIN_BASELINE} earlier flights to compare with.")
    elif st.button("Run check", type="primary", key="chk-run"):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, f in enumerate([cand, *base]):
                d = Path(tmp) / str(i)
                d.mkdir()
                p = d / Path(f.name).name
                p.write_bytes(f.getvalue())
                paths.append(str(p))
            try:
                outcome = run_check(kind, paths[0], paths[1:], triage=want_triage)
            except Exception as e:  # a corrupt or unsupported log should not take the page down
                outcome = CheckOutcome(error=f"These logs could not be read: {e}")
        st.session_state["chk-result-upload"] = (outcome, kind)
    if "chk-result-upload" in st.session_state:
        shown_outcome, shown_kind = st.session_state["chk-result-upload"]
        show_check(shown_outcome, shown_kind)

    with st.expander("Why the telemetry and onboard logs differ"):
        st.markdown(
            "ArduPilot records every flight twice: telemetry sent to the ground station (`.tlog`) and the "
            "autopilot's own onboard log (`.bin`). The same two flights, read both ways:"
        )
        st.table(pd.DataFrame([
            {"": "time armed (s)", "boat .bin": 107.3, "boat .tlog": 107.0, "rover .bin": 243.1, "rover .tlog": 243.0},
            {"": "mean cross-track error (m)", "boat .bin": 1.400, "boat .tlog": 1.405, "rover .bin": 0.598, "rover .tlog": 0.600},
            {"": "worst cross-track error (m)", "boat .bin": 4.381, "boat .tlog": 4.381, "rover .bin": 2.959, "rover .tlog": 2.959},
            {"": "lowest battery (V)", "boat .bin": 11.90, "boat .tlog": 11.90, "rover .bin": 12.53, "rover .tlog": 12.53},
            {"": "in-flight mode changes", "boat .bin": 0, "boat .tlog": 0, "rover .bin": 17, "rover .tlog": 17},
        ]).style.hide(axis="index"))
        st.markdown(
            "Where they disagree, the cause is known: DataFlash stores EKF variances at 0.01 resolution; "
            "the `.bin` catches a half-second arm/disarm that 1 Hz telemetry misses; and DataFlash has error "
            f"records telemetry doesn't carry. [Full output]({REPO}/blob/main/docs/cross-format-check.txt)"
        )
