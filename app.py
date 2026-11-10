"""Web demo: the CI result, a way to try your own logs, the stability and estimator checks, and the cross-format check.

    streamlit run app.py
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from autonomy_eval import bisect as bisect_mod
from autonomy_eval.checks import CHECKS, CheckOutcome, run_check
from autonomy_eval.compare import MIN_BASELINE, Comparison, scan, vehicle_kind
from autonomy_eval.metrics import FlightSummary, load
from autonomy_eval.report import LABELS, SEVERITY

EXAMPLES = Path(__file__).parent / "examples"
REPO = "https://github.com/rkb32/autonomy-eval"
HISTORIES = {
    "Sailboat - 8 daily builds": "ardurover-surface-boat",
    "Ground rover - 8 daily builds": "ardurover-ground-rover",
}
STATUS_COLORS = {"REGRESSED": "#d62728", "changed": "#ff9f1c", "ok": "#9aa0a6"}

st.set_page_config(page_title="autonomy-eval", page_icon="🛰️", layout="wide")

# Visual layer only. Colors that must work in light and dark mode are derived from the current text
# color (currentColor / color-mix), so they flip with the theme instead of being hard-coded per mode.
STYLE = """
<style>
:root { --ae-red: #d62728;
  --ae-gh: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Cpath d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38v-1.33c-2.23.48-2.7-1.07-2.7-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.06-.49.06-.49.8.06 1.23.83 1.23.83.72 1.22 1.88.87 2.34.66.07-.52.28-.87.51-1.07-1.78-.2-3.65-.89-3.65-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8Z'/%3E%3C/svg%3E");
  --ae-serif: "Source Serif", ui-serif, Georgia, serif; }
[data-testid="stMainBlockContainer"] { max-width: 1120px; padding-top: 2.5rem; padding-bottom: 4rem; }

/* hero */
.ae-hero { padding: 0.5rem 0 1.75rem; margin-bottom: 0.5rem;
  border-bottom: 1px solid color-mix(in srgb, currentColor 12%, transparent); }
.ae-hero h1 { font-family: var(--ae-serif); font-weight: 600; font-size: clamp(2.4rem, 6vw, 3.6rem);
  letter-spacing: -0.025em; line-height: 1.05; margin: 0 0 1rem; padding: 0; }
.ae-lede { font-family: var(--ae-serif); font-size: clamp(1.1rem, 2.2vw, 1.32rem); line-height: 1.55;
  max-width: 46rem; margin: 0 0 1.4rem; opacity: 0.92; }
.ae-lede strong { color: var(--ae-red); font-weight: 600; }
.ae-cta { display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.5rem 0.95rem; border-radius: 0.5rem;
  font-size: 0.92rem; font-weight: 600; text-decoration: none !important; color: inherit !important;
  border: 1px solid color-mix(in srgb, currentColor 22%, transparent); transition: border-color .15s, background .15s; }
.ae-cta:hover { border-color: var(--ae-red); background: color-mix(in srgb, var(--ae-red) 8%, transparent); }
.ae-cta::before { content: ""; width: 16px; height: 16px; background: currentColor;
  -webkit-mask: var(--ae-gh) center / contain no-repeat; mask: var(--ae-gh) center / contain no-repeat; }

/* tabs */
[data-testid="stTabs"] [role="tablist"] { gap: 1.75rem; overflow-x: auto; scrollbar-width: none;
  border-bottom: 1px solid color-mix(in srgb, currentColor 12%, transparent); }
[data-testid="stTab"] { padding: 0.85rem 0 0.75rem; height: auto; background: transparent; }
[data-testid="stTab"] p { font-size: 1rem !important; font-weight: 600; opacity: 0.6; white-space: nowrap; }
[data-testid="stTab"]:hover p, [data-testid="stTab"][aria-selected="true"] p { opacity: 1; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top: 1.6rem; }

/* tab intro paragraphs */
[class*="st-key-ae-intro"] p { font-size: 1.04rem; line-height: 1.65; max-width: 48rem; opacity: 0.85; }

/* widget labels read as section headings */
[data-testid="stSelectbox"] label p, [data-testid="stFileUploader"] label p {
  font-family: var(--ae-serif); font-size: 1.2rem; font-weight: 600; opacity: 1; }
[data-testid="stSelectbox"] { padding-top: 1.4rem; margin-top: 0.6rem;
  border-top: 1px solid color-mix(in srgb, currentColor 10%, transparent); }
[data-testid="stSelectbox"] > div:last-child { max-width: 26rem; }

/* history switch as a segmented control */
[class*="st-key-ae-history"] [role="radiogroup"] { gap: 0.5rem; flex-wrap: wrap; }
[class*="st-key-ae-history"] [role="radiogroup"] label { margin: 0; padding: 0.35rem 0.85rem; border-radius: 999px;
  border: 1px solid color-mix(in srgb, currentColor 16%, transparent); }
[class*="st-key-ae-history"] [role="radiogroup"] label:has(input:checked) {
  border-color: var(--ae-red); background: color-mix(in srgb, var(--ae-red) 8%, transparent); }

/* result summary */
.ae-summary { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.25rem 0 0.9rem; }
.ae-stat { display: flex; align-items: baseline; gap: 0.45rem; padding: 0.55rem 0.9rem; border-radius: 0.5rem;
  border: 1px solid color-mix(in srgb, currentColor 12%, transparent); min-width: 7.5rem; }
.ae-stat b { font-family: var(--ae-serif); font-size: 1.5rem; font-weight: 600; line-height: 1; }
.ae-stat span { font-size: 0.82rem; letter-spacing: 0.04em; text-transform: uppercase; opacity: 0.7; }
.ae-stat.bad { border-color: color-mix(in srgb, var(--ae-red) 55%, transparent);
  background: color-mix(in srgb, var(--ae-red) 7%, transparent); }
.ae-stat.bad b { color: var(--ae-red); }
.ae-stat.warn b { color: #e08600; }

/* dataframes */
[data-testid="stDataFrame"] { border-radius: 0.5rem; overflow: hidden; }

/* alerts: hairline card with a colored left rule instead of a filled block */
[data-testid="stAlertContainer"] { background: color-mix(in srgb, currentColor 3%, transparent) !important;
  border: 1px solid color-mix(in srgb, currentColor 12%, transparent); border-left-width: 3px; border-radius: 0.5rem; }
[data-testid="stAlertContentError"], [data-testid="stAlertContentWarning"],
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentSuccess"] { color: inherit; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) { border-left-color: var(--ae-red); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) { border-left-color: #e08600; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) { border-left-color: #9aa0a6; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { border-left-color: #2f9e6a; }

/* uploader as an obvious empty state */
[data-testid="stFileUploaderDropzone"] { border: 1.5px dashed color-mix(in srgb, currentColor 32%, transparent);
  background: transparent; padding: 1.6rem 1.25rem; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--ae-red); }

/* closing notes under each tab */
[class*="st-key-ae-note"] { margin-top: 1.25rem; padding: 1rem 1.2rem; border-radius: 0.5rem;
  border: 1px solid color-mix(in srgb, currentColor 12%, transparent); border-left: 3px solid #9aa0a6; }
.st-key-ae-note-boat { border-left-color: var(--ae-red) !important; }
[class*="st-key-ae-note"] p { line-height: 1.65; margin: 0; }

@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; padding-top: 3.25rem; }
  [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1.1rem; }
  .ae-stat { min-width: 0; flex: 1 1 40%; }
}
</style>
"""
st.html(STYLE)


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
    line = alt.Chart(df).mark_line(color="#9aa0a6", strokeWidth=1).encode(x=x, y=y)
    dots = alt.Chart(df).mark_circle(size=140, opacity=1).encode(
        x=x, y=y,
        color=alt.Color("flagged:N", scale=alt.Scale(domain=["normal", "flagged on this metric"],
                                                     range=["#9aa0a6", "#d62728"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["build", alt.Tooltip("value:Q", format=".4g"), "flagged"],
    )
    median = alt.Chart(pd.DataFrame({"median": [df["value"].median()]})).mark_rule(
        strokeDash=[4, 4], color="#9aa0a6").encode(y="median:Q")
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
        st.caption(f"not localized: {b.error}")
        return
    if not b.suspects:
        st.caption(f"None of the {b.total_commits} commit(s) in this range touch code related to "
                   f"{label(c.regressions[0].metric)}. That itself is evidence this is noise, not a firmware bug.")
    else:
        st.caption(f"{len(b.suspects)} of {b.total_commits} commit(s) touch related code:")
        for s in b.suspects:
            st.markdown(f"- `{s.sha}` {s.message} — {', '.join(s.files)}")
    if b.ai:
        icon = "🟢" if "false" in b.ai.label else ("🔴" if "true" in b.ai.label else "🟡")
        st.markdown(f"{icon} **AI read: {b.ai.label}** — {b.ai.reason}")
    else:
        st.caption("Set ANTHROPIC_API_KEY to also get an AI likely-true/false-positive read on this.")


def show_detail(c: Comparison, ordered_runs: list[FlightSummary]) -> None:
    cand = c.candidate
    st.markdown(f"**{cand.label}** ({vehicle_kind(cand.vehicle, cand.mav_type)}), compared against "
                f"{len(c.baseline)} other runs: {', '.join(b.label for b in c.baseline)}")
    for w in c.warnings:
        st.caption(f"note: {w}")
    if not c.flag_count:
        st.success("No regressions: every metric is within the normal range of the other runs.")
        return
    if c.findings:
        st.dataframe(pd.DataFrame([{
            "": {"worse": "🔴 worse", "better": "🟢 better", "changed": "🟠 changed"}[f.verdict],
            "what": label(f.metric),
            "this run": f.value,
            "normal (median)": f.median,
            "change": f"{f.pct:+.1f}%" if f.pct is not None else "",
        } for f in c.findings]), hide_index=True, use_container_width=True)
    if c.regressions:
        show_bisection(c, ordered_runs)
    if c.new_alarms:
        st.error("New warnings/errors never seen before:\n\n" +
                 "\n".join(f"- [{SEVERITY[m.severity]}] {m.text}" for m in c.new_alarms))
    info = [m for m in c.new_messages if not m.alarming]
    if info:
        st.info("New messages (informational):\n\n" + "\n".join(f"- {m.text}" for m in info))
    if c.missing_messages:
        st.info("Missing messages (seen in every other run, absent here):\n\n" +
                "\n".join(f"- {m}" for m in c.missing_messages))
    if cand.source.startswith("http"):
        st.caption(f"Raw log: {cand.source}")


def show_triage(verdict: str, items: list[str]) -> None:
    if verdict == "LIKELY REAL":
        st.error(f"Triage: likely real. Outside the baseline spread on: {', '.join(items)}")
    elif verdict == "POSSIBLE":
        st.warning(f"Triage: possible, not confirmed. Only {items[0]} is outside the spread. "
                   "One item is usually noise; rerun before treating it as real.")
    else:
        st.success("Triage: within noise. The run matches the baseline spread; rerun before treating it as real.")


def show_check(outcome: CheckOutcome, kind: str) -> None:
    if outcome.error:
        st.error(outcome.error)
        return
    c = outcome.comparison
    st.markdown(f"**{c.candidate.label}**, compared against {len(c.baseline)} earlier flight(s)")
    if kind == "step":
        st.info("Information only: step metrics never fail a build.")
    elif c.failed:
        st.error("Regression: outside the normal range of the earlier flights.")
    else:
        st.success("No regressions: within the normal range of the earlier flights.")
    if outcome.triage:
        show_triage(*outcome.triage)
    if c.findings:
        st.dataframe(pd.DataFrame([{
            "": {"worse": "🔴 worse", "better": "🟢 better", "changed": "🟠 changed"}[f.verdict],
            "what": label(f.metric),
            "this run": f.value,
            "normal (median)": f.median,
            "change": f"{f.pct:+.1f}%" if f.pct is not None else "",
        } for f in c.findings]), hide_index=True, use_container_width=True)
    for w in c.warnings:
        st.caption(f"note: {w}")


def results_section(runs: list[FlightSummary], key: str, default_metric: str | None = None) -> None:
    results = scan(runs)
    df = scan_table(results)
    counts = {s: sum(status(c) == s for c in results) for s in ("REGRESSED", "changed", "ok")}
    st.html(
        '<div class="ae-summary">'
        f'<div class="ae-stat"><b>{len(results)}</b><span>runs</span></div>'
        f'<div class="ae-stat{" bad" if counts["REGRESSED"] else ""}"><b>{counts["REGRESSED"]}</b><span>regressed</span></div>'
        f'<div class="ae-stat{" warn" if counts["changed"] else ""}"><b>{counts["changed"]}</b><span>changed</span></div>'
        f'<div class="ae-stat"><b>{counts["ok"]}</b><span>ok</span></div>'
        '</div>'
    )
    st.dataframe(
        df.style.map(lambda v: f"color: {STATUS_COLORS.get(v, 'inherit')}; font-weight: 600", subset=["status"]),
        hide_index=True, use_container_width=True,
    )
    flagged_metrics = [f.metric for c in results for f in c.findings if not f.metric.startswith("event:")]
    metrics = sorted({m for r in runs for m in r.metrics})
    default = default_metric or (flagged_metrics[0] if flagged_metrics else "armed_time_s")
    metric = st.selectbox("Compare one metric across runs", metrics, index=metrics.index(default) if default in metrics else 0,
                          format_func=label, key=f"{key}-metric")
    st.altair_chart(metric_chart(results, metric), use_container_width=True)
    worst = next((c.candidate.label for c in results if c.failed), results[0].candidate.label)
    choice = st.selectbox("Look at one run in detail", [c.candidate.label for c in results],
                          index=[c.candidate.label for c in results].index(worst), key=f"{key}-detail")
    show_detail(next(c for c in results if c.candidate.label == choice), runs)


st.html(
    '<header class="ae-hero">'
    '<h1>autonomy-eval</h1>'
    '<p class="ae-lede">Catches <strong>flight regressions</strong>: compares a test flight\'s log against earlier '
    "flights of the same vehicle and reports what got worse than normal. Existing ArduPilot log tools look at one "
    "flight at a time; this answers <em>did this build fly worse than the last ones did?</em></p>"
    f'<a class="ae-cta" href="{REPO}" target="_blank" rel="noopener">Code on GitHub</a>'
    "</header>"
)

tab_ci, tab_upload, tab_checks, tab_formats = st.tabs(
    ["Real ArduPilot CI result", "Try your own logs", "Stability and estimator checks", "Same flight, two formats"])

with tab_ci:
    st.container(key="ae-intro-ci").markdown(
        "ArduPilot's CI reruns the same test missions on every firmware build and publishes the "
        "telemetry. Each build below is compared against the other seven."
    )
    history = st.container(key="ae-history").radio(
        "History", list(HISTORIES), horizontal=True, label_visibility="collapsed")
    folder = HISTORIES[history]
    results_section(load_history(folder), key=folder,
                    default_metric="armed_time_s" if folder == "ardurover-surface-boat" else None)
    if folder == "ardurover-surface-boat":
        st.container(key="ae-note-boat").markdown(
            "**What happened in build bf080274:** the boat tacked one extra time, which alone explains "
            "the 13 extra seconds and the worse tracking. \"Possible cause\" above (found automatically, "
            "not written by hand) agrees: the one commit it flags only relocates a parameter table, not "
            "control logic, so the likely cause is the simulator making a different tack decision -- a "
            "flaky test, not a firmware bug. Knowing which is the point."
        )
    else:
        st.container(key="ae-note-rover").markdown(
            "No build regressed. This history was held out while the rules were developed; the first "
            "run on it produced two false alarms, which were fixed "
            f"([before and after]({REPO}#honest-results))."
        )

with tab_upload:
    st.container(key="ae-intro-upload").markdown(
        f"Upload at least {MIN_BASELINE + 1} logs of the **same vehicle and mission**: MAVLink telemetry "
        "(`.tlog`), ArduPilot DataFlash (`.bin`), or summaries saved by `autonomy-eval summarize -o` "
        "(`.json`). Logs are grouped by the vehicle their contents say they are, and each run is compared "
        "against the rest. Files are processed in memory for this session and not kept. Upload them "
        "oldest build first: a regression's \"possible cause\" looks for the ArduPilot commits between "
        "the upload right before it and itself."
    )
    files = st.file_uploader("Flight logs", type=["tlog", "bin", "json"], accept_multiple_files=True)
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
        for msg in failed:
            st.warning(f"Could not read {msg}")
        st.session_state["uploaded_runs"] = runs
    runs = st.session_state.get("uploaded_runs", [])
    groups: dict[tuple, list[FlightSummary]] = {}
    for r in runs:
        groups.setdefault((r.vehicle, r.mav_type), []).append(r)
    for (vehicle, mav_type), group in groups.items():
        st.subheader(f"{vehicle_kind(vehicle, mav_type)}: {len(group)} log(s)")
        if len(group) <= MIN_BASELINE:
            st.warning(f"Need more than {MIN_BASELINE} logs of this vehicle to compare them.")
            continue
        results_section(group, key=f"upload-{vehicle}-{mav_type}")

with tab_checks:
    st.markdown(
        "The stability and estimator checks from the command line, on your own flights. They read raw "
        "DataFlash `.bin` logs, not summaries, and each needs a flight made for it: a SystemID chirp for "
        "stability margins, a SITL run that records ground truth for the estimator check, or GUIDED "
        "position steps for the step response."
    )
    kind = st.radio("Check", list(CHECKS), format_func=lambda k: CHECKS[k][0], horizontal=True, key="chk-kind")
    st.caption(CHECKS[kind][1])
    cand = st.file_uploader("New flight (.bin)", type=["bin"], accept_multiple_files=False, key="chk-candidate")
    base = st.file_uploader(f"Earlier flights of the same mission (.bin, at least {MIN_BASELINE})",
                            type=["bin"], accept_multiple_files=True, key="chk-baseline")
    want_triage = st.checkbox("Also say whether a flagged result is likely real or noise", value=True,
                              key="chk-triage")
    if cand is None:
        st.caption("Upload a new flight to start.")
    elif len(base) < MIN_BASELINE:
        st.caption(f"Upload at least {MIN_BASELINE} earlier flights to compare against.")
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
                outcome = CheckOutcome(error=f"Could not read these logs: {e}")
        show_check(outcome, kind)

with tab_formats:
    st.container(key="ae-intro-formats").markdown(
        "ArduPilot records every flight twice: telemetry sent to the ground station (`.tlog`) and the "
        "autopilot's own onboard log (`.bin`). The same two flights, read both ways:"
    )
    st.dataframe(pd.DataFrame([
        {"": "time armed (s)", "boat .bin": 107.3, "boat .tlog": 107.0, "rover .bin": 243.1, "rover .tlog": 243.0},
        {"": "mean cross-track error (m)", "boat .bin": 1.400, "boat .tlog": 1.405, "rover .bin": 0.598, "rover .tlog": 0.600},
        {"": "worst cross-track error (m)", "boat .bin": 4.381, "boat .tlog": 4.381, "rover .bin": 2.959, "rover .tlog": 2.959},
        {"": "lowest battery (V)", "boat .bin": 11.90, "boat .tlog": 11.90, "rover .bin": 12.53, "rover .tlog": 12.53},
        {"": "in-flight mode changes", "boat .bin": 0, "boat .tlog": 0, "rover .bin": 17, "rover .tlog": 17},
    ]), hide_index=True, use_container_width=True)
    st.container(key="ae-note-formats").markdown(
        "Where they disagree, the cause is known: DataFlash stores EKF variances at 0.01 resolution; "
        "the `.bin` catches a half-second arm/disarm that 1 Hz telemetry misses; and DataFlash has error "
        f"records telemetry doesn't carry. [Full output]({REPO}/blob/main/docs/cross-format-check.txt)"
    )
