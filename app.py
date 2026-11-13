"""Web demo: the CI result, a way to try your own logs, and the stability and estimator checks.

    streamlit run app.py
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from autonomy_eval import bisect as bisect_mod
from autonomy_eval.checks import CHECKS, CheckOutcome, run_check, run_sample
from autonomy_eval.compare import MIN_BASELINE, Comparison, scan, vehicle_kind
from autonomy_eval.metrics import FlightSummary, load
from autonomy_eval.report import LABELS, SEVERITY

EXAMPLES = Path(__file__).parent / "examples"
REPO = "https://github.com/rkb32/autonomy-eval"
HISTORIES = {
    "Sailboat - 8 daily builds": "ardurover-surface-boat",
    "Ground rover - 8 daily builds": "ardurover-ground-rover",
}
STATUS_COLORS = {"REGRESSED": "#d62728", "changed": "#e08600", "ok": "#9aa0a6"}
SAMPLES = [
    ("sysid", "Run stability sample (about 300 MB, first run is slow)"),
    ("nees", "Run estimator sample"),
    ("step", "Run step response sample"),
]

st.set_page_config(page_title="Autonomy-eval", layout="centered")

# Small visual layer: monospace numbers in tables, a quiet loading placeholder, and a keyboard hint.
STYLE = """
<style>
[data-testid="stTable"] table { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.86rem; font-variant-numeric: tabular-nums; }
[data-testid="stTable"] td, [data-testid="stTable"] th { white-space: nowrap; }
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
        st.caption(f"Could not find the cause: {b.error}")
        return
    if not b.suspects:
        st.caption(f"None of the {b.total_commits} commit(s) in this range touch code related to "
                   f"{label(c.regressions[0].metric)}. That suggests noise, not a firmware bug.")
    else:
        st.caption(f"{len(b.suspects)} of {b.total_commits} commit(s) touch related code:")
        for s in b.suspects:
            st.markdown(f"- `{s.sha}` {s.message} — {', '.join(s.files)}")
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
    st.markdown(f"**{cand.label}** ({vehicle_kind(cand.vehicle, cand.mav_type)}), compared against "
                f"{len(c.baseline)} other runs: {', '.join(b.label for b in c.baseline)}")
    for w in c.warnings:
        st.caption(f"note: {w}")
    if not c.flag_count:
        st.success("No regressions: every metric is within the normal range of the other runs.")
        return
    if c.findings:
        st.table(findings_table(c).style.hide(axis="index"))
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


@st.cache_data(show_spinner=False)
def cached_sample(kind: str) -> CheckOutcome:
    """Each visitor shares the result: the slow check runs once per sample, not once per visit."""
    return run_sample(kind)


def regression_text(outcome: CheckOutcome) -> str:
    """One verdict line for a failed check, with the triage folded in so the page never says two things."""
    if outcome.triage:
        verdict, items = outcome.triage
        if verdict == "LIKELY REAL":
            return f"Regression, likely real: outside the normal range on {', '.join(items)}."
        if verdict == "POSSIBLE":
            return (f"Regression, possibly noise: only {items[0]} is outside the normal range. "
                    "Rerun before trusting it.")
        return "Regression, probably noise: the run matches the normal range of the earlier flights."
    return "Regression: outside the normal range of the earlier flights."


def show_check(outcome: CheckOutcome, kind: str) -> None:
    if outcome.error:
        st.error(outcome.error)
        return
    c = outcome.comparison
    st.markdown(f"**{c.candidate.label}**, compared against {len(c.baseline)} earlier flight(s)")
    if kind == "step":
        st.info("Information only: step metrics never fail a build.")
    elif c.failed:
        st.error(regression_text(outcome))
    else:
        st.success("No regressions: within the normal range of the earlier flights.")
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
    with st.expander("Chart and run detail"):
        metric = st.selectbox("Compare one metric across runs", metrics,
                              index=metrics.index(default) if default in metrics else 0,
                              format_func=label, key=f"{key}-metric")
        st.altair_chart(metric_chart(results, metric), width="stretch")
        choice = st.selectbox("Look at one run in detail", labels, index=labels.index(worst), key=f"{key}-detail")
        show_detail(next(c for c in results if c.candidate.label == choice), runs)


def results_section(runs: list[FlightSummary], key: str, default_metric: str | None = None) -> None:
    results = scan(runs)
    df = scan_table(results)
    counts = {s: sum(status(c) == s for c in results) for s in ("REGRESSED", "changed", "ok")}
    worst = next((c.candidate.label for c in results if c.failed), results[0].candidate.label)
    # One verdict line first; the table, chart and per-run detail below it are for checking the verdict.
    if counts["REGRESSED"]:
        st.error(f"{counts['REGRESSED']} of {len(results)} builds regressed. The first one is {worst}.")
    elif counts["changed"]:
        st.warning(f"No regressions. {counts['changed']} of {len(results)} builds changed on at least one metric.")
    else:
        st.success(f"No regressions in {len(results)} builds.")

    only_changed = st.toggle("Only builds that changed", key=f"{key}-only")
    shown = df[df["status"] != "ok"] if only_changed else df
    if shown.empty:
        st.caption("No build changed. Turn the filter off to see all builds.")
    else:
        st.table(shown.style.map(lambda v: f"color: {STATUS_COLORS.get(v, 'inherit')}; font-weight: 600",
                                 subset=["status"]).hide(axis="index"))
    with st.popover("What does normal mean?"):
        st.markdown(
            "Normal is compared against the other runs, metric by metric, using their median. "
            "A metric is flagged when this run moves outside the range those runs show."
        )
    st.download_button("Download builds (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{key}-builds.csv", mime="text/csv", key=f"{key}-csv")
    detail_section(results, runs, key, default_metric, worst)


st.title("Autonomy-eval")
st.markdown(
    "Did this build fly worse than the earlier ones? This compares a flight log against earlier flights of the "
    "same vehicle and flags anything outside the normal range. Most ArduPilot log tools look at one flight at a time."
)
st.link_button("Source on GitHub", REPO)
st.html('<p class="ae-keys">Keys: <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> switch tabs</p>')

tab_ci, tab_upload, tab_checks = st.tabs(["Example result", "Check your logs", "Check one flight"])

with tab_ci:
    st.markdown(
        "ArduPilot's CI reruns the same test missions on every firmware build and publishes the telemetry. "
        "Each build is compared against the other seven."
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
            "the 13 extra seconds and the worse tracking. The \"possible cause\" above (found automatically, "
            "not written by hand) agrees: the one commit it flags only relocates a parameter table, not "
            "control logic, so the likely cause is the simulator making a different tack decision: a "
            "flaky test, not a firmware bug."
        )
    else:
        st.markdown(
            "No build regressed. This history was held out while the rules were developed; the first "
            "run on it produced two false alarms, which were fixed "
            f"([before and after]({REPO}#honest-results))."
        )

with tab_upload:
    st.markdown(f"Upload at least {MIN_BASELINE + 1} logs of the **same vehicle and mission**:")
    st.markdown(
        "- **Formats:** MAVLink telemetry (`.tlog`), ArduPilot DataFlash (`.bin`), or summaries saved by "
        "`autonomy-eval summarize -o` (`.json`).\n"
        "- **Oldest build first.** A regression's possible cause looks at the commits between it and the "
        "build right before it.\n"
        "- **Private:** files are read in memory for this session and not kept."
    )
    files = st.file_uploader("Flight logs", type=["tlog", "bin", "json"], accept_multiple_files=True)
    if not files:
        st.info("No logs yet? The Example result tab (key 1) shows the same comparison on real build history.")
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
        st.caption(f"Read {len(runs)} of {len(files)} file(s)" + (f"; {len(failed)} could not be read." if failed else "."))
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
            st.warning(f"Need more than {MIN_BASELINE} logs of this vehicle to compare them.")
            continue
        results_section(group, key=f"upload-{vehicle}-{mav_type}")

with tab_checks:
    st.markdown(
        "Run the stability and estimator checks from the command line, on your own flights or a sample. They read "
        "raw DataFlash `.bin` logs, and each needs a flight made for it: a SystemID chirp for stability margins, "
        "a SITL run with ground truth for the estimator, or GUIDED position steps for the step response."
    )
    kind = st.segmented_control("Check", list(CHECKS), format_func=lambda k: CHECKS[k][0],
                                default=next(iter(CHECKS)), key="chk-kind") or next(iter(CHECKS))
    st.caption(CHECKS[kind][1])

    st.subheader("Try a sample flight")
    st.caption("SITL flights with a known fault. The first run downloads them once; later runs reuse them.")
    for sample_kind, button_text in SAMPLES:
        if st.button(button_text, key=f"chk-sample-{sample_kind}"):
            wait = ("Downloading about 300 MB, first run only..." if sample_kind == "sysid"
                    else "Downloading the sample flights, first run only...")
            slot = st.empty()
            slot.html(skeleton(wait))
            try:
                st.session_state[f"chk-result-{sample_kind}"] = cached_sample(sample_kind)
                st.toast(f"{CHECKS[sample_kind][0]} sample ready")
            except Exception as e:  # no network, or a bad download, should not take the page down
                st.session_state[f"chk-result-{sample_kind}"] = CheckOutcome(
                    error=f"Could not load the sample flights: {e}")
            slot.empty()
        if f"chk-result-{sample_kind}" in st.session_state:
            st.markdown(f"**Sample: {CHECKS[sample_kind][0]}**")
            show_check(st.session_state[f"chk-result-{sample_kind}"], sample_kind)

    st.subheader("Check your own flights")
    cand = st.file_uploader("New flight (.bin)", type=["bin"], accept_multiple_files=False, key="chk-candidate")
    base = st.file_uploader(f"Earlier flights of the same mission (.bin, at least {MIN_BASELINE})",
                            type=["bin"], accept_multiple_files=True, key="chk-baseline")
    want_triage = st.checkbox("Also say whether a flagged result looks real or like noise", value=True,
                              key="chk-triage")
    if cand is None:
        st.session_state.pop("chk-result-upload", None)
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
