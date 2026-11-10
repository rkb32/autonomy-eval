"""Flight regression detection for autopilot telemetry logs.

  autonomy-eval summarize LOG [-o out.json]        one flight's metrics (save as JSON to reuse as baseline)
  autonomy-eval check NEW --baseline OLD [OLD ...] is NEW worse than the baseline? exit 1 if so (CI gate)
  autonomy-eval scan LOG LOG LOG ...               leave-one-out: which run in a history is the odd one out?

LOG is a MAVLink telemetry log (.tlog), an ArduPilot DataFlash log (.bin), or a summary .json
written by `summarize -o`. The format is detected from the file contents, not the extension.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

from . import bisect
from .compare import MIN_BASELINE, Incomparable, compare, scan, vehicle_kind
from .metrics import FlightSummary, load, summarize
from .report import render, render_bisection, render_scan, to_dict, to_dict_bisection
from .margins import sysid_margins
from .ekf_health import innovation_metrics
from .nees import nees_from_file
from .stepresp import step_metrics


def _load_all(paths: list[str]):
    if len(paths) == 1:
        return [load(paths[0])]
    with ProcessPoolExecutor() as pool:
        return list(pool.map(load, paths))


def cmd_summarize(args) -> int:
    s = load(args.log)
    if args.output:
        with open(args.output, "w") as f:
            f.write(s.to_json())
        print(f"wrote {args.output}  ({s.vehicle} @ {s.label}, {len(s.metrics)} metrics, {len(s.events)} event kinds)")
    else:
        print(s.to_json())
    return 0


def _bisect_check(result, baseline, do_it: bool):
    if not (do_it and result.regressions and baseline):
        return None
    b = bisect.localize(result.regressions[0].metric, baseline[-1].label, result.candidate.label)
    if b is not None and not b.error:
        b.ai = bisect.ai_verdict(b)
    return b


def cmd_check(args) -> int:
    candidate, *baseline = _load_all([args.log, *args.baseline])
    result = compare(candidate, baseline)
    b = _bisect_check(result, baseline, args.bisect)
    if args.json:
        d = to_dict(result)
        if b:
            d["bisection"] = to_dict_bisection(b)
        print(json.dumps(d, indent=1))
    else:
        print(render(result) + (render_bisection(b) if b else ""))
    return 1 if result.failed else 0


def cmd_scan(args) -> int:
    groups: dict[tuple, list] = {}
    for run in _load_all(args.logs):
        groups.setdefault((run.vehicle, run.mav_type), []).append(run)
    all_results, bisections = [], {}
    for (vehicle, mav_type), runs in groups.items():
        if len(runs) <= MIN_BASELINE:
            print(f"skipping {len(runs)} {vehicle_kind(vehicle, mav_type)} run(s): need more than {MIN_BASELINE} to scan", file=sys.stderr)
            continue
        results = scan(runs)
        all_results += results
        if args.bisect:
            for r in results:
                if r.failed:
                    bisections[id(r)] = bisect.for_comparison(r, runs)
        if not args.json:
            print(f"== {vehicle_kind(vehicle, mav_type)}: {len(runs)} runs ==")
            print(render_scan(results))
            for r in results:
                if r.failed:
                    b = bisections.get(id(r))
                    print("\n" + "-" * 60 + "\n" + render(r) + (render_bisection(b) if b else ""))
            print()
    if args.json:
        dicts = []
        for r in all_results:
            d = to_dict(r)
            b = bisections.get(id(r))
            if b:
                d["bisection"] = to_dict_bisection(b)
            dicts.append(d)
        print(json.dumps(dicts, indent=1))
    return 1 if any(r.failed for r in all_results) else 0


def _step_flight(path: str) -> FlightSummary:
    """Only the step-response metrics, so the gate judges steps and nothing else in the flight."""
    s = summarize(path)
    return FlightSummary(source=str(path), vehicle=s.vehicle, firmware=s.firmware, mav_type=s.mav_type,
                         metrics=step_metrics(path))


def cmd_step_check(args) -> int:
    paths = [args.log, *args.baseline]
    with ProcessPoolExecutor() as pool:
        flights = list(pool.map(_step_flight, paths))
    for f in flights:
        if not f.metrics:
            print(f"error: no position-controller steps found in {f.source}", file=sys.stderr)
            return 2
    candidate, *baseline = flights
    result = compare(candidate, baseline)
    if args.json:
        print(json.dumps(to_dict(result), indent=1))
    else:
        print(render(result))
        print("\n(step metrics are information only: this check does not fail the build)")
    return 0    # info only: position-error metrics are the gate (see README)


def _margin_flight(path: str) -> FlightSummary:
    """Only the stability margins of a SystemID chirp, so the gate judges the loop and nothing else.

    If the margins cannot be measured (coherence too low at the crossover, which happens near instability),
    the flight gets margin_unmeasurable = 1, which the gate treats as a regression, not as a pass.
    """
    s = summarize(path)
    m = sysid_margins(path)
    m["margin_unmeasurable"] = 0.0 if ("margin_pm_deg" in m and "margin_gm_db" in m) else 1.0
    return FlightSummary(source=str(path), vehicle=s.vehicle, firmware=s.firmware, mav_type=s.mav_type,
                         metrics=m)


def _missing_in_candidate(candidate: FlightSummary, baseline: list[FlightSummary]) -> list[str]:
    """Metrics every baseline flight has but the candidate lacks: the gate cannot judge those."""
    common = set.intersection(*(set(b.metrics) for b in baseline)) if baseline else set()
    return sorted(common - set(candidate.metrics))


def cmd_sysid_check(args) -> int:
    paths = [args.log, *args.baseline]
    with ProcessPoolExecutor() as pool:
        flights = list(pool.map(_margin_flight, paths))
    if all(f.metrics.get("margin_unmeasurable") == 1.0 for f in flights):
        print("error: no SystemID chirp with usable margins in any of these flights", file=sys.stderr)
        return 2
    candidate, *baseline = flights
    result = compare(candidate, baseline)     # unmeasurable margins are a flagged metric, not an error
    if args.json:
        print(json.dumps(to_dict(result), indent=1))
    else:
        print(render(result))
    return 1 if result.failed else 0


def _nees_flight(path: str) -> FlightSummary:
    """Only the estimator-consistency numbers, so the gate judges the EKF and nothing else."""
    s = summarize(path)
    return FlightSummary(source=str(path), vehicle=s.vehicle, firmware=s.firmware, mav_type=s.mav_type,
                         metrics={**{f"nees_{k}": v for k, v in nees_from_file(path).items()},
                                  **innovation_metrics(path)})


def cmd_nees_check(args) -> int:
    paths = [args.log, *args.baseline]
    with ProcessPoolExecutor() as pool:
        flights = list(pool.map(_nees_flight, paths))
    for f in flights:
        if not f.metrics:
            print(f"error: no in-flight EKF data in {f.source}", file=sys.stderr)
            return 2
    candidate, *baseline = flights
    missing = _missing_in_candidate(candidate, baseline)
    if missing:
        print(f"error: cannot judge {candidate.source}: missing {', '.join(missing)}", file=sys.stderr)
        return 2
    result = compare(candidate, baseline)
    if args.json:
        print(json.dumps(to_dict(result), indent=1))
    else:
        print(render(result))
    return 1 if result.failed else 0


TRIAGE_MIN_ITEMS = 2   # one flagged item is often noise (a single metric crossed its band by chance)


def triage_verdict(result) -> tuple[str, list[str]]:
    """'LIKELY REAL', 'POSSIBLE' or 'WITHIN NOISE', with the items outside the baseline spread."""
    names = sorted({f.metric for f in result.regressions} | {m.text for m in result.new_alarms})
    decisive = {"margin_unmeasurable"} | {m.text for m in result.new_alarms}   # physical signals, never noise
    if len(names) >= TRIAGE_MIN_ITEMS or decisive & set(names):
        return "LIKELY REAL", names
    if names:
        return "POSSIBLE", names
    return "WITHIN NOISE", names


def cmd_triage(args) -> int:
    """Label a failed or suspicious run: likely real (outside the baseline spread) or within noise (rerun it)."""
    builders = {"summary": None, "sysid": _margin_flight, "nees": _nees_flight, "step": _step_flight}
    builder = builders[args.kind]
    if builder is None:
        candidate, *baseline = _load_all([args.log, *args.baseline])
    else:
        with ProcessPoolExecutor() as pool:
            candidate, *baseline = list(pool.map(builder, [args.log, *args.baseline]))
    result = compare(candidate, baseline)
    label, names = triage_verdict(result)
    if label == "LIKELY REAL":
        print(f"LIKELY REAL: {candidate.label} is outside the baseline spread on {len(names)} item(s):")
        for n in names:
            print(f"  - {n}")
        if args.bisect and result.regressions:
            b = bisect.localize(result.regressions[0].metric, baseline[-1].label, candidate.label)
            if b is not None and not b.error:
                print(render_bisection(b))
        return 1
    if label == "POSSIBLE":
        print(f"POSSIBLE, NOT CONFIRMED: only {names[0]} is outside the baseline spread. "
              f"One item is usually noise; rerun before treating it as real.")
        return 0
    print(f"WITHIN NOISE: {candidate.label} matches the baseline spread; rerun before treating it as real.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="autonomy-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summarize", help="metrics for one flight")
    s.add_argument("log")
    s.add_argument("-o", "--output", help="write the summary to this .json file")
    s.set_defaults(fn=cmd_summarize)

    c = sub.add_parser("check", help="compare a new flight against a baseline of earlier ones")
    c.add_argument("log")
    c.add_argument("--baseline", nargs="+", required=True, help="oldest to newest")
    c.add_argument("--json", action="store_true")
    c.add_argument("--bisect", action="store_true",
                    help="for a regression, fetch the GitHub commits between the last baseline build and this "
                         "one that touch related code (needs network; ANTHROPIC_API_KEY adds a likely-TP/FP read)")
    c.set_defaults(fn=cmd_check)

    sc = sub.add_parser("scan", help="find the odd run out in a history of runs")
    sc.add_argument("logs", nargs="+", help="oldest to newest")
    sc.add_argument("--json", action="store_true")
    sc.add_argument("--bisect", action="store_true", help="see `check --bisect`")
    sc.set_defaults(fn=cmd_scan)

    st = sub.add_parser("step-check", help="step-response gate: overshoot and settling of a GUIDED step flight")
    st.add_argument("log")
    st.add_argument("--baseline", nargs="+", required=True, help="step flights of the previous build(s)")
    st.add_argument("--json", action="store_true")
    st.set_defaults(fn=cmd_step_check)

    sy = sub.add_parser("sysid-check", help="stability-margin gate: gain and phase margin of a SystemID chirp")
    sy.add_argument("log")
    sy.add_argument("--baseline", nargs="+", required=True, help="SystemID flights of the previous build(s)")
    sy.add_argument("--json", action="store_true")
    sy.set_defaults(fn=cmd_sysid_check)

    ne = sub.add_parser("nees-check", help="estimator-consistency gate: is the EKF overconfident against truth?")
    ne.add_argument("log")
    ne.add_argument("--baseline", nargs="+", required=True, help="flights of the previous build(s)")
    ne.add_argument("--json", action="store_true")
    ne.set_defaults(fn=cmd_nees_check)

    tr = sub.add_parser("triage", help="label a failed run: likely real, or within noise (rerun)")
    tr.add_argument("log")
    tr.add_argument("--baseline", nargs="+", required=True, help="passing runs of the same test")
    tr.add_argument("--kind", choices=["summary", "sysid", "nees", "step"], default="summary",
                    help="which metrics to triage: flight summary (default), SystemID margins, EKF NEES, or step response")
    tr.add_argument("--bisect", action="store_true", help="for a likely-real result, localize the commit range")
    tr.set_defaults(fn=cmd_triage)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except Incomparable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:                       # a missing optional package or a corrupt log is not a regression
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
