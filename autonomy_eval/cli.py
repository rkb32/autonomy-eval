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
from .metrics import load
from .report import render, render_bisection, render_scan, to_dict, to_dict_bisection


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

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except Incomparable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
