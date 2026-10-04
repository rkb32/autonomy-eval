"""Does the detector catch a regression whose size and cause are known?

Faults are injected into ArduPilot SITL by changing one parameter before the same mission is flown, at
three strengths each. Every faulted run is checked against a baseline of unmodified runs, and the
baseline runs are checked against each other to count false alarms.

  python scripts/sitl_runs.py --runs 8 --out runs/baseline
  python scripts/fault_benchmark.py collect --out runs --per-level 3     # about 75 s per flight
  python scripts/fault_benchmark.py report --out runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# fault name -> (parameter, strengths from mild to severe)
FAULTS: dict[str, tuple[str, list[float]]] = {
    "vibration": ("SIM_VIB_MOT_MAX", [5, 15, 30]),
    "wind": ("SIM_WIND_SPD", [2, 5, 8]),
    "weak-position-gain": ("PSC_POSXY_P", [0.6, 0.35, 0.2]),
    "high-roll-rate-gain": ("ATC_RAT_RLL_P", [0.25, 0.4, 0.6]),
}


def level_dir(out: Path, fault: str, value: float) -> Path:
    return out / f"{fault}_{value:g}"


def cmd_collect(args) -> int:
    import sitl_runs
    import subprocess

    for fault, (param, values) in FAULTS.items():
        for value in values:
            d = level_dir(args.out, fault, value)
            if len(list(d.glob("run*.BIN"))) >= args.per_level:
                continue
            print(f"== {fault}: {param}={value:g}", flush=True)
            subprocess.run([sys.executable, str(HERE / "sitl_runs.py"), "--runs", str(args.per_level),
                            "--out", str(d), "--param", f"{param}={value:g}"], check=False)
    return 0


def cmd_report(args) -> int:
    from autonomy_eval.compare import compare, scan
    from autonomy_eval.metrics import summarize

    baseline_logs = sorted((args.out / "baseline").glob("run*.BIN"))
    baseline = [summarize(p) for p in baseline_logs]
    loo = scan(baseline)
    print(f"false alarms: {sum(r.failed for r in loo)} of {len(loo)} baseline runs flagged when checked against the other {len(baseline) - 1}\n")
    print(f"{'fault':22} {'strength':>9}  {'detected':>9}  metrics that regressed (worst first)")
    for fault, (param, values) in FAULTS.items():
        for value in values:
            runs = [summarize(p) for p in sorted(level_dir(args.out, fault, value).glob("run*.BIN"))]
            if not runs:
                print(f"{fault:22} {value:>9g}  {'no runs':>9}")
                continue
            results = [compare(r, baseline) for r in runs]
            seen: dict[str, int] = {}
            for res in results:
                for f in res.regressions:
                    seen[f.metric] = seen.get(f.metric, 0) + 1
            detected = sum(res.failed for res in results)
            top = ", ".join(f"{m} ({n}/{len(runs)})" for m, n in sorted(seen.items(), key=lambda kv: -kv[1])[:4])
            print(f"{fault:22} {value:>9g}  {detected:>4}/{len(runs):<4}  {top}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--per-level", type=int, default=3)
    c.set_defaults(fn=cmd_collect)
    r = sub.add_parser("report")
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
