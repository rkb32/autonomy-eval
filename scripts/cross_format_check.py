"""Check that a .bin log and the .tlog of the same flight produce the same summary.

ArduPilot CI records every flight twice: the harness saves MAVLink telemetry for the whole session
in one .tlog, and the autopilot writes its own DataFlash .bin per boot. For each .bin given, this
finds the flight in the .tlog with the same vehicle type and armed duration, summarizes just that
stretch of telemetry, and prints the two summaries side by side.

    python scripts/cross_format_check.py SESSION.tlog A.BIN B.BIN ...
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autonomy_eval.metrics import summarize, summarize_records  # noqa: E402
from autonomy_eval.telemetry import Armed, Reboot, VehicleType, read  # noqa: E402

MARGIN_S = 5.0
MATCH_TOLERANCE_S = 2.0


def tlog_flights(path: str) -> list[tuple[float, float, int | None]]:
    flights, armed, start, mav_type = [], False, 0.0, None
    for r in read(path):
        if isinstance(r, VehicleType):
            mav_type = r.mav_type
        elif isinstance(r, Reboot) and armed:
            flights.append((start, r.t, mav_type))
            armed = False
        elif isinstance(r, Armed):
            if r.armed and not armed:
                start = r.t
            elif armed and not r.armed:
                flights.append((start, r.t, mav_type))
            armed = r.armed
    return flights


def main(tlog: str, bins: list[str]) -> int:
    print(f"reading {tlog} ...", file=sys.stderr)
    flights = tlog_flights(tlog)
    windows = []
    for b in bins:
        s = summarize(b)
        target = s.metrics.get("armed_time_s", 0)
        candidates = [f for f in flights if f[2] == s.mav_type and abs((f[1] - f[0]) - target) <= MATCH_TOLERANCE_S]
        if len(candidates) != 1:
            print(f"{b}: {len(candidates)} matching flights in the tlog (need exactly 1); skipped")
            continue
        windows.append((b, s, candidates[0]))

    print(f"re-reading {tlog} for {len(windows)} flight(s) ...", file=sys.stderr)
    wanted = [(a - MARGIN_S, z + MARGIN_S) for _, _, (a, z, _) in windows]
    buckets: list[list] = [[] for _ in windows]
    for r in read(tlog):
        for i, (a, z) in enumerate(wanted):
            if a <= r.t <= z:
                buckets[i].append(r)

    for (b, bin_s, (a, z, _)), records in zip(windows, buckets):
        tlog_s = summarize_records(records, f"{tlog}@{a:.0f}-{z:.0f}s")
        print(f"\n{Path(b).name}  vs  tlog {a:.1f}-{z:.1f}s")
        print(f"  {'metric':28} {'.bin':>10} {'.tlog':>10}  diff")
        for k in sorted(set(bin_s.metrics) | set(tlog_s.metrics)):
            x, y = bin_s.metrics.get(k), tlog_s.metrics.get(k)
            if x is None or y is None:
                note = "only in one format"
            else:
                rel = abs(x - y) / max(abs(x), abs(y), 1e-9)
                note = "" if rel < 0.01 else f"{100 * rel:.0f}%"
            print(f"  {k:28} {str(x):>10} {str(y):>10}  {note}")
        same = set(bin_s.events) & set(tlog_s.events)
        print(f"  in-flight message kinds: {len(same)} in both, "
              f"{len(set(bin_s.events) - same)} only in .bin, {len(set(tlog_s.events) - same)} only in .tlog")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2:]))
