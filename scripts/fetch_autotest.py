"""Download a series of ArduPilot CI (autotest) telemetry logs and save their summaries as JSON.

ArduPilot's CI reruns the same test missions on every build and publishes the telemetry at
autotest.ardupilot.org, but only keeps the last several days. Summaries are small enough to commit,
so a baseline outlives the logs it came from.

    python scripts/fetch_autotest.py ArduCopter-TerrainFailsafe --max-mb 20 --out examples

Summaries land in OUT/<vehicle-kind>/, grouped by what the telemetry says the vehicle is. Raw logs
are cached in --cache and not re-downloaded.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autonomy_eval.compare import vehicle_kind  # noqa: E402
from autonomy_eval.metrics import summarize  # noqa: E402

SERVER = "https://autotest.ardupilot.org/"


def list_logs(prefix: str) -> list[str]:
    html = urllib.request.urlopen(SERVER, timeout=30).read().decode()
    return sorted(set(re.findall(rf'href="({re.escape(prefix)}-autotest-\d+\.tlog)"', html)))


def size_mb(name: str) -> float:
    """0 when the file is gone: the server's index keeps listing logs it has already deleted."""
    req = urllib.request.Request(SERVER + name, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length") or 0) / 1e6
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return 0
        raise


def fetch(name: str, cache: Path) -> Path:
    path = cache / name
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(SERVER + name, tmp)
        tmp.rename(path)
    return path


def export(job: tuple[Path, Path]) -> str:
    path, out = job
    s = summarize(path)
    s.source = SERVER + path.name
    stamp = re.search(r"-(\d+)\.tlog$", path.name).group(1)
    folder = out / vehicle_kind(s.vehicle, s.mav_type).lower().replace(" ", "-")
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{stamp}-{s.label}.json"
    dest.write_text(s.to_json() + "\n")
    return str(dest)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("test", help="autotest name prefix, e.g. ArduCopter-TerrainFailsafe")
    p.add_argument("--max-mb", type=float, default=20, help="skip logs larger than this")
    p.add_argument("--out", type=Path, default=Path("examples"))
    p.add_argument("--cache", type=Path, default=Path("data/autotest"))
    args = p.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in list_logs(args.test):
        if (args.cache / name).exists():
            paths.append(args.cache / name)
            continue
        mb = size_mb(name)
        if not 0 < mb <= args.max_mb:
            print(f"skip {name} ({mb:.0f} MB)")
            continue
        print(f"download {name} ({mb:.1f} MB)")
        paths.append(fetch(name, args.cache))

    with ProcessPoolExecutor() as pool:
        for dest in pool.map(export, [(path, args.out) for path in paths]):
            print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
