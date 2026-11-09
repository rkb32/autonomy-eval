"""Fly the same Copter mission N times in fresh ArduPilot SITL containers and collect the DataFlash logs.

Used to build a baseline (identical runs) and to inject faults (--param NAME=VALUE, set before arming), so
the detector can be tested against regressions whose size and cause are known.

  python scripts/sitl_runs.py --runs 8 --out runs/baseline
  python scripts/sitl_runs.py --runs 8 --out runs/vibe --param SIM_VIB_MOT_MAX=30

Needs Docker, a volume with ArduPilot built for SITL (default name: ardupilot-src, built with
`./waf configure --board sitl && ./waf copter`), and pymavlink on the host.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
HOME = "-35.363261,149.165230,584,353"
PORT = 5760


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


def fly(params: list[str], speedup: int, volume: str, image: str, run_dir: Path, name: str, attempts: int = 4,
        script: str = "sitl_fly.py", timeout: int = 600) -> bool:
    # Every attempt gets a fresh SITL: a retry on a container that already ran would not be the same flight.
    for attempt in range(attempts):
        docker("rm", "-f", name, check=False)
        docker("run", "-d", "--name", name, "--user", "root", "-p", f"127.0.0.1:{PORT}:{PORT}",
               "-v", f"{volume}:/ardupilot", "-v", f"{run_dir.resolve()}:/runs", "-w", "/runs", image,
               "/ardupilot/build/sitl/bin/arducopter", "--model", "+", "--speedup", str(speedup),
               "--defaults", "/ardupilot/Tools/autotest/default_params/copter.parm", "--home", HOME, "-I0")
        try:
            time.sleep(20)                             # SITL boot
            cmd = [sys.executable, str(HERE / script), "--timeout", str(timeout)]
            for p in params:
                cmd += ["--param", p]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode == 0:
                return True
            # stderr carries the traceback; printing only stdout hid the real error
            print(f"    attempt {attempt + 1} failed: stdout={done.stdout.strip().splitlines()[-1:]} "
                  f"stderr={done.stderr.strip().splitlines()[-3:]}", flush=True)
        finally:
            docker("rm", "-f", name, check=False)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--param", action="append", default=[], help="NAME=VALUE set before arming (a fault to inject)")
    ap.add_argument("--speedup", type=int, default=5)
    ap.add_argument("--volume", default="ardupilot-src")
    # pinned by digest so a new upstream build cannot silently change the SITL under test
    ap.add_argument("--image", default="ardupilot/ardupilot-dev-base@sha256:d81c02a9d86ea88b297a66910a6a1f077e8c14cd9ad10e1620b44033e8e371bf")
    ap.add_argument("--flight", default="sitl_fly.py", help="flight script in scripts/ (e.g. sitl_step.py)")
    ap.add_argument("--timeout", type=int, default=600, help="wall-clock cap per flight, seconds")
    args = ap.parse_args()
    if Path(args.flight).name != args.flight or not (HERE / args.flight).is_file():
        ap.error(f"--flight must be the name of a script in scripts/, got {args.flight!r}")

    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for i in range(1, args.runs + 1):
        run_dir = args.out / f"work{i:02d}"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True)
        print(f"run {i}/{args.runs}", flush=True)
        ok = fly(args.param, args.speedup, args.volume, args.image, run_dir, f"sitl-runs-{i}", script=args.flight,
                 timeout=args.timeout)
        logs = sorted((run_dir / "logs").glob("*.BIN"), key=lambda p: p.stat().st_size, reverse=True)
        if ok and logs:
            shutil.copy(logs[0], args.out / f"run{i:02d}.BIN")
        else:
            failed += 1
            print(f"  run {i} produced no usable log", flush=True)
        shutil.rmtree(run_dir, ignore_errors=True)
    print(f"done: {args.runs - failed}/{args.runs} logs in {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
