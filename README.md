# autonomy-eval

Catches flight regressions. Give it the telemetry log from a new test flight plus the logs from
earlier flights of the same vehicle, and it tells you what got worse than normal, in plain
language, with an exit code CI can gate on.

Existing ArduPilot log tools (UAVLogViewer, MAVExplorer, and several ML "anomaly detectors" on
GitHub) look at **one flight at a time**. None of them answer the question a team asks after
every firmware change: *did this build fly worse than the last twenty did?*

## What it found in real data

ArduPilot's CI reruns the same test missions on every firmware build and publishes the telemetry.
Eight consecutive daily builds of one of those missions (a sailboat running two waypoint missions),
scanned leave-one-out, each build against the other seven:

```
$ autonomy-eval scan examples/*/*.json

== ArduRover surface boat: 8 runs ==
  bf080274  REGRESSED   3 flag(s)  mean cross-track error: 1.68 m, baseline 1.2 m  (+40.3%)
  b2b1b3d2  ok          0 flag(s)
  665c0dee  ok          0 flag(s)
  9165d224  ok          0 flag(s)
  368dc0c4  ok          0 flag(s)
  0d38ef16  changed     1 flag(s)  missing message: PreArm: Mode requires mission
  9256cd77  ok          0 flag(s)
  a64bad1a  ok          0 flag(s)

bf080274  (ArduRover, https://autotest.ardupilot.org/ArduCopter-TerrainFailsafe-autotest-1789583517282588.tlog)
REGRESSIONS (worse than every-day variation):
  mean cross-track error: 1.68 m, baseline 1.2 m  (+40.3%)
  time armed (mission time): 257.0 s, baseline 244 s  (+5.3%)
OTHER CHANGES:
  event: Sailboat: Tacking: 8, baseline 7  (+14.3%)
```

One build out of eight tacked one extra time, which alone explains the 13 extra seconds and the
worse tracking. Seven builds flagged nothing.

**Was it a firmware bug?** Probably not, and that is worth knowing too. The
[commits between bf080274 and the next build](https://github.com/ArduPilot/ardupilot/compare/bf080274...b2b1b3d2)
are build tooling and parameter-table placement, with no Rover or sailboat code. The likelier cause
is the simulator making a different tack decision on one run. The tool found a run that genuinely
behaved differently; the diff says to treat it as a flaky test, not a regression.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate        # or: source .venv/bin/activate
pip install -e ".[dev]"

autonomy-eval scan examples/*/*.json                   # reproduce the result above (no download)
autonomy-eval check NEW.tlog --baseline OLD1.tlog OLD2.tlog OLD3.tlog ...
autonomy-eval summarize flight.tlog -o flight.json     # save a summary to reuse as baseline
```

`check` exits `0` if clean, `1` on a regression (for CI), `2` if the logs can't be compared.
Logs can be MAVLink telemetry (`.tlog`, what a ground station saves), ArduPilot DataFlash (`.bin`,
what the autopilot writes to its SD card), or saved `.json` summaries so CI doesn't re-parse
history. The format is detected from the file's contents, not its name.

To pull fresh runs from ArduPilot CI: `python scripts/fetch_autotest.py ArduCopter-TerrainFailsafe`.

## Six things the real data got wrong that a demo wouldn't have

Each of these broke a naive first version.

1. **The filename lies.** The logs are named `ArduCopter-TerrainFailsafe`, but the telemetry says
   they're a Rover sailboat and a Rover ground vehicle. The tool identifies the vehicle from its own
   heartbeat, and `scan` sorts a folder of mixed logs by what's actually in them.
2. **Same name, different vehicle.** Both are `ArduRover` firmware, but one is MAV_TYPE 11 (boat)
   and one is MAV_TYPE 10 (ground rover). Comparing by name would silently compare a boat to a car;
   `check` refuses.
3. **Someone else is talking.** The CI harness writes into the same log (system 250). Its long
   messages get split into 50-character chunks, and in a single log 777 of those chunks don't start
   with the harness's `AT-` prefix, so filtering by prefix leaks harness chatter into the vehicle's data.
   Filtering by MAVLink source system, and reassembling chunks by their `id`, fixes it.
4. **Time runs backwards.** The simulator reboots mid-test and `time_boot_ms` restarts at zero.
   Durations are computed on a clock stitched across reboots.
5. **The index lists files that are gone.** ArduPilot's log server keeps deleted logs in its
   listing (my first download was an HTML 404 page saved as `.bin`). That's also why the summaries
   are committed in `examples/`: the raw logs will disappear, the result shouldn't.
6. **The harness is in the autopilot's own log, too.** ArduPilot writes text it *receives* into its
   `.bin` as `SRC=250/250:...`. In one rover log, 63 of 71 "in-flight message kinds" were the
   harness talking. DataFlash has no per-record source ID, but that prefix names the sender, so
   the same rule applies: only the vehicle's own messages count.

## Same flight, two formats

ArduPilot CI records every flight twice: telemetry for the whole session in one `.tlog`, and the
autopilot's own `.bin` per boot. `scripts/cross_format_check.py` finds a `.bin` flight inside the
session `.tlog` and summarizes both. Two flights from build a64bad1a
([full output](docs/cross-format-check.txt)):

| metric | boat `.bin` | boat `.tlog` | rover `.bin` | rover `.tlog` |
|---|---|---|---|---|
| time armed | 107.3 s | 107.0 s | 243.1 s | 243.0 s |
| mean cross-track error | 1.400 m | 1.405 m | 0.598 m | 0.600 m |
| worst cross-track error | 4.381 m | 4.381 m | 2.959 m | 2.959 m |
| lowest battery | 11.90 V | 11.90 V | 12.53 V | 12.53 V |
| in-flight mode changes | 0 | 0 | 17 | 17 |
| in-flight message kinds | 2 | 2 (same 2) | 8 | 5 (all 5 shared) |

Where they disagree, I traced why before calling it validated:

- **EKF variances** differ by up to 0.01. DataFlash stores them as int16 × 0.01, so a true 0.019
  is logged as 0.01 or 0.02. Below the 0.05 reporting threshold.
- **Rover: 2 flights in `.bin`, 1 in `.tlog`.** The `.bin` records an arm at 62.72 s and a disarm
  at 63.26 s. Telemetry heartbeats are 1 Hz and never see a half-second arm. `.bin` is right.
- **Rover: 9 errors in `.bin`, 0 in `.tlog`.** DataFlash has `ERR` records (here, fence
  failsafes) that have no telemetry equivalent. That accounts for one of the three message kinds
  only in `.bin`; I haven't traced the other two.

`check` warns when a comparison mixes formats, since these differences are expected.

## How it decides something is a regression

- **Median and MAD, not mean and stdev.** CI history always contains a bad run or two. With a
  mean, one outlier widens the "normal" range enough to hide the next real regression; there's a
  test that fails if you switch back.
- **IQR as well as MAD.** Simulated runs are so repeatable that more than half often share one
  exact value, and MAD collapses to zero. Some metrics are bimodal across builds (cross-track p95
  lands on 5.30 m or 5.64 m); IQR sees the second cluster so it isn't flagged.
- **Statistically unusual isn't the same as mattering.** Each metric also has a minimum change
  worth a human's time (vibration: 1 m/s/s, when ArduPilot calls anything under 30 acceptable).
- **Count only what happens in flight.** Boot banners and calibration messages repeat a varying
  number of times depending on when recording started. Pre-flight messages count by presence only.
- **Fail CI only on things that are worse or alarming.** A new warning/error or a worse metric
  fails; a message that disappears, or a failed pre-arm check that retried and passed, is reported.

## Honest results

| data | runs | regressions flagged | false regressions |
|---|---|---|---|
| sailboat (the rules were developed on this) | 8 | 1 (bf080274) | 0 |
| ground rover, first run, **before** looking at it | 8 | 2 | **2** |
| ground rover, after the two fixes below | 8 | 0 | 0 |

The first run on data the tool hadn't seen produced two false alarms, recorded in
[docs/heldout-v1-before-fixes.txt](docs/heldout-v1-before-fixes.txt): a failed pre-arm check that
retried and passed was treated as a critical error, and a 4.6% vibration change that was
statistically unusual but physically meaningless failed the build. Both are fixed, and both fixes
are general rules rather than special cases, but after fixing them the rover set is no longer
unseen data. The real test is the next week of CI builds, which didn't exist when these rules were
written.

## Known gaps

- `.bin` support is validated on two flights recorded in both formats, not on a history of builds:
  the CI server overwrites `.bin` logs daily, so a `.bin` baseline has to be collected over days.
- DataFlash text has no severity, so in `.bin` logs errors come only from `ERR` records; a new
  warning that exists only as text won't fail the build.
- It has not yet caught a **confirmed** firmware regression; the one anomaly it found traces to
  simulator nondeterminism. The strongest next step is to run it on the builds around a known
  ArduPilot bug fix.
- Metric directions ("higher is worse") and the minimum-change thresholds are hand-set for
  ground and surface vehicles; copters and planes would want their own (e.g. altitude tracking).
  Vehicle type for `.bin` logs is mapped from `FRAME_CLASS` for Rover and Copter frames only.
