# autonomy-eval

**Live demo: https://autonomy-eval.streamlit.app** (no install; try it with your own logs)

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

## Where to look

Naming the bad build isn't the useful part -- pointing at *why* is. Build labels are ArduPilot git
SHAs, so `--bisect` fetches the real commit range from GitHub and keeps only the commits that touch
source directories plausible for the metric that regressed (navigation code for cross-track error,
`AP_InertialSensor` for vibration, and so on), instead of a person reading a compare link by hand,
which is how the paragraph above was originally written:

```
$ autonomy-eval scan examples/*/*.json --bisect
...
POSSIBLE CAUSE (bf080274...b2b1b3d2, https://github.com/ArduPilot/ardupilot/compare/bf080274...b2b1b3d2):
  1 of 15 commit(s) touch related code:
    a4e55aa9  AC_AttitudeControl: place PosControl parameter conversion tables in rodata
      libraries/AC_AttitudeControl/AC_PosControl.cpp
```

That one commit only relocates a parameter table; it doesn't touch control logic, which is the
same conclusion the hand-written paragraph reached. A directory match is a lead, not a verdict --
`AC_PosControl.cpp` is in the right neighborhood but the change itself is a memory-layout change.
That's a real limitation of matching by file path alone, which is why `--bisect` will also ask a
model for a single "likely true positive / likely false positive" read if `ANTHROPIC_API_KEY` is
set, grounded only in the commit messages and files it was just given (it's told explicitly not to
invent ArduPilot internals it wasn't shown). Skipped silently without a key. Both are leads for a
person to check, not a diagnosis: a commit touching the right file is worth reading, not proof.

When a build is the *first* one in the history (as bf080274 is here), there's no earlier build in
the dataset to diff against, so `--bisect` checks the range to the *next* build instead -- the same
substitution the hand-written analysis made.

## Upgrade checks: new firmware vs your last build

The main use: you flashed a new build, flew the same mission, and want to know what changed.
`check` compares the flight, and because `.bin` logs carry the parameter table, it also diffs
parameters against what every baseline run agreed on (a parameter the baseline runs disagree
about — per-boot calibration like `INS_GYROFFS_*` — is skipped, so only real changes are
reported). Metrics follow the vehicle: Copter logs get position and attitude tracking error
(`PSCN`/`PSCE`, `ATT`), Rover and Plane get cross-track error (`NTUN`).

Validated in SITL (ArduPilot's simulator, in Docker, same mission flown repeatedly):

- **4.6.2 → 4.6.3** (patch release): no metric regressions, 0 of 1397 parameters changed.
  "Nothing changed" is the correct answer for a patch release, and that's what it said.
- **4.6.2 → 4.7.1** (minor release): no flight regressions. Under ~345 parameter-table renames
  (4.7 moved params to unit-suffixed names, e.g. `ANGLE_MAX` → `ATC_ANGLE_MAX`), it surfaced the
  3 real default changes: `ATC_INPUT_TC` 0.15 → 0.10, `EK3_SRC_OPTIONS` 1 → 0,
  `LOG_MAV_RATEMAX` 0 → 10.
- **Does a rename silently drop a custom value?** Tested directly: set a custom `ANGLE_MAX` on
  4.6.2, booted 4.7.1 on the same parameter storage. The value carried into `ATC_ANGLE_MAX` with
  the unit conversion applied (2500 centidegrees → 25.0 degrees). ArduPilot's migration tables
  do their job — and the diff makes every rename visible so you can verify your own settings.

## Does it catch a regression whose size and cause are known?

Fault injection in SITL: 8 identical baseline flights, then the same mission flown 3 times per
fault level with one parameter changed. `scripts/sitl_runs.py` + `scripts/fault_benchmark.py`
(~75 s per flight, Docker).

| injected fault | flights | detected |
|---|---|---|
| weak position gain (`PSC_POSXY_P` 0.6 / 0.35 / 0.2) | 9 | **9/9** (position error metrics) |
| wind (`SIM_WIND_SPD` 2 / 5 / 8 m/s) | 9 | 6/9 — 2 m/s sits under the 0.1 m reporting floor, by design |
| high roll-rate gain (`ATC_RAT_RLL_P` 0.25 / 0.4 / 0.6) | 9 | 3/9 — only 0.6; the 0.4 signal (attitude error +57%) sits under absolute floors set for real hardware |
| vibration (`SIM_ACC1_RND` 1.5 / 4 / 10, with `SIM_VIB_FREQ_X/Y/Z` = 100 Hz) | 9 | **9/9** (`VIBE` p95: 0.049 at baseline, 1.10 / 2.92 / 7.30 at the three levels) |
| baseline checked against itself (leave-one-out) | 8 | **0 false alarms** |

The vibration row needs a static vibration frequency. `SIM_ACC1_RND` only scales the vibration sine;
without `SIM_VIB_FREQ_*` it adds nothing, which is why an earlier version of this benchmark showed 0/9.

### Step response, stability margins and estimator consistency (SystemID and GPS-noise flights)

| check | what it measures | result |
|---|---|---|
| step response (`step-check`) | overshoot, settling time, steady-state error of GUIDED position steps | **0/15** detected across `PSC_POSXY_P` 0.2–2.0. Steady-state error is the only metric that moves with gain, and it stays under the 0.1 m floor until gain 0.2. Now **information only**: `step-check` always exits 0. |
| stability margins (`sysid-check`) | gain and phase margin of the roll-rate loop from a SystemID chirp | baseline (4 flights): phase margin 38–41°, gain margin 9.0–9.3 dB. Roll gain 0.4 (3 flights): gain margin about 0 dB, phase margin about −92°. **3/3** flagged, 0 false alarms. |
| estimator consistency (`nees-check`) | EKF normalized estimation error (NEES) against SITL truth, core 0 only | 0 false alarms in leave-one-out. Did **not** detect the altitude-sinusoid fault at 2 m or 4 m (0/3 each): vertical NEES stays at about 0.10 in every flight. Horizontal NEES is noisy (0.1–0.8 at baseline). An earlier version reported detection, but it mixed in a second EKF core and was wrong. |
| innovation health (`nees-check`, same flights) | p95 of the EKF's innovations (measurement minus prediction) | 0 false alarms. Did **not** detect the altitude-sinusoid fault at 2 m or 4 m (0/3 each). |

The step-response gate was built to catch controller-gain changes, and at the floors above it caught none
of the 15 faulted flights. The position-error gate already catches weak position gain (9/9, above), so the
step metrics are reported for information only. Lowering the steady-state floor to 0.05 m would catch gain
0.2; that is one fault level, so the floor was not changed.

## Web demo

```bash
pip install -e ".[demo,sysid]" && streamlit run app.py
```

Four tabs: the CI result above with a chart of each metric across builds, an upload box for your
own `.tlog` / `.bin` logs, the stability and estimator checks (`sysid-check`, `nees-check`, `step-check`,
and triage) on your own `.bin` logs, and the cross-format check. It runs the same code as the CLI.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate        # or: source .venv/bin/activate
pip install -e ".[dev]"

autonomy-eval scan examples/*/*.json                   # reproduce the result above (no download)
autonomy-eval sysid-check NEW.bin --baseline OLD.bin ...   # stability margins of a SystemID chirp (pip install -e ".[sysid]")
autonomy-eval nees-check NEW.bin --baseline OLD.bin ...    # EKF consistency against SITL truth (SIM2 logs)
autonomy-eval step-check NEW.bin --baseline OLD.bin ...    # step response; information only, always exits 0
autonomy-eval triage LOG --baseline PASS.bin ...           # one flagged item = possible; two or more = likely real
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
- It has not yet caught a **confirmed** firmware regression in the wild; detection is validated
  against injected SITL faults of known size (table above), and the one CI anomaly it found traces
  to simulator nondeterminism.
- The minimum-change floors are absolute and hand-set for real hardware. SITL baselines are far
  quieter (vibration floor 1.0 m/s/s vs a SITL baseline of 0.05), so mild simulated faults can sit
  under them — that's why the 0.4 roll-gain level isn't flagged. Whether floors should scale with
  the baseline is an open design question.
- **Motor vibration is not benchmarked.** The accelerometer row (table above) reproduces: with `SIM_VIB_FREQ_X/Y/Z=100`,
  `SIM_ACC1_RND` 1.5, 4 and 10 give vibration p95 1.1, 2.92–2.93 and 7.31–7.32 m/s/s over 3 runs each, matching the 1.10, 2.92 and 7.30 in the table. The v2 batch set
  `SIM_VIB_MOT_MAX=100` without the frequency instead, and gave different values: 7.1–9.3 at `SIM_ACC1_RND=4` (3 runs: 9.32, 9.10, 7.14) and 16.8–23.5
  at `SIM_ACC1_RND=10` (3 runs: 23.52, 20.86, 16.76). A landing-end rerun gave 7.7–9.7 at 4 and 18.1–23.8 at 10. The motor 1.5 level gave 2.93–3.22 over 3 runs. So the motor batch measures a different
  signal, and its real-world fidelity is unverified. The 1.5 flights ended at the landing timeout without disarming. The 4 and 10 batches, old and rerun,
  all logged LAND_COMPLETE and disarmed about 0.5 s later, so their endings match.
  At motor 1.5 with `EK3_IMU_MASK=2` (EKF uses IMU1 only), 3 of 3 runs logged LAND_COMPLETE and disarmed 0.5 s later. The unmasked 1.5 flights (`EK3_IMU_MASK=3`, same sim parameters, 3 runs) never disarmed, and none logged LAND_COMPLETE (one logged LAND_MAYBE). That fits a lane-switch explanation, but the mechanism is not proven. The mask runs' vibration (3.47, 3.11, 3.11) is left out of the table because the mask changes the estimator's input.
- **Stability-margin floors are a guess.** The 5° phase-margin and 1 dB gain-margin floors come from one
  fault level and four baseline flights. They are not validated against other faults.
- **The margin is of the attitude cascade at the SystemID injection point, not of the rate loop alone.** ArduCopter
  adds the chirp to the rate target after the angle controller (`libraries/AC_AttitudeControl/AC_AttitudeControl_Multi.cpp:453`),
  so the angle loop's feedback is inside the measured response. Angle feedback matters most at low frequency, so
  the crossover estimate is closest to the rate loop. A pure rate-loop margin would need the angle loop disabled,
  which this check does not do. The phase is anchored at its lowest-frequency value (the principal angle).
- **Estimator checks did not catch the fault we injected, and that fault was mislabelled.** `SIM_GPS_NOISE` adds a
  vertical sinusoid to the simulated GPS altitude only (`libraries/SITL/SIM_GPS.cpp:367`), not horizontal noise.
  Neither NEES nor innovation health flagged it at 2 m or 4 m. Horizontal GPS velocity error (`SIM_GPS_VERR_X=1`
  and `SIM_GPS_VERR_Y=1`) is flagged by `check` in all 3 runs, and again on the landing-end script (velocity variance p95 0.33 vs 0.02). `nees-check` on run 1 shows velocity NEES 1.85 and 1.32
  vs baseline 0.10 and 0.06, and velocity innovation p95 0.73 vs 0.04. Baseline false-alarm rate, leave-one-out on the 6 landing-end baseline flights (`v6-base`): `ekf_vel_var_p95` is 0.02 in every run, and `check` flags 0 of 6 (all metrics within the normal range). That is 0/6, which bounds the rate only loosely: 95% upper bound about 39%. A single-metric flag from `triage`
  is reported as "possible", not as a real regression.
- **Unmeasurable margins count as a regression.** If the loop has no usable crossover (coherence falls below 0.6
  near instability), `sysid-check` flags `margin_unmeasurable` and fails. Roll gain 0.4 is flagged this way, not
  by a measured negative margin.
- **Step-response gate is information only** (see the table above). Its steady-state floor was not tuned.
- Vehicle type for `.bin` logs is mapped from `FRAME_CLASS` for Rover and Copter frames only.
- `--bisect` only works when build labels are real ArduPilot git SHAs (not e.g. `.json` summaries
  saved under your own filenames), and matches commits by source directory, which can name a file
  in the right neighborhood whose actual change is unrelated (the bf080274 case above) -- worth a
  human's five minutes, not a verdict. Unauthenticated GitHub API calls are rate-limited to 60/hour.
