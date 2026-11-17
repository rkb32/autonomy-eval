# SITL gyro vibration (test-only patch)

Stock ArduPilot SITL in this build does not put a sinusoid on the gyro. `SIM_VIB_FREQ_*` is
declared but not applied to the sensor feed, and `SIM_VIB_MOT_*` only changes simulated RPM.

`sitl_vibration_sensor_feed.patch` adds one. It is applied to `libraries/SITL/SIM_Aircraft.cpp`
in `fill_fdm`, on both the plain and the smoothed rate paths. It changes only the sensor feed,
not the physics or the controllers.

- Base: ArduPilot 4.6.2, commit `1ebd4d9`.
- Parameters: `SIM_VIB_FREQ_X` = frequency in Hz, `SIM_VIB_FREQ_Y` = amplitude in rad/s, applied
  to all three axes. Off when either is zero.
- Apply: `git apply scripts/sitl_vibration_sensor_feed.patch` inside the ArduPilot tree, then rebuild
  with `./waf copter -j14`.
- Not for upstream. Verified only in local SITL: an 80 Hz, 0.05 rad/s injection shows a gyro peak
  at 74 to 82 Hz, about 84,500 times the median above 25 Hz.
