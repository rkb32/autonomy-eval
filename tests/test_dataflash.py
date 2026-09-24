from pathlib import Path

import pytest

from autonomy_eval.compare import Incomparable, compare
from autonomy_eval.dataflash import RELAYED_RE, _vehicle_type
from autonomy_eval.metrics import load
from autonomy_eval.telemetry import is_dataflash

# A real DataFlash log from ArduPilot CI (autotest.ardupilot.org, Rover-Scripting-00000088.BIN):
# ArduRover build a64bad1a booting under the test harness, never armed.
BOOT = Path(__file__).parent / "data" / "rover-boot-a64bad1a.BIN"


def test_real_bin_is_identified_from_its_own_contents():
    s = load(BOOT)
    assert is_dataflash(BOOT)
    assert (s.vehicle, s.firmware, s.mav_type) == ("ArduRover", "a64bad1a", 10)
    assert s.quality["format"] == "dataflash"


def test_text_relayed_from_the_test_harness_is_dropped():
    s = load(BOOT)                               # 71 of this log's 179 text records are "SRC=250/250:..."
    assert s.preflight and not any(k.startswith("SRC=") for k in [*s.events, *s.preflight])


def test_never_armed_log_has_no_flight():
    s = load(BOOT)
    assert s.metrics["flights"] == 0 and s.metrics["armed_time_s"] == 0


@pytest.mark.parametrize("text, relayed", [
    ("SRC=250/250:AT-0000.1: Arm motors with MAVLink cmd", True),
    ("SRC=255/190:hello from a ground station", True),
    ("Throttle armed", False),
    ("PreArm: SRC=1 is not a prefix here", False),
])
def test_relayed_text_pattern(text, relayed):
    assert bool(RELAYED_RE.match(text)) is relayed


@pytest.mark.parametrize("firmware, frame_class, mav_type", [
    ("ArduRover", 1, 10),     # ground rover
    ("ArduRover", 2, 11),     # boat: same firmware name, different vehicle
    ("ArduCopter", 1, 2),     # quad
    ("ArduCopter", 6, 4),     # traditional helicopter
    ("ArduPlane", None, 1),
    ("ArduCopter", 99, None), # unknown frame: say so rather than guess
])
def test_vehicle_type_from_frame_class(firmware, frame_class, mav_type):
    got = [v.mav_type for v in _vehicle_type(0.0, firmware, frame_class)]
    assert got == ([mav_type] if mav_type is not None else [])


def test_bin_and_tlog_of_different_vehicles_are_still_refused():
    rover = load(BOOT)
    boats = [load(p) for p in sorted((Path(__file__).parents[1] / "examples" / "ardurover-surface-boat").glob("*.json"))]
    with pytest.raises(Incomparable, match="ground rover.*surface boat"):
        compare(rover, boats)
