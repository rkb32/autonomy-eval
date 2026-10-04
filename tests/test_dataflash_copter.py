"""Copter DataFlash behavior, using a fake log so each case is a few records instead of a 10 MB file."""
from autonomy_eval import dataflash
from autonomy_eval.metrics import summarize_records


class Msg:
    def __init__(self, kind, **fields):
        self._kind = kind
        self.__dict__.update(fields)

    def get_type(self):
        return self._kind


class FakeLog:
    def __init__(self, msgs):
        self.msgs, self.i = msgs, 0

    def recv_match(self, type=None):
        kinds = [type] if isinstance(type, str) else type
        while self.i < len(self.msgs):
            m = self.msgs[self.i]
            self.i += 1
            if kinds is None or m.get_type() in kinds:
                return m
        return None

    def rewind(self):
        self.i = 0


def banner(t, vehicle="ArduCopter"):
    return Msg("MSG", TimeUS=t * 1_000_000, Message=f"{vehicle} V4.6.2 (1ebd4d99)", ID=0)


def vibe(t, x=1.0):
    return Msg("VIBE", TimeUS=t * 1_000_000, IMU=0, VibeX=x, VibeY=x, VibeZ=x, Clip=0)


def arm(t, state):
    return Msg("ARM", TimeUS=t * 1_000_000, ArmState=state, ArmChecks=0, Forced=0, Method=0)


def summary(msgs):
    return summarize_records(dataflash._read(FakeLog(msgs)), "x.bin")


def test_log_that_starts_at_arming_counts_the_whole_file_as_flight():
    # Only the disarm is in the file: the arm record came before logging began.
    s = summary([banner(10), vibe(11), vibe(40), arm(50, 0)])
    assert s.metrics["flights"] == 1
    assert s.metrics["armed_time_s"] == 40.0
    assert s.metrics["vibe_p95"] == 1.0


def test_log_with_both_arm_and_disarm_is_unchanged():
    s = summary([banner(0), vibe(1, x=99.0), arm(10, 1), vibe(20), arm(30, 0), vibe(40, x=99.0)])
    assert s.metrics["flights"] == 1
    assert s.metrics["armed_time_s"] == 20.0
    assert s.metrics["vibe_p95"] == 1.0           # the samples outside the armed window are ignored


def test_log_with_no_arm_record_has_no_flight():
    assert summary([banner(0), vibe(1)]).metrics["flights"] == 0


def test_copter_position_error_is_desired_minus_actual_in_the_horizontal_plane():
    s = summary([
        banner(0), arm(1, 1),
        Msg("PSCN", TimeUS=2_000_000, DPN=3.0, PN=0.0), Msg("PSCE", TimeUS=2_000_000, DPE=4.0, PE=0.0),
        arm(5, 0)])
    assert s.metrics["pos_err_mean_m"] == 5.0
    assert s.metrics["pos_err_p95_m"] == 5.0


def test_copter_attitude_error_is_the_worse_of_roll_and_pitch():
    s = summary([
        banner(0), arm(1, 1),
        Msg("ATT", TimeUS=2_000_000, DesRoll=2.0, Roll=0.0, DesPitch=-1.0, Pitch=1.5),
        arm(5, 0)])
    assert s.metrics["att_err_p95_deg"] == 2.5


def test_attitude_error_is_only_for_copter():
    s = summary([
        banner(0, "ArduRover"), arm(1, 1),
        Msg("ATT", TimeUS=2_000_000, DesRoll=2.0, Roll=0.0, DesPitch=0.0, Pitch=0.0),
        arm(5, 0)])
    assert "att_err_p95_deg" not in s.metrics


def test_copter_reached_command_counts_as_a_waypoint():
    texts = [Msg("MSG", TimeUS=2_000_000 + i, Message=f"Reached command #{i}", ID=0) for i in range(1, 4)]
    s = summary([banner(0), arm(1, 1), *texts, arm(10, 0)])
    assert s.metrics["waypoints_reached"] == 3
