from types import SimpleNamespace

from autonomy_eval.telemetry import Clock, TextAssembler


def _text(text, id=0, severity=6):
    return SimpleNamespace(id=id, severity=severity, text=text)


def test_clock_is_monotonic_across_a_reboot():
    c = Clock()
    for ms in (0, 60_000, 120_000):
        c.update(ms)
    assert c.update(500) == 120.5          # rebooted: time_boot_ms restarted near zero
    assert c.update(10_000) == 130.0
    assert c.reboots == 1


def test_clock_ignores_jitter_between_message_types():
    c = Clock()
    c.update(10_000)
    c.update(9_800)                        # a message stamped 200 ms earlier, not a reboot
    assert c.reboots == 0


def test_unchunked_text_passes_through():
    a = TextAssembler()
    assert [t.text for t in a.feed(1.0, _text("ArduPilot Ready"))] == ["ArduPilot Ready"]


def test_chunks_are_joined_and_flushed_on_the_next_message():
    a = TextAssembler()
    assert a.feed(1.0, _text("AT-0000.1: Received: TIMESYNC {tc1 : 1034677000, t", id=2)) == []
    assert a.feed(1.0, _text("s1 : 137250}", id=2)) == []
    out = a.feed(1.1, _text("Barometer 1 calibration complete"))
    assert [t.text for t in out] == [
        "AT-0000.1: Received: TIMESYNC {tc1 : 1034677000, ts1 : 137250}",
        "Barometer 1 calibration complete",
    ]


def test_a_new_chunk_id_starts_a_new_message():
    a = TextAssembler()
    a.feed(1.0, _text("first part ", id=3))
    out = a.feed(2.0, _text("second message", id=4))
    assert [t.text for t in out] == ["first part "]
    assert [t.text for t in a.flush()] == ["second message"]
