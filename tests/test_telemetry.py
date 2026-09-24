from autonomy_eval.telemetry import Clock, TextAssembler, is_dataflash


def test_clock_is_monotonic_across_a_reboot():
    c = Clock()
    for s in (0, 60, 120):
        assert not c.update(s)
    assert c.update(0.5)                   # rebooted: boot time restarted near zero
    assert c.now == 120.5
    c.update(10)
    assert c.now == 130.0 and c.reboots == 1


def test_clock_ignores_jitter_between_message_types():
    c = Clock()
    c.update(10.0)
    assert not c.update(9.8)               # a message stamped 200 ms earlier, not a reboot


def test_unchunked_text_passes_through():
    a = TextAssembler()
    assert [t.text for t in a.feed(1.0, 0, 6, "ArduPilot Ready")] == ["ArduPilot Ready"]


def test_chunks_are_joined_and_flushed_on_the_next_message():
    a = TextAssembler()
    assert a.feed(1.0, 2, 4, "AT-0000.1: Received: TIMESYNC {tc1 : 1034677000, t") == []
    assert a.feed(1.0, 2, 4, "s1 : 137250}") == []
    out = a.feed(1.1, 0, 6, "Barometer 1 calibration complete")
    assert [(t.text, t.severity) for t in out] == [
        ("AT-0000.1: Received: TIMESYNC {tc1 : 1034677000, ts1 : 137250}", 4),
        ("Barometer 1 calibration complete", 6),
    ]


def test_a_new_chunk_id_starts_a_new_message():
    a = TextAssembler()
    a.feed(1.0, 3, None, "first part ")
    out = a.feed(2.0, 4, None, "second message")
    assert [t.text for t in out] == ["first part "]
    assert [t.text for t in a.flush()] == ["second message"]


def test_format_is_detected_from_content_not_extension(tmp_path):
    fake_bin = tmp_path / "actually-a-tlog.bin"
    fake_bin.write_bytes(b"\x00\x05\xa8\x1c\x82\x9c\x5e\x00\xfd\x09")   # tlog timestamp + MAVLink2 magic
    assert not is_dataflash(fake_bin)
    real = tmp_path / "flight.tlog"
    real.write_bytes(b"\xa3\x95\x80\x80\x59FMT")
    assert is_dataflash(real)
