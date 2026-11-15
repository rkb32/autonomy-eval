"""A run's label is its firmware, unless the app gives it a title because several runs share that firmware."""
from autonomy_eval.metrics import FlightSummary


def test_label_is_the_firmware_when_there_is_no_title():
    assert FlightSummary(source="a.ulg", firmware="1.17.0").label == "1.17.0"


def test_title_overrides_the_label():
    s = FlightSummary(source="a.ulg", firmware="1.17.0")
    s.title = "1.17.0 · a.ulg"
    assert s.label == "1.17.0 · a.ulg"


def test_label_falls_back_to_the_file_stem_without_firmware():
    assert FlightSummary(source="logs/flight-7.ulg").label == "flight-7"
