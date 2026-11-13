"""Readers for the web views. The log-reading tests use a sample flight when one is present in the temp cache."""
import os
from pathlib import Path

import numpy as np
import pytest

from autonomy_eval import flightdata

SAMPLE = Path(os.environ.get("TEMP", "")) / "autonomy-eval-samples" / "sysid-candidate-rollgain.bin"
needs_sample = pytest.mark.skipif(not SAMPLE.exists(), reason="no sample flight in the temp cache")


def test_known_copter_mode_numbers_get_names():
    assert flightdata.COPTER_MODES[4] == "GUIDED"
    assert flightdata.COPTER_MODES[6] == "RTL"


@needs_sample
def test_derived_signal_refuses_calls_other_than_abs_and_sqrt():
    with pytest.raises(ValueError):
        flightdata.derived_signal(SAMPLE, "BAT", "__import__('os')")


@needs_sample
def test_derived_signal_refuses_unknown_fields():
    with pytest.raises(ValueError):
        flightdata.derived_signal(SAMPLE, "BAT", "NoSuchField * 2")


@needs_sample
def test_derived_signal_multiplies_fields():
    t, v = flightdata.derived_signal(SAMPLE, "BAT", "Volt * 2")
    assert t.size > 0
    assert np.all(np.isfinite(v))
    assert 20 < float(np.median(v)) < 30     # about 12.6 V doubled


@needs_sample
def test_spectrum_stops_at_half_the_sample_rate():
    f, p, fs = flightdata.gyro_spectrum(SAMPLE)
    assert abs(f[-1] - fs / 2) < fs / 100


@needs_sample
def test_fault_checks_return_five_named_rows_with_a_status():
    rows = flightdata.fault_checks(SAMPLE)
    assert len(rows) >= 5
    assert all(r["status"] in {"found", "clear", "not in log"} for r in rows)
