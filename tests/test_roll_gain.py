"""Roll-rate tracking summary and parameter lookup. Sample-dependent tests skip when no sample flight is present."""
import os
from pathlib import Path

import pytest

from autonomy_eval import flightdata

SAMPLE = Path(os.environ.get("TEMP", "")) / "autonomy-eval-samples" / "sysid-candidate-rollgain.bin"
needs_sample = pytest.mark.skipif(not SAMPLE.exists(), reason="no sample flight in the temp cache")


@needs_sample
def test_missing_parameter_returns_none():
    assert flightdata.param_value(SAMPLE, "NO_SUCH_PARAM_XYZ") is None


@needs_sample
def test_roll_tracking_summary_reports_rms_gain_and_samples():
    s = flightdata.roll_tracking_summary(SAMPLE)
    assert s["rms"] >= 0
    assert s["samples"] > 0
    assert s["gain"] is None or s["gain"] > 0
