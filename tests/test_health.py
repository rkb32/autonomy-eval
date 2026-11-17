"""Timing-health helpers, checked on values where the answer is known by construction."""
import numpy as np

from autonomy_eval.health import gap_ratio, overrun_fraction


def test_a_single_double_length_gap_gives_ratio_two():
    times = list(range(0, 400, 40)) + list(range(440, 800, 40))   # one 80 us gap (360 to 440) among 40 us steps
    assert abs(gap_ratio(times) - 2.0) < 1e-9


def test_a_uniform_clock_gives_ratio_one():
    assert abs(gap_ratio(np.arange(0, 4000, 40)) - 1.0) < 1e-9


def test_too_few_samples_give_no_gap_ratio():
    assert gap_ratio([0, 40]) is None


def test_overrun_fraction_is_overruns_over_loops():
    assert abs(overrun_fraction([100, 100], [5, 15]) - 0.10) < 1e-12


def test_no_loops_give_no_overrun_fraction():
    assert overrun_fraction([], []) is None
