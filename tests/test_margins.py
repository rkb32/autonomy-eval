"""Stability margins from a known open-loop transfer function, where the answer is known analytically."""
import math

import numpy as np

from autonomy_eval.margins import margins_from_open_loop, unmeasured_margin_bound

F = np.logspace(-1, 1.5, 6000)


def test_integrator_has_90_degrees_phase_margin_at_1hz_and_no_gain_margin():
    L = 1.0 / (1j * F)                       # |L| = 1 at 1 Hz, phase -90 everywhere
    gm, pm, fc = margins_from_open_loop(F, L)
    assert gm is None
    assert abs(fc - 1.0) < 0.01
    assert abs(pm - 90.0) < 0.2


def test_third_order_loop_margins_match_the_closed_form():
    # L = K / (1 + j f)^3, K = 4. Phase crosses -180 at f = sqrt(3) where |L| = K / 8,
    # so GM = 20 log10(8 / 4) = 6.02 dB. |L| = 1 where (1 + x^2)^1.5 = K, x = sqrt(K^(2/3) - 1).
    K = 4.0
    L = K / (1 + 1j * F) ** 3
    gm, pm, fc = margins_from_open_loop(F, L)
    x_c = math.sqrt(K ** (2 / 3) - 1)
    expected_pm = 180 - 3 * math.degrees(math.atan(x_c))
    assert abs(gm - 20 * math.log10(8 / K)) < 0.05
    assert abs(pm - expected_pm) < 0.2
    assert abs(fc - x_c) < 0.01


def test_triple_integrator_reads_its_raw_phase_convention():
    # L = 1 / (j f)^3 has |L| = 1 at 1 Hz. Its phase is +90 at the lowest frequency (the principal
    # value), so the raw convention gives 180 + 90 = 270. The margin is only as good as that anchor.
    L = 1.0 / (1j * F) ** 3
    gm, pm, fc = margins_from_open_loop(F, L)
    assert abs(fc - 1.0) < 0.01
    assert abs(pm - 270.0) < 0.5


def test_unstable_third_order_loop_has_negative_phase_margin():
    # K = 1000 pushes the crossover to x = sqrt(K^(2/3) - 1) where the lag is -3 atan(x), past -180.
    K = 1000.0
    L = K / (1 + 1j * F) ** 3
    gm, pm, fc = margins_from_open_loop(F, L)
    x_c = math.sqrt(K ** (2 / 3) - 1)
    assert abs(pm - (180 - 3 * math.degrees(math.atan(x_c)))) < 0.5
    assert pm < 0


def test_gaps_are_not_bridged_by_crossings():
    # A downward crossing of |L| = 1 that spans a NaN gap must not be used.
    L = 4.0 / (1 + 1j * F) ** 3
    L[2000:2500] = np.nan
    gm, pm, fc = margins_from_open_loop(F, L)
    assert gm is not None or pm is not None  # the loop still has its crossings outside the gap
    assert fc is None or not (F[2000] <= fc <= F[2500])


def test_low_gain_loop_has_no_crossover():
    L = 0.2 / (1 + 1j * F) ** 3              # never reaches |L| = 1
    gm, pm, fc = margins_from_open_loop(F, L)
    assert pm is None and fc is None
    assert gm is not None                    # phase still crosses -180


def test_gap_while_gain_is_above_one_gives_a_phase_margin_bound():
    # Measured only below 5 Hz, where |L| = 7.5 and the phase is about -236 deg: the crossover is in the gap.
    L = 1000.0 / (1 + 1j * F) ** 3
    L[F > 5.0] = np.nan
    f_last, gain, pm_bound = unmeasured_margin_bound(F, L)
    assert abs(f_last - 5.0) < 0.05
    assert gain > 1.0
    assert abs(pm_bound - (180 - 3 * math.degrees(math.atan(f_last)))) < 0.5


def test_gap_with_gain_already_below_one_gives_no_bound():
    L = 0.2 / (1 + 1j * F) ** 3
    L[F > 5.0] = np.nan
    assert unmeasured_margin_bound(F, L) is None


def test_no_gap_gives_no_bound():
    L = 1000.0 / (1 + 1j * F) ** 3           # measured everywhere, so there is nothing to bound
    assert unmeasured_margin_bound(F, L) is None
