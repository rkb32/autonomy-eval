"""NEES on synthetic errors whose consistency is known by construction."""
import numpy as np

from autonomy_eval.nees import nees_series


def test_consistent_filter_averages_one():
    rng = np.random.default_rng(0)
    p = np.full(200000, 0.04)                       # filter says sigma = 0.2 m
    e = rng.normal(0.0, np.sqrt(p))                 # errors really have sigma = 0.2 m
    assert abs(np.mean(nees_series(e, p)) - 1.0) < 0.02


def test_overconfident_filter_averages_above_one():
    rng = np.random.default_rng(1)
    p = np.full(200000, 0.04)                       # filter says sigma = 0.2 m
    e = rng.normal(0.0, 2 * np.sqrt(p))             # errors are twice as big: sigma = 0.4 m
    assert abs(np.mean(nees_series(e, p)) - 4.0) < 0.1


def test_bad_variances_are_dropped():
    e = np.array([0.1, 0.1, 0.1])
    p = np.array([0.0, -1.0, np.nan])
    assert len(nees_series(e, p)) == 0
