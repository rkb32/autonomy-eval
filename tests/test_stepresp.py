"""Step-response analysis on synthetic responses whose overshoot, settling time and error are known."""
import math

from autonomy_eval.stepresp import analyse_axis

DT = 0.1
T = [i * DT for i in range(300)]          # 30 s at 10 Hz


def make_ref(changes):
    """Desired position: starts at 0, then takes each (index, value) in turn."""
    jump = dict(changes)
    ref, cur = [], 0.0
    for i in range(len(T)):
        cur = jump.get(i, cur)
        ref.append(cur)
    return ref


def test_first_order_step_settles_without_overshoot():
    ref = make_ref([(5, 10.0)])
    tk = T[5]
    act = [0.0 if x < tk else 10.0 * (1 - math.exp(-(x - tk))) for x in T]
    (r,) = analyse_axis("N", T, ref, act)
    assert r.overshoot_pct < 0.5
    assert r.settled
    # settles into the 5% band (0.5 m of a 10 m step) at t = ln(20) = 3.0 s
    assert abs(r.settle_s - math.log(20)) < 0.2
    assert r.ss_err_m < 0.05
    assert abs(r.size_m - 10.0) < 1e-9


def test_overshoot_is_percent_of_step_size():
    ref = make_ref([(5, 10.0)])
    tk = T[5]

    def act_at(x):
        if x < tk:
            return 0.0
        y = x - tk
        return 10.0 + math.exp(-4 * (y - 1.0) ** 2)     # peaks at 11 m, one second in

    (r,) = analyse_axis("N", T, ref, [act_at(x) for x in T])
    assert abs(r.overshoot_pct - 10.0) < 0.5


def test_down_step_undershoot_counts_as_overshoot():
    ref = make_ref([(5, 10.0), (100, 0.0)])
    act = []
    for x in T:
        if x < T[5]:
            act.append(0.0)
        elif x < T[100]:
            act.append(10.0 * (1 - math.exp(-(x - T[5]))))
        else:
            y = x - T[100]
            act.append(-math.exp(-4 * (y - 1.0) ** 2))   # goes to -1 m, one second into the step down
    up, down = analyse_axis("E", T, ref, act)
    assert abs(up.size_m - 10.0) < 1e-9 and up.overshoot_pct < 0.5
    assert abs(down.size_m - 10.0) < 1e-9
    assert abs(down.overshoot_pct - 10.0) < 0.5


def test_oscillation_that_never_settles_is_flagged():
    ref = make_ref([(5, 10.0)])
    act = [0.0 if x < T[5] else 10.0 + 2.0 * math.sin(3 * (x - T[5])) for x in T]
    (r,) = analyse_axis("E", T, ref, act)
    assert not r.settled
    assert r.settle_s > 20


def test_no_reference_change_means_no_steps():
    assert analyse_axis("N", T, [0.0] * len(T), [0.0] * len(T)) == []


def test_change_below_step_threshold_is_ignored():
    ref = make_ref([(5, 0.5)])
    act = [0.0 if x < T[5] else 0.5 for x in T]
    assert analyse_axis("N", T, ref, act) == []


def test_slow_ramp_is_still_a_step():
    # 3 m at 0.3 m/s takes 10 s: a slow ramp, below the 0.5 m/s the old per-sample threshold caught
    ref = [min(3.0, max(0.0, 0.3 * (x - T[5]))) for x in T]
    results = analyse_axis("N", T, ref, list(ref))
    assert len(results) == 1
    assert abs(results[0].size_m - 3.0) < 0.2
    assert results[0].overshoot_pct < 1.0


def test_s_curve_start_is_walked_back_to_rest():
    # velocity starts at zero, so the first moving sample is late; the step still starts at the rest point
    # an 8 s S-curve, so the motion ends well before the 25 s hold that the steady-state window uses
    ref = [0.0 if x < T[5] else 10.0 * (1 - math.cos(math.pi * min(1.0, (x - T[5]) / 8.0))) / 2 for x in T]
    (r,) = analyse_axis("N", T, ref, list(ref))
    assert abs(r.size_m - 10.0) < 0.2
    assert abs(r.start_s - T[5]) < 0.3
