"""Innovation summary on known values."""
from autonomy_eval.ekf_health import p95_abs


def test_p95_uses_absolute_values():
    values = [-1.0] * 95 + [0.5] * 5          # 95 samples of |1.0|, 5 of |0.5|
    assert abs(p95_abs(values) - 1.0) < 1e-9


def test_empty_or_nonfinite_input_gives_none():
    assert p95_abs([]) is None
    assert p95_abs([float("nan")]) is None
