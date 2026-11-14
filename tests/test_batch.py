"""Batch (sensor-rate) gyro decoding on synthetic ISBH/ISBD-shaped data."""
import numpy as np

from autonomy_eval import batch

MUL = 938.0   # INT16_MAX / radians(2000), as logged in ISBH for the default gyro range
FS = 1000.0


def make_block(n, values, mul=MUL):
    """values: gyro rad/s, shape (3, 1024). Returns headers and packets for block n."""
    raw = np.round(values * mul).astype(np.int16)
    packets = {}
    for seq in range(batch.ISBD_PER_BLOCK):
        s = seq * batch.SAMPLES_PER_ISBD
        chunk = raw[:, s:s + batch.SAMPLES_PER_ISBD]
        packets[seq] = (chunk[0], chunk[1], chunk[2])
    return {n: (FS, mul)}, {n: packets}


def sine_values(hz=250.0, amp=1.0, n_samples=1024):
    # 250 Hz is 4 samples per cycle at 1 kHz, so the sine hits its peak exactly on the grid
    t = np.arange(n_samples) / FS
    x = amp * np.sin(2 * np.pi * hz * t)
    return np.vstack([x, np.zeros_like(x), np.zeros_like(x)])


def test_gyro_scale_is_raw_divided_by_mul():
    headers, packets = make_block(0, sine_values())
    blocks, skipped = batch.gyro_blocks(headers, packets)
    assert skipped == 0
    assert len(blocks) == 1
    rate, gyro = blocks[0]
    assert rate == FS
    assert gyro.shape == (3, 1024)
    # one LSB at this scale is 1/938 rad/s, so the decoded peak matches to within about that
    assert abs(np.max(gyro[0]) - 1.0) < 2 / MUL


def test_incomplete_block_is_skipped_and_counted():
    headers, packets = make_block(0, sine_values())
    del packets[0][5]
    blocks, skipped = batch.gyro_blocks(headers, packets)
    assert blocks == []
    assert skipped == 1


def test_spectrum_finds_an_injected_80_hz_tone():
    # four blocks of an 80 Hz gyro tone (rad/s) on x; the peak must land within one bin of 80 Hz
    blocks = []
    for n in range(4):
        h, p = make_block(n, sine_values(hz=80.0, amp=0.5))
        b, _ = batch.gyro_blocks(h, p)
        blocks.extend(b)
    f, psd, fs = batch.spectrum_from_blocks(blocks, axis=0)
    assert fs == FS
    assert abs(f[np.argmax(psd)] - 80.0) < (f[1] - f[0])


def test_blocks_are_kept_separate_in_order():
    h0, p0 = make_block(0, sine_values())
    h1, p1 = make_block(1, sine_values(hz=125.0))
    blocks, skipped = batch.gyro_blocks({**h0, **h1}, {**p0, **p1})
    assert skipped == 0
    assert [g.shape for _, g in blocks] == [(3, 1024), (3, 1024)]
    assert abs(np.max(blocks[1][1][0]) - 1.0) < 2 / MUL
