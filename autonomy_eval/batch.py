"""Decode IMU batch (sensor-rate) gyro samples from DataFlash ISBH/ISBD messages.

ArduPilot stores each raw gyro sample as int16 = gyro_rad_s * mul, where mul is the ISBH field.
ISBH type 1 is GYRO (IMU_SENSOR_TYPE_GYRO); type 0 is accel. Each block N has one ISBH and
32 ISBD messages (seqno 0..31) of 32 samples each. Gyro and accel blocks alternate in time,
so blocks are kept separate (never joined across the gaps between them).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pymavlink import DFReader
from scipy.signal import welch

GYRO_TYPE = 1
SAMPLES_PER_ISBD = 32
ISBD_PER_BLOCK = 32


def read_batch_messages(path: str | Path, instance: int = 0):
    """Return (headers, packets): gyro block headers keyed by N, and ISBD packets keyed by N then seqno."""
    headers: dict[int, tuple[float, float]] = {}
    packets: dict[int, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    reader = DFReader.DFReader_binary(str(path), zero_time_base=True)
    try:
        while True:
            m = reader.recv_match(type=["ISBH", "ISBD"])
            if m is None:
                break
            if m.get_type() == "ISBH":
                if m.type == GYRO_TYPE and m.instance == instance:
                    headers[m.N] = (float(m.smp_rate), float(m.mul))
            else:
                packets.setdefault(m.N, {})[m.seqno] = (np.asarray(m.x), np.asarray(m.y), np.asarray(m.z))
    finally:
        reader.close()
    return headers, packets


def gyro_blocks(headers, packets):
    """Complete gyro blocks as rad/s arrays. Returns (blocks, skipped) where blocks is a list of (rate_hz, gyro[3, 1024])."""
    blocks, skipped = [], 0
    for n in sorted(headers):
        rate, mul = headers[n]
        seqs = packets.get(n, {})
        if sorted(seqs) != list(range(ISBD_PER_BLOCK)):
            skipped += 1
            continue
        xs = np.concatenate([seqs[s][0] for s in range(ISBD_PER_BLOCK)])
        ys = np.concatenate([seqs[s][1] for s in range(ISBD_PER_BLOCK)])
        zs = np.concatenate([seqs[s][2] for s in range(ISBD_PER_BLOCK)])
        blocks.append((rate, np.vstack([xs, ys, zs]) / mul))
    return blocks, skipped


def batch_gyro(path: str | Path, instance: int = 0):
    """Gyro blocks from a log: (blocks, skipped). See gyro_blocks."""
    headers, packets = read_batch_messages(path, instance)
    return gyro_blocks(headers, packets)


def spectrum_from_blocks(blocks, axis: int = 0, nperseg: int = 256):
    """Welch per block, averaged over blocks, for one gyro axis. Returns (freq_hz, psd, fs_hz)."""
    if not blocks:
        raise ValueError("no complete gyro batch blocks in this log")
    fs = float(np.median([rate for rate, _ in blocks]))
    psds = []
    for _, gyro in blocks:
        f, p = welch(gyro[axis], fs=fs, nperseg=min(nperseg, gyro.shape[1]))
        psds.append(p)
    return f, np.mean(psds, axis=0), fs


def batch_spectrum(path: str | Path, axis: int = 0, instance: int = 0, nperseg: int = 256):
    """Power spectrum of one gyro axis from a log's batch blocks. Returns (freq_hz, psd, fs_hz)."""
    blocks, _ = batch_gyro(path, instance)
    return spectrum_from_blocks(blocks, axis, nperseg)
