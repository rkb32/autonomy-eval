"""Stability margins of the roll-rate loop from a SystemID chirp (scripts/sitl_sysid.py flights).

ArduCopter's SystemID mode (ArduCopter/mode_systemid.cpp) excites the rate loop with a chirp and logs
the target and measured roll rate as SIDD records (Targ, Gx). From those:

  T(f) = Pxy(f) / Pxx(f)        closed-loop response, a cross-spectral estimate (Welch)
  L(f) = T / (1 - T)            open-loop transfer function of the loop
  gain margin  = -20 log10 |L| at the frequency where the phase of L crosses -180 degrees
  phase margin = 180 + phase(L) at the frequency where |L| crosses 1

Only frequencies where the coherence is at least COH_MIN are used, so noise-dominated bins do not
create fake crossings.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

COH_MIN = 0.6
F_MIN_HZ, F_MAX_HZ = 0.5, 40.0      # the chirp's SID_F_START_HZ and SID_F_STOP_HZ defaults
NPERSEG_S = 4.0                     # Welch segment length in seconds


def read_sidd(path: str | Path):
    """(time in s, target roll rate, measured roll rate) from SIDD records."""
    from pymavlink import DFReader

    t, targ, gyro = [], [], []
    log = DFReader.DFReader_binary(str(path), zero_time_base=True)
    try:
        while True:
            m = log.recv_match(type="SIDD")
            if m is None:
                break
            t.append(m.TimeUS / 1e6)
            targ.append(m.Targ)
            gyro.append(m.Gx)
    finally:
        log.close()
    return np.asarray(t), np.asarray(targ), np.asarray(gyro)


def closed_loop(t, target, measured, nperseg_s: float = NPERSEG_S):
    """Frequencies, closed-loop T, and coherence, from equal-length time series (uniform sampling assumed)."""
    from scipy.signal import coherence, csd, welch

    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    nperseg = min(len(target), int(nperseg_s * fs))
    f, pxx = welch(target, fs=fs, nperseg=nperseg)
    _, pxy = csd(target, measured, fs=fs, nperseg=nperseg)
    _, coh = coherence(target, measured, fs=fs, nperseg=nperseg)
    T = pxy / pxx
    return f, T, coh


def margins_from_open_loop(f, L):
    """(gain margin dB, phase margin deg, crossover Hz) from L(f). Each is None if the loop has no such crossing."""
    f = np.asarray(f, dtype=float)
    L = np.asarray(L, dtype=complex)
    mag = np.abs(L)
    # Unwrap over the finite bins only: np.unwrap would carry one NaN gap through every later bin.
    finite = np.isfinite(L)
    phase = np.full(len(L), np.nan)
    if finite.any():
        phase[finite] = np.degrees(np.unwrap(np.angle(L[finite])))
    logf = np.log10(np.clip(f, 1e-12, None))     # the zero-frequency bin is NaN anyway; avoid log(0)

    def downward_crossings(y, level):
        """Indices (i, frac) where y falls through level between sample i and i+1; NaN gaps are skipped."""
        out = []
        for i in range(len(y) - 1):
            a, b = y[i] - level, y[i + 1] - level
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            if a > 0 >= b:
                out.append((i, a / (a - b)))
        return out

    gm = pm = fc = None
    # Gain crossover: the highest frequency where |L| falls through 1. A loop that starts just under 1
    # at the lowest frequency has an upward crossing there, which is not a crossover.
    gc = downward_crossings(mag, 1.0)
    if gc:
        i, frac = gc[-1]
        lf = logf[i] + frac * (logf[i + 1] - logf[i])
        fc = 10 ** lf
        ph = phase[i] + frac * (phase[i + 1] - phase[i])
        pm = 180.0 + ph      # raw: below 0 means the loop is unstable at crossover
    # Phase crossover: the lowest frequency where the phase falls through -180 degrees.
    pc = downward_crossings(phase, -180.0)
    if pc:
        i, frac = pc[0]
        m = mag[i] + frac * (mag[i + 1] - mag[i])
        if np.isfinite(m) and m > 0:
            gm = -20 * math.log10(m)
    return gm, pm, fc


def _open_loop(path: str | Path):
    """(frequencies, open-loop L) with NaN where coherence is too low, or None if too little data."""
    t, targ, gyro = read_sidd(path)
    if len(t) < 64:
        return None
    f, T, coh = closed_loop(t, targ, gyro)
    keep = (coh >= COH_MIN) & (f >= F_MIN_HZ) & (f <= F_MAX_HZ) & (np.abs(1 - T) > 1e-9)
    if keep.sum() < 8:
        return None
    # Bins that fail the coherence test stay in place as NaN, so crossings never bridge a gap.
    L = np.full(len(f), np.nan + 0j)
    L[keep] = T[keep] / (1 - T[keep])
    return f, L


def sysid_margins(path: str | Path) -> dict[str, float]:
    """Margin metrics for one SystemID flight. Empty if the chirp did not give usable margins."""
    loop = _open_loop(path)
    if loop is None:
        return {}
    f, L = loop
    gm, pm, _ = margins_from_open_loop(f, L)
    out = {}
    if pm is not None:
        out["margin_pm_deg"] = round(float(pm), 2)
    if gm is not None:
        out["margin_gm_db"] = round(float(gm), 2)
    return out


def unmeasured_margin_bound(f, L):
    """For a loop whose crossover falls in a coherence gap: (last measured Hz, |L| there, phase margin bound deg).

    Uses the first run of measured bins. If |L| is still above 1 at its end, the gain crossover is above
    that frequency, and the phase keeps falling, so the phase margin is at most the value returned.
    None if there is no gap after that run, or |L| is already below 1 there.
    """
    f = np.asarray(f, dtype=float)
    L = np.asarray(L, dtype=complex)
    finite = np.isfinite(L)
    if not finite.any():
        return None
    start = int(np.argmax(finite))
    end = start
    while end + 1 < len(L) and finite[end + 1]:
        end += 1
    if end + 1 >= len(L) or not abs(L[end]) > 1.0:
        return None
    phase = np.degrees(np.unwrap(np.angle(L[start:end + 1])))
    return float(f[end]), float(abs(L[end])), float(180.0 + phase[-1])


def margin_note(path: str | Path) -> str | None:
    """One line for the reports (web page and CLI): why a SystemID flight's margins could not be measured, if the data says."""
    loop = _open_loop(path)
    if loop is None:
        return None
    bound = unmeasured_margin_bound(*loop)
    if bound is None:
        return None
    f_last, gain, pm_bound = bound
    return (f"Margins not measured: coherence drops below {COH_MIN} above {f_last:.1f} Hz. "
            f"At {f_last:.1f} Hz the gain is still {gain:.2f}, so the phase margin is likely under {pm_bound:.0f} deg.")
