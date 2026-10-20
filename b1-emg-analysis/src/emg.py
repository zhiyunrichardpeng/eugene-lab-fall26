"""Core sEMG processing: envelope, onset detection, and order-consistency statistics.

Pipeline (per channel):
    raw -> remove DC -> band-pass 20-450 Hz -> notch (mains) -> full-wave rectify
        -> low-pass 6 Hz  = envelope

Onset (Hodges & Bui 1996 style threshold):
    first sample where envelope > baseline_mean + k * baseline_sd
    and stays above for at least `min_duration` seconds.
"""
import numpy as np
from scipy import signal, stats


def envelope(x, fs, band=(20.0, 450.0), notch=50.0, lowpass=6.0):
    """Return the linear envelope of a raw EMG signal (1-D array)."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    nyq = fs / 2.0
    hi = min(band[1], 0.95 * nyq)
    b, a = signal.butter(4, [band[0] / nyq, hi / nyq], btype="bandpass")
    x = signal.filtfilt(b, a, x)
    if notch:
        for f0 in np.arange(notch, hi, notch):  # mains + harmonics inside the band
            bn, an = signal.iirnotch(f0, Q=30.0, fs=fs)
            x = signal.filtfilt(bn, an, x)
    x = np.abs(x)
    b, a = signal.butter(4, lowpass / nyq, btype="lowpass")
    return signal.filtfilt(b, a, x)


def detect_onset(env, fs, baseline, k=3.0, min_duration=0.05, search_from=None, min_rise=0.0):
    """Onset index of `env`, or None if the threshold is never sustained.

    baseline: (start, stop) sample indices of a rest segment.
    search_from: only look for onsets at/after this index (default: end of baseline).
    min_rise: threshold is at least this fraction of (peak - baseline mean). Needed for
        heavily smoothed envelopes, whose baseline SD is tiny and 3 SD fires on drift.
    """
    base = env[baseline[0]:baseline[1]]
    start = baseline[1] if search_from is None else search_from
    thr = max(base.mean() + k * base.std(),
              base.mean() + min_rise * (env[start:].max() - base.mean()))
    above = env[start:] > thr
    n = max(1, int(round(min_duration * fs)))
    # run length of consecutive True values ending at each sample
    run = np.convolve(above.astype(int), np.ones(n, dtype=int), mode="valid")
    hits = np.flatnonzero(run == n)
    return None if hits.size == 0 else start + int(hits[0])


def activation_ratio(env, baseline):
    """Peak / baseline-mean. Values near 1 mean the muscle barely activated."""
    base = env[baseline[0]:baseline[1]].mean()
    return env[baseline[1]:].max() / max(base, 1e-12)


def half_peak_time(env, start, stop):
    """Index (absolute) where env first reaches 50% of its peak within [start, stop)."""
    seg = env[start:stop]
    lo = np.median(env[:start]) if start > 0 else seg.min()
    level = lo + 0.5 * (seg.max() - lo)
    return start + int(np.argmax(seg >= level))


def kendall_w(rankings):
    """Kendall's coefficient of concordance.

    rankings: (m raters, n items) array of ranks (1..n). Returns W in [0, 1].
    No tie correction (fine for onset orders, which rarely tie exactly).
    """
    R = np.asarray(rankings, dtype=float)
    m, n = R.shape
    if n < 2:
        return np.nan
    S = ((R.sum(axis=0) - m * (n + 1) / 2.0) ** 2).sum()
    return 12.0 * S / (m ** 2 * (n ** 3 - n))


def onset_ranks(onsets):
    """Map {muscle: onset_time} -> {muscle: rank} (1 = earliest). Missing onsets ignored."""
    items = [(t, m) for m, t in onsets.items() if t is not None and np.isfinite(t)]
    order = [m for _, m in sorted(items)]
    return {m: i + 1 for i, m in enumerate(order)}


def kendall_tau(order_a, order_b):
    """Kendall's tau between two {muscle: time} dicts on their shared muscles."""
    shared = [m for m in order_a if m in order_b
              and order_a[m] is not None and order_b[m] is not None]
    if len(shared) < 2:
        return np.nan
    return stats.kendalltau([order_a[m] for m in shared], [order_b[m] for m in shared]).statistic
