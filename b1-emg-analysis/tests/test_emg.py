"""Sanity check: synthetic EMG bursts with known onsets must be recovered."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import emg  # noqa: E402

FS = 2048


def burst(onset_s, dur=6.0, total=8.0, amp=8.0, rng=None):
    t = np.arange(int(total * FS)) / FS
    x = rng.normal(0, 1.0, t.size)                              # background noise
    x += 0.5 * np.sin(2 * np.pi * 50 * t)                       # mains hum
    on = (t >= onset_s) & (t < onset_s + dur)
    ramp = np.clip((t - onset_s) / 0.05, 0, 1)                  # 50 ms rise
    x[on] += amp * ramp[on] * rng.normal(0, 1.0, on.sum())
    return x


def test_onsets_recovered():
    rng = np.random.default_rng(0)
    true = {"zyg": 1.00, "oor": 1.12, "ooc": 1.30}
    found = {}
    for m, t0 in true.items():
        env = emg.envelope(burst(t0, rng=rng), FS)
        idx = emg.detect_onset(env, FS, baseline=(int(0.2 * FS), int(0.9 * FS)))
        found[m] = idx / FS
        assert abs(found[m] - t0) < 0.06, (m, found[m], t0)
    assert list(emg.onset_ranks(found)) == ["zyg", "oor", "ooc"]


def test_kendall():
    assert emg.kendall_w([[1, 2, 3], [1, 2, 3]]) == 1.0
    assert emg.kendall_w([[1, 2, 3], [3, 2, 1]]) == 0.0
    assert emg.kendall_tau({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}) == 1.0


if __name__ == "__main__":
    test_onsets_recovered()
    test_kendall()
    print("all tests passed")
