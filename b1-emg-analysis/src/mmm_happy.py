"""MMM sample (1 participant): envelopes of the 'happy' clips, plus MediaPipe smile motion.

The MMM sEMG is already an RMS envelope at 30 Hz (500 ms window), so we only
normalise it. Onset = first frame above baseline mean + 3 SD for >= 3 frames
(100 ms), baseline = first 0.5 s of the clip.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import emg

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mmm_sample" / "MMM"
FIG = ROOT / "figures"
FS = 30.0

# muscles that map onto the lab model's tissues (right side; model OOC_R / OOR)
MUSCLES = {
    "Zygomaticus_r": "Zygomaticus (no model tissue)",
    "Orbicularis Oris_r": "Orbicularis oris (OOR)",
    "Orbicularis Oculi_r": "Orbicularis oculi (OOC_R)",
    "Levator Labii Superioris_r": "Levator labii sup. (no model tissue)",
    "Lateraler Frontalis_r": "Frontalis (frontalis)",
    "Masseter_r": "Masseter (jaw)",
}
BASE = (0, int(0.5 * FS))


def load(rel):
    e = pd.read_csv(DATA / rel, index_col=0).clip(lower=0)
    return e


def mediapipe_smile(index):
    idx = pd.read_csv(DATA / "MMM_3DMM.csv", dtype={"pid": str})
    rel = idx.loc[idx["index"] == index, "MediaPipe"].iloc[0]
    mp = pd.read_csv(DATA / rel)
    return mp[["mouthSmileLeft", "mouthSmileRight"]].mean(axis=1).to_numpy()


def main():
    FIG.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "MMM.csv", dtype={"pid": str})
    clips = df[(df.task == "happy") & df.emg.notna()]
    rows, ratios = [], {}
    fig, axes = plt.subplots(len(clips), 1, figsize=(9, 2.6 * len(clips)), sharex=True)
    for ax, (_, c) in zip(np.atleast_1d(axes), clips.iterrows()):
        e = load(c.emg)
        t = np.arange(len(e)) / FS
        onsets = {}
        for col, label in MUSCLES.items():
            x = e[col].to_numpy()
            x = x / x.max()
            i = emg.detect_onset(x, FS, BASE, k=3.0, min_duration=0.1, min_rise=0.1)
            ratios.setdefault(label, []).append(emg.activation_ratio(x, BASE))
            onsets[label] = None if i is None else i / FS
            (ln,) = ax.plot(t, x, lw=1.2, label=label)
            if i is not None:
                ax.axvline(i / FS, color=ln.get_color(), ls=":", lw=1)
        smile = mediapipe_smile(c["index"])
        ax.plot(t[: len(smile)], smile / smile.max(), "k--", lw=1.5, label="MediaPipe smile (video)")
        j = emg.detect_onset(smile / smile.max(), FS, BASE, k=3.0, min_duration=0.1, min_rise=0.1)
        onsets["Video: mouth smile"] = None if j is None else j / FS
        ax.set_title(c["index"], fontsize=9, loc="left")
        ax.set_ylabel("norm. envelope")
        rows.append({"clip": c["index"], **onsets})
    np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    np.atleast_1d(axes)[0].legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "mmm_happy_envelopes.png", dpi=150)

    table = pd.DataFrame(rows).set_index("clip")
    table.to_csv(ROOT / "results_mmm_happy_onsets.csv")
    print(table.round(2).T.to_string())
    print("\nmedian peak/baseline ratio (≈1 means barely active):")
    for m, r in ratios.items():
        print(f"  {m:40s} {np.median(r):6.1f}")
    ranks = [emg.onset_ranks(r.to_dict()) for _, r in table.iterrows()]
    common = sorted(set.intersection(*(set(r) for r in ranks)))
    R = np.array([[r[m] for m in common] for r in ranks])
    # re-rank on the common set
    R = np.argsort(np.argsort(R, axis=1), axis=1) + 1
    print(f"\nKendall's W across {len(ranks)} happy clips ({len(common)} muscles): {emg.kendall_w(R):.2f}")


if __name__ == "__main__":
    main()
