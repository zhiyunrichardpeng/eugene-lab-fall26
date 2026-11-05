"""Mimetic Interfaces 2015, phase A: muscle onset latencies for voluntary smiles.

12 participants (the dataset authors excluded 6, 7, 14 for not performing the tasks),
10 smiles each, raw sEMG 2048 Hz from the right side of the face.

Per smile trial (cue = task start label, taken from video by the dataset authors):
    baseline = [-1.5 s, -0.25 s] before the cue (neutral-expression task)
    onset    = first time after the cue the envelope exceeds
               max(baseline + 3 SD, baseline + 10 % of the rise) for >= 50 ms
    latency  = onset - cue
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matio import load_from_mat
from scipy import stats

import emg

warnings.filterwarnings("ignore", module="matio")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mimetic2015" / "Data"
FIG = ROOT / "figures"
FS = 2048
MUSCLES = ["corrugator", "zygomaticus", "orb_oris", "orb_oculi"]   # column order in Eraw
MODEL = {"corrugator": "—", "zygomaticus": "— (moves OOR passively)",
         "orb_oris": "OOR", "orb_oculi": "OOC_R"}
INCLUDED = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 15]
SMILE = 2
PRE, POST = 2.0, 6.0                 # segment window around the cue (s)
BASE = (-1.5, -0.25)                 # baseline window (s, relative to cue)
SEARCH = 3.0                         # look for onsets within 3 s after the cue
ACTIVE_RATIO = 2.0                   # peak/baseline below this = "not active"


def load_phase_a(pid):
    p = load_from_mat(DATA / f"data{pid:02d}A.mat")["D"].properties
    names = [str(np.squeeze(n)) for n in p["varnames"].ravel()]
    cols = dict(zip(names, (np.asarray(d) for d in p["data"].ravel())))
    return cols["Eraw"], cols["L"].ravel()


def analyse_participant(pid):
    raw, labels = load_phase_a(pid)
    env = np.column_stack([emg.envelope(raw[:, c], FS, notch=50.0) for c in range(raw.shape[1])])
    cues = np.flatnonzero(labels == SMILE)
    rows, segs = [], []
    for trial, cue in enumerate(cues, 1):
        a, b = cue - int(PRE * FS), cue + int(POST * FS)
        if a < 0 or b > len(env):
            continue
        seg = env[a:b]
        base = (int((PRE + BASE[0]) * FS), int((PRE + BASE[1]) * FS))
        search_to = int((PRE + SEARCH) * FS)
        segs.append(seg)
        for c, m in enumerate(MUSCLES):
            x = seg[:search_to, c]
            i = emg.detect_onset(x, FS, base, k=3.0, min_duration=0.05,
                                 search_from=int(PRE * FS), min_rise=0.1)
            ratio = emg.activation_ratio(seg[:, c], base)
            rows.append({"participant": pid, "trial": trial, "muscle": m,
                         "latency_s": np.nan if i is None else i / FS - PRE,
                         "activation_ratio": ratio})
    return pd.DataFrame(rows), np.array(segs)


def plot_example(pid, segs, trials):
    t = np.arange(segs.shape[1]) / FS - PRE
    fig, axes = plt.subplots(len(MUSCLES), 1, figsize=(8, 7), sharex=True)
    for c, (ax, m) in enumerate(zip(axes, MUSCLES)):
        for k, s in enumerate(segs):
            ax.plot(t, s[:, c], color="0.6", lw=0.6)
            lat = trials.query("trial == @k + 1 and muscle == @m").latency_s
            if len(lat) and np.isfinite(lat.iloc[0]):
                ax.plot(lat.iloc[0], np.interp(lat.iloc[0], t, s[:, c]), "o", ms=3, color="C3")
        ax.plot(t, np.median(segs[:, :, c], axis=0), color="k", lw=1.5)
        ax.axvline(0, color="C0", ls="--", lw=1)
        ax.set_ylabel(f"{m}\n(µV)", fontsize=8)
    axes[0].set_title(f"Participant {pid}: 10 smiles (grey), median (black), onsets (red), cue (blue)",
                      fontsize=9, loc="left")
    axes[-1].set_xlabel("time from smile cue (s)")
    fig.tight_layout()
    fig.savefig(FIG / f"mimetic_P{pid:02d}_smile_trials.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    all_trials = []
    for pid in INCLUDED:
        df, segs = analyse_participant(pid)
        all_trials.append(df)
        if pid == INCLUDED[0]:
            plot_example(pid, segs, df)
        print(f"participant {pid:2d}: {df.trial.nunique()} smiles")
    trials = pd.concat(all_trials, ignore_index=True)
    trials.to_csv(ROOT / "results_mimetic_smile_trials.csv", index=False)

    # which muscles are actually active in a smile?
    act = trials.groupby("muscle").activation_ratio.median()
    active = [m for m in MUSCLES if act[m] >= ACTIVE_RATIO]
    print("\nmedian peak/baseline ratio:\n" + act.reindex(MUSCLES).round(1).to_string())
    print("active muscles used for ordering:", active)

    # per-participant median latency and order
    med = (trials[trials.muscle.isin(active)]
           .groupby(["participant", "muscle"]).latency_s.median().unstack()[active])
    order = med.apply(lambda r: " → ".join(r.dropna().sort_values().index), axis=1)
    table = med.round(3).assign(order=order)
    table.to_csv(ROOT / "results_mimetic_smile_order.csv")
    print("\nmedian onset latency (s) per participant:\n" + table.to_string())

    # consistency
    def ranks(frame):
        return frame.rank(axis=1).dropna().to_numpy()
    w_between = emg.kendall_w(ranks(med))
    w_within = []
    for pid, g in trials[trials.muscle.isin(active)].groupby("participant"):
        r = g.pivot(index="trial", columns="muscle", values="latency_s")[active]
        if len(r.dropna()) >= 3:
            w_within.append(emg.kendall_w(ranks(r)))
    print(f"\nKendall's W between participants (median orders): {w_between:.2f}")
    print(f"Kendall's W within participants (across 10 trials): median {np.median(w_within):.2f} "
          f"(range {np.min(w_within):.2f}–{np.max(w_within):.2f})")

    # pairwise lags relative to zygomaticus, tested across participants
    print("\nlag relative to zygomaticus (participant medians):")
    lags = {}
    for m in active:
        if m == "zygomaticus":
            continue
        d = (med[m] - med["zygomaticus"]).dropna()
        lags[m] = d
        p = stats.wilcoxon(d).pvalue if len(d) >= 5 else np.nan
        print(f"  {m:10s} median {1000 * d.median():6.0f} ms   later in {np.sum(d > 0)}/{len(d)}"
              f" participants   Wilcoxon p = {p:.3f}")

    # main figure
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [med[m].dropna().to_numpy() * 1000 for m in active]
    ax.boxplot(data, vert=False, widths=0.5, showfliers=False)
    for k, d in enumerate(data, 1):
        ax.scatter(d, np.full_like(d, k) + np.random.default_rng(k).uniform(-0.12, 0.12, d.size),
                   s=18, color="C3", zorder=3)
    ax.set_yticks(range(1, len(active) + 1),
                  [f"{m}\n(model: {MODEL[m]})" for m in active], fontsize=8)
    ax.set_xlabel("onset latency after smile cue (ms)")
    ax.set_title(f"Voluntary smile, {len(med)} participants × 10 trials "
                 f"(dots = participant medians)\nKendall's W between participants = {w_between:.2f}",
                 fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "mimetic_smile_onsets.png", dpi=150)
    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
