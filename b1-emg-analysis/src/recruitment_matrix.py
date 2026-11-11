"""Recruitment matrix: for each MaM task x electrode site, are people's muscles active, and what does
the model's target pose do at that spot?

Why target poses and not runs: in all three runs we have, the solver ends within ~1% of the artist's
target at every electrode site (see results_model_runs.csv). So the target tells us where a muscle
ends up for any expression, including ones nobody has run yet. It says nothing about the path.

Human side, per clip and channel:
    a(t) = (envelope - rest) / ref
    rest = median of the session's Face-At-Rest clip (fallback: 10th percentile of the clip)
    ref  = 95th percentile over the session's standardized movement clips (near-max effort)
    hold level = 90th percentile of a(t) over the clip
Per participant: median over clips. A site is active if the participant median is >= 0.10.

Model side, per target pose and electrode pair (first tissue layer, as in fair_compare):
    still      the site moves < 0.1 mm
    shortens   eps > +2%
    stretches  eps < -2%
    moves      it moves but its length barely changes
    no muscle  the model has no driven muscle under this electrode

Usage:  python recruitment_matrix.py ../data/model/electrodes.json
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fair_compare as fc
import mam_happy
import model_run
import virtual_electrode as ve

# MaM task -> model target poses (FACS action units). Tasks without a matching target are left out.
PAIRS = {
    "Forehead": ["AU1", "AU2"],
    "Eye-Tight": ["AU6"],            # AU7 left out: its eye and frontalis meshes equal AU6's (see below)
    "Smile-Closed": ["AU12"],
    "Smile-Open": ["AU6_12_25"],
    "happy": ["AU6_12_25"],
    "Lip-Pucker": ["AU22", "phoneme_UW"],  # pucker is AU18, which the model lacks; AU22 and the UW pout are closest
    "surprised": ["AU1", "AU2"],     # brows only, the model has no jaw-drop target
    "angry": ["AU4"],                # brow lowering only
}
# Inputs found to be unreliable (checked 2026-11-11), excluded instead of interpreted:
# - target AU7: its OrbicularisOculi_l/_r and Frontalis meshes are identical to AU6 (max difference
#   0.0000); only the lips differ. Lid tightening should change the eyes, so this looks like a copy.
# - MaM "Lateraler Frontalis" l/r: near rest in Forehead (0.02) and Eye-Gentle, but active in Snarl,
#   Nose, Lip-Pucker, Smile-Open. Across 13 tasks its profile matches masseter and procerus
#   (r = 0.66-0.68), not frontalis; the medial frontalis channel behaves as expected (1.22 in Forehead).
#   Possibly mislabelled or crosstalk; to be checked with the dataset authors.
# - MaM "Levator Labii Superioris" l/r: peaks in Forehead (1.25) and tracks medial frontalis (r = 0.68-0.74),
#   while "Lateraler Frontalis" peaks in Nose and Snarl, which is what levator labii does. That fits a
#   swap of the two channels, but "Levator" is also high in happy (0.73), which fits real levator.
#   Unresolved, so both pairs are left out of the verdicts.
UNRELIABLE_CHANNELS = {"Lateraler Frontalis_l", "Lateraler Frontalis_r",
                       "Levator Labii Superioris_l", "Levator Labii Superioris_r"}
ACTIVE = 0.10
MOVE_MIN_CM = fc.MOVE_MIN_CM
SHORTEN = fc.SHORTEN


def human_levels(tasks):
    """DataFrame (task, channel) -> median, IQR and % of participants active, from participant medians."""
    df = mam_happy.index()
    use = df[df.emg.notna() & (df.sync_grade <= 1)]
    use = use[[(mam_happy.ROOT / p).exists() for p in use.emg]]
    refs = mam_happy.session_refs(use)
    rest = {}
    for r in use[use.task == "Face-At-Rest"].itertuples():
        e = pd.read_csv(mam_happy.ROOT / r.emg, index_col=0)
        if len(e):
            rest[(r.pid, r.rec)] = e.median()

    rows = []
    for r in use[use.task.isin(tasks)].itertuples():
        ref = refs.get((r.pid, r.rec))
        e = pd.read_csv(mam_happy.ROOT / r.emg, index_col=0)
        if ref is None or e.empty:
            continue
        base = rest.get((r.pid, r.rec), e.quantile(0.10))
        hold = ((e - base) / ref.reindex(e.columns)).quantile(0.90)
        rows.append(hold.rename(r.index).to_frame().T.assign(task=r.task, pid=r.pid))
    clips = pd.concat(rows)
    per_pid = clips.groupby(["task", "pid"]).median(numeric_only=True)
    out = per_pid.groupby("task").agg(["median", lambda x: (x >= ACTIVE).mean() * 100])
    return per_pid, clips.groupby("task").size()


def model_states(targets, electrodes_path, shift_cm=None):
    """{target: {channel: (state, eps_pct, moved_cm)}} read on the target pose at each electrode pair.

    shift_cm: optional (dx, dy) in cm added to both electrodes of every pair, to test how much the
    result depends on exact placement (x = participant's right, y = up; the face looks towards -z).
    """
    pos, upc = ve.load_electrodes(electrodes_path)
    if shift_cm is not None:
        offset = np.array([shift_cm[0], shift_cm[1], 0.0]) * upc
        pos = {ch: p + offset for ch, p in pos.items()}
    skin = {}
    out = {}
    for t in targets:
        pose = model_run.load_target(fc.TARGET_DIR / f"dataV22_{t}.py")
        states = {}
        for ch, tissue in ve.CHANNEL_TISSUE.items():
            if tissue is None or tissue not in pose:
                states[ch] = ("no muscle", np.nan, np.nan)
                continue
            if tissue not in skin:
                skin[tissue] = model_run.load_skinning(fc.LBS_DIR, tissue)[0]
            bind = skin[tissue]
            moved = []
            for p in pos[ch]:
                idx, w, _ = ve.surface_patch(bind, p, upc, fc.LAYER_CM * upc)
                moved.append(np.linalg.norm(w @ (pose[tissue][idx] - bind[idx])))
            moved_cm = max(moved) / upc
            eps = ve.static_shortening(bind, pose[tissue], pos[ch], upc, fc.LAYER_CM * upc)
            if moved_cm < MOVE_MIN_CM:
                state = "still"
            elif eps > SHORTEN:
                state = "shortens"
            elif eps < -SHORTEN:
                state = "stretches"
            else:
                state = "moves"
            states[ch] = (state, 100 * eps, moved_cm)
        out[t] = states
    return out


SHIFT_CM = 0.5   # electrode placement varies by several mm between people


ROBUST_MIN = 8   # of 9 placements (original + 8 shifted) must give the same state


def placement_stability(targets, electrodes_path, base):
    """Per cell, over 9 placements (original + 5 mm shifts every 45 degrees):
    (% with the same state as the original, % shortening, % stretching, robust?)."""
    angles = np.deg2rad(np.arange(0, 360, 45))
    shifted = [model_states(targets, electrodes_path, (SHIFT_CM * np.cos(a), SHIFT_CM * np.sin(a))) for a in angles]
    out = {}
    for t in targets:
        out[t] = {}
        for ch in base[t]:
            states = [base[t][ch][0]] + [s[t][ch][0] for s in shifted]
            same = sum(st == base[t][ch][0] for st in states)
            out[t][ch] = (100 * same / len(states), 100 * states.count("shortens") / len(states),
                          100 * states.count("stretches") / len(states), same >= ROBUST_MIN)
    return out


def verdict(active, state):
    if active:
        return {"shortens": "agree: active", "stretches": "sign conflict",
                "moves": "moves, no shortening"}.get(state, "missing agonist")
    return {"shortens": "over-recruitment"}.get(state, "agree: silent")


def main(electrodes_path):
    per_pid, n_clips = human_levels(list(PAIRS))
    targets = sorted({t for ts in PAIRS.values() for t in ts})
    model = model_states(targets, electrodes_path)
    stable = placement_stability(targets, electrodes_path, model)
    channels = [c for c in ve.CHANNEL_TISSUE if c not in UNRELIABLE_CHANNELS]

    rows = []
    for task, ts in PAIRS.items():
        P = per_pid.loc[task]
        for t in ts:
            for ch in channels:
                med = P[ch].median()
                pct = 100 * (P[ch] >= ACTIVE).mean()
                state, eps, moved = model[t][ch]
                if ch in UNRELIABLE_CHANNELS:
                    continue
                rows.append({"task": task, "target": t, "channel": ch, "human_median": med,
                             "human_pct_active": pct, "participants": len(P), "clips": int(n_clips[task]),
                             "model_state": state, "model_eps_pct": eps, "model_moved_cm": moved,
                             "placements_same_state_pct": stable[t][ch][0],
                             "placements_shortening_pct": stable[t][ch][1],
                             "placements_stretching_pct": stable[t][ch][2],
                             "robust_to_placement": stable[t][ch][3],
                             "verdict": verdict(med >= ACTIVE, state)})
    table = pd.DataFrame(rows)
    table.to_csv(fc.OUT / "results_recruitment_matrix.csv", index=False)

    # figure: human activation heatmap, model state as a symbol in each cell
    labels = [f"{task} | {t}" for task, ts in PAIRS.items() for t in ts]
    H = table.pivot_table(index=["task", "target"], columns="channel", values="human_median", sort=False)[channels]
    S = table.pivot_table(index=["task", "target"], columns="channel", values="model_state", aggfunc="first", sort=False)[channels]
    sym = {"shortens": "▼", "stretches": "▲", "moves": "○", "still": "·", "no muscle": ""}
    fig, ax = plt.subplots(figsize=(14, 0.45 * len(labels) + 2.6))
    im = ax.imshow(H.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(labels)):
        for j in range(len(channels)):
            s = sym[S.iat[i, j]]
            if s:
                ax.text(j, i, s, ha="center", va="center", fontsize=10,
                        color="crimson" if s in "▲" else "k")
    ax.set_xticks(range(len(channels)), [c.replace("Supercilii", "Sup.").replace("Superioris", "Sup.") for c in channels],
                  rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, label="human activation (median, 0 = rest, 1 ≈ max)")
    ax.set_title("MaM tasks vs model target poses at each electrode site\n"
                 "▼ model shortens   ▲ model stretches   ○ moves, no length change   · still   blank: no driven muscle",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(fc.OUT / "figures" / "recruitment_matrix.png", dpi=150)

    counts = table[table.model_state != "no muscle"].groupby("verdict").size()
    print(f"clips per task: {n_clips.to_dict()}")
    print("\nsites the model has a muscle for, verdict counts:\n" + counts.to_string())
    print("\nmissing agonists (human active, no driven muscle), % of participants active:")
    miss = table[(table.verdict == "missing agonist") & (table.model_state == "no muscle")]
    print(miss.groupby("channel").human_pct_active.median().sort_values(ascending=False).round(0).to_string())
    m = table[table.model_state != "no muscle"]
    m.to_csv(fc.OUT / "results_recruitment_matrix_model_sites.csv", index=False)
    print(f"\nplacement (original + 8 shifts of 5 mm): {m.robust_to_placement.sum()} of {len(m)} cells keep the "
          f"same model state in >= {ROBUST_MIN}/9 placements")
    print("\nverdicts, all cells vs robust cells only:")
    print(pd.DataFrame({"all": m.groupby("verdict").size(),
                        "robust": m[m.robust_to_placement].groupby("verdict").size()}).fillna(0).astype(int).to_string())
    show = m.round(2)
    pd.set_option("display.width", 220)
    print("\n" + show[["task", "target", "channel", "human_median", "human_pct_active", "model_state",
                       "model_eps_pct", "placements_shortening_pct", "placements_stretching_pct",
                       "robust_to_placement", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
