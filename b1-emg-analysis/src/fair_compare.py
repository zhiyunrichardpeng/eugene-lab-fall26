"""Fair comparison, happy: the model at each MaM electrode site vs that electrode's EMG.

Both sides go on the same 0 -> 1 expression-progress clock (model: path length travelled;
human: tracked smile). We never compare units (EMG is not movement); we compare
  1. recruitment: is the site active in humans, and does the model drive it?
  2. sign: where humans contract, does the model shorten the muscle there (eps > 0)?
  3. timing/shape: when does each site rise along the expression, against the spread
     between people (noise ceiling: each person vs the median of everyone else)?

Usage:  python fair_compare.py ../data/model/hpy ../data/model/electrodes.json
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mam_happy
import model_run
import virtual_electrode as ve

OUT = Path(__file__).resolve().parents[1]
ACTIVE = 0.10          # human: a(apex) above this = active (10% of near-max effort)
SHORTEN = 0.02         # model: |eps| above 2% = clear length change


def rise(curve):
    """Normalise a curve on the grid to its rise from grid 0 to its maximum: 0 -> 1."""
    c = np.asarray(curve, float)
    span = c.max() - c[0]
    return (c - c[0]) / span if span > 1e-9 else np.full_like(c, np.nan)


def t50(r):
    """Progress at which a normalised rise first reaches one half."""
    hit = np.flatnonzero(r >= 0.5)
    return mam_happy.GRID[hit[0]] if hit.size else np.nan


def model_signals(run_dir, electrodes_path, radius_scale=1.0):
    paths, _ = model_run.load_run(run_dir)
    prog = model_run.progress(paths)
    pos, upc = ve.load_electrodes(electrodes_path)
    on_grid = mam_happy.first_passage(prog, mam_happy.GRID)
    drv = set(model_run.driven(paths))
    rows = {}
    for ch, tissue in ve.CHANNEL_TISSUE.items():
        if tissue is None or tissue not in drv:
            rows[ch] = {"tissue": tissue, "driven": False}
            continue
        eps, trans, _ = ve.pair_signals(paths[tissue], pos[ch], radius_scale * upc)
        rows[ch] = {"tissue": tissue, "driven": True, "eps": eps[on_grid], "trans": trans[on_grid]}
    return rows


def human_curves():
    long, skipped = mam_happy.all_curves()
    per_pid = long.groupby(["channel", "pid", "grid"]).a.median().unstack("grid")
    return per_pid, long.clip_id.nunique(), skipped


def noise_ceiling(per_pid_rises):
    """RMSE of each participant's rise curve vs the median of the others (median, 90th pct)."""
    R = np.vstack(per_pid_rises)
    errs = []
    for i in range(len(R)):
        others = np.nanmedian(np.delete(R, i, axis=0), axis=0)
        errs.append(np.sqrt(np.nanmean((R[i] - others) ** 2)))
    return np.nanmedian(errs), np.nanpercentile(errs, 90)


def classify(h_active, m):
    if not m["driven"]:
        return "missing agonist" if h_active else "agree: silent"
    eps_end = m["eps"][-1]
    if not h_active:
        return "over-recruitment"
    if eps_end > SHORTEN:
        return "agree: contracts"
    if eps_end < -SHORTEN:
        return "sign conflict: model stretches"
    return "driven, no length change"


def main(run_dir, electrodes_path):
    model = model_signals(run_dir, electrodes_path)
    per_pid, n_clips, skipped = human_curves()
    sens = {s: model_signals(run_dir, electrodes_path, s) for s in (0.5, 2.0)}

    rows = []
    for ch, m in model.items():
        P = per_pid.loc[ch]
        apex = P[1.0]
        h_active = apex.median() > ACTIVE
        h_rises = [rise(r) for r in P.to_numpy()]
        h_med_rise = rise(P.median().to_numpy())
        row = {
            "channel": ch, "model_tissue": m["tissue"] or "-",
            "human_apex_median": apex.median(), "human_apex_iqr": apex.quantile(.75) - apex.quantile(.25),
            "human_pct_active": 100 * (apex > ACTIVE).mean(),
            "human_t50": t50(h_med_rise),
            "model_driven": m["driven"],
            "verdict": classify(h_active, m),
        }
        if m["driven"]:
            m_rise = rise(m["trans"])
            ceil_med, ceil_90 = noise_ceiling(h_rises)
            row.update({
                "model_eps_end_pct": 100 * m["eps"][-1],
                "model_eps_end_pct_r0.5": 100 * sens[0.5][ch]["eps"][-1],
                "model_eps_end_pct_r2": 100 * sens[2.0][ch]["eps"][-1],
                "model_t50": t50(m_rise),
                "shape_rmse_model": np.sqrt(np.nanmean((m_rise - h_med_rise) ** 2)),
                "shape_rmse_ceiling_median": ceil_med, "shape_rmse_ceiling_p90": ceil_90,
            })
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "results_fair_compare_happy.csv", index=False)

    # figure: the four driven sites, human rise (median + IQR) vs model rise
    driven = [r["channel"] for r in rows if r["model_driven"]]
    fig, axes = plt.subplots(1, len(driven), figsize=(3.4 * len(driven), 3.2), sharey=True)
    g = mam_happy.GRID
    for ax, ch in zip(np.atleast_1d(axes), driven):
        H = np.vstack([rise(r) for r in per_pid.loc[ch].to_numpy()])
        ax.fill_between(g, np.nanpercentile(H, 25, 0), np.nanpercentile(H, 75, 0), color="C0", alpha=.25, lw=0)
        ax.plot(g, rise(per_pid.loc[ch].median().to_numpy()), color="C0", label="human EMG (median, IQR)")
        ax.plot(g, rise(model[ch]["trans"]), color="C1", label="model: movement at site")
        e = model[ch]["eps"]
        ax.plot(g, e / np.abs(e).max(), color="C1", ls="--", label="model: shortening (+) / stretch (-)")
        ax.axhline(0, color="k", lw=.5)
        ax.set_title(ch, fontsize=9)
        ax.set_xlabel("expression progress")
    np.atleast_1d(axes)[0].set_ylabel("normalised rise")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Happy: model at MaM electrode sites vs EMG ({per_pid.index.get_level_values('pid').nunique()} participants, {n_clips} clips)", fontsize=10)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "fair_compare_happy.png", dpi=150)

    pd.set_option("display.width", 200)
    print(f"clips used {n_clips}, skipped {skipped}")
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
