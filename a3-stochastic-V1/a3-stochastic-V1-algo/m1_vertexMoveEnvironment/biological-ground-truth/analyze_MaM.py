"""
=============================================================================
MaM sEMG analysis: exploration charts + model-ready human recruitment
=============================================================================
SIGGRAPH TOG figure style (as in sample_MaM.py): Times, ticks inside on all
four sides, TrueType fonts, every figure saved as 600 DPI PNG and as PDF.

Usage (full release, video-derived hold windows):
    python analyze_MaM.py
    python analyze_MaM.py --root /data/MaM_Public --mmm-root /data/MMM/MMM

Usage (local copy without MediaPipe parameters):
    python analyze_MaM.py --root MaM --hold emg

Channel corrections (mam_data.SCHAEDE_COLUMN_SWAPS, EXCLUDED_CHANNELS) are applied by
default; --as-released reproduces the uncorrected numbers.

Output (default visualization/analyze_MaM/, never commit it, MaM is under a
data-use agreement):
    F01..F15 *.png / *.pdf, clips_inventory.csv, features_long.csv,
    human_recruitment_maya.csv, human_recruitment_maya_summary.csv, findings.md
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import mam_data as m

HERE = Path(__file__).resolve().parent

# ===================== PALETTE =====================
# Categorical slots, fixed order (validated: CVD and normal-vision separation pass on
# adjacent pairs; aqua, yellow and magenta sit below 3:1, so they always come with a legend).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
MUSCLE_COLOR = dict(zip(m.MUSCLES_OF_INTEREST, SERIES))
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]).with_extremes(bad=SURFACE)
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948", "#a52b2a"]).with_extremes(bad=SURFACE)
DOUBLE_COL = 7.0  # inches, TOG two-column figure width
# ===================================================


def apply_style() -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42,  # TrueType in PDF, no Type-3 bitmaps
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.labelpad": 3,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": INK,
        "legend.fancybox": False,
        "lines.linewidth": 1.2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


# ----------------------------------------------------------------------------
# Shared drawing helpers
# ----------------------------------------------------------------------------

def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}")


def ink_on(rgba) -> str:
    r, g, b = to_rgb(rgba)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else INK


def heatmap(ax, values: np.ndarray, cmap, vmin: float, vmax: float, xlabels, ylabels, xrot: float = 55):
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=xrot, ha="right" if xrot else "center", rotation_mode="anchor")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_xticks(np.arange(-0.5, len(xlabels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.2)
    ax.tick_params(which="both", top=False, right=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def colorbar(fig, im, ax, label: str, **kwargs):
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015, **kwargs)
    cb.set_label(label, fontweight="bold")
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(direction="out", length=2, width=0.6, right=True, left=False)
    return cb


def boxes(ax, groups: list[np.ndarray], positions, color: str, width: float = 0.32, vert: bool = True,
          fliers: bool = True) -> None:
    keep = [(np.asarray(g)[np.isfinite(g)], p) for g, p in zip(groups, positions)]
    keep = [(g, p) for g, p in keep if len(g)]
    if not keep:
        return
    bp = ax.boxplot([g for g, _ in keep], positions=[p for _, p in keep], widths=width, orientation="vertical" if vert else "horizontal",
                    patch_artist=True, showfliers=fliers, whis=(5, 95),
                    flierprops=dict(marker="o", markersize=1.6, markerfacecolor=color, markeredgewidth=0, alpha=0.6),
                    medianprops=dict(color=color, linewidth=1.6), whiskerprops=dict(color=color, linewidth=0.8),
                    capprops=dict(color=color, linewidth=0.8), boxprops=dict(edgecolor=color, linewidth=0.8))
    for patch in bp["boxes"]:
        patch.set_facecolor((*to_rgb(color), 0.12))


def pct(x) -> str:
    return f"{100 * x:.0f}%"


def task_label(task: str) -> str:
    return task[0].upper() + task[1:] if task in m.EMOTION_TASKS else task


def mapped_rows() -> list[tuple[m.MayaExpression, str, str]]:
    return [(e, t, r) for e, t, r in m.maya_rows() if t is not None]


def row_label(expr: m.MayaExpression, task: str | None, role: str) -> str:
    name = f"{expr.name} (alt)" if role == "alt" else expr.name
    if not task:
        return f"{name} · no MaM task"
    return name if task.lower() == expr.name.lower() else f"{name} · {task_label(task)}"


def chan_label(ds: m.Dataset, channel: str) -> str:
    label = f"{m.SITE_LABELS[channel[:-2]]} {channel[-1].upper()}"
    return f"{label} (excl.)" if channel in ds.excluded else label


def legend_below(fig, handles, ncols: int) -> None:
    """Legend centred under everything drawn so far, rotated tick labels included."""
    bottom = fig.get_tightbbox(fig.canvas.get_renderer()).y0 / fig.get_figheight()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, bottom - 0.01), ncols=ncols)


def grid_axes(n: int, ncols: int, height: float, sharex: bool = True):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL, height), sharex=sharex, squeeze=False)
    return fig, axes.ravel()


# ----------------------------------------------------------------------------
# Analysis state
# ----------------------------------------------------------------------------

@dataclass
class Analysis:
    ds: m.Dataset
    site: pd.DataFrame  # clip x site
    sess: pd.DataFrame  # session x task x site
    part: pd.DataFrame  # participant x task x site
    grp: pd.DataFrame  # task x site
    chan_part: pd.DataFrame  # participant x task x channel
    chan_grp: pd.DataFrame  # task x channel
    rel: pd.DataFrame
    repeat: pd.DataFrame
    top: pd.DataFrame
    oo: pd.DataFrame
    oo_grade0: pd.DataFrame
    mmm_oo: pd.DataFrame | None
    cross: pd.DataFrame | None
    mmm_cross: pd.DataFrame | None
    scan_released: pd.DataFrame
    scan_loaded: pd.DataFrame
    sim_released: pd.DataFrame
    sim_loaded: pd.DataFrame


def analyse(ds: m.Dataset, mmm: m.Dataset | None) -> Analysis:
    site = m.site_values(ds.features)
    sess = m.session_values(site)
    part = m.participant_values(sess)
    chan = ds.features.drop(columns="site").rename(columns={"channel": "site"})
    chan_part = m.participant_values(m.session_values(chan))
    grade0 = ds.features[ds.features["sync_grade"] == 0]
    has_blend = len(ds.blend) > 0 and ds.hold_source == "mediapipe"
    return Analysis(
        ds=ds, site=site, sess=sess, part=part, grp=m.group_values(part),
        chan_part=chan_part, chan_grp=m.group_values(chan_part),
        rel=m.reliability(sess), repeat=m.imitation_repeatability(site), top=m.top_sites(sess),
        oo=m.oculi_oris(ds.features), oo_grade0=m.oculi_oris(grade0),
        mmm_oo=m.oculi_oris(mmm.features) if mmm else None,
        cross=m.blendshape_crosscheck(ds) if has_blend else None,
        mmm_cross=m.blendshape_crosscheck(mmm) if mmm is not None and len(mmm.blend) else None,
        scan_released=m.label_swap_scan(ds, released=True), scan_loaded=m.label_swap_scan(ds, released=False),
        sim_released=m.profile_similarity(ds, released=True), sim_loaded=m.profile_similarity(ds, released=False),
    )


def group_matrix(grp: pd.DataFrame, rows, cols) -> np.ndarray:
    table = grp.pivot(index="task", columns="site", values="median")
    return table.reindex(index=list(rows), columns=list(cols)).to_numpy(dtype=float)


def tasks_present(a: Analysis, include_rest: bool = True) -> list[str]:
    present = set(a.ds.ok["task"])
    return [t for t in m.TASKS if t in present and (include_rest or t != m.REST_TASK)]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def fig_inventory(a: Analysis, out: Path) -> None:
    clips = a.ds.clips
    pids = sorted(clips["pid"].unique())
    recs = sorted(clips["rec"].unique(), key=lambda r: (int(r[:-1]), r))
    ok_counts = clips[clips["status"] == "ok"].groupby(["pid", "rec"]).size()
    totals = clips.groupby(["pid", "rec"]).size()
    grid = np.full((len(pids), len(recs)), np.nan)
    for (pid, rec), total in totals.items():
        grid[pids.index(pid), recs.index(rec)] = ok_counts.get((pid, rec), 0)

    fig = plt.figure(figsize=(DOUBLE_COL, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], width_ratios=[1, 1.15], hspace=0.3, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    im = heatmap(ax, grid, SEQ, 0, max(np.nanmax(grid), 1), [f"{r}" for r in recs], pids, xrot=0)
    for i, pid in enumerate(pids):
        for j, rec in enumerate(recs):
            if np.isfinite(grid[i, j]):
                total = totals[(pid, rec)]
                text = f"{grid[i, j]:.0f}" if grid[i, j] == total else f"{grid[i, j]:.0f}/{total}"
                ax.text(j, i, text, ha="center", va="center", fontsize=6.5, color=ink_on(im.cmap(im.norm(grid[i, j]))),
                        fontweight="bold" if grid[i, j] < total else "normal")
    ax.set_xlabel("Sensor session")
    ax.set_ylabel("Participant")
    ax.set_title("(a) Usable clips per session", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    ok = clips[clips["status"] == "ok"]
    grades = ok.pivot_table(index="pid", columns="sync_grade", values="index", aggfunc="count").reindex(pids).fillna(0)
    left = np.zeros(len(pids))
    for k, grade in enumerate(sorted(grades.columns)):
        ax.barh(range(len(pids)), grades[grade], left=left, height=0.62, color=SERIES[k % len(SERIES)],
                edgecolor=SURFACE, linewidth=1.0, label=f"grade {grade:.0f}")
        left += grades[grade].to_numpy()
    ax.set_yticks(range(len(pids)), pids)
    ax.set_ylim(len(pids) - 0.5, -0.5)
    ax.set_xlim(0, left.max() * 1.3)
    ax.set_xlabel("Usable clips")
    ax.set_title("(b) sync_grade per participant", loc="left")
    ax.legend(loc="center right", handlelength=1.0)
    ax.tick_params(axis="y", length=0)

    for col, mode in enumerate(["schaede", "emotion"]):
        ax = fig.add_subplot(gs[1, col])
        lengths = ok.loc[ok["mode"] == mode, "n_frames"].astype(float) / m.FPS
        ax.hist(lengths, bins=30, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
        ax.set_xlim(left=0)
        ax.set_xlabel("Clip length (s)")
        ax.set_ylabel("Clips")
        ax.set_title(f"({'cd'[col]}) {mode.capitalize()} clips, median {lengths.median():.1f} s", loc="left")
        ax.minorticks_on()
        short = int((lengths < lengths.median() - 0.5).sum())
        if mode == "emotion" and short:
            ax.text(0.03, 0.95, f"{short} of {len(lengths)} clips are\nshorter than {lengths.median() - 0.5:.1f} s",
                    transform=ax.transAxes, va="top", fontsize=6.5, color=INK2)

    problems = clips[clips["status"] != "ok"].groupby(["session", "status"]).size()
    note = "Not usable: " + "; ".join(f"{s} {st.replace('_', ' ')} ({n})" for (s, st), n in problems.items()) if len(problems) else ""
    fig.text(0.01, -0.01, note, fontsize=6.5, color=INK2, ha="left", va="top", wrap=True)
    save(fig, out, "F01_inventory")


def session_levels(ds: m.Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Session x channel, raw units: Face-At-Rest median and schaede 95th percentile (excluded channels too)."""
    rest, p95 = {}, {}
    for session, group in ds.ok.groupby("session"):
        schaede = np.concatenate([ds.raw[i] for i in group.index[group["mode"] == "schaede"]])
        rest[session] = ds.baselines[session].rest
        p95[session] = np.percentile(schaede, 95, axis=0)
    return pd.DataFrame(rest, index=m.CHANNELS).T, pd.DataFrame(p95, index=m.CHANNELS).T


def fig_channel_quality(a: Analysis, out: Path) -> None:
    rest, p95 = session_levels(a.ds)
    order = list(m.DISPLAY_CHANNELS)
    flat = (p95 / rest < 1.5)[order].sum()

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.3))
    x = np.arange(len(order))
    boxes(ax, [rest[c].to_numpy() for c in order], x - 0.19, SERIES[1], width=0.3)
    boxes(ax, [p95[c].to_numpy() for c in order], x + 0.19, SERIES[0], width=0.3)
    ax.set_yscale("log")
    ax.set_xticks(x, [chan_label(a.ds, c) for c in order], rotation=55, ha="right", rotation_mode="anchor", fontsize=6.5)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(r"Envelope ($\mu$V)")
    for i, c in enumerate(order):
        if flat[c]:
            ax.text(i, 1.0, f"{flat[c]}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_title("Per-session channel levels (numbers on top: sessions where schaede p95 < 1.5 × rest, i.e. a flat channel)",
                 loc="left", pad=10)
    ax.tick_params(axis="x", top=False)
    legend_below(fig, [Patch(facecolor=(*to_rgb(SERIES[1]), 0.12), edgecolor=SERIES[1], label="Face-At-Rest median"),
                       Patch(facecolor=(*to_rgb(SERIES[0]), 0.12), edgecolor=SERIES[0], label="Schaede 95th percentile")], 2)
    save(fig, out, "F02_channel_quality")


def fig_task_channel_heatmap(a: Analysis, out: Path) -> None:
    tasks = tasks_present(a)
    cols = list(m.DISPLAY_CHANNELS)
    values = group_matrix(a.chan_grp, tasks, cols)
    vmax = 2.0
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.9))
    im = heatmap(ax, values, SEQ, 0, vmax, [chan_label(a.ds, c) for c in cols],
                 [task_label(t) for t in tasks])
    for i, task in enumerate(tasks):
        if task != m.REST_TASK and np.isfinite(values[i]).any():
            j = int(np.nanargmax(values[i]))
            ax.plot(j, i, "o", markersize=3.2, color=ink_on(im.cmap(im.norm(values[i, j]))))
    n_schaede = sum(t in m.SCHAEDE_TASKS for t in tasks)
    ax.axhline(n_schaede - 0.5, color=INK, linewidth=0.8)
    colorbar(fig, im, ax, "Group median, % of reference", extend="max",
             format=matplotlib.ticker.FuncFormatter(lambda v, _: pct(v)))
    ax.set_title("Recruitment per task and channel (dot: strongest channel)", loc="left")
    save(fig, out, "F03_task_channel_heatmap")


def fig_top_site_share(a: Analysis, out: Path) -> None:
    tasks = tasks_present(a, include_rest=False)
    share = (a.top.groupby("task")["top_site"].value_counts(normalize=True).unstack(fill_value=0)
             .reindex(index=tasks, columns=list(m.SITES), fill_value=0))
    n = a.top.groupby("task").size().reindex(tasks)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.3))
    im = heatmap(ax, share.to_numpy(), SEQ, 0, 1, [m.SITE_LABELS[s] for s in m.SITES],
                 [f"{task_label(t)} (n={n[t]})" for t in tasks], xrot=35)
    for i in range(len(tasks)):
        for j in range(len(m.SITES)):
            v = share.iat[i, j]
            if v >= 0.10:
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.5, color=ink_on(im.cmap(im.norm(v))))
    ax.axhline(sum(t in m.SCHAEDE_TASKS for t in tasks) - 0.5, color=INK, linewidth=0.8)
    colorbar(fig, im, ax, "Sessions with this top site", format=matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Which muscle leads each task (% of sessions, shown from 10%)", loc="left")
    save(fig, out, "F04_top_site_share")


def fig_time_courses(a: Analysis, out: Path) -> None:
    rows = mapped_rows()
    fig, axes = grid_axes(len(rows) + 1, 4, 5.9)
    t = (np.arange(-2 * m.FPS, 6 * m.FPS)) / m.FPS
    for ax, (expr, task, role) in zip(axes, rows):
        traces = m.aligned_traces(a.ds, task)
        for site in m.MUSCLES_OF_INTEREST:
            stack = np.stack(traces.loc[traces["site"] == site, "frames"].to_numpy()) if len(traces) else np.full((1, len(t)), np.nan)
            with warnings.catch_warnings():  # time points no participant reaches are all-NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(stack, axis=0)
                q25, q75 = np.nanpercentile(stack, [25, 75], axis=0)
            ax.fill_between(t, 100 * q25, 100 * q75, color=MUSCLE_COLOR[site], alpha=0.10, linewidth=0)
            ax.plot(t, 100 * med, color=MUSCLE_COLOR[site], linewidth=1.1)
        ax.axvline(0, color=HAIRLINE, linewidth=0.8, zorder=0)
        ax.axhline(0, color=HAIRLINE, linewidth=0.8, zorder=0)
        ax.set_title(row_label(expr, task, role), loc="left", fontsize=7.5)
        ax.set_xlim(t[0], t[-1])
    for ax in axes[len(rows):]:
        ax.axis("off")
    axes[len(rows)].legend(handles=[Line2D([], [], color=MUSCLE_COLOR[s], linewidth=1.6, label=m.SITE_LABELS[s]) for s in m.MUSCLES_OF_INTEREST],
                           loc="center", title="Muscle (L/R mean)", title_fontproperties={"weight": "bold", "size": 7})
    for ax in axes[::4]:
        ax.set_ylabel("% of reference")
    for ax in axes[len(rows) - 4: len(rows)]:
        ax.set_xlabel("Time from first hold onset (s)")
        ax.xaxis.set_tick_params(labelbottom=True)
    fig.suptitle("Envelope time course: median across participants, band = IQR", x=0.01, ha="left", fontsize=8.5, fontweight="bold")
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, out, "F05_time_courses")


def typical_clip(a: Analysis, task: str) -> str | None:
    clips = m.wide(a.site[a.site["task"] == task], ["index"]).dropna()
    target = a.grp[a.grp["task"] == task].set_index("site")["median"].reindex(list(m.RANK_SITES))
    if clips.empty or target.isna().all():
        return None
    return str((clips - target.to_numpy()).pow(2).sum(axis=1).idxmin())


def fig_exemplar_raw(a: Analysis, out: Path) -> None:
    rows = mapped_rows()
    fig, axes = grid_axes(len(rows) + 1, 4, 5.9, sharex=False)
    cols = {s: [m.CHANNELS.index(f"{s}_{side}") for side in "lr"] for s in m.MUSCLES_OF_INTEREST}
    for ax, (expr, task, role) in zip(axes, rows):
        idx = typical_clip(a, task)
        if idx is None:
            ax.axis("off")
            continue
        raw, hold = a.ds.raw[idx], a.ds.hold[idx]
        t = np.arange(len(raw)) / m.FPS
        edges = np.flatnonzero(np.diff(np.r_[0, hold.astype(int), 0])) if task != m.REST_TASK else []
        for start, stop in zip(edges[::2], edges[1::2]):
            ax.axvspan(start / m.FPS, stop / m.FPS, color="#f0efec", linewidth=0, zorder=0)
        for site, c in cols.items():
            ax.plot(t, raw[:, c].mean(axis=1), color=MUSCLE_COLOR[site], linewidth=0.9)
        ax.set_title(f"{row_label(expr, task, role)} · {a.ds.clips.at[idx, 'session']}", loc="left", fontsize=7)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(bottom=0)
    for ax in axes[len(rows):]:
        ax.axis("off")
    handles = [Line2D([], [], color=MUSCLE_COLOR[s], linewidth=1.6, label=m.SITE_LABELS[s]) for s in m.MUSCLES_OF_INTEREST]
    handles.append(Patch(facecolor="#f0efec", label=f"Hold window ({a.ds.hold_source})"))
    axes[len(rows)].legend(handles=handles, loc="center", title="Muscle (L/R mean)", title_fontproperties={"weight": "bold", "size": 7})
    for ax in axes[::4]:
        ax.set_ylabel(r"Activation ($\mu$V)")
    for ax in axes[len(rows) - 4: len(rows)]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Most typical clip per expression (closest to the group median), raw envelope",
                 x=0.01, ha="left", fontsize=8.5, fontweight="bold")
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, out, "F06_exemplar_raw")


def fig_reliability(a: Analysis, out: Path) -> None:
    tasks = tasks_present(a, include_rest=False)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.4), gridspec_kw={"width_ratios": [3.2, 1]})
    x = np.arange(len(tasks))
    for kind, color, dx in [("within", SERIES[0], -0.19), ("between", SERIES[1], 0.19)]:
        sub = a.rel[a.rel["kind"] == kind]
        # thousands of between-participant pairs per task: whiskers (5-95%) only, no flier wall
        boxes(ax, [sub.loc[sub["task"] == t, "rho"].to_numpy() for t in tasks], x + dx, color, width=0.32, fliers=kind == "within")
    ax.set_xticks(x, [task_label(t) for t in tasks], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(tasks) - 0.4)
    ax.axhline(0, color=HAIRLINE, linewidth=0.8, zorder=0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(r"Spearman $\rho$ of site vectors")
    ax.set_title("(a) Same participant across sessions vs different participants", loc="left")
    ax.tick_params(axis="x", top=False)

    emotions = [t for t in m.EMOTION_TASKS if t in set(a.repeat["task"])]
    boxes(ax2, [a.repeat.loc[a.repeat["task"] == t, "rho"].to_numpy() for t in emotions], np.arange(len(emotions)), SERIES[0], width=0.5)
    ax2.set_xticks(range(len(emotions)), [task_label(t) for t in emotions], rotation=45, ha="right", rotation_mode="anchor")
    ax2.set_xlim(-0.6, max(len(emotions), 1) - 0.4)
    ax2.set_ylim(-1.05, 1.05)
    ax2.axhline(0, color=HAIRLINE, linewidth=0.8, zorder=0)
    ax2.set_title("(b) Repeated imitations", loc="left")
    ax2.tick_params(axis="x", top=False)
    fig.tight_layout(w_pad=1.0)
    legend_below(fig, [Patch(facecolor=(*to_rgb(SERIES[0]), 0.12), edgecolor=SERIES[0], label="Same participant, other session (or imitation)"),
                       Patch(facecolor=(*to_rgb(SERIES[1]), 0.12), edgecolor=SERIES[1], label="Different participants (whiskers 5-95%)")], 2)
    save(fig, out, "F07_reliability")


def fig_asymmetry(a: Analysis, out: Path) -> None:
    tasks = tasks_present(a, include_rest=False)
    cp = a.chan_part.assign(muscle=a.chan_part["site"].str[:-2], side=a.chan_part["site"].str[-1])
    lr = cp.pivot_table(index=["pid", "task", "muscle"], columns="side", values="norm")
    left, right = lr["l"].clip(lower=0), lr["r"].clip(lower=0)
    total = left + right
    index = ((left - right) / total).where(total >= 0.1)
    values = (index.groupby(level=["task", "muscle"]).median().unstack()
              .reindex(index=tasks, columns=list(m.SITES)).to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.3))
    im = heatmap(ax, values, DIV, -0.5, 0.5, [m.SITE_LABELS[s] for s in m.SITES], [task_label(t) for t in tasks], xrot=35)
    ax.axhline(sum(t in m.SCHAEDE_TASKS for t in tasks) - 0.5, color=INK, linewidth=0.8)
    cb = colorbar(fig, im, ax, "(L − R) / (L + R), median", extend="both")
    cb.ax.text(2.6, 1.0, "left\nstronger", transform=cb.ax.transAxes, fontsize=6.5, va="top", color=INK2)
    cb.ax.text(2.6, 0.0, "right\nstronger", transform=cb.ax.transAxes, fontsize=6.5, va="bottom", color=INK2)
    ax.set_title("Left/right asymmetry (blank: below 10% of reference on both sides, or a side excluded)", loc="left")
    save(fig, out, "F08_asymmetry")


def session_coactivation(ds: m.Dataset) -> np.ndarray:
    mats = [m.channel_corr(np.concatenate([ds.norm(i) for i in group.index])) for _, group in ds.ok.groupby("session")]
    with warnings.catch_warnings():  # excluded channels are NaN in every session
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack([c for c in mats if c is not None]), axis=0)


def fig_coactivation(a: Analysis, out: Path) -> np.ndarray:
    corr = session_coactivation(a.ds)
    valid = np.flatnonzero(np.isfinite(np.diag(corr)))
    sub = corr[np.ix_(valid, valid)]
    dist = np.clip(1 - np.nan_to_num(sub, nan=0.0), 0, 2)
    np.fill_diagonal(dist, 0)
    order = valid[leaves_list(linkage(squareform(dist, checks=False), method="average"))]
    labels = [chan_label(a.ds, c) for c in np.array(m.CHANNELS)[order]]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.82, DOUBLE_COL * 0.74))
    im = heatmap(ax, corr[np.ix_(order, order)], DIV, -1, 1, labels, labels)
    colorbar(fig, im, ax, "Frame-level Pearson r, median over sessions")
    omitted = f"; {len(m.CHANNELS) - len(valid)} excluded channel omitted" if len(valid) < len(m.CHANNELS) else ""
    ax.set_title(f"Channel co-activation, clustered (blocks = shared activity or crosstalk{omitted})", loc="left")
    save(fig, out, "F09_coactivation")
    return corr


def fig_oculi_oris(a: Analysis, out: Path) -> None:
    panels = [("(a) All usable clips", a.oo), ("(b) sync_grade 0 clips only", a.oo_grade0)]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.4), sharey=True)
    rng = np.random.default_rng(0)
    tasks = list(m.OCULI_ORIS_TASKS)
    for ax, (title, oo) in zip(axes, panels):
        for i, task in enumerate(tasks):
            d = 100 * oo.loc[oo["task"] == task, "d"].dropna().to_numpy()
            ax.scatter(i - 0.08 + rng.uniform(-0.17, 0.17, len(d)), d, s=9, color=SERIES[0], alpha=0.55,
                       edgecolors=SURFACE, linewidths=0.4, zorder=2)
            if len(d):
                ax.plot([i - 0.32, i + 0.16], [np.median(d)] * 2, color=INK, linewidth=1.6, zorder=3)
            expected = m.OCULI_ORIS_EXPECTED.get(task)
            text = f"{int((np.sign(d) == expected).sum())}/{len(d)}" if expected else "no expected\nsign"
            ax.text(i, 1.0, text, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7,
                    color=INK if expected else INK2, fontweight="bold" if expected else "normal")
            if a.mmm_oo is not None:
                dm = 100 * a.mmm_oo.loc[a.mmm_oo["task"] == task, "d"].to_numpy()
                ax.scatter(np.full(len(dm), i + 0.3), dm, s=26, marker="D", color=SERIES[1], edgecolors=SURFACE,
                           linewidths=0.6, zorder=4)
        ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
        ax.set_xticks(range(len(tasks)), tasks)
        ax.set_xlim(-0.6, len(tasks) - 0.4)
        ax.set_title(title, loc="left", pad=16)
        ax.tick_params(axis="x", top=False)
    axes[0].set_ylabel("Orb. oculi − Orb. oris (% of reference)")
    axes[0].text(0.01, 0.98, "oculi stronger ↑", transform=axes[0].transAxes, fontsize=6.5, color=INK2, va="top")
    axes[0].text(0.01, 0.02, "oris stronger ↓", transform=axes[0].transAxes, fontsize=6.5, color=INK2, va="bottom")
    handles = [Line2D([], [], marker="o", linestyle="", markersize=4, color=SERIES[0], label=f"{a.ds.name} session"),
               Line2D([], [], color=INK, linewidth=1.6, label="Median")]
    if a.mmm_oo is not None:
        handles.append(Line2D([], [], marker="D", linestyle="", markersize=4.5, color=SERIES[1], label="MMM session (pid 99)"))
    fig.tight_layout(w_pad=0.8)
    legend_below(fig, handles, len(handles))
    save(fig, out, "F10_oculi_oris_headline")


CROSS_MAIN = (("OOC", "OOR"), ("eye", "pucker"))
CROSS_LR = (("OOC_l", "OOC_r"), ("eyeBlinkLeft", "eyeBlinkRight"))


def cross_matrix(cross: pd.DataFrame, rows, cols) -> np.ndarray:
    return np.array([[cross[f"{r}~{c}"].median() for c in cols] for r in rows])


def fig_blendshape_crosscheck(a: Analysis, out: Path) -> None:
    panels = [(f"(a) {a.ds.name}: muscle vs movement", a.cross, CROSS_MAIN, ("Orb. oculi", "Orb. oris"), ("Eye blink/squint", "Pucker/funnel")),
              (f"(b) {a.ds.name}: side check", a.cross, CROSS_LR, ("Orb. oculi L", "Orb. oculi R"), ("eyeBlinkLeft", "eyeBlinkRight"))]
    if a.mmm_cross is not None:
        panels.append(("(c) MMM: muscle vs movement", a.mmm_cross, CROSS_MAIN, ("Orb. oculi", "Orb. oris"), ("Eye blink/squint", "Pucker/funnel")))
    fig, axes = plt.subplots(1, len(panels), figsize=(DOUBLE_COL, 2.5), gridspec_kw={"wspace": 0.55})
    for ax, (title, cross, (rows, cols), ylabels, xlabels) in zip(np.atleast_1d(axes), panels):
        mat = cross_matrix(cross, rows, cols)
        im = heatmap(ax, mat, DIV, -1, 1, xlabels, ylabels, xrot=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=8, color=ink_on(im.cmap(im.norm(mat[i, j]))))
        diag = (cross[f"{rows[0]}~{cols[0]}"] > cross[f"{rows[0]}~{cols[1]}"]) & (cross[f"{rows[1]}~{cols[1]}"] > cross[f"{rows[1]}~{cols[0]}"])
        ax.set_title(f"{title}\ndiagonal wins in {int(diag.sum())}/{len(cross)} sessions", loc="left", fontsize=7.5)
    colorbar(fig, im, np.atleast_1d(axes).tolist(), "Median Pearson r over sessions")
    save(fig, out, "F11_blendshape_crosscheck")


def fig_maya_heatmap(a: Analysis, out: Path) -> None:
    rows = m.maya_rows()
    values = np.full((len(rows), len(m.SITES)), np.nan)
    for i, (_, task, _) in enumerate(rows):
        if task:
            values[i] = group_matrix(a.grp, [task], m.SITES)[0]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 5.0))
    labels = [f"{row_label(e, t, r)}  [{e.au_recipe}]" for e, t, r in rows]
    im = heatmap(ax, values, SEQ, 0, 2.0, [m.SITE_LABELS[s] for s in m.SITES], labels, xrot=35)
    for i, (_, task, _) in enumerate(rows):
        if not task:
            ax.add_patch(Rectangle((-0.5, i - 0.5), len(m.SITES), 1, facecolor="none", edgecolor=HAIRLINE, hatch="////", linewidth=0))
        elif task != m.REST_TASK:
            j = int(np.nanargmax(values[i]))
            ax.text(j, i, pct(values[i, j]), ha="center", va="center", fontsize=6.5, color=ink_on(im.cmap(im.norm(values[i, j]))))
    for j, site in enumerate(m.SITES):
        if site in m.MODEL_DRIVEN_SITES:
            ax.plot(j, -0.85, marker="v", markersize=4.5, color=INK, clip_on=False)
    ax.set_ylim(len(rows) - 0.5, -1.15)
    ax.annotate("▼ driven by the face model's solver   ·   hatched: no matching MaM task   ·   number: strongest site",
                (0, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points", fontsize=6.5, color=INK2, va="bottom")
    colorbar(fig, im, ax, "Group median, % of reference", extend="max",
             format=matplotlib.ticker.FuncFormatter(lambda v, _: pct(v)))
    ax.set_title("Human recruitment per Maya expression", loc="left", pad=14)
    save(fig, out, "F12_maya_site_heatmap")


def fig_maya_muscles(a: Analysis, out: Path) -> None:
    rows = [r for r in mapped_rows() if r[1] != m.REST_TASK]
    fig, axes = grid_axes(len(rows), 5, 4.6, sharex=True)
    rng = np.random.default_rng(1)
    x = np.arange(len(m.MUSCLES_OF_INTEREST))
    for ax, (expr, task, role) in zip(axes, rows):
        sub = a.part[a.part["task"] == task]
        for k, site in enumerate(m.MUSCLES_OF_INTEREST):
            v = 100 * sub.loc[sub["site"] == site, "norm"].dropna().to_numpy()
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=MUSCLE_COLOR[site], alpha=0.65,
                       edgecolors=SURFACE, linewidths=0.3, zorder=2)
            if len(v):
                ax.plot([k - 0.3, k + 0.3], [np.median(v)] * 2, color=INK, linewidth=1.4, zorder=3)
        ax.axhline(0, color=HAIRLINE, linewidth=0.8, zorder=0)
        ax.set_title(row_label(expr, task, role), loc="left", fontsize=7)
        ax.set_xticks(x, [m.SITE_LABELS[s] for s in m.MUSCLES_OF_INTEREST], rotation=55, ha="right", rotation_mode="anchor", fontsize=6.5)
        ax.tick_params(axis="x", top=False)
        ax.set_xlim(-0.6, len(x) - 0.4)
    for ax in axes[len(rows):]:
        ax.axis("off")
    for ax in axes[::5]:
        ax.set_ylabel("% of reference")
    fig.suptitle("Muscles of interest per expression: one dot per participant, bar = median",
                 x=0.01, ha="left", fontsize=8.5, fontweight="bold")
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    save(fig, out, "F13_maya_muscles_of_interest")


def fig_reliability_ceiling(a: Analysis, out: Path) -> None:
    rows = [r for r in mapped_rows() if r[1] != m.REST_TASK]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.62, 3.6))
    y = np.arange(len(rows))
    within = a.rel[a.rel["kind"] == "within"]
    between = a.rel[a.rel["kind"] == "between"]
    boxes(ax, [within.loc[within["task"] == t, "rho"].to_numpy() for _, t, _ in rows], y, SERIES[0], width=0.5, vert=False)
    med_between = [between.loc[between["task"] == t, "rho"].median() for _, t, _ in rows]
    ax.scatter(med_between, y, marker="D", s=18, color=SERIES[1], edgecolors=SURFACE, linewidths=0.5, zorder=4)
    ax.set_yticks(y, [row_label(e, t, r) for e, t, r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(-1.02, 1.02)
    ax.axvline(0, color=HAIRLINE, linewidth=0.8, zorder=0)
    ax.set_xlabel(r"Spearman $\rho$, 10 sites (masseter excluded)")
    ax.tick_params(axis="y", right=False)
    ax.legend(handles=[Patch(facecolor=(*to_rgb(SERIES[0]), 0.12), edgecolor=SERIES[0], label="Same participant, other session (ceiling)"),
                       Line2D([], [], marker="D", linestyle="", markersize=4.5, color=SERIES[1], label="Different participants, median")],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=1, fontsize=6.5)
    ax.set_title("Agreement ceiling for a model-vs-human comparison", loc="left")
    save(fig, out, "F14_reliability_ceiling")


# Expected sign of (Lateraler Frontalis - Levator Labii Superioris): brow raising vs upper-lip raising.
LABEL_TASKS = {"Forehead": 1, "surprised": 1, "Snarl": -1, "disgusted": -1}


def frontalis_minus_levator(a: Analysis, released: bool) -> pd.DataFrame:
    """Session x task: normalized lateral frontalis minus levator labii, optionally with schaede labels as released."""
    f = a.ds.features
    f = f[f["task"].isin(LABEL_TASKS) & f["site"].isin(["Lateraler Frontalis", "Levator Labii Superioris"])]
    if released and a.ds.schaede_swaps:
        rename = {x: y for pair in a.ds.schaede_swaps for x, y in (pair, pair[::-1])}
        f = f.assign(site=np.where(f["mode"] == "schaede", f["site"].replace(rename), f["site"]))
    w = m.session_values(m.site_values(f)).pivot_table(index=["session", "task"], columns="site", values="norm")
    return (w["Lateraler Frontalis"] - w["Levator Labii Superioris"]).rename("d").reset_index()


def scan_summary(scan: pd.DataFrame) -> pd.DataFrame:
    return scan.groupby("candidate")["gain"].agg(share=lambda g: (g > 0).mean(), median="median", n="size")


def fig_label_check(a: Analysis, out: Path) -> None:
    fixed = bool(a.ds.schaede_swaps)
    released = scan_summary(a.scan_released)
    top = released["share"].idxmax() if len(released) else None
    panels = [("(a) Swap scan, labels as released", released)]
    if fixed:
        panels.append(("(b) After the swap fix", scan_summary(a.scan_loaded)))
    fig = plt.figure(figsize=(DOUBLE_COL, 6.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.35], height_ratios=[1, 0.95], wspace=0.42, hspace=0.48)
    for k, (title, summ) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        rest = summ.drop(index=top, errors="ignore")
        ax.scatter(rest["median"], 100 * rest["share"], s=10, color=MUTED, alpha=0.7, edgecolors=SURFACE, linewidths=0.4)
        if top in summ.index:
            xy = (summ.at[top, "median"], 100 * summ.at[top, "share"])
            ax.scatter(*xy, s=28, color=SERIES[1], edgecolors=SURFACE, linewidths=0.6, zorder=3)
            ax.annotate(top.replace(" ↔ ", " ↔\n"), xy, xytext=(0.05, 0.86), textcoords="axes fraction", fontsize=6.5,
                        va="top", arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.6, shrinkB=3))
        ax.axhline(50, color=HAIRLINE, linewidth=0.8, zorder=0)
        ax.axvline(0, color=HAIRLINE, linewidth=0.8, zorder=0)
        ax.set_ylim(-3, 103)
        ax.set_xlabel("Median gain in agreement r")
        ax.set_title(title, loc="left", fontsize=7.5)
        if k == 0:
            ax.set_ylabel("Sessions improved (%)")
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(2)
    loaded, as_released = frontalis_minus_levator(a, released=False), frontalis_minus_levator(a, released=True)
    for i, task in enumerate(LABEL_TASKS):
        d = 100 * loaded.loc[loaded["task"] == task, "d"].dropna().to_numpy()
        ax.scatter(i - 0.13 + rng.uniform(-0.1, 0.1, len(d)), d, s=7, color=SERIES[0], alpha=0.6, edgecolors=SURFACE, linewidths=0.3, zorder=2)
        ax.text(i, 1.0, f"{int((np.sign(d) == LABEL_TASKS[task]).sum())}/{len(d)}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.5, fontweight="bold")
        if fixed and task in m.SCHAEDE_TASKS:
            r = 100 * as_released.loc[as_released["task"] == task, "d"].dropna().to_numpy()
            ax.scatter(i + 0.13 + rng.uniform(-0.1, 0.1, len(r)), r, s=7, color=SERIES[1], alpha=0.6, edgecolors=SURFACE, linewidths=0.3, zorder=2)
    ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(LABEL_TASKS)), [task_label(t) for t in LABEL_TASKS], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(LABEL_TASKS) - 0.4)
    ax.tick_params(axis="x", top=False)
    ax.set_ylabel("Frontalis lat. − Lev. labii sup. (%)")
    ax.set_title("(c) Brow raise vs upper-lip raise", loc="left", fontsize=7.5, pad=12)

    ax = fig.add_subplot(gs[1, :])
    order = list(m.DISPLAY_CHANNELS)
    x = np.arange(len(order))
    sims = [(a.sim_released, SERIES[1], -0.19)] + ([(a.sim_loaded, SERIES[0], 0.19)] if fixed else [])
    for sim, color, dx in sims:
        boxes(ax, [sim.loc[sim["channel"] == c, "similarity"].to_numpy() for c in order], x + dx, color, width=0.32)
    ax.axhline(0, color=HAIRLINE, linewidth=0.8, zorder=0)
    ax.set_xticks(x, [chan_label(a.ds, c) for c in order], rotation=55, ha="right", rotation_mode="anchor", fontsize=6.5)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-1.05, 1.05)
    ax.tick_params(axis="x", top=False)
    ax.set_ylabel("Profile r")
    ax.set_title("(d) Does each channel co-activate with the others the same way in schaede and emotion clips?", loc="left", fontsize=7.5)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=3.5, color=SERIES[1], label="Schaede labels as released")]
    if fixed:
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=3.5, color=SERIES[0], label="As analysed (after the swap fix)"))
    legend_below(fig, handles, len(handles))
    save(fig, out, "F15_label_check")


# ----------------------------------------------------------------------------
# Tables and findings
# ----------------------------------------------------------------------------

def write_tables(a: Analysis, out: Path) -> pd.DataFrame:
    a.ds.clips.drop(columns=["emg", "MediaPipe"]).to_csv(out / "clips_inventory.csv", index=False)
    a.ds.features.to_csv(out / "features_long.csv", index=False)
    m.maya_recruitment(a.sess).to_csv(out / "human_recruitment_maya.csv", index=False)
    within = a.rel[a.rel["kind"] == "within"].groupby("task")["rho"].median().rename("ceiling_rho_within")
    between = a.rel[a.rel["kind"] == "between"].groupby("task")["rho"].median().rename("rho_between")
    rows = []
    for expr, task, role in m.maya_rows():
        head = {"maya_expression": expr.name, "au_recipe": expr.au_recipe, "pairing": expr.pairing, "role": role, "mam_task": task}
        sub = a.grp[a.grp["task"] == task]
        if task is None or sub.empty:
            rows.append(head)
            continue
        for _, r in sub.iterrows():
            rows.append({**head, "site": r["site"], "model_driven": r["site"] in m.MODEL_DRIVEN_SITES,
                         "median": r["median"], "q25": r["q25"], "q75": r["q75"], "n_participants": r["n_participants"],
                         "n_sessions": int(a.sess[(a.sess["task"] == task) & (a.sess["site"] == r["site"])]["norm"].notna().sum()),
                         "ceiling_rho_within": within.get(task, np.nan), "rho_between": between.get(task, np.nan)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "human_recruitment_maya_summary.csv", index=False)
    return summary


def raw_rise(ds: m.Dataset, task: str, channels: list[str]) -> pd.DataFrame:
    """Session x channel: hold median minus rest median in raw units, imitations combined by median."""
    cols = [m.CHANNELS.index(c) for c in channels]
    rows = []
    for idx, row in ds.ok[ds.ok["task"] == task].iterrows():
        rise = np.median(ds.raw[idx][ds.hold[idx]][:, cols], axis=0) - ds.baselines[row["session"]].rest[cols]
        rows.append({"session": row["session"], **dict(zip(channels, rise))})
    return pd.DataFrame(rows, columns=["session", *channels]).groupby("session").median()


def kn(oo: pd.DataFrame, task: str, col: str = "d") -> tuple[int, int]:
    d = oo.loc[oo["task"] == task, col].dropna()
    return int((np.sign(d) == m.OCULI_ORIS_EXPECTED[task]).sum()), len(d)


def write_findings(a: Analysis, summary: pd.DataFrame, corr: np.ndarray, out: Path) -> None:
    ds, lines = a.ds, []
    add = lines.append
    clips = ds.clips
    add(f"# {ds.name} sEMG findings")
    add("")
    add(f"Generated by `analyze_MaM.py`. Hold windows: **{ds.hold_source}**.")
    if ds.hold_source == "emg":
        add("The windows come from the summed EMG itself, which biases them toward the signal being measured.")
        add("Rerun on the full release with `--hold mediapipe` before quoting numbers.")
    add("Units: % of reference, i.e. rest-subtracted hold median divided by the session's 95th-percentile schaede contraction per channel.")
    add("")
    add("## Inventory")
    add("")
    ok = clips[clips["status"] == "ok"]
    add(f"{clips['pid'].nunique()} participants, {clips['session'].nunique()} sensor sessions, {len(clips)} sensor clips, of which {len(ok)} are usable.")
    for (session, status), n in clips[clips["status"] != "ok"].groupby(["session", "status"]).size().items():
        add(f"- `{session}`: {n} clip{'s' if n != 1 else ''} {status.replace('_', ' ')}.")
    neg = ok["n_negative"].sum() / max(ok["n_frames"].sum() * len(m.CHANNELS), 1)
    add(f"Negative envelope samples (resampling ringing, clamped to 0): {100 * neg:.3f}% of all samples.")
    add("")
    add("## Channel quality")
    add("")
    rest, p95 = session_levels(ds)
    dyn = p95 / rest
    flat = [(session, c, rest.at[session, c]) for session in dyn.index for c in m.CHANNELS if dyn.at[session, c] < 1.5]
    add(f"{len(flat)} session-channels barely rise above rest (schaede p95 < 1.5 x rest median); their normalized values are noise-dominated.")
    for session, c, rest in flat:
        add(f"- `{session}` {c} (rest median {rest:.0f} µV)")
    add("")
    add("## Channel corrections")
    add("")
    if ds.schaede_swaps or ds.excluded:
        pairs = ", ".join(f"{x} ↔ {y}" for x, y in ds.schaede_swaps)
        add(f"Swapped back in schaede clips on load: {pairs or 'none'} (`mam_data.SCHAEDE_COLUMN_SWAPS`).")
        add(f"Excluded from every normalized value: {', '.join(ds.excluded) or 'none'} (`mam_data.EXCLUDED_CHANNELS`); its site rests on the other side.")
        add("Run with `--as-released` to reproduce the uncorrected numbers.")
    else:
        add("This run uses the channels exactly as released.")
    released = scan_summary(a.scan_released).sort_values("share", ascending=False)
    if len(released) > 1:
        add(f"Swap scan over {len(released)} candidates (every site pair, every left/right pair), labels as released: "
            f"{released.index[0]} improves schaede/emotion agreement in {pct(released['share'].iloc[0])} of {released['n'].iloc[0]} sessions "
            f"(median gain {released['median'].iloc[0]:+.2f}); the runner-up, {released.index[1]}, in {pct(released['share'].iloc[1])}.")
    if ds.schaede_swaps:
        after = scan_summary(a.scan_loaded)
        add(f"After the swap fix no candidate improves agreement in more than {pct(after['share'].max())} of sessions ({after['share'].idxmax()}).")
    loaded, as_released = frontalis_minus_levator(a, False), frontalis_minus_levator(a, True)
    for task, sign in LABEL_TASKS.items():
        d = loaded.loc[loaded["task"] == task, "d"].dropna()
        r = as_released.loc[as_released["task"] == task, "d"].dropna()
        word = "frontalis lat. > lev. labii" if sign > 0 else "lev. labii > frontalis lat."
        extra = f" (as released: {int((np.sign(r) == sign).sum())}/{len(r)})" if ds.schaede_swaps and task in m.SCHAEDE_TASKS else ""
        add(f"- {task_label(task)}: {word} in {int((np.sign(d) == sign).sum())}/{len(d)} sessions{extra}.")
    sim = a.sim_loaded.groupby("channel")["similarity"].median().sort_values()
    add(f"Schaede/emotion profile similarity per channel {'after the swap fix' if ds.schaede_swaps else 'as released'} (median r over sessions): lowest {sim.index[0]} {sim.iloc[0]:+.2f}, "
        f"next {sim.index[1]} {sim.iloc[1]:+.2f}, median channel {sim.median():+.2f}.")
    for channel in ds.excluded:
        right = channel[:-1] + ("r" if channel.endswith("l") else "l")
        for task in ("Lip-Pucker", "Depress-Lip"):
            rise = raw_rise(ds, task, [channel, right])
            add(f"- {task}: {channel} rises more above rest than {right} in {int((rise[channel] > rise[right]).sum())}/{len(rise)} sessions (raw µV).")
    add("Worth reporting to the dataset authors, together with the MMM oculi/oris swap below.")
    add("")
    add("## Orbicularis oculi vs oris labels")
    add("")
    for task in m.OCULI_ORIS_EXPECTED:
        k, n = kn(a.oo, task)
        k0, n0 = kn(a.oo_grade0, task)
        kl, _ = kn(a.oo, task, "d_l")
        kr, _ = kn(a.oo, task, "d_r")
        word = "oculi > oris" if m.OCULI_ORIS_EXPECTED[task] > 0 else "oris > oculi"
        add(f"- {task}: {word} in {k}/{n} sessions (left {kl}, right {kr}); sync_grade 0 only: {k0}/{n0}.")
    pattern = m.swap_pattern(a.oo).value_counts()
    add(f"Across {' and '.join(m.SWAP_TASKS)}: {pattern.get('expected', 0)} sessions as expected, "
        f"{pattern.get('swapped', 0)} fully swapped, {pattern.get('mixed', 0)} mixed.")
    if a.mmm_oo is not None:
        mmm_pattern = m.swap_pattern(a.mmm_oo)
        add(f"MMM (pid 99): {', '.join(f'`{s}` {p}' for s, p in mmm_pattern.items())}.")
        for task in m.SWAP_TASKS:
            vals = a.mmm_oo.loc[a.mmm_oo["task"] == task, "d"]
            add(f"- MMM {task}: oculi − oris = {', '.join(pct(v) for v in vals)}.")
        if pattern.get("swapped", 0) == 0 and (mmm_pattern == "swapped").all():
            add(f"Verdict: no {ds.name} session shows the swap, while both MMM sessions do, so the swap is specific to the MMM release (column labels or electrode placement), not to the protocol.")
    add("")
    if a.cross is not None:
        add("## Blendshape cross-check")
        add("")
        for name, cross in [(ds.name, a.cross), ("MMM", a.mmm_cross)]:
            if cross is None:
                continue
            mat = cross_matrix(cross, *CROSS_MAIN)
            add(f"{name}: r(oculi, eye) {mat[0, 0]:+.2f}, r(oculi, pucker) {mat[0, 1]:+.2f}, r(oris, eye) {mat[1, 0]:+.2f}, r(oris, pucker) {mat[1, 1]:+.2f} (median over sessions).")
        lr = cross_matrix(a.cross, *CROSS_LR)
        add(f"Side check: r(OOC_l, eyeBlinkLeft) {lr[0, 0]:+.2f}, r(OOC_l, eyeBlinkRight) {lr[0, 1]:+.2f}, r(OOC_r, eyeBlinkLeft) {lr[1, 0]:+.2f}, r(OOC_r, eyeBlinkRight) {lr[1, 1]:+.2f}.")
        add("Both eyes usually close together, so the side check is weak; only a clearly larger off-diagonal suggests a side swap.")
        add("")
    add("## Recruitment per Maya expression")
    add("")
    add("| Maya expression | MaM task | Top site | Median | Driven in model | Sessions led by it | Ceiling rho (same person) | rho (other people) |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for expr, task, role in m.maya_rows():
        name = f"{expr.name} (alt)" if role == "alt" else expr.name
        sub = summary[(summary["maya_expression"] == expr.name) & (summary["mam_task"] == task)].dropna(subset=["median"]) if task else pd.DataFrame()
        if sub.empty or task == m.REST_TASK:
            note = "baseline" if task == m.REST_TASK else "no MaM task"
            add(f"| {name} | {task or '-'} | {note} | | | | | |")
            continue
        best = sub.loc[sub["median"].idxmax()]
        if best["median"] < 0.1:
            add(f"| {name} | {task} | none above 10% | {pct(best['median'])} | | | {best['ceiling_rho_within']:.2f} | {best['rho_between']:.2f} |")
            continue
        share = (a.top.loc[a.top["task"] == task, "top_site"] == best["site"]).mean()
        add(f"| {name} | {task} | {m.SITE_LABELS[best['site']]} | {pct(best['median'])} | {'yes' if best['model_driven'] else 'no'} | "
            f"{pct(share)} | {best['ceiling_rho_within']:.2f} | {best['rho_between']:.2f} |")
    add("")
    add("## Co-activation")
    add("")
    pairs = [(corr[i, j], m.CHANNELS[i], m.CHANNELS[j]) for i in range(len(m.CHANNELS)) for j in range(i + 1, len(m.CHANNELS))
             if m.CHANNEL_SIDE[i] == m.CHANNEL_SIDE[j] and np.isfinite(corr[i, j])]
    add("Strongest same-side channel pairs (frame-level r, median over sessions), the first place to look for crosstalk:")
    for r, c1, c2 in sorted(pairs, reverse=True)[:6]:
        add(f"- {c1} ~ {c2}: {r:+.2f}")
    add("")
    add("## Notes")
    add("")
    add("- Cheese is AU6 in the piano UI's EXPRESSION_COMMANDS but AU6+12+25 ('Cheese (maximum)') in run_pipeline_update.py; it is paired with Smile-Open here.")
    add("- Schaede clips repeat each movement about three times; time courses align to the first hold onset.")
    add("- The model renders Sad as AU4 only, while human sad recruits AU1+4+15, so Sad is a fair pairing at best.")
    add("- Open Mouth, Lower the Brow, Cheek Raise and AU12L + AU2L have no MaM task and keep empty rows in the join table.")
    add("- Session design (same day or not) is unknown, so session-to-session agreement is reported as between-session, not test-retest.")
    (out / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  findings.md")


# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=HERE / "MaM_Public", help="dataset root holding MaM.csv (default: MaM_Public next to this script)")
    parser.add_argument("--name", default="MaM", choices=["MaM", "MMM"], help="index file stem in --root")
    parser.add_argument("--out", type=Path, default=HERE / "visualization" / "analyze_MaM", help="output folder")
    parser.add_argument("--hold", default="mediapipe", choices=["mediapipe", "emg"], help="what defines the hold window")
    parser.add_argument("--mmm-root", type=Path, help="MMM release root (holding MMM.csv), drawn as a reference in the oculi/oris figures")
    parser.add_argument("--as-released", action="store_true",
                        help="use the channels exactly as released, without mam_data's swap fix and exclusions")
    args = parser.parse_args(argv)

    apply_style()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.name} from {args.root} (hold: {args.hold})")
    try:
        ds = m.build_dataset(args.root, args.name, args.hold, corrections=not args.as_released)
        mmm = m.build_dataset(args.mmm_root, "MMM", args.hold) if args.mmm_root else None
    except FileNotFoundError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    a = analyse(ds, mmm)
    print(f"Writing to {args.out}")
    fig_inventory(a, args.out)
    fig_channel_quality(a, args.out)
    fig_task_channel_heatmap(a, args.out)
    fig_top_site_share(a, args.out)
    fig_time_courses(a, args.out)
    fig_exemplar_raw(a, args.out)
    fig_reliability(a, args.out)
    fig_asymmetry(a, args.out)
    corr = fig_coactivation(a, args.out)
    fig_oculi_oris(a, args.out)
    if a.cross is not None:
        fig_blendshape_crosscheck(a, args.out)
    else:
        print("  F11_blendshape_crosscheck skipped: needs --hold mediapipe")
    fig_maya_heatmap(a, args.out)
    fig_maya_muscles(a, args.out)
    fig_reliability_ceiling(a, args.out)
    fig_label_check(a, args.out)
    summary = write_tables(a, args.out)
    write_findings(a, summary, corr, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
