"""
=============================================================================
Ziva solver run vs MaM human recruitment
=============================================================================
Compares one m1_vertexMove evaluation run (a results folder holding config.json,
log0.txt and deltaB_temp.pkl) with the MaM sEMG features written by analyze_MaM.py.

The model reports joint displacement and MaM reports sEMG, so the figure compares
patterns, never magnitudes. One row per target task:
    share     which muscle carries the transition: model joint travel vs human |EMG change|
    change    human EMG change from the start task to the target task, signed
plus the orbicularis oculi right/left balance in the first target task.

Usage (from this folder, after analyze_MaM.py has written features_long.csv):
    python compare_ziva_run.py 261011-221704
    python compare_ziva_run.py RUN --start surprised --targets Eye-Tight Eye-Gentle

Output (default visualization/compare_ziva_run/, never commit it, MaM is under a
data-use agreement): ziva_<run>_vs_MaM.png / .pdf
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import mam_data as m
from analyze_MaM import DOUBLE_COL, HAIRLINE, INK, INK2, MUSCLE_COLOR, SURFACE, apply_style, save, task_label

HERE = Path(__file__).resolve().parent

# Joint-offset layout of the m1_vertexMove envs (base_action_prime / delta_b0_conca), in order:
# config.json key holding the joint count of each part, 3 values per joint.
RUN_PARTS = (
    ("oris", "joint_num"),
    ("jaw", "joint_rotate_num"),  # rotation angles, not a translation
    ("oculi_l", "joint_ooc_l_num"),
    ("oculi_r", "joint_ooc_r_num"),
    ("frontalis", "joint_frontalis_num"),
)
FRONTALIS = ("Lateraler Frontalis", "Medialer Frontalis")
# Human sites the solver drives (the model has one frontalis mesh for both frontalis sites).
CHANGE_SITES = ("Orbicularis Oculi", "Orbicularis Oris") + FRONTALIS
SHARE_COLUMNS = (
    ("Orbicularis Oculi", "Orb. oculi", MUSCLE_COLOR["Orbicularis Oculi"]),
    ("Orbicularis Oris", "Orb. oris", MUSCLE_COLOR["Orbicularis Oris"]),
    ("Frontalis", "Frontalis\n(lat. + med.)", MUSCLE_COLOR["Lateraler Frontalis"]),
)
MODEL_MARKER = dict(marker="D", s=26, facecolor=SURFACE, edgecolor=INK, linewidth=1.0, zorder=4)


@dataclass(frozen=True)
class Run:
    name: str
    task: str
    travel: dict[str, float]  # part -> summed net displacement over its joints (mm)

    def share(self) -> dict[str, float]:
        """Model share per SHARE_COLUMNS muscle; the jaw has no MaM counterpart and is left out."""
        parts = {"Orbicularis Oculi": self.travel["oculi_l"] + self.travel["oculi_r"],
                 "Orbicularis Oris": self.travel["oris"], "Frontalis": self.travel["frontalis"]}
        total = sum(parts.values())
        return {k: v / total for k, v in parts.items()}


def load_run(run_dir: Path) -> Run:
    config = json.loads((run_dir / "config.json").read_text())
    match = re.search(r"This task:\s*(.+)", (run_dir / "log0.txt").read_text())
    with open(run_dir / "deltaB_temp.pkl", "rb") as fh:
        delta = np.asarray(pickle.load(fh)["delta_b0_conca"], dtype=float)
    sizes = [3 * int(config[key]) for _, key in RUN_PARTS]
    if sum(sizes) != len(delta):
        raise ValueError(f"{run_dir}: delta_b0_conca has {len(delta)} values, config.json implies {sum(sizes)}")
    travel = {}
    for (part, _), chunk in zip(RUN_PARTS, np.split(delta, np.cumsum(sizes)[:-1])):
        travel[part] = float(np.linalg.norm(chunk.reshape(-1, 3), axis=1).sum())
    return Run(run_dir.name, match.group(1).strip() if match else "task not logged", travel)


def human_values(features_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Participant x task x site (L/R mean) and participant x task x channel, as in analyze_MaM."""
    features = pd.read_csv(features_csv, dtype={"pid": str, "rec": str})
    site = m.participant_values(m.session_values(m.site_values(features)))
    chan = features.drop(columns="site").rename(columns={"channel": "site"})
    channel = m.participant_values(m.session_values(chan))
    return site, channel


def paired_change(site: pd.DataFrame, start: str, target: str) -> pd.DataFrame:
    """Participant x site: target minus start, in % of reference, for participants with both tasks."""
    wide = site[site["site"].isin(CHANGE_SITES)].pivot_table(index=["pid", "site"], columns="task", values="norm")
    return (100 * (wide[target] - wide[start])).unstack("site").dropna()


def change_share(change: pd.DataFrame) -> pd.DataFrame:
    """Participant x SHARE_COLUMNS muscle: share of the summed |change| over the driven sites."""
    absolute = change.abs()
    share = absolute.div(absolute.sum(axis=1), axis=0)
    share["Frontalis"] = share[list(FRONTALIS)].sum(axis=1)
    return share[[c for c, _, _ in SHARE_COLUMNS]].dropna()


def right_left(channel: pd.DataFrame, task: str, site: str) -> pd.Series:
    """Participant: right/left raw rise above rest (µV); normalized values are ~1 per side by construction."""
    sub = channel[(channel["task"] == task) & channel["site"].str.startswith(f"{site}_")]
    wide = sub.pivot_table(index="pid", columns="site", values="delta_uv")
    left, right = wide[f"{site}_l"], wide[f"{site}_r"]
    return (right / left).where((left > 0) & (right > 0)).dropna()


def strip(ax, x: float, values: np.ndarray, color: str, rng: np.random.Generator) -> None:
    """F13 look: one dot per participant, black bar at the median."""
    ax.scatter(x + rng.uniform(-0.18, 0.18, len(values)), values, s=9, color=color, alpha=0.7,
               edgecolors=SURFACE, linewidths=0.4, zorder=2)
    ax.plot([x - 0.3, x + 0.3], [np.median(values)] * 2, color=INK, linewidth=1.4, zorder=3)


def model_point(ax, x: float, y: float, label: str) -> None:
    """Model diamond, its value written right of the median bar so it clears the jittered dots."""
    ax.scatter(x, y, **MODEL_MARKER)
    ax.text(x + 0.34, y, label, va="center", fontsize=6.5, color=INK2, zorder=5)


def categorical_axis(ax, labels, room_right: float = 0.0) -> None:
    """room_right widens the last slot so a model_point label fits inside the frame."""
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", rotation_mode="anchor", fontsize=6.5)
    ax.tick_params(axis="x", top=False)
    ax.set_xlim(-0.6, len(labels) - 0.4 + room_right)


def fig_comparison(run: Run, site: pd.DataFrame, channel: pd.DataFrame, start: str, targets: list[str], out: Path) -> None:
    rng = np.random.default_rng(1)
    fig = plt.figure(figsize=(DOUBLE_COL, 2.35 * len(targets) + 0.4))
    grid = fig.add_gridspec(len(targets), 3, width_ratios=[1, 1.25, 0.5])
    model_share = run.share()

    for row, target in enumerate(targets):
        change = paired_change(site, start, target)
        share = change_share(change)

        ax = fig.add_subplot(grid[row, 0])
        for k, (col, _, color) in enumerate(SHARE_COLUMNS):
            strip(ax, k, 100 * share[col].to_numpy(), color, rng)
            model_point(ax, k, 100 * model_share[col], f"{100 * model_share[col]:.0f}%")
        ax.set_ylim(-5, 105)
        ax.set_ylabel("Share of the change (%)")
        ax.set_title(f"{task_label(target)}: who carries the change", loc="left", fontsize=7.5)
        categorical_axis(ax, [label for _, label, _ in SHARE_COLUMNS], room_right=0.35)

        ax = fig.add_subplot(grid[row, 1])
        for k, s in enumerate(CHANGE_SITES):
            strip(ax, k, change[s].to_numpy(), MUSCLE_COLOR[s], rng)
        ax.axhline(0, color=HAIRLINE, linewidth=0.8, zorder=0)
        ax.set_ylabel("EMG change (% of ref.)")
        ax.set_title(f"Humans, {task_label(start)} → {task_label(target)}  (n = {len(change)})", loc="left", fontsize=7.5)
        categorical_axis(ax, [m.SITE_LABELS[s] for s in CHANGE_SITES])

    ratio = right_left(channel, targets[0], "Orbicularis Oculi")
    model_ratio = run.travel["oculi_r"] / run.travel["oculi_l"]
    ax = fig.add_subplot(grid[0, 2])
    strip(ax, 0, ratio.to_numpy(), MUSCLE_COLOR["Orbicularis Oculi"], rng)
    model_point(ax, 0, model_ratio, f"{model_ratio:.2f}")
    ax.set_yscale("log")
    ax.set_ylim(0.4, 2.5)
    ax.set_yticks([0.5, 1, 2], ["0.5", "1", "2"])
    ax.minorticks_off()
    ax.axhline(1, color=HAIRLINE, linewidth=0.8, zorder=0)
    ax.set_ylabel("Orb. oculi right / left\n(above 1: right stronger)")
    ax.set_title(f"{task_label(targets[0])}: right / left", loc="left", fontsize=7.5)
    ax.set_xticks([0], [f"n = {len(ratio)}"])
    ax.tick_params(axis="x", top=False)
    ax.set_xlim(-0.6, 0.9)

    legend = fig.add_subplot(grid[1:, 2]) if len(targets) > 1 else None
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=4.5, markerfacecolor=INK2, markeredgecolor=SURFACE, label="Participant"),
        Line2D([], [], color=INK, linewidth=1.4, label="Human median"),
        Line2D([], [], linestyle="none", marker="D", markersize=5, markerfacecolor=SURFACE, markeredgecolor=INK, label="Ziva model"),
    ]
    if legend is not None:
        legend.axis("off")
        legend.legend(handles=handles, loc="center left", frameon=True, borderaxespad=0)
    else:
        fig.legend(handles=handles, loc="lower center", ncols=3)

    fig.suptitle(f"Ziva run {run.name} ({run.task}) vs MaM humans", x=0.01, y=0.995, ha="left", va="top",
                 fontsize=8.5, fontweight="bold")
    fig.text(0.01, 0.962, "Model: net joint travel. Humans: sEMG, share of |change| over the four driven sites; right/left from raw µV. "
             "Compare patterns, not magnitudes.", ha="left", va="top", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.8, w_pad=0.6)
    save(fig, out, f"ziva_{run.name}_vs_MaM")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="results folder of one run (config.json, log0.txt, deltaB_temp.pkl)")
    parser.add_argument("--features", type=Path, default=HERE / "visualization" / "analyze_MaM" / "features_long.csv",
                        help="features_long.csv written by analyze_MaM.py")
    parser.add_argument("--start", default="surprised", help="MaM task matching the run's start keyframe")
    parser.add_argument("--targets", nargs="+", default=["Eye-Tight", "Eye-Gentle"],
                        help="MaM tasks matching the run's target keyframe; the first also gives the right/left panel")
    parser.add_argument("--out", type=Path, default=HERE / "visualization" / "compare_ziva_run", help="output folder")
    args = parser.parse_args(argv)

    unknown = [t for t in [args.start, *args.targets] if t not in m.TASKS]
    if unknown:
        print(f"error: unknown MaM task(s) {unknown}; choose from {list(m.TASKS)}", file=sys.stderr)
        return 2
    try:
        run = load_run(args.run)
        site, channel = human_values(args.features)
    except (FileNotFoundError, KeyError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    apply_style()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Run {run.name}: {run.task}")
    for part, value in run.travel.items():
        print(f"  {part:10s} joint travel {value:8.2f}")
    print(f"Writing to {args.out}")
    fig_comparison(run, site, channel, args.start, args.targets, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
