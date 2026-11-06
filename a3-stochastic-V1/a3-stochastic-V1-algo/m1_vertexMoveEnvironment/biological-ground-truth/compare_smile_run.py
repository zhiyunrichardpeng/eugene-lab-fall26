"""
=============================================================================
Ziva smile chain vs MaM human smile
=============================================================================
Analyzes a chain of m1_vertexMove smile evaluation runs, such as happy-job
(neutral -> AU6 -> AU12 -> AU25, the parts of Happy = AU6+AU12+AU25), and
compares it with the MaM sEMG features of the matching smile task.
Each stage is one results folder holding config.json, log0.txt and
deltaB_temp.pkl; a single run is a chain of one stage.

A keyframe is a single-AU pose (AU6) or a combined one (AU6_12_25). A stage
between single-AU keyframes can undo the previous one (AU6 -> AU12 relaxes the
cheek raise), so the chain's travel is summed over all stages as the solver
performed it.

MaM normalizes every task against Face-At-Rest, so the human "change" is the
task value itself. The model reports joint displacement and MaM reports sEMG,
so the figure compares patterns, not magnitudes:
    stages       model joint travel per driven part in each stage
    recruitment  human EMG per site during the task
    share        which driven muscle carries the smile: chain joint travel vs human EMG
    right/left   side balance: chain oculi and oris joint travel vs human EMG

Model left/right follows the run's own naming: the side of the orbicularis
oculi "l" joints is left.

Usage (from this folder, after analyze_MaM.py has written features_long.csv):
    python compare_smile_run.py happy-job
    python compare_smile_run.py job_261019-194513/261019-194523 --task Smile-Closed

Output (default visualization/compare_smile_run/, do not commit it, MaM is under a
data-use agreement): <chain>_vs_MaM_<task>.png / .pdf, the panels above, and
<chain>_vs_MaM_<task>_all_muscles.png / .pdf, the share of the whole face per muscle
(every human site, the model's parts beside them)
=============================================================================
"""

from __future__ import annotations

import argparse
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
from analyze_MaM import DOUBLE_COL, HAIRLINE, INK, INK2, MUSCLE_COLOR, MUTED, SURFACE, apply_style, save, task_label
from compare_ziva_run import (CHANGE_SITES, FRONTALIS, SHARE_COLUMNS, Run, categorical_axis, change_share, human_values, load_run,
                              model_point, right_left, strip)

HERE = Path(__file__).resolve().parent
SMILE_TASKS = ("happy", "Smile-Closed", "Smile-Open")
SIDE_SITES = ("Orbicularis Oculi", "Zygomaticus", "Orbicularis Oris")
# Log names of the per-part keyframe distances: part -> (initial label, final label).
DISTANCE_LABELS = {
    "oris": ("initial distance", "distance"),
    "oculi_l": ("initial_ooc_l_distance", "distance_ooc_l"),
    "oculi_r": ("initial_ooc_r_distance", "distance_ooc_r"),
    "frontalis": ("initial_frontalis_distance", "distance_frontalis"),
}
# Stage bar segments: label, travel parts, color, hatch.
STAGE_SEGMENTS = (
    ("Orb. oris", ("oris",), MUSCLE_COLOR["Orbicularis Oris"], None),
    ("Orb. oculi l", ("oculi_l",), MUSCLE_COLOR["Orbicularis Oculi"], None),
    ("Orb. oculi r", ("oculi_r",), MUSCLE_COLOR["Orbicularis Oculi"], "////"),
    ("Frontalis", ("frontalis",), MUSCLE_COLOR["Lateraler Frontalis"], None),
)


@dataclass(frozen=True)
class Stage:
    run: Run
    label: str  # "AU0 to AU6", from the log's task line
    steps: int
    success: bool
    distance: dict[str, tuple[float, float]]  # part -> (start, end) distance to the target keyframe
    side_travel: dict[str, float]  # "l"/"r" -> summed oris joint displacement on that side (mm)


def log_number(log: str, pattern: str) -> float:
    found = re.search(pattern + r":?\s*([-\d.eE+]+)", log)
    if found is None:
        raise ValueError(f"log0.txt has no '{pattern}'")
    return float(found.group(1))


def stage_label(run_dir: Path, task: str) -> str:
    """'dataV22_AU0_neutral to dataV22_AU6_12_25' -> 'AU0 to AU6+12+25'; flags a folder name that names other AUs."""
    poses = [re.search(r"AU(\d+L?(?:_\d+L?)*)", p) for p in re.split(r"\s+to\s+", task)]
    if len(poses) == 2 and all(poses):
        label = " to ".join("AU" + p.group(1).replace("_", "+") for p in poses)
        aus = [au for p in poses for au in p.group(1).split("_")]
    else:
        label, aus = task, re.findall(r"AU(\d+L?)", task)
    suffix = run_dir.name[run_dir.name.find("AU"):] if "AU" in run_dir.name else ""
    folder = re.findall(r"\d+L?", suffix)  # "AU0-AU6" and "AU12-25" both name two AUs
    if folder and folder != aus:
        print(f"warning: {run_dir.name} logs task '{task}', its folder name says {suffix}", file=sys.stderr)
        label += f" (folder: {suffix})"
    return label


def load_stage(run_dir: Path) -> Stage:
    run = load_run(run_dir)
    log = (run_dir / "log0.txt").read_text()
    done = re.search(r"done at No\. (\d+) step, success flag\s*(\d)", log)
    if done is None:
        raise ValueError(f"{run_dir}: log0.txt has no evaluator result line")
    final = log[log.index("final distance_jaw"):]
    distance = {part: (log_number(log, start), log_number(final, rf"\b{end}"))
                for part, (start, end) in DISTANCE_LABELS.items()}

    with open(run_dir / "deltaB_temp.pkl", "rb") as fh:
        pkl = pickle.load(fh)
    joints = np.asarray(pkl["x_joint_"], dtype=float)
    delta = np.asarray(pkl["delta_b0_conca"], dtype=float)[: joints.size].reshape(-1, 3)
    left_sign = np.sign(np.mean(np.asarray(pkl["x_ooc_l_joint_"], dtype=float)[:, 0]))
    side_travel = {side: float(np.linalg.norm(delta[np.sign(joints[:, 0]) == sign], axis=1).sum())
                   for side, sign in (("l", left_sign), ("r", -left_sign))}
    return Stage(run, stage_label(run_dir, run.task), int(done.group(1)), done.group(2) == "1", distance, side_travel)


def find_stages(paths: list[Path]) -> list[Path]:
    """Run folders (holding log0.txt) under the given paths, in run-timestamp order."""
    runs = {p.parent for path in paths for p in ([path / "log0.txt"] if (path / "log0.txt").exists() else path.rglob("log0.txt"))}
    if not runs:
        raise FileNotFoundError(f"no run folder with log0.txt under {', '.join(map(str, paths))}")
    return sorted(runs, key=lambda p: p.name)


def order_chain(stages: list[Stage]) -> list[Stage]:
    """Orders stages so each starts at the pose the previous one ends at, from the logged tasks."""
    poses = {id(s): [p.strip() for p in re.split(r"\s+to\s+", s.run.task)] for s in stages}
    ends = {poses[id(s)][1] for s in stages}
    starts = [s for s in stages if poses[id(s)][0] not in ends]
    if len(starts) != 1:
        raise ValueError(f"stages do not form one chain: {', '.join(s.run.task for s in stages)}")
    chain, rest = starts, [s for s in stages if s is not starts[0]]
    while rest:
        nxt = [s for s in rest if poses[id(s)][0] == poses[id(chain[-1])][1]]
        if len(nxt) != 1:
            raise ValueError(f"chain breaks after '{chain[-1].run.task}': remaining {', '.join(s.run.task for s in rest)}")
        chain.append(nxt[0])
        rest.remove(nxt[0])
    return chain


def chain_total(stages: list[Stage]) -> tuple[Run, dict[str, float]]:
    travel = {part: sum(s.run.travel[part] for s in stages) for part in stages[0].run.travel}
    side = {k: sum(s.side_travel[k] for s in stages) for k in "lr"}
    return Run("chain", " | ".join(s.label for s in stages), travel), side


def model_ratios(total: Run, side: dict[str, float]) -> dict[str, float]:
    """Model right/left per SIDE_SITES site it drives."""
    return {"Orbicularis Oculi": total.travel["oculi_r"] / total.travel["oculi_l"] if total.travel["oculi_l"] else np.nan,
            "Orbicularis Oris": side["r"] / side["l"] if side["l"] else np.nan}


def task_norm(site: pd.DataFrame, task: str) -> pd.DataFrame:
    """Participant x site: EMG in % of reference for one task (already relative to Face-At-Rest)."""
    sub = site[site["task"] == task]
    return (100 * sub.pivot_table(index="pid", columns="site", values="norm")).dropna(axis=1, how="all")


def report(stages: list[Stage], norm: pd.DataFrame, share: pd.DataFrame, ratios: dict[str, pd.Series], task: str) -> None:
    total, side = chain_total(stages)
    for k, s in enumerate(stages, 1):
        print(f"Stage {k} {s.run.name}: {s.run.task}")
        print(f"  {s.steps} steps, {'success' if s.success else 'NOT reached'}")
        for part, (start, end) in s.distance.items():
            if max(start, end) > 1e-3:  # parts the keyframes leave in place log ~1e-6
                print(f"  {part:10s} distance {start:7.2f} -> {end:6.2f}, joint travel {s.run.travel[part]:8.2f}")
    print("Chain total joint travel:")
    for part, value in total.travel.items():
        print(f"  {part:10s} {value:8.2f}")

    print(f"Humans, {task_label(task)} (n = {len(norm)}): median EMG, % of reference")
    for s, value in norm.median().reindex([s for s in m.SITES if s in norm]).items():
        print(f"  {m.SITE_LABELS[s]:20s} {value:6.1f}")
    model_share = total.share()
    print("Share of the driven sites (human median vs chain):")
    for col, label, _ in SHARE_COLUMNS:
        print(f"  {label.splitlines()[0]:12s} {100 * share[col].median():5.0f}%  model {100 * model_share[col]:5.0f}%")
    model = model_ratios(total, side)
    for s, ratio in ratios.items():
        extra = f", model {model[s]:.2f}" if s in model else ""
        print(f"  {m.SITE_LABELS[s]} right / left (median, n = {len(ratio)}): {ratio.median():.2f}{extra}")


def stage_panel(ax, stages: list[Stage]) -> None:
    """One stacked bar per stage: joint travel per driven part (the jaw has no MaM counterpart)."""
    y = np.arange(len(stages))[::-1]
    left = np.zeros(len(stages))
    for label, parts, color, hatch in STAGE_SEGMENTS:
        width = np.array([sum(s.run.travel[p] for p in parts) for s in stages])
        ax.barh(y, width, left=left, height=0.6, color=color, hatch=hatch, edgecolor=SURFACE, linewidth=0.5, label=label)
        left += width
    ax.set_yticks(y, [f"{k}  {s.label.replace(' (', chr(10) + '(')}\n{s.steps} steps{'' if s.success else ', failed'}"
                      for k, s in enumerate(stages, 1)], fontsize=6.5)
    ax.tick_params(axis="y", right=False)
    ax.set_xlim(0, left.max() * 1.9)  # room for the legend right of the longest bar
    ax.set_xlabel("Joint travel (mm)")
    ax.set_title(f"Model: {len(stages)} stage{'s' if len(stages) > 1 else ''}", loc="left", fontsize=7.5)
    ax.legend(loc="lower right", fontsize=6, handlelength=1.2, borderaxespad=0.3)


def clip_top(ax, norm: pd.DataFrame, sites: list[str]) -> None:
    """Keep the y range on the bulk of the data; outliers above it sit on the top edge with their value."""
    top = 1.15 * max(norm[s].quantile(0.95) for s in sites)
    for k, s in enumerate(sites):
        for value in norm[s][norm[s] > top]:
            ax.scatter(k, 0.95 * top, marker="^", s=14, color=MUSCLE_COLOR.get(s, MUTED), zorder=4)
            ax.text(k + 0.15, 0.95 * top, f"{value:.0f}", va="center", fontsize=6, color=INK2)
    ax.set_ylim(top=top)


def fig_chain(name: str, stages: list[Stage], norm: pd.DataFrame, share: pd.DataFrame, ratios: dict[str, pd.Series],
              task: str, out: Path) -> None:
    rng = np.random.default_rng(1)
    total, side = chain_total(stages)
    fig = plt.figure(figsize=(DOUBLE_COL, 5.2))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1.25, 0.75])

    stage_panel(fig.add_subplot(grid[0, 0]), stages)

    ax = fig.add_subplot(grid[0, 1:])
    sites = [s for s in m.SITES if s in norm]
    for k, s in enumerate(sites):
        strip(ax, k, norm[s].dropna().to_numpy(), MUSCLE_COLOR.get(s, MUTED), rng)
    clip_top(ax, norm, sites)
    ax.axhline(0, color=HAIRLINE, linewidth=0.8, zorder=0)
    ax.set_ylabel("EMG (% of ref.)")
    ax.set_title(f"Humans, {task_label(task)}: recruitment per site  (n = {len(norm)})", loc="left", fontsize=7.5)
    categorical_axis(ax, [m.SITE_LABELS[s] + (" *" if s in CHANGE_SITES else "") for s in sites])
    ax.text(0.98, 0.96, "* site the model drives", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK2)

    ax = fig.add_subplot(grid[1, 0:2])
    model_share = total.share()
    for k, (col, _, color) in enumerate(SHARE_COLUMNS):
        strip(ax, k, 100 * share[col].to_numpy(), color, rng)
        model_point(ax, k, 100 * model_share[col], f"{100 * model_share[col]:.0f}%")
    ax.set_ylim(-5, 105)
    ax.set_ylabel("Share of the smile (%)")
    ax.set_title(f"{task_label(task)}: who carries the smile among the driven sites", loc="left", fontsize=7.5)
    categorical_axis(ax, [label for _, label, _ in SHARE_COLUMNS], room_right=0.35)

    ax = fig.add_subplot(grid[1, 2])
    model = model_ratios(total, side)
    for k, (s, ratio) in enumerate(ratios.items()):
        strip(ax, k, ratio.to_numpy(), MUSCLE_COLOR[s], rng)
        if np.isfinite(model.get(s, np.nan)):
            model_point(ax, k, model[s], f"{model[s]:.2f}")
    ax.set_yscale("log")
    ax.set_ylim(0.22, 4.6)
    ax.set_yticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.axhline(1, color=HAIRLINE, linewidth=0.8, zorder=0)
    ax.set_ylabel("Right / left\n(above 1: right stronger)")
    counts = {len(r) for r in ratios.values()}
    same_n = len(counts) == 1
    ax.set_title("Side balance" + (f"  (n = {counts.pop()})" if same_n else ""), loc="left", fontsize=7.5)
    categorical_axis(ax, [m.SITE_LABELS[s] + ("" if same_n else f"\nn = {len(r)}") for s, r in ratios.items()],
                     room_right=0.35)

    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=4.5, markerfacecolor=INK2, markeredgecolor=SURFACE, label="Participant"),
        Line2D([], [], color=INK, linewidth=1.4, label="Human median"),
        Line2D([], [], linestyle="none", marker="D", markersize=5, markerfacecolor=SURFACE, markeredgecolor=INK,
               label="Ziva model (chain total)" if len(stages) > 1 else "Ziva model"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Ziva smile {'chain' if len(stages) > 1 else 'run'} {name} ({' | '.join(s.label for s in stages)}) vs MaM humans", x=0.01, y=0.995,
                 ha="left", va="top", fontsize=8.5, fontweight="bold")
    fig.text(0.01, 0.962, "Model: net joint travel summed over the stages. Humans: sEMG relative to Face-At-Rest; share of |EMG| "
             "over the four driven sites; right/left from raw µV.",
             ha="left", va="top", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 0.955), h_pad=0.8, w_pad=0.6)
    save(fig, out, f"{name}_vs_MaM_{task}")


def all_muscle_shares(norm: pd.DataFrame, total: Run) -> pd.DataFrame:
    """Muscle x (human, model): share of the whole face, in %.

    Human: each participant's |EMG| over all sites (frontalis lat. + med. as one, like the model's
    single frontalis mesh) as fractions of 1, averaged over participants so the column sums to 100.
    Model: joint travel per part over the oris, oculi and frontalis parts; the jaw is a rotation,
    not mm, so it stays out. NaN where the model has no counterpart.
    """
    human = norm.abs()
    human.insert(0, "Frontalis", human[list(FRONTALIS)].sum(axis=1, min_count=2))
    human = human[["Frontalis"] + [s for s in m.SITES if s in human and s not in FRONTALIS]].dropna()
    human = 100 * human.div(human.sum(axis=1), axis=0).mean()
    model = pd.Series({k: 100 * v for k, v in total.share().items()})
    return pd.DataFrame({"human": human, "model": model.reindex(human.index)})


def fig_all_muscles(name: str, stages: list[Stage], norm: pd.DataFrame, task: str, out: Path) -> None:
    total, _ = chain_total(stages)
    shares = all_muscle_shares(norm, total)
    print("Share of the whole face, % (human mean vs model):")
    for muscle, row in shares.iterrows():
        model = "not in model" if np.isnan(row["model"]) else f"{row['model']:5.0f}%"
        print(f"  {m.SITE_LABELS.get(muscle, muscle):20s} {row['human']:5.0f}%  model {model}")

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, 3.6))
    y = np.arange(len(shares))[::-1]
    h = 0.38
    ax.barh(y + h / 2, shares["human"], height=h, color=MUTED, label=f"Humans, mean (n = {len(norm)})")
    ax.barh(y - h / 2, shares["model"].fillna(0), height=h, color=INK, label="Ziva model")
    for k, (muscle, row) in zip(y, shares.iterrows()):
        ax.text(row["human"] + 1, k + h / 2, f"{row['human']:.0f}%", va="center", fontsize=6, color=INK2)
        model = "not in model" if np.isnan(row["model"]) else f"{row['model']:.0f}%"
        ax.text((0 if np.isnan(row["model"]) else row["model"]) + 1, k - h / 2, model, va="center", fontsize=6,
                color=INK2)
    ax.set_yticks(y, ["Frontalis (lat. + med.)" if s == "Frontalis" else m.SITE_LABELS[s] for s in shares.index])
    ax.tick_params(axis="y", right=False, length=0)
    ax.set_xlim(0, 1.15 * np.nanmax(shares.to_numpy()) + 8)
    ax.set_xlabel("Share of the whole face (%)")
    ax.set_title(f"{task_label(task)}: every muscle, humans vs model ({name})", loc="left", fontsize=7.5)
    ax.legend(loc="lower right", fontsize=6.5)
    fig.text(0.01, 0.01, "Humans: |sEMG| relative to Face-At-Rest. Model: joint travel; the jaw (a rotation) is left out.",
             ha="left", va="bottom", fontsize=6, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, f"{name}_vs_MaM_{task}_all_muscles")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+",
                        help="stage run folders, or a job folder holding them (e.g. happy-job); stages are chained by their logged from/to poses")
    parser.add_argument("--features", type=Path, default=HERE / "visualization" / "analyze_MaM" / "features_long.csv",
                        help="features_long.csv written by analyze_MaM.py")
    parser.add_argument("--task", default="happy", choices=SMILE_TASKS, help="MaM task matching the chain's end expression")
    parser.add_argument("--name", help="chain name in the title and file name (default: the first path's folder name)")
    parser.add_argument("--out", type=Path, default=HERE / "visualization" / "compare_smile_run", help="output folder")
    args = parser.parse_args(argv)

    try:
        stages = order_chain([load_stage(run_dir) for run_dir in find_stages(args.runs)])
        site, channel = human_values(args.features)
    except (FileNotFoundError, KeyError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    norm = task_norm(site, args.task)
    share = change_share(norm[list(CHANGE_SITES)].dropna())
    ratios = {s: right_left(channel, args.task, s) for s in SIDE_SITES}
    name = args.name or args.runs[0].resolve().name

    apply_style()
    args.out.mkdir(parents=True, exist_ok=True)
    report(stages, norm, share, ratios, args.task)
    print(f"Writing to {args.out}")
    fig_chain(name, stages, norm, share, ratios, args.task, args.out)
    fig_all_muscles(name, stages, norm, args.task, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
