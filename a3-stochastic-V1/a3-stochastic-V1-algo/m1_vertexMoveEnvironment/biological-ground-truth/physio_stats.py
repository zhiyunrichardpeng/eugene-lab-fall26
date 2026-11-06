"""
=============================================================================
Physiological validation statistics: Ziva runs vs MaM human recruitment
=============================================================================
For every model run paired with a MaM task, compares which driven muscle
carries the movement (the recruitment share over orbicularis oculi,
orbicularis oris and frontalis) with the human sEMG shares, and reports:

    share        model share, human median [IQR], model percentile among humans
    similarity   cosine similarity and total variation distance (TVD, the
                 fraction of recruitment that would have to move between
                 muscles) from the model to each participant, median with a
                 participant bootstrap 95% CI
    reference    the same distances between pairs of participants (between-person
                 spread), and between two sessions of the same person (noise
                 ceiling)
    typicality   fraction of participants who are further from the others
                 than the model is (high: the model looks like a typical human)
    leader       does the model's largest driven muscle lead in humans too
                 (share of participants, Wilson 95% CI)
    coverage     share of the human whole-face |EMG change| that falls on the
                 muscles the model drives
    side         model right/left ratio, percentile among human ratios

The model reports joint travel and MaM reports sEMG, so only patterns are
compared, never magnitudes.

Usage (from this folder, after analyze_MaM.py has written features_long.csv):
    python physio_stats.py

Output (visualization/physio_stats/, never commit it, MaM is under a data-use
agreement): physio_stats.csv, one row per comparison
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import mam_data as m
from compare_smile_run import chain_total, find_stages, load_stage, model_ratios, order_chain
from compare_ziva_run import FRONTALIS, SHARE_COLUMNS, right_left

HERE = Path(__file__).resolve().parent
MUSCLES = [c for c, _, _ in SHARE_COLUMNS]
N_BOOT = 10_000


@dataclass(frozen=True)
class Comparison:
    label: str
    runs: tuple[str, ...]  # run folders, or a job folder chained by compare_smile_run
    start: str  # MaM task matching the start keyframe
    target: str  # MaM task matching the end keyframe


COMPARISONS = (
    Comparison("AU1 to AU7", ("261011-221704",), "surprised", "Eye-Tight"),
    Comparison("AU1 to AU7", ("261011-221704",), "surprised", "Eye-Gentle"),
    Comparison("AU0 to AU12, run A", ("261023-235713",), "Face-At-Rest", "Smile-Closed"),
    Comparison("AU0 to AU12, run B", ("job_261019-194513",), "Face-At-Rest", "Smile-Closed"),
    Comparison("AU0 to AU6+12+25", ("hpy",), "Face-At-Rest", "happy"),
    Comparison("AU0 to AU6+12+25", ("hpy",), "Face-At-Rest", "Smile-Open"),
    Comparison("AU0 to AU6 to AU12 to AU25", ("happy-job",), "Face-At-Rest", "happy"),
)


def change(site: pd.DataFrame, keys: list[str], start: str, target: str) -> pd.DataFrame:
    """Unit (participant or session) x site: |target - start| over all sites, units with both tasks."""
    wide = site.pivot_table(index=keys + ["site"], columns="task", values="norm")
    return (wide[target] - wide[start]).abs().unstack("site").dropna(how="all")


def shares(absolute: pd.DataFrame) -> pd.DataFrame:
    """Unit x MUSCLES: share of the summed |change| over the driven sites, frontalis lat. + med. as one."""
    driven = absolute[["Orbicularis Oculi", "Orbicularis Oris", *FRONTALIS]].dropna()
    out = pd.DataFrame({"Orbicularis Oculi": driven["Orbicularis Oculi"], "Orbicularis Oris": driven["Orbicularis Oris"],
                        "Frontalis": driven[list(FRONTALIS)].sum(axis=1)})
    return out.div(out.sum(axis=1), axis=0)[MUSCLES].dropna()


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def tvd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(a - b).sum(axis=-1)


def leave_one_out(h: np.ndarray, metric) -> np.ndarray:
    """Each participant against the mean of all the others."""
    n = len(h)
    others = (h.sum(axis=0) - h) / (n - 1)
    return metric(h, others)


def pairwise(h: np.ndarray, metric) -> np.ndarray:
    """Every pair of participants."""
    return np.array([metric(h[i], h[j]) for i, j in combinations(range(len(h)), 2)])


def boot_median(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(values), (N_BOOT, len(values)))
    return tuple(np.percentile(np.median(values[idx], axis=1), [2.5, 97.5]))


def session_ceiling(session_shares: pd.DataFrame, metric) -> np.ndarray:
    """Per participant: mean distance between pairs of their own sessions."""
    out = []
    for _, group in session_shares.groupby(level="pid"):
        if len(group) > 1:
            out.append(np.mean([metric(group.iloc[i].to_numpy(), group.iloc[j].to_numpy())
                                for i, j in combinations(range(len(group)), 2)]))
    return np.array(out)


def percentile(values: np.ndarray, x: float) -> float:
    """Mid-rank percentile of x among values (50 = human median)."""
    return 100 * (np.mean(values < x) + 0.5 * np.mean(values == x))


def wilson(k: int, n: int) -> tuple[float, float]:
    ci = binomtest(k, n).proportion_ci(method="wilson")
    return ci.low, ci.high


def evaluate(comp: Comparison, part_site: pd.DataFrame, sess_site: pd.DataFrame, channel: pd.DataFrame,
             rng: np.random.Generator) -> dict:
    stages = order_chain([load_stage(d) for d in find_stages([HERE / r for r in comp.runs])])
    total, side = chain_total(stages)
    model = np.array([total.share()[c] for c in MUSCLES])

    absolute = change(part_site, ["pid"], comp.start, comp.target)
    human = shares(absolute)
    h = human.to_numpy()
    sess = shares(change(sess_site, ["pid", "session"], comp.start, comp.target))

    row = {"comparison": comp.label, "mam": f"{comp.start} to {comp.target}", "n": len(h),
           "stages": len(stages), "all_reached": all(s.success for s in stages)}
    for k, c in enumerate(MUSCLES):
        q1, med, q3 = np.percentile(h[:, k], [25, 50, 75])
        row[f"{c}: model"] = model[k]
        row[f"{c}: human median"] = med
        row[f"{c}: human IQR"] = (q1, q3)
        row[f"{c}: model pct"] = percentile(h[:, k], model[k])

    for name, metric in (("cos", cosine), ("tvd", tvd)):
        to_model = metric(h, model[None, :])
        between = pairwise(h, metric)
        ceiling = session_ceiling(sess, metric)
        row[f"{name} model"] = np.median(to_model)
        row[f"{name} model CI"] = boot_median(to_model, rng)
        row[f"{name} between people"] = np.median(between)
        row[f"{name} same person"] = np.median(ceiling) if len(ceiling) else np.nan
        row["n ceiling"] = len(ceiling)

    # Typicality: the model's TVD to the human mean, against each participant's TVD to the others' mean.
    model_dev = tvd(model, h.mean(axis=0))
    row["typicality"] = np.mean(leave_one_out(h, tvd) > model_dev)

    leader = int(np.argmax(model))
    k = int((h.argmax(axis=1) == leader).sum())
    row["model leader"] = MUSCLES[leader]
    row["humans same leader"] = (k, len(h), *wilson(k, len(h)))

    full = absolute.dropna()
    driven = full[["Orbicularis Oculi", "Orbicularis Oris", *FRONTALIS]].sum(axis=1)
    coverage = (driven / full.sum(axis=1)).to_numpy()
    row["coverage"] = np.median(coverage)
    row["coverage CI"] = boot_median(coverage, rng)
    row["top human site"] = full.div(full.sum(axis=1), axis=0).mean().idxmax()

    ratios = model_ratios(total, side)
    for s, r in ratios.items():
        human_r = right_left(channel, comp.target, s).to_numpy()
        row[f"{s} R/L model"] = r
        usable = len(human_r) > 0 and np.isfinite(r)
        row[f"{s} R/L human median"] = np.median(human_r) if len(human_r) else np.nan
        row[f"{s} R/L model pct"] = percentile(human_r, r) if usable else np.nan
    return row


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def print_row(r: dict) -> None:
    print(f"\n=== {r['comparison']}  vs  MaM {r['mam']}  (n = {r['n']} participants, {r['stages']} stage(s)"
          f"{'' if r['all_reached'] else ', NOT all reached'})")
    print("  share of the driven sites      model   human median [IQR]   model percentile")
    for c in MUSCLES:
        q1, q3 = r[f"{c}: human IQR"]
        print(f"    {c:20s} {pct(r[f'{c}: model']):>12s}   {pct(r[f'{c}: human median']):>5s} [{pct(q1)}, {pct(q3)}]"
              f"{r[f'{c}: model pct']:>16.0f}")
    for name, label in (("cos", "cosine (1 = same)"), ("tvd", "TVD (0 = same)")):
        lo, hi = r[f"{name} model CI"]
        print(f"  {label:20s} model-human {r[f'{name} model']:.2f} [{lo:.2f}, {hi:.2f}]   "
              f"between people {r[f'{name} between people']:.2f}   same person {r[f'{name} same person']:.2f} "
              f"(n = {r['n ceiling']})")
    print(f"  typicality: {pct(r['typicality'])} of participants deviate more from the group than the model")
    k, n, lo, hi = r["humans same leader"]
    print(f"  model leader {r['model leader']}: leads in {k}/{n} participants ({pct(k / n)}, 95% CI {pct(lo)} to {pct(hi)})")
    lo, hi = r["coverage CI"]
    print(f"  coverage: driven muscles carry {pct(r['coverage'])} [{pct(lo)}, {pct(hi)}] of the human whole-face change; "
          f"top human site {r['top human site']}")
    for s in ("Orbicularis Oculi", "Orbicularis Oris"):
        print(f"  {s} right/left: model {r[f'{s} R/L model']:.2f}, human median {r[f'{s} R/L human median']:.2f}, "
              f"model percentile {r[f'{s} R/L model pct']:.0f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=HERE / "visualization" / "analyze_MaM" / "features_long.csv",
                        help="features_long.csv written by analyze_MaM.py")
    parser.add_argument("--out", type=Path, default=HERE / "visualization" / "physio_stats", help="output folder")
    args = parser.parse_args(argv)

    try:
        features = pd.read_csv(args.features, dtype={"pid": str, "rec": str})
    except FileNotFoundError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    sess_site = m.session_values(m.site_values(features))
    part_site = m.participant_values(sess_site)
    channel = m.participant_values(m.session_values(features.drop(columns="site").rename(columns={"channel": "site"})))

    rng = np.random.default_rng(0)
    rows = []
    for comp in COMPARISONS:
        try:
            rows.append(evaluate(comp, part_site, sess_site, channel, rng))
        except (FileNotFoundError, KeyError, ValueError) as err:
            print(f"error: {comp.label} vs {comp.target}: {err}", file=sys.stderr)
            return 2
        print_row(rows[-1])

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "physio_stats.csv", index=False)
    print(f"\nWrote {args.out / 'physio_stats.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
