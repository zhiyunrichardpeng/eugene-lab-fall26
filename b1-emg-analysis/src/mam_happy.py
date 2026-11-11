"""MaM "happy" clips on a common 0 -> 1 expression-progress clock.

For each usable happy clip (sensor session, sync_grade <= 1, non-empty EMG):
  1. Normalise each channel: a(t) = (envelope(t) - rest) / ref
       rest = median envelope before the smile starts (this clip)
       ref  = 95th percentile of that channel over all of the session's standardized
              ("schaede") movement clips, i.e. a near-maximal effort for that person and site
     So 0 = rest and ~1 = near-maximal contraction; microvolts and skin impedance drop out.
  2. Smile progress from the video: s(t) = mean(mouthSmileLeft, mouthSmileRight) (MediaPipe).
     Progress runs 0 (smile onset) -> 1 (smile apex), using its running maximum so it never
     goes backwards.
  3. Read a(t) at the first frame where progress reaches each grid value (first passage).

Caveat that comes with the data: MaM envelopes use a ~500 ms RMS window, so fine timing is blurred.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "mam" / "MaM_Public"
GRID = np.linspace(0.0, 1.0, 21)
MIN_SMILE_RISE = 0.2     # skip clips where the tracked smile barely changes


def index():
    df = pd.read_csv(ROOT / "MaM.csv", dtype={"pid": str})
    d3 = pd.read_csv(ROOT / "MaM_3DMM.csv", dtype={"pid": str})[["index", "MediaPipe"]]
    df = df.merge(d3, on="index", how="left")
    return df


def session_refs(df, q=95):
    """{(pid, rec): Series of per-channel reference levels from the session's schaede clips}."""
    refs = {}
    sch = df[(df["mode"] == "schaede") & df.emg.notna()]
    for (pid, rec), g in sch.groupby(["pid", "rec"]):
        frames = [pd.read_csv(ROOT / p, index_col=0) for p in g.emg]
        frames = [f for f in frames if len(f)]
        if frames:
            refs[(pid, rec)] = pd.concat(frames).quantile(q / 100.0)
    return refs


def first_passage(x, grid):
    """Frame index where the non-decreasing x first reaches each grid value."""
    return np.searchsorted(x, grid - 1e-12, side="left").clip(0, len(x) - 1)


def clip_curves(row, ref):
    """Normalised EMG of one clip at each progress grid point: DataFrame (grid x channel), or None."""
    emg = pd.read_csv(ROOT / row.emg, index_col=0)
    if emg.empty or not isinstance(row.MediaPipe, str):
        return None
    mp = pd.read_csv(ROOT / row.MediaPipe)
    n = min(len(emg), len(mp))
    emg, mp = emg.iloc[:n], mp.iloc[:n]

    s = mp[["mouthSmileLeft", "mouthSmileRight"]].mean(axis=1).rolling(3, center=True, min_periods=1).mean().to_numpy()
    apex = int(np.argmax(s))
    if apex < 3:
        return None
    base = np.percentile(s[: apex + 1], 10)
    rise = s[apex] - base
    if rise < MIN_SMILE_RISE:
        return None
    onset = int(np.flatnonzero(s[: apex + 1] <= base + 0.05 * rise)[-1])

    rest = emg.iloc[max(0, onset - 15): onset + 1].median()
    a = (emg - rest) / ref.reindex(emg.columns)
    prog = np.maximum.accumulate(np.clip((s[onset: apex + 1] - base) / rise, 0, 1))
    idx = onset + first_passage(prog, GRID)
    out = a.iloc[idx].reset_index(drop=True)
    out.index = GRID
    return out


def all_curves():
    """Long table: pid, clip_id, grid, channel, a  (one row per clip x grid point x channel)."""
    df = index()
    happy = df[(df.task == "happy") & df.emg.notna() & (df.sync_grade <= 1)]
    refs = session_refs(df[(df.pid + "_" + df.rec).isin(set(happy.pid + "_" + happy.rec))])
    rows, skipped = [], 0
    for r in happy.itertuples():
        ref = refs.get((r.pid, r.rec))
        c = None if ref is None else clip_curves(r, ref)
        if c is None:
            skipped += 1
            continue
        c = c.rename_axis("grid").reset_index().melt(id_vars="grid", var_name="channel", value_name="a")
        c["pid"], c["clip_id"] = r.pid, r.index
        rows.append(c)
    return pd.concat(rows, ignore_index=True), skipped
