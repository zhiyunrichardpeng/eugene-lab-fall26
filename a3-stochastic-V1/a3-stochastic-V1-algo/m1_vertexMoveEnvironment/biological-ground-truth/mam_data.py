"""
=============================================================================
MaM data layer: load, window and normalize facial sEMG for model comparison
=============================================================================
Reads the Mimics and Muscles (MaM) release, or MMM, its single-subject
companion with the same layout, through the dataset index CSVs.
Nothing here plots; analyze_MaM.py owns the figures.

Units:
    raw   RMS envelope in the recorder's microvolt-scale export units
    norm  (hold median - rest median) / reference contraction, per channel
          and session. Rest is the Face-At-Rest median, the reference is the
          95th percentile of (EMG - rest) over all schaede frames of the
          session (MaM has no MVC recording).
=============================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.stats import rankdata

FPS = 30

# Muscle sites in anatomical order, top of the face to the bottom.
SITES = (
    "Medialer Frontalis",
    "Lateraler Frontalis",
    "Corrugator Supercilii",
    "Depressor Supercilii Procerus",
    "Orbicularis Oculi",
    "Levator Labii Superioris",
    "Zygomaticus",
    "Orbicularis Oris",
    "Depressor Ang. Oris",
    "Mentalis",
    "Masseter",
)
SITE_LABELS = {
    "Medialer Frontalis": "Frontalis med.",
    "Lateraler Frontalis": "Frontalis lat.",
    "Corrugator Supercilii": "Corrugator",
    "Depressor Supercilii Procerus": "Depr. sup./procerus",
    "Orbicularis Oculi": "Orb. oculi",
    "Levator Labii Superioris": "Lev. labii sup.",
    "Zygomaticus": "Zygomaticus",
    "Orbicularis Oris": "Orb. oris",
    "Depressor Ang. Oris": "Depr. ang. oris",
    "Mentalis": "Mentalis",
    "Masseter": "Masseter",
}
# Column order inside every EMG CSV: alphabetical, _l before _r.
CHANNELS = tuple(sorted(f"{site}_{side}" for site in SITES for side in "lr"))
CHANNEL_SITE = np.array([c[:-2] for c in CHANNELS])
CHANNEL_SIDE = np.array([c[-1] for c in CHANNELS])
DISPLAY_CHANNELS = tuple(f"{site}_{side}" for site in SITES for side in "lr")

# Muscles the facial model work cares about (from sample_MaM.py).
# MaM records one Zygomaticus channel per side, so major and minor are not separable.
MUSCLES_OF_INTEREST = (
    "Orbicularis Oculi",
    "Orbicularis Oris",
    "Lateraler Frontalis",
    "Medialer Frontalis",
    "Zygomaticus",
)
# Structures the solver drives (frontalis, orbicularis oculi L/R, orbicularis oris, jaw).
# Masseter stands in for the jaw with the opposite sign, so rank metrics leave it out.
MODEL_DRIVEN_SITES = (
    "Medialer Frontalis",
    "Lateraler Frontalis",
    "Orbicularis Oculi",
    "Orbicularis Oris",
    "Masseter",
)
RANK_SITES = tuple(s for s in SITES if s != "Masseter")

REST_TASK = "Face-At-Rest"
SCHAEDE_TASKS = (
    "Face-At-Rest",
    "Forehead",
    "Eye-Gentle",
    "Eye-Tight",
    "Nose",
    "Smile-Closed",
    "Smile-Open",
    "Lip-Pucker",
    "Cheeks",
    "Snarl",
    "Depress-Lip",
)
EMOTION_TASKS = ("angry", "disgusted", "fearful", "happy", "sad", "surprised")
TASKS = SCHAEDE_TASKS + EMOTION_TASKS

# MediaPipe blendshapes whose summed activity defines each task's hold window.
HOLD_BLENDSHAPES = {
    "Forehead": ("browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
    "Eye-Gentle": ("eyeBlinkLeft", "eyeBlinkRight"),
    "Eye-Tight": ("eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight"),
    "Nose": ("noseSneerLeft", "noseSneerRight"),
    "Smile-Closed": ("mouthSmileLeft", "mouthSmileRight"),
    "Smile-Open": ("mouthSmileLeft", "mouthSmileRight", "jawOpen"),
    "Lip-Pucker": ("mouthPucker", "mouthFunnel"),
    "Cheeks": ("cheekPuff",),
    "Snarl": ("noseSneerLeft", "noseSneerRight", "mouthUpperUpLeft", "mouthUpperUpRight"),
    "Depress-Lip": ("mouthLowerDownLeft", "mouthLowerDownRight"),
    "angry": ("browDownLeft", "browDownRight"),
    "disgusted": ("noseSneerLeft", "noseSneerRight", "mouthUpperUpLeft", "mouthUpperUpRight"),
    "fearful": ("browInnerUp", "eyeWideLeft", "eyeWideRight", "mouthStretchLeft", "mouthStretchRight"),
    "happy": ("mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft", "cheekSquintRight"),
    "sad": ("browInnerUp", "mouthFrownLeft", "mouthFrownRight"),
    "surprised": ("browInnerUp", "browOuterUpLeft", "browOuterUpRight", "jawOpen", "eyeWideLeft", "eyeWideRight"),
}
HOLD_BLENDSHAPES["Nose-Pucker"] = HOLD_BLENDSHAPES["Nose"]  # MMM's label for the same exercise

# Blendshape groups kept per clip for the oculi/oris label check.
EYE_BLENDSHAPES = ("eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight")
PUCKER_BLENDSHAPES = ("mouthPucker", "mouthFunnel")
BLEND_SUMMARY = ("eye", "pucker", "eyeBlinkLeft", "eyeBlinkRight")


@dataclass(frozen=True)
class MayaExpression:
    name: str
    au_recipe: str
    tasks: tuple[str, ...]  # primary MaM task first, then alternates; empty when MaM has none
    pairing: str


# The 14 expressions of the Maya piano UI, with AU recipes from EMOTION_DROPDOWN_ITEMS and
# EXPRESSION_COMMANDS in b99-frame1to4-get-b1to3/d4 interactive interface of animation/h99-UI-mayascipt-V1.py.
MAYA_EXPRESSIONS = (
    MayaExpression("Neutral", "AU0", ("Face-At-Rest",), "exact"),
    MayaExpression("Smile", "AU12", ("Smile-Closed",), "good"),
    MayaExpression("Happy", "AU6+AU12+AU25", ("happy",), "good"),
    MayaExpression("Sad", "AU4", ("sad",), "fair: human sad is AU1+4+15"),
    MayaExpression("Frown", "AU4", ("angry",), "fair"),
    MayaExpression("Surprise", "AU1+AU2+AU25", ("surprised",), "good"),
    MayaExpression("Squint", "AU7", ("Eye-Gentle", "Eye-Tight"), "fair"),
    MayaExpression("Open Mouth", "AU25", (), "none"),
    MayaExpression("Raise the Brow", "AU1+AU2", ("Forehead",), "good"),
    MayaExpression("Lower the Brow", "AU4", (), "none"),
    MayaExpression("Cheek Raise", "AU6", (), "none: MaM Cheeks is cheek inflation, not AU6"),
    MayaExpression("Pucker (WUU)", "AU22", ("Lip-Pucker",), "fair"),
    MayaExpression("Cheese", "AU6 / AU6+AU12+AU25", ("Smile-Open",), "good; AU6 in the piano UI, AU6+12+25 in run_pipeline_update.py"),
    MayaExpression("AU12L + AU2L", "AU12L+AU2L", (), "none"),
)


def maya_rows() -> list[tuple[MayaExpression, str | None, str]]:
    """(expression, MaM task or None, role) rows, one per primary/alternate task."""
    rows = []
    for expr in MAYA_EXPRESSIONS:
        if not expr.tasks:
            rows.append((expr, None, "none"))
        for i, task in enumerate(expr.tasks):
            rows.append((expr, task, "primary" if i == 0 else "alt"))
    return rows


# Column pairs whose labels are swapped in a release's schaede clips, corrected on load.
# MaM: in schaede clips the column named Levator Labii Superioris leads Forehead (brow raise) in
# 77/77 sessions while Lateraler Frontalis stays at rest, and Lateraler Frontalis leads Snarl;
# the same sessions' emotion clips are anatomically right (surprise -> frontalis, disgust ->
# levator labii). Swapping the pair in schaede clips makes their channel co-activation match the
# emotion clips of the same session in 73/76 sessions; no other pair comes close (label_swap_scan,
# figure F15). Both sides are affected.
SCHAEDE_COLUMN_SWAPS = {"MaM": (("Lateraler Frontalis", "Levator Labii Superioris"),)}

# Channels whose released signal is unusable, excluded from every normalized value.
# MaM Depressor Supercilii Procerus_l: in schaede clips it carries mouth activity (it beats the
# right side in Lip-Pucker and Depress-Lip in 77/77 sessions, where a brow depressor should be
# silent) and its co-activation profile resembles Mentalis instead of its own emotion-clip
# profile (median similarity -0.14; profile_similarity, figure F15). No swap explains it. Its
# reference contraction comes from those schaede frames, so emotion values are skewed as well;
# the Depressor Supercilii Procerus site therefore rests on the right side alone.
EXCLUDED_CHANNELS = {"MaM": ("Depressor Supercilii Procerus_l",)}


def channel_permutation(site_pairs) -> np.ndarray:
    """Column order that swaps each (site_a, site_b) pair on both sides; an involution."""
    perm = np.arange(len(CHANNELS))
    for a, b in site_pairs:
        for side in "lr":
            i, j = CHANNELS.index(f"{a}_{side}"), CHANNELS.index(f"{b}_{side}")
            perm[i], perm[j] = perm[j], perm[i]
    return perm


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_index(root: Path, name: str) -> pd.DataFrame:
    """Sensor-session clips from {name}.csv, with the MediaPipe path joined from {name}_3DMM.csv."""
    idx = pd.read_csv(root / f"{name}.csv", dtype=str)
    idx = idx[idx["rec"].str.endswith("S")].copy()  # N sessions carry no sEMG
    mp_csv = root / f"{name}_3DMM.csv"
    if mp_csv.exists():
        mp = pd.read_csv(mp_csv, dtype=str, usecols=["index", "MediaPipe"])
        idx = idx.merge(mp, on="index", how="left")
    else:
        idx["MediaPipe"] = np.nan
    # identifier: {pid}_{rec}_{mode}_{task_seq}_{task}; MMM.csv has no task_seq column
    idx["task_seq"] = idx["index"].str.split("_").str[3].astype(int)
    idx["sync_grade"] = pd.to_numeric(idx["sync_grade"], errors="coerce")
    idx["session"] = idx["pid"] + "_" + idx["rec"]
    cols = ["index", "session", "pid", "rec", "mode", "task", "task_seq", "sync_grade", "emg", "MediaPipe"]
    return idx[cols].reset_index(drop=True)


def missing_blendshape_files(index: pd.DataFrame, root: Path) -> list[str]:
    paths = index["MediaPipe"].dropna()
    return [p for p in paths if not (root / p).exists()]


def load_emg(path: Path) -> np.ndarray | None:
    """frames x 22 float32 envelope, unclamped; None for an empty file."""
    # skipinitialspace: a few files on disk were re-saved with space-aligned columns
    emg = pd.read_csv(path, index_col=0, skipinitialspace=True)
    if emg.empty:
        return None
    if tuple(emg.columns) != CHANNELS:
        raise ValueError(f"{path}: unexpected channels {list(emg.columns)}")
    return emg.to_numpy(np.float32)


def load_blendshapes(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def blend_summary(blend: pd.DataFrame) -> np.ndarray:
    """frames x 4 (eye, pucker, eyeBlinkLeft, eyeBlinkRight) for the label check."""
    return np.column_stack([
        blend[list(EYE_BLENDSHAPES)].sum(axis=1),
        blend[list(PUCKER_BLENDSHAPES)].sum(axis=1),
        blend["eyeBlinkLeft"],
        blend["eyeBlinkRight"],
    ]).astype(np.float32)


# ----------------------------------------------------------------------------
# Windowing and normalization
# ----------------------------------------------------------------------------

def hold_mask(activity: np.ndarray) -> np.ndarray:
    """Frames at or above half the range of the 5-frame median-filtered activity.

    Half the range rather than half the maximum, because blendshapes and EMG both
    sit on a nonzero resting baseline.
    """
    smooth = median_filter(np.asarray(activity, dtype=np.float64), size=5, mode="nearest")
    lo, hi = smooth.min(), smooth.max()
    if not hi > lo:
        return np.ones(len(smooth), dtype=bool)
    return smooth >= lo + 0.5 * (hi - lo)


@dataclass(frozen=True)
class Baseline:
    rest: np.ndarray  # (22,) Face-At-Rest median, raw units
    ref: np.ndarray  # (22,) reference contraction above rest; NaN where the channel never rises


def session_baseline(rest_frames: np.ndarray, schaede_frames: np.ndarray) -> Baseline:
    rest = np.median(rest_frames, axis=0)
    ref = np.percentile(schaede_frames - rest, 95, axis=0)
    return Baseline(rest=rest, ref=np.where(ref > 1e-6, ref, np.nan))


@dataclass
class Dataset:
    name: str
    hold_source: str
    clips: pd.DataFrame  # inventory: one row per sensor clip, with status
    features: pd.DataFrame  # ok clips x channels, long format
    raw: dict[str, np.ndarray]  # index -> frames x 22, clamped at zero
    hold: dict[str, np.ndarray]  # index -> frames, bool
    blend: dict[str, np.ndarray]  # index -> frames x 4 (BLEND_SUMMARY); only where MediaPipe exists
    baselines: dict[str, Baseline]  # session -> rest/reference
    schaede_swaps: tuple = ()  # SCHAEDE_COLUMN_SWAPS pairs applied on load
    excluded: tuple = ()  # EXCLUDED_CHANNELS applied: raw kept, normalized values NaN

    def norm(self, index: str) -> np.ndarray:
        """frames x 22 normalized envelope of one ok clip."""
        b = self.baselines[self.clips.at[index, "session"]]
        return (self.raw[index] - b.rest) / b.ref

    @property
    def ok(self) -> pd.DataFrame:
        return self.clips[self.clips["status"] == "ok"]


def build_dataset(root: Path, name: str, hold: str, corrections: bool = True) -> Dataset:
    """Load every sensor clip, window it, and compute per-channel hold features.

    hold is "mediapipe" (video-derived window, independent of the EMG being measured)
    or "emg" (summed normalized EMG; biased toward the signal, for runs without MediaPipe).
    corrections applies this release's SCHAEDE_COLUMN_SWAPS and EXCLUDED_CHANNELS; False keeps
    the channels exactly as released.
    Statuses: ok, no_emg, missing_file, empty, no_blendshapes, no_rest.
    """
    if hold not in ("mediapipe", "emg"):
        raise ValueError(f"hold must be 'mediapipe' or 'emg', not {hold!r}")
    index = load_index(root, name)
    if hold == "mediapipe":
        missing = missing_blendshape_files(index, root)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} MediaPipe files listed in {name}_3DMM.csv are missing under {root}, "
                f"e.g. {missing[:3]}. Run with --hold emg or point --root at the full release."
            )

    swaps = SCHAEDE_COLUMN_SWAPS.get(name, ()) if corrections else ()
    excluded = EXCLUDED_CHANNELS.get(name, ()) if corrections else ()
    keep = ~np.isin(CHANNELS, excluded)
    schaede_perm = channel_permutation(swaps)
    clips = index.set_index("index", drop=False)
    clips.index.name = None
    clips["status"] = "ok"
    clips["n_frames"] = 0
    clips["n_negative"] = 0
    raw: dict[str, np.ndarray] = {}
    blend: dict[str, np.ndarray] = {}
    task_activity: dict[str, np.ndarray] = {}  # summed task blendshapes, for the MediaPipe hold window
    for idx, row in clips.iterrows():
        if pd.isna(row["emg"]):
            clips.at[idx, "status"] = "no_emg"
            continue
        path = root / row["emg"]
        if not path.exists():
            clips.at[idx, "status"] = "missing_file"
            continue
        emg = load_emg(path)
        if emg is None:
            clips.at[idx, "status"] = "empty"
            continue
        if row["mode"] == "schaede":
            emg = emg[:, schaede_perm]
        mp = row["MediaPipe"]
        if isinstance(mp, str) and (root / mp).exists():
            shapes = load_blendshapes(root / mp)
            # sEMG can be a few frames shorter than the video; both start at frame 0
            n = min(len(emg), len(shapes))
            emg, shapes = emg[:n], shapes.iloc[:n]
            blend[idx] = blend_summary(shapes)
            if hold == "mediapipe" and row["task"] != REST_TASK:
                task_activity[idx] = shapes[list(HOLD_BLENDSHAPES[row["task"]])].sum(axis=1).to_numpy()
        elif hold == "mediapipe":
            clips.at[idx, "status"] = "no_blendshapes"
            continue
        clips.at[idx, "n_frames"] = len(emg)
        clips.at[idx, "n_negative"] = int((emg < 0).sum())
        raw[idx] = np.clip(emg, 0, None)

    baselines: dict[str, Baseline] = {}
    for session, group in clips[clips["status"] == "ok"].groupby("session"):
        rest = [raw[i] for i in group.index[group["task"] == REST_TASK]]
        schaede = [raw[i] for i in group.index[group["mode"] == "schaede"]]
        if not rest:
            clips.loc[group.index, "status"] = "no_rest"
            continue
        b = session_baseline(np.concatenate(rest), np.concatenate(schaede))
        baselines[session] = Baseline(rest=b.rest, ref=np.where(keep, b.ref, np.nan))
    for idx in clips.index[clips["status"] == "no_rest"]:
        raw.pop(idx, None)
        blend.pop(idx, None)

    holds: dict[str, np.ndarray] = {}
    records = []
    for idx, row in clips[clips["status"] == "ok"].iterrows():
        b = baselines[row["session"]]
        frames = raw[idx]
        if row["task"] == REST_TASK:
            mask = np.ones(len(frames), dtype=bool)
        elif hold == "mediapipe":
            mask = hold_mask(task_activity[idx])
        else:
            mask = hold_mask(np.nansum((frames - b.rest) / b.ref, axis=1))
        holds[idx] = mask
        hold_uv = np.where(keep, np.median(frames[mask], axis=0), np.nan)
        delta = hold_uv - b.rest
        records.append(pd.DataFrame({
            "index": idx,
            "channel": CHANNELS,
            "site": CHANNEL_SITE,
            "side": CHANNEL_SIDE,
            "hold_uv": hold_uv,
            "rest_uv": b.rest,
            "delta_uv": delta,
            "norm": delta / b.ref,
        }))
    clips["n_hold"] = pd.Series({k: int(v.sum()) for k, v in holds.items()}, dtype="Int64")
    clips["hold_source"] = np.where(clips["status"] == "ok", hold, "")
    meta = clips[["session", "pid", "rec", "mode", "task", "task_seq", "sync_grade"]]
    features = pd.concat(records, ignore_index=True).join(meta, on="index")
    return Dataset(name, hold, clips, features, raw, holds, blend, baselines, swaps, excluded)


# ----------------------------------------------------------------------------
# Aggregation: clip -> session -> participant -> group
# ----------------------------------------------------------------------------

CLIP_KEYS = ["index", "session", "pid", "rec", "mode", "task", "task_seq", "sync_grade"]


def site_values(features: pd.DataFrame) -> pd.DataFrame:
    """Clip x site: left and right averaged."""
    return features.groupby(CLIP_KEYS + ["site"], as_index=False, dropna=False)[["norm", "delta_uv"]].mean()


def session_values(site_df: pd.DataFrame) -> pd.DataFrame:
    """Session x task x site: median over repeated imitations (emotions have 4 per session)."""
    grouped = site_df.groupby(["session", "pid", "rec", "mode", "task", "site"], as_index=False)
    out = grouped[["norm", "delta_uv"]].median()
    out["n_clips"] = grouped.size()["size"].to_numpy()
    return out


def participant_values(session_df: pd.DataFrame) -> pd.DataFrame:
    """Participant x task x site: mean over sessions."""
    grouped = session_df.groupby(["pid", "mode", "task", "site"], as_index=False)
    out = grouped[["norm", "delta_uv"]].mean()
    out["n_sessions"] = grouped.size()["size"].to_numpy()
    return out


def group_values(participant_df: pd.DataFrame) -> pd.DataFrame:
    """Task x site: median and IQR across participants."""
    grouped = participant_df.groupby(["task", "site"])["norm"]
    return pd.DataFrame({
        "median": grouped.median(),
        "q25": grouped.quantile(0.25),
        "q75": grouped.quantile(0.75),
        "n_participants": grouped.count(),
    }).reset_index()


def wide(df: pd.DataFrame, keys: list[str], sites=RANK_SITES) -> pd.DataFrame:
    return df.pivot_table(index=keys, columns="site", values="norm")[list(sites)]


def rank_corr_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise Spearman rho between the rows of an n x k matrix."""
    ranks = rankdata(vectors, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(ranks)


def reliability(session_df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho of session site vectors, per task, for every session pair.

    kind is "within" (same participant, different sessions: the ceiling a model can
    reach against this person) or "between" (different participants).
    """
    vecs = wide(session_df[session_df["task"] != REST_TASK], ["task", "pid", "session"]).dropna()
    rows = []
    for task, sub in vecs.groupby(level="task"):
        if len(sub) < 2:
            continue
        rho = rank_corr_matrix(sub.to_numpy())
        pids = sub.index.get_level_values("pid").to_numpy()
        a, b = np.triu_indices(len(sub), k=1)
        rows.append(pd.DataFrame({
            "task": task,
            "kind": np.where(pids[a] == pids[b], "within", "between"),
            "pid_a": pids[a],
            "pid_b": pids[b],
            "rho": rho[a, b],
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["task", "kind", "pid_a", "pid_b", "rho"])


def imitation_repeatability(site_df: pd.DataFrame) -> pd.DataFrame:
    """Per session and emotion: median Spearman rho among the repeated imitations."""
    emo = site_df[site_df["mode"] == "emotion"]
    vecs = wide(emo, ["session", "task", "index"]).dropna()
    rows = []
    for (session, task), sub in vecs.groupby(level=["session", "task"]):
        if len(sub) < 2:
            continue
        rho = rank_corr_matrix(sub.to_numpy())
        rows.append({"session": session, "task": task, "rho": np.nanmedian(rho[np.triu_indices(len(sub), k=1)])})
    return pd.DataFrame(rows, columns=["session", "task", "rho"])


def top_sites(session_df: pd.DataFrame) -> pd.DataFrame:
    """Session x task: the site with the highest normalized value."""
    vecs = wide(session_df[session_df["task"] != REST_TASK], ["task", "pid", "session"], SITES).dropna(how="all")
    return vecs.idxmax(axis=1).rename("top_site").reset_index()


OCULI_ORIS_TASKS = ("Eye-Gentle", "Eye-Tight", "Lip-Pucker", "Smile-Closed")
# Expected sign of (orbicularis oculi - orbicularis oris) if the labels are right.
OCULI_ORIS_EXPECTED = {"Eye-Gentle": 1, "Eye-Tight": 1, "Lip-Pucker": -1}
# Gentle closure barely rises above rest, so only the two forceful tasks decide a swap.
SWAP_TASKS = ("Eye-Tight", "Lip-Pucker")


def oculi_oris(features: pd.DataFrame) -> pd.DataFrame:
    """Session x task: normalized OOC, OOR and their difference, per side and averaged."""
    sub = features[features["task"].isin(OCULI_ORIS_TASKS) & features["site"].isin(["Orbicularis Oculi", "Orbicularis Oris"])]
    per_clip = sub.pivot_table(index=["session", "pid", "task", "index"], columns="channel", values="norm")
    per_session = per_clip.groupby(level=["session", "pid", "task"]).median()
    out = pd.DataFrame(index=per_session.index)
    for side in "lr":
        out[f"d_{side}"] = per_session[f"Orbicularis Oculi_{side}"] - per_session[f"Orbicularis Oris_{side}"]
    out["ooc"] = per_session[["Orbicularis Oculi_l", "Orbicularis Oculi_r"]].mean(axis=1)
    out["oor"] = per_session[["Orbicularis Oris_l", "Orbicularis Oris_r"]].mean(axis=1)
    out["d"] = out["ooc"] - out["oor"]
    return out.reset_index()


def swap_pattern(oo: pd.DataFrame) -> pd.Series:
    """Per session: "expected", "swapped" (both SWAP_TASKS reversed) or "mixed"."""
    signs = oo[oo["task"].isin(SWAP_TASKS)].pivot(index="session", columns="task", values="d")
    signs = signs.dropna()
    agree = np.sign(signs) == pd.Series(OCULI_ORIS_EXPECTED)[signs.columns]
    return pd.Series(np.select([agree.all(axis=1), (~agree).all(axis=1)], ["expected", "swapped"], "mixed"), index=signs.index)


def blendshape_crosscheck(ds: Dataset) -> pd.DataFrame:
    """Per session: Pearson r between orbicularis envelopes and MediaPipe blendshapes.

    Frames of all schaede clips of the session are concatenated, so the between-task
    contrast (eyes closing in eye tasks, lips pushing in Lip-Pucker) drives r.
    If the oculi/oris labels are right, the eye blendshapes track OOC and the pucker
    blendshapes track OOR.
    """
    ooc = [CHANNELS.index(f"Orbicularis Oculi_{s}") for s in "lr"]
    oor = [CHANNELS.index(f"Orbicularis Oris_{s}") for s in "lr"]
    rows = []
    ok = ds.ok[(ds.ok["mode"] == "schaede")]
    for session, group in ok.groupby("session"):
        ids = [i for i in group.index if i in ds.blend]
        if not ids:
            continue
        emg = np.concatenate([ds.norm(i) for i in ids])
        bs = np.concatenate([ds.blend[i] for i in ids])
        signals = {
            "OOC": np.nanmean(emg[:, ooc], axis=1),
            "OOR": np.nanmean(emg[:, oor], axis=1),
            "OOC_l": emg[:, ooc[0]],
            "OOC_r": emg[:, ooc[1]],
        }
        blends = dict(zip(BLEND_SUMMARY, bs.T))
        row = {"session": session, "pid": group["pid"].iloc[0]}
        for e, b in [("OOC", "eye"), ("OOC", "pucker"), ("OOR", "eye"), ("OOR", "pucker"),
                     ("OOC_l", "eyeBlinkLeft"), ("OOC_l", "eyeBlinkRight"),
                     ("OOC_r", "eyeBlinkLeft"), ("OOC_r", "eyeBlinkRight")]:
            x, y = signals[e], blends[b]
            good = np.isfinite(x) & np.isfinite(y)
            row[f"{e}~{b}"] = np.corrcoef(x[good], y[good])[0, 1] if good.sum() > 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aligned_traces(ds: Dataset, task: str, sites=MUSCLES_OF_INTEREST, pre: int = 2 * FPS, post: int = 6 * FPS) -> pd.DataFrame:
    """Normalized site envelopes aligned to hold onset, aggregated imitation -> session -> participant.

    Returns rows per (pid, site) with a frames column holding a float array of length pre + post.
    """
    cols = [[CHANNELS.index(f"{s}_{side}") for side in "lr"] for s in sites]
    per_session: dict[tuple[str, str], list[np.ndarray]] = {}
    for idx, row in ds.ok[ds.ok["task"] == task].iterrows():
        norm = ds.norm(idx)
        onset = int(np.argmax(ds.hold[idx])) if task != REST_TASK else 0
        window = np.full((pre + post, len(sites)), np.nan)
        lo, hi = max(onset - pre, 0), min(onset + post, len(norm))
        sites_trace = np.column_stack([np.nanmean(norm[:, c], axis=1) for c in cols])
        window[lo - (onset - pre): hi - (onset - pre)] = sites_trace[lo:hi]
        per_session.setdefault((row["pid"], row["session"]), []).append(window)
    per_pid: dict[str, list[np.ndarray]] = {}
    with warnings.catch_warnings():  # frames past a clip edge are NaN in every imitation
        warnings.simplefilter("ignore", RuntimeWarning)
        for (pid, _), windows in per_session.items():
            per_pid.setdefault(pid, []).append(np.nanmedian(np.stack(windows), axis=0))
        means = {pid: np.nanmean(np.stack(sessions), axis=0) for pid, sessions in per_pid.items()}
    rows = []
    for pid, mean in means.items():
        for k, s in enumerate(sites):
            rows.append({"pid": pid, "site": s, "frames": mean[:, k]})
    return pd.DataFrame(rows, columns=["pid", "site", "frames"])


def maya_recruitment(session_df: pd.DataFrame) -> pd.DataFrame:
    """Model-join table: Maya expression x MaM task x session x site.

    Expressions without a MaM task, or whose task has no usable session, keep one row
    with empty human columns, so every expression appears in the join.
    """
    parts = []
    for expr, task, role in maya_rows():
        head = {"maya_expression": expr.name, "au_recipe": expr.au_recipe, "pairing": expr.pairing, "role": role}
        sub = session_df[session_df["task"] == task].rename(columns={"task": "mam_task", "delta_uv": "delta_uv_median"})
        parts.append(sub.assign(**head) if len(sub) else pd.DataFrame([{**head, "mam_task": task}]))
    cols = ["maya_expression", "au_recipe", "pairing", "role", "mam_task", "pid", "rec", "session", "site", "norm", "delta_uv_median", "n_clips"]
    return pd.concat(parts, ignore_index=True)[cols]


def swap_candidates() -> dict[str, np.ndarray]:
    """Every site pair swapped on both sides, and every site swapped left/right."""
    out = {f"{SITE_LABELS[a]} \u2194 {SITE_LABELS[b]}": channel_permutation([(a, b)])
           for i, a in enumerate(SITES) for b in SITES[i + 1:]}
    for site in SITES:
        perm = np.arange(len(CHANNELS))
        i, j = CHANNELS.index(f"{site}_l"), CHANNELS.index(f"{site}_r")
        perm[i], perm[j] = j, i
        out[f"{SITE_LABELS[site]} L \u2194 R"] = perm
    return out


def channel_corr(frames: np.ndarray) -> np.ndarray | None:
    """22 x 22 Pearson r over frames; channels without data stay NaN. None if too few frames."""
    cols = np.isfinite(frames).any(axis=0)
    frames = frames[:, cols]
    frames = frames[np.isfinite(frames).all(axis=1)]
    if len(frames) <= 50:
        return None
    out = np.full((len(CHANNELS), len(CHANNELS)), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[np.ix_(cols, cols)] = np.corrcoef(frames.T)
    return out


def _mode_corrs(ds: Dataset, released: bool):
    """Per session: (schaede, emotion) channel correlation of raw frames.

    Raw frames, because Pearson r ignores per-channel offset and scale, so excluded channels
    still show; released=True undoes the applied column swaps in schaede frames.
    """
    undo = channel_permutation(ds.schaede_swaps) if released else np.arange(len(CHANNELS))
    for session, group in ds.ok.groupby("session"):
        mats = []
        for mode in ("schaede", "emotion"):
            ids = group.index[group["mode"] == mode]
            frames = np.concatenate([ds.raw[i] for i in ids]) if len(ids) else np.empty((0, len(CHANNELS)))
            mats.append(channel_corr(frames[:, undo] if mode == "schaede" else frames))
        if mats[0] is not None and mats[1] is not None:
            yield session, mats[0], mats[1]


def _r(a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[good], b[good])[0, 1]) if good.sum() > 2 else np.nan


def profile_similarity(ds: Dataset, released: bool = True) -> pd.DataFrame:
    """Per session and channel: does the channel co-activate with the others the same way in
    schaede and emotion clips? r between its two correlation-matrix rows, itself left out."""
    rows = []
    for session, schaede, emotion in _mode_corrs(ds, released):
        for i, channel in enumerate(CHANNELS):
            others = np.arange(len(CHANNELS)) != i
            rows.append({"session": session, "channel": channel, "similarity": _r(schaede[i, others], emotion[i, others])})
    return pd.DataFrame(rows, columns=["session", "channel", "similarity"])


def label_swap_scan(ds: Dataset, released: bool = True) -> pd.DataFrame:
    """Per session and candidate swap: change in agreement between schaede and emotion co-activation.

    Agreement is the Pearson r between the upper triangles of the session's channel correlation
    matrices over schaede frames and over emotion frames. A positive gain means the candidate
    swap, applied to schaede frames only, makes the two clip types look like the same electrodes.
    released=True scores against the labels as released (undoing any applied correction);
    False scores against the labels as loaded, i.e. what is left after the correction.
    """
    iu = np.triu_indices(len(CHANNELS), k=1)
    candidates = swap_candidates()
    rows = []
    for session, schaede, emotion in _mode_corrs(ds, released):
        base = _r(schaede[iu], emotion[iu])
        for name, perm in candidates.items():
            swapped = schaede[np.ix_(perm, perm)]
            rows.append({"session": session, "candidate": name, "gain": _r(swapped[iu], emotion[iu]) - base})
    return pd.DataFrame(rows, columns=["session", "candidate", "gain"])
