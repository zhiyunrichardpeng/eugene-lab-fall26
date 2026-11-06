"""Tests for mam_data.py against the real local files (skipped when a dataset is absent).

Run from this folder:  python -m pytest test_mam_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mam_data as m

HERE = Path(__file__).resolve().parent
MAM = next((p for p in (HERE / "MaM_Public", HERE / "MaM") if (p / "MaM.csv").exists()), None)
MMM = next((p for p in HERE.parents if (p / "MMM" / "MMM" / "MMM.csv").exists()), None)
MMM = MMM / "MMM" / "MMM" if MMM else None

needs_mam = pytest.mark.skipif(MAM is None, reason="MaM dataset not found next to this file")
needs_mmm = pytest.mark.skipif(MMM is None, reason="MMM dataset not found in a parent folder")


@pytest.fixture(scope="module")
def mam():
    return m.build_dataset(MAM, "MaM", "emg")


@needs_mam
def test_emg_has_22_alphabetical_channels():
    emg = m.load_emg(MAM / "emg/01_3S/01_3S_emotion_00_angry.csv")
    assert emg.shape[1] == 22
    assert list(m.CHANNELS) == sorted(m.CHANNELS)


@needs_mam
def test_space_aligned_csv_reads_like_the_rest():
    emg = m.load_emg(MAM / "emg/04_2S/04_2S_emotion_02_sad.csv")
    assert emg is not None and emg.shape == (180, 22) and np.isfinite(emg).all()


@needs_mam
def test_empty_file_is_reported_not_raised(mam):
    assert m.load_emg(MAM / "emg/09_7S/09_7S_emotion_01_surprised.csv") is None
    assert mam.clips.at["09_7S_emotion_01_surprised", "status"] == "empty"


@needs_mam
def test_session_without_rest_clip_is_excluded(mam):
    no_rest = mam.clips[mam.clips["status"] == "no_rest"]
    assert set(no_rest["session"]) == {"10_8S"}
    assert "10_8S" not in mam.baselines


@needs_mam
def test_traces_are_non_negative_and_references_positive(mam):
    assert all((raw >= 0).all() for raw in mam.raw.values())
    for baseline in mam.baselines.values():
        ref = baseline.ref[np.isfinite(baseline.ref)]
        assert (ref > 0).all()


@needs_mam
def test_rest_hold_covers_every_frame_and_task_hold_is_partial(mam):
    rest = mam.ok[mam.ok["task"] == m.REST_TASK].index[0]
    assert mam.hold[rest].all()
    tight = mam.ok[mam.ok["task"] == "Eye-Tight"].index[0]
    assert 0 < mam.hold[tight].sum() < len(mam.hold[tight])


@needs_mam
def test_every_ok_clip_has_22_features(mam):
    counts = mam.features.groupby("index").size()
    assert len(counts) == len(mam.ok) and (counts == 22).all()


@needs_mam
def test_maya_tasks_exist_in_mam():
    tasks = set(pd.read_csv(MAM / "MaM.csv", dtype=str)["task"])
    mapped = {task for expr in m.MAYA_EXPRESSIONS for task in expr.tasks}
    assert len(m.MAYA_EXPRESSIONS) == 14
    assert mapped <= tasks
    assert set(m.TASKS) == tasks


@needs_mam
def test_hold_blendshapes_cover_every_task():
    assert set(m.TASKS) - {m.REST_TASK} <= set(m.HOLD_BLENDSHAPES)


def test_hold_mask_threshold_is_half_the_range():
    activity = np.r_[np.full(20, 0.3), np.full(10, 1.0), np.full(20, 0.3)]
    mask = m.hold_mask(activity)
    assert mask.sum() == 10 and mask[20:30].all()
    assert m.hold_mask(np.full(10, 0.5)).all()


def test_rank_corr_matrix_matches_scipy():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(4, 10))
    rho = m.rank_corr_matrix(vecs)
    assert np.isclose(rho[0, 1], spearmanr(vecs[0], vecs[1]).statistic)


@needs_mam
def test_mediapipe_mode_fails_fast_without_files():
    if not m.missing_blendshape_files(m.load_index(MAM, "MaM"), MAM):
        pytest.skip("full release present, MediaPipe files available")
    with pytest.raises(FileNotFoundError, match="MediaPipe files"):
        m.build_dataset(MAM, "MaM", "mediapipe")


@needs_mmm
def test_mediapipe_hold_on_mmm():
    ds = m.build_dataset(MMM, "MMM", "mediapipe")
    assert (ds.clips["status"] == "ok").all()
    for idx in ds.ok.index:
        assert len(ds.hold[idx]) == len(ds.raw[idx]) == len(ds.blend[idx])
    tight = ds.ok[ds.ok["task"] == "Eye-Tight"].index[0]
    assert 0 < ds.hold[tight].sum() < len(ds.hold[tight])
    assert not m.blendshape_crosscheck(ds).empty
    # Known MMM finding: its oculi/oris columns read swapped in both sessions.
    assert (m.swap_pattern(m.oculi_oris(ds.features)) == "swapped").all()


def test_maya_recruitment_keeps_empty_expressions():
    session_df = pd.DataFrame({
        "session": ["01_3S"], "pid": ["01"], "rec": ["3S"], "mode": ["schaede"], "task": ["Forehead"],
        "site": ["Medialer Frontalis"], "norm": [1.0], "delta_uv": [50.0], "n_clips": [1],
    })
    table = m.maya_recruitment(session_df)
    assert set(table["maya_expression"]) == {e.name for e in m.MAYA_EXPRESSIONS}
    empty = table[table["maya_expression"] == "Cheek Raise"]
    assert len(empty) == 1 and empty["norm"].isna().all()
    brow = table[table["maya_expression"] == "Raise the Brow"]
    assert brow["norm"].tolist() == [1.0]


@needs_mam
def test_schaede_frontalis_levator_fix(mam):
    released = m.build_dataset(MAM, "MaM", "emg", corrections=False)
    assert mam.schaede_swaps and not released.schaede_swaps

    def forehead_fl_minus_lls(ds):
        site = m.session_values(m.site_values(ds.features))
        w = site[site["task"] == "Forehead"].pivot(index="session", columns="site", values="norm")
        return w["Lateraler Frontalis"] - w["Levator Labii Superioris"]

    assert (forehead_fl_minus_lls(released) < 0).all()  # brow raise led by "levator labii" as released
    assert (forehead_fl_minus_lls(mam) > 0).mean() > 0.95
    emotion = mam.ok[mam.ok["mode"] == "emotion"].index[0]
    assert np.array_equal(mam.raw[emotion], released.raw[emotion])  # emotion clips untouched


@needs_mam
def test_label_scan_finds_the_swap_and_nothing_after_fix(mam):
    known = "Frontalis lat. ↔ Lev. labii sup."
    share = lambda scan: (scan.assign(up=scan["gain"] > 0).groupby("candidate")["up"].mean())
    before = share(m.label_swap_scan(mam, released=True))
    after = share(m.label_swap_scan(mam, released=False))
    assert before.idxmax() == known and before[known] > 0.9
    assert after[known] < 0.1 and after.max() < 0.9


def test_channel_permutation_is_an_involution():
    perm = m.channel_permutation([("Lateraler Frontalis", "Levator Labii Superioris")])
    assert np.array_equal(perm[perm], np.arange(len(m.CHANNELS)))
    assert m.CHANNELS[perm[m.CHANNELS.index("Lateraler Frontalis_r")]] == "Levator Labii Superioris_r"


@needs_mam
def test_excluded_channel_has_no_normalized_values(mam):
    dsp_l = mam.features[mam.features["channel"] == "Depressor Supercilii Procerus_l"]
    assert dsp_l[["norm", "delta_uv", "hold_uv"]].isna().all().all()
    site = m.site_values(mam.features)
    dsp = site[site["site"] == "Depressor Supercilii Procerus"]
    right = mam.features[mam.features["channel"] == "Depressor Supercilii Procerus_r"]
    assert np.allclose(dsp.set_index("index")["norm"].sort_index(), right.set_index("index")["norm"].sort_index(), equal_nan=True)


@needs_mam
def test_left_brow_depressor_carries_mouth_activity_as_released():
    released = m.build_dataset(MAM, "MaM", "emg", corrections=False)
    chan = m.session_values(released.features.drop(columns="site").rename(columns={"channel": "site"}))
    w = chan.pivot_table(index=["session", "task"], columns="site", values="norm")
    for task in ("Lip-Pucker", "Depress-Lip"):
        t = w.xs(task, level="task")
        assert (t["Depressor Supercilii Procerus_l"] > t["Depressor Supercilii Procerus_r"]).mean() > 0.95
    sim = m.profile_similarity(released).groupby("channel")["similarity"].median()
    assert sim["Depressor Supercilii Procerus_l"] < 0 < sim.drop("Depressor Supercilii Procerus_l").min()
