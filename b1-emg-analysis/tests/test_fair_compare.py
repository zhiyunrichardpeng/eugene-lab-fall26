"""Checks for the fair-comparison tool on synthetic data with known answers.

Run:  ../.venv/bin/python test_fair_compare.py   (from the tests/ folder)
"""
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fair_compare as fc
import mam_happy
import model_run
import virtual_electrode as ve


def make_run(folder, steps=50, contract=0.2):
    """A fake run: OOR landmarks on a line contract towards x=0 by `contract`, others still."""
    rng = np.random.default_rng(0)
    rest = {k: rng.normal(size=(300, 3)) * 10 for k in model_run.TISSUES}
    rest["OOR"][:, 0] = np.linspace(-50, 50, 300)
    final = {k: v.copy() for k, v in rest.items()}
    final["OOR"][:, 0] *= 1 - contract
    total = np.zeros(3603)
    for k, (sl, _) in model_run.TISSUES.items():
        total[sl] = (final[k] - rest[k]).ravel()
    actions = [total / steps for _ in range(steps)]
    pickle.dump({"final_epoch_action_list": actions}, open(folder / "final_epoch_action_list.pkl", "wb"))
    d = {"delta_b0_conca": total}
    for k, (_, key) in model_run.TISSUES.items():
        d[key] = final[k]
    pickle.dump(d, open(folder / "deltaB_temp.pkl", "wb"))
    return rest, final


def test_safe_loader_blocks_code():
    class Evil:
        def __reduce__(self):
            return (print, ("should never run",))
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "evil.pkl"
        pickle.dump(Evil(), open(p, "wb"))
        try:
            model_run.safe_load(p)
        except pickle.UnpicklingError:
            return
        raise AssertionError("unsafe pickle was loaded")


def test_rebuilds_paths_and_progress():
    with tempfile.TemporaryDirectory() as t:
        rest, final = make_run(Path(t))
        paths, _ = model_run.load_run(t)
        assert np.allclose(paths["OOR"][0], rest["OOR"]) and np.allclose(paths["OOR"][-1], final["OOR"])
        assert model_run.driven(paths) == ["OOR"]
        p = model_run.progress(paths)
        assert p[0] == 0 and abs(p[-1] - 1) < 1e-12 and np.all(np.diff(p) >= 0)
        assert abs(p[25] - 0.5) < 1e-9          # constant speed -> halfway at the middle step


def test_progress_ignores_jitter():
    """Overshoot-and-return must not count as progress; wasted motion must be reported."""
    X = np.array([0, 1, 3, 2, 3, 2.5, 4.0])                 # net 4, path 1+2+1+1+0.5+1.5 = 7
    paths = {"OOR": np.zeros((len(X), 300, 3))}
    paths["OOR"][:, 0, 0] = X
    p = model_run.progress(paths)
    assert np.all(np.diff(p) >= 0) and p[-1] == 1.0
    assert abs(p[2] - 0.75) < 1e-12 and abs(p[3] - 0.75) < 1e-12   # stepping back is not progress
    assert abs(model_run.wasted_motion(paths) - 7 / 4) < 1e-12


def test_shortening_sign_and_size():
    with tempfile.TemporaryDirectory() as t:
        make_run(Path(t), contract=0.2)
        paths, _ = model_run.load_run(t)
        pair = np.array([[-20.0, 0, 0], [20.0, 0, 0]])
        eps, trans, _ = ve.pair_signals(paths["OOR"], pair, radius=3.0)
        assert abs(eps[-1] - 0.2) < 0.02, eps[-1]   # 20% contraction recovered
    with tempfile.TemporaryDirectory() as t:
        make_run(Path(t), contract=-0.1)              # stretch
        paths, _ = model_run.load_run(t)
        eps, _, _ = ve.pair_signals(paths["OOR"], pair, radius=3.0)
        assert eps[-1] < -0.05


def test_weights_fallback_and_normalised():
    lm = np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0], [300, 0, 0.0]])
    idx, w, dmin = ve.weights_for_point(lm, np.array([1.0, 0, 0]), radius=5, min_k=3)
    assert idx.size == 3 and abs(w.sum() - 1) < 1e-12 and dmin == 1.0


def test_first_passage_and_rise():
    x = np.array([0, .1, .1, .4, .7, 1.0])
    idx = mam_happy.first_passage(x, np.array([0, .1, .5, 1.0]))
    assert list(idx) == [0, 1, 4, 5]
    r = fc.rise([2, 3, 4, 6])
    assert np.allclose(r, [0, .25, .5, 1]) and fc.t50(np.r_[r, [1] * 17]) == mam_happy.GRID[2]
    assert np.isnan(fc.rise([1, 1, 1])).all()


def test_classify():
    still = {"driven": False}
    assert fc.classify(True, still) == "missing agonist"
    assert fc.classify(False, still) == "agree: silent"
    assert fc.classify(True, {"driven": True, "eps": np.array([0, .05])}) == "agree: contracts"
    assert fc.classify(True, {"driven": True, "eps": np.array([0, -.05])}).startswith("sign conflict")
    assert fc.classify(False, {"driven": True, "eps": np.array([0, .05])}) == "over-recruitment"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok ", name)
    print("all tests passed")
