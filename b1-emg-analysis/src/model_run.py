"""Read one lab simulation run and rebuild every landmark's path over the solver steps.

A run folder (e.g. the "hpy" happy run) holds:
    final_epoch_action_list.pkl  one action vector per solver step (the landmark moves)
    deltaB_temp.pkl              final landmark positions and final muscle meshes

Action layout (from evaluator.py / the environment): 3 numbers per joint, in this order
    OOR 300 joints | jaw rotation 1 | OOC_l 300 | OOC_r 300 | frontalis 300   = 3603

Landmark position at step t = initial position + sum of the moves up to t.
The initial position is not stored, so it is recovered as final - total move.
"""
import pickle
from pathlib import Path

import numpy as np

# Pickle can run arbitrary code, so only plain numpy arrays are allowed through.
_ALLOWED = {
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def safe_load(path):
    with open(path, "rb") as fh:
        return _SafeUnpickler(fh).load()


# (name in our code, slice of the action vector, key of final joints in deltaB_temp.pkl)
TISSUES = {
    "OOR": (slice(0, 900), "x_joint_"),
    "OOC_l": (slice(903, 1803), "x_ooc_l_joint_"),
    "OOC_r": (slice(1803, 2703), "x_ooc_r_joint_"),
    "frontalis": (slice(2703, 3603), "x_frontalis_joint_"),
}
JAW = slice(900, 903)


def load_run(run_dir):
    """Return {tissue: (T+1, 300, 3) landmark positions, step 0 = rest}, plus jaw rotation (T+1, 3)."""
    run_dir = Path(run_dir)
    actions = np.asarray(safe_load(run_dir / "final_epoch_action_list.pkl")["final_epoch_action_list"], float)
    final = safe_load(run_dir / "deltaB_temp.pkl")
    total = np.asarray(final["delta_b0_conca"], float)
    if not np.allclose(actions.sum(axis=0), total):
        raise ValueError("actions do not add up to delta_b0_conca; unknown run format")

    cum = np.vstack([np.zeros((1, actions.shape[1])), np.cumsum(actions, axis=0)])
    paths = {}
    for name, (sl, key) in TISSUES.items():
        final_joints = np.asarray(final[key], float)               # (300, 3)
        rest = final_joints - total[sl].reshape(-1, 3)
        paths[name] = rest[None] + cum[:, sl].reshape(len(cum), -1, 3)
    return paths, cum[:, JAW]


SKIN_FILES = {"OOR": "oor", "OOC_l": "ooc_l", "OOC_r": "ooc_r", "frontalis": "frontalis"}


def load_skinning(lbs_dir, tissue):
    """Bind-pose vertices (N, 3) and skin weights (N, 300) of one muscle, from automation/*.json."""
    import json
    f = SKIN_FILES[tissue]
    bind = np.asarray(json.load(open(Path(lbs_dir) / f"{f}_bind_pose.json")), float)
    weights = np.asarray(json.load(open(Path(lbs_dir) / f"{f}_skin_weights.json")), float)
    return bind, weights


TARGET_KEYS = {"OOR": "x_middleres_OrbicularisOris1", "OOC_l": "x_middleres_OrbicularisOculi_l1",
               "OOC_r": "x_middleres_OrbicularisOculi_r1", "frontalis": "x_middleres_Frontalis1"}


def load_target(pose_file):
    """Artist-sculpted target meshes {tissue: (N, 3)} from a dataV22_*.py pose file.

    In these files index 0 is the rest shape (equal to the bind pose) and index 1 the expression.
    The file is read as data with ast.literal_eval, never executed.
    """
    import ast
    want = {v: k for k, v in TARGET_KEYS.items()}
    out = {}
    for node in ast.parse(Path(pose_file).read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in want:
            out[want[node.targets[0].id]] = np.asarray(ast.literal_eval(node.value), float)
    return out


def driven(paths, tol=1e-9):
    """Tissues the solver actually moved in this run."""
    return [k for k, p in paths.items() if np.abs(p[-1] - p[0]).max() > tol]


def progress(paths):
    """Expression progress 0 -> 1 for each step: net progress towards the final pose.

    p(t) = 1 - |X(t) - X_end| / |X_0 - X_end| over all driven landmarks, with a running
    maximum so the clock never goes backwards.

    Not path length: the solver travels 2.6-3x further than the net displacement (it jitters
    on its way), so a path-length clock mostly counts jitter and makes real movement look early.
    """
    X = np.concatenate([paths[k].reshape(len(paths[k]), -1) for k in driven(paths)], axis=1)
    d = np.linalg.norm(X - X[-1], axis=1)
    return np.maximum.accumulate(1.0 - d / d[0])


def wasted_motion(paths):
    """Total landmark path length divided by net displacement (1.0 = straight to the target)."""
    X = np.concatenate([paths[k].reshape(len(paths[k]), -1) for k in driven(paths)], axis=1)
    return np.linalg.norm(np.diff(X, axis=0), axis=1).sum() / np.linalg.norm(X[-1] - X[0])
