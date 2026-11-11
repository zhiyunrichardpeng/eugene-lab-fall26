"""Virtual electrodes: what the model does at the spot of each real MaM electrode.

An electrode records from a ~1 cm patch, so one model landmark is too small and the whole
muscle is too big. A virtual landmark is the Gaussian-weighted average of the landmarks of the
muscle under the electrode that lie within radius r (r ~ 1 cm, the electrode pickup scale).

For each bipolar pair (two electrodes laid along the fibres) we measure:
    shortening  eps(t) = 1 - |v1(t) - v2(t)| / |v1(0) - v2(0)|   (>0 = contracting, <0 = stretched)
    translation |m(t) - m(0)|, m = midpoint of the pair            (how far that spot travelled)

Electrode positions come from electrodes.json (Fridlund placement on the lab head, by Oleh and
Richard), which is in the same coordinate space as the muscle meshes.
"""
import json

import numpy as np

# MaM channel -> model tissue under it. None = no muscle the solver drives at that site.
CHANNEL_TISSUE = {
    "Orbicularis Oris_l": "OOR", "Orbicularis Oris_r": "OOR",
    "Orbicularis Oculi_l": "OOC_l", "Orbicularis Oculi_r": "OOC_r",
    "Medialer Frontalis_l": "frontalis", "Medialer Frontalis_r": "frontalis",
    "Lateraler Frontalis_l": "frontalis", "Lateraler Frontalis_r": "frontalis",
    "Zygomaticus_l": None, "Zygomaticus_r": None,
    "Levator Labii Superioris_l": None, "Levator Labii Superioris_r": None,
    "Depressor Ang. Oris_l": None, "Depressor Ang. Oris_r": None,
    "Mentalis_l": None, "Mentalis_r": None,
    "Corrugator Supercilii_l": None, "Corrugator Supercilii_r": None,
    "Depressor Supercilii Procerus_l": None, "Depressor Supercilii Procerus_r": None,
    "Masseter_l": None, "Masseter_r": None,
}


def load_electrodes(path):
    """Return ({channel: (2, 3) rest positions}, units_per_cm)."""
    e = json.load(open(path))
    pos = {ch: np.array([el["rest_position"] for el in c["electrodes"]], float)
           for ch, c in e["channels"].items()}
    return pos, float(e["units_per_cm"])


def weights_for_point(rest_landmarks, point, radius, min_k=3):
    """Indices and normalised Gaussian weights (sigma = radius / 2) of landmarks within radius.

    If fewer than min_k landmarks fall inside the radius, the min_k nearest are used, so a
    sparse area still gets a stable average. Also returns the distance to the nearest landmark.
    """
    d = np.linalg.norm(rest_landmarks - point, axis=1)
    idx = np.flatnonzero(d <= radius)
    if idx.size < min_k:
        idx = np.argsort(d)[:min_k]
    sigma = radius / 2.0
    w = np.exp(-(d[idx] ** 2) / (2.0 * sigma ** 2))
    return idx, w / w.sum(), float(d.min())


def virtual_path(path, idx, w):
    """path: (T, N, 3) landmark positions. Returns the virtual landmark over time, (T, 3)."""
    return (path[:, idx, :] * w[None, :, None]).sum(axis=1)


def surface_patch(rest_vertices, point, radius, layer):
    """Mesh vertices directly under an electrode, first tissue layer only.

    The inward direction is taken from the electrode to the nearest vertex. A vertex is kept if it
    lies within `radius` sideways of that line and within `layer` of the top surface, so the
    deeper side of the muscle (e.g. the inner lip surface of the orbicularis oris ring, which sits
    behind a gap of a few units) is excluded. Weights: Gaussian in sideways distance, sigma = radius / 2.
    """
    rel = rest_vertices - point
    dist = np.linalg.norm(rel, axis=1)
    n = rel[dist.argmin()] / dist.min()
    depth = rel @ n
    side = np.linalg.norm(rel - np.outer(depth, n), axis=1)
    under = side <= radius
    top = depth[under].min()
    idx = np.flatnonzero(under & (depth <= top + layer))
    w = np.exp(-(side[idx] ** 2) / (2.0 * (radius / 2.0) ** 2))
    return idx, w / w.sum(), float(dist.min())


def mesh_pair_signals(landmark_path, bind, weights, electrodes, radius, layer):
    """Like pair_signals, but on the full muscle mesh rebuilt by linear blend skinning.

    vertex(t) = bind + weights @ (landmarks(t) - landmarks(0)), the lab's own LBS formula
    (pytorch_analytical_gradient.LinearBlendSkinning); only the patch vertices are rebuilt.
    """
    moves = landmark_path - landmark_path[0]                          # (T, K, 3)
    v, info = [], []
    for p in electrodes:
        idx, w, dmin = surface_patch(bind, p, radius, layer)
        verts = bind[idx][None] + np.einsum("nk,tkd->tnd", weights[idx], moves)
        v.append((verts * w[None, :, None]).sum(axis=1))
        info.append({"n_vertices": int(idx.size), "nearest_vertex": dmin})
    length = np.linalg.norm(v[0] - v[1], axis=1)
    eps = 1.0 - length / length[0]
    mid = (v[0] + v[1]) / 2.0
    return eps, np.linalg.norm(mid - mid[0], axis=1), info


def static_shortening(bind, deformed, electrodes, radius, layer):
    """Shortening between the two virtual electrodes for one fixed deformed mesh (e.g. a target pose)."""
    v0, v1 = [], []
    for p in electrodes:
        idx, w, _ = surface_patch(bind, p, radius, layer)
        v0.append(w @ bind[idx])
        v1.append(w @ deformed[idx])
    return 1.0 - np.linalg.norm(v1[0] - v1[1]) / np.linalg.norm(v0[0] - v0[1])


def pair_signals(path, electrodes, radius):
    """Shortening and translation over time for one electrode pair on one muscle path."""
    v = []
    info = []
    for p in electrodes:
        idx, w, dmin = weights_for_point(path[0], p, radius)
        v.append(virtual_path(path, idx, w))
        info.append({"n_landmarks": int(idx.size), "nearest_landmark": dmin})
    length = np.linalg.norm(v[0] - v[1], axis=1)
    eps = 1.0 - length / length[0]
    mid = (v[0] + v[1]) / 2.0
    trans = np.linalg.norm(mid - mid[0], axis=1)
    return eps, trans, info
