"""Tests for electrode_placement.py on a synthetic spherical head (no Maya, no MaM data).

The head is a UV sphere facing +z, with +y up and +x the subject's left, in mm.
Run from this folder:  python -m pytest test_electrode_placement.py
"""

import json

import numpy as np
import pytest

import electrode_placement as ep
import mam_data as m

RADIUS = 80.0


def uv_sphere(radius=RADIUS, rings=90, segments=120, center=(0.0, 0.0, 0.0)):
    """Closed triangle mesh; an even segment count makes it mirror-symmetric in x."""
    theta = np.linspace(0, np.pi, rings + 1)[1:-1]
    phi = np.arange(segments) * 2 * np.pi / segments
    t, p = np.meshgrid(theta, phi, indexing="ij")
    ring = np.stack([np.sin(t) * np.cos(p), np.cos(t), np.sin(t) * np.sin(p)], axis=-1).reshape(-1, 3)
    verts = np.vstack([[0, 1, 0], ring, [0, -1, 0]]) * radius + np.asarray(center)
    top, bottom = 0, len(verts) - 1
    idx = lambda r, s: 1 + r * segments + s % segments
    tris = []
    for s in range(segments):
        tris.append((top, idx(0, s + 1), idx(0, s)))
        tris.append((bottom, idx(rings - 2, s), idx(rings - 2, s + 1)))
        for r in range(rings - 2):
            a, b, c, d = idx(r, s), idx(r, s + 1), idx(r + 1, s), idx(r + 1, s + 1)
            tris += [(a, b, d), (a, d, c)]
    return verts, np.asarray(tris)


def on_sphere(x, y):
    """Point on the front of the sphere at subject-left x and height y."""
    return np.array([x, y, np.sqrt(RADIUS**2 - x**2 - y**2)])


# Rough adult proportions (mm), mirrored for the right side.
LANDMARK_XY = {
    "endocanthion": (16, 10), "exocanthion": (45, 8), "pupil": (31, 9),
    "brow_inner": (17, 24), "brow_middle": (33, 28), "alar_curvature": (17, -20),
    "cheilion": (25, -42), "gonion": (58, -45), "preauricular": (68, 10),
}


def synthetic_landmarks(verts):
    """Landmark name -> index of the vertex nearest the intended spot."""
    points = {"pogonion": on_sphere(0, -65)}
    for name, (x, y) in LANDMARK_XY.items():
        points[f"{name}_l"], points[f"{name}_r"] = on_sphere(x, y), on_sphere(-x, y)
    return {name: int(np.argmin(np.linalg.norm(verts - p, axis=1))) for name, p in points.items()}


@pytest.fixture(scope="module")
def head():
    verts, tris = uv_sphere()
    return ep.Surface(verts, tris), synthetic_landmarks(verts)


@pytest.fixture(scope="module")
def placement(head):
    surface, landmarks = head
    return ep.place_electrodes(surface, landmarks)


@pytest.fixture(scope="module")
def fridlund(head):
    surface, landmarks = head
    return ep.place_electrodes(surface, landmarks, levator_extra_cm=0, zygomaticus_extra_cm=0)


def test_every_mam_channel_gets_one_pair(placement):
    assert set(placement.channels) == set(m.CHANNELS)
    assert all(len(pair) == 2 for pair in placement.channels.values())


def test_landmark_set_is_complete_and_checked(head):
    surface, landmarks = head
    assert set(landmarks) == set(ep.LANDMARKS)
    with pytest.raises(ValueError, match="cheilion_l"):
        ep.place_electrodes(surface, {k: v for k, v in landmarks.items() if k != "cheilion_l"})


def test_head_frame_axes(head):
    surface, landmarks = head
    frame = ep.HeadFrame.from_landmarks(ep.landmark_positions(surface, landmarks))
    np.testing.assert_allclose(frame.left, [1, 0, 0], atol=0.02)
    np.testing.assert_allclose(frame.up, [0, 1, 0], atol=0.02)
    np.testing.assert_allclose(frame.forward, [0, 0, 1], atol=0.02)


def test_swapped_sides_are_rejected(head):
    surface, landmarks = head
    swapped = {k[:-1] + {"l": "r", "r": "l"}[k[-1]] if k[-2:] in ("_l", "_r") else k: v for k, v in landmarks.items()}
    with pytest.raises(ValueError, match="left"):
        ep.place_electrodes(surface, swapped)


def test_electrodes_lie_on_the_skin(placement):
    for pair in placement.channels.values():
        for e in pair:
            assert abs(np.linalg.norm(e.rest_position) - RADIUS) < 0.1


def test_pairs_are_about_one_centimetre_apart(placement):
    for channel, pair in placement.channels.items():
        assert 9.0 < placement.spacing(channel) < 11.5, channel


def test_left_and_right_are_mirror_images(placement):
    mirror = np.array([-1.0, 1.0, 1.0])
    for site in m.SITES:
        left, right = placement.channels[f"{site}_l"], placement.channels[f"{site}_r"]
        for el, er in zip(left, right):
            np.testing.assert_allclose(el.rest_position * mirror, er.rest_position, atol=1.0, err_msg=site)


def test_mirror_mismatch_is_small_on_a_symmetric_head(head, placement):
    surface, landmarks = head
    frame = ep.HeadFrame.from_landmarks(ep.landmark_positions(surface, landmarks))
    gaps = placement.mirror_mismatch(frame)
    assert set(gaps) == set(m.SITES) and max(gaps.values()) < 1.0


def test_mirror_mismatch_catches_an_asymmetric_pick(head):
    surface, landmarks = head
    verts = surface.vertices
    moved = dict(landmarks, cheilion_l=int(np.argmin(np.linalg.norm(verts - on_sphere(25, -32), axis=1))))  # 1 cm high
    gaps = ep.place_electrodes(surface, moved).mirror_mismatch(
        ep.HeadFrame.from_landmarks(ep.landmark_positions(surface, moved)))
    assert gaps["Orbicularis Oris"] > 8 and gaps["Lateraler Frontalis"] < 1.0


def test_rules_follow_the_guideline_geometry(head, fridlund):
    """Distances are along the skin (about the chord at 1 cm), directions in the head frame."""
    placement = fridlund
    surface, landmarks = head
    pos = ep.landmark_positions(surface, landmarks)
    oris1, oris2 = (e.rest_position for e in placement.channels["Orbicularis Oris_l"])
    assert np.linalg.norm(pos["cheilion_l"] - oris1) == pytest.approx(10, abs=0.5)  # 1 cm below the cheilion
    assert oris1[1] < pos["cheilion_l"][1] - 7 and abs(oris1[0] - pos["cheilion_l"][0]) < 0.5
    assert oris2[0] < oris1[0]  # second electrode medial (toward the midline)
    zyg1, zyg2 = (e.rest_position for e in placement.channels["Zygomaticus_l"])
    midpoint = (pos["cheilion_l"] + pos["preauricular_l"]) / 2
    assert np.linalg.norm(zyg1[:2] - midpoint[:2]) < 1.5  # midway along cheilion-preauricular
    assert zyg2[0] < zyg1[0] and zyg2[1] < zyg1[1]  # 1 cm inferior and medial
    lat1, lat2 = (e.rest_position for e in placement.channels["Lateraler Frontalis_l"])
    corner = np.array([pos["pupil_l"][0], pos["brow_middle_l"][1]])  # pupil vertical x brow height
    assert np.linalg.norm(lat1 - ep.Surface(surface.vertices, surface.triangles).project(
        np.r_[corner, pos["brow_middle_l"][2]], np.zeros(3)).position) == pytest.approx(np.hypot(10, 10), abs=1)
    assert lat1[0] > corner[0] + 7 and lat1[1] > corner[1] + 7  # lateral and superior
    assert lat2[1] > lat1[1] + 7 and abs(lat2[0] - lat1[0]) < 1  # second straight above
    ment1, ment2 = (e.rest_position for e in placement.channels["Mentalis_l"])
    assert ment1[0] == pytest.approx(5, abs=1) and ment1[1] > pos["pogonion"][1] > ment2[1]


def test_rigid_motion_moves_electrodes_with_the_head(head, placement):
    surface, landmarks = head
    angle = np.deg2rad(30)
    rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    shift = np.array([5.0, -120.0, 40.0])
    moved = ep.place_electrodes(ep.Surface(surface.vertices @ rot.T + shift, surface.triangles), landmarks)
    for channel, pair in placement.channels.items():
        for e, e_moved in zip(pair, moved.channels[channel]):
            np.testing.assert_allclose(e.rest_position @ rot.T + shift, e_moved.rest_position, atol=1e-6)


def test_barycentric_positions_follow_a_deformed_mesh(head, placement):
    surface, _ = head
    np.testing.assert_allclose(placement.positions(surface.vertices)["Masseter_r"],
                               [e.rest_position for e in placement.channels["Masseter_r"]], atol=1e-9)
    shifted = placement.positions(surface.vertices + [0.0, 3.0, 0.0])
    np.testing.assert_allclose(shifted["Masseter_r"][:, 1],
                               [e.rest_position[1] + 3.0 for e in placement.channels["Masseter_r"]], atol=1e-9)


def test_projection_takes_the_outermost_surface():
    outer_v, outer_t = uv_sphere(RADIUS)
    inner_v, inner_t = uv_sphere(RADIUS * 0.75)
    surface = ep.Surface(np.vstack([outer_v, inner_v]), np.vstack([outer_t, inner_t + len(outer_v)]))
    hit = surface.project_radial(np.array([10.0, 5.0, 50.0]), center=np.zeros(3))
    assert abs(np.linalg.norm(hit.position) - RADIUS) < 0.1


def test_json_round_trip(tmp_path, placement):
    path = tmp_path / "electrodes.json"
    placement.save(path, mesh_name="skin_mesh")
    loaded = ep.Placement.load(path)
    assert json.loads(path.read_text())["mesh"] == "skin_mesh"
    for channel, pair in placement.channels.items():
        for a, b in zip(pair, loaded.channels[channel]):
            assert a.triangle == b.triangle
            np.testing.assert_allclose(a.barycentric, b.barycentric)
            np.testing.assert_allclose(a.rest_position, b.rest_position)


def test_maya_tool_uses_the_same_landmarks(monkeypatch):
    import sys
    import types
    maya = types.ModuleType("maya")
    maya.cmds = types.ModuleType("maya.cmds")
    maya.api = types.ModuleType("maya.api")
    maya.api.OpenMaya = types.ModuleType("maya.api.OpenMaya")
    for name, mod in {"maya": maya, "maya.cmds": maya.cmds, "maya.api": maya.api,
                      "maya.api.OpenMaya": maya.api.OpenMaya}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "maya_electrode_tools", raising=False)
    import maya_electrode_tools as met
    assert [n for n, _ in met.LANDMARKS] == list(ep.LANDMARKS)


def test_cli_reads_the_maya_export_formats(tmp_path):
    """Files shaped like export_mesh / LandmarkPicker.save output go through the CLI end to end."""
    import subprocess
    import sys
    from pathlib import Path
    verts, tris = uv_sphere()
    (tmp_path / "skin_mesh.json").write_text(json.dumps(
        {"mesh": "skin_mesh", "vertices": verts.tolist(), "triangles": tris.tolist()}))
    (tmp_path / "landmarks.json").write_text(json.dumps({"mesh": "skin_mesh", "landmarks": synthetic_landmarks(verts)}))
    out = tmp_path / "electrodes.json"
    run = subprocess.run([sys.executable, "electrode_placement.py", str(tmp_path / "skin_mesh.json"),
                          str(tmp_path / "landmarks.json"), "-o", str(out)],
                         cwd=Path(__file__).parent, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "<- check" not in run.stdout
    assert set(json.loads(out.read_text())["channels"]) == set(m.CHANNELS)


def test_normal_near_averages_over_a_crease():
    # A convex 90 degree edge along x, like a lip corner: the top strip faces up (+y), the front one +z.
    xs = np.linspace(-10, 10, 11)
    up = [(x, 0.0, z) for z in np.linspace(-10, 0, 6) for x in xs]
    front = [(x, y, 0.0) for y in np.linspace(0, -10, 6)[1:] for x in xs]
    verts = np.array(up + front)
    n = len(xs)
    rows = [list(range(r * n, (r + 1) * n)) for r in range(len(verts) // n)]
    order = rows  # one grid: back of the top strip, over the edge, down the front
    tris = []
    for r0, r1 in zip(order, order[1:]):
        for i in range(n - 1):
            tris += [(r0[i], r0[i + 1], r1[i + 1]), (r0[i], r1[i + 1], r1[i])]
    surface = ep.Surface(verts, np.asarray(tris))
    normal = surface.normal_near(np.zeros(3), 5.0, center=np.array([0.0, -50.0, -50.0]))
    # One triangle at the fold faces straight up or straight forward; the patch average is in between.
    assert normal == pytest.approx(np.array([0, 1, 1]) / np.sqrt(2), abs=0.05)


def test_model_shifts_move_levator_and_zygomaticus_outward(head, placement, fridlund):
    surface, landmarks = head
    pos = ep.landmark_positions(surface, landmarks)
    lev, lev0 = (c.channels["Levator Labii Superioris_l"][0].rest_position for c in (placement, fridlund))
    assert lev[0] - lev0[0] == pytest.approx(ep.LEVATOR_EXTRA_LATERAL_CM * ep.UNITS_PER_CM, rel=0.25)
    zyg, zyg0 = (c.channels["Zygomaticus_l"][0].rest_position for c in (placement, fridlund))
    to_ear = ep.unit(pos["preauricular_l"] - pos["cheilion_l"])
    assert (zyg - zyg0) @ to_ear == pytest.approx(ep.ZYGOMATICUS_EXTRA_TOWARD_EAR_CM * ep.UNITS_PER_CM, rel=0.25)
    assert placement.spacing("Zygomaticus_l") == pytest.approx(ep.UNITS_PER_CM, rel=0.1)


def test_respace_brings_pairs_into_range_and_keeps_manual_moves(head, placement, tmp_path):
    surface, landmarks = head
    path = tmp_path / "electrodes.json"
    placement.save(path, "sphere")
    data = json.loads(path.read_text())
    zyg = data["channels"]["Zygomaticus_l"]["electrodes"]
    first_before = list(zyg[0]["rest_position"])
    zyg[0]["manual"] = True
    a, b = (np.asarray(e["rest_position"]) for e in zyg)
    far = surface.project(a + 1.4 * (b - a), np.zeros(3))  # pull the pair to ~1.4 cm
    zyg[1].update(triangle=far.triangle, vertices=surface.triangles[far.triangle].tolist(),
                  barycentric=far.barycentric.tolist(), rest_position=far.position.tolist())
    changed = ep.respace(data, surface, np.zeros(3))
    assert any(c.startswith("Zygomaticus_l") for c in changed)
    for entry in data["channels"].values():
        p, q = (np.asarray(e["rest_position"]) for e in entry["electrodes"])
        assert 0.95 * ep.UNITS_PER_CM <= np.linalg.norm(p - q) <= 1.05 * ep.UNITS_PER_CM
    assert zyg[0]["rest_position"] == first_before and zyg[0]["manual"]  # anchor and flag untouched
    new_dir = ep.unit(np.asarray(zyg[1]["rest_position"]) - a)
    assert new_dir @ ep.unit(b - a) > 0.99  # same pair orientation
