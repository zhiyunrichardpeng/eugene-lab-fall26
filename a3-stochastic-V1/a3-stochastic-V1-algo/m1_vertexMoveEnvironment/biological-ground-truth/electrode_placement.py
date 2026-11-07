"""
=============================================================================
MaM electrode pairs on our skin mesh (step 1 of the virtual-electrode design)
=============================================================================
Places both electrodes of every MaM sEMG channel on the model's skin_mesh,
following the bipolar placement atlas of Fridlund & Cacioppo (1986), Fig. 1 and
its text (p. 571). Each rule is a centimetre offset from an anthropometric
landmark (Farkas 1981): endocanthion, exocanthion, cheilion, gonion, ...

Input  skin mesh exported from Maya (maya_electrode_tools.export_mesh) and the
       landmark vertices picked on it (maya_electrode_tools.LandmarkPicker).
Output one pair per channel in mam_data.CHANNELS, each electrode stored as a
       skin triangle plus barycentric weights, so its position follows any
       deformed copy of the mesh (Placement.positions).

Offsets are taken in a head frame built from the landmarks, then the target is
dropped onto the outer skin along the local surface normal (a ray from inside
the head, at the gonion midpoint, picks the outer skin over inner sheets such as
the mouth bag). Every pair spacing is printed so the 1 cm rules can
be checked on the real mesh.

Interpretations of the guideline (my calls, flagged in the output):
  - "slightly" offsets use SLIGHT_CM = 0.2 cm; the guideline gives no number.
  - Corrugator and depressor supercilii/procerus share their first electrode
    position, as the guideline literally places both "directly above the brow
    on the vertical through the endocanthion".
  - Depressor supercilii/procerus second electrode: 1 cm medial-inferior,
    "on the side of the nasal bone".
  - "Directly above the brow" = the picked brow upper border point.
  - Distances are along the skin; directions are as seen from the front where
    the skin faces forward. "Midway along cheilion-preauricular" is the skin
    point in front of the straight-line midpoint.

Usage (from this folder):
    python electrode_placement.py skin_mesh.json landmarks.json -o electrodes.json

No MaM data is read; outputs describe our model only.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mam_data as m

UNITS_PER_CM = 10.0  # the Maya scene is in mm
SLIGHT_CM = 0.2
FRONT_FACING = 0.5  # cos of the angle between skin normal and the forward axis
SOURCE = "Fridlund & Cacioppo (1986), Guidelines for Human Electromyographic Research, Fig. 1 and p. 571"

SIDED_LANDMARKS = {
    "endocanthion": "inner corner of the eye fissure",
    "exocanthion": "outer corner of the eye fissure",
    "pupil": "skin point in front of the pupil centre, centered gaze (sets the pupil vertical)",
    "brow_inner": "upper border of the brow, straight above the endocanthion",
    "brow_middle": "upper border of the middle of the brow",
    "alar_curvature": "most lateral point of the nostril (ala nasi) baseline",
    "cheilion": "lip corner",
    "gonion": "most lateral point of the jaw angle",
    "preauricular": "bony dimple above the posterior edge of the zygomatic arch, in front of the ear",
}
MIDLINE_LANDMARKS = {"pogonion": "most anterior midpoint of the chin"}
LANDMARKS = {f"{name}_{side}": f"{text} ({'subject left' if side == 'l' else 'subject right'})"
             for name, text in SIDED_LANDMARKS.items() for side in "lr"} | MIDLINE_LANDMARKS


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


@dataclass(frozen=True)
class Hit:
    triangle: int
    barycentric: np.ndarray  # (3,), weights of the triangle's vertices
    position: np.ndarray  # (3,)


class Surface:
    """Triangle mesh: vertices (V, 3), triangles (T, 3) vertex indices."""

    def __init__(self, vertices, triangles):
        self.vertices = np.asarray(vertices, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (V, 3), got {self.vertices.shape}")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(f"triangles must be (T, 3), got {self.triangles.shape}")

    @classmethod
    def load(cls, path: Path) -> tuple[Surface, str]:
        data = json.loads(Path(path).read_text())
        return cls(data["vertices"], data["triangles"]), data.get("mesh", "")

    def _line_hits(self, origin: np.ndarray, direction: np.ndarray):
        """All triangle crossings of the line origin + t * direction, any sign of t (Moller-Trumbore)."""
        a, b, c = (self.vertices[self.triangles[:, k]] for k in range(3))
        e1, e2 = b - a, c - a
        pvec = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tvec = origin - a
        u = np.einsum("ij,ij->i", tvec, pvec) * inv
        qvec = np.cross(tvec, e1)
        v = (qvec @ direction) * inv
        t = np.einsum("ij,ij->i", e2, qvec) * inv
        eps = 1e-9
        k = np.flatnonzero(ok & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps))
        return k, t[k], u[k], v[k]

    def _hit(self, k: int, u: float, v: float, position: np.ndarray) -> Hit:
        bary = np.clip(np.array([1 - u - v, u, v]), 0, 1)
        return Hit(int(k), bary / bary.sum(), position)

    def project_radial(self, point: np.ndarray, center: np.ndarray) -> Hit:
        """Outermost crossing of the ray from center (inside the head) through point."""
        direction = unit(point - center)
        k, t, u, v = self._line_hits(center, direction)
        out = t > 0
        if not out.any():
            raise ValueError(f"no skin found along the ray from {center} through {point}")
        i = np.flatnonzero(out)[np.argmax(t[out])]
        return self._hit(k[i], u[i], v[i], center + t[i] * direction)

    def project_frontal(self, point: np.ndarray, forward: np.ndarray) -> Hit:
        """Frontmost crossing of the line through point along forward: the skin spot seen in front of it."""
        k, t, u, v = self._line_hits(point, forward)
        if not len(k):
            raise ValueError(f"no skin in front of or behind {point}")
        i = int(np.argmax(t))
        return self._hit(k[i], u[i], v[i], point + t[i] * forward)

    def project(self, point: np.ndarray, center: np.ndarray) -> Hit:
        """Point dropped onto the outer skin along the local surface normal.

        The radial ray only finds the outer skin near the point; the final position comes from the
        line through the point along that triangle's normal, so tangential offsets keep their length.
        """
        near = self.project_radial(point, center)
        a, b, c = self.vertices[self.triangles[near.triangle]]
        normal = unit(np.cross(b - a, c - a))
        k, t, u, v = self._line_hits(point, normal)
        positions = point + t[:, None] * normal
        i = int(np.argmin(np.linalg.norm(positions - near.position, axis=1)))  # the outer sheet, not inner ones
        return self._hit(k[i], u[i], v[i], positions[i])


@dataclass(frozen=True)
class HeadFrame:
    origin: np.ndarray  # gonion midpoint, inside the head; rays start here
    left: np.ndarray  # toward the subject's left
    up: np.ndarray
    forward: np.ndarray  # direction the face looks

    @classmethod
    def from_landmarks(cls, pos: dict[str, np.ndarray]) -> HeadFrame:
        mid = lambda name: (pos[f"{name}_l"] + pos[f"{name}_r"]) / 2
        left = unit(pos["exocanthion_l"] - pos["exocanthion_r"])
        forward = mid("endocanthion") - mid("preauricular")  # close to the Frankfort horizontal
        forward = unit(forward - (forward @ left) * left)
        up = np.cross(forward, left)
        frame = cls(mid("gonion"), left, up, forward)
        if (pos["pogonion"] - mid("endocanthion")) @ up >= 0 or (mid("cheilion") - mid("preauricular")) @ forward <= 0:
            raise ValueError("landmarks put the chin above the eyes or the mouth behind the ears: "
                             "left and right look swapped (pick from the subject's left, not the viewer's)")
        return frame

    def lateral(self, side: str) -> np.ndarray:
        return self.left if side == "l" else -self.left


def landmark_positions(surface: Surface, landmarks: dict[str, int]) -> dict[str, np.ndarray]:
    missing = sorted(set(LANDMARKS) - set(landmarks))
    if missing:
        raise ValueError(f"missing landmarks: {', '.join(missing)}")
    return {name: surface.vertices[int(landmarks[name])] for name in LANDMARKS}


@dataclass(frozen=True)
class Electrode:
    triangle: int
    vertices: np.ndarray  # (3,) skin vertex ids of that triangle
    barycentric: np.ndarray
    rest_position: np.ndarray


class Placement:
    def __init__(self, channels: dict[str, tuple[Electrode, Electrode]], landmarks: dict[str, int],
                 units_per_cm: float = UNITS_PER_CM, slight_cm: float = SLIGHT_CM):
        self.channels = channels
        self.landmarks = landmarks
        self.units_per_cm = units_per_cm
        self.slight_cm = slight_cm

    def spacing(self, channel: str) -> float:
        a, b = self.channels[channel]
        return float(np.linalg.norm(a.rest_position - b.rest_position))

    def positions(self, vertices: np.ndarray) -> dict[str, np.ndarray]:
        """Channel -> (2, 3) electrode positions on a deformed copy of the skin (same topology)."""
        vertices = np.asarray(vertices, dtype=float)
        return {ch: np.array([e.barycentric @ vertices[e.vertices] for e in pair])
                for ch, pair in self.channels.items()}

    def save(self, path: Path, mesh_name: str) -> None:
        data = {
            "mesh": mesh_name,
            "source": SOURCE,
            "units_per_cm": self.units_per_cm,
            "slight_cm": self.slight_cm,
            "landmarks": {k: int(v) for k, v in self.landmarks.items()},
            "channels": {ch: {"spacing": round(self.spacing(ch), 4),
                              "electrodes": [{"triangle": e.triangle, "vertices": e.vertices.tolist(),
                                              "barycentric": e.barycentric.tolist(),
                                              "rest_position": e.rest_position.tolist()} for e in pair]}
                         for ch, pair in self.channels.items()},
        }
        Path(path).write_text(json.dumps(data, indent=1))

    @classmethod
    def load(cls, path: Path) -> Placement:
        data = json.loads(Path(path).read_text())
        channels = {ch: tuple(Electrode(e["triangle"], np.asarray(e["vertices"]), np.asarray(e["barycentric"]), np.asarray(e["rest_position"]))
                              for e in entry["electrodes"]) for ch, entry in data["channels"].items()}
        return cls(channels, data["landmarks"], data["units_per_cm"], data["slight_cm"])


def place_electrodes(surface: Surface, landmarks: dict[str, int], units_per_cm: float = UNITS_PER_CM,
                     slight_cm: float = SLIGHT_CM) -> Placement:
    """One electrode pair per MaM channel, by the Fridlund & Cacioppo (1986) rules."""
    pos = landmark_positions(surface, landmarks)
    frame = HeadFrame.from_landmarks(pos)
    cm, slight = units_per_cm, slight_cm * units_per_cm

    def snap(target: np.ndarray) -> Electrode:
        hit = surface.project(target, frame.origin)
        return Electrode(hit.triangle, surface.triangles[hit.triangle].copy(), hit.barycentric, hit.position)

    def walk(start: np.ndarray, *steps: tuple[float, np.ndarray]) -> Electrode:
        """Move from start by the summed (distance, direction) steps, each measured along the skin.

        The offset is turned into the tangent plane at start and keeps its length, so "1 cm below"
        means 1 cm on the skin, as with a ruler on the face, not a 1 cm vertical drop.
        Where the skin faces forward, the out-of-plane part is removed along the forward axis, so the
        step keeps its direction as seen from the front ("vertical lines" are frontal-view lines);
        where it faces sideways (masseter), it is removed along the normal.
        """
        here = snap(start)
        a, b, c = surface.vertices[here.vertices]
        normal = unit(np.cross(b - a, c - a))
        if normal @ (here.rest_position - frame.origin) < 0:
            normal = -normal  # point out of the head whatever the triangle winding
        slide = frame.forward if normal @ frame.forward > FRONT_FACING else normal
        on_skin = lambda v: unit(v - (v @ normal) / (slide @ normal) * slide)
        return snap(here.rest_position + sum(d * on_skin(direction) for d, direction in steps))

    def on_vertical(height_point: np.ndarray, vertical_point: np.ndarray) -> np.ndarray:
        """height_point moved sideways onto the vertical line through vertical_point."""
        return height_point + ((vertical_point - height_point) @ frame.left) * frame.left

    up, down = frame.up, -frame.up
    channels = {}
    for s in "lr":
        lat, med = frame.lateral(s), -frame.lateral(s)
        p = lambda name: pos[f"{name}_{s}"]
        rules = {}

        # Medial frontalis: vertical 1 cm medial of the endocanthion vertical; inferior electrode 1.5 cm
        # above the upper edge of the inner brow; second 1 cm superior and slightly lateral.
        e1 = walk(on_vertical(p("brow_inner"), p("endocanthion")), (cm, med), (1.5 * cm, up))
        rules["Medialer Frontalis"] = (e1, walk(e1.rest_position, (cm, up), (slight, lat)))

        # Lateral frontalis: vertical 1 cm lateral of the pupil vertical; inferior electrode 1 cm above the
        # upper border of the middle of the brow; second 1 cm superior.
        e1 = walk(on_vertical(p("brow_middle"), p("pupil")), (cm, lat), (cm, up))
        rules["Lateraler Frontalis"] = (e1, walk(e1.rest_position, (cm, up)))

        # Corrugator: directly above the brow on the endocanthion vertical; second 1 cm lateral and
        # slightly superior, on the border of the eyebrow.
        brow_above_en = snap(on_vertical(p("brow_inner"), p("endocanthion")))
        rules["Corrugator Supercilii"] = (brow_above_en, walk(brow_above_en.rest_position, (cm, lat), (slight, up)))

        # Depressor supercilii / procerus: same first spot; second 1 cm away on the side of the nasal bone.
        rules["Depressor Supercilii Procerus"] = (brow_above_en, walk(brow_above_en.rest_position, (cm, unit(med + down))))

        # Orbicularis oculi (inferior orbital): 1 cm below the exocanthion; second 1 cm medial and
        # slightly inferior, parallel to the lower eyelid.
        e1 = walk(p("exocanthion"), (cm, down))
        rules["Orbicularis Oculi"] = (e1, walk(e1.rest_position, (cm, med), (slight, down)))

        # Levator labii superioris (+ alaeque nasi): 1 cm lateral of the ala nasi baseline, 0.5 cm below the
        # alar curvature point; second 1 cm superior and slightly lateral.
        e1 = walk(p("alar_curvature"), (cm, lat), (0.5 * cm, down))
        rules["Levator Labii Superioris"] = (e1, walk(e1.rest_position, (cm, up), (slight, lat)))

        # Zygomatic major: midway along cheilion-preauricular; second 1 cm toward the mouth on that line.
        e1 = snap(surface.project_frontal((p("cheilion") + p("preauricular")) / 2, frame.forward).position)
        rules["Zygomaticus"] = (e1, walk(e1.rest_position, (cm, unit(p("cheilion") - p("preauricular")))))

        # Orbicularis oris (inferior): 1 cm below the cheilion; second 1 cm medial and slightly inferior,
        # parallel to the lower lip border.
        e1 = walk(p("cheilion"), (cm, down))
        rules["Orbicularis Oris"] = (e1, walk(e1.rest_position, (cm, med), (slight, down)))

        # Depressor anguli oris: 1 cm lateral and 1 cm inferior to the cheilion; second 1 cm inferior.
        e1 = walk(p("cheilion"), (cm, lat), (cm, down))
        rules["Depressor Ang. Oris"] = (e1, walk(e1.rest_position, (cm, down)))

        # Mentalis: 0.5 cm lateral of the midline, 0.5 cm above and 0.5 cm below the pogonion.
        rules["Mentalis"] = (walk(pos["pogonion"], (0.5 * cm, lat), (0.5 * cm, up)),
                             walk(pos["pogonion"], (0.5 * cm, lat), (0.5 * cm, down)))

        # Masseter: on the gonion-exocanthion line, 2 cm from the gonion; second 1 cm superior and
        # slightly medial (toward the mouth).
        e1 = walk(p("gonion"), (2 * cm, unit(p("exocanthion") - p("gonion"))))
        toward_mouth = p("cheilion") - e1.rest_position
        toward_mouth = unit(toward_mouth - (toward_mouth @ up) * up)
        rules["Masseter"] = (e1, walk(e1.rest_position, (cm, up), (slight, toward_mouth)))

        if set(rules) != set(m.SITES):
            raise AssertionError(f"rules do not cover mam_data.SITES: {set(m.SITES) ^ set(rules)}")
        channels |= {f"{site}_{s}": pair for site, pair in rules.items()}

    return Placement({ch: channels[ch] for ch in m.CHANNELS}, dict(landmarks), units_per_cm, slight_cm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", type=Path, help="skin mesh JSON from maya_electrode_tools.export_mesh")
    parser.add_argument("landmarks", type=Path, help="landmark JSON from maya_electrode_tools.LandmarkPicker")
    parser.add_argument("-o", "--out", type=Path, default=Path("electrodes.json"))
    parser.add_argument("--units-per-cm", type=float, default=UNITS_PER_CM)
    args = parser.parse_args()

    surface, mesh_name = Surface.load(args.mesh)
    landmarks = json.loads(args.landmarks.read_text())["landmarks"]
    placement = place_electrodes(surface, landmarks, args.units_per_cm)
    placement.save(args.out, mesh_name)

    pos = landmark_positions(surface, landmarks)
    ex = np.linalg.norm(pos["exocanthion_l"] - pos["exocanthion_r"]) / args.units_per_cm
    print(f"exocanthion-exocanthion width: {ex:.1f} cm (adults are roughly 8-10 cm; far off means wrong units)")
    print(f"{'channel':36s} spacing (cm)")
    for ch in m.DISPLAY_CHANNELS:
        flag = "" if 0.85 <= placement.spacing(ch) / args.units_per_cm <= 1.2 else "   <- check"
        print(f"{ch:36s} {placement.spacing(ch) / args.units_per_cm:5.2f}{flag}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
