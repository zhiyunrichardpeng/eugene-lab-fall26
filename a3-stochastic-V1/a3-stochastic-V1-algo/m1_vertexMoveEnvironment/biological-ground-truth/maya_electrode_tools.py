"""
=============================================================================
Maya side of the electrode placement (step 1 of the virtual-electrode design)
=============================================================================
Runs inside Maya (Python 3, no numpy needed). The geometry itself lives in
electrode_placement.py, which runs outside Maya.

1. Open the Ziva scene at the neutral pose (AU0) and, in the Script Editor:

       import sys; sys.path.append(r"<this folder>")
       import maya_electrode_tools as met
       met.export_mesh("skin_mesh", r"<this folder>/electrodes/skin_mesh.json")
       met.LandmarkPicker("skin_mesh", r"<this folder>/electrodes/landmarks.json")

   In the picker, select one vertex on the skin, press "Set" on the matching
   row, repeat for all 19 landmarks, then "Save". Left/right are the SUBJECT's
   sides, not the viewer's. "Show" drops locators to check the picks.

2. Outside Maya (this folder):

       python electrode_placement.py electrodes/skin_mesh.json electrodes/landmarks.json -o electrodes/electrodes.json

3. Back in Maya, check the result on the mesh:

       met.show_electrodes(r"<this folder>/electrodes/electrodes.json", "skin_mesh")
=============================================================================
"""

import json
import os
import re

import maya.api.OpenMaya as om
import maya.cmds as cmds

# Same names and order as electrode_placement.LANDMARKS (kept in sync by hand: Maya has no numpy).
SIDED = (
    ("endocanthion", "inner corner of the eye fissure"),
    ("exocanthion", "outer corner of the eye fissure"),
    ("pupil", "skin in front of the pupil centre, centered gaze"),
    ("brow_inner", "upper border of the brow, straight above the endocanthion"),
    ("brow_middle", "upper border of the middle of the brow"),
    ("alar_curvature", "most lateral point of the nostril baseline"),
    ("cheilion", "lip corner"),
    ("gonion", "most lateral point of the jaw angle"),
    ("preauricular", "bony dimple above the back of the zygomatic arch, in front of the ear"),
)
LANDMARKS = tuple((f"{name}_{s}", f"{text} (subject {'left' if s == 'l' else 'right'})")
                  for name, text in SIDED for s in "lr") + (("pogonion", "most anterior midpoint of the chin"),)


def _fn_mesh(mesh):
    sel = om.MSelectionList()
    sel.add(mesh)
    return om.MFnMesh(sel.getDagPath(0))


def _points(mesh):
    return [(p.x, p.y, p.z) for p in _fn_mesh(mesh).getPoints(om.MSpace.kWorld)]


def export_mesh(mesh, path):
    """Write world-space vertices and triangles of mesh to JSON (export at the neutral pose)."""
    fn = _fn_mesh(mesh)
    _, ids = fn.getTriangles()
    ids = list(ids)
    data = {"mesh": mesh, "vertices": _points(mesh),
            "triangles": [ids[i:i + 3] for i in range(0, len(ids), 3)]}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh)
    print(f"exported {mesh}: {len(data['vertices'])} vertices, {len(data['triangles'])} triangles -> {path}")


class LandmarkPicker:
    """Window: one row per landmark; "Set" stores the selected vertex of the mesh."""

    WINDOW = "electrodeLandmarkPicker"

    def __init__(self, mesh, path):
        self.mesh, self.path = mesh, path
        self.picked = {}
        if os.path.exists(path):
            with open(path) as fh:
                self.picked = {k: int(v) for k, v in json.load(fh).get("landmarks", {}).items()}
        if cmds.window(self.WINDOW, exists=True):
            cmds.deleteUI(self.WINDOW)
        cmds.window(self.WINDOW, title=f"Landmarks on {mesh}", widthHeight=(560, 640))
        cmds.scrollLayout(childResizable=True)
        cmds.columnLayout(adjustableColumn=True, rowSpacing=2)
        self.fields = {}
        for name, text in LANDMARKS:
            cmds.rowLayout(numberOfColumns=4, columnWidth4=(130, 80, 50, 50), annotation=text)
            cmds.text(label=name, align="left", annotation=text)
            self.fields[name] = cmds.textField(text=str(self.picked.get(name, "")), editable=False)
            cmds.button(label="Set", command=lambda _, n=name: self.set_from_selection(n))
            cmds.button(label="Select", command=lambda _, n=name: self.select(n))
            cmds.setParent("..")
        cmds.separator(height=8)
        cmds.rowLayout(numberOfColumns=2)
        cmds.button(label="Save", width=120, command=lambda _: self.save())
        cmds.button(label="Show", width=120, command=lambda _: self.show())
        cmds.setParent("..")
        cmds.showWindow(self.WINDOW)

    def set_from_selection(self, name):
        sel = cmds.ls(selection=True, flatten=True) or []
        vtx = [s for s in sel if ".vtx[" in s]
        if len(vtx) != 1 or vtx[0].split(".vtx[")[0].split("|")[-1] != self.mesh.split("|")[-1]:
            cmds.warning(f"select exactly one vertex on {self.mesh}")
            return
        self.picked[name] = int(re.search(r"\[(\d+)\]", vtx[0]).group(1))
        cmds.textField(self.fields[name], edit=True, text=str(self.picked[name]))

    def select(self, name):
        if name in self.picked:
            cmds.select(f"{self.mesh}.vtx[{self.picked[name]}]")

    def save(self):
        missing = [n for n, _ in LANDMARKS if n not in self.picked]
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump({"mesh": self.mesh, "landmarks": self.picked}, fh, indent=1)
        print(f"saved {len(self.picked)} landmarks -> {self.path}"
              + (f"; still missing: {', '.join(missing)}" if missing else ""))

    def show(self):
        points = _points(self.mesh)
        group = _fresh_group("electrode_landmarks_GRP")
        for name, idx in self.picked.items():
            loc = cmds.spaceLocator(name=f"lm_{name}")[0]
            cmds.xform(loc, worldSpace=True, translation=points[idx])
            cmds.setAttr(f"{loc}.localScale", 2, 2, 2)
            cmds.parent(loc, group)


def _fresh_group(name):
    if cmds.objExists(name):
        cmds.delete(name)
    return cmds.group(empty=True, name=name)


def show_electrodes(path, mesh, radius=1.5):
    """Spheres at each electrode on the mesh's CURRENT shape (works on deformed poses too), pairs joined by a line."""
    with open(path) as fh:
        data = json.load(fh)
    points = _points(mesh)
    group = _fresh_group("electrodes_GRP")
    for channel, entry in data["channels"].items():
        tag = re.sub(r"[^A-Za-z0-9_]", "", channel.replace(" ", "_"))
        ends = []
        for i, e in enumerate(entry["electrodes"], start=1):
            pos = [sum(w * points[v][axis] for w, v in zip(e["barycentric"], e["vertices"])) for axis in range(3)]
            sphere = cmds.polySphere(name=f"elec_{tag}_{i}", radius=radius, subdivisionsX=8, subdivisionsY=6)[0]
            cmds.xform(sphere, worldSpace=True, translation=pos)
            cmds.parent(sphere, group)
            ends.append(pos)
        line = cmds.curve(name=f"pair_{tag}", degree=1, point=ends)
        cmds.parent(line, group)
    print(f"placed {2 * len(data['channels'])} electrodes from {path}")
