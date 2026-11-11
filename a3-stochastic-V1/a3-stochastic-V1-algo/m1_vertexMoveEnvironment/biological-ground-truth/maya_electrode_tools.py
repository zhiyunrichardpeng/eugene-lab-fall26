"""
=============================================================================
Maya side of the electrode placement (step 1 of the virtual-electrode design)
=============================================================================
Runs inside Maya (Python 3, no numpy needed). The geometry itself lives in
electrode_placement.py, which runs outside Maya. Outputs land in
electrodes/ next to this file unless a path is given.

0. Open the Ziva scene, go to the neutral (AU0) frame, then in the Script
   Editor (Python tab):

       import sys, importlib
       sys.path.append(r"<this folder>")
       import maya_electrode_tools as met; importlib.reload(met)

   The skin is |jack01_MODEL|skin_cloth (met.SKIN, the Ziva-attached skin;
   not the larger skin_cloth_copy_v5_pull_back* copies). It sits in the
   display layer final_render_layer, which may be hidden:
   met.show_skin() turns that layer on and isolates the skin in the viewport.

1. Export the skin and pick the 19 landmarks:

       met.export_mesh()          # -> electrodes/skin_mesh.json (gitignored)
       met.LandmarkPicker()       # -> electrodes/landmarks.json

   In vertex mode (select the skin, F9), click one vertex, press "Set" on its
   row; "Save" writes the file ("Save to" / "Browse..." change it, picks are
   kept). Left/right are the SUBJECT's sides: _l is on your right when you
   face the model.

2. Check the picks:

       met.check_landmarks()      # mirror table + labelled locators

   Rows marked "<- check" are swapped or far from mirror-symmetric.

3. Outside Maya (this folder):

       python electrode_placement.py electrodes/skin_mesh.json electrodes/landmarks.json \
           -o electrodes/electrodes.json --units-per-cm 15.8

   Fix any spacing or mirror flag it prints by re-picking those landmarks.

4. Back in Maya:

       met.show_electrodes()      # spheres + pair lines on the current shape

5. Optional manual realign, at the neutral frame, after met.show_electrodes():

       met.slide_on_skin()        # dragged spheres now slide on the skin
       # move elec_<channel>_1 / _2 spheres with the Move tool (W)
       met.save_electrodes()      # snap to the skin, write electrodes/electrodes.json
       met.slide_on_skin(False)   # back to normal moving

   Moved electrodes are marked "manual": true. Rerunning electrode_placement.py
   rewrites electrodes.json from the rules and drops manual moves, so commit
   the file right after saving.

   Delete electrodes_GRP / electrode_landmarks_GRP / lm_check_GRP, and turn
   final_render_layer back off, before saving the scene.
=============================================================================
"""

import json
import os
import re

import maya.api.OpenMaya as om
import maya.cmds as cmds

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "electrodes")
DEFAULT_MESH = os.path.join(OUT, "skin_mesh.json")
DEFAULT_LANDMARKS = os.path.join(OUT, "landmarks.json")
DEFAULT_ELECTRODES = os.path.join(OUT, "electrodes.json")
SKIN = "|jack01_MODEL|skin_cloth"
SKIN_LAYER = "final_render_layer"

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


def show_skin(mesh=SKIN):
    """Turn on the skin's display layer and isolate the skin in the focused viewport (not saved with the scene)."""
    if cmds.objExists(SKIN_LAYER):
        cmds.setAttr(SKIN_LAYER + ".visibility", True)
    cmds.select(mesh)
    panel = cmds.getPanel(withFocus=True)
    if cmds.getPanel(typeOf=panel) == "modelPanel":
        cmds.modelEditor(panel, edit=True, polymeshes=True)
        cmds.isolateSelect(panel, state=True)
        cmds.isolateSelect(panel, addSelected=True)
    cmds.viewFit()
    print(f"showing {mesh}; Ctrl+1 in the viewport leaves isolate mode")


def export_mesh(mesh=SKIN, path=DEFAULT_MESH):
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

    def __init__(self, mesh=SKIN, path=DEFAULT_LANDMARKS):
        self.mesh, self.path = mesh, path
        self.picked = {}
        if os.path.exists(self.path):
            with open(self.path) as fh:
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
        cmds.rowLayout(numberOfColumns=3, adjustableColumn=2, columnWidth3=(60, 380, 90))
        cmds.text(label="Save to", align="left")
        self.path_field = cmds.textField(text=self.path, changeCommand=lambda value: self.set_path(value))
        cmds.button(label="Browse...", command=lambda _: self.browse())
        cmds.setParent("..")
        cmds.rowLayout(numberOfColumns=2)
        cmds.button(label="Save", width=120, command=lambda _: self.save())
        cmds.button(label="Show", width=120, command=lambda _: self.show())
        cmds.setParent("..")
        cmds.showWindow(self.WINDOW)

    def set_path(self, path):
        """Change where Save writes; the picks made so far are kept."""
        self.path = os.path.abspath(path)
        cmds.textField(self.path_field, edit=True, text=self.path)
        print(f"landmarks will be saved to {self.path}")

    def browse(self):
        start = os.path.dirname(self.path) if os.path.isdir(os.path.dirname(self.path)) else HERE
        chosen = cmds.fileDialog2(fileMode=0, caption="Save landmarks as", fileFilter="JSON (*.json)",
                                  startingDirectory=start)
        if chosen:
            self.set_path(chosen[0])

    def set_from_selection(self, name):
        sel = cmds.ls(selection=True, flatten=True, long=True) or []
        vtx = [s for s in sel if ".vtx[" in s]
        if len(vtx) != 1 or _transform(vtx[0].split(".vtx[")[0]) != _transform(self.mesh):
            # Deformed meshes report vertices on their shape node (skin_clothShape.vtx[..]), so compare transforms.
            msg = f"{name}: select exactly one vertex on {self.mesh} (selection: {sel[:3]}{' ...' if len(sel) > 3 else ''})"
            print(msg)
            cmds.warning(msg)
            return
        self.picked[name] = int(re.search(r"\[(\d+)\]", vtx[0]).group(1))
        cmds.textField(self.fields[name], edit=True, text=str(self.picked[name]))
        print(f"{name} = vertex {self.picked[name]}")

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


def _transform(node):
    """Full path of the transform for a mesh transform or its shape node."""
    path = cmds.ls(node, long=True)[0]
    if cmds.nodeType(path) == "mesh":
        path = cmds.listRelatives(path, parent=True, fullPath=True)[0]
    return path


def _fresh_group(name):
    if cmds.objExists(name):
        cmds.delete(name)
    return cmds.group(empty=True, name=name)


def check_landmarks(path=DEFAULT_LANDMARKS, mesh=SKIN, tolerance=8.0):
    """Print a left/right mirror table of the picked landmarks and drop labelled locators (lm_check_GRP).

    x is the distance from the midline along the eye-corner axis (left +, right -); "asym" is how far
    the two sides are from mirror images there, "off-axis" the height/depth difference. Both in scene units.
    """
    with open(path) as fh:
        lm = {k: int(v) for k, v in json.load(fh)["landmarks"].items()}
    points = _points(mesh)
    pos = {k: points[v] for k, v in lm.items()}
    sub = lambda a, b: [a[i] - b[i] for i in range(3)]
    dot = lambda a, b: sum(a[i] * b[i] for i in range(3))
    d = sub(pos["exocanthion_l"], pos["exocanthion_r"])
    width = dot(d, d) ** 0.5
    left = [c / width for c in d]
    mid = [(pos["endocanthion_l"][i] + pos["endocanthion_r"][i]) / 2 for i in range(3)]
    print(f"eye-corner width: {width:.1f} units (use --units-per-cm {width / 9.0:.1f} for a 9 cm adult width)")
    print(f"{'landmark':15s} {'x left':>8s} {'x right':>8s} {'asym':>6s} {'off-axis':>8s}")
    for name in sorted({k[:-2] for k in lm if k[-2:] in ("_l", "_r")}):
        if name + "_l" not in pos or name + "_r" not in pos:
            print(f"{name:15s} missing a side")
            continue
        l, r = sub(pos[name + "_l"], mid), sub(pos[name + "_r"], mid)
        xl, xr = dot(l, left), dot(r, left)
        off = [l[i] - xl * left[i] - (r[i] - xr * left[i]) for i in range(3)]
        off_axis = dot(off, off) ** 0.5
        bad = xl <= 0 or xr >= 0 or abs(xl + xr) > tolerance or off_axis > tolerance or lm[name + "_l"] == lm[name + "_r"]
        print(f"{name:15s} {xl:8.1f} {xr:8.1f} {abs(xl + xr):6.1f} {off_axis:8.1f}{'  <- check' if bad else ''}")

    group = _fresh_group("lm_check_GRP")
    for name, p in pos.items():
        loc = cmds.spaceLocator(name="lmc_" + name)[0]
        cmds.xform(loc, worldSpace=True, translation=p)
        cmds.setAttr(loc + ".localScale", 4, 4, 4)
        ann = cmds.annotate(loc, text=name, point=[p[0], p[1] + 6, p[2] + 6])
        cmds.parent(cmds.listRelatives(ann, parent=True)[0], group)
        cmds.parent(loc, group)


def show_electrodes(path=DEFAULT_ELECTRODES, mesh=SKIN, radius=1.5):
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
            for attr, value in (("electrodeChannel", channel), ("electrodeIndex", str(i - 1))):
                cmds.addAttr(sphere, longName=attr, dataType="string")
                cmds.setAttr(f"{sphere}.{attr}", value, type="string")
            cmds.parent(sphere, group)
            ends.append(pos)
        line = cmds.curve(name=f"pair_{tag}", degree=1, point=ends)
        cmds.parent(line, group)
    print(f"placed {2 * len(data['channels'])} electrodes from {path}")


def slide_on_skin(on=True, mesh=SKIN):
    """Make the skin live so moved electrode spheres stay on its surface (Modify > Make Live)."""
    if on:
        cmds.makeLive(mesh)
        print(f"{mesh} is live: spheres moved with W slide on it; met.slide_on_skin(False) to stop")
    else:
        cmds.makeLive(none=True)
        print("skin no longer live")


def _triangle_offsets(fn):
    """Index of each polygon's first triangle in the export_mesh triangle list (same getTriangles order)."""
    counts, _ = fn.getTriangles()
    offsets, total = [], 0
    for c in counts:
        offsets.append(total)
        total += c
    return offsets


def _barycentric(p, a, b, c):
    """Barycentric weights of p projected into triangle abc, clamped to the triangle."""
    sub = lambda u, v: [u[i] - v[i] for i in range(3)]
    dot = lambda u, v: sum(u[i] * v[i] for i in range(3))
    v0, v1, v2 = sub(b, a), sub(c, a), sub(p, a)
    d00, d01, d11, d20, d21 = dot(v0, v0), dot(v0, v1), dot(v1, v1), dot(v2, v0), dot(v2, v1)
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-20:
        return [1.0, 0.0, 0.0]
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    weights = [max(0.0, 1 - v - w), max(0.0, v), max(0.0, w)]
    total = sum(weights)
    return [x / total for x in weights]


def _snap(fn, offsets, point):
    """Closest skin spot to point: (triangle id in the export, its 3 vertex ids, barycentric, position)."""
    hit, face = fn.getClosestPoint(om.MPoint(*point), om.MSpace.kWorld)
    hit = (hit.x, hit.y, hit.z)
    poly = om.MItMeshPolygon(fn.dagPath())
    poly.setIndex(face)
    _, ids = poly.getTriangles(om.MSpace.kWorld)
    ids = list(ids)
    best = None
    for k in range(0, len(ids), 3):
        tri = ids[k:k + 3]
        corners = [tuple(fn.getPoint(v, om.MSpace.kWorld))[:3] for v in tri]
        bary = _barycentric(hit, *corners)
        pos = [sum(bary[j] * corners[j][axis] for j in range(3)) for axis in range(3)]
        err = sum((pos[axis] - hit[axis]) ** 2 for axis in range(3))
        if best is None or err < best[0]:
            best = (err, offsets[face] + k // 3, tri, bary, pos)
    return best[1:]


def save_electrodes(path=DEFAULT_ELECTRODES, mesh=SKIN, group="electrodes_GRP"):
    """Snap the electrode spheres onto the skin and write their surface positions back to path.

    Run at the neutral frame: rest_position is taken from the skin as it is now. Only spheres that
    moved more than 0.01 units are rewritten (and marked "manual"); the rest keep their entries.
    """
    with open(path) as fh:
        data = json.load(fh)
    fn = _fn_mesh(mesh)
    offsets = _triangle_offsets(fn)
    units = data.get("units_per_cm", 10.0)
    moved = []
    for sphere in cmds.listRelatives(group, children=True, type="transform", fullPath=True) or []:
        if not cmds.attributeQuery("electrodeChannel", node=sphere, exists=True):
            continue
        channel = cmds.getAttr(sphere + ".electrodeChannel")
        index = int(cmds.getAttr(sphere + ".electrodeIndex"))
        entry = data["channels"][channel]["electrodes"][index]
        here = cmds.xform(sphere, query=True, worldSpace=True, translation=True)
        old = [sum(w * tuple(fn.getPoint(v, om.MSpace.kWorld))[axis] for w, v in zip(entry["barycentric"], entry["vertices"]))
               for axis in range(3)]
        if sum((here[a] - old[a]) ** 2 for a in range(3)) ** 0.5 < 0.01:
            continue
        triangle, vertices, bary, pos = _snap(fn, offsets, here)
        entry.update(triangle=triangle, vertices=vertices, barycentric=bary, rest_position=pos, manual=True)
        cmds.xform(sphere, worldSpace=True, translation=pos)
        moved.append(f"{channel} #{index + 1}")
    for channel, entry in data["channels"].items():
        a, b = (e["rest_position"] for e in entry["electrodes"])
        entry["spacing"] = round(sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5, 4)
        if any(e.get("manual") for e in entry["electrodes"]):
            print(f"  {channel:34s} spacing {entry['spacing'] / units:.2f} cm")
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)
    print(f"saved {len(moved)} moved electrode(s) -> {path}: {', '.join(moved) or 'none moved'}")
