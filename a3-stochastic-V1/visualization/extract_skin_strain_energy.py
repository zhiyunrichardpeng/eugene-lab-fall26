"""
=============================================================================
Extract Skin Elastic Strain Energy (Maya Script)
=============================================================================
Run this script inside Maya's Script Editor (Python tab).

Purpose:
  - Observe the mesh "skin_cloth_copy_v5_pull_back" (the skin).
  - Play the animation sequentially from frame 0 to frame 50 so that
    Ziva Dynamics is properly evaluated at each frame.
  - At each frame (from frame 15 onward), compute the elastic strain energy
    of the skin mesh relative to the NEUTRAL pose (frame 0).
  - The elastic strain energy formula (same as evaluator.py):
        E_strain = 0.5 * k_tissue * (d * scaling_factor)^2
    where d is the per-vertex Euclidean distance from rest, summed over all
    vertices.
  - Save the result (frames 15–50) as a CSV file for external plotting.

How to use:
  1. Open your .mb file (V→N deformation) in Maya.
  2. Make sure Ziva Dynamics is active (the solver is not disabled).
  3. Open Script Editor → Python tab.
  4. Paste this entire script and run it (Ctrl+Enter or the ▶ button).
  5. The CSV will be saved to:
       C:/Users/zpa18/a3-stochastic-V1/visualization/skin_strain_energy_data.csv
=============================================================================
"""

import maya.cmds as cmds
import maya.OpenMaya as OpenMaya
import os
import csv
import numpy as np

# ===================== CONFIGURATION =====================
MESH_NAME = "skin_cloth_copy_v5_pull_back"
START_FRAME = 0       # Playback must start from 0 for Ziva to initialize
OBS_START_FRAME = 15  # We only *record* from this frame onward
REST_FRAME = 0        # The neutral pose — strain is measured relative to this
END_FRAME = 50        # Last frame to evaluate

# Physical parameters (same as evaluator.py)
K_TISSUE = 10000          # Stiffness of soft tissues (N/m)
SCALING_FACTOR = 9.62e-4  # Maya unit → metres conversion

OUTPUT_DIR = r"C:/Users/zpa18/a3-stochastic-V1/visualization"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "skin_strain_energy_data.csv")
# =========================================================


def get_mesh_vertices(mesh_name):
    """
    Return all vertex positions of `mesh_name` as a numpy array of shape (N, 3)
    using OpenMaya for speed.
    """
    cmds.select(mesh_name, visible=True)

    selection = OpenMaya.MSelectionList()
    OpenMaya.MGlobal.getActiveSelectionList(selection)

    dagPath = OpenMaya.MDagPath()
    iterSel = OpenMaya.MItSelectionList(selection, OpenMaya.MFn.kMesh)
    iterSel.getDagPath(dagPath)

    fn_mesh = OpenMaya.MFnMesh(dagPath)
    points = OpenMaya.MPointArray()
    fn_mesh.getPoints(points, OpenMaya.MSpace.kWorld)

    num_pts = points.length()
    coords = np.zeros((num_pts, 3))
    for i in range(num_pts):
        p = points[i]
        coords[i, 0] = p.x
        coords[i, 1] = p.y
        coords[i, 2] = p.z

    return coords


def compute_strain_energy(verts_current, verts_rest, k, scale):
    """
    Compute the elastic strain energy of the skin mesh.

    Uses the same spring formula as evaluator.py (line 1240):
        strain_energy = 0.5 * k_tissue * (strain_distance * scaling_factor)^2

    strain_distance here is the MEAN per-vertex Euclidean distance between
    the current pose and the neutral (rest) pose.  Using the mean (rather
    than the aggregated L2 norm) normalises out the vertex count, so the
    energy value is independent of mesh resolution and comparable to the
    landmark-based calculation in evaluator.py.
    """
    diff = verts_current - verts_rest                         # (N, 3)
    per_vertex_dist = np.sqrt(np.sum(diff ** 2, axis=1))      # (N,)

    # Mean per-vertex distance (normalises out vertex count)
    strain_distance = np.mean(per_vertex_dist)

    # Elastic strain energy
    strain_energy = 0.5 * k * (strain_distance * scale) ** 2
    return strain_energy


# ----- Sanity check -----
if not cmds.objExists(MESH_NAME):
    raise RuntimeError(
        "Mesh '{}' not found in the scene. "
        "Please check the mesh name and try again.".format(MESH_NAME)
    )

print("=" * 60)
print("  Extracting skin elastic strain energy")
print("  Mesh : {}".format(MESH_NAME))
print("  Frames: {} -> {}  (observe from frame {})".format(START_FRAME, END_FRAME, OBS_START_FRAME))
print("  Rest frame: {}".format(REST_FRAME))
print("  k_tissue = {}  |  scaling_factor = {}".format(K_TISSUE, SCALING_FACTOR))
print("=" * 60)

# ----- Play through all frames, store vertex positions -----
all_verts = {}

for frame in range(START_FRAME, END_FRAME + 1):
    cmds.currentTime(frame, edit=True)
    all_verts[frame] = get_mesh_vertices(MESH_NAME)
    if frame % 5 == 0 or frame == END_FRAME:
        print("  Captured frame {} / {}".format(frame, END_FRAME))

print("Playback complete. Captured {} frames.".format(len(all_verts)))

# ----- Compute strain energy at each observed frame -----
rest_verts = all_verts[REST_FRAME]

frames_list = []
energy_list = []

for frame in range(OBS_START_FRAME, END_FRAME + 1):
    energy = compute_strain_energy(all_verts[frame], rest_verts, K_TISSUE, SCALING_FACTOR)
    frames_list.append(frame)
    energy_list.append(energy)
    print("  Frame {:3d}  |  Elastic Strain Energy: {:.6e} J".format(frame, energy))

# ----- Save to CSV -----
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

with open(OUTPUT_CSV, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "elastic_strain_energy_J"])
    for fr, e in zip(frames_list, energy_list):
        writer.writerow([fr, e])

print("\nCSV saved to: {}".format(OUTPUT_CSV))
print("Done! Now run 'plot_skin_strain_energy.py' externally to generate the figure.")
