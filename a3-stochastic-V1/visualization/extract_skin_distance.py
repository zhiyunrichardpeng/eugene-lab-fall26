"""
=============================================================================
Extract Skin Distance (Maya Script)
=============================================================================
Run this script inside Maya's Script Editor (Python tab).

Purpose:
  - Observe the mesh "skin_cloth_copy_v5_pull_back" (the skin).
  - Play the animation sequentially from frame 0 to frame 50 so that
    Ziva Dynamics is properly evaluated at each frame.
  - Record vertex positions at every frame.
  - After playback, compute the mean Euclidean distance between each frame's
    vertex positions and the TARGET pose (frame 50).
  - Save the result (frames 15–50) as a CSV file for external plotting.

How to use:
  1. Open your .mb file (N→Y deformation) in Maya.
  2. Make sure Ziva Dynamics is active (the solver is not disabled).
  3. Open Script Editor → Python tab.
  4. Paste this entire script and run it (Ctrl+Enter or the ▶ button).
  5. The CSV will be saved to:
       C:/Users/zpa18/a3-stochastic-V1/visualization/skin_distance_data.csv
=============================================================================
"""

import maya.cmds as cmds
import maya.OpenMaya as OpenMaya
import os
import csv

# ===================== CONFIGURATION =====================
MESH_NAME = "skin_cloth_copy_v5_pull_back"
START_FRAME = 0       # Playback must start from 0 (or 1) for Ziva to initialize
OBS_START_FRAME = 15  # We only *record* from this frame onward
TARGET_FRAME = 50     # The target pose frame
END_FRAME = 50        # Last frame to evaluate

OUTPUT_DIR = r"C:/Users/zpa18/a3-stochastic-V1/visualization"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "skin_distance_data.csv")
# =========================================================

import numpy as np


def get_mesh_vertices(mesh_name):
    """
    Return all vertex positions of `mesh_name` as a numpy array of shape (N, 3)
    using OpenMaya for speed.
    """
    # Select the mesh
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


def compute_mean_euclidean_distance(verts_a, verts_b):
    """
    Compute the mean per-vertex Euclidean distance between two vertex arrays.
    Both arrays must have the same shape (N, 3).
    Returns distance in centimetres (Maya default unit).
    """
    diff = verts_a - verts_b
    per_vertex_dist = np.sqrt(np.sum(diff ** 2, axis=1))  # shape (N,)
    return np.mean(per_vertex_dist)


# ----- Sanity check: does the mesh exist? -----
if not cmds.objExists(MESH_NAME):
    raise RuntimeError(
        "Mesh '{}' not found in the scene. "
        "Please check the mesh name and try again.".format(MESH_NAME)
    )

print("=" * 60)
print("  Extracting skin distance data")
print("  Mesh : {}".format(MESH_NAME))
print("  Frames: {} → {}  (observe from frame {})".format(START_FRAME, END_FRAME, OBS_START_FRAME))
print("  Target frame: {}".format(TARGET_FRAME))
print("=" * 60)

# ----- Pass 1: Play through all frames, store vertex positions -----
all_verts = {}  # frame -> (N, 3) array

for frame in range(START_FRAME, END_FRAME + 1):
    cmds.currentTime(frame, edit=True)
    all_verts[frame] = get_mesh_vertices(MESH_NAME)
    if frame % 5 == 0 or frame == END_FRAME:
        print("  Captured frame {} / {}".format(frame, END_FRAME))

print("Playback complete. Captured {} frames.".format(len(all_verts)))

# ----- Compute distances relative to the target frame -----
target_verts = all_verts[TARGET_FRAME]

frames_list = []
distances_list = []

for frame in range(OBS_START_FRAME, END_FRAME + 1):
    dist = compute_mean_euclidean_distance(all_verts[frame], target_verts)
    frames_list.append(frame)
    distances_list.append(dist)
    print("  Frame {:3d}  |  Mean Euclidean Distance to target: {:.6f} cm".format(frame, dist))

# ----- Save to CSV -----
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

with open(OUTPUT_CSV, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "mean_euclidean_distance_cm"])
    for fr, d in zip(frames_list, distances_list):
        writer.writerow([fr, d])

print("\nCSV saved to: {}".format(OUTPUT_CSV))
print("Done! Now run 'plot_skin_distance.py' externally to generate the figure.")
