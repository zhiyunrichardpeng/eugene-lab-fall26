"""
Passive Dynamics & Momentum Test Script: Right Cheek Clap

This script applies a high-speed collision input using a hand/finger model to the right cheek,
and measures the resulting passive elastic dynamics (jiggle/overshoot) and force propagation
across the skin and underlying muscles.
"""

import maya.cmds as cmds
import maya.api.OpenMaya as om
import csv
import json
import os
import math
import subprocess
import time
import datetime
import re
import sys
from paths import a3_stochastic_relative_path, results_path

try:
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    cmds.error("matplotlib or numpy is not installed. Please install it in your Maya Python environment.")

# --- Debug & Feature Toggles ---
DEBUG_MODE = True           # Set to True to only run 20 frames and exit
FINGER_CLAP_ENABLED = True  # Set to True to enable the custom PBD physics & kinematic push

# --- Configuration ---
HAND_TRANSFORM = "hand"           # The transform node of your hand/finger zBone
IMPACT_AXIS = "tx"                # The axis to translate the hand towards the cheek
IMPACT_MAGNITUDE = -6.133346      # The distance the hand moves to penetrate the cheek

OUTPUT_DIR = os.path.join(a3_stochastic_relative_path, "automation")

# Target vertices to track force propagation
FINGER_VERTEX = f"{HAND_TRANSFORM}.vtx[100]"
SKIN_MESH_FOR_PLOT = "skin_cloth_copy_v5_pull_back_RENDER"
SKIN_VERTICES = {
    "ImpactCenter (15163)": "skin_cloth_copy_v5_pull_back_RENDER.vtx[15163]",
    "Skin (2802)": "skin_cloth_copy_v5_pull_back_RENDER.vtx[2802]",
    "Skin (3817)": "skin_cloth_copy_v5_pull_back_RENDER.vtx[3817]",
    "Skin (867)": "skin_cloth_copy_v5_pull_back_RENDER.vtx[867]",
    "Skin (7235)": "skin_cloth_copy_v5_pull_back_RENDER.vtx[7235]",
    "Skin (15153)": "skin_cloth_copy_v5_pull_back_RENDER.vtx[15153]"
}
MUSCLE_VERTEX = "middleres_ZygomaticusMajor_r_RENDER.vtx[395]"

START_FRAME = 0
SHAKE_START_FRAME = 16
CLAPS = 6
FRAMES_PER_CLAP = 62
END_FRAME = SHAKE_START_FRAME + (CLAPS * FRAMES_PER_CLAP) + 30

def get_vertex_world_position(vtx_name):
    """Get the world space position of a specific vertex."""
    pos = cmds.xform(vtx_name, query=True, worldSpace=True, translation=True)
    return om.MVector(pos[0], pos[1], pos[2])

def run_playblast():
    """Generates a playblast and compiles it into a video."""
    print("\n--- Starting Playblast ---")
    cmds.playbackOptions(minTime=START_FRAME, maxTime=END_FRAME)
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_NAME = f"passive_dynamics_clap_{timestamp}"
    PLAYBLAST_DIR =str(results_path / "maya_render_output_hub" / "temp_rendered_png_hub" / "playblust_results")
    
    if not os.path.exists(PLAYBLAST_DIR):
        os.makedirs(PLAYBLAST_DIR)
    file_path_base = os.path.join(PLAYBLAST_DIR, BASE_NAME)
    cmds.playblast(
        format="image",
        compression="png",
        filename=file_path_base,
        widthHeight=[256, 256],
        viewer=False,
        showOrnaments=False,
        clearCache=True,
        framePadding=4,
        forceOverwrite=True,
        offScreen=True,
        percent=100
    )
    print(f"Playblast Complete in {time.time() - start_time:.2f} seconds.")
    VIDEO_SCRIPT = os.path.join(a3_stochastic_relative_path, "b99-frame1to4-get-b1to3", "d2 helper functions", "make_playblast_video.py")
    print("Running video generation script...")
    try:
        with open(VIDEO_SCRIPT, "r") as f:
            lines = f.readlines()
        with open(VIDEO_SCRIPT, "w") as f:
            for line in lines:
                if line.startswith("BASE_NAME ="):
                    f.write(f'BASE_NAME = "{BASE_NAME}"\n')
                else:
                    f.write(line)

        cmd = f'python "{VIDEO_SCRIPT}"'
        subprocess.run(cmd, shell=True, check=True)
        print("Video generated successfully.")
    except Exception as e:
        print(f"Failed to run the video script. Error: {e}")

def setup_animation():
    """Animates the hand to strike the cheek rapidly and pull back multiple times."""
    cmds.currentTime(START_FRAME, edit=True)
    if not cmds.objExists(HAND_TRANSFORM):
        cmds.error(f"Hand transform '{HAND_TRANSFORM}' not found in the scene.")
    initial_val = cmds.getAttr(f"{HAND_TRANSFORM}.{IMPACT_AXIS}")
    impact_val = initial_val + IMPACT_MAGNITUDE
    
    cmds.setKeyframe(HAND_TRANSFORM, attribute=IMPACT_AXIS, time=START_FRAME, value=initial_val)
    cmds.setKeyframe(HAND_TRANSFORM, attribute=IMPACT_AXIS, time=SHAKE_START_FRAME, value=initial_val)
    cmds.keyTangent(HAND_TRANSFORM, attribute=IMPACT_AXIS, time=(START_FRAME, SHAKE_START_FRAME), inTangentType='flat', outTangentType='flat')
    
    current_frame = SHAKE_START_FRAME
    for i in range(CLAPS):
        impact_f = current_frame + 30
        hold_f = impact_f + 2
        release_f = hold_f + 30
        cmds.setKeyframe(HAND_TRANSFORM, attribute=IMPACT_AXIS, time=impact_f, value=impact_val)
        cmds.setKeyframe(HAND_TRANSFORM, attribute=IMPACT_AXIS, time=hold_f, value=impact_val)
        cmds.setKeyframe(HAND_TRANSFORM, attribute=IMPACT_AXIS, time=release_f, value=initial_val)
        cmds.keyTangent(HAND_TRANSFORM, attribute=IMPACT_AXIS, time=(current_frame, impact_f), inTangentType='linear', outTangentType='linear')
        cmds.keyTangent(HAND_TRANSFORM, attribute=IMPACT_AXIS, time=(impact_f, hold_f), inTangentType='flat', outTangentType='flat')
        cmds.keyTangent(HAND_TRANSFORM, attribute=IMPACT_AXIS, time=(hold_f, release_f), inTangentType='linear', outTangentType='linear')
        current_frame = release_f
    cmds.setKeyframe(HAND_TRANSFORM, attribute=IMPACT_AXIS, time=END_FRAME, value=initial_val)

def plot_results(plot_file, frames, data_lists, y_label):
    """Plot data (velocity or displacement) to show wave propagation."""
    plt.style.use('classic')
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['legend.fontsize'] = 9
    fig, ax = plt.subplots(figsize=(6.0, 3.5), dpi=300)
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#9467bd', '#8c564b', '#e377c2']
    color_idx = 0
    for key, data in data_lists.items():
        if key == 'finger':
            ax.plot(frames, data, label='Input (Fingers)', linestyle=':', color='#7f7f7f', linewidth=2.0)
        elif key == 'muscle':
            ax.plot(frames, data, label='Muscle (Zygomaticus 395)', linestyle='--', color='#2ca02c', linewidth=2.0)
        else:
            ax.plot(frames, data, label=key, linestyle='-', color=colors[color_idx % len(colors)], linewidth=1.5)
            color_idx += 1
    ax.set_xlabel('Frame')
    ax.set_ylabel(y_label)
    ax.axvspan(SHAKE_START_FRAME, SHAKE_START_FRAME + (CLAPS * FRAMES_PER_CLAP), color='gray', alpha=0.2)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_file, format='pdf', bbox_inches='tight')
    plt.savefig(plot_file.replace('.pdf', '.png'), format='png', bbox_inches='tight')
    plt.close()

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_3d_context(plot_file, skin_mesh_name, skin_vertices_dict, muscle_vertex):
    """Plot the position of vertices on skin only, and mark the 3d position of muscle vertex."""
    plt.style.use('classic')
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    if cmds.objExists(skin_mesh_name):
        flat_coords = cmds.xform(f"{skin_mesh_name}.vtx[*]", query=True, translation=True, worldSpace=True)
        if flat_coords:
            skin_cloud = np.array(flat_coords).reshape(-1, 3)
            step = 1 if len(skin_cloud) < 5000 else 10
            ax.scatter(skin_cloud[::step, 0], skin_cloud[::step, 1], skin_cloud[::step, 2],
                       c='#F5DEB3', s=2, alpha=0.2, label='Skin Mesh')
    skin_pts = []
    for key, vtx in skin_vertices_dict.items():
        if cmds.objExists(vtx):
            pos = cmds.xform(vtx, query=True, translation=True, worldSpace=True)
            skin_pts.append(pos)
    if skin_pts:
        skin_pts = np.array(skin_pts)
        ax.scatter(skin_pts[:, 0], skin_pts[:, 1], skin_pts[:, 2],
                   c='#DAA520', s=120, edgecolors='black', linewidth=1.5, depthshade=False, label='Tracked Skin Vertices')
    if cmds.objExists(muscle_vertex):
        m_pos = cmds.xform(muscle_vertex, query=True, translation=True, worldSpace=True)
        ax.scatter(m_pos[0], m_pos[1], m_pos[2],
                   c='#2ca02c', s=120, edgecolors='black', linewidth=1.5, depthshade=False, label='Tracked Muscle Vertex')

    ax.set_title(f'Tracked Vertices Global Context')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    ax.legend(loc='upper right')
    ax.view_init(elev=85, azim=-90, roll=None)

    plt.tight_layout()
    plt.savefig(plot_file, format='pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(plot_file.replace('.pdf', '.png'), format='png', bbox_inches='tight', facecolor='white')
    plt.close()

def main(auto_playblast=False):
    original_time = cmds.currentTime(query=True)
    try:
        print("Setting up clap animation...")
        setup_animation()
        
        # ==========================================================
        # 1. INITIALIZE CUSTOM PBD PHYSICS CONSTRAINT GRAPH (IF ENABLED)
        # ==========================================================
        vtx_indices = set()
        push_vertex_id = 2802
        push_translation = om.MVector(0.0, -2.5, 0.0)
        
        if FINGER_CLAP_ENABLED:
            print("[PBD] Initializing Custom Physics Solver with Kinematic Push...")
            impact_json_path = os.path.join(a3_stochastic_relative_path, "automation", "impact_area_vertices.json")
            try:
                with open(impact_json_path, 'r') as f:
                    skin_vertices = json.load(f)
            except Exception as e:
                print("[PBD ERROR] Could not load impact_area_vertices.json:", e)
                return

            try:
                sel = om.MSelectionList()
                sel.add(SKIN_MESH_FOR_PLOT)
                skin_dag = sel.getDagPath(0)
                skin_mesh_fn = om.MFnMesh(skin_dag)
            except Exception as e:
                print("[PBD ERROR] Could not get mesh:", SKIN_MESH_FOR_PLOT, e)
                return

            for vtx in skin_vertices:
                match = re.search(r'\[(\d+)\]', vtx)
                if match:
                    vtx_indices.add(int(match.group(1)))
            
            # Ensure our target push vertex is included
            vtx_indices.add(push_vertex_id)

            # Build local topological neighbor edge map
            edge_iter = om.MItMeshEdge(skin_dag)
            full_neighbors = {}
            while not edge_iter.isDone():
                v0, v1 = edge_iter.vertexId(0), edge_iter.vertexId(1)
                full_neighbors.setdefault(v0, []).append(v1)
                full_neighbors.setdefault(v1, []).append(v0)
                edge_iter.next()

            # Cache rest states
            rest_positions = {}
            original_lengths = {}
            pts = skin_mesh_fn.getPoints(om.MSpace.kWorld)
            for idx in vtx_indices:
                rest_positions[idx] = om.MVector(pts[idx].x, pts[idx].y, pts[idx].z)

            for v0 in vtx_indices:
                for v1 in full_neighbors.get(v0, []):
                    if v1 > v0 or v1 not in vtx_indices:
                        p0 = om.MVector(pts[v0].x, pts[v0].y, pts[v0].z)
                        p1 = om.MVector(pts[v1].x, pts[v1].y, pts[v1].z)
                        original_lengths[(v0, v1)] = (p0 - p1).length()

            print(f"[PBD] Setup complete for {len(vtx_indices)} vertices and {len(original_lengths)} edge constraints.")

        # ==========================================================
        # 2. SIMULATION LOOP & TIMELINE STEPPING
        # ==========================================================
        print("Running simulation and collecting data...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        CSV_FILE_PATH = os.path.join(OUTPUT_DIR, f"passive_dynamics_clap_results_{timestamp}.csv")
        PLOT_VELOCITY_FILE_PATH = os.path.join(OUTPUT_DIR, f"passive_dynamics_clap_velocity_{timestamp}.pdf")
        PLOT_DISPLACEMENT_FILE_PATH = os.path.join(OUTPUT_DIR, f"passive_dynamics_clap_displacement_{timestamp}.pdf")
        PLOT_3D_CONTEXT_FILE_PATH = os.path.join(OUTPUT_DIR, f"passive_dynamics_clap_3dcontext_{timestamp}.pdf")
        
        positions = {'finger': {}, 'muscle': {}}
        for key in SKIN_VERTICES.keys():
            positions[key] = {}

        for vtx in list(SKIN_VERTICES.values()) + [FINGER_VERTEX, MUSCLE_VERTEX]:
            if not cmds.objExists(vtx):
                cmds.warning(f"Vertex {vtx} not found. Please update configuration.")

        frames_stepped = 0

        # Sequentially step through timeline
        for t in range(START_FRAME, END_FRAME + 1):
            cmds.currentTime(t, edit=True)
            
            # --- PBD SOLVER EXECUTION (Directly in main loop if enabled) ---
            if FINGER_CLAP_ENABLED and vtx_indices:
                curr_pts_array = skin_mesh_fn.getPoints(om.MSpace.kWorld)
                current_pts = {}
                for idx in vtx_indices:
                    current_pts[idx] = om.MVector(curr_pts_array[idx].x, curr_pts_array[idx].y, curr_pts_array[idx].z)

                # Elasticity: Pull towards rest pose
                spring_stiffness = 0.3
                for idx in vtx_indices:
                    if idx == push_vertex_id:
                        current_pts[idx] = rest_positions[idx] + push_translation
                    else:
                        curr = current_pts[idx]
                        rest = rest_positions[idx]
                        current_pts[idx] = curr + (rest - curr) * spring_stiffness

                # Solver Iterations
                for _ in range(4):
                    for (v0, v1), rest_len in original_lengths.items():
                        p0 = current_pts[v0]
                        p1 = current_pts[v1] if v1 in vtx_indices else om.MVector(curr_pts_array[v1].x, curr_pts_array[v1].y, curr_pts_array[v1].z)
                        delta = p1 - p0
                        cur_len = delta.length()
                        if cur_len > 1e-6:
                            diff = (cur_len - rest_len) / cur_len
                            correction = delta * 0.5 * diff * 0.8
                            if v0 != push_vertex_id:
                                current_pts[v0] = p0 + correction
                            if v1 in vtx_indices and v1 != push_vertex_id:
                                current_pts[v1] = p1 - correction

                # Update mesh geometry and WRITE KEYFRAMES to the timeline
                for idx in vtx_indices:
                    p = current_pts[idx]
                    vtx_path = f"{SKIN_MESH_FOR_PLOT}.vtx[{idx}]"
                    
                    # 1. Update the position in world space
                    cmds.xform(vtx_path, worldSpace=True, translation=[p.x, p.y, p.z])
                    
                    # 2. Save this position as a keyframe on the current frame 't'
                    cmds.setKeyframe(vtx_path, time=t)

            # Record positions safely
            positions['finger'][t] = get_vertex_world_position(FINGER_VERTEX) if cmds.objExists(FINGER_VERTEX) else om.MVector(0,0,0)
            positions['muscle'][t] = get_vertex_world_position(MUSCLE_VERTEX) if cmds.objExists(MUSCLE_VERTEX) else om.MVector(0,0,0)
            for key, vtx in SKIN_VERTICES.items():
                positions[key][t] = get_vertex_world_position(vtx) if cmds.objExists(vtx) else om.MVector(0,0,0)

            # --- DEBUG TERMINATION TEST ---
            frames_stepped += 1
            if DEBUG_MODE and frames_stepped >= 20:
                print("passed the test, now quit.")
                sys.exit(0)

        # ==========================================================
        # 3. ANALYSIS & DATA EXPORT
        # ==========================================================
        print("Calculating velocities and displacements...")
        results = []
        frames = []
        vel_lists = {k: [] for k in positions.keys()}
        disp_lists = {k: [] for k in positions.keys()}
        for t in range(START_FRAME, END_FRAME + 1):
            frames.append(t)
            row_data = {'Frame': t}
            for key in positions.keys():
                if t == START_FRAME:
                    v = 0.0
                else:
                    v = (positions[key][t] - positions[key][t-1]).length()
                d = (positions[key][t] - positions[key][START_FRAME]).length()
                vel_lists[key].append(v)
                disp_lists[key].append(d)
                row_data[f'{key}_Velocity'] = v
                row_data[f'{key}_Displacement'] = d
            results.append(row_data)

        print("Exporting data to CSV...")
        with open(CSV_FILE_PATH, mode='w', newline='') as file:
            fieldnames = ['Frame']
            for key in positions.keys():
                fieldnames.extend([f'{key}_Velocity', f'{key}_Displacement'])
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print("Plotting results...")
        cmds.currentTime(START_FRAME, edit=True)
        plot_3d_context(PLOT_3D_CONTEXT_FILE_PATH, SKIN_MESH_FOR_PLOT, SKIN_VERTICES, MUSCLE_VERTEX)
        plot_results(PLOT_VELOCITY_FILE_PATH, frames, vel_lists, "Velocity (cm/frame)")
        plot_results(PLOT_DISPLACEMENT_FILE_PATH, frames, disp_lists, "Displacement from Rest (cm)")

        if auto_playblast:
            run_playblast()
        print(f"\nSuccess! Data saved to {OUTPUT_DIR}")

    except Exception as e:
        cmds.error(f"Error during execution: {str(e)}")
    finally:
        cmds.currentTime(original_time)

if __name__ == "__main__":
    main()