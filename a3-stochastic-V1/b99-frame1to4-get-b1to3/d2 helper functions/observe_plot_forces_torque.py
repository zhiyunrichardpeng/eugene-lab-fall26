import os
import csv
import math
import numpy as np
import datetime

# Try to import Maya commands
try:
    import maya.cmds as cmds
    IN_MAYA = True
except ImportError:
    IN_MAYA = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from paths import b99_frame1to4_path

# --- CONFIGURATION ---
# Define the meshes and vertices to observe
OOR_MESH = "middleres_OrbicularisOris"
# We will pick 3 arbitrary vertices if they exist. You can change these indices.
OOR_VERTICES = [100, 300, 800] 

# Set your mandible (jaw) mesh name here. 
JAW_MESH = "middleres_Jaw" # Replace with actual jaw mesh name if different
JAW_VERTICES = [29, 177, 350]

# Unit Conversion: Maya units to Standard SI units (meters, kg, seconds)
# 1 Maya Unit = 0.0962 cm = 0.000962 meters
MAYA_TO_METERS = 0.000962
MASS_KG = 1.0 # Assumed mass of the tissue/bone in kg

# Approximate TMJ (Jaw Pivot) in world space for Torque calculation. 
# We define it here in Maya units, we will convert to meters internally.
JAW_PIVOT_MAYA = np.array([0.0, 8.0, -2.0]) 

# Simulation parameters
DT = 1.0 / 24.0  # Assuming 24 FPS
OUTPUT_DIR = str(b99_frame1to4_path / "d4 interactive interface of animation" / "force_observation_results")


# --- MAYA DATA EXTRACTION ---
def extract_physics_trajectories(start_frame=1, end_frame=60):
    """Steps through the Maya timeline and records positions."""
    if not IN_MAYA:
        print("[ERROR] extract_physics_trajectories must be run inside Maya.")
        return None
        
    targets = {
        'oor_1': f"{OOR_MESH}.vtx[{OOR_VERTICES[0]}]",
        'oor_2': f"{OOR_MESH}.vtx[{OOR_VERTICES[1]}]",
        'oor_3': f"{OOR_MESH}.vtx[{OOR_VERTICES[2]}]",
        'jaw_1': f"{JAW_MESH}.vtx[{JAW_VERTICES[0]}]",
        'jaw_2': f"{JAW_MESH}.vtx[{JAW_VERTICES[1]}]",
        'jaw_3': f"{JAW_MESH}.vtx[{JAW_VERTICES[2]}]"
    }
    
    # Verify existence
    for name, vtx in targets.items():
        if not cmds.objExists(vtx):
            print(f"[WARNING] Vertex {vtx} does not exist. Tracking might fail.")

    data = {name: [] for name in targets.keys()}
    frames = list(range(start_frame, end_frame + 1))
    
    print(f"[PHYSICS OBSERVATION] Stepping timeline from {start_frame} to {end_frame}...")
    for f in frames:
        cmds.currentTime(f, edit=True)
        for name, vtx in targets.items():
            try:
                pos = cmds.xform(vtx, query=True, translation=True, worldSpace=True)
                data[name].append(pos)
            except Exception:
                data[name].append([0,0,0])
                
    # Convert to numpy arrays and to Standard SI Units (Meters)
    for name in data:
        data[name] = np.array(data[name]) * MAYA_TO_METERS
        
    return frames, data

# --- KINEMATICS & FORCE CALCULATION ---
def compute_derivatives(positions, dt):
    """Computes velocity and acceleration (proportional to force) using finite differences."""
    # positions: (N, 3)
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    
    if len(positions) < 3:
        return velocities, accelerations
        
    # Central difference for interior points
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    # Forward/backward difference for edges
    velocities[0] = (positions[1] - positions[0]) / dt
    velocities[-1] = (positions[-1] - positions[-2]) / dt
    
    # Acceleration = dv/dt (Force = m*a, we assume m=1)
    accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2 * dt)
    accelerations[0] = (velocities[1] - velocities[0]) / dt
    accelerations[-1] = (velocities[-1] - velocities[-2]) / dt
    
    return velocities, accelerations

def compute_torque(positions, forces, pivot):
    """Computes Torque = r x F."""
    # positions: (N, 3), forces: (N, 3), pivot: (3,)
    torques = np.zeros_like(positions)
    for i in range(len(positions)):
        r = positions[i] - pivot
        torques[i] = np.cross(r, forces[i])
    return torques

# --- PLOTTING (SIGGRAPH TOG STYLE) ---
def setup_siggraph_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.dpi'] = 300

def colorline(x, y, z=None, cmap=plt.get_cmap('viridis'), linewidth=3, alpha=1.0, ax=None):
    """Helper to plot a line with varying color mapped to time."""
    t = np.linspace(0, 1, len(x))
    
    if z is None:
        # 2D case
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        lc.set_array(t)
        lc.set_linewidth(linewidth)
        lc.set_alpha(alpha)
        ax.add_collection(lc)
        return lc
    else:
        # 3D case
        # Matplotlib 3D doesn't support LineCollection3D easily with color arrays, 
        # so we plot individual segments.
        for i in range(len(x) - 1):
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], 
                    color=cmap(t[i]), linewidth=linewidth, alpha=alpha)

def plot_visualizations(frames, forces_dict, torques_dict, out_dir):
    setup_siggraph_style()
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Colors
    colors = {'oor_1': '#4A90D9', 'oor_2': '#E85D75', 'oor_3': '#50C878', 
              'jaw_1': '#F5A623', 'jaw_2': '#9B59B6', 'jaw_3': '#E67E22'}
    labels = {'oor_1': 'OOR Vtx 1', 'oor_2': 'OOR Vtx 2', 'oor_3': 'OOR Vtx 3', 
              'jaw_1': f'Jaw Vtx {JAW_VERTICES[0]}', 'jaw_2': f'Jaw Vtx {JAW_VERTICES[1]}', 'jaw_3': f'Jaw Vtx {JAW_VERTICES[2]}'}

    # 1. 1D Force Component (Y-Axis) - Shows gravity/pulling effect
    plt.figure(figsize=(10, 6))
    for name, f_arr in forces_dict.items():
        plt.plot(frames, f_arr[:, 1], linewidth=2.5, color=colors[name], label=f"{labels[name]} ($F_y$)")
    
    plt.title('Force Y-Component Over Time (Vertical Pull)')
    plt.xlabel('Frame')
    plt.ylabel('Force ($F_y$) [Newtons]')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"force_y_component_{timestamp}.png"))
    plt.close()

    # 2. Force Magnitude
    plt.figure(figsize=(10, 6))
    for name, f_arr in forces_dict.items():
        mags = np.linalg.norm(f_arr, axis=1)
        plt.plot(frames, mags, linewidth=2.5, color=colors[name], label=f"{labels[name]} ($||F||$)")
    
    plt.title('Force Magnitude Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Force Magnitude $||F||$ [Newtons]')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"force_magnitude_{timestamp}.png"))
    plt.close()

    # 3. Mandible Torque (Magnitude and Pitch Component)
    plt.figure(figsize=(12, 7))
    line_styles = ['-', '--', ':']
    for idx, name in enumerate(['jaw_1', 'jaw_2', 'jaw_3']):
        jaw_t = torques_dict[name]
        t_mags = np.linalg.norm(jaw_t, axis=1)
        plt.plot(frames, t_mags, linewidth=2.5, color=colors[name], linestyle=line_styles[0], label=f'Mag $||\\tau||$ ({labels[name]})')
        plt.plot(frames, jaw_t[:, 0], linewidth=1.5, color=colors[name], linestyle=line_styles[1], label=f'Pitch $\\tau_x$ ({labels[name]})')
    
    plt.title('Mandible Torque Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Torque $\\tau$ [N*m]')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"mandible_torque_{timestamp}.png"))
    plt.close()

    # 4. Fancy 3D Force Trajectory
    # Let's plot it for both the Jaw and one of the OOR vertices
    for target_name in ['jaw_1', 'oor_1']:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        f_data = forces_dict[target_name]
        colorline(f_data[:, 0], f_data[:, 1], f_data[:, 2], cmap=plt.get_cmap('plasma'), linewidth=3, ax=ax)
        
        # Plot start and end points
        ax.scatter([f_data[0, 0]], [f_data[0, 1]], [f_data[0, 2]], color='green', s=100, label='Start')
        ax.scatter([f_data[-1, 0]], [f_data[-1, 1]], [f_data[-1, 2]], color='red', s=100, label='End')
        
        ax.set_title(f'3D Force Trajectory of {labels[target_name]}\n(Color maps from Start -> End)')
        ax.set_xlabel('$F_x$ [N]')
        ax.set_ylabel('$F_y$ [N]')
        ax.set_zlabel('$F_z$ [N]')
        
        # Adjust tick formatting if values are extremely small (like 1e-18)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))
        ax.ticklabel_format(style='sci', axis='z', scilimits=(-3,3))

        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"force_trajectory_3d_{target_name}_{timestamp}.png"))
        plt.close()
    
    print(f"[SUCCESS] All plots saved to: {out_dir}")

def export_to_csv(frames, positions_dict, forces_dict, torques_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"physics_forces_observation_{timestamp}.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Frame']
        for name in positions_dict.keys():
            header.extend([f"{name}_pos_X", f"{name}_pos_Y", f"{name}_pos_Z"])
            header.extend([f"{name}_force_X", f"{name}_force_Y", f"{name}_force_Z"])
            if name.startswith('jaw_'):
                header.extend([f"{name}_torque_X", f"{name}_torque_Y", f"{name}_torque_Z"])
        writer.writerow(header)
        
        # Data
        for i, frame in enumerate(frames):
            row = [frame]
            for name in positions_dict.keys():
                pos = positions_dict[name][i]
                force = forces_dict[name][i]
                row.extend([pos[0], pos[1], pos[2], force[0], force[1], force[2]])
                
                if name.startswith('jaw_'):
                    torque = torques_dict[name][i]
                    row.extend([torque[0], torque[1], torque[2]])
            writer.writerow(row)
            
    print(f"[SUCCESS] Data exported to CSV: {csv_path}")

# --- MAIN EXECUTION ---
def main():
    if not IN_MAYA:
        print("[INFO] Maya not detected. If you want to plot existing CSV data, you would load it here.")
        print("[INFO] Please run this script INSIDE Maya to capture the trajectories.")
        return

    # Configuration for observation window
    start_time = int(cmds.playbackOptions(query=True, minTime=True))
    end_time = int(cmds.playbackOptions(query=True, maxTime=True))
    
    print(f"\n{'='*50}")
    print(f"Starting Physics Observation: Frames {start_time} to {end_time}")
    print(f"{'='*50}")

    # 1. Extract Positions
    frames, positions_dict = extract_physics_trajectories(start_frame=start_time, end_frame=end_time)
    
    if frames is None:
        return
        
    forces_dict = {}
    torques_dict = {}
    
    # 2. Compute Kinematics (Force & Torque)
    jaw_pivot_m = JAW_PIVOT_MAYA * MAYA_TO_METERS
    for name, pos_arr in positions_dict.items():
        vel, acc = compute_derivatives(pos_arr, dt=DT)
        # F = m*a
        forces_dict[name] = acc * MASS_KG 
        
        if name.startswith('jaw_'):
            torques_dict[name] = compute_torque(pos_arr, forces_dict[name], jaw_pivot_m)
            
    # 3. Export to CSV
    export_to_csv(frames, positions_dict, forces_dict, torques_dict, OUTPUT_DIR)
    
    # 4. Plot (TOG SIGGRAPH STYLE)
    try:
        plot_visualizations(frames, forces_dict, torques_dict, OUTPUT_DIR)
    except Exception as e:
        print(f"[ERROR] Plotting failed (maybe matplotlib is missing/failing in Maya?): {e}")

if __name__ == "__main__":
    main()
