import os
import datetime
import numpy as np
from paths import b99_frame1to4_path

# Try to import Maya commands
try:
    # pyrefly: ignore [missing-import]
    import maya.cmds as cmds
    IN_MAYA = True
except ImportError:
    IN_MAYA = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
PLOT_OOR = True
OOR_MESH = "middleres_OrbicularisOris"
OOR_VERTICES = [0, 100] 

PLOT_JAW = False # Muted as requested
JAW_MESH = "middleres_Jaw" 
JAW_VERTICES = [29, 177, 350]

PLOT_SKIN = True
SKIN_MESH = "skin_cloth_copy_v5_pull_back" 
SKIN_VERTICES = [11109]

OUTPUT_DIR = str(b99_frame1to4_path / "d4 interactive interface of animation" / "force_observation_results")
FRAME_TO_PLOT = 0

# --- DATA EXTRACTION ---
def get_mesh_point_cloud(mesh_name):
    """Retrieves all vertices of a mesh as an (N, 3) numpy array."""
    if not cmds.objExists(mesh_name):
        print(f"[WARNING] Mesh '{mesh_name}' not found.")
        return None
        
    flat_coords = cmds.xform(f"{mesh_name}.vtx[*]", query=True, translation=True, worldSpace=True)
    if not flat_coords:
        return None
        
    # Reshape from [x1, y1, z1, x2, y2, z2...] to [[x1, y1, z1], [x2, y2, z2]...]
    points = np.array(flat_coords).reshape(-1, 3)
    return points

def get_specific_vertices(mesh_name, indices):
    """Retrieves specific vertex coordinates as an (N, 3) numpy array."""
    points = []
    for idx in indices:
        vtx_name = f"{mesh_name}.vtx[{idx}]"
        if cmds.objExists(vtx_name):
            pos = cmds.xform(vtx_name, query=True, translation=True, worldSpace=True)
            points.append(pos)
        else:
            print(f"[WARNING] Vertex {vtx_name} not found.")
    return np.array(points) if points else None

# --- PLOTTING ---
def setup_siggraph_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.dpi'] = 300

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres, etc.."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_mesh_context():
    if not IN_MAYA:
        print("[ERROR] Must be run inside Maya.")
        return

    print(f"[INFO] Stepping to frame {FRAME_TO_PLOT}...")
    cmds.currentTime(FRAME_TO_PLOT, edit=True)

    print("[INFO] Extracting mesh point clouds...")
    oor_cloud = get_mesh_point_cloud(OOR_MESH) if PLOT_OOR else None
    jaw_cloud = get_mesh_point_cloud(JAW_MESH) if PLOT_JAW else None
    skin_cloud = get_mesh_point_cloud(SKIN_MESH) if PLOT_SKIN else None
    
    oor_tracked = get_specific_vertices(OOR_MESH, OOR_VERTICES) if PLOT_OOR else None
    jaw_tracked = get_specific_vertices(JAW_MESH, JAW_VERTICES) if PLOT_JAW else None
    skin_tracked = get_specific_vertices(SKIN_MESH, SKIN_VERTICES) if PLOT_SKIN else None

    setup_siggraph_style()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Full Context Meshes (Point Clouds)
    if oor_cloud is not None:
        ax.scatter(oor_cloud[:, 0], oor_cloud[:, 1], oor_cloud[:, 2], 
                   c='#4A90D9', s=1, alpha=0.05, label='OOR Mesh')
                   
    if jaw_cloud is not None:
        # Subsample jaw if it's too dense to keep plot clean
        step = 1 if len(jaw_cloud) < 5000 else 5
        ax.scatter(jaw_cloud[::step, 0], jaw_cloud[::step, 1], jaw_cloud[::step, 2], 
                   c='gray', s=1, alpha=0.05, label='Jaw Mesh')

    if skin_cloud is not None:
        step = 1 if len(skin_cloud) < 5000 else 10
        ax.scatter(skin_cloud[::step, 0], skin_cloud[::step, 1], skin_cloud[::step, 2], 
                   c='#F5DEB3', s=2, alpha=0.2, label='Skin Mesh')

    # Plot the specific tracked vertices
    if oor_tracked is not None:
        ax.scatter(oor_tracked[:, 0], oor_tracked[:, 1], oor_tracked[:, 2], 
                   c='#E85D75', s=120, edgecolors='black', linewidth=1.5, depthshade=False, label='Tracked OOR Vertices')
                   
        # Add labels to points
        for i, idx in enumerate(OOR_VERTICES):
            ax.text(oor_tracked[i, 0], oor_tracked[i, 1], oor_tracked[i, 2], f' OOR[{idx}]', color='black', fontsize=9)

    if jaw_tracked is not None:
        ax.scatter(jaw_tracked[:, 0], jaw_tracked[:, 1], jaw_tracked[:, 2], 
                   c='#50C878', s=120, edgecolors='black', linewidth=1.5, depthshade=False, label='Tracked Jaw Vertices')
                   
        for i, idx in enumerate(JAW_VERTICES):
            ax.text(jaw_tracked[i, 0], jaw_tracked[i, 1], jaw_tracked[i, 2], f' Jaw[{idx}]', color='black', fontsize=9)

    if skin_tracked is not None:
        ax.scatter(skin_tracked[:, 0], skin_tracked[:, 1], skin_tracked[:, 2], 
                   c='#DAA520', s=120, edgecolors='black', linewidth=1.5, depthshade=False, label='Tracked Skin Vertices')
                   
        for i, idx in enumerate(SKIN_VERTICES):
            ax.text(skin_tracked[i, 0], skin_tracked[i, 1], skin_tracked[i, 2], f' Skin[{idx}]', color='black', fontsize=9)

    ax.set_title(f'Tracked Vertices Global Context (Frame {FRAME_TO_PLOT})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Ensure axes are equally scaled so faces don't look squished
    set_axes_equal(ax)
    
    ax.legend(loc='upper right')
    
    # Adjust viewing angle for better facial profile/3/4 view (rotated 90 degrees)
    ax.view_init(elev=85, azim=-90, roll=None)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"mesh_vertex_context_{timestamp}.png")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white')
    plt.close()
    
    print(f"[SUCCESS] Saved context plot to: {output_path}")

if __name__ == "__main__":
    plot_mesh_context()
