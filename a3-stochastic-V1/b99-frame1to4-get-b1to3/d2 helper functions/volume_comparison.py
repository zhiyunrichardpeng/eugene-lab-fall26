"""
Volume Comparison Script (OpenMaya 2.0)
Calculates and compares the volume of a mesh between physical simulation and a linear interpolation baseline.
Exports data to CSV and generates a SIGGRAPH TOG format plot.
"""

import maya.cmds as cmds
import maya.api.OpenMaya as om
import csv
import os
from paths import a3_stochastic_relative_path

try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError:
    cmds.error("matplotlib is not installed. Please install it in your Maya Python environment.")

def get_mesh_fn(mesh_name):
    """Get MFnMesh from a mesh node name."""
    selection_list = om.MSelectionList()
    try:
        selection_list.add(mesh_name)
    except Exception:
        cmds.error(f"Could not find mesh: {mesh_name}. Please check the name.")
        
    dag_path = selection_list.getDagPath(0)
    
    # Ensure it's a mesh shape or we get the shape if transform is selected
    if dag_path.apiType() == om.MFn.kTransform:
        dag_path.extendToShape()
        
    if dag_path.apiType() != om.MFn.kMesh:
        cmds.error(f"Object {mesh_name} is not a polygon mesh.")
        
    return om.MFnMesh(dag_path)

def calculate_mesh_volume(points, triangle_vertices):
    """
    Calculate the volume of a closed triangulated mesh using the signed volume of tetrahedra
    formed by each triangle and the origin. We center the mesh first to avoid floating point precision issues.
    """
    # 1. Calculate centroid to prevent float precision noise when mesh is far from origin
    cx, cy, cz = 0.0, 0.0, 0.0
    for p in points:
        cx += p.x
        cy += p.y
        cz += p.z
    n = len(points)
    if n > 0:
        cx /= n
        cy /= n
        cz /= n

    volume = 0.0
    
    # Iterate through the triangles and calculate signed volume
    num_triangles = len(triangle_vertices) // 3
    for i in range(num_triangles):
        v0_idx = triangle_vertices[i * 3]
        v1_idx = triangle_vertices[i * 3 + 1]
        v2_idx = triangle_vertices[i * 3 + 2]
        
        # Convert to MVector and shift by centroid
        p0 = om.MVector(points[v0_idx].x - cx, points[v0_idx].y - cy, points[v0_idx].z - cz)
        p1 = om.MVector(points[v1_idx].x - cx, points[v1_idx].y - cy, points[v1_idx].z - cz)
        p2 = om.MVector(points[v2_idx].x - cx, points[v2_idx].y - cy, points[v2_idx].z - cz)
        
        # Signed volume of tetrahedron from origin: (p0 dot (p1 cross p2)) / 6
        # In OM2: ^ is cross product, * is dot product
        signed_volume = p0 * (p1 ^ p2)
        volume += signed_volume
        
    return abs(volume / 6.0)

def plot_results(csv_file_path, plot_file_path, frames, kinematic_loss, physics_loss):
    """Plot the results using a SIGGRAPH TOG style formatting."""
    # SIGGRAPH TOG typical style: Serif font, clear lines, simple styling
    plt.style.use('classic')
    
    # Configure Matplotlib parameters for a scientific paper look
    matplotlib.rcParams['font.family'] = 'serif'
    # Fallback to general serif if Times New Roman is unavailable
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['legend.fontsize'] = 9
    matplotlib.rcParams['xtick.labelsize'] = 8
    matplotlib.rcParams['ytick.labelsize'] = 8
    
    # Single column width in SIGGRAPH TOG is ~3.5 inches
    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
    
    ax.plot(frames, kinematic_loss, label='Kinematic (Linear)', linestyle='--', color='#d62728', linewidth=1.5)
    ax.plot(frames, physics_loss, label='Physics (Simulated)', linestyle='-', color='#1f77b4', linewidth=1.5)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Volume Loss (%)')
    # Titles are typically omitted in TOG plots in favor of figure captions
    # ax.set_title('Volume Preservation') 
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best')
    
    # Adjust layout so labels are not cut off
    plt.tight_layout()
    
    # Save as PDF for the paper, and PNG for easy viewing
    plt.savefig(plot_file_path, format='pdf', bbox_inches='tight')
    plt.savefig(plot_file_path.replace('.pdf', '.png'), format='png', bbox_inches='tight')
    plt.close()

def main():
    # Setup Variables
    # Check if the UI's proxy render mesh exists, otherwise use raw Ziva mesh
    base_mesh = "middleres_OrbicularisOris"
    mesh_name = base_mesh + "_RENDER" if cmds.objExists(base_mesh + "_RENDER") else base_mesh
    # Automatically get the current timeline range from the Maya scene
    start_frame = int(cmds.playbackOptions(query=True, minTime=True))
    end_frame = int(cmds.playbackOptions(query=True, maxTime=True))
    
    output_dir = str(a3_stochastic_relative_path / "visualization")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    csv_file_path = os.path.join(output_dir, "volume_comparison4.csv")
    plot_file_path = os.path.join(output_dir, "volume_comparison4.pdf")
    
    # Store original time
    original_time = cmds.currentTime(query=True)
    
    try:
        print(f"Starting volume evaluation on {mesh_name} from frame {start_frame} to {end_frame}...")
        
        # 1. Initialize & Gather Data Sequentially for Ziva
        # First, jump to the start frame to reset any cache
        cmds.currentTime(start_frame, edit=True)
        mesh_fn = get_mesh_fn(mesh_name)
        
        # Get connectivity (assuming topology doesn't change)
        triangle_counts, triangle_vertices = mesh_fn.getTriangles()
        
        # We will collect the vertex positions at each frame sequentially
        # so Ziva has a chance to evaluate correctly without jumping around
        physics_positions_per_frame = {}
        
        print("Stepping timeline linearly to allow Ziva to evaluate...")
        for t in range(start_frame, end_frame + 1):
            cmds.currentTime(t, edit=True)
            mesh_fn = get_mesh_fn(mesh_name)
            physics_positions_per_frame[t] = mesh_fn.getPoints(om.MSpace.kWorld)
            
        # 2. Extract Key Poses
        p_rest = physics_positions_per_frame[start_frame]
        p_target = physics_positions_per_frame[end_frame]
        
        rest_volume = calculate_mesh_volume(p_rest, triangle_vertices)
        
        results = []
        frames = []
        kinematic_loss_list = []
        physics_loss_list = []
        
        # 3. Evaluate Results
        print("Calculating volumes...")
        for t in range(start_frame, end_frame + 1):
            # Calculate weight w
            w = float(t - start_frame) / float(end_frame - start_frame) if end_frame > start_frame else 0.0
            
            # Kinematic calculation (Linear Interpolation)
            p_kinematic = om.MPointArray()
            p_kinematic.setLength(len(p_rest))
            for i in range(len(p_rest)):
                p_kinematic[i] = om.MPoint(
                    (1.0 - w) * p_rest[i].x + w * p_target[i].x,
                    (1.0 - w) * p_rest[i].y + w * p_target[i].y,
                    (1.0 - w) * p_rest[i].z + w * p_target[i].z,
                    1.0
                )
            kinematic_volume = calculate_mesh_volume(p_kinematic, triangle_vertices)
            
            # Physics calculation (from collected Scene Data)
            p_physics = physics_positions_per_frame[t]
            physics_volume = calculate_mesh_volume(p_physics, triangle_vertices)
            
            # Loss Percentages
            kinematic_loss = ((rest_volume - kinematic_volume) / rest_volume) * 100.0 if rest_volume > 0 else 0.0
            physics_loss = ((rest_volume - physics_volume) / rest_volume) * 100.0 if rest_volume > 0 else 0.0
            
            # Keep track of data
            results.append({
                'Frame': t,
                'Rest_Volume': rest_volume,
                'Kinematic_Volume': kinematic_volume,
                'Physics_Volume': physics_volume,
                'Kinematic_Volume_Loss_Percentage': kinematic_loss,
                'Physics_Volume_Loss_Percentage': physics_loss
            })
            frames.append(t)
            kinematic_loss_list.append(kinematic_loss)
            physics_loss_list.append(physics_loss)
            
        # 3. Data Export to CSV
        with open(csv_file_path, mode='w', newline='') as file:
            fieldnames = ['Frame', 'Rest_Volume', 'Kinematic_Volume', 'Physics_Volume', 'Kinematic_Volume_Loss_Percentage', 'Physics_Volume_Loss_Percentage']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            
        print(f"Data successfully exported to: {csv_file_path}")
        
        # 4. Plot using SIGGRAPH TOG format
        plot_results(csv_file_path, plot_file_path, frames, kinematic_loss_list, physics_loss_list)
        print(f"Plot successfully saved to: {plot_file_path}")
        print("Done!")
        
    finally:
        # Restore original time
        cmds.currentTime(original_time)

if __name__ == "__main__":
    main()
