import maya.cmds as cmds
import matplotlib.pyplot as plt
import os
import math
from paths import b99_frame1to4_path

def debug_blendshape_morph(snap_frame=374, buffer_frames=15, target_mesh="middleres_OrbicularisOris", vertex_idx=0):
    """
    Steps the timeline from (snap_frame - 5) to (snap_frame + buffer_frames + 5).
    Records the Y-position of a specific vertex on the mesh to verify the morph is smooth.
    Plots and saves the result.
    """
    print(f"--- Starting Visual Debug for {target_mesh} (Vertex {vertex_idx}) ---")
    
    frames = []
    y_positions = []
    
    start_f = snap_frame - 5
    end_f = snap_frame + buffer_frames + 5
    
    # Store current time
    orig_time = cmds.currentTime(query=True)
    
    for f in range(start_f, end_f + 1):
        cmds.currentTime(f, edit=True)
        # Get absolute world translation of the vertex
        pos = cmds.xform(f"{target_mesh}.vtx[{vertex_idx}]", query=True, translation=True, worldSpace=True)
        
        frames.append(f)
        y_positions.append(pos[1])  # Y axis
        print(f"Frame {f}: Vertex Y = {pos[1]:.4f}")
        
    # Restore time
    cmds.currentTime(orig_time, edit=True)
    
    # === PLOT THE DATA ===
    plt.figure(figsize=(10, 6))
    plt.plot(frames, y_positions, marker='o', linestyle='-', color='b', label='Vertex Y Position')
    
    # Mark the snap zone
    plt.axvspan(snap_frame, snap_frame + buffer_frames, color='red', alpha=0.2, label='Blendshape Active')
    plt.axvline(x=snap_frame, color='red', linestyle='--', label='Snap Occurs')
    
    plt.title(f"Smooth Snap Debug: {target_mesh} Vertex {vertex_idx}")
    plt.xlabel("Frame")
    plt.ylabel("World Y Position")
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    debug_dir = str(b99_frame1to4_path / "debug_results")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
        
    plot_path = os.path.join(debug_dir, f"vertex_morph_plot_{snap_frame}.png")
    plt.savefig(plot_path)
    print(f"Saved plot to: {plot_path}")
    plt.close()

# Run the debug (assuming snap happened at 374, change this if needed)
debug_blendshape_morph(snap_frame=374, buffer_frames=15)
