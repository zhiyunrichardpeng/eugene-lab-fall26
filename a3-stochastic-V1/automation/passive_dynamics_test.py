"""
Passive Dynamics & Momentum Test Script
This script applies a sudden step input to the 'head' joint and measures the
resulting passive elastic dynamics (jiggle/overshoot) of the skin and muscle.
"""

import maya.cmds as cmds
import maya.api.OpenMaya as om
import csv
import os
import math
from paths import a3_stochastic_relative_path

try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError:
    cmds.error("matplotlib is not installed. Please install it in your Maya Python environment.")

# --- Configuration ---
# Update these names to match your scene exactly
ANIMATION_TARGETS = ["middleres_Jaw", "middleres_skull"]
SKIN_VERTEX = "skin_cloth_copy_v5_pull_back.vtx[5000]" 
MUSCLE_VERTEX = "middleres_OrbicularisOris.vtx[100]"

START_FRAME = 0
SHAKE_START_FRAME = 16
SHAKE_LOOPS = 3
FRAMES_PER_LOOP = 10
SHAKE_END_FRAME = SHAKE_START_FRAME + (SHAKE_LOOPS * FRAMES_PER_LOOP)
END_FRAME = SHAKE_END_FRAME + 40 # Give it time to settle afterwards

OUTPUT_DIR = os.path.join(a3_stochastic_relative_path," automation")
CSV_FILE_PATH = os.path.join(OUTPUT_DIR, "passive_dynamics_results_V5.csv")
PLOT_FILE_PATH = os.path.join(OUTPUT_DIR, "passive_dynamics_plot_V5.pdf")

def get_vertex_world_position(vtx_name):
    """Get the world space position of a specific vertex."""
    pos = cmds.xform(vtx_name, query=True, worldSpace=True, translation=True)
    return om.MVector(pos[0], pos[1], pos[2])

def setup_animation():
    """Applies a smooth left-right shake input to the target meshes."""
    # Ensure auto-key is off or we manually key
    cmds.currentTime(START_FRAME, edit=True)
    
    magnitude = 5.0  
    
    for target in ANIMATION_TARGETS:
        # Store initial translation
        initial_tx = cmds.getAttr(f"{target}.tx")
        
        # Frame 0 to 15 (equilibrium)
        cmds.setKeyframe(target, attribute='tx', time=START_FRAME, value=initial_tx)
        cmds.setKeyframe(target, attribute='tx', time=SHAKE_START_FRAME, value=initial_tx)
        
        # Keyframes to create the shakes
        shake_frames = []
        current_f = SHAKE_START_FRAME
        
        for i in range(SHAKE_LOOPS):
            # A loop takes FRAMES_PER_LOOP frames.
            # We roughly divide it into 4 quarters: center->left->center->right->center
            q = FRAMES_PER_LOOP / 4.0
            shake_frames.append((current_f + q, initial_tx + magnitude))         # Left
            shake_frames.append((current_f + 2*q, initial_tx))                   # Center
            shake_frames.append((current_f + 3*q, initial_tx - magnitude))       # Right
            current_f += FRAMES_PER_LOOP
            shake_frames.append((current_f, initial_tx))                         # Center
            
        for frame, val in shake_frames:
            cmds.setKeyframe(target, attribute='tx', time=frame, value=val)
        
        # Set tangents to 'auto' to ensure a smooth, human-like ease in/ease out
        cmds.keyTangent(target, attribute='tx', time=(START_FRAME, SHAKE_END_FRAME), inTangentType='auto', outTangentType='auto')

def plot_results(csv_file, plot_file, frames, input_vel, skin_vel, muscle_vel):
    """Plot the velocities using SIGGRAPH TOG style."""
    plt.style.use('classic')
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['legend.fontsize'] = 9
    matplotlib.rcParams['xtick.labelsize'] = 8
    matplotlib.rcParams['ytick.labelsize'] = 8
    
    fig, ax = plt.subplots(figsize=(5.5, 3.0), dpi=300)
    
    ax.plot(frames, input_vel, label='Input (Head Joint)', linestyle=':', color='#7f7f7f', linewidth=1.5)
    ax.plot(frames, muscle_vel, label='Muscle (OOR)', linestyle='--', color='#2ca02c', linewidth=1.5)
    ax.plot(frames, skin_vel, label='Skin', linestyle='-', color='#1f77b4', linewidth=1.5)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Velocity (cm/frame)')
    
    # Highlight the shake phase
    ax.axvspan(SHAKE_START_FRAME, SHAKE_END_FRAME, color='gray', alpha=0.2, label='Input Signal (Shake Phase)')
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(plot_file, format='pdf', bbox_inches='tight')
    plt.savefig(plot_file.replace('.pdf', '.png'), format='png', bbox_inches='tight')
    plt.close()

def main():
    original_time = cmds.currentTime(query=True)
    
    try:
        print("Setting up animation...")
        setup_animation()
        
        print("Running simulation and collecting data...")
        positions = {'head': {}, 'skin': {}, 'muscle': {}}
        
        # Sequentially step through the timeline to evaluate physics
        for t in range(START_FRAME, END_FRAME + 1):
            cmds.currentTime(t, edit=True)
            
            # Record positions
            # For head, track a point offset from the joint to capture angular velocity
            # Or just track the raw rotation/translation. Here we track a point relative to it.
            # Using the joint's transform world position
            # Using the skull's transform world position as the reference
            positions['head'][t] = get_vertex_world_position(ANIMATION_TARGETS[1])
            positions['skin'][t] = get_vertex_world_position(SKIN_VERTEX)
            positions['muscle'][t] = get_vertex_world_position(MUSCLE_VERTEX)
            
        print("Calculating velocities...")
        results = []
        frames = []
        input_vel_list = []
        skin_vel_list = []
        muscle_vel_list = []
        
        for t in range(START_FRAME, END_FRAME + 1):
            if t == START_FRAME:
                v_head, v_skin, v_muscle = 0.0, 0.0, 0.0
            else:
                # Calculate velocity as distance moved from previous frame
                v_head = (positions['head'][t] - positions['head'][t-1]).length()
                v_skin = (positions['skin'][t] - positions['skin'][t-1]).length()
                v_muscle = (positions['muscle'][t] - positions['muscle'][t-1]).length()
                
            # If tracking the joint center isn't accurate enough for some reason,
            # we can fall back to reading the translation attribute directly.
            if v_head < 1e-4:
                tx_t = cmds.getAttr(f"{ANIMATION_TARGETS[1]}.tx", time=t)
                tx_prev = cmds.getAttr(f"{ANIMATION_TARGETS[1]}.tx", time=t-1) if t > START_FRAME else tx_t
                v_head = abs(tx_t - tx_prev)
                
            frames.append(t)
            input_vel_list.append(v_head)
            skin_vel_list.append(v_skin)
            muscle_vel_list.append(v_muscle)
            
            results.append({
                'Frame': t,
                'Input_Velocity': v_head,
                'Skin_Velocity': v_skin,
                'Muscle_Velocity': v_muscle
            })
            
        print("Exporting data to CSV...")
        with open(CSV_FILE_PATH, mode='w', newline='') as file:
            fieldnames = ['Frame', 'Input_Velocity', 'Skin_Velocity', 'Muscle_Velocity']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            
        print("Plotting results...")
        plot_results(CSV_FILE_PATH, PLOT_FILE_PATH, frames, input_vel_list, skin_vel_list, muscle_vel_list)
        
        print(f"Success! Saved to {OUTPUT_DIR}")
        
    except Exception as e:
        cmds.error(f"Error during execution: {str(e)}")
    finally:
        cmds.currentTime(original_time)

if __name__ == "__main__":
    main()
