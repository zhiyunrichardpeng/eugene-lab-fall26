"""
Passive Dynamics Observer Script (Linear Evaluation)
This script directly evaluates the Maya timeline linearly to record positions 
of the bone, skin, and muscle. By evaluating linearly, it works perfectly 
with Ziva cached playback and allows pure passive jiggle observation.
"""

import maya.cmds as cmds
import maya.api.OpenMaya as om
import csv
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError:
    cmds.error("matplotlib is not installed. Please install it in your Maya Python environment.")

# --- Configuration ---
BONE_REFERENCE = "middleres_skull"
SKIN_VERTEX = "skin_cloth_copy_v5_pull_back.vtx[11109]" 
MUSCLE_VERTEX = "middleres_OrbicularisOris.vtx[100]"

OUTPUT_DIR = os.path.join(a3_stochastic_relative_path,automation"
CSV_FILE_PATH = os.path.join(OUTPUT_DIR, "observed_passive_dynamics_V3_SKIN.csv")
PLOT_FILE_PATH = os.path.join(OUTPUT_DIR, "observed_passive_dynamics_plot_V3_SKIN.pdf")

def get_vertex_world_position(vtx_name):
    """Get the world space position of a specific vertex."""
    pos = cmds.xform(vtx_name, query=True, worldSpace=True, translation=True)
    return om.MVector(pos[0], pos[1], pos[2])

def plot_results(csv_file, plot_file, frames, skin_stretch, muscle_stretch, skin_rel_vel, muscle_rel_vel):
    """Plot the relative metrics using SIGGRAPH TOG style."""
    plt.style.use('classic')
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['legend.fontsize'] = 9
    matplotlib.rcParams['xtick.labelsize'] = 8
    matplotlib.rcParams['ytick.labelsize'] = 8
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 5.0), dpi=300, sharex=True)
    
    # --- Plot 1: Relative Displacement (Stretch) ---
    ax1.plot(frames, muscle_stretch, label='Muscle Stretch (OOR)', linestyle='-', color='#2ca02c', linewidth=1.5)
    ax1.plot(frames, skin_stretch, label='Skin Stretch', linestyle='-', color='#1f77b4', linewidth=1.5)
    
    ax1.set_ylabel('Stretch / Sag (cm)')
    ax1.set_title('Tissue Displacement Relative to Skull')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # --- Plot 2: Relative Velocity (Jiggle Speed) ---
    ax2.plot(frames, muscle_rel_vel, label='Muscle Jiggle Velocity', linestyle='--', color='#2ca02c', linewidth=1.5)
    ax2.plot(frames, skin_rel_vel, label='Skin Jiggle Velocity', linestyle='--', color='#1f77b4', linewidth=1.5)
    
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Relative Velocity (cm/frame)')
    ax2.set_title('Tissue Velocity Relative to Skull (Pure Jiggle)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(plot_file, format='pdf', bbox_inches='tight')
    plt.savefig(plot_file.replace('.pdf', '.png'), format='png', bbox_inches='tight')
    plt.close()

def main():
    original_time = cmds.currentTime(query=True)
    
    try:
        start_frame = int(cmds.playbackOptions(q=True, minTime=True))
        end_frame = int(cmds.playbackOptions(q=True, maxTime=True))
        
        print(f"--- Running Standalone Observer ({start_frame} to {end_frame}) ---")
        
        # Dictionary to store recorded data
        positions = {'head': {}, 'skin': {}, 'muscle': {}}
        
        # Jump to start frame to reset any live caches properly
        cmds.currentTime(start_frame, edit=True)
        
        # 1. Sequentially step through the timeline
        print("Stepping timeline linearly to read simulation/cache data...")
        for t in range(start_frame, end_frame + 1):
            cmds.currentTime(t, edit=True)
            try:
                positions['head'][t] = get_vertex_world_position(BONE_REFERENCE)
                positions['skin'][t] = get_vertex_world_position(SKIN_VERTEX)
                positions['muscle'][t] = get_vertex_world_position(MUSCLE_VERTEX)
            except Exception as e:
                print(f"[Warning] Frame {t}: Could not read vertices. Error: {e}")
                
        print("Calculating relative dynamics (isolating passive jiggle)...")
        results = []
        frames = []
        skin_stretch_list = []
        muscle_stretch_list = []
        skin_rel_vel_list = []
        muscle_rel_vel_list = []
        
        # Validate data was captured
        if not positions['head']:
            cmds.error("No valid position data was recorded.")
            return
            
        # Get resting distances at the start frame
        rest_dist_skin = (positions['skin'][start_frame] - positions['head'][start_frame]).length()
        rest_dist_muscle = (positions['muscle'][start_frame] - positions['head'][start_frame]).length()
        
        for t in range(start_frame, end_frame + 1):
            if t not in positions['head']:
                continue
                
            frames.append(t)
            
            # 2. Calculate Stretch (Distance between tissue and head, minus the resting distance)
            curr_dist_skin = (positions['skin'][t] - positions['head'][t]).length()
            curr_dist_muscle = (positions['muscle'][t] - positions['head'][t]).length()
            
            s_skin = curr_dist_skin - rest_dist_skin
            s_muscle = curr_dist_muscle - rest_dist_muscle
            
            skin_stretch_list.append(s_skin)
            muscle_stretch_list.append(s_muscle)
            
            # 3. Calculate Relative Velocity
            if t == start_frame:
                v_skin, v_muscle = 0.0, 0.0
            else:
                rel_pos_skin_curr = positions['skin'][t] - positions['head'][t]
                rel_pos_skin_prev = positions['skin'][t-1] - positions['head'][t-1]
                v_skin = (rel_pos_skin_curr - rel_pos_skin_prev).length()
                
                rel_pos_muscle_curr = positions['muscle'][t] - positions['head'][t]
                rel_pos_muscle_prev = positions['muscle'][t-1] - positions['head'][t-1]
                v_muscle = (rel_pos_muscle_curr - rel_pos_muscle_prev).length()
                
            skin_rel_vel_list.append(v_skin)
            muscle_rel_vel_list.append(v_muscle)
            
            results.append({
                'Frame': t,
                'Skin_Stretch': s_skin,
                'Muscle_Stretch': s_muscle,
                'Skin_Relative_Velocity': v_skin,
                'Muscle_Relative_Velocity': v_muscle
            })
            
        print("Exporting data to CSV...")
        with open(CSV_FILE_PATH, mode='w', newline='') as file:
            fieldnames = ['Frame', 'Skin_Stretch', 'Muscle_Stretch', 'Skin_Relative_Velocity', 'Muscle_Relative_Velocity']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            
        print("Plotting relative dynamics...")
        plot_results(CSV_FILE_PATH, PLOT_FILE_PATH, frames, skin_stretch_list, muscle_stretch_list, skin_rel_vel_list, muscle_rel_vel_list)
        
        print(f"[Observer] Success! Saved passive dynamics observations to {OUTPUT_DIR}")
        
    except Exception as e:
        cmds.error(f"Error during execution: {str(e)}")
    finally:
        cmds.currentTime(original_time)

if __name__ == "__main__":
    main()
