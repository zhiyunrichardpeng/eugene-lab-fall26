# have to run this code in maya
# open this maya file: 
# C:\keyLabLocal_D\c99-maya-ziva-workshop\experiment_files\g10-10 rest pose 1200 landmarks v9.mb

base_frames_global = 10
# commandPort -name "localhost:7001" -sourceType "mel";
# ensure_proxy_render_meshes()

import maya.cmds as cmds
import os
import re
import math
import pickle
import numpy as np
import json
# pyrefly: ignore [missing-import]
import maya.OpenMaya as OpenMaya
import csv
import datetime


import sys
sys.path.insert(0, r"C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo\example_train")
sys.path.insert(0, r"C:\Users\zpa18\a3-stochastic-V1\a1-blendshape-guided-animation-pipeline\database")
sys.path.insert(0, r"C:\Users\zpa18\a3-stochastic-V1\b98-sampling-joint")
sys.path.insert(0, r"C:\Users\zpa18\a3-stochastic-V1\b98-sampling-joint\300joints_suite_good_old")
sys.path.insert(0, r"C:\Users\zpa18\a3-stochastic-V1\b98-sampling-joint\joint_positions\joint_position_ooc_l")
sys.path.insert(0, r"C:\Users\zpa18\a3-stochastic-V1\b98-sampling-joint\joint_positions\joint_position_ooc_r")
sys.path.insert(0, r"C:\Users\zpa18\a3-stochastic-V1\b98-sampling-joint\joint_positions\joint_position_frontalis")
sys.path.insert(0, r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\b31 dataV22 backup hub")
sys.path.insert(0, r"C:\Users\zpa18\a3-stochastic-V1\b95-debug-use")



# Get neutral pose positions from joint_position modules (TRUE neutral reference)
# These are the same positions used during training, NOT from Maya scene
import joint_position
import joint_position_first_wave
import joint_position_third_wave
import joint_position_ooc_l_base_0
import joint_position_ooc_l_base_pose11_0
import joint_position_ooc_l_base_pose11_AU1_0
import joint_position_ooc_l_base_pose11_AU1_AU2_0
import joint_position_ooc_r_base_0
import joint_position_ooc_r_base_pose11_0
import joint_position_ooc_r_base_pose11_AU1_0
import joint_position_ooc_r_base_pose11_AU1_AU2_0
import joint_position_frontalis_base_0
import joint_position_frontalis_base_AU1_0
import joint_position_frontalis_base_AU1_AU2_0


# --- 0. DEVELOPMENT CONFIGURATION ---
DEVELOPMENT_MODE = False # Set to False for production use
VERBOSE_MODE = False # Set to True to see all detailed debug prints

# --- CUSTOM PRINT FUNCTION (Verbose Level Control) ---
import builtins
_original_print = builtins.print
def smart_print(*args, **kwargs):
    global VERBOSE_MODE
    if VERBOSE_MODE:
        _original_print(*args, **kwargs)
    else:
        text_args = ' '.join([str(a) for a in args])
        text_lower = text_args.lower()
        if any(keyword in text_lower for keyword in ['illegal', 'finished', 'complete!', 'all transitions complete']):
            _original_print(*args, **kwargs)

print = smart_print

# In development mode, the script will use the hardcoded sequence below
# and skip processing if the deformation file already exists.
# DEV_SEQUENCE_STRING = "Neutral to phoneme_Y to phoneme_UW to phoneme_V to phoneme_N to phoneme_AA to phoneme_T to phoneme_IY to phoneme_T to phoneme_N to phoneme_IY to phoneme_T to phoneme_Y to phoneme_EH to phoneme_T"
DEV_SEQUENCE_STRING = "Neutral to phoneme_N to phoneme_AA to phoneme_T to phoneme_IY to phoneme_V to phoneme_N to phoneme_Y to phoneme_UW"
# Neutral to phoneme_N to phoneme_IY to phoneme_T to phoneme_Y to phoneme_EH to phoneme_T
# Neutral to phoneme_AA

# --- ANIMATION CHAINING CONFIGURATION ---
# Stores the last animation's metadata for chaining
LAST_ANIMATION_END_FRAME = 14  # Default rest pose frame
LAST_ANIMATION_PHONEME = "dataV22_AU0_neutral"  # Default starting phoneme
CONTINUE_FROM_LAST = True  # Set to True to continue from last animation

# --- OVERLAP CONFIGURATION ---
# Set to 0.4 to use overlap-aware checkpoints from [overlap40] DB keys
# Set to 0.0 to use standard (no-overlap) checkpoints
OVERLAP_RATIO = 0.0  # 0.0 = standard, 0.4 = 40% overlap co-articulation

# --- FEEDBACK CONTROL CONFIGURATION ---
# Dissipating Error Offset (Feedback Control Loop)
ENABLE_FEEDBACK_CONTROL = True  # Enable/disable feedback control loop
ERROR_DISSIPATION_RATE = 1.0 # 0.5  # How much of the error to apply (0.0 to 1.0)
# 0.5 means 50% of error is corrected in current transition, 50% persists to next
# Higher value = more aggressive correction, faster convergence
# Lower value = smoother correction, more gradual convergence

# Global error accumulator for feedback control
ACCUMULATED_ERROR = None  # Will store the accumulated position error across transitions

# --- DEBUG CONFIGURATION ---
DEBUG_OUTPUT_DIR = r"C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\d4 interactive interface of animation\h96-debug-dissipating-error"
DEBUG_ENABLED = True  # Enable detailed debugging and logging
DEBUG_DATA = []  # Stores debug information for each transition

# --- KEYBOARD MODE CONFIGURATION ---
# Piano-style keyboard mode: Press keys to trigger phoneme transitions
KEYBOARD_MODE_ENABLED = False  # Enable/disable keyboard mode
KEYBOARD_CURRENT_PHONEME = "dataV22_AU0_neutral"  # Current phoneme state
KEYBOARD_SCRIPT_JOBS = []  # Store script job IDs for cleanup

# Key to Phoneme mapping (using intuitive letters)
# Press the letter that corresponds to the phoneme sound
KEYBOARD_PHONEME_MAP = {
    '0': {'phoneme': 'dataV22_AU0_neutral', 'label': 'Neutral', 'key': '0'},
    'n': {'phoneme': 'dataV22_phoneme_N', 'label': 'N', 'key': 'n'},
    'a': {'phoneme': 'dataV22_phoneme_AA', 'label': 'AA', 'key': 'a'},
    'i': {'phoneme': 'dataV22_phoneme_IY', 'label': 'IY', 'key': 'i'},
    't': {'phoneme': 'dataV22_phoneme_T', 'label': 'T', 'key': 't'},
    'e': {'phoneme': 'dataV22_phoneme_EH', 'label': 'EH', 'key': 'e'},
    'u': {'phoneme': 'dataV22_phoneme_UW', 'label': 'UW', 'key': 'u'},
    'v': {'phoneme': 'dataV22_phoneme_V', 'label': 'V', 'key': 'v'},
    'y': {'phoneme': 'dataV22_phoneme_Y', 'label': 'Y', 'key': 'y'},
    's': {'phoneme': 'dataV22_AU12', 'label': 'Smile', 'key': 's'},
    'f': {'phoneme': 'dataV22_AU4', 'label': 'Frown', 'key': 'f'},
    'b': {'phoneme': 'dataV22_AU2', 'label': 'Brow', 'key': 'b'},
    'm': {'phoneme': 'dataV22_AU25', 'label': 'Mouth', 'key': 'm'},
    'q': {'phoneme': 'dataV22_AU7', 'label': 'Squint', 'key': 'q'},
    'c': {'phoneme': 'dataV22_AU6', 'label': 'Cheek', 'key': 'c'},
    'p': {'phoneme': 'dataV22_AU22', 'label': 'Pucker', 'key': 'p'},
}

# Reverse lookup: phoneme to key info
PHONEME_TO_KEY = {v['phoneme']: v for k, v in KEYBOARD_PHONEME_MAP.items()}

# --- 1. DEFORMATION PARAMETERS DATABASE (Your "Friendly Database") ---
#
# Load deformation path database and state map from JSON file
# The deformation_paths key maps "StateA_to_StateB" to file paths
# The state_map key normalizes user input (e.g., "au12" becomes "AU12")
#
def load_database():
    """Load the deformation path database and state map from the JSON file"""
    # Use hardcoded base directory path since __file__ is not available in Maya
    base_dir = r"c:\Users\zpa18\a3-stochastic-V1"
    db_path = os.path.join(base_dir, "automation", "deformation_path_db.json")

    try:
        with open(db_path, 'r') as f:
            data = json.load(f)
            return data.get('deformation_paths', {}), data.get('state_map', {})
    except FileNotFoundError:
        print(f"WARNING: Database file not found at {db_path}")
        return {}, {}
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON database: {e}")
        return {}, {}

DEFORMATION_PATH_DB, STATE_MAP = load_database()

# Base path for saving the combined deformation parameters
SEQUENCE_DB_PATH = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\sequence_combined_deformation_parameters_hub"

# Global variables for pose configuration (set by parse_sequence_string)
NEUTRAL_TO_OTHER_POSE_FLAG = 1  # Default to neutral start
PREVIOUS_POSE_FOLDER = ""  # Will be set if non-neutral start

# List of all soft tissue meshes to process for anti-drift morphing (excludes rigid bodies)
SOFT_TISSUE_MESHES = [
    "polySurface11", "middleres_ZygomaticusMinor_r", "polySurface1", "polySurface3", 
    "middleres_OrbicularisOculi_r", "middleres_Temporalis_l", "middleres_NasalisTransverse", 
    "middleres_DepressorSeptiNasi", "middleres_ZygomaticusMajor_r", "middleres_LateralCartige", 
    "middleres_ZygomaticusMinor_l", "middleres_LevatorLabiiSuperioris_r", "middleres_Frontalis", 
    "middleres_Temporalis_r", "polySurface14", "middleres_Procerus", "middleres_DepressorSupercilii_r", 
    "polySurface12", "polySurface5", "polySurface4", "middleres_NostrilWing", "middleres_MasseterDeep", 
    "polySurface2", "middleres_DepressorSupercilii_l", "polySurface10", "middleres_Mentalis", 
    "middleres_AuricularisAnterior", "middleres_AlarCartidge", "polySurface8", "polySurface7", 
    "middleres_DepressorLabiiInferioris_l", "middleres_LevatorLabiiSuperioris_l", "middleres_ZygomaticusMajor_l", 
    "middleres_AuricularisPosterior", "polySurface9", "polySurface13", "polySurface6", 
    "middleres_DepressorLabiiInferioris_r", "middleres_OrbicularisOculi_l", "middleres_AuricularisSuperior", 
    "middleres_OrbicularisOris", "skin_cloth_copy_v5_pull_back"
]


def normalize_task_key(task_string):
    """Converts a task string into a standardized key format for lookups."""
    import re
    key = str(task_string).lower()
    key = key.replace('_to_', ' to ')  # bring to a common format
    key = re.sub(r'\s*to\s*', '_to_', key)
    key = re.sub(r'\s+', '_', key)
    return key


def auto_configure_previous_pose_folder(from_pose):
    """
    Automatically searches for the GENERATED_RESULTS_PATH based on the from_pose.
    This mimics the logic from run_pipeline_update.py for finding prerequisite tasks.

    Args:
        from_pose: The starting pose (e.g., "phoneme_N", "AU6")

    Returns:
        The path to the previous pose folder, or None if not found
    """
    # Check if starting from neutral pose
    if normalize_task_key(from_pose) == 'datav22_au0_neutral':
        print(f"[INFO] Task starts from neutral pose. No previous pose folder needed.")
        return None

    # Build prerequisite task string
    source_task_string = f"dataV22_AU0_neutral to {from_pose}"
    normalized_source_key = normalize_task_key(source_task_string)

    print(f"[INFO] Task starts from non-neutral pose '{from_pose}'. Looking for prerequisite task '{source_task_string}'...")
    print(f"--- DEBUG: Prerequisite Lookup ---")
    print(f"  Normalized key to find: '{normalized_source_key}'")

    # Create normalized key map
    db_key_map = {normalize_task_key(k): k for k in DEFORMATION_PATH_DB}

    # Look up the original key
    original_source_key = db_key_map.get(normalized_source_key)

    if not original_source_key:
        print(f"[ERROR] Cannot perform task. Prerequisite task '{source_task_string}' not found in database.")
        print(f"[ERROR] Please run that task first or manually set the previous_pose_folder.")
        return None

    # Get the path from the database
    GENERATED_RESULTS_PATH = DEFORMATION_PATH_DB[original_source_key]
    print(f"[INFO] Found prerequisite path: {GENERATED_RESULTS_PATH}")

    return GENERATED_RESULTS_PATH


# --- 2. MAIN DEFORMATION SCRIPT (Your "Part 2") ---
#
# This is your entire processing script, converted into a function.
# It now takes 'phonemes_list' and 'log_paths_list' as arguments
# instead of having them hard-coded.
#
# --- 2. NEW DEFORMATION HELPERS AND MAIN FUNCTION ---

# Global live tracking arrays to avoid redundant simulation passes
LIVE_DEBUG_FRAMES = []
LIVE_DEBUG_POSITIONS = []

def step_timeline_forward(target_frame):
    """
    Steps the timeline forward one frame at a time from current to target_frame.
    This ensures Ziva's physics cache is perfectly evaluated and we never jump backwards.
    It simultaneously collects live debug data so we don't have to simulate again at the end.
    """
    global LIVE_DEBUG_FRAMES, LIVE_DEBUG_POSITIONS
    
    current_time = int(cmds.currentTime(query=True))
    target = int(target_frame)
    
    # Live target logic (prioritize proxy render mesh if it exists)
    t_mesh = "middleres_OrbicularisOris_RENDER" if cmds.objExists("middleres_OrbicularisOris_RENDER") else "middleres_OrbicularisOris"
    
    # If starting fresh at frame 0, collect frame 0 data
    if current_time == 0 and len(LIVE_DEBUG_FRAMES) == 0:
        try:
            pos = cmds.xform(f"{t_mesh}.vtx[0]", query=True, translation=True, worldSpace=True)
            LIVE_DEBUG_FRAMES.append(0)
            LIVE_DEBUG_POSITIONS.append(pos[1])
        except Exception:
            pass
            
    if current_time < target:
        print(f"  [PHYSICS] Stepping timeline sequentially from {current_time} to {target} to preserve Ziva cache...")
        for f in range(current_time + 1, target + 1):
            cmds.currentTime(f, edit=True)
            
            # Collect live position
            try:
                # Always check for render mesh in case it was created during this step
                live_mesh = "middleres_OrbicularisOris_RENDER" if cmds.objExists("middleres_OrbicularisOris_RENDER") else "middleres_OrbicularisOris"
                pos = cmds.xform(f"{live_mesh}.vtx[0]", query=True, translation=True, worldSpace=True)
                LIVE_DEBUG_FRAMES.append(f)
                LIVE_DEBUG_POSITIONS.append(pos[1])
            except Exception:
                pass
                
    elif current_time > target:
        print(f"  [CRITICAL ERROR] Current time ({current_time}) is greater than target ({target}). Ziva forbids backward jumps! Skipping timeline change.")
        return

# --- HIDDEN SNAP (ZIVA RESET) ---
def insert_ziva_hidden_snap(snap_frame):
    """
    Inserts a Hidden Snap at a specific frame by keyframing the Ziva solver's startFrame.
    This is the bulletproof way to completely flush accumulated FEM numerical drift.
    It forces Ziva to dump its internal velocity and stress and re-initialize the rest state.
    """
    solvers = cmds.ls(type='zSolver')
    if not solvers:
        print(f"[PHYSICS] Warning: No zSolver found. Could not insert hidden snap at frame {snap_frame}.")
        return

    for solver_shape in solvers:
        # cmds.ls(type='zSolver') returns the SHAPE node (e.g., zSolver1Shape)
        # But 'startFrame' is an attribute on the TRANSFORM node (e.g., zSolver1)
        solver_transform = cmds.listRelatives(solver_shape, parent=True)[0]

        # The ultimate Ziva flush: Animate the startFrame!
        if cmds.attributeQuery('startFrame', node=solver_transform, exists=True):
            # 1. Maintain the original start frame (usually 1) right up until the snap
            original_start = cmds.getAttr(f"{solver_transform}.startFrame", time=1)
            if original_start is None: 
                original_start = 1
                
            cmds.setKeyframe(f"{solver_transform}.startFrame", time=snap_frame - 1, value=original_start)
            
            # 2. At the exact snap frame, set the startFrame to the current frame!
            cmds.setKeyframe(f"{solver_transform}.startFrame", time=snap_frame, value=snap_frame)
        else:
            print(f"[PHYSICS] Warning: 'startFrame' attribute not found on {solver_transform}.")

        # ALSO flush any native Maya nCloth (nucleus) solvers!
        # If the skin uses nCloth, failing to reset it while Ziva resets causes it to explode violently.
        nucleus_nodes = cmds.ls(type='nucleus')
        for nucleus in nucleus_nodes:
            if cmds.attributeQuery('startFrame', node=nucleus, exists=True):
                orig_start = cmds.getAttr(f"{nucleus}.startFrame", time=1)
                if orig_start is None:
                    orig_start = 1
                    
                cmds.setKeyframe(f"{nucleus}.startFrame", time=snap_frame - 1, value=orig_start)
                cmds.setKeyframe(f"{nucleus}.startFrame", time=snap_frame, value=snap_frame)
                print(f"[PHYSICS] Inserted 'Hidden Snap' for nucleus ({nucleus}) at frame {snap_frame}.")

        # Also briefly disable/enable the solver shape to ensure Maya triggers a graph re-evaluation
        if cmds.attributeQuery('enable', node=solver_shape, exists=True):
            cmds.setKeyframe(f"{solver_shape}.enable", time=snap_frame - 1, value=1)
            cmds.setKeyframe(f"{solver_shape}.enable", time=snap_frame, value=0)
            cmds.setKeyframe(f"{solver_shape}.enable", time=snap_frame + 1, value=1)

    print(f"\n{'*'*50}")
    print(f"[PHYSICS] Inserted 'Hidden Snap' for Ziva at frame {snap_frame} (Keyed startFrame).")
    print(f"{'*'*50}\n")


def insert_smooth_snap_blendshape(snap_frame, buffer_frames=15):
    """
    Creates perfect linear blendshape morphs that mathematically ignore the underlying Ziva physics.
    It morphs strictly from the 'sagged' snapshots to the '15-frame equilibrium' poses for ALL soft tissues.
    """
    valid_meshes = [m for m in SOFT_TISSUE_MESHES if cmds.objExists(m)]
    print(f"[SMOOTH SNAP] Processing {len(valid_meshes)} valid soft-tissue meshes for anti-drift morphing...")

    # 1. Step to the "sagged" state BEFORE the snap (Strictly forward)
    step_timeline_forward(snap_frame - 1)
    
    for target_mesh in valid_meshes:
        sagged_name = f"{target_mesh}_sagged_snapshot_{snap_frame}"
        if cmds.objExists(sagged_name):
            cmds.delete(sagged_name)
        sagged_snapshot = cmds.duplicate(target_mesh, name=sagged_name)[0]
        if cmds.attributeQuery('visibility', node=sagged_snapshot, exists=True):
            cmds.setAttr(f"{sagged_snapshot}.visibility", 0)

    print(f"[PHYSICS] Flushing Ziva solver to clear drift...")
    insert_ziva_hidden_snap(snap_frame)

    # 2. Step Ziva forward through the 15 buffer frames so it settles into its CURRENT localized equilibrium
    print(f"[SMOOTH SNAP] Letting Ziva physics settle for {buffer_frames} frames to find local equilibrium...")
    for f in range(snap_frame, snap_frame + buffer_frames + 1):
        cmds.currentTime(f, edit=True)

    # 3. Take the NEW equilibrium snapshot at the settled frame!
    for target_mesh in valid_meshes:
        equil_name = f"{target_mesh}_equil_snapshot_{snap_frame}"
        if cmds.objExists(equil_name):
            cmds.delete(equil_name)
        equil_snapshot = cmds.duplicate(target_mesh, name=equil_name)[0]
        if cmds.attributeQuery('visibility', node=equil_snapshot, exists=True):
            cmds.setAttr(f"{equil_snapshot}.visibility", 0)

    # 4. Create or get Proxy Render Meshes and set up Blendshapes
    for target_mesh in valid_meshes:
        render_mesh = f"{target_mesh}_RENDER"
        bs_name = f"master_bs_{target_mesh}"
        
        if not cmds.objExists(render_mesh):
            render_mesh_node = cmds.duplicate(target_mesh, name=render_mesh)[0]
            
            # Hide the live Ziva mesh, show the render proxy!
            cmds.setAttr(f"{target_mesh}.visibility", 0)
            cmds.setAttr(f"{render_mesh_node}.visibility", 1)
            
            # Create master blendshape connecting Live Ziva to Proxy
            cmds.blendShape([target_mesh], render_mesh_node, name=bs_name, origin='local')
            # Default state: Follow live Ziva 100%
            cmds.setAttr(f"{bs_name}.{target_mesh}", 1.0)
            
        # 5. Add the Snapshots to the Master Blendshape
        sagged_name = f"{target_mesh}_sagged_snapshot_{snap_frame}"
        equil_name = f"{target_mesh}_equil_snapshot_{snap_frame}"
        
        existing_targets = cmds.listAttr(f"{bs_name}.w", multi=True) or []
        next_idx = len(existing_targets)
        
        cmds.blendShape(bs_name, edit=True, target=(render_mesh, next_idx, sagged_name, 1.0))
        sagged_attr = f"{bs_name}.{sagged_name}"
        
        cmds.blendShape(bs_name, edit=True, target=(render_mesh, next_idx+1, equil_name, 1.0))
        equil_attr = f"{bs_name}.{equil_name}"
        
        live_ziva_attr = f"{bs_name}.{target_mesh}"

        # 6. Animate the Proxy Morph (90% / 10% Smooth In)
        # At snap_frame - 1: Follow Live Ziva
        cmds.setKeyframe(live_ziva_attr, time=snap_frame - 1, value=1.0)
        cmds.setKeyframe(sagged_attr, time=snap_frame - 1, value=0.0)
        cmds.setKeyframe(equil_attr, time=snap_frame - 1, value=0.0)
        
        # At snap_frame: Disconnect Live Ziva (because it flushes to Neutral), Switch to Sagged!
        cmds.setKeyframe(live_ziva_attr, time=snap_frame, value=0.0)
        cmds.setKeyframe(sagged_attr, time=snap_frame, value=1.0)
        cmds.setKeyframe(equil_attr, time=snap_frame, value=0.0)
        
        # At snap_frame + buffer_frames (15): Reach 90% of the Equilibrium target
        cmds.setKeyframe(live_ziva_attr, time=snap_frame + buffer_frames, value=0.0)
        cmds.setKeyframe(sagged_attr, time=snap_frame + buffer_frames, value=0.1)
        cmds.setKeyframe(equil_attr, time=snap_frame + buffer_frames, value=0.9)

        # 15 frames AFTER buffer (frame 30): Morph remaining 10% smoothly back to live Ziva!
        smooth_in_end = snap_frame + buffer_frames + 15
        cmds.setKeyframe(live_ziva_attr, time=smooth_in_end, value=1.0)
        cmds.setKeyframe(sagged_attr, time=smooth_in_end, value=0.0)
        cmds.setKeyframe(equil_attr, time=smooth_in_end, value=0.0)

        # CRITICAL FIX: Maya's default Auto Tangents overshoot when curves go from 0->1->0.1.
        # This overshoot causes the weights to NOT sum to 1.0 between frames (like at frame 87).
        # When weights don't equal 1.0, the Neutral base mesh bleeds in, causing a massive snap!
        # Forcing STRICT LINEAR interpolation guarantees they always crossfade perfectly to 1.0.
        cmds.keyTangent(live_ziva_attr, inTangentType='linear', outTangentType='linear')
        cmds.keyTangent(sagged_attr, inTangentType='linear', outTangentType='linear')
        cmds.keyTangent(equil_attr, inTangentType='linear', outTangentType='linear')

    # 7. Manually calculate missing debug data for the buffer frames (for Orbicularis Oris only)
    print(f"[SMOOTH SNAP] Calculating plot data for buffer transition...")
    try:
        if "middleres_OrbicularisOris" in valid_meshes:
            sagged_y = cmds.xform(f"middleres_OrbicularisOris_sagged_snapshot_{snap_frame}.vtx[0]", query=True, translation=True, worldSpace=True)[1]
            equil_y = cmds.xform(f"middleres_OrbicularisOris_equil_snapshot_{snap_frame}.vtx[0]", query=True, translation=True, worldSpace=True)[1]
            
            global LIVE_DEBUG_FRAMES, LIVE_DEBUG_POSITIONS
            for f in range(snap_frame, snap_frame + buffer_frames + 1):
                progress = (f - snap_frame) / float(buffer_frames)
                w_sagged = 1.0 - (0.9 * progress)
                w_equil = 0.9 * progress
                
                y_pos = (sagged_y * w_sagged) + (equil_y * w_equil)
                LIVE_DEBUG_FRAMES.append(f)
                LIVE_DEBUG_POSITIONS.append(y_pos)
            
    except Exception as e:
        print(f"[DEBUG] Failed to compute manual plot points: {e}")

    print(f"[SMOOTH SNAP] Successfully injected perfect 90/10 proxy morphs on all soft tissues!")


def retrieve_landmark_positions(joint_groups_to_query, target_frame=14, is_first_transition=True):
    """
    Retrieve landmark positions at a specific frame.

    Args:
        joint_groups_to_query: Dictionary of joint groups and their landmark counts
        target_frame: Frame number to retrieve positions from (default 14 for neutral pose)
        is_first_transition: If True, iterate through frames 0-14 to ensure Ziva simulation proceeds frame-by-frame
    """
    # For the first transition, step through frames to ensure Ziva simulation proceeds correctly
    if is_first_transition:
        print(f"  [FIRST TRANSITION] Stepping through frames 0-{target_frame} for Ziva simulation...")
        step_timeline_forward(target_frame)
        print(f"  Reached target frame {target_frame}")
        
        # --- NEW: Capture Global Equilibrium Snapshot for Ziva Snaps ---
        target_mesh = "middleres_OrbicularisOris"
        equil_name = f"{target_mesh}_GLOBAL_EQUIL_SNAPSHOT"
        if cmds.objExists(target_mesh):
            if cmds.objExists(equil_name):
                cmds.delete(equil_name)
            equil_snapshot = cmds.duplicate(target_mesh, name=equil_name)[0]
            if cmds.attributeQuery('visibility', node=equil_snapshot, exists=True):
                cmds.setAttr(f"{equil_snapshot}.visibility", 0)
            print(f"  [SMOOTH SNAP] Captured global equilibrium snapshot at frame {target_frame}!")
            
    else:
        # For subsequent transitions, jump directly to the target frame
        print(f"  Retrieving landmark positions at frame {target_frame}")
        step_timeline_forward(target_frame)

    retrieved_positions = {}
    for joint_group_name, num_landmarks in joint_groups_to_query.items():
        landmark_locations = []
        print(f"  Retrieving landmarks for group: {joint_group_name}")
        for i in range(num_landmarks):
            joint_name = f'{joint_group_name}_{i + 1:02d}'
            if cmds.objExists(joint_name):
                location = cmds.xform(joint_name, query=True, translation=True, worldSpace=True)
                landmark_locations.append(location)
            else:
                print(f"  Warning: Joint '{joint_name}' not found.")
        retrieved_positions[joint_group_name] = landmark_locations
    return retrieved_positions


def retrieve_tissue_vertices(mesh_name='middleres_OrbicularisOris'):
    """
    Retrieve vertex positions for a tissue mesh (oor only for debugging).

    Args:
        mesh_name: Name of the mesh (default: 'oor')

    Returns:
        List of vertex coordinates [x1, y1, z1, x2, y2, z2, ...]
    """
    if not cmds.objExists(mesh_name):
        print(f"[DEBUG WARNING] Mesh '{mesh_name}' not found")
        return []

    try:
        # Get all vertex positions in world space
        all_vertex_positions = cmds.xform(f"{mesh_name}.vtx[*]",
                                         query=True,
                                         translation=True,
                                         worldSpace=True)
        return all_vertex_positions
    except Exception as e:
        print(f"[DEBUG ERROR] Failed to retrieve vertices for {mesh_name}: {e}")
        return []


def compute_landmark_distance(landmarks1, landmarks2):
    """
    Compute average Euclidean distance between two sets of landmarks.

    Args:
        landmarks1: Flattened list [x1, y1, z1, x2, y2, z2, ...]
        landmarks2: Flattened list [x1, y1, z1, x2, y2, z2, ...]

    Returns:
        tuple: (mean_distance, max_distance, total_error_magnitude)
    """
    if len(landmarks1) != len(landmarks2):
        print(f"[DEBUG WARNING] Landmark length mismatch: {len(landmarks1)} vs {len(landmarks2)}")
        return 0.0, 0.0, 0.0

    landmarks1 = np.array(landmarks1)
    landmarks2 = np.array(landmarks2)

    # Reshape to (N, 3) for easier computation
    n_landmarks = len(landmarks1) // 3
    lm1 = landmarks1.reshape(n_landmarks, 3)
    lm2 = landmarks2.reshape(n_landmarks, 3)

    # Compute per-landmark distance
    distances = np.linalg.norm(lm1 - lm2, axis=1)

    mean_dist = np.mean(distances)
    max_dist = np.max(distances)
    total_error = np.linalg.norm(landmarks1 - landmarks2)

    return mean_dist, max_dist, total_error


def compute_vertex_distance(vertices1, vertices2):
    """
    Compute average Euclidean distance between two sets of vertices.

    Args:
        vertices1: Flattened list [x1, y1, z1, x2, y2, z2, ...]
        vertices2: Flattened list [x1, y1, z1, x2, y2, z2, ...]

    Returns:
        tuple: (mean_distance, max_distance, total_error_magnitude)
    """
    if len(vertices1) != len(vertices2) or len(vertices1) == 0:
        print(f"[DEBUG WARNING] Vertex length mismatch or empty: {len(vertices1)} vs {len(vertices2)}")
        return 0.0, 0.0, 0.0

    vertices1 = np.array(vertices1)
    vertices2 = np.array(vertices2)

    # Reshape to (N, 3)
    n_vertices = len(vertices1) // 3
    v1 = vertices1.reshape(n_vertices, 3)
    v2 = vertices2.reshape(n_vertices, 3)

    # Compute per-vertex distance
    distances = np.linalg.norm(v1 - v2, axis=1)

    mean_dist = np.mean(distances)
    max_dist = np.max(distances)
    total_error = np.linalg.norm(vertices1 - vertices2)

    return mean_dist, max_dist, total_error


def save_debug_csv(debug_data, filename="debug_errors.csv"):
    """
    Save debug data to CSV file.

    Args:
        debug_data: List of dictionaries with debug information
        filename: Output CSV filename
    """
    if not debug_data:
        print("[DEBUG] No debug data to save")
        return

    csv_path = os.path.join(DEBUG_OUTPUT_DIR, filename)

    # Ensure directory exists
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

    try:
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['transition', 'phoneme_from', 'phoneme_to', 'frame',
                         'landmark_mean_hat_tilde', 'landmark_max_hat_tilde', 'landmark_total_hat_tilde',
                         'vertex_mean_hat_tilde', 'vertex_max_hat_tilde', 'vertex_total_hat_tilde',
                         'vertex_mean_hat_ground', 'vertex_max_hat_ground', 'vertex_total_hat_ground',
                         'error_magnitude', 'accumulated_error_magnitude',
                         'correction_magnitude', 'prev_error_magnitude', 'new_error_magnitude', 'residual_error_magnitude',
                         'correction_applied', 'dissipation_rate']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in debug_data:
                writer.writerow(row)

        print(f"[DEBUG] Saved debug data to {csv_path}")
    except Exception as e:
        print(f"[DEBUG ERROR] Failed to save CSV: {e}")


def visualize_errors(debug_data):
    """
    Create comprehensive visualization for error dissipation analysis.
    4 panels:
      1. Landmark error (mean distance) per transition
      2. Error decomposition: accumulated, new, correction, overlap offset
      3. Per-frame correction dissipation ramp
      4. Error flow: stacked view of how error is built and corrected

    Args:
        debug_data: List of dictionaries with debug information
    """
    if not debug_data:
        print("[DEBUG] No debug data to visualize")
        return

    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    import matplotlib.pyplot as plt

    # SIGGRAPH quality settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9

    # Ensure directory exists
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

    # Extract data for plotting
    transitions = []
    transition_labels = []
    for d in debug_data:
        transitions.append(d['transition'])
        pf = d['phoneme_from'].split('_')[-1]
        if pf == 'neutral': pf = 'AU0'
        pt = d['phoneme_to'].split('_')[-1]
        if pt == 'neutral': pt = 'AU0'
        transition_labels.append(f"{pf} to {pt}")

    unit_to_mm = 0.962
    landmark_mean = [d['landmark_mean_hat_tilde'] * unit_to_mm for d in debug_data]
    landmark_max = [d['landmark_max_hat_tilde'] * unit_to_mm for d in debug_data]
    error_mag = [d.get('error_magnitude', 0) for d in debug_data]
    acc_error_mag = [d.get('accumulated_error_magnitude', 0) for d in debug_data]
    correction_mag = [d.get('correction_magnitude', 0) for d in debug_data]
    prev_error_mag = [d.get('prev_error_magnitude', 0) for d in debug_data]
    new_error_mag = [d.get('new_error_magnitude', 0) for d in debug_data]
    overlap_corr_mag = [d.get('overlap_correction_magnitude', 0) for d in debug_data]

    # --- Create 4-panel figure ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=200)

    # ---- PANEL 1: Landmark Error (Mean & Max) ----
    ax1 = axes[0, 0]
    ax1.plot(transitions, landmark_mean, 'o-', color='#4A90D9', linewidth=2, markersize=7,
             label='Mean Landmark Error', zorder=3)
    ax1.fill_between(transitions, 0, landmark_mean, color='#4A90D9', alpha=0.15)
    ax1.plot(transitions, landmark_max, 's--', color='#E85D75', linewidth=1.5, markersize=5,
             label='Max Landmark Error', alpha=0.7)
    ax1.set_xlabel('Transition')
    ax1.set_ylabel('Distance (mm)')
    ax1.set_title('Landmark Error per Transition')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_xticks(transitions)
    ax1.set_xticklabels(transition_labels, rotation=45, ha='right')

    # ---- PANEL 2: Error Decomposition ----
    ax2 = axes[0, 1]
    bar_width = 0.18
    x = np.arange(len(transitions))

    ax2.bar(x - 1.5*bar_width, prev_error_mag, bar_width, label='Previous Accumulated ||e_acc||',
            color='#F5A623', edgecolor='#333', linewidth=0.5, alpha=0.85)
    ax2.bar(x - 0.5*bar_width, new_error_mag, bar_width, label='New Error ||e_new||',
            color='#E85D75', edgecolor='#333', linewidth=0.5, alpha=0.85)
    ax2.bar(x + 0.5*bar_width, correction_mag, bar_width, label='Correction Applied ||c||',
            color='#50C878', edgecolor='#333', linewidth=0.5, alpha=0.85)
    ax2.bar(x + 1.5*bar_width, overlap_corr_mag, bar_width, label='Overlap Offset ||delta||',
            color='#9B59B6', edgecolor='#333', linewidth=0.5, alpha=0.85)

    # Add value labels on bars
    for bars_x, vals in [(x - 1.5*bar_width, prev_error_mag),
                          (x - 0.5*bar_width, new_error_mag),
                          (x + 0.5*bar_width, correction_mag),
                          (x + 1.5*bar_width, overlap_corr_mag)]:
        for bx, val in zip(bars_x, vals):
            if val > 0:
                ax2.text(bx, val + max(max(prev_error_mag + new_error_mag, default=1), 1) * 0.02,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    ax2.set_xlabel('Transition')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Error Decomposition per Transition')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_xticks(x)
    ax2.set_xticklabels(transition_labels, rotation=45, ha='right')

    # ---- PANEL 3: Per-Frame Correction Dissipation ----
    ax3 = axes[1, 0]
    if 'PER_FRAME_DEBUG_HISTORY' in globals() and PER_FRAME_DEBUG_HISTORY:
        global_frame_counter = 0
        colors_cycle = ['#4A90D9', '#E85D75', '#50C878', '#F5A623', '#9B59B6',
                        '#E67E22', '#1ABC9C', '#C0392B']

        for idx, entry in enumerate(PER_FRAME_DEBUG_HISTORY):
            data = entry['data']
            label = entry['label']
            frames = range(global_frame_counter, global_frame_counter + len(data))
            color = colors_cycle[idx % len(colors_cycle)]

            ax3.plot(frames, data, label=label, linewidth=1.5, color=color)
            ax3.fill_between(frames, 0, data, alpha=0.1, color=color)

            # Mark transition boundary
            if global_frame_counter > 0:
                ax3.axvline(x=global_frame_counter, color='gray', linestyle=':', alpha=0.5)

            global_frame_counter += len(data)

        ax3.set_xlabel('Maya Frame')
        ax3.set_ylabel('Correction Magnitude')
        ax3.set_title('Per-Frame Correction Dissipation (Linear Ramp)')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.2)
    else:
        ax3.text(0.5, 0.5, 'No per-frame data collected\n(PER_FRAME_DEBUG_HISTORY empty)',
                ha='center', va='center', fontsize=12, color='gray', transform=ax3.transAxes)
        ax3.set_title('Per-Frame Correction Dissipation')

    # ---- PANEL 4: Error Flow (cumulative view) ----
    ax4 = axes[1, 1]
    # Show how accumulated error evolves: starts at 0, each transition adds new_error
    # and subtracts correction
    cum_error = [0]
    for i, d in enumerate(debug_data):
        prev = cum_error[-1]
        new_e = d.get('new_error_magnitude', 0)
        corr = d.get('correction_magnitude', 0)
        # After transition: residual = prev*(1-rate) + new_error
        rate = d.get('dissipation_rate', 1.0)
        residual = prev * (1 - rate) + new_e
        cum_error.append(residual)

    ax4.plot(range(len(cum_error)), cum_error, 'o-', color='#E85D75', linewidth=2,
             markersize=6, label='Accumulated Error ||e_acc||', zorder=3)
    ax4.fill_between(range(len(cum_error)), 0, cum_error, color='#E85D75', alpha=0.1)

    # Annotate steady state
    if len(cum_error) > 3:
        ss = cum_error[-1]
        ax4.axhline(y=ss, color='gray', linestyle='--', alpha=0.5)
        ax4.text(len(cum_error) - 1, ss * 1.05, f'e_inf ~ {ss:.1f}', fontsize=9, color='gray')

    x_labels_flow = ['Start'] + transition_labels
    ax4.set_xticks(range(len(cum_error)))
    ax4.set_xticklabels(x_labels_flow, rotation=45, ha='right')
    ax4.set_xlabel('Transition')
    ax4.set_ylabel('||e_acc|| (Accumulated Error)')
    ax4.set_title(f'Error Evolution (gamma = {debug_data[0].get("dissipation_rate", "?")})')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.2)

    plt.suptitle('Feedback Control: Error Dissipation Analysis',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    # Save figure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(DEBUG_OUTPUT_DIR, f"error_analysis_{timestamp}.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"[DEBUG] Saved visualization to {plot_path}")

    plt.close('all')


def load_ideal_target_positions(target_phoneme):
    """
    Load the ideal target landmark positions from precomputed neutral-to-target deformation.

    This loads the "ground truth" positions (B_tilde, C_tilde, etc.) that we would reach
    if we started from a perfect neutral pose and applied the deformation.

    Args:
        target_phoneme: The target phoneme (e.g., "dataV22_phoneme_Y", "dataV22_phoneme_UW")

    Returns:
        Dictionary with flattened ideal positions for each joint group, or None if not found
        {
            'x_joint_conca': [...],  # oor landmarks (900 values = 300 landmarks * 3)
            'x_joint_rotation_conca': [...],  # jaw rotation (3 values)
            'x_ooc_l_joint_conca': [...],  # ooc_l landmarks (900 values)
            'x_ooc_r_joint_conca': [...],  # ooc_r landmarks (900 values)
            'x_frontalis_joint_conca': [...]  # frontalis landmarks (900 values)
        }
    """
    print(f"\n[FEEDBACK CONTROL] Computing ideal target positions for {target_phoneme}")

    # Build the key for looking up the neutral-to-target path
    neutral_to_target_key = f"dataV22_AU0_neutral_to_{target_phoneme}"

    # Try different key formats
    db_key_map = {normalize_task_key(k): k for k in DEFORMATION_PATH_DB}
    normalized_key = normalize_task_key(neutral_to_target_key)
    original_key = db_key_map.get(normalized_key)

    if not original_key:
        # Try without the extra "to" connector
        neutral_to_target_key = f"dataV22_AU0_neutral to {target_phoneme}"
        normalized_key = normalize_task_key(neutral_to_target_key)
        original_key = db_key_map.get(normalized_key)

    if not original_key:
        print(f"[FEEDBACK CONTROL WARNING] Could not find neutral-to-{target_phoneme} in database")
        print(f"  Tried keys: {neutral_to_target_key}")
        return None

    # Get the path to the precomputed deformation folder
    deformation_path = DEFORMATION_PATH_DB[original_key]

    # Load combined_checkpoints_base{base_frames_global}.pkl (contains accumulated deformation parameters)
    checkpoints_file = os.path.join(deformation_path, f"combined_checkpoints_base{base_frames_global}.pkl")
    
    if not os.path.exists(checkpoints_file):
        print(f"[FEEDBACK CONTROL WARNING] combined_checkpoints_base{base_frames_global}.pkl not found at {checkpoints_file}")
        return None

    try:
        with open(checkpoints_file, 'rb') as f:
            checkpoints = pickle.load(f)

        print(f"[FEEDBACK CONTROL] Loaded {len(checkpoints)} checkpoints from {checkpoints_file}")

        # Get the LAST checkpoint (final accumulated deformation)
        final_checkpoint = checkpoints[-1]

        # Expected dimensions
        n = 300 * 3  # oor landmarks
        joint_rotate_num = 1
        joint_ooc_l_num = 300
        joint_ooc_r_num = 300
        joint_frontalis_num = 300

        # Parse the final checkpoint
        delta_b_final = np.array(final_checkpoint)



        # OOR landmarks (300 landmarks = 900 coordinates)
        x_joint_first_wave = getattr(joint_position_first_wave, "x_joint_position_1")
        x_joint_second_wave = getattr(joint_position, "x_joint_position_1")
        x_joint_third_wave = getattr(joint_position_third_wave, "x_joint_position_1")
        x_joint_combined = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave
        # Flatten: [[x1,y1,z1], [x2,y2,z2], ...] -> [x1,y1,z1,x2,y2,z2,...]
        x_neutral = [item for sublist in x_joint_combined for item in sublist]

        # OOC_L landmarks (300 landmarks = 900 coordinates)
        x_joint_first_wave = getattr(joint_position_ooc_l_base_0, "x_joint_position_1")
        x_joint_second_wave = getattr(joint_position_ooc_l_base_pose11_0, "x_joint_position_1")
        x_joint_third_wave = getattr(joint_position_ooc_l_base_pose11_AU1_0, "x_joint_position_1")
        x_joint_fourth_wave = getattr(joint_position_ooc_l_base_pose11_AU1_AU2_0, "x_joint_position_1")
        x_joint_combined = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave + x_joint_fourth_wave
        # Flatten
        x_neutral_ooc_l = [item for sublist in x_joint_combined for item in sublist]

        # OOC_R landmarks (300 landmarks = 900 coordinates)
        x_joint_first_wave = getattr(joint_position_ooc_r_base_0, "x_joint_position_1")
        x_joint_second_wave = getattr(joint_position_ooc_r_base_pose11_0, "x_joint_position_1")
        x_joint_third_wave = getattr(joint_position_ooc_r_base_pose11_AU1_0, "x_joint_position_1")
        x_joint_fourth_wave = getattr(joint_position_ooc_r_base_pose11_AU1_AU2_0, "x_joint_position_1")
        x_joint_combined = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave + x_joint_fourth_wave
        # Flatten
        x_neutral_ooc_r = [item for sublist in x_joint_combined for item in sublist]

        # Frontalis landmarks (300 landmarks = 900 coordinates)
        x_joint_first_wave = getattr(joint_position_frontalis_base_0, "x_joint_position_1")
        x_joint_second_wave = getattr(joint_position_frontalis_base_AU1_0, "x_joint_position_1")
        x_joint_third_wave = getattr(joint_position_frontalis_base_AU1_AU2_0, "x_joint_position_1")
        x_joint_combined = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave
        # Flatten
        x_neutral_frontalis = [item for sublist in x_joint_combined for item in sublist]

        print(f"[FEEDBACK CONTROL] Loaded TRUE neutral positions from joint_position modules:")
        print(f"  OOR: {len(x_neutral)} coords, OOC_L: {len(x_neutral_ooc_l)} coords")
        print(f"  OOC_R: {len(x_neutral_ooc_r)} coords, Frontalis: {len(x_neutral_frontalis)} coords")

        # Compute ideal positions: x_ideal = x_neutral + delta_b
        x_ideal = np.array(x_neutral) + delta_b_final[0:n]
        x_ideal_rotation = np.array([0, 0, 0]) + delta_b_final[n:n + 3*joint_rotate_num]

        slice_start = n + 3*joint_rotate_num
        slice_end = slice_start + 3*joint_ooc_l_num
        x_ideal_ooc_l = np.array(x_neutral_ooc_l) + delta_b_final[slice_start:slice_end]

        slice_start = slice_end
        slice_end = slice_start + 3*joint_ooc_r_num
        x_ideal_ooc_r = np.array(x_neutral_ooc_r) + delta_b_final[slice_start:slice_end]

        slice_start = slice_end
        slice_end = slice_start + 3*joint_frontalis_num
        x_ideal_frontalis = np.array(x_neutral_frontalis) + delta_b_final[slice_start:slice_end]

        ideal_positions = {
            'x_joint_conca': x_ideal.tolist(),
            'x_joint_rotation_conca': x_ideal_rotation.tolist(),
            'x_ooc_l_joint_conca': x_ideal_ooc_l.tolist(),
            'x_ooc_r_joint_conca': x_ideal_ooc_r.tolist(),
            'x_frontalis_joint_conca': x_ideal_frontalis.tolist()
        }

        print(f"[FEEDBACK CONTROL] Computed ideal positions: {len(ideal_positions['x_joint_conca'])} oor coordinates")
        return ideal_positions

    except Exception as e:
        print(f"[FEEDBACK CONTROL ERROR] Failed to compute ideal positions: {e}")
        import traceback
        traceback.print_exc()
        return None

def initial_joint_position(target_frame=14, is_first_transition=True):
    """
    Gets initial joint positions for the deformation at a specific frame.

    Args:
        target_frame: Frame number to retrieve positions from (default 14 for neutral pose)
        is_first_transition: If True, iterate through frames 0-14 to ensure Ziva simulation proceeds frame-by-frame

    Uses global variables:
    - NEUTRAL_TO_OTHER_POSE_FLAG: 1 if starting from neutral, 0 if starting from another pose
    - PREVIOUS_POSE_FOLDER: Path to previous pose data (used when NEUTRAL_TO_OTHER_POSE_FLAG = 0)
    """
    print(f"--- Getting Initial Joint Positions at frame {target_frame} ---")
    joint_groups_to_query = {
        'oor': 300, 'ooc_l': 300,
        'ooc_r': 300, 'frontalis': 300
    }
    retrieved_positions = retrieve_landmark_positions(joint_groups_to_query, target_frame, is_first_transition)

    # Determine jaw rotation based on whether starting from neutral or another pose
    if NEUTRAL_TO_OTHER_POSE_FLAG == 0:
        # Starting from non-neutral pose - load previous pose rotation data
        print(f"  Loading previous pose data from: {PREVIOUS_POSE_FOLDER}")
        try:
            with open(PREVIOUS_POSE_FOLDER + "/deltaB_temp.pkl", 'rb') as f:
                data_t = pickle.load(f)
            x_joint_rotation = data_t['x_joint_rotation_']
            print(f"  Loaded x_joint_rotation from previous pose")
        except Exception as e:
            print(f"  [WARNING] Could not load previous pose data: {e}")
            print(f"  Using neutral pose rotation instead")
            x_joint_rotation = [[0, 0, 0]]
    else:
        # Starting from neutral pose - use neutral rotation
        print(f"  Using neutral pose rotation")
        # For neutral pose, jaw rotation is simply [0, 0, 0]
        x_joint_rotation = [[0, 0, 0]]

    # Flatten retrieved landmark positions
    x_joint_conca = [item for sublist in retrieved_positions.get('oor', []) for item in sublist]
    x_ooc_l_joint_conca = [item for sublist in retrieved_positions.get('ooc_l', []) for item in sublist]
    x_ooc_r_joint_conca = [item for sublist in retrieved_positions.get('ooc_r', []) for item in sublist]
    x_frontalis_joint_conca = [item for sublist in retrieved_positions.get('frontalis', []) for item in sublist]

    # Flatten x_joint_rotation to concatenated form
    x_joint_rotation_conca = [item for sublist in x_joint_rotation for item in sublist]

    print(f"  Retrieved {len(x_joint_conca)//3} oor landmarks")
    print(f"  Jaw rotation: {x_joint_rotation_conca}")

    return x_joint_conca, x_joint_rotation_conca, x_ooc_l_joint_conca, x_ooc_r_joint_conca, x_frontalis_joint_conca

def compute_position_error(actual_positions, ideal_positions):
    """
    Compute the error between actual and ideal landmark positions.

    Args:
        actual_positions: Dictionary with actual positions (same format as ideal_positions)
        ideal_positions: Dictionary with ideal positions from neutral-to-target deformation

    Returns:
        Dictionary with position errors for each joint group
    """
    if ideal_positions is None:
        return None

    error = {}
    for key in ['x_joint_conca', 'x_joint_rotation_conca', 'x_ooc_l_joint_conca',
                'x_ooc_r_joint_conca', 'x_frontalis_joint_conca']:
        actual = np.array(actual_positions.get(key, []))
        ideal = np.array(ideal_positions.get(key, []))

        if len(actual) == 0 or len(ideal) == 0 or len(actual) != len(ideal):
            error[key] = np.zeros_like(actual) if len(actual) > 0 else np.array([])
        else:
            error[key] = ideal - actual

    return error


def apply_dissipating_error(positions, accumulated_error, dissipation_rate):
    """
    Apply the dissipating error to the positions.

    Args:
        positions: Dictionary with current positions
        accumulated_error: Dictionary with accumulated error from previous transition
        dissipation_rate: How much of the error to apply (0.0 to 1.0)

    Returns:
        Updated positions with error correction applied
    """
    if accumulated_error is None:
        return positions

    corrected = {}
    for key in positions.keys():
        pos = np.array(positions[key])
        err = np.array(accumulated_error.get(key, []))

        if len(err) == 0 or len(pos) != len(err):
            corrected[key] = pos
        else:
            # Apply a fraction of the error as correction
            correction = err * dissipation_rate
            corrected[key] = pos + correction

    return corrected


def beta_function(x_joint_conca, x_joint_rotation_conca, au22_x_joint_conca3, au22_x_joint_rotation_conca3, au22_x_ooc_l_joint_conca3, au22_x_ooc_r_joint_conca3, au22_x_frontalis_joint_conca3, start_frame, keyframe_max):
    time_temp = str(keyframe_max)
    cmds.currentTime(time_temp, edit=True)

    x_joint_init_conca = x_joint_conca
    x_joint_rotation_init_conca = x_joint_rotation_conca

    x_joint_ = au22_x_joint_conca3
    x_joint_rotation_ = au22_x_joint_rotation_conca3
    x_ooc_l_joint_ = au22_x_ooc_l_joint_conca3
    x_ooc_r_joint_ = au22_x_ooc_r_joint_conca3
    x_frontalis_joint_ = au22_x_frontalis_joint_conca3

    relative_rotation_value = np.array(x_joint_rotation_) - np.array(x_joint_rotation_init_conca)
    cmds.rotate(relative_rotation_value[0], relative_rotation_value[1], relative_rotation_value[2], 'jaw01', relative=True, ws=True)
    cmds.setKeyframe('jaw01', time=time_temp)

    joint_num = 300
    for i_temp2 in range(joint_num):
        joint_name = f'oor_{i_temp2 + 1:02d}'
        cmds.move(x_joint_[3*i_temp2], x_joint_[3*i_temp2+1], x_joint_[3*i_temp2+2], joint_name, absolute=True)
        cmds.setKeyframe(joint_name, time=time_temp)

    joint_ooc_l_num = 300
    for i_temp2 in range(joint_ooc_l_num):
        joint_name = f'ooc_l_{i_temp2 + 1:02d}'
        cmds.move(x_ooc_l_joint_[3*i_temp2], x_ooc_l_joint_[3*i_temp2+1], x_ooc_l_joint_[3*i_temp2+2], joint_name, absolute=True)
        cmds.setKeyframe(joint_name, time=time_temp)

    joint_ooc_r_num = 300
    for i_temp2 in range(joint_ooc_r_num):
        joint_name = f'ooc_r_{i_temp2 + 1:02d}'
        cmds.move(x_ooc_r_joint_[3*i_temp2], x_ooc_r_joint_[3*i_temp2+1], x_ooc_r_joint_[3*i_temp2+2], joint_name, absolute=True)
        cmds.setKeyframe(joint_name, time=time_temp)
        
    joint_frontalis_num = 300
    for i_temp2 in range(joint_frontalis_num):
        joint_name = f'frontalis_{i_temp2 + 1:02d}'
        cmds.move(x_frontalis_joint_[3*i_temp2], x_frontalis_joint_[3*i_temp2+1], x_frontalis_joint_[3*i_temp2+2], joint_name, absolute=True)
        cmds.setKeyframe(joint_name, time=time_temp)

def mayaSave2file(save_path):
    print(f"Saving Maya file to: {save_path}")
    cmds.file(rename=save_path)
    cmds.file(save=True, force=True, type='mayaBinary')

def perform_the_deformation(source_path, start_frame_offset=14, continue_from_last=False, correction_vectors=None):
    """
    Apply deformation to Maya rig with Linear Error Dissipation.
    
    Args:
        source_path: Direct path to the folder containing combined_checkpoints_base{base_frames_global}.pkl
        correction_vectors: Dict containing total correction needed for each joint group.
                            { 'x_joint_conca': np.array([...]), ... }
    """
    print("--- Starting Deformation Application (With Linear Error Dissipation) ---")
    
    # Use source_path directly (no more SEQUENCE_DB_PATH)
    checkpoints_path = os.path.join(source_path, f"combined_checkpoints_base{base_frames_global}.pkl")
    if not os.path.exists(checkpoints_path):
        raise RuntimeError(f"Deformation file not found! {checkpoints_path}")

    with open(checkpoints_path, "rb") as file:
        combined_checkpoints = pickle.load(file)

    # Dimensions
    n = 300 * 3
    joint_rotate_num = 1
    joint_ooc_l_num = 300
    joint_ooc_r_num = 300
    joint_frontalis_num = 300
    expected_dimension = n + (3*joint_rotate_num) + (3*joint_ooc_l_num) + (3*joint_ooc_r_num) + (3*joint_frontalis_num)

    # Initialize positions
    if not continue_from_last:
        x_joint, x_joint_rotation, x_ooc_l_joint, x_ooc_r_joint, x_frontalis_joint = initial_joint_position(start_frame_offset, is_first_transition=True)
        # Set initial keyframe
        cmds.currentTime(start_frame_offset, edit=True)
        # ... (Set keyframes code omitted for brevity, same as before) ...
    else:
        x_joint, x_joint_rotation, x_ooc_l_joint, x_ooc_r_joint, x_frontalis_joint = initial_joint_position(start_frame_offset, is_first_transition=False)

    x_joint_init = np.array(x_joint)
    x_joint_rotation_init = np.array(x_joint_rotation)
    x_ooc_l_joint_init = np.array(x_ooc_l_joint)
    x_ooc_r_joint_init = np.array(x_ooc_r_joint)
    x_frontalis_joint_init = np.array(x_frontalis_joint)

    # Data collection for per-frame plotting
    per_frame_debug_data = [] 
    total_steps = len(combined_checkpoints)

    # Load overlap metadata if it exists
    metadata_path = os.path.join(source_path, "overlap_metadata.pkl")
    frame_boundaries = []
    phonemes = []
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            frame_boundaries = metadata.get('frame_boundaries', [])
            phonemes = metadata.get('phonemes', [])

    # --- THE LOOP ---
    for i, delta_b0_conca_np in enumerate(combined_checkpoints):
        frame = start_frame_offset + 1 + i
        delta_b0_conca = np.array(delta_b0_conca_np)

        # Padding logic...
        if len(delta_b0_conca) < expected_dimension:
            padding = np.zeros(expected_dimension - len(delta_b0_conca))
            delta_b0_conca = np.concatenate((delta_b0_conca, padding))

        # 1. Calculate Standard Physics Deformation (Open Loop)
        x_joint_ = x_joint_init + delta_b0_conca[0:n]
        x_joint_rotation_ = x_joint_rotation_init + delta_b0_conca[n:n + 3 * joint_rotate_num]
        
        slice_start = n + 3 * joint_rotate_num
        slice_end = slice_start + 3 * joint_ooc_l_num
        x_ooc_l_joint_ = x_ooc_l_joint_init + delta_b0_conca[slice_start:slice_end]

        slice_start = slice_end
        slice_end = slice_start + 3 * joint_ooc_r_num
        x_ooc_r_joint_ = x_ooc_r_joint_init + delta_b0_conca[slice_start:slice_end]

        slice_start = slice_end
        slice_end = slice_start + 3 * joint_frontalis_num
        x_frontalis_joint_ = x_frontalis_joint_init + delta_b0_conca[slice_start:slice_end]

        # 2. Apply Distributed Error Correction (Feedback Loop)
        current_correction_mag = 0.0
        
        if correction_vectors is not None:
            # Calculate interpolation weight scaling specifically for the current phase!
            current_phase_start = 0
            current_phase_end = total_steps
            if frame_boundaries:
                past_bounds = [b for b in frame_boundaries if b <= i]
                current_phase_start = past_bounds[-1] if past_bounds else 0
                future_bounds = [b for b in frame_boundaries if b > i]
                current_phase_end = future_bounds[0] if future_bounds else total_steps

            phase_len = current_phase_end - current_phase_start
            if phase_len == 0: phase_len = 1
            progress = (i - current_phase_start + 1) / float(phase_len)
            
            # Apply portion of the error
            if 'x_joint_conca' in correction_vectors:
                correction = correction_vectors['x_joint_conca'] * progress
                x_joint_ = x_joint_ + correction
                current_correction_mag = np.linalg.norm(correction)

            if 'x_ooc_l_joint_conca' in correction_vectors:
                x_ooc_l_joint_ = x_ooc_l_joint_ + (correction_vectors['x_ooc_l_joint_conca'] * progress)
            
            if 'x_ooc_r_joint_conca' in correction_vectors:
                x_ooc_r_joint_ = x_ooc_r_joint_ + (correction_vectors['x_ooc_r_joint_conca'] * progress)

            if 'x_frontalis_joint_conca' in correction_vectors:
                x_frontalis_joint_ = x_frontalis_joint_ + (correction_vectors['x_frontalis_joint_conca'] * progress)

        # Collect per-frame data
        per_frame_debug_data.append(current_correction_mag)

        # Pass corrected positions to beta_function (Maya Keyframing)
        prev_x_joint = x_joint_init if i == 0 else x_joint
        prev_x_joint_rotation = x_joint_rotation_init if i == 0 else x_joint_rotation
        
        beta_function(
            prev_x_joint, prev_x_joint_rotation, x_joint_, x_joint_rotation_,
            x_ooc_l_joint_, x_ooc_r_joint_, x_frontalis_joint_,
            frame-1, frame
        )
        
        # --- COLLECT LIVE POSITION FOR DEBUG PLOTTING ---
        try:
            live_mesh = "middleres_OrbicularisOris_RENDER" if cmds.objExists("middleres_OrbicularisOris_RENDER") else "middleres_OrbicularisOris"
            pos = cmds.xform(f"{live_mesh}.vtx[0]", query=True, translation=True, worldSpace=True)
            LIVE_DEBUG_FRAMES.append(frame)
            LIVE_DEBUG_POSITIONS.append(pos[1])
        except Exception:
            pass
        
        # Update state
        x_joint = x_joint_
        x_joint_rotation = x_joint_rotation_

        # 4. Compute Dynamic Feedback Correction for NEXT segment
        if ENABLE_FEEDBACK_CONTROL and frame_boundaries and phonemes and i in frame_boundaries:
            phase_idx = frame_boundaries.index(i)
            # Find the phoneme target we just arrived at
            if phase_idx + 1 < len(phonemes):
                current_phoneme = phonemes[phase_idx + 1]
                print(f"[ACTIVE FEEDBACK] Hit phase boundary for {current_phoneme}. Calculating intermediate correction...")
                
                joint_groups = {'oor': 300, 'ooc_l': 300, 'ooc_r': 300, 'frontalis': 300}
                actual_positions = retrieve_landmark_positions(joint_groups, frame, is_first_transition=False)
                actual_flat = {
                    'x_joint_conca': [item for sublist in actual_positions.get('oor', []) for item in sublist],
                    'x_joint_rotation_conca': [0, 0, 0],
                    'x_ooc_l_joint_conca': [item for sublist in actual_positions.get('ooc_l', []) for item in sublist],
                    'x_ooc_r_joint_conca': [item for sublist in actual_positions.get('ooc_r', []) for item in sublist],
                    'x_frontalis_joint_conca': [item for sublist in actual_positions.get('frontalis', []) for item in sublist]
                }
                
                ideal_positions = load_ideal_target_positions(current_phoneme)
                if ideal_positions is not None:
                    new_error = compute_position_error(actual_flat, ideal_positions)
                    if correction_vectors is None:
                        correction_vectors = {}
                    for key in new_error.keys():
                        err = np.array(new_error[key])
                        if len(err) > 0:
                            correction_vectors[key] = err * ERROR_DISSIPATION_RATE
                    print(f"   -> Enqueued correction vector magnitude: {np.linalg.norm(correction_vectors.get('x_joint_conca', [0])):.6f}")


    # Finalize
    final_frame = start_frame_offset + len(combined_checkpoints)
    playback_end = final_frame + 36
    cmds.playbackOptions(minTime=1, maxTime=playback_end)
    
    return final_frame, per_frame_debug_data


def run_deformation_processing(phonemes_list, log_paths_list, sequence_folder_name):
    """
    Runs the full deformation parameter processing pipeline.
    """
    print("--- Starting Deformation Processing ---")
    print(f"Phonemes: {phonemes_list}")
    print(f"Log Paths: {log_paths_list}")
    print(f"Output Folder: {sequence_folder_name}")

    # --- Your script starts here ---
    
    overlap_ratio = 0.4 # 0.1, 0.4

    # Configuration for the 3 separate paths
    # (THIS IS NOW AN ARGUMENT)
    log_paths = log_paths_list

    # 1. Separate the folder name
    base_path = r"\\bbysvm-nfs1.its.sfu.ca\GrUViLab\Data\richardPeng\SFU_PhD_cloud\2-Projects\1-physically based facial animation\3-Experiments\a3-stochastic-V1-backup-and-results\results\sequence_combined_deformation_parameters_hub"
    # (THIS IS NOW AN ARGUMENT)
    folder_name = sequence_folder_name

    # 2. Create the full path
    log_path_output = os.path.join(base_path, folder_name)

    # 3. Create the folder if it doesn't exist
    if not os.path.exists(log_path_output):
        os.makedirs(log_path_output)
        print(f"Created directory: {log_path_output}")
    else:
        print(f"Directory already exists: {log_path_output}")

    def get_total_steps_from_separate_logs():
        """Extracts step counts from log files"""
        total_steps_convergence = [0]  # Starting with 0 for the initial phase
        
        for i, log_path in enumerate(log_paths):
            file_path = os.path.join(log_path, "log0.txt")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                    match = re.search(r'done at No\. (\d+) step', content)
                    if match:
                        total_steps_convergence.append(int(match.group(1)))
                        print(f"Log {i+1}: Found {match.group(1)} steps")
            else:
                print(f"File not found: {file_path}")
        
        print("total_steps_convergence =", total_steps_convergence)
        return total_steps_convergence

    # Get step counts from separate logs
    total_steps_convergence = get_total_steps_from_separate_logs()

    # Calculate accumulated phase boundaries
    phase_boundaries = np.cumsum(total_steps_convergence).tolist()
    phase_boundaries_old = phase_boundaries.copy()

    print(f"Phase boundaries: {phase_boundaries}")

    # (THIS IS NOW AN ARGUMENT)
    phonemes = phonemes_list
    total_duration = 1.68 # You can also make this an argument if needed

    # Define phoneme-specific duration mappings based on your reference
    # Reference: "not even you" -> ["Neutral", "N", "AA", "T", "IY", "V", "N", "Y", "UW"]
    # timestamps = [0, 0.23, 0.52, 0.67, 0.89, 0.98, 1.14, 1.16, 1.68]
    # This means: Neutral->N: 0.23, N->AA: 0.29, AA->T: 0.15, T->IY: 0.22, IY->V: 0.09, V->N: 0.16, N->Y: 0.02, Y->UW: 0.52

    # You can customize these durations per phoneme transition
    # For now, we'll use equal intervals but you can modify this dictionary
    phoneme_duration_map = {
        # Add custom durations here if needed
        # Format: ("phoneme_from", "phoneme_to"): duration
    }

    # generate equal interval's timestemps (or custom if you define mappings above)
    if len(phonemes) <= 1:
        print("Error: Need at least 2 phonemes (a start and an end) to generate timestamps.")
        return # Exit if not enough phonemes

    # For "You've not eaten it yet", using natural female speech timing
    # Phoneme breakdown: Neutral -> Y -> UW -> V -> N -> AA -> T -> IY -> T -> N -> IY -> T -> Y -> EH -> T
    # Word breakdown: "You've" (Y,UW,V) | "not" (N,AA,T) | "eaten" (IY,T,N) | "it" (IY,T) | "yet" (Y,EH,T)
    if len(phonemes) == 15:  # Specific case for "You've not eaten it yet" (14 phonemes + Neutral = 15 total)
        # Natural female speech timing - total duration = 1.68
        # You've: 0.0-0.35 (0.35s), not: 0.35-0.68 (0.33s), eaten: 0.68-1.10 (0.42s), it: 1.10-1.35 (0.25s), yet: 1.35-1.68 (0.33s)
        timestamps = [
            0.00,   # Neutral (start)
            0.08,   # Y (you)
            0.22,   # UW (you've)
            0.35,   # V (you've)
            0.45,   # N (not)
            0.58,   # AA (not)
            0.68,   # T (not)
            0.82,   # IY (ea-)
            0.95,   # T (-ten)
            1.10,   # N (eaten)
            1.20,   # IY (i-)
            1.35,   # T (it)
            1.43,   # Y (ye-)
            1.58,   # EH (-et)
            1.68    # T (yet)
        ]
    else:
        # Default: equal intervals
        timestamps = [i * total_duration / (len(phonemes) - 1) for i in range(len(phonemes))]

    print(f"expressions: {phonemes}")
    print(f"Timestamps: {[round(t, 2) for t in timestamps]}")

    def adjust_durations_with_overlap(timestamps, overlap_ratio):
        if len(timestamps) < 2:
            return timestamps, []
        
        original_durations = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if len(original_durations) < 2: # Only one duration, no overlap possible
            return timestamps, original_durations

        adjusted_durations = []
        for i in range(len(original_durations)-1):
            adjusted_duration = original_durations[i] - original_durations[i+1] * overlap_ratio
            adjusted_durations.append(max(0, adjusted_duration))
        
        adjusted_durations.append(original_durations[-1])
        
        adjusted_timestamps = [timestamps[0]]
        for duration in adjusted_durations:
            adjusted_timestamps.append(adjusted_timestamps[-1] + duration)
        
        return adjusted_timestamps, adjusted_durations

    # Apply the duration adjustment
    adjusted_timestamps, adjusted_durations = adjust_durations_with_overlap(timestamps, overlap_ratio)
    print("Original timestamps:", timestamps)
    print("Adjusted timestamps:", adjusted_timestamps)

    total_duration = timestamps[-1]

    # Calculate new phase durations and weights
    if not adjusted_timestamps or len(adjusted_timestamps) < 2:
        print("Error: Could not calculate adjusted timestamps.")
        return

    adjusted_phase_durations = [adjusted_timestamps[i+1] - adjusted_timestamps[i] for i in range(len(adjusted_timestamps)-1)]
    
    # Avoid division by zero if total_duration is 0
    if total_duration == 0:
        print("Warning: Total duration is 0. Setting phase weights equally.")
        adjusted_phase_weights = [1.0 / len(adjusted_phase_durations)] * len(adjusted_phase_durations) if adjusted_phase_durations else []
    else:
        adjusted_phase_weights = [duration/total_duration for duration in adjusted_phase_durations]

    phase_durations = adjusted_phase_durations
    phase_weights = adjusted_phase_weights

    # Must-include indices are exactly the accumulated phase boundaries
    must_include_indices = phase_boundaries.copy()
    must_include_indices_old = must_include_indices.copy()

    # Time-preserving overlap: keep the same number of checkpoints (and thus
    # the same number of animation frames) regardless of overlap_ratio.
    # OLD (duration-shrinking):
    #   time_reduced_percentage = (timestamps[-1] - adjusted_timestamps[-1])/timestamps[-1] if timestamps[-1] > 0 else 0
    #   num_checkpoints = math.floor(36 * (1 - time_reduced_percentage))
    num_checkpoints = 36

    buffer_ratio = 0.001

    # Adaptive phase ranges based on phonemes length
    phase_ranges = []
    if len(phase_boundaries) != len(phonemes):
        print(f"Error: Mismatch! Got {len(phase_boundaries)} phase boundaries but {len(phonemes)} phonemes.")
        print("This usually means the number of log paths + 1 does not equal the number of phonemes.")
        return

    for i in range(len(phonemes) - 1):
        phase_ranges.append(
            (phase_boundaries[i], phase_boundaries[i+1], phase_weights[i])
        )

    print("\nPhase Ranges:")
    for i, (start, end, weight) in enumerate(phase_ranges):
        print(f"Phase {i+1} ({phonemes[i]} to {phonemes[i+1]}): {start}-{end} (weight: {weight:.3f})")

    # Load action lists from separate paths - all files have same name
    action_lists = []
    for i, log_path in enumerate(log_paths):
        file_path = os.path.join(log_path, "final_epoch_action_list.pkl")
        
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                if 'final_epoch_action_list' in data:
                    action_lists.append(data['final_epoch_action_list'])
                elif 'action_list' in data:
                    action_lists.append(data['action_list'])
                else:
                    action_lists.append(data)
                print(f"Successfully loaded phase {i+1} from {log_path} with {len(action_lists[-1])} actions")
        except FileNotFoundError:
            print(f"Warning: Could not find action list at {file_path}")
            action_lists.append([])
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            action_lists.append([])

    action_lists_old = action_lists.copy()

    # Apply 40% overlap between phases
    def modify_phase_boundaries(phase_boundaries, front_phase, back_phase):
        phase_length = phase_boundaries[back_phase] - phase_boundaries[front_phase]
        co_expression_steps = int(phase_length * overlap_ratio)
        
        for i in range(back_phase, len(phase_boundaries)):
            phase_boundaries[i] -= co_expression_steps
        
        return phase_boundaries, co_expression_steps

    def modify_action_lists(action_lists, current_boundaries, front_phase, back_phase, co_expression_steps):
        flat_actions = []
        for phase in action_lists:
            flat_actions.extend(phase)
        
        front_boundary = current_boundaries[front_phase]
        
        # Check for sufficient length
        if (front_boundary + co_expression_steps) > len(flat_actions):
            print(f"Warning: Overlap of {co_expression_steps} exceeds available actions ({len(flat_actions)}). Capping overlap.")
            co_expression_steps = len(flat_actions) - front_boundary
            if co_expression_steps < 0: co_expression_steps = 0
            
        if (front_boundary - co_expression_steps) < 0:
             print(f"Warning: Overlap of {co_expression_steps} is too large for previous phase. Capping overlap.")
             co_expression_steps = front_boundary
             if co_expression_steps < 0: co_expression_steps = 0


        l1 = flat_actions[:front_boundary - co_expression_steps]
        l2 = flat_actions[front_boundary - co_expression_steps:front_boundary]
        l3 = flat_actions[front_boundary:front_boundary + co_expression_steps]
        l4 = flat_actions[front_boundary + co_expression_steps:]
        
        # Ensure l2 and l3 are of same length for zipping
        min_overlap_len = min(len(l2), len(l3))
        if min_overlap_len < co_expression_steps:
            print(f"Adjusting overlap blending to {min_overlap_len} steps due to phase length.")
            l2 = l2[:min_overlap_len]
            l3 = l3[:min_overlap_len]

        l2_prime = []
        for a, b in zip(l2, l3):
            blended = [x + y for x, y in zip(a, b)]
            l2_prime.append(blended)
        
        # Reconstruct flat list
        modified_flat = l1 + l2_prime + l4
        
        action_lists_new = []
        current_idx = 0
        
        for i in range(len(current_boundaries)-1):
            if i == front_phase:
                phase_length = (current_boundaries[i+1] - current_boundaries[i]) - co_expression_steps
            else:
                phase_length = (current_boundaries[i+1] - current_boundaries[i])
            
            phase_actions = modified_flat[current_idx:current_idx + phase_length]
            action_lists_new.append(phase_actions)
            current_idx += len(phase_actions)
        
        return action_lists_new

    # Apply overlap between phases
    action_lists_current = [lst.copy() for lst in action_lists]
    phase_boundaries_current = phase_boundaries.copy()

    # This loop only runs if there are 3+ phonemes (2+ phases)
    for front_phase in range(len(phase_boundaries) - 2):
        back_phase = front_phase + 1
        
        print(f"\nApplying {overlap_ratio*100}% overlap between phase {front_phase+1} and {back_phase+1}")
        
        phase_boundaries_new, co_expression_steps = modify_phase_boundaries(
            phase_boundaries_current.copy(), front_phase, back_phase
        )
        
        action_lists_new = modify_action_lists(
            action_lists_current, phase_boundaries_current, front_phase, back_phase, co_expression_steps
        )
        
        action_lists_current = action_lists_new
        phase_boundaries_current = phase_boundaries_new

    # Update final results after overlap
    action_lists = action_lists_current
    phase_boundaries = phase_boundaries_current
    must_include_indices = phase_boundaries

    print(f"\nCode modification complete - now handles {len(log_paths)} separate log files with {overlap_ratio*100}% overlap")

    # Calculate checkpoint counts per phase
    total_weights = sum(weight for _, _, weight in phase_ranges)
    if total_weights == 0: total_weights = 1 # Avoid division by zero
    
    remaining_checkpoints = num_checkpoints - len(must_include_indices)
    
    if remaining_checkpoints < 0:
        print(f"Warning: More 'must-include' points ({len(must_include_indices)}) than total checkpoints ({num_checkpoints}).")
        remaining_checkpoints = 0
        phase_counts = [0] * len(phase_ranges)
    else:
        phase_counts = [max(1, int(round(weight * remaining_checkpoints / total_weights))) for _, _, weight in phase_ranges]
        while sum(phase_counts) != remaining_checkpoints:
            if sum(phase_counts) < remaining_checkpoints:
                phase_idx = np.argmax([(weight * remaining_checkpoints / total_weights) - count for (_, _, weight), count in zip(phase_ranges, phase_counts)])
                phase_counts[phase_idx] += 1
            else:
                phase_idx = np.argmin([(weight * remaining_checkpoints / total_weights) - count for (_, _, weight), count in zip(phase_ranges, phase_counts)])
                phase_counts[phase_idx] -= 1

    # Generate samples with dynamic spacing
    # IMPORTANT: Must include both START and END of each phase to ensure proper phoneme coverage
    checkpoints = []
    for i, (start, end, _) in enumerate(phase_ranges):
        phase_length = end - start
        if phase_length <= 0:
            continue # Skip empty phases

        # For each phase, we MUST include:
        # 1. The START (beginning of the phoneme transition)
        # 2. The END (completion of the phoneme transition)
        # 3. Any intermediate points based on phase_counts

        # Always include start and end
        phase_checkpoints = [start, end]

        # Add intermediate points if we have enough checkpoints allocated
        if phase_counts[i] > 2:
            # We already have start and end, so distribute the remaining points
            num_intermediate = phase_counts[i] - 2

            buffer = int(buffer_ratio * phase_length)
            valid_start = start + buffer
            valid_end = end - buffer

            if valid_start < valid_end and num_intermediate > 0:
                intermediate_indices = np.linspace(
                    valid_start,
                    valid_end,
                    num=num_intermediate,
                    dtype=int
                ).tolist()
                phase_checkpoints.extend(intermediate_indices)
        elif phase_counts[i] == 1:
            # If only 1 checkpoint allocated, use the end point (final pose)
            phase_checkpoints = [end]

        checkpoints.extend(phase_checkpoints)
        print(f"Phase {i+1} ({phonemes[i]} to {phonemes[i+1]}): checkpoints at {phase_checkpoints[:5]}{'...' if len(phase_checkpoints) > 5 else ''}")

    # Combine with must-include points and sort
    checkpoints = sorted(set(checkpoints + must_include_indices))

    # CRITICAL: Ensure the LAST checkpoint is the final frame of the last phase
    # This ensures the final phoneme (e.g., phoneme_T) is fully reached
    final_phase_end = phase_boundaries[-1] if phase_boundaries else 0
    if final_phase_end not in checkpoints:
        checkpoints.append(final_phase_end)
        checkpoints = sorted(set(checkpoints))
        print(f"[FORCED] Added final checkpoint at {final_phase_end} to ensure final phoneme is reached")

    # Final adjustment to ensure exactly num_checkpoints
    # IMPORTANT: Never remove the first or last checkpoint
    while len(checkpoints) > num_checkpoints:
        if len(checkpoints) <= 2:
            break  # Keep at least start and end

        distances = []
        for i in range(1, len(checkpoints)-1):  # Skip first and last
            distances.append((i, checkpoints[i+1] - checkpoints[i-1]))
        if distances:
            min_dist_idx = min(distances, key=lambda x: x[1])[0]
            # Don't remove if it's a phase boundary
            if checkpoints[min_dist_idx] not in phase_boundaries:
                checkpoints.pop(min_dist_idx)
            else:
                # Remove the closest non-boundary checkpoint
                for idx, _ in sorted(distances, key=lambda x: x[1]):
                    if checkpoints[idx] not in phase_boundaries:
                        checkpoints.pop(idx)
                        break
                else:
                    checkpoints.pop(min_dist_idx)  # Fallback
        else:
            checkpoints.pop(-2)  # Remove second to last if no middle points

    while len(checkpoints) < num_checkpoints:
        max_gap = 0
        insert_idx = 1
        for i in range(1, len(checkpoints)):
            gap = checkpoints[i] - checkpoints[i-1]
            if gap > max_gap:
                max_gap = gap
                insert_idx = i
        if max_gap > 1:  # Only add if gap is larger than 1
            new_point = checkpoints[insert_idx-1] + max_gap//2
            checkpoints.insert(insert_idx, new_point)
        else:
            break  # Can't add more checkpoints without duplicates
        checkpoints = sorted(checkpoints)

    print("\nOptimized checkpoints:", checkpoints)
    print(f"First checkpoint: {checkpoints[0]} (should be 0 - neutral start)")
    print(f"Last checkpoint: {checkpoints[-1]} (should be {phase_boundaries[-1]} - final phoneme)")
    print(f"Final phase boundary: {phase_boundaries[-1]}")

    print("\nCount per phase:")
    for i, (start, end, _) in enumerate(phase_ranges):
        count = sum(1 for x in checkpoints if start <= x <= end)
        has_start = start in checkpoints
        has_end = end in checkpoints
        print(f"Phase {i+1} ({phonemes[i]} -> {phonemes[i+1]}): {count} checkpoints [start: {'Y' if has_start else 'N'}, end: {'Y' if has_end else 'N'}]")
    print(f"\nTotal checkpoints: {len(checkpoints)} (target: {num_checkpoints})")

    # Calculate total length across all phases
    total_lengths = [len(lst) for lst in action_lists]
    cumulative_lengths = np.cumsum([0] + total_lengths)
    total_length_all = cumulative_lengths[-1]
    print(f"Total length across all phases: {total_length_all}")

    if total_length_all == 0:
        print("ERROR: No action list data loaded. Cannot generate checkpoints.")
        return

    print(f"Maximum checkpoint index requested: {max(checkpoints) if checkpoints else 'N/A'}")

    # Filter out checkpoints that exceed available frames
    # CRITICAL FIX: Allow the final phase boundary even if it equals total_length_all
    # This ensures the final phoneme (e.g., phoneme_T) is reached
    valid_checkpoints = [cp for cp in checkpoints if cp < total_length_all or (cp == phase_boundaries[-1] and cp <= total_length_all)]
    missing_count = len(checkpoints) - len(valid_checkpoints)
    
    if missing_count > 0:
        print(f"Found {missing_count} checkpoints beyond available frames (max: {total_length_all - 1})")
        
        # Add replacement points
        while len(valid_checkpoints) < num_checkpoints:
            # Find largest gap
            max_gap = 0
            insert_idx = 1
            for i in range(1, len(valid_checkpoints)):
                gap = valid_checkpoints[i] - valid_checkpoints[i-1]
                if gap > max_gap:
                    max_gap = gap
                    insert_idx = i
            
            if max_gap > 1: # Only add if gap is larger than 1
                new_point = valid_checkpoints[insert_idx-1] + max_gap//2
                valid_checkpoints.insert(insert_idx, new_point)
                print(f"Adding replacement checkpoint at {new_point}")
            else: # No gaps, just add near the end
                new_point = valid_checkpoints[-1] - 1 if len(valid_checkpoints) > 0 else 0
                if new_point < 0: new_point = 0
                valid_checkpoints.insert(len(valid_checkpoints)-1, new_point)
                print(f"Adding replacement checkpoint at {new_point} (near end)")
            
            valid_checkpoints = sorted(list(set(valid_checkpoints))) # re-sort and de-duplicate


    checkpoint_indices = sorted(list(set(valid_checkpoints))) # Final de-dupe
    
    # Ensure we have the correct number *finally*
    while len(checkpoint_indices) < num_checkpoints:
        checkpoint_indices.append(checkpoint_indices[-1] if checkpoint_indices else 0)
    while len(checkpoint_indices) > num_checkpoints:
        checkpoint_indices.pop()


    print(f"\nFinal checkpoint indices ({len(checkpoint_indices)} points):")
    print(checkpoint_indices)

    # Initialize parameters
    self_n = 300 * 3
    joint_rotate_num = 1
    correct_dimension = self_n + 3 * joint_rotate_num

    # # Let's dynamically determine the correct dimension size
    # if action_lists and action_lists[0] and action_lists[0][0]:
    #     actual_dimension = len(action_lists[0][0])
    #     print(f"\nDetected actual dimension size: {actual_dimension}")
    #     if actual_dimension != correct_dimension:
    #         print(f"ADJUSTING: Using actual dimension {actual_dimension} instead of calculated {correct_dimension}")
    #         correct_dimension = actual_dimension
    # else:
    #     print(f"Warning: Could not detect action list dimension. Using default: {correct_dimension}")

    # Let's dynamically determine the correct dimension size
    if action_lists and action_lists[0]:
        actual_dimension = len(action_lists[0][0])
        print(f"\nDetected actual dimension size: {actual_dimension}")
        print(f"Current self_n + 3*joint_rotate_num = {self_n} + {3*joint_rotate_num} = {self_n + 3*joint_rotate_num}")
        
        # Update to use the correct dimension
        if actual_dimension != (self_n + 3 * joint_rotate_num):
            print(f"ADJUSTING: Using actual dimension {actual_dimension} instead of calculated {self_n + 3 * joint_rotate_num}")
            correct_dimension = actual_dimension
        else:
            correct_dimension = self_n + 3 * joint_rotate_num
    else:
        correct_dimension = self_n + 3 * joint_rotate_num
        print(f"Using default dimension: {correct_dimension}")



    # def accumulate_checkpoints(action_lists, checkpoint_indices):
    #     delta_b0_conca = [0] * correct_dimension
    #     checkpoints_data_list = []
    #     global_index = 0
    #     found_indices = set()
        
    #     print(f"\nStarting checkpoint accumulation...")
    #     print(f"Using dimension size: {correct_dimension}")
        
    #     total_frames = sum(len(lst) for lst in action_lists)
    #     print(f"Total frames available: {total_frames}")

    #     if not checkpoint_indices:
    #          print("Error: No checkpoint indices provided.")
    #          return []
        
    #     # Adjust last checkpoint
    #     if checkpoint_indices[-1] >= total_frames and total_frames > 0:
    #         print(f"Adjusting last checkpoint from {checkpoint_indices[-1]} to {total_frames - 1}")
    #         checkpoint_indices[-1] = total_frames - 1
            
    #     print(f"Checkpoint indices: {checkpoint_indices}")
        
    #     checkpoint_data_verification = []
        
    #     for phase_idx, action_list in enumerate(action_lists):
    #         if not action_list:
    #             continue
                
    #         for action_idx, action in enumerate(action_list):
    #             if len(action) != correct_dimension:
    #                 print(f"ERROR: Phase {phase_idx+1}, Action {action_idx} has wrong dimension! Expected {correct_dimension}, got {len(action)}")
    #                 continue # Skip this action
                    
    #             delta_b0_conca = [delta_b0_conca[j] + action[j] for j in range(correct_dimension)]
                
    #             if global_index in checkpoint_indices:
    #                 if global_index not in found_indices: # Avoid duplicates
    #                     non_zero_count = sum(1 for x in delta_b0_conca if x != 0)
    #                     checkpoint_data_verification.append({
    #                         'index': global_index, 'non_zero_count': non_zero_count,
    #                     })
                        
    #                     checkpoints_data_list.append((global_index, delta_b0_conca.copy()))
    #                     found_indices.add(global_index)
    #                     print(f"Checkpoint at {global_index} (Phase {phase_idx+1}, Action {action_idx}) - {non_zero_count}/{correct_dimension} non-zero values")
                
    #             global_index += 1
        
    #     print(f"\n=== CHECKPOINT DATA VERIFICATION ===")
    #     for data in checkpoint_data_verification[:5]:
    #         print(f"Checkpoint: Index {data['index']}, {data['non_zero_count']} non-zero")
        
    #     missing = set(checkpoint_indices) - found_indices
    #     if missing:
    #         print(f"\nWarning: Missing {len(missing)} checkpoints: {sorted(missing)}")
    #         print(f"Maximum available index was {global_index-1}")
        
    #     print(f"\nSuccessfully captured {len(checkpoints_data_list)}/{len(checkpoint_indices)} checkpoints")
    #     return checkpoints_data_list

    # # Process all phases
    # combined_checkpoints = accumulate_checkpoints(action_lists, checkpoint_indices)


    def accumulate_checkpoints(action_lists, checkpoint_indices):
        # Use the correct dimension
        delta_b0_conca = [0] * correct_dimension
        checkpoints = []
        global_index = 0
        found_indices = set()
        
        print(f"\nStarting checkpoint accumulation...")
        print(f"Using dimension size: {correct_dimension}")
        
        # Calculate total number of frames
        total_frames = sum(len(lst) for lst in action_lists)
        print(f"Total frames available: {total_frames}")
        
        # Debug: Check first few actions to see if they contain non-zero values
        print(f"\nChecking first 3 actions across all phases:")
        action_count = 0
        for phase_idx, action_list in enumerate(action_lists):
            if not action_list:
                continue
            for action_idx, action in enumerate(action_list[:3]):  # Check first 3 actions
                non_zero_count = sum(1 for x in action if x != 0)
                print(f"Phase {phase_idx+1}, Action {action_idx}: {non_zero_count}/{len(action)} non-zero elements")
                action_count += 1
                if action_count >= 9:  # Limit output
                    break
            if action_count >= 9:
                break
        
        # Adjust last checkpoint if it's exactly one beyond available frames
        if checkpoint_indices[-1] == total_frames:
            print(f"Note: Last checkpoint index {checkpoint_indices[-1]} equals total frames")
            checkpoint_indices[-1] = total_frames - 1
            print(f"Adjusted last checkpoint to index {checkpoint_indices[-1]}")
        
        print(f"Checkpoint indices: {checkpoint_indices}")
        
        checkpoint_data = []  # Store checkpoint data for verification
        
        for phase_idx, action_list in enumerate(action_lists):
            if not action_list:
                continue
                
            for action_idx, action in enumerate(action_list):
                # Update the accumulated state
                delta_b0_conca = [delta_b0_conca[j] + action[j] for j in range(len(delta_b0_conca))]
                
                # Check if current global index matches any checkpoint
                if global_index in checkpoint_indices:
                    # Verify this checkpoint has non-zero values
                    non_zero_count = sum(1 for x in delta_b0_conca if x != 0)
                    checkpoint_data.append({
                        'index': global_index,
                        'non_zero_count': non_zero_count,
                        'data_sample': delta_b0_conca[:5]  # First 5 elements
                    })
                    
                    checkpoints.append((global_index, delta_b0_conca.copy()))
                    found_indices.add(global_index)
                    print(f"Checkpoint at {global_index} (Phase {phase_idx+1}, Action {action_idx}) - {non_zero_count}/{len(delta_b0_conca)} non-zero values")
                
                global_index += 1
        
        # Print checkpoint data verification
        print(f"\n=== CHECKPOINT DATA VERIFICATION ===")
        for i, data in enumerate(checkpoint_data[:5]):  # Show first 5 checkpoints
            print(f"Checkpoint {i}: Index {data['index']}, {data['non_zero_count']} non-zero, sample: {data['data_sample']}")
        
        # Final verification
        missing = set(checkpoint_indices) - found_indices
        if missing:
            print(f"\nWarning: Missing {len(missing)} checkpoints: {sorted(missing)}")
            print(f"Maximum available index was {global_index-1}")
        
        print(f"\nSuccessfully captured {len(checkpoints)}/{len(checkpoint_indices)} checkpoints")
        return checkpoints



    # Process all phases
    # combined_checkpoints = accumulate_checkpoints(action_lists, checkpoint_indices)


    # Process all phases
    combined_checkpoints = accumulate_checkpoints(action_lists, checkpoint_indices)

    # # Additional verification
    # print(f"\n=== FINAL VERIFICATION ===")
    # print(f"Total checkpoints captured: {len(combined_checkpoints)}")

    # if combined_checkpoints:
    #     all_zero_checkpoints = 0
    #     for idx, (checkpoint_idx, checkpoint_data) in enumerate(combined_checkpoints):
    #         if sum(1 for x in checkpoint_data if x != 0) == 0:
    #             all_zero_checkpoints += 1
        
    #     if all_zero_checkpoints > 0:
    #         print(f"TOTAL: {all_zero_checkpoints}/{len(combined_checkpoints)} checkpoints contain all zeros")
    #     else:
    #         print("SUCCESS: All checkpoints contain non-zero data")
            
    #     print(f"\nFirst checkpoint sample (first 10 elements): {combined_checkpoints[0][1][:10]}")
    # else:
    #     print("ERROR: No checkpoints were captured!")
    #     return # Stop if nothing was processed

    # # Save results
    # output_path = os.path.join(log_path_output, "combined_checkpoints.pkl")
    # with open(output_path, 'wb') as f:
    #     pickle.dump([cp[1] for cp in combined_checkpoints], f)

    # print(f"\nFinal checkpoint indices captured:")
    # for idx, _ in combined_checkpoints:
    #     print(f"- {idx}")

    # print(f"\nResults saved to: {output_path}")

    # # Verification
    # try:
    #     with open(output_path, "rb") as f:
    #         loaded_checkpoints = pickle.load(f)
    #     print(f"Verification: Loaded {len(loaded_checkpoints)} checkpoints")
    # except Exception as e:
    #     print(f"Error verifying output: {e}")

    # print("\n--- Deformation Processing Completed Successfully! ---")


    # Additional verification after processing
    print(f"\n=== FINAL VERIFICATION ===")
    print(f"Total checkpoints captured: {len(combined_checkpoints)}")

    if combined_checkpoints:
        # Check if checkpoints contain meaningful data
        all_zero_checkpoints = 0
        for idx, (checkpoint_idx, checkpoint_data) in enumerate(combined_checkpoints):
            non_zero_count = sum(1 for x in checkpoint_data if x != 0)
            if non_zero_count == 0:
                all_zero_checkpoints += 1
                if all_zero_checkpoints <= 3:  # Show first 3 zero checkpoints
                    print(f"WARNING: Checkpoint {idx} (index {checkpoint_idx}) contains ALL ZEROS!")
        
        if all_zero_checkpoints > 0:
            print(f"TOTAL: {all_zero_checkpoints}/{len(combined_checkpoints)} checkpoints contain all zeros")
        else:
            print("SUCCESS: All checkpoints contain non-zero data")
            
        # Show sample of first checkpoint data
        print(f"\nFirst checkpoint sample (first 10 elements): {combined_checkpoints[0][1][:10]}")
    else:
        print("ERROR: No checkpoints were captured!")


    # Save results
    output_path = os.path.join(log_path_output, f"combined_checkpoints_base{base_frames_global}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump([cp[1] for cp in combined_checkpoints], f)

    print(f"\nFinal checkpoint indices captured:")
    for idx, _ in combined_checkpoints:
        print(f"- {idx}")

    print(f"\nResults saved to: {output_path}")

    # Save overlap metadata to allow dynamic error correction in perform_the_deformation
    frame_boundaries = []
    for b in must_include_indices:
        for idx, (checkpoint_idx, _) in enumerate(combined_checkpoints):
            if checkpoint_idx == b:
                frame_boundaries.append(idx)
                break

    metadata = {
        'frame_boundaries': frame_boundaries,
        'phonemes': phonemes_list
    }
    metadata_path = os.path.join(log_path_output, "overlap_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Overlap metadata saved to: {metadata_path}")

    # Verification
    try:
        with open(output_path, "rb") as f:
            loaded_checkpoints = pickle.load(f)
        print(f"Verification: Loaded {len(loaded_checkpoints)} checkpoints")
        print("len of combined_checkpoints", len(loaded_checkpoints))
        # print("combined_checkpoints", loaded_checkpoints[0:1])    
    except Exception as e:
        print(f"Error verifying output: {e}")

    print("\nCode execution completed successfully!")    
    # --- Your script ends here ---


def run_deformation_processing_to_source(phonemes_list, log_paths_list, source_path, num_checkpoints=36):
    """
    Generates combined_checkpoints_base{base_frames_global}.pkl directly to the SOURCE path (from DEFORMATION_PATH_DB).
    This is a simplified version that generates checkpoints to the original training output folder.
    
    Args:
        phonemes_list: List of phonemes for this transition
        log_paths_list: List of paths (from DEFORMATION_PATH_DB)
        source_path: The source path where to save combined_checkpoints_base{base_frames_global}.pkl
        num_checkpoints: The number of frames (checkpoints) to generate for this transition
    """
    print("--- Starting Deformation Processing (to Source Path) ---")
    print(f"Phonemes: {phonemes_list}")
    print(f"Log Paths: {log_paths_list}")
    print(f"Output Path: {source_path}")

    overlap_ratio = 0
    log_paths = log_paths_list
    log_path_output = source_path  # Use the source path directly

    def get_total_steps_from_separate_logs():
        """Extracts step counts from log files"""
        total_steps_convergence = [0]
        
        for i, log_path in enumerate(log_paths):
            file_path = os.path.join(log_path, "log0.txt")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                    match = re.search(r'done at No\. (\d+) step', content)
                    if match:
                        total_steps_convergence.append(int(match.group(1)))
                        print(f"Log {i+1}: Found {match.group(1)} steps")
            else:
                print(f"File not found: {file_path}")
        
        print("total_steps_convergence =", total_steps_convergence)
        return total_steps_convergence

    total_steps_convergence = get_total_steps_from_separate_logs()
    phase_boundaries = np.cumsum(total_steps_convergence).tolist()
    print(f"Phase boundaries: {phase_boundaries}")

    # Load action lists
    action_lists = []
    for i, log_path in enumerate(log_paths):
        file_path = os.path.join(log_path, "final_epoch_action_list.pkl")
        
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                if 'final_epoch_action_list' in data:
                    action_lists.append(data['final_epoch_action_list'])
                elif 'action_list' in data:
                    action_lists.append(data['action_list'])
                else:
                    action_lists.append(data)
                print(f"Successfully loaded phase {i+1} from {log_path} with {len(action_lists[-1])} actions")
        except FileNotFoundError:
            print(f"Warning: Could not find action list at {file_path}")
            action_lists.append([])
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            action_lists.append([])

    # Flatten all actions
    all_actions = []
    for phase_actions in action_lists:
        all_actions.extend(phase_actions)

    if len(all_actions) == 0:
        print("ERROR: No actions found. Cannot generate checkpoints.")
        return

    # Calculate checkpoint selection
    total_steps = len(all_actions)
    
    if total_steps <= num_checkpoints:
        checkpoint_indices = list(range(total_steps))
    else:
        step_size = total_steps / num_checkpoints
        checkpoint_indices = [int(i * step_size) for i in range(num_checkpoints)]
        if checkpoint_indices[-1] != total_steps - 1:
            checkpoint_indices[-1] = total_steps - 1

    print(f"Selected {len(checkpoint_indices)} checkpoint indices")

    # Calculate cumulative deformation at each checkpoint
    combined_checkpoints = []
    cumulative = None
    
    for i, checkpoint_idx in enumerate(checkpoint_indices):
        if cumulative is None:
            cumulative = np.array(all_actions[0])
        
        prev_idx = checkpoint_indices[i-1] if i > 0 else 0
        for j in range(prev_idx + 1, checkpoint_idx + 1):
            if j < len(all_actions):
                cumulative = cumulative + np.array(all_actions[j])
        
        combined_checkpoints.append((checkpoint_idx, cumulative.tolist()))

    # Save results to source path
    output_path = os.path.join(log_path_output, f"combined_checkpoints_base{base_frames_global}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump([cp[1] for cp in combined_checkpoints], f)

    print(f"Results saved to: {output_path}")
    
    # Verification
    try:
        with open(output_path, "rb") as f:
            loaded_checkpoints = pickle.load(f)
        print(f"Verification: Loaded {len(loaded_checkpoints)} checkpoints")
    except Exception as e:
        print(f"Error verifying output: {e}")

    print("\nDeformation processing to source completed!")


# --- 3. MAYA UI AND PIPELINE SCRIPT ---

def run_maya_render():
    """
    (PLACEHOLDER) This simulates the render process.
    """
    print("--- (SIMULATION) ---")
    print("Starting Maya render...")
    print("This is where cmds.render() would be called.")
    print("...Render complete.")
    print("--- (SIMULATION) ---")


def run_playblast_script():
    playblast_script = r"C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\d2 helper functions\do_playblast.py"
    print(f"\n--- Calling Playblast Script ---")
    if os.path.exists(playblast_script):
        try:
            import sys
            import importlib
            script_dir = r"C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\d2 helper functions"
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            import do_playblast
            importlib.reload(do_playblast)
            do_playblast.main()
            print("[PLAYBLAST] Playblast generated successfully.")
        except Exception as e:
            print(f"[PLAYBLAST ERROR] Failed to run playblast script: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[PLAYBLAST ERROR] Script not found: {playblast_script}")


def combine_pngs_to_mp4():
    """
    (PLACEHOLDER) This simulates the PNG-to-MP4 script.
    """
    print("--- (SIMULATION) ---")
    print("Combining PNGs to MP4...")
    print("This is where your FFMPEG or subprocess script would run.")
    
    fake_video_path = "C:/maya_project/movies/SIMULATED_video.mp4"
    print(f"Video successfully saved to {fake_video_path}")
    print("--- (SIMULATION) ---")
    return fake_video_path


# --- SENTENCE-TO-PHONEME CONVERSION (Natural Language Support) ---
# Maps English grapheme patterns to our available phoneme inventory.
# Our inventory: N, AA, T, IY, V, Y, UW, EH
# Missing common phonemes are mapped to the closest available one.

# Word-level dictionary for common words (hand-tuned for our limited inventory)
WORD_TO_PHONEMES = {
    # Common words from expected sentences
    "not": ["N", "AA", "T"],
    "even": ["IY", "V", "EH", "N"],
    "you": ["Y", "UW"],
    "you've": ["Y", "UW", "V"],
    "youve": ["Y", "UW", "V"],
    "yet": ["Y", "EH", "T"],
    "eaten": ["IY", "T", "EH", "N"],
    "it": ["IY", "T"],
    "the": ["EH"],          # /dh/ not available, approximate
    "a": ["AA"],
    "an": ["AA", "N"],
    "in": ["IY", "N"],
    "on": ["AA", "N"],
    "at": ["AA", "T"],
    "to": ["T", "UW"],
    "too": ["T", "UW"],
    "two": ["T", "UW"],
    "no": ["N", "AA"],      # /oh/ not available, use AA
    "yes": ["Y", "EH"],     # /s/ not available, skip
    "i": ["AA", "IY"],      # diphthong /ai/
    "eye": ["AA", "IY"],
    "me": ["EH", "IY"],     # /m/ not available, approximate with EH
    "my": ["EH", "AA", "IY"],
    "we": ["UW", "IY"],
    "he": ["IY"],
    "be": ["IY"],
    "net": ["N", "EH", "T"],
    "vet": ["V", "EH", "T"],
    "tin": ["T", "IY", "N"],
    "ten": ["T", "EH", "N"],
    "tan": ["T", "AA", "N"],
    "van": ["V", "AA", "N"],
    "vine": ["V", "AA", "IY", "N"],
    "view": ["V", "Y", "UW"],
    "new": ["N", "Y", "UW"],
    "knew": ["N", "Y", "UW"],
    "tune": ["T", "UW", "N"],
    "noon": ["N", "UW", "N"],
    "teen": ["T", "IY", "N"],
    "neat": ["N", "IY", "T"],
    "eat": ["IY", "T"],
    "tea": ["T", "IY"],
    "vat": ["V", "AA", "T"],
    "nave": ["N", "EH", "IY", "V"],
    "knit": ["N", "IY", "T"],
    "native": ["N", "EH", "IY", "T", "IY", "V"],
    "invite": ["IY", "N", "V", "AA", "IY", "T"],
    "nut": ["N", "AA", "T"],
    "naivety": ["N", "AA", "IY", "V", "T", "IY"],
    "unity": ["Y", "UW", "N", "IY", "T", "IY"],
    "university": ["Y", "UW", "N", "IY", "V", "EH", "T", "IY"],
    "hi": ["AA", "IY"],
    "hey": ["EH", "IY"],
    "hello": ["EH", "AA"],  # /h,l,oh/ not available
    "bye": ["AA", "IY"],
    "have": ["AA", "V"],     # /h/ not available
    "haven't": ["AA", "V", "EH", "N", "T"],
    "haven't": ["AA", "V", "EH", "N", "T"],
    "had": ["AA"],           # /h,d/ not available
    "hate": ["EH", "IY", "T"],  # /h/ not available
    "every": ["EH", "V", "IY"],
    "never": ["N", "EH", "V", "EH"],
    "nine": ["N", "AA", "IY", "N"],
    "nineteen": ["N", "AA", "IY", "N", "T", "IY", "N"],
    "note": ["N", "AA", "T"],
    "vote": ["V", "AA", "T"],
    "voted": ["V", "AA", "T", "EH"],
    "yacht": ["Y", "AA", "T"],
    "tv": ["T", "IY", "V", "IY"],
}

# Phoneme-to-internal-name mapping
PHONEME_NAME_TO_INTERNAL = {
    "N":  "dataV22_phoneme_N",
    "AA": "dataV22_phoneme_AA",
    "T":  "dataV22_phoneme_T",
    "IY": "dataV22_phoneme_IY",
    "V":  "dataV22_phoneme_V",
    "Y":  "dataV22_phoneme_Y",
    "UW": "dataV22_phoneme_UW",
    "EH": "dataV22_phoneme_EH",
}

# Available phoneme set (for validation)
AVAILABLE_PHONEMES = set(PHONEME_NAME_TO_INTERNAL.keys())

# Simple letter-to-phoneme fallback rules (very approximate)
LETTER_TO_PHONEME = {
    'a': 'AA', 'b': None, 'c': None, 'd': None, 'e': 'EH',
    'f': None, 'g': None, 'h': None, 'i': 'IY', 'j': None,
    'k': None, 'l': None, 'm': None, 'n': 'N',  'o': 'AA',
    'p': None, 'q': None, 'r': None, 's': None, 't': 'T',
    'u': 'UW', 'v': 'V',  'w': 'UW', 'x': None, 'y': 'Y',
    'z': None,
}


# Expression command patterns: maps natural language phrases to AU sequences
EXPRESSION_COMMANDS = {
    # Single expressions from neutral
    "smile":              ["dataV22_AU0_neutral", "dataV22_AU12"],
    "open the mouth":     ["dataV22_AU0_neutral", "dataV22_AU25"],
    "open mouth":         ["dataV22_AU0_neutral", "dataV22_AU25"],
    "open the lips":      ["dataV22_AU0_neutral", "dataV22_AU25"],
    "lips part":          ["dataV22_AU0_neutral", "dataV22_AU25"],
    "raise the brow":     ["dataV22_AU0_neutral", "dataV22_AU1", "dataV22_AU2"],
    "raise brow":         ["dataV22_AU0_neutral", "dataV22_AU1", "dataV22_AU2"],
    "raise the eyebrow":  ["dataV22_AU0_neutral", "dataV22_AU1", "dataV22_AU2"],
    "raise eyebrow":      ["dataV22_AU0_neutral", "dataV22_AU1", "dataV22_AU2"],
    "raise eyebrows":     ["dataV22_AU0_neutral", "dataV22_AU1", "dataV22_AU2"],
    "lower the brow":     ["dataV22_AU0_neutral", "dataV22_AU4"],
    "lower brow":         ["dataV22_AU0_neutral", "dataV22_AU4"],
    "frown":              ["dataV22_AU0_neutral", "dataV22_AU4"],
    "squint":             ["dataV22_AU0_neutral", "dataV22_AU7"],
    "tighten lids":       ["dataV22_AU0_neutral", "dataV22_AU7"],
    "cheek raise":        ["dataV22_AU0_neutral", "dataV22_AU6"],
    "raise cheeks":       ["dataV22_AU0_neutral", "dataV22_AU6"],
    "wuu":                ["dataV22_AU0_neutral", "dataV22_AU22"],
    "wuu sound":          ["dataV22_AU0_neutral", "dataV22_AU22"],
    "pucker":             ["dataV22_AU0_neutral", "dataV22_AU22"],
    "cheese":             ["dataV22_AU0_neutral", "dataV22_AU6"],
    "sad":                    ["dataV22_AU0_neutral", "dataV22_AU4"],
    # Compound expressions
    "smile and open mouth":   ["dataV22_AU0_neutral", "dataV22_AU12", "dataV22_AU25"],
    "surprise":               ["dataV22_AU0_neutral", "dataV22_AU1", "dataV22_AU2", "dataV22_AU25"],
    "happy":                  ["dataV22_AU0_neutral", "dataV22_AU6", "dataV22_AU12"],
}

# --- DROPDOWN MENU DEFINITIONS ---
# Phoneme dropdown items: (display_label, internal_state_name)
PHONEME_DROPDOWN_ITEMS = [
    ("-- Select Phoneme --", ""),
    ("Phoneme N  (net)",     "dataV22_phoneme_N"),
    ("Phoneme AA (pot)",     "dataV22_phoneme_AA"),
    ("Phoneme T  (test)",    "dataV22_phoneme_T"),
    ("Phoneme IY (beat)",    "dataV22_phoneme_IY"),
    ("Phoneme V  (vet)",     "dataV22_phoneme_V"),
    ("Phoneme Y  (yet)",     "dataV22_phoneme_Y"),
    ("Phoneme UW (lute)",    "dataV22_phoneme_UW"),
    ("Phoneme EH (bet)",     "dataV22_phoneme_EH"),
]

# FACS (Facial Action Coding System) dropdown items
FACS_DROPDOWN_ITEMS = [
    ("-- Select FACS Action Unit --", ""),
    ("AU 0  - Neutral",              "dataV22_AU0_neutral"),
    ("AU 1  - Inner Brow Raiser",    "dataV22_AU1"),
    ("AU 2  - Outer Brow Raiser",    "dataV22_AU2"),
    ("AU 4  - Brow Lowerer",         "dataV22_AU4"),
    ("AU 6  - Cheek Raiser",         "dataV22_AU6"),
    ("AU 7  - Lid Tightener",        "dataV22_AU7"),
    ("AU 12 - Smile",                "dataV22_AU12"),
    ("AU 22 - WUU Sound",            "dataV22_AU22"),
    ("AU 25 - Lips Part",            "dataV22_AU25"),
]

# Emotion / Expression dropdown items
EMOTION_DROPDOWN_ITEMS = [
    ("-- Select Expression --", ""),
    ("Smile",                   "smile"),
    ("Happy",                   "happy"),
    ("Sad",                     "sad"),
    ("Frown",                   "frown"),
    ("Surprise",                "surprise"),
    ("Squint",                  "squint"),
    ("Open Mouth (Lips Part)",  "open mouth"),
    ("Raise the Brow (AU1+AU2)","raise the brow"),
    ("Lower the Brow",         "lower brow"),
    ("Cheek Raise",            "cheek raise"),
    ("Pucker (WUU)",           "pucker"),
    ("Cheese",                 "cheese"),
]


def sentence_to_phonemes(sentence):
    """
    Convert an English sentence to a sequence of phoneme internal names.
    Uses a word dictionary first, then falls back to letter-by-letter mapping.
    Skips phonemes not in our inventory with a warning.
    
    Args:
        sentence: English sentence string (e.g., "not even you")
    
    Returns:
        List of internal phoneme names (e.g., ["dataV22_phoneme_N", "dataV22_phoneme_AA", ...])
        Always prepends Neutral as the starting pose.
    """
    # Match words OR punctuation marks
    tokens = re.findall(r"[a-z']+|[.,!?;:]", sentence.lower())
    
    if not tokens:
        return []
    
    phoneme_sequence = []
    skipped_chars = []
    
    for token in tokens:
        if token in ".,!?;:":
            print(f"  [PUNCTUATION] '{token}' -> Pause (Neutral)")
            phoneme_sequence.append("dataV22_AU0_neutral")
            continue
            
        # Strip trailing/leading punctuation from word just in case
        clean_word = token.strip("'\"")
        if not clean_word:
            continue
            
        if clean_word in WORD_TO_PHONEMES:
            # Use dictionary lookup
            word_phonemes = WORD_TO_PHONEMES[clean_word]
            print(f"  [WORD] '{clean_word}' -> {word_phonemes}")
            for p in word_phonemes:
                if p in AVAILABLE_PHONEMES:
                    phoneme_sequence.append(PHONEME_NAME_TO_INTERNAL[p])
                else:
                    print(f"  [WARNING] Phoneme '{p}' not in inventory, skipping")
                    skipped_chars.append(p)
        else:
            # Fallback: letter-by-letter
            print(f"  [LETTER-BY-LETTER] '{clean_word}' (not in dictionary)")
            for ch in clean_word:
                mapped = LETTER_TO_PHONEME.get(ch)
                if mapped and mapped in AVAILABLE_PHONEMES:
                    phoneme_sequence.append(PHONEME_NAME_TO_INTERNAL[mapped])
                elif mapped is None and ch.isalpha():
                    print(f"  [WARNING] Letter '{ch}' in word '{clean_word}' has no phoneme mapping, skipping")
                    skipped_chars.append(ch)
    
    if skipped_chars:
        print(f"\n  [SUMMARY] Skipped characters/phonemes with no mapping: {skipped_chars}")
        print(f"  [SUMMARY] Animation will continue without these sounds.\n")
    
    # Remove consecutive duplicates (e.g., two N's in a row)
    deduped = []
    for p in phoneme_sequence:
        if not deduped or deduped[-1] != p:
            deduped.append(p)
    
    return deduped


def detect_input_mode(input_text):
    """
    Detect whether the input is:
      1. 'machine'    - Machine language: contains 'to' with phoneme/AU names
      2. 'expression' - Expression command: matches a known expression phrase
      3. 'sentence'   - Natural English sentence to be converted to phonemes
    
    Returns:
        One of: 'machine', 'expression', 'sentence'
    """
    lower = input_text.strip().lower()
    
    # Check if it's an expression command
    if lower in EXPRESSION_COMMANDS:
        return 'expression'
    
    # Check for machine language indicators:
    # Contains "phoneme_", "au", "neutral", "datav22", or structured "X to Y" with known state names
    machine_indicators = ['phoneme_', 'au0', 'au1', 'au2', 'au4', 'au6', 'au7', 'au12', 'au22', 'au25', 'datav22']
    if any(ind in lower for ind in machine_indicators):
        return 'machine'
    
    # Check if all words (excluding "to") are in STATE_MAP
    cleaned = lower.replace(",", " ")
    parts = [p for p in cleaned.split() if p and p != 'to']
    if parts and all(p in STATE_MAP for p in parts):
        return 'machine'
    
    # Default to sentence mode
    return 'sentence'


def parse_sequence_string(input_text):
    """
    This function translates the "human" input into the
    phonemes list and the list of log paths.
    
    Supports 3 input modes (auto-detected):
      1. Machine language: "Neutral to phoneme_N to phoneme_AA" or "AU2 to AU7"
      2. Expression command: "smile", "open the mouth", "raise the brow"
      3. Natural sentence:  "not even you" -> auto-converted to phoneme sequence
    
    If a phoneme is not in our inventory, a warning is printed but the
    animation continues by skipping to the next available phoneme.
    """
    print(f"Parsing user input: '{input_text}'")
    
    # Detect input mode
    mode = detect_input_mode(input_text)
    print(f"[INPUT MODE] Detected: {mode}")
    
    if mode == 'expression':
        # Expression command mode
        lower = input_text.strip().lower()
        phonemes = EXPRESSION_COMMANDS[lower]
        print(f"[EXPRESSION] '{lower}' -> {phonemes}")
    
    elif mode == 'sentence':
        # Natural sentence mode
        print(f"[SENTENCE MODE] Converting English sentence to phoneme sequence...")
        phoneme_list = sentence_to_phonemes(input_text)
        
        if not phoneme_list:
            raise ValueError(
                f"Could not extract any phonemes from sentence: '{input_text}'. "
                f"Try using machine language format instead, e.g. 'Neutral to phoneme_N to phoneme_AA'"
            )
        
        # Prepend neutral as starting pose
        phonemes = ["dataV22_AU0_neutral"] + phoneme_list
        
        print(f"\n[SENTENCE MODE] Converted sentence to {len(phonemes)} states:")
        print(f"  {' -> '.join([p.replace('dataV22_', '') for p in phonemes])}")
    
    else:
        # Machine language mode (original behavior)
        # Clean up the input string
        cleaned_text = input_text.lower().replace(",", " ").replace("to", " ")
        
        # Split by spaces and get non-empty parts, then map to standard names
        parts = [part for part in cleaned_text.split() if part]
        
        phonemes = []
        skipped = []
        for part in parts:
            if part in STATE_MAP:
                phonemes.append(STATE_MAP[part])
            else:
                print(f"[WARNING] Unknown state '{part}' is not in our inventory. Skipping.")
                skipped.append(part)
        
        if skipped:
            print(f"[WARNING] Skipped unknown states: {skipped}")
            print(f"[WARNING] Animation will continue without these states.")

    if len(phonemes) < 2:
        raise ValueError(f"Input sequence is too short. Need at least 2 states. Got: {phonemes}")

    print(f"Parsed phonemes: {phonemes}")

    # Build the log_paths list
    log_paths = []

    # Create a normalized key map for the database
    # This maps normalized keys to original keys in the database
    db_key_map = {}
    for original_key in DEFORMATION_PATH_DB.keys():
        normalized_key = normalize_task_key(original_key)
        db_key_map[normalized_key] = original_key

    for i in range(len(phonemes) - 1):
        state_a = phonemes[i]
        state_b = phonemes[i+1]

        # Try multiple possible key formats
        transition_key_underscore = f"{state_a}_to_{state_b}"
        transition_key_space = f"{state_a} to {state_b}"

        # First try direct lookup
        path = DEFORMATION_PATH_DB.get(transition_key_underscore)
        if path is None:
            path = DEFORMATION_PATH_DB.get(transition_key_space)

        # If direct lookup fails, try normalized lookup
        if path is None:
            # Normalize the key and look up in the normalized map
            normalized_key = normalize_task_key(transition_key_underscore)
            original_key = db_key_map.get(normalized_key)
            if original_key:
                path = DEFORMATION_PATH_DB[original_key]
                print(f"Found path via normalized lookup: '{normalized_key}' -> '{original_key}'")

        if path is None:
            # Handle missing path
            raise ValueError(
                f"Path not found in database for transition: '{state_a}' to '{state_b}'. "
                f"Tried keys: '{transition_key_underscore}' and '{transition_key_space}'. "
                f"Please add it to DEFORMATION_PATH_DB."
            )

        print(f"Found path for '{state_a}' to '{state_b}'")
        log_paths.append(path)

    # Auto-configure previous_pose_folder if needed
    # This checks if the first phoneme is NOT neutral
    global NEUTRAL_TO_OTHER_POSE_FLAG, PREVIOUS_POSE_FOLDER

    from_pose = phonemes[0]
    IS_NEUTRAL_TO_OTHER_POSE = (normalize_task_key(from_pose) == 'datav22_au0_neutral')

    if not IS_NEUTRAL_TO_OTHER_POSE:
        # neutral_to_other_pose_flag = 0 case
        print(f"\n[AUTO-CONFIG] Detected non-neutral starting pose: '{from_pose}'")
        previous_pose_folder = auto_configure_previous_pose_folder(from_pose)

        if previous_pose_folder:
            print(f"[AUTO-CONFIG] Found previous pose folder: {previous_pose_folder}")

            # Set global variables for use in initial_joint_position()
            NEUTRAL_TO_OTHER_POSE_FLAG = 0
            PREVIOUS_POSE_FOLDER = previous_pose_folder

            # Update the d1-adhoc-for-b99.py file
            d1_b99_file = r"c:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\d1-adhoc-for-b99.py"
            try:
                with open(d1_b99_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Update previous_pose_folder
                content = re.sub(
                    r'previous_pose_folder\s*=\s*r"[^"]*"',
                    f'previous_pose_folder = r"{previous_pose_folder}"',
                    content
                )

                # Update neutral_to_other_pose_flag to 0
                content = re.sub(
                    r'neutral_to_other_pose_flag\s*=\s*[01]',
                    'neutral_to_other_pose_flag = 0',
                    content
                )

                with open(d1_b99_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                print(f"[AUTO-CONFIG] Updated d1-adhoc-for-b99.py:")
                print(f"  - neutral_to_other_pose_flag = 0")
                print(f"  - previous_pose_folder = {previous_pose_folder}")
            except Exception as e:
                print(f"[WARNING] Could not auto-update d1-adhoc-for-b99.py: {e}")
        else:
            print(f"[WARNING] Could not find previous pose folder. Manual configuration may be needed.")
            NEUTRAL_TO_OTHER_POSE_FLAG = 0
            PREVIOUS_POSE_FOLDER = ""
    else:
        # neutral_to_other_pose_flag = 1 case
        print(f"\n[AUTO-CONFIG] Starting from neutral pose. No previous pose configuration needed.")

        # Set global variables for use in initial_joint_position()
        NEUTRAL_TO_OTHER_POSE_FLAG = 1
        PREVIOUS_POSE_FOLDER = ""

        # Update the d1-adhoc-for-b99.py file to set flag to 1
        d1_b99_file = r"c:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\d1-adhoc-for-b99.py"
        try:
            with open(d1_b99_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Update neutral_to_other_pose_flag to 1
            content = re.sub(
                r'neutral_to_other_pose_flag\s*=\s*[01]',
                'neutral_to_other_pose_flag = 1',
                content
            )

            with open(d1_b99_file, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"[AUTO-CONFIG] Updated d1-adhoc-for-b99.py:")
            print(f"  - neutral_to_other_pose_flag = 1")
        except Exception as e:
            print(f"[WARNING] Could not auto-update d1-adhoc-for-b99.py: {e}")

    return phonemes, log_paths


def generate_short_sequence_name(user_text, phonemes):
    """
    Generate a short, meaningful folder name based on the sequence.

    Args:
        user_text: The original user input text
        phonemes: List of phoneme names

    Returns:
        A short folder name
    """
    import hashlib
    from datetime import datetime

    # Map common sequences to friendly names
    friendly_names = {
        "neutral to phoneme_y to phoneme_uw to phoneme_v to phoneme_n to phoneme_aa to phoneme_t to phoneme_iy to phoneme_t to phoneme_n to phoneme_iy to phoneme_t to phoneme_y to phoneme_eh to phoneme_t": "youve_not_eaten_it_yet",
        "neutral to phoneme_n to phoneme_aa to phoneme_t to phoneme_iy to phoneme_v to phoneme_n to phoneme_y to phoneme_uw": "not_even_you",
    }

    # Normalize the user text for comparison
    normalized_input = user_text.lower().replace(",", " ").strip()
    normalized_input = " ".join(normalized_input.split())  # Remove extra spaces

    # Check if we have a friendly name
    if normalized_input in friendly_names:
        return "seq_" + friendly_names[normalized_input]

    # Otherwise, create a hash-based name with timestamp
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    text_hash = hashlib.md5(user_text.encode()).hexdigest()[:8]

    # Use first and last phoneme for context, plus length
    first_phoneme = phonemes[0].replace("dataV22_", "").replace("phoneme_", "").replace("AU0_neutral", "neu")
    last_phoneme = phonemes[-1].replace("dataV22_", "").replace("phoneme_", "")
    
    if len(phonemes) > 2:
        # Build an acronym of the phonemes so it's unique
        middle_phonemes = "_".join([p.replace("dataV22_phoneme_", "") for p in phonemes[1:-1]])
        return f"seq_{first_phoneme}_{middle_phonemes}_{last_phoneme}"
    else:
        return f"seq_{first_phoneme}_to_{last_phoneme}"
    # _{timestamp}_{text_hash}"

def get_transition_frames(p_from, p_to, base_frames=30):
    """
    Calculate dynamic transition length based on phoneme categories.
    Neutral to Consonant is quick, Consonant to Vowel is longer, etc.
    """
    # Clean up the names
    f = p_from.replace('dataV22_phoneme_', '').replace('dataV22_AU0_neutral', 'Neutral')
    t = p_to.replace('dataV22_phoneme_', '').replace('dataV22_AU0_neutral', 'Neutral')
    
    # Treat some specific AUs as Neutral/Consonants for fallback
    if f.startswith('dataV22_AU'): f = 'Neutral'
    if t.startswith('dataV22_AU'): t = 'Neutral'
    
    vowels = ['AA', 'IY', 'UW', 'EH']
    consonants = ['N', 'T', 'V', 'Y']
    
    if f == 'Neutral':
        return int(base_frames * 0.6)  # ~21 frames
    elif f in consonants and t in vowels:
        return int(base_frames * 1.3)  # ~45 frames
    elif f in vowels and t in consonants:
        return int(base_frames * 0.6)  # ~21 frames
    elif f in vowels and t in vowels:
        return int(base_frames * 1.2)  # ~42 frames
    elif f in consonants and t in consonants:
        return int(base_frames * 0.7)  # ~24 frames
    else:
        return base_frames


def run_single_transition(phoneme_from, phoneme_to, start_frame, continue_mode, is_first_transition=False):
    global ACCUMULATED_ERROR, PER_FRAME_DEBUG_HISTORY

    print(f"\n{'='*60}")
    print(f"RUNNING TRANSITION: {phoneme_from} -> {phoneme_to}")
    print(f"{'='*60}\n")

    # Build transition key for database lookup
    from_name = phoneme_from.replace('dataV22_', '').replace('AU0_neutral', 'neutral')
    to_name = phoneme_to.replace('dataV22_', '')
    user_text = f"{from_name} to {to_name}"
    phonemes, log_paths = parse_sequence_string(user_text)
    
    # Get the source path from the database
    # If OVERLAP_RATIO > 0, try the overlap-aware key first (e.g., "... [overlap40]")
    source_path = log_paths[0] if log_paths else None
    
    # --- SNAP COMPATIBILITY ---
    # Disable overlap blending if the transition involves the Neutral pose,
    # because the Hidden Snap requires a clean, non-overlapped start/end.
    is_neutral_involved = ("AU0_neutral" in phoneme_from or "AU0_neutral" in phoneme_to)
    
    if OVERLAP_RATIO > 0 and not is_neutral_involved:
        overlap_pct = int(OVERLAP_RATIO * 100)
        # The overlap keys in the DB look like: "dataV22_AU0_neutral to dataV22_phoneme_V [overlap40]"
        overlap_key = f"{phonemes[0]} to {phonemes[1]} [overlap{overlap_pct}]"
        overlap_path = DEFORMATION_PATH_DB.get(overlap_key)
        if overlap_path is not None:
            source_path = overlap_path
            print(f"[OVERLAP] Using overlap-aware checkpoints: '{overlap_key}'")
            print(f"[OVERLAP] Path: {source_path}")
        else:
            print(f"[OVERLAP] No overlap key '{overlap_key}' found in DB, falling back to standard")
            print(f"[OVERLAP] Standard path: {source_path}")
    elif is_neutral_involved and OVERLAP_RATIO > 0:
        print("[OVERLAP] Bypassing overlap blending because transition involves Neutral pose (Snap compatibility).")
        print(f"[OVERLAP] Standard path: {source_path}")
    
    if source_path is None:
        raise RuntimeError(f"Could not find path for transition: {phoneme_from} -> {phoneme_to}")
    
    # Check if combined_checkpoints_base{base_frames_global}.pkl exists in the SOURCE path (from database)
    checkpoint_file_path = os.path.join(source_path, f"combined_checkpoints_base{base_frames_global}.pkl")
    
    # Calculate dynamic frames based on phonemes
    transition_frames = get_transition_frames(phoneme_from, phoneme_to, base_frames=base_frames_global)
    num_checkpoints = transition_frames + 1
    
    print(f"[CHECKPOINT] Looking for: {checkpoint_file_path}")
    
    generate_new = False
    if os.path.exists(checkpoint_file_path):
        try:
            with open(checkpoint_file_path, 'rb') as f:
                existing_checkpoints = pickle.load(f)
            if len(existing_checkpoints) != num_checkpoints:
                print(f"[CHECKPOINT] Existing file has {len(existing_checkpoints)} frames, but {num_checkpoints} are needed. Regenerating...")
                generate_new = True
            else:
                print(f"[CHECKPOINT] Found existing combined_checkpoints_base{base_frames_global}.pkl in source path with correct frame count.")
        except Exception as e:
            print(f"[CHECKPOINT] Error loading existing file: {e}. Regenerating...")
            generate_new = True
    else:
        print(f"[CHECKPOINT] Not found, generating to source path...")
        generate_new = True
        
    if generate_new:
        # Generate combined_checkpoints_base{base_frames_global}.pkl to the SOURCE path (not sequence_combined_deformation_parameters_hub)
        run_deformation_processing_to_source(phonemes, log_paths, source_path, num_checkpoints=num_checkpoints)

    # --- PREPARE CORRECTION VECTOR (Do NOT Apply to Scene Here) ---
    correction_vectors = None
    
    if ENABLE_FEEDBACK_CONTROL and ACCUMULATED_ERROR is not None and not is_first_transition:
        print(f"[FEEDBACK CONTROL] Calculating correction vector for smooth dissipation...")
        
        # We prepare the FULL correction vector needed.
        # perform_the_deformation will split this up over the frames.
        correction_vectors = {}
        for key in ACCUMULATED_ERROR.keys():
            err = np.array(ACCUMULATED_ERROR[key])
            if len(err) > 0:
                # We want to add 'err * rate' by the end of the transition
                correction_vectors[key] = err * ERROR_DISSIPATION_RATE
        
        print(f"[FEEDBACK CONTROL] Vector prepared. Magnitude: {np.linalg.norm(correction_vectors['x_joint_conca']):.6f}")

    # --- PERFORM DEFORMATION (With Correction Passed In) ---
    final_frame, per_frame_data = perform_the_deformation(source_path, start_frame, continue_mode, correction_vectors)

    # --- STORE PER-FRAME DATA FOR PLOTTING ---
    if DEBUG_ENABLED:
        if 'PER_FRAME_DEBUG_HISTORY' not in globals():
            PER_FRAME_DEBUG_HISTORY = []
        PER_FRAME_DEBUG_HISTORY.append({
            'label': f"{from_name}->{to_name}",
            'data': per_frame_data
        })

    # --- COMPUTE ERROR FOR NEXT TRANSITION (Same as before) ---
    if ENABLE_FEEDBACK_CONTROL:
        # ... (Your existing error calculation logic for ACCUMULATED_ERROR remains here) ...
        # Copied strictly from your provided code to ensure continuity:
        ideal_positions = load_ideal_target_positions(phoneme_to)
        if ideal_positions is not None:
            joint_groups = {'oor': 300, 'ooc_l': 300, 'ooc_r': 300, 'frontalis': 300}
            actual_positions = retrieve_landmark_positions(joint_groups, final_frame, is_first_transition=False)
            
            actual_flat = {
                'x_joint_conca': [item for sublist in actual_positions.get('oor', []) for item in sublist],
                'x_joint_rotation_conca': [0, 0, 0],
                'x_ooc_l_joint_conca': [item for sublist in actual_positions.get('ooc_l', []) for item in sublist],
                'x_ooc_r_joint_conca': [item for sublist in actual_positions.get('ooc_r', []) for item in sublist],
                'x_frontalis_joint_conca': [item for sublist in actual_positions.get('frontalis', []) for item in sublist]
            }
            
            # --- OVERLAP CORRECTION ---
            # If this transition was generated with overlap (midpoint split), the
            # actual position at the boundary differs from the true target because:
            #   - The latter half of L2 (outgoing) was NOT applied
            #   - The first half of L3 (incoming) WAS prematurely absorbed
            #
            # Load the correction offset and apply it:
            #   corrected_actual = actual + correction_offset
            #   error = ideal - corrected_actual
            #
            overlap_corr_mag_debug = 0.0  # Track for visualization
            overlap_correction_path = os.path.join(source_path, "overlap_correction.pkl")
            if os.path.exists(overlap_correction_path):
                try:
                    with open(overlap_correction_path, 'rb') as f:
                        correction_data = pickle.load(f)
                    
                    correction_offset = np.array(correction_data['overlap_correction_offset'])
                    overlap_corr_mag_debug = float(np.linalg.norm(correction_offset))
                    print(f"[OVERLAP CORRECTION] Loaded correction offset (magnitude: {overlap_corr_mag_debug:.6f})")
                    
                    # The correction offset is a flat vector of dimension N (e.g., 3603).
                    # Split it into the same tissue-group structure as actual_flat.
                    # Layout: OOR (900) + jaw (3) + OOC_L (900) + OOC_R (900) + Frontalis (900)
                    n_lm = 300 * 3  # 900
                    idx = 0
                    
                    oor_corr = correction_offset[idx:idx + n_lm]; idx += n_lm
                    jaw_corr = correction_offset[idx:idx + 3]; idx += 3
                    ooc_l_corr = correction_offset[idx:idx + n_lm]; idx += n_lm
                    ooc_r_corr = correction_offset[idx:idx + n_lm]; idx += n_lm
                    frontalis_corr = correction_offset[idx:idx + n_lm]; idx += n_lm
                    
                    # Apply correction: corrected_actual = actual + correction_offset
                    actual_flat['x_joint_conca'] = (np.array(actual_flat['x_joint_conca']) + oor_corr).tolist()
                    actual_flat['x_joint_rotation_conca'] = (np.array(actual_flat['x_joint_rotation_conca']) + jaw_corr).tolist()
                    actual_flat['x_ooc_l_joint_conca'] = (np.array(actual_flat['x_ooc_l_joint_conca']) + ooc_l_corr).tolist()
                    actual_flat['x_ooc_r_joint_conca'] = (np.array(actual_flat['x_ooc_r_joint_conca']) + ooc_r_corr).tolist()
                    actual_flat['x_frontalis_joint_conca'] = (np.array(actual_flat['x_frontalis_joint_conca']) + frontalis_corr).tolist()
                    
                    print(f"[OVERLAP CORRECTION] Applied correction to actual positions before error computation")
                    
                except Exception as e:
                    print(f"[OVERLAP CORRECTION] Error loading correction: {e}")
                    print(f"[OVERLAP CORRECTION] Proceeding without correction")
            
            new_error = compute_position_error(actual_flat, ideal_positions)
            
            # --- DEBUG: Collect debug data for this transition ---
            if DEBUG_ENABLED:
                # Compute landmark distances (actual vs ideal)
                lm_mean, lm_max, lm_total = compute_landmark_distance(
                    actual_flat['x_joint_conca'],
                    ideal_positions['x_joint_conca']
                )
                
                # Compute vertex distances if OOR tissue mesh exists
                oor_vertices_actual = retrieve_tissue_vertices('middleres_OrbicularisOris')
                v_mean, v_max, v_total = 0.0, 0.0, 0.0
                v_ground_mean, v_ground_max, v_ground_total = 0.0, 0.0, 0.0
                
                # Calculate error magnitudes
                error_mag = np.linalg.norm(new_error['x_joint_conca']) if len(new_error['x_joint_conca']) > 0 else 0.0
                
                # Calculate correction magnitude if correction was applied
                correction_mag = 0.0
                prev_error_mag = 0.0
                new_error_mag = error_mag
                residual_error_mag = 0.0
                
                if correction_vectors is not None and 'x_joint_conca' in correction_vectors:
                    correction_mag = np.linalg.norm(correction_vectors['x_joint_conca'])
                
                if ACCUMULATED_ERROR is not None and 'x_joint_conca' in ACCUMULATED_ERROR:
                    prev_error_mag = np.linalg.norm(ACCUMULATED_ERROR['x_joint_conca'])
                
                # Build debug entry
                debug_entry = {
                    'transition': len(DEBUG_DATA) + 1,
                    'phoneme_from': phoneme_from.replace('dataV22_', ''),
                    'phoneme_to': phoneme_to.replace('dataV22_', ''),
                    'frame': final_frame,
                    'landmark_mean_hat_tilde': lm_mean,
                    'landmark_max_hat_tilde': lm_max,
                    'landmark_total_hat_tilde': lm_total,
                    'vertex_mean_hat_tilde': v_mean,
                    'vertex_max_hat_tilde': v_max,
                    'vertex_total_hat_tilde': v_total,
                    'vertex_mean_hat_ground': v_ground_mean,
                    'vertex_max_hat_ground': v_ground_max,
                    'vertex_total_hat_ground': v_ground_total,
                    'error_magnitude': error_mag,
                    'accumulated_error_magnitude': prev_error_mag + error_mag,
                    'correction_magnitude': correction_mag,
                    'prev_error_magnitude': prev_error_mag,
                    'new_error_magnitude': new_error_mag,
                    'residual_error_magnitude': residual_error_mag,
                    'correction_applied': correction_vectors is not None,
                    'dissipation_rate': ERROR_DISSIPATION_RATE,
                    'overlap_correction_magnitude': overlap_corr_mag_debug
                }
                
                DEBUG_DATA.append(debug_entry)
                print(f"[DEBUG] Debug entry saved (transition #{len(DEBUG_DATA)})")
            
            # Update ACCUMULATED_ERROR logic (Residual + New)
            if ACCUMULATED_ERROR is not None:
                residual_error = {}
                for key in new_error.keys():
                    prev_err = np.array(ACCUMULATED_ERROR.get(key, []))
                    new_err_val = np.array(new_error[key])
                    # IMPORTANT: Since we applied (prev * rate) during the transition,
                    # the residual is prev * (1 - rate).
                    if len(prev_err) > 0:
                        residual_error[key] = (prev_err * (1.0 - ERROR_DISSIPATION_RATE)) + new_err_val
                    else:
                        residual_error[key] = new_err_val
                ACCUMULATED_ERROR = residual_error
            else:
                ACCUMULATED_ERROR = new_error
        else:
            # ideal_positions is None - cannot compute ideal target, but we should still
            # initialize ACCUMULATED_ERROR with zero error to prevent subsequent transitions from failing
            print(f"[FEEDBACK CONTROL WARNING] Could not load ideal positions for {phoneme_to}")
            print(f"[FEEDBACK CONTROL WARNING] Initializing zero error - subsequent corrections may not work properly")
            
            if ACCUMULATED_ERROR is None:
                # Initialize with zero errors so subsequent transitions can still function
                n = 300 * 3
                ACCUMULATED_ERROR = {
                    'x_joint_conca': np.zeros(n),
                    'x_joint_rotation_conca': np.zeros(3),
                    'x_ooc_l_joint_conca': np.zeros(n),
                    'x_ooc_r_joint_conca': np.zeros(n),
                    'x_frontalis_joint_conca': np.zeros(n)
                }
                print(f"[FEEDBACK CONTROL] Initialized ACCUMULATED_ERROR with zeros")

    return final_frame


def run_full_pipeline(*args):
    """
    This is the main function called by the "Go" button.
    It chains all your steps together.

    NEW BEHAVIOR: If the input contains multiple phonemes (3+), it will:
    1. Break them down into individual transitions
    2. Run each transition sequentially using existing transition files
    3. Chain them together automatically
    4. Apply feedback control loop to correct accumulated drift
    """
    # Declare global variables at the top of the function
    global LAST_ANIMATION_END_FRAME, LAST_ANIMATION_PHONEME, CONTINUE_FROM_LAST, ACCUMULATED_ERROR, PER_FRAME_DEBUG_HISTORY, LIVE_DEBUG_FRAMES, LIVE_DEBUG_POSITIONS

    import time
    pipeline_start_time = time.time()

    print("========= STARTING FULL PIPELINE =========")

    try:
        # --- Development Mode Logic ---
        if DEVELOPMENT_MODE:
            print("--- RUNNING IN DEVELOPMENT MODE ---")
            user_text = DEV_SEQUENCE_STRING
            cmds.textField('sequence_input_field', edit=True, text=user_text) # Update UI for visibility
        else:
            user_text = cmds.textField('sequence_input_field', query=True, text=True)

        if not user_text:
            cmds.warning("Input field is empty!")
            return

        # Parse the phonemes
        phonemes, _ = parse_sequence_string(user_text)

        # --- NEW LOGIC: Break down into individual transitions ---
        # Restored multi-phoneme transition support
        if len(phonemes) > 2:
            print(f"\n{'='*60}")
            print(f"MULTI-PHONEME SEQUENCE DETECTED: {len(phonemes)} phonemes")
            print(f"Breaking down into {len(phonemes)-1} individual transitions...")
            if ENABLE_FEEDBACK_CONTROL:
                print(f"FEEDBACK CONTROL: ENABLED (dissipation rate: {ERROR_DISSIPATION_RATE})")
            print(f"OVERLAP: {'ENABLED (' + str(int(OVERLAP_RATIO*100)) + '%)' if OVERLAP_RATIO > 0 else 'DISABLED (standard checkpoints)'}")
            print(f"{'='*60}\n")

            # Reset accumulated error and debug data if not continuing from last
            if not CONTINUE_FROM_LAST:
                ACCUMULATED_ERROR = None
                DEBUG_DATA.clear()
                # Also clear PER_FRAME_DEBUG_HISTORY to prevent old data from showing in plots
                PER_FRAME_DEBUG_HISTORY = []
                LIVE_DEBUG_FRAMES.clear()
                LIVE_DEBUG_POSITIONS.clear()
                print("[FEEDBACK CONTROL] Accumulated error reset for new sequence")
                print("[DEBUG] Debug data and per-frame history cleared\n")

            # --- CREATE PROXY RENDER MESHES IMMEDIATELY ---
            ensure_proxy_render_meshes()

            # First transition: Start from frame 14 (or current start frame if continuing)
            start_frame = LAST_ANIMATION_END_FRAME if CONTINUE_FROM_LAST else 14
            initial_start_frame = start_frame  # Save for timing metrics
            continue_mode = CONTINUE_FROM_LAST

            for i in range(len(phonemes) - 1):
                phoneme_from = phonemes[i]
                phoneme_to = phonemes[i + 1]

                print(f"\n[STEP {i+1}/{len(phonemes)-1}] Transition: {phoneme_from} -> {phoneme_to}")

                # Run this single transition with feedback control
                # First transition (i==0 and not continuing) is marked as is_first_transition=True
                is_first = (i == 0 and not CONTINUE_FROM_LAST)
                final_frame = run_single_transition(phoneme_from, phoneme_to, start_frame, continue_mode, is_first_transition=is_first)

                # Update state for next transition
                LAST_ANIMATION_END_FRAME = final_frame
                LAST_ANIMATION_PHONEME = phoneme_to
                start_frame = final_frame
                continue_mode = True  # All subsequent transitions are in continue mode

                print(f"[STATE UPDATE] Next transition will start from frame {final_frame}")

                # --- DYNAMIC HIDDEN SNAP ---
                # To prevent gravity/FEM drift, we reset the Ziva solver invisibly when 
                # the face returns to a perfect Neutral pose (like at periods or commas).
                if phoneme_to == "dataV22_AU0_neutral":
                    snap_frame = final_frame + 1
                    print(f"[PHYSICS] Character reached Neutral at frame {final_frame}. Inserting hidden snap at {snap_frame} to clear physics drift...")
                    insert_ziva_hidden_snap(snap_frame)
                    
                    # --- EQUILIBRIUM SETTLING BUFFER ---
                    # Because the snap reverts the tissue to the Frame 0 un-sagged state, 
                    # we must give the solver 15 frames of pure Neutral pose to let gravity 
                    # pull the tissue back down to its natural equilibrium
                    print(f"[PHYSICS] Adding 15-frame equilibrium settling buffer...")
                    
                    # Insert the smooth morphing blendshape using those 15 frames!
                    insert_smooth_snap_blendshape(snap_frame, buffer_frames=15)
                    
                    LAST_ANIMATION_END_FRAME = snap_frame + 15
                    start_frame = LAST_ANIMATION_END_FRAME
                    print(f"[PHYSICS] Next transition will now safely start from frame {start_frame} at full equilibrium.")

            pipeline_end_time = time.time()
            total_time = pipeline_end_time - pipeline_start_time
            total_frames_simulated = LAST_ANIMATION_END_FRAME - initial_start_frame
            time_per_frame = total_time / total_frames_simulated if total_frames_simulated > 0 else 0

            print(f"\n{'='*60}")
            print(f"ALL TRANSITIONS COMPLETE!")
            print(f"Total frames: {initial_start_frame} -> {LAST_ANIMATION_END_FRAME} ({total_frames_simulated} simulated)")
            print(f"Total simulation time: {total_time:.2f} seconds")
            print(f"Average time per frame: {time_per_frame:.2f} seconds/frame")
            print(f"Final phoneme: {LAST_ANIMATION_PHONEME}")
            print(f"{'='*60}\n")

            # --- ZIVA RELIABLE SIMULATION DEBUG RUN ---
            generate_ziva_debug_plot()

            # --- DEBUG: Save and visualize debug data ---
            if DEBUG_ENABLED:
                if len(DEBUG_DATA) > 0:
                    print(f"\n[DEBUG] Saving debug data ({len(DEBUG_DATA)} transitions)...")
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_filename = f"debug_errors_{timestamp}.csv"

                    try:
                        save_debug_csv(DEBUG_DATA, csv_filename)
                        print(f"[DEBUG] CSV saved successfully")
                    except Exception as e:
                        print(f"[DEBUG ERROR] Failed to save CSV: {e}")

                    try:
                        visualize_errors(DEBUG_DATA)
                        print(f"[DEBUG] Visualization saved successfully")
                    except Exception as e:
                        print(f"[DEBUG ERROR] Failed to save visualization: {e}")
                        import traceback
                        traceback.print_exc()

                    print(f"[DEBUG] Debug data saved to: {DEBUG_OUTPUT_DIR}")
                    print(f"[DEBUG] - CSV file: {csv_filename}")
                    print(f"[DEBUG] - Visualization: error_analysis_{timestamp}.png")
                else:
                    print(f"[DEBUG WARNING] No debug data collected (DEBUG_DATA is empty)")
            else:
                print(f"[DEBUG] Debug disabled (DEBUG_ENABLED = {DEBUG_ENABLED})")

            # --- Playback bounds adjustment ---
            cmds.playbackOptions(minTime=0)
            print("[PLAYBACK] Timeline bounds adjusted, but strictly avoiding backward jump to frame 0.")

            # --- Run Rendering ---
            run_maya_render()

            # --- Combine to Video ---
            final_video = combine_pngs_to_mp4()

            # --- Notify User ---
            if final_video:
                debug_msg = f"\n\nDebug data saved to:\n{DEBUG_OUTPUT_DIR}" if DEBUG_ENABLED else ""
                # cmds.confirmDialog(title='Complete!',
                #                    message=f"Multi-Phoneme Pipeline Finished!\n\n{len(phonemes)-1} transitions completed\n\nFrames: 14 to {LAST_ANIMATION_END_FRAME}\n\nFinal phoneme: {LAST_ANIMATION_PHONEME.replace('dataV22_', '')}\n\nFinal video (simulated):\n{final_video}{debug_msg}")
                print(f"Multi-Phoneme Pipeline Finished! Frames: 14 to {LAST_ANIMATION_END_FRAME}. Final video: {final_video}{debug_msg}")

        else:
            # --- ORIGINAL LOGIC: Python-based Overlap Blending ---
            print(f"\n[CLASSIC OVERLAP MODE] Running {phonemes[0]} -> {phonemes[-1]} ({len(phonemes)} phonemes)")

            sequence_folder_name = generate_short_sequence_name(user_text, phonemes)
            log_paths = parse_sequence_string(user_text)[1]

            # --- Conditional Processing ---
            # base_path = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\sequence_combined_deformation_parameters_hub"
            base_path = r"\\bbysvm-nfs1.its.sfu.ca\GrUViLab\Data\richardPeng\SFU_PhD_cloud\2-Projects\1-physically based facial animation\3-Experiments\a3-stochastic-V1-backup-and-results\results\sequence_combined_deformation_parameters_hub"
            checkpoint_file_path = os.path.join(base_path, sequence_folder_name, f"combined_checkpoints_base{base_frames_global}.pkl")

            should_run_processing = True
            if os.path.exists(checkpoint_file_path):
                print(f"Deformation file already exists at {checkpoint_file_path}. SKIPPING processing.")
                should_run_processing = False

            if should_run_processing:
                print("--- Running Deformation Processing ---")
                run_deformation_processing(phonemes, log_paths, sequence_folder_name)

            # --- Determine start frame and continue mode ---
            # (global variables already declared at the top of the function)
            start_frame = LAST_ANIMATION_END_FRAME if CONTINUE_FROM_LAST else 14
            continue_mode = CONTINUE_FROM_LAST

            print(f"\n=== ANIMATION CHAINING INFO ===")
            print(f"Continue from last: {continue_mode}")
            print(f"Start frame: {start_frame}")
            print(f"Last phoneme: {LAST_ANIMATION_PHONEME}")
            print(f"New sequence: {phonemes[0]} -> {phonemes[-1]}")
            print(f"===============================\n")

            # --- Perform the Deformation on the Rig ---
            full_source_path = os.path.join(base_path, sequence_folder_name)
            final_frame, _ = perform_the_deformation(full_source_path, start_frame, continue_mode)

            # --- Update global state for next animation ---
            LAST_ANIMATION_END_FRAME = final_frame
            LAST_ANIMATION_PHONEME = phonemes[-1]  # Store the last phoneme
            print(f"\n[CHAINING] Animation complete. Final frame: {final_frame}")
            print(f"[CHAINING] Last phoneme: {LAST_ANIMATION_PHONEME}")
            print(f"[CHAINING] Ready for next animation from frame {final_frame}\n")

            # --- Run Rendering ---
            run_maya_render()

            # --- Combine to Video ---
            final_video = combine_pngs_to_mp4()

            # --- Notify User ---
            if final_video:
                # cmds.confirmDialog(title='Complete!',
                #                    message=f"Pipeline Finished!\n\nDeformation data in:\n{sequence_folder_name}\n\nFrames: {start_frame} to {final_frame}\n\nFinal video (simulated):\n{final_video}\n\nNext animation will start from frame {final_frame}")
                print(f"Pipeline Finished! Frames: {start_frame} to {final_frame}. Final video: {final_video}")

    except Exception as e:
        # Catch any errors from parsing or processing
        print(f"ERROR: {e}")
        # cmds.confirmDialog(title='Pipeline Error',
        #                    message=f"An error occurred:\n\n{e}")

    if cmds.checkBox('auto_playblast_checkbox', exists=True) and cmds.checkBox('auto_playblast_checkbox', query=True, value=True):
        run_playblast_script()

    print("========= PIPELINE FINISHED =========")


def toggle_continue_mode(*args):
    """Toggle the CONTINUE_FROM_LAST global variable based on checkbox"""
    global CONTINUE_FROM_LAST
    CONTINUE_FROM_LAST = cmds.checkBox('continue_checkbox', query=True, value=True)
    print(f"[UI] Continue from last animation: {CONTINUE_FROM_LAST}")

def toggle_feedback_control(*args):
    """Toggle the ENABLE_FEEDBACK_CONTROL global variable based on checkbox"""
    global ENABLE_FEEDBACK_CONTROL
    ENABLE_FEEDBACK_CONTROL = cmds.checkBox('feedback_checkbox', query=True, value=True)
    print(f"[UI] Feedback control enabled: {ENABLE_FEEDBACK_CONTROL}")

def update_dissipation_rate(*args):
    """Update the ERROR_DISSIPATION_RATE from slider"""
    global ERROR_DISSIPATION_RATE
    ERROR_DISSIPATION_RATE = cmds.floatSliderGrp('dissipation_slider', query=True, value=True)
    print(f"[UI] Error dissipation rate: {ERROR_DISSIPATION_RATE}")

def reset_animation_chain(*args):
    """Reset animation chaining to start from neutral and reopen rest pose scene"""
    global LAST_ANIMATION_END_FRAME, LAST_ANIMATION_PHONEME, CONTINUE_FROM_LAST, ACCUMULATED_ERROR, DEBUG_DATA

    # Reset global variables
    LAST_ANIMATION_END_FRAME = 14
    LAST_ANIMATION_PHONEME = "dataV22_AU0_neutral"
    CONTINUE_FROM_LAST = False
    ACCUMULATED_ERROR = None  # Reset feedback control error
    DEBUG_DATA.clear()  # Clear debug data
    cmds.checkBox('continue_checkbox', edit=True, value=False)

    # Reopen the rest pose scene
    # rest_pose_scene = r"C:\keyLabLocal_D\c99-maya-ziva-workshop\different pose as rest pose\g10-10 rest pose 1200 landmarks v9.mb"
    rest_pose_scene = r"C:\keyLabLocal_D\c99-maya-ziva-workshop\different pose as rest pose\Neutral V7.5-2 frame 0 and 14 have keys inproved right eye muscles.mb"
    try:
        cmds.file(rest_pose_scene, open=True, force=True)
        print(f"[UI] Animation chain reset and rest pose scene reopened: {rest_pose_scene}")
        cmds.confirmDialog(title='Reset', message='Animation chain reset!\n\nRest pose scene reopened.\nNext animation will start from frame 14 (neutral pose).')
    except Exception as e:
        print(f"[ERROR] Failed to open rest pose scene: {e}")
        cmds.confirmDialog(title='Reset Error', message=f'Animation chain reset, but failed to open scene:\n{rest_pose_scene}\n\nError: {e}')

# --- KEYBOARD MODE FUNCTIONS ---

def keyboard_transition_to_phoneme(target_key):
    """
    Triggered when a mapped key is pressed. Performs transition to the target phoneme.
    
    Args:
        target_key: The key that was pressed (e.g., '1', '2', etc.)
    """
    global KEYBOARD_CURRENT_PHONEME, LAST_ANIMATION_END_FRAME, LAST_ANIMATION_PHONEME
    
    if not KEYBOARD_MODE_ENABLED:
        print(f"[KEYBOARD] Keyboard mode is disabled")
        return
    
    if target_key not in KEYBOARD_PHONEME_MAP:
        print(f"[KEYBOARD] Unknown key: {target_key}")
        return
    
    target_info = KEYBOARD_PHONEME_MAP[target_key]
    target_phoneme = target_info['phoneme']
    target_label = target_info['label']
    
    # Check if already at target phoneme
    if KEYBOARD_CURRENT_PHONEME == target_phoneme:
        print(f"[KEYBOARD] Already at {target_label}")
        return
    
    # Get current phoneme label for display
    current_label = PHONEME_TO_KEY.get(KEYBOARD_CURRENT_PHONEME, {}).get('label', 'Unknown')
    
    print(f"\n{'='*50}")
    print(f"[KEYBOARD] Key '{target_key}' pressed: {current_label} -> {target_label}")
    print(f"{'='*50}")
    
    try:
        # Perform the transition
        start_frame = LAST_ANIMATION_END_FRAME
        
        # Run the transition (reusing existing function)
        final_frame = smart_single_transition(
            KEYBOARD_CURRENT_PHONEME, 
            target_phoneme, 
            start_frame, 
            is_first_transition=False
        )
        
        # Update state
        KEYBOARD_CURRENT_PHONEME = target_phoneme
        LAST_ANIMATION_END_FRAME = final_frame
        LAST_ANIMATION_PHONEME = target_phoneme
        
        # Update UI display
        update_keyboard_status_display()
        
        print(f"[KEYBOARD] Transition complete. Now at {target_label} (frame {final_frame})")
        
    except Exception as e:
        print(f"[KEYBOARD ERROR] Failed to transition: {e}")
        import traceback
        traceback.print_exc()


def create_keyboard_command(key):
    """Create a command function for a specific key"""
    def command_func():
        keyboard_transition_to_phoneme(key)
    return command_func


def setup_keyboard_hotkeys():
    """Set up keyboard mode - now uses text field approach instead of hotkeys"""
    print("[KEYBOARD] Keyboard mode ready!")
    print("[KEYBOARD] Click on the input field and press keys to trigger transitions:")
    for key, info in KEYBOARD_PHONEME_MAP.items():
        print(f"  '{key}' = {info['label']}")


def cleanup_keyboard_hotkeys():
    """Cleanup keyboard mode resources"""
    global KEYBOARD_SCRIPT_JOBS
    
    # Clear any script jobs
    for job_id in KEYBOARD_SCRIPT_JOBS:
        try:
            if cmds.scriptJob(exists=job_id):
                cmds.scriptJob(kill=job_id, force=True)
        except:
            pass
    
    KEYBOARD_SCRIPT_JOBS = []
    print("[KEYBOARD] Keyboard mode disabled")


def on_keyboard_input_changed(*args):
    """
    Called when the keyboard input field changes.
    Processes the last character typed and triggers the appropriate transition.
    """
    global KEYBOARD_CURRENT_PHONEME
    
    if not KEYBOARD_MODE_ENABLED:
        return
    
    try:
        # Get the current text in the field
        current_text = cmds.textField('keyboard_input_field', query=True, text=True)
        
        if not current_text:
            return
        
        # Get the last character typed (lowercase for matching)
        last_char = current_text[-1].lower()
        
        # Clear the field immediately for next input
        cmds.textField('keyboard_input_field', edit=True, text='')
        
        # Check if this key is mapped
        if last_char in KEYBOARD_PHONEME_MAP:
            keyboard_transition_to_phoneme(last_char)
        else:
            print(f"[KEYBOARD] Unknown key: '{last_char}' - Valid keys: {', '.join(KEYBOARD_PHONEME_MAP.keys())}")
            
    except Exception as e:
        print(f"[KEYBOARD ERROR] Input processing failed: {e}")


def toggle_keyboard_mode(*args):
    """Toggle keyboard mode on/off"""
    global KEYBOARD_MODE_ENABLED, KEYBOARD_CURRENT_PHONEME
    
    KEYBOARD_MODE_ENABLED = cmds.checkBox('keyboard_mode_checkbox', query=True, value=True)
    
    if KEYBOARD_MODE_ENABLED:
        # Reset to current state
        KEYBOARD_CURRENT_PHONEME = LAST_ANIMATION_PHONEME
        setup_keyboard_hotkeys()
        update_keyboard_status_display()
        
        # Focus on the keyboard input field
        try:
            cmds.setFocus('keyboard_input_field')
        except:
            pass
        
        print(f"[KEYBOARD] Keyboard mode ENABLED")
        print(f"[KEYBOARD] Click on the input field below and type keys!")
        print(f"[KEYBOARD] Current phoneme: {PHONEME_TO_KEY.get(KEYBOARD_CURRENT_PHONEME, {}).get('label', 'Unknown')}")
    else:
        cleanup_keyboard_hotkeys()
        print(f"[KEYBOARD] Keyboard mode DISABLED")


def update_keyboard_status_display():
    """Update the keyboard status display in the UI"""
    try:
        if cmds.text('keyboard_status_text', exists=True):
            current_label = PHONEME_TO_KEY.get(KEYBOARD_CURRENT_PHONEME, {}).get('label', 'Unknown')
            current_key = PHONEME_TO_KEY.get(KEYBOARD_CURRENT_PHONEME, {}).get('key', '?')
            cmds.text('keyboard_status_text', edit=True, 
                     label=f"Current: [{current_key}] {current_label}  |  Frame: {LAST_ANIMATION_END_FRAME}")
    except:
        pass


def ensure_proxy_render_meshes():
    """Ensure _RENDER proxies exist. Must jump to Frame 0 to duplicate un-deformed Neutral mesh."""
    valid_meshes = [m for m in SOFT_TISSUE_MESHES if cmds.objExists(m)]
    needs_proxy = any(not cmds.objExists(f"{m}_RENDER") for m in valid_meshes)
    
    if needs_proxy:
        print(f"[INITIALIZATION] Jumping to Frame 0 to safely capture perfect Neutral proxy meshes...")
        current_time = cmds.currentTime(query=True)
        cmds.currentTime(0, edit=True)
        
        print(f"[INITIALIZATION] Creating Proxy Render Meshes for {len(valid_meshes)} soft tissues...")
        for target_mesh in valid_meshes:
            render_mesh = f"{target_mesh}_RENDER"
            bs_name = f"master_bs_{target_mesh}"
            if not cmds.objExists(render_mesh):
                render_mesh_node = cmds.duplicate(target_mesh, name=render_mesh)[0]
                
                cmds.setAttr(f"{target_mesh}.visibility", 0)
                cmds.setAttr(f"{render_mesh_node}.visibility", 1)
                
                # Link it to follow the live Ziva mesh 100% by default
                cmds.blendShape([target_mesh], render_mesh_node, name=bs_name, origin='local')
                cmds.setAttr(f"{bs_name}.{target_mesh}", 1.0)
                
        if current_time > 0:
            print(f"[INITIALIZATION] Re-simulating physics from 0 to {current_time} to restore cache...")
            for f in range(1, int(current_time) + 1):
                cmds.currentTime(f, edit=True)
        else:
            cmds.currentTime(current_time, edit=True)

def generate_ziva_debug_plot():
    """Generate the Ziva Live Debug Visualization from accumulated data."""
    print("\n" + "="*60)
    print("RUNNING ZIVA RELIABLE DEBUG VISUALIZATION (Plotting Pre-collected Live Data)")
    print("="*60)
    try:
        import matplotlib.pyplot as plt
        import os
        import datetime
        
        target_mesh = "middleres_OrbicularisOris_RENDER"
        if not cmds.objExists(target_mesh):
            target_mesh = "middleres_OrbicularisOris"
        vertex_idx = 0
        
        if LIVE_DEBUG_FRAMES:
            plt.figure(figsize=(15, 6))
            plt.plot(LIVE_DEBUG_FRAMES, LIVE_DEBUG_POSITIONS, marker='.', linestyle='-', color='b', markersize=2, label=f'Vertex Y Position ({target_mesh})')
            plt.title(f"Ziva Live Sequential Debug: {target_mesh} Vertex {vertex_idx}")
            plt.xlabel("Frame")
            plt.ylabel("World Y Position")
            plt.grid(True)
            plt.legend()
            
            debug_dir = r"C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\debug_results"
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(debug_dir, f"FULL_vertex_morph_plot_{timestamp_str}.png")
            plt.savefig(plot_path)
            print(f"\n[DEBUG] Successfully saved reliable sequential plot to: {plot_path}")
            plt.close()
            
            csv_path = os.path.join(debug_dir, f"FULL_vertex_morph_data_{timestamp_str}.csv")
            with open(csv_path, "w") as f:
                f.write("frame,vertex_y\n")
                for i in range(len(LIVE_DEBUG_FRAMES)):
                    f.write(f"{LIVE_DEBUG_FRAMES[i]},{LIVE_DEBUG_POSITIONS[i]:.6f}\n")
            print(f"[DEBUG] Successfully saved raw live stepping data to: {csv_path}")
        else:
            print("[DEBUG] Warning: No live data was collected. Did the timeline step?")
    except Exception as e:
        print(f"[DEBUG] Failed to run sequential debug plot: {e}")

def apply_neutral_hidden_snap(final_frame):
    """
    Applies the hidden snap and 15-frame equilibrium buffer to clear physics drift.
    """
    snap_frame = final_frame + 1
    print(f"[PHYSICS] Character reached Neutral at frame {final_frame}. Inserting hidden snap at {snap_frame} to clear physics drift...")
    
    print(f"[PHYSICS] Adding 15-frame equilibrium settling buffer...")
    insert_smooth_snap_blendshape(snap_frame, buffer_frames=15)
    
    return snap_frame + 15

# --- SMART TRANSITION HELPER ---
def smart_single_transition(current_pose, target_pose, start_frame, is_first_transition=False):
    """
    Attempts to transition from current to target_pose.
    If the direct path doesn't exist, it falls back to current -> neutral -> target_pose.
    """
    ensure_proxy_render_meshes()
    
    task_string = f"{current_pose} to {target_pose}"
    normalized_key = normalize_task_key(task_string)
    db_key_map = {normalize_task_key(k): k for k in DEFORMATION_PATH_DB}
    
    neutral = "dataV22_AU0_neutral"
    
    # Check if path exists or if it's already a transition involving neutral
    if normalized_key in db_key_map or current_pose == neutral or target_pose == neutral:
        final_frame = run_single_transition(current_pose, target_pose, start_frame, continue_mode=True, is_first_transition=is_first_transition)
        if target_pose == neutral:
            final_frame = apply_neutral_hidden_snap(final_frame)
            
        generate_ziva_debug_plot()
        return final_frame
    
    # Path doesn't exist, and neither is neutral. Do fallback.
    c_short = current_pose.replace('dataV22_', '')
    t_short = target_pose.replace('dataV22_', '')
    
    print(f"\n[WARNING] Path '{c_short} -> {t_short}' does not exist!")
    print(f"[INFO] Routing through Neutral pose: {c_short} -> Neutral -> {t_short}")
    
    # Step 1: current to neutral
    intermediate_frame = run_single_transition(
        current_pose, neutral, start_frame,
        continue_mode=True, is_first_transition=is_first_transition
    )
    
    intermediate_frame = apply_neutral_hidden_snap(intermediate_frame)
    
    # Step 2: neutral to target
    final_frame = run_single_transition(
        neutral, target_pose, intermediate_frame,
        continue_mode=True, is_first_transition=False
    )
    
    generate_ziva_debug_plot()
    return final_frame

# --- DROPDOWN CALLBACK FUNCTIONS (MetaHuman-style: immediate deformation on click) ---

def dropdown_immediate_transition(target_phoneme, label_text):
    """
    Immediately deform from the current pose to the target pose.
    This is the MetaHuman-style behavior: click = instant transition.
    """
    global LAST_ANIMATION_END_FRAME, LAST_ANIMATION_PHONEME, KEYBOARD_CURRENT_PHONEME

    current = LAST_ANIMATION_PHONEME
    if current == target_phoneme:
        print(f"[CLICK] Already at {label_text}")
        return

    current_short = current.replace('dataV22_', '')
    target_short = target_phoneme.replace('dataV22_', '')
    print(f"\n{'='*50}")
    print(f"[CLICK] {current_short} -> {target_short}")
    print(f"{'='*50}")

    try:
        start_frame = LAST_ANIMATION_END_FRAME
        final_frame = smart_single_transition(
            current, target_phoneme, start_frame,
            is_first_transition=False
        )
        LAST_ANIMATION_END_FRAME = final_frame
        LAST_ANIMATION_PHONEME = target_phoneme
        KEYBOARD_CURRENT_PHONEME = target_phoneme
        update_status_display()
        print(f"[CLICK] Done. Now at {target_short} (frame {final_frame})")
        if cmds.checkBox('auto_playblast_checkbox', exists=True) and cmds.checkBox('auto_playblast_checkbox', query=True, value=True):
            run_playblast_script()
    except Exception as e:
        print(f"[CLICK ERROR] {e}")
        import traceback
        traceback.print_exc()


def dropdown_immediate_expression(command_key):
    """Run a multi-step expression command (e.g. 'raise the brow' = AU1 then AU2)."""
    global LAST_ANIMATION_END_FRAME, LAST_ANIMATION_PHONEME, KEYBOARD_CURRENT_PHONEME

    if command_key not in EXPRESSION_COMMANDS:
        print(f"[CLICK ERROR] Unknown expression: {command_key}")
        return

    sequence = EXPRESSION_COMMANDS[command_key]
    # Skip the neutral prefix if we're already past it -- start from current pose
    current = LAST_ANIMATION_PHONEME
    # If the expression starts with neutral and we're already at neutral, skip it
    targets = list(sequence)
    if targets and targets[0] == current:
        targets = targets[1:]
    if not targets:
        print(f"[CLICK] Already at expression '{command_key}'")
        return

    print(f"\n{'='*50}")
    print(f"[EXPRESSION] '{command_key}' ({len(targets)} steps)")
    print(f"{'='*50}")

    try:
        for t in targets:
            t_short = t.replace('dataV22_', '')
            c_short = LAST_ANIMATION_PHONEME.replace('dataV22_', '')
            if t == LAST_ANIMATION_PHONEME:
                continue
            print(f"  [{c_short} -> {t_short}]")
            start_frame = LAST_ANIMATION_END_FRAME
            final_frame = smart_single_transition(
                LAST_ANIMATION_PHONEME, t, start_frame,
                is_first_transition=False
            )
            LAST_ANIMATION_END_FRAME = final_frame
            LAST_ANIMATION_PHONEME = t
            KEYBOARD_CURRENT_PHONEME = t
        update_status_display()
        print(f"[EXPRESSION] Done: '{command_key}'")
        if cmds.checkBox('auto_playblast_checkbox', exists=True) and cmds.checkBox('auto_playblast_checkbox', query=True, value=True):
            run_playblast_script()
    except Exception as e:
        print(f"[EXPRESSION ERROR] {e}")
        import traceback
        traceback.print_exc()


def update_status_display():
    """Update the status bar in the UI."""
    try:
        if cmds.text('status_bar_text', exists=True):
            short = LAST_ANIMATION_PHONEME.replace('dataV22_', '').replace('_', ' ')
            cmds.text('status_bar_text', edit=True,
                      label=f"  Current Pose: {short}   |   Frame: {LAST_ANIMATION_END_FRAME}  ")
    except:
        pass
    # Also update keyboard status if it exists
    try:
        if cmds.text('keyboard_status_text', exists=True):
            short = LAST_ANIMATION_PHONEME.replace('dataV22_', '')
            cmds.text('keyboard_status_text', edit=True,
                      label=f"Current: {short}  |  Frame: {LAST_ANIMATION_END_FRAME}")
    except:
        pass


def make_custom_dropdown(button_id, items, callback, bg_color):
    """Creates a button with a popupMenu to bypass Maya's optionMenu color bug."""
    default_label = items[0][0]
    btn = cmds.button(button_id, label=default_label, backgroundColor=bg_color, height=26)
    pop = cmds.popupMenu(parent=btn, button=1)
    
    for label_text, internal_val in items:
        # We must use default arguments in the lambda to capture loop variables safely
        def _cmd(x, l=label_text, v=internal_val, b=button_id, c=callback):
            cmds.button(b, edit=True, label=l)
            c(l, v)
        cmds.menuItem(parent=pop, label=label_text, command=_cmd)
    return btn


def handle_phoneme_selection(label, internal):
    if internal:
        dropdown_immediate_transition(internal, label)


def handle_facs_selection(label, internal):
    if internal:
        dropdown_immediate_transition(internal, label)


def handle_emotion_selection(label, command_key):
    if command_key:
        dropdown_immediate_expression(command_key)


def toggle_verbose_mode(*args):
    global VERBOSE_MODE
    VERBOSE_MODE = cmds.checkBox('verbose_mode_checkbox', query=True, value=True)
    _original_print(f"[UI] Verbose mode: {VERBOSE_MODE}")


def initialize_simulation_action(*args):
    """Initializes the Ziva simulation physics by running frame 0 to 14."""
    _original_print("\n--- INITIALIZING ZIVA PHYSICS ---")
    _original_print("Evaluating physics to Neutral (Frame 14). Please wait...")
    
    # CRITICAL TRADE-OFF: full quality (substeps=2, maxNewton=5) causes 5-minute freezes.
    # Interactive speed (substeps=1, maxNewton=1) causes "crumpled newspaper" mesh explosions.
    # We enforce a "Middle Ground" just for initialization to prevent both!
    solvers = cmds.ls(type='zSolver')
    old_settings = {}
    if solvers:
        for solver in solvers:
            try:
                sub = cmds.getAttr(f"{solver}.substeps")
                newton = cmds.getAttr(f"{solver}.maxNewtonIterations")
                old_settings[solver] = (sub, newton)
                cmds.setAttr(f"{solver}.substeps", 1)
                cmds.setAttr(f"{solver}.maxNewtonIterations", 2)
            except: pass
    
    # Actually run the timeline to solve physics to frame 14
    step_timeline_forward(14)
    
    # Restore user's previous settings
    if solvers:
        for solver, (sub, newton) in old_settings.items():
            try:
                cmds.setAttr(f"{solver}.substeps", sub)
                cmds.setAttr(f"{solver}.maxNewtonIterations", newton)
            except: pass
    
    global LAST_ANIMATION_END_FRAME, LAST_ANIMATION_PHONEME
    LAST_ANIMATION_END_FRAME = 14
    LAST_ANIMATION_PHONEME = "dataV22_AU0_neutral"
    
    update_keyboard_status_display()
    _original_print("--- INITIALIZATION COMPLETE ---")
    if cmds.text('status_bar_text', exists=True):
        cmds.text('status_bar_text', edit=True, label=f'  Current Pose: Neutral   |   Frame: 14  ')


def precompute_keyboard_transitions(*args):
    """
    Precomputes all combined_checkpoints_base{base_frames_global}.pkl files required for keyboard transitions.
    This runs in the background (without playing the timeline) so users don't 
    experience any delay when hitting the keys for the first time.
    """
    print("\n" + "="*50)
    print("--- PRECOMPUTING KEYBOARD TRANSITIONS ---")
    print("="*50)
    
    phonemes_list = [info['phoneme'] for key, info in KEYBOARD_PHONEME_MAP.items()]
    neutral = "dataV22_AU0_neutral"
    
    transitions_to_process = []
    db_key_map = {normalize_task_key(k): k for k in DEFORMATION_PATH_DB}
    
    for p1 in phonemes_list:
        for p2 in phonemes_list:
            if p1 == p2: continue
            
            task_string = f"{p1} to {p2}"
            norm_key = normalize_task_key(task_string)
            
            if norm_key in db_key_map:
                transitions_to_process.append((p1, p2))
            else:
                # smart_single_transition will route through neutral
                if p1 != neutral:
                    transitions_to_process.append((p1, neutral))
                if p2 != neutral:
                    transitions_to_process.append((neutral, p2))
                    
    # Remove duplicates
    transitions_to_process = list(set(transitions_to_process))
    total_transitions = len(transitions_to_process)
    _original_print(f"Found {total_transitions} unique transitions to pre-compute.")
    
    # Create a progress window that user can pause/cancel
    cmds.progressWindow(title='Precomputing Cache',
                        progress=0,
                        status='Precomputing: 0%',
                        isInterruptable=True,
                        maxValue=total_transitions)
    
    paused = False
    for idx, (p_from, p_to) in enumerate(transitions_to_process):
        # Check if the user clicked "Cancel" in the progress window to Pause it
        if cmds.progressWindow(query=True, isCancelled=True):
            _original_print("--- PRECOMPUTATION PAUSED BY USER ---")
            _original_print("You can resume later by clicking the button again.")
            paused = True
            break
            
        try:
            from_name = p_from.replace('dataV22_', '').replace('AU0_neutral', 'neutral')
            to_name = p_to.replace('dataV22_', '')
            user_text = f"{from_name} to {to_name}"
            
            cmds.progressWindow(edit=True, progress=idx, status=f'Processing: {user_text}')
            
            phonemes, log_paths = parse_sequence_string(user_text)
            source_path = log_paths[0] if log_paths else None
            
            if source_path is None:
                continue
                
            checkpoint_file_path = os.path.join(source_path, f"combined_checkpoints_base{base_frames_global}.pkl")
            transition_frames = get_transition_frames(p_from, p_to, base_frames=base_frames_global)
            num_checkpoints = transition_frames + 1
            
            generate_new = False
            if os.path.exists(checkpoint_file_path):
                try:
                    with open(checkpoint_file_path, 'rb') as f:
                        existing_checkpoints = pickle.load(f)
                    if len(existing_checkpoints) != num_checkpoints:
                        generate_new = True
                except:
                    generate_new = True
            else:
                generate_new = True
                
            if generate_new:
                _original_print(f"\n[PRECOMPUTE {idx+1}/{total_transitions}] Generating {user_text} -> {num_checkpoints} frames")
                run_deformation_processing_to_source(phonemes, log_paths, source_path, num_checkpoints=num_checkpoints)
            else:
                _original_print(f"[PRECOMPUTE {idx+1}/{total_transitions}] Already cached: {user_text}")
                
        except Exception as e:
            _original_print(f"[PRECOMPUTE ERROR] Failed on {p_from} -> {p_to}: {e}")
            
    cmds.progressWindow(endProgress=True)
            
    if not paused:
        _original_print("\n" + "="*50)
        _original_print("--- PRECOMPUTATION COMPLETE! ---")
        _original_print("All keyboard keys will now trigger instantly.")
        _original_print("="*50 + "\n")
        if cmds.text('keyboard_status_text', exists=True):
            cmds.text('keyboard_status_text', edit=True, label="Precomputation Complete! Ready.")
    else:
        if cmds.text('keyboard_status_text', exists=True):
            cmds.text('keyboard_status_text', edit=True, label="Precomputation Paused.")


def create_automation_ui():
    """
    Builds a polished, artist-friendly UI window inspired by MetaHuman.
    Primary: Click-to-deform dropdowns (immediate transition on selection).
    Secondary: Text input for sentences / machine language (collapsed).
    """
    window_id = 'expression_automator_ui'

    if cmds.window(window_id, exists=True):
        cmds.deleteUI(window_id)

    # --- Color palette ---
    bg_main   = [1.00, 1.00, 1.00]   # Window background (White)
    bg_header = [0.92, 0.95, 0.99]   # Section header (Pastel blue)
    bg_status = [1.00, 0.92, 0.95]   # Status bar (Pastel pink)
    bg_field  = [0.98, 0.98, 0.98]   # Input field (Light grey)
    bg_dropdown = [1.00, 0.92, 0.95] # Pastel pink to match status bar
    btn_go    = [0.95, 0.25, 0.45]   # Pink/Magenta button
    btn_reset = [0.45, 0.65, 0.95]   # Light blue button
    accent    = [0.90, 0.94, 0.98]   # Pastel blue accent

    cmds.window(window_id, title='Facial Expression Automator', widthHeight=(520, 580),
                backgroundColor=bg_main)

    main_scroll = cmds.scrollLayout(childResizable=True, backgroundColor=bg_main)
    main_col = cmds.columnLayout(adjustableColumn=True, rowSpacing=2,
                                 columnOffset=('both', 8), backgroundColor=bg_main)

    # ==========================================================
    #  HEADER / TITLE BAR
    # ==========================================================
    cmds.separator(height=4, style='none')
    cmds.text(label='  Facial Expression Automator  ',
              font='boldLabelFont', align='center',
              backgroundColor=bg_header, height=28)
    cmds.text(label='Click any item below to deform instantly',
              align='center', font='smallPlainLabelFont',
              backgroundColor=bg_header)
    cmds.separator(height=6, style='none')

    global LAST_ANIMATION_END_FRAME, LAST_ANIMATION_PHONEME
    
    cmds.button(label='Initialize Simulation (Run Physics to Frame 14)', height=30,
                backgroundColor=[0.25, 0.65, 0.45], command=initialize_simulation_action)
    cmds.separator(height=6, style='none')

    # ==========================================================
    #  STATUS BAR
    # ==========================================================
    cmds.text('status_bar_text',
              label=f'  Current Pose: Neutral   |   Frame: {LAST_ANIMATION_END_FRAME}  ',
              align='center', font='boldLabelFont', height=26,
              backgroundColor=bg_status)
    cmds.separator(height=6, style='none')

    # ==========================================================
    #  SECTION 1: PHONEMES (click = instant deformation)
    # ==========================================================
    cmds.text(label='  Phonemes', align='left', font='boldLabelFont',
              backgroundColor=bg_header, height=22)
    cmds.separator(height=2, style='none')

    cmds.rowLayout(numberOfColumns=2, columnWidth2=(252, 252),
                   columnAttach=[(1, 'both', 2), (2, 'both', 2)])
    # Left: phoneme dropdown
    cmds.columnLayout(adjustableColumn=True)
    make_custom_dropdown('phoneme_btn', PHONEME_DROPDOWN_ITEMS, handle_phoneme_selection, bg_dropdown)
    cmds.setParent('..')
    # Right: FACS dropdown
    cmds.columnLayout(adjustableColumn=True)
    make_custom_dropdown('facs_btn', FACS_DROPDOWN_ITEMS, handle_facs_selection, bg_dropdown)
    cmds.setParent('..')
    cmds.setParent('..')  # end row

    cmds.separator(height=4, style='none')

    # ==========================================================
    #  SECTION 2: EXPRESSIONS / EMOTIONS
    # ==========================================================
    cmds.text(label='  Expressions / Emotions', align='left', font='boldLabelFont',
              backgroundColor=bg_header, height=22)
    cmds.separator(height=2, style='none')

    make_custom_dropdown('emotion_btn', EMOTION_DROPDOWN_ITEMS, handle_emotion_selection, bg_dropdown)

    cmds.separator(height=6, style='none')

    # ==========================================================
    #  RESET BUTTON
    # ==========================================================
    cmds.button(label='Reset to Neutral (Reopen Rest Pose)', height=32,
                backgroundColor=btn_reset, command=reset_animation_chain)
    cmds.separator(height=4, style='none')

    # ==========================================================
    #  SECTION 3: TYPE-IN MODE (collapsed for power users)
    # ==========================================================
    cmds.frameLayout(label='  Type-In Mode', collapsable=True, collapse=True,
                     borderStyle='etchedIn', marginWidth=8, marginHeight=6,
                     backgroundColor=bg_main)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=4)

    cmds.text(label='Auto-detects 3 formats:', font='smallBoldLabelFont', align='left')
    cmds.text(label='  Machine:    Neutral to phoneme_N to phoneme_AA',
              align='left', font='smallPlainLabelFont')
    cmds.text(label='  Sentence:   not even you  (auto-converted to phonemes)',
              align='left', font='smallPlainLabelFont')
    cmds.text(label='  Expression: smile / open the mouth / raise the brow',
              align='left', font='smallPlainLabelFont')
    cmds.separator(height=4, style='none')

    cmds.textField('sequence_input_field',
                   placeholderText='e.g.  not even you   OR   smile   OR   neutral to phoneme_Y',
                   backgroundColor=bg_field, height=26)
    cmds.separator(height=2, style='none')
    cmds.button(label='Generate from Text', height=28,
                backgroundColor=btn_go, command=run_full_pipeline)

    cmds.setParent('..')
    cmds.setParent('..')

    # ==========================================================
    #  SECTION 4: OPTIONS (collapsed)
    # ==========================================================
    cmds.frameLayout(label='  Options', collapsable=True, collapse=True,
                     borderStyle='etchedIn', marginWidth=8, marginHeight=6,
                     backgroundColor=bg_main)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=4)

    cmds.checkBox('continue_checkbox',
                  label='Continue from last animation (chain)',
                  value=CONTINUE_FROM_LAST,
                  changeCommand=toggle_continue_mode)

    cmds.text(label=f'Last: Frame {LAST_ANIMATION_END_FRAME}, '
              f'Pose: {LAST_ANIMATION_PHONEME.replace("dataV22_", "")}',
              align='left', font='smallPlainLabelFont')

    cmds.separator(height=4, style='none')
    cmds.text(label='Feedback Control:', align='left', font='smallBoldLabelFont')

    cmds.checkBox('feedback_checkbox',
                  label='Enable feedback control (correct drift)',
                  value=ENABLE_FEEDBACK_CONTROL,
                  changeCommand=toggle_feedback_control)

    cmds.floatSliderGrp('dissipation_slider',
                       label='Dissipation:',
                       field=True,
                       minValue=0.0, maxValue=1.0,
                       value=ERROR_DISSIPATION_RATE,
                       step=0.05,
                       changeCommand=update_dissipation_rate,
                       columnWidth3=[80, 50, 200])

    cmds.separator(height=4, style='none')
    cmds.checkBox('verbose_mode_checkbox',
                  label='Verbose Output (Development Mode)',
                  value=VERBOSE_MODE,
                  changeCommand=toggle_verbose_mode)

    cmds.separator(height=4, style='none')
    cmds.text(label='Playblast:', align='left', font='smallBoldLabelFont')

    cmds.checkBox('auto_playblast_checkbox',
                  label='Auto-run playblast after deformation',
                  value=False)
                  
    cmds.button(label='Run Playblast Now', height=24,
                backgroundColor=[0.45, 0.75, 0.95], command=lambda x: run_playblast_script())

    cmds.setParent('..')
    cmds.setParent('..')

    # ==========================================================
    #  SECTION 5: KEYBOARD MODE (collapsed)
    # ==========================================================
    cmds.frameLayout(label='  Keyboard Mode (piano-style)', collapsable=True, collapse=True,
                     borderStyle='etchedIn', marginWidth=8, marginHeight=6,
                     backgroundColor=bg_main)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=4)

    cmds.checkBox('keyboard_mode_checkbox',
                  label='Enable keyboard mode',
                  value=KEYBOARD_MODE_ENABLED,
                  changeCommand=toggle_keyboard_mode)

    cmds.rowLayout(numberOfColumns=6, columnWidth6=(75, 75, 75, 75, 75, 75),
                   columnAttach=[(1, 'both', 1), (2, 'both', 1), (3, 'both', 1), (4, 'both', 1), (5, 'both', 1), (6, 'both', 1)])
    cmds.text(label="[0] Neu", align='center', backgroundColor=accent)
    cmds.text(label="[n] N", align='center', backgroundColor=accent)
    cmds.text(label="[a] AA", align='center', backgroundColor=accent)
    cmds.text(label="[i] IY", align='center', backgroundColor=accent)
    cmds.text(label="[t] T", align='center', backgroundColor=accent)
    cmds.text(label="[e] EH", align='center', backgroundColor=accent)
    cmds.setParent('..')

    cmds.rowLayout(numberOfColumns=6, columnWidth6=(75, 75, 75, 75, 75, 75),
                   columnAttach=[(1, 'both', 1), (2, 'both', 1), (3, 'both', 1), (4, 'both', 1), (5, 'both', 1), (6, 'both', 1)])
    cmds.text(label="[u] UW", align='center', backgroundColor=accent)
    cmds.text(label="[v] V", align='center', backgroundColor=accent)
    cmds.text(label="[y] Y", align='center', backgroundColor=accent)
    cmds.text(label="[s] Smile", align='center', backgroundColor=btn_reset)
    cmds.text(label="[f] Frown", align='center', backgroundColor=btn_reset)
    cmds.text(label="[b] Brow", align='center', backgroundColor=btn_reset)
    cmds.setParent('..')

    cmds.rowLayout(numberOfColumns=6, columnWidth6=(75, 75, 75, 75, 75, 75),
                   columnAttach=[(1, 'both', 1), (2, 'both', 1), (3, 'both', 1), (4, 'both', 1), (5, 'both', 1), (6, 'both', 1)])
    cmds.text(label="[m] Mouth", align='center', backgroundColor=btn_reset)
    cmds.text(label="[q] Squint", align='center', backgroundColor=btn_reset)
    cmds.text(label="[c] Cheek", align='center', backgroundColor=btn_reset)
    cmds.text(label="[p] Pucker", align='center', backgroundColor=btn_reset)
    cmds.text(label="", align='center')
    cmds.text(label="", align='center')
    cmds.setParent('..')

    cmds.textField('keyboard_input_field',
                   placeholderText='Type: n, a, i, t, e, u, v, y, s, f, b, m, q, c, p, 0',
                   textChangedCommand=on_keyboard_input_changed,
                   backgroundColor=bg_field, height=26)
                   
    cmds.separator(height=6, style='none')
    cmds.button(label='Precompute All Keyboard Transitions (Fix First-Time Lag)', height=28,
                backgroundColor=[0.45, 0.75, 0.55], command=precompute_keyboard_transitions)
    cmds.text('keyboard_status_text', label='Ready', align='center', font='smallPlainLabelFont')

    cmds.setParent('..')
    cmds.setParent('..')

    # -- Footer --
    cmds.separator(height=4, style='none')
    cmds.setParent('..')  # end main columnLayout
    cmds.setParent('..')  # end scrollLayout
    cmds.showWindow(window_id)


# --- Run this line in the Maya Script Editor to show the window ---
create_automation_ui()