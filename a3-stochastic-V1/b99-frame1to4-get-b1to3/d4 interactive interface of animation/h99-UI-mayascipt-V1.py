# have to run this code in maya
# open this maya file: 
# C:\keyLabLocal_D\c99-maya-ziva-workshop\experiment_files\g10-10 rest pose 1200 landmarks v9.mb


# commandPort -name "localhost:7001" -sourceType "mel";

import maya.cmds as cmds
import os
import re
import math
import pickle
import numpy as np
import json
import maya.OpenMaya as OpenMaya
import csv
import datetime
import inspect
from config import a3_stochastic_relative_path, b98_sampling_path, a1_blendshape_path

import sys
sys.path.insert(0, str(a3_stochastic_relative_path / "a3-stochastic-V1-algo"/ "example_train"))
sys.path.insert(0, str(a1_blendshape_path / "database"))
sys.path.insert(0, str(b98_sampling_path))
sys.path.insert(0, str(b98_sampling_path/ "300joints_suite_good_old"))
sys.path.insert(0, str(b98_sampling_path/ "joint_positions"/ "joint_position_ooc_l"))
sys.path.insert(0, str(b98_sampling_path/ "joint_positions"/ "joint_position_ooc_r"))
sys.path.insert(0, str(b98_sampling_path/ "joint_positions"/ "joint_position_frontalis"))
sys.path.insert(0, r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\b31 dataV22 backup hub")
sys.path.insert(0, str(a3_stochastic_relative_path / "b95-debug-use"))



# Get neutral pose positions from joint_position modules (TRUE neutral reference)
# These are the same positions used during training, NOT from Maya scene
import joint_position
import joint_position_first_wave
import joint_position_third_wave
import joint_position_oor_four_hundred

oor_joint_modules = [
    joint_position_first_wave,
    joint_position,
    joint_position_third_wave
    # joint_position_oor_four_hundred
]

# import joint_position_oor_full


# oor_joint_modules = [
#     joint_position_oor_full
# ]

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
VERBOSE_MODE = True # Set to True to see all detailed debug prints

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
DEBUG_OUTPUT_DIR = str(a3_stochastic_relative_path / "b99-frame1to4-get-b1to3"/ "d4 interactive interface of animation"/ "h96-debug-dissipating-error")
DEBUG_ENABLED = True  # Enable detailed debugging and logging
CURRENT_JAW_LOG_FILE = ""
DEBUG_DATA = []  # Stores debug information for each transition

# --- KEYBOARD MODE CONFIGURATION ---
# Piano-style keyboard mode: Press keys to trigger phoneme transitions
KEYBOARD_MODE_ENABLED = False  # Enable/disable keyboard mode
KEYBOARD_CURRENT_PHONEME = "dataV22_AU0_neutral"  # Current phoneme state
KEYBOARD_SCRIPT_JOBS = []  # Store script job IDs for cleanup

# Key to Phoneme mapping (using intuitive letters)
# Press the letter that corresponds to the phoneme sound
KEYBOARD_PHONEME_MAP = {
    '0': {'phoneme': 'dataV22_AU0_neutral', 'label': 'Neutral', 'key': '0', 'octave': 'mid'},
    'n': {'phoneme': 'dataV22_phoneme_N', 'label': 'N', 'key': 'n', 'octave': 'low'},
    'a': {'phoneme': 'dataV22_phoneme_AA', 'label': 'AA', 'key': 'a', 'octave': 'low'},
    'i': {'phoneme': 'dataV22_phoneme_IY', 'label': 'IY', 'key': 'i', 'octave': 'low'},
    't': {'phoneme': 'dataV22_phoneme_T', 'label': 'T', 'key': 't', 'octave': 'low'},
    'e': {'phoneme': 'dataV22_phoneme_EH', 'label': 'EH', 'key': 'e', 'octave': 'low'},
    'u': {'phoneme': 'dataV22_phoneme_UW', 'label': 'UW', 'key': 'u', 'octave': 'low'},
    'v': {'phoneme': 'dataV22_phoneme_V', 'label': 'V', 'key': 'v', 'octave': 'low'},
    'y': {'phoneme': 'dataV22_phoneme_Y', 'label': 'Y', 'key': 'y', 'octave': 'low'},
    's': {'phoneme': 'dataV22_AU12', 'label': 'Smile', 'key': 's', 'octave': 'mid'},
    'f': {'phoneme': 'dataV22_AU4', 'label': 'Frown', 'key': 'f', 'octave': 'high'},
    'b': {'phoneme': 'dataV22_AU2', 'label': 'Brow', 'key': 'b', 'octave': 'high'},
    'm': {'phoneme': 'dataV22_AU25', 'label': 'Mouth', 'key': 'm', 'octave': 'mid'},
    'q': {'phoneme': 'dataV22_AU7', 'label': 'Squint', 'key': 'q', 'octave': 'high'},
    'c': {'phoneme': 'dataV22_AU6', 'label': 'Cheek', 'key': 'c', 'octave': 'mid'},
    'p': {'phoneme': 'dataV22_AU22', 'label': 'Pucker', 'key': 'p', 'octave': 'mid'},
    '1': {'phoneme': 'chord_dataV22_AU6___dataV22_AU12___dataV22_AU25', 'label': 'Happy', 'key': '1', 'octave': 'mid'},
    '2': {'phoneme': 'dataV22_AU4', 'label': 'Sad', 'key': '2', 'octave': 'mid'},
    '3': {'phoneme': 'chord_dataV22_AU1___dataV22_AU2___dataV22_AU25', 'label': 'Surprise', 'key': '3', 'octave': 'mid'},
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
    base_dir = str(a3_stochastic_relative_path)
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
MERGED_DEFORMATION_DB_PATH = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\merged_deformation_parameters_hub"

# Global variables for pose configuration (set by parse_sequence_string)
NEUTRAL_TO_OTHER_POSE_FLAG = 1  # Default to neutral start
PREVIOUS_POSE_FOLDER = ""  # Will be set if non-neutral start
TEETH_HACK_ENABLED = True
teeth_correction_weight = 0.25

FINGER_CLAP_ENABLED = False
RENDER_AFTER_DEMO_ENABLED = False
RENDER_AFTER_TYPEIN_ENABLED = False

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

TEETH_TONGUE_MESH_PATTERNS = ['*loTeeth01_geo*', '*tongue01_geo*']


def get_teeth_tongue_meshes():
    """Returns unique base transform names for teeth and tongue meshes, excluding any _RENDER proxies."""
    meshes = []
    for pattern in TEETH_TONGUE_MESH_PATTERNS:
        found = cmds.ls(pattern, type='transform') or cmds.ls(pattern) or []
        for m in found:
            if not m.endswith("_RENDER") and not m.endswith("_sagged_snapshot") and not m.endswith("_equil_snapshot"):
                if m not in meshes:
                    meshes.append(m)
    return meshes


def ensure_teeth_tongue_render_meshes(at_frame_zero=False):
    """
    Creates _RENDER versions of teeth and tongue meshes (e.g. jack01_c_loTeeth01_geo_RENDER,
    jack01_c_tongue01_geo_RENDER), sets up master blendshapes to follow live Ziva meshes,
    hides the live Ziva meshes (visibility=0), and shows the _RENDER meshes (visibility=1).
    """
    valid_meshes = [m for m in get_teeth_tongue_meshes() if cmds.objExists(m)]
    needs_creation = any(not cmds.objExists(f"{m}_RENDER") for m in valid_meshes)
    
    if needs_creation and at_frame_zero:
        curr_time = cmds.currentTime(query=True)
        if curr_time != 0:
            cmds.currentTime(0, edit=True)

    for target_mesh in valid_meshes:
        # Extract a clean, short name without pipes (hierarchy) or colons (namespaces)
        clean_name = target_mesh.split('|')[-1].split(':')[-1]
        render_mesh = f"{clean_name}_RENDER"
        bs_name = f"master_bs_{clean_name}"
        
        if not cmds.objExists(render_mesh):
            render_mesh_node = cmds.duplicate(target_mesh, name=render_mesh)[0]
            
            # Create master blendshape connecting Live Ziva to Proxy
            if not cmds.objExists(bs_name):
                cmds.blendShape([target_mesh], render_mesh_node, name=bs_name, origin='local')
                cmds.setAttr(f"{bs_name}.{clean_name}", 1.0)
                
            print(f"[TEETH/TONGUE RENDER] Created RENDER mesh: {render_mesh_node} for {target_mesh}")

        # Always ensure correct visibility: hide live Ziva mesh, show RENDER mesh
        if cmds.attributeQuery('visibility', node=target_mesh, exists=True):
            cmds.setAttr(f"{target_mesh}.visibility", 0)
        if cmds.attributeQuery('visibility', node=render_mesh, exists=True):
            cmds.setAttr(f"{render_mesh}.visibility", 1)


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

DEBUG_MODE = True

# def step_timeline_forward(target_frame):
#     """
#     Steps the timeline forward one frame at a time from current to target_frame.
#     This ensures Ziva's physics cache is perfectly evaluated and we never jump backwards.
#     It simultaneously collects live debug data so we don't have to simulate again at the end.
#     """
#     global LIVE_DEBUG_FRAMES, LIVE_DEBUG_POSITIONS
    
#     current_time = int(cmds.currentTime(query=True))
#     target = int(target_frame)
    
#     # Live target logic (prioritize proxy render mesh if it exists)
#     t_mesh = "middleres_OrbicularisOris_RENDER" if cmds.objExists("middleres_OrbicularisOris_RENDER") else "middleres_OrbicularisOris"
    
#     # If starting fresh at frame 0, collect frame 0 data
#     if current_time == 0 and len(LIVE_DEBUG_FRAMES) == 0:
#         try:
#             pos = cmds.xform(f"{t_mesh}.vtx[0]", query=True, translation=True, worldSpace=True)
#             LIVE_DEBUG_FRAMES.append(0)
#             LIVE_DEBUG_POSITIONS.append(pos[1])
#         except Exception:
#             pass
            
#     if current_time < target:
#         print(f"  [PHYSICS] Stepping timeline sequentially from {current_time} to {target} to preserve Ziva cache...")
#         for f in range(current_time + 1, target + 1):
#             cmds.currentTime(f, edit=True)
            
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
        
        frames_stepped = 0
        for f in range(current_time + 1, target + 1):
            cmds.currentTime(f, edit=True)
            frames_stepped += 1
            
            # # --- DEBUG MODE TERMINATION ---
            # if DEBUG_MODE and frames_stepped >= 20:
            #     print("passed the test, now quit.")
            #     sys.exit(0)  # Exit with status code 0
                            
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
        # This overshoot causes the weights to NOT sum to 1.0 between frames (like at frame 31-32).
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
    """
    print(f"\n[FEEDBACK CONTROL] Computing ideal target positions for {target_phoneme}")

    # Compute TRUE neutral positions first since we need them in both cases
    x_joint_combined = []
    for mod in oor_joint_modules:
        x_joint_combined.extend(getattr(mod, "x_joint_position_1"))
    x_neutral = [item for sublist in x_joint_combined for item in sublist]

    x_joint_first_wave = getattr(joint_position_ooc_l_base_0, "x_joint_position_1")
    x_joint_second_wave = getattr(joint_position_ooc_l_base_pose11_0, "x_joint_position_1")
    x_joint_third_wave = getattr(joint_position_ooc_l_base_pose11_AU1_0, "x_joint_position_1")
    x_joint_fourth_wave = getattr(joint_position_ooc_l_base_pose11_AU1_AU2_0, "x_joint_position_1")
    x_joint_combined = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave + x_joint_fourth_wave
    x_neutral_ooc_l = [item for sublist in x_joint_combined for item in sublist]

    x_joint_first_wave = getattr(joint_position_ooc_r_base_0, "x_joint_position_1")
    x_joint_second_wave = getattr(joint_position_ooc_r_base_pose11_0, "x_joint_position_1")
    x_joint_third_wave = getattr(joint_position_ooc_r_base_pose11_AU1_0, "x_joint_position_1")
    x_joint_fourth_wave = getattr(joint_position_ooc_r_base_pose11_AU1_AU2_0, "x_joint_position_1")
    x_joint_combined = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave + x_joint_fourth_wave
    x_neutral_ooc_r = [item for sublist in x_joint_combined for item in sublist]

    x_joint_first_wave = getattr(joint_position_frontalis_base_0, "x_joint_position_1")
    x_joint_second_wave = getattr(joint_position_frontalis_base_AU1_0, "x_joint_position_1")
    x_joint_third_wave = getattr(joint_position_frontalis_base_AU1_AU2_0, "x_joint_position_1")
    x_joint_combined = x_joint_first_wave + x_joint_second_wave + x_joint_third_wave
    x_neutral_frontalis = [item for sublist in x_joint_combined for item in sublist]

    if target_phoneme == "dataV22_AU0_neutral":
        print(f"[FEEDBACK CONTROL] Target is Neutral Pose. Returning true absolute zero positions.")
        return {
            'x_joint_conca': x_neutral,
            'x_joint_rotation_conca': [0.0, 0.0, 0.0],
            'x_ooc_l_joint_conca': x_neutral_ooc_l,
            'x_ooc_r_joint_conca': x_neutral_ooc_r,
            'x_frontalis_joint_conca': x_neutral_frontalis
        }

    if is_combined_expression_pose(target_phoneme):
        short = target_phoneme.replace('dataV22_', '')
        checkpoints_file = os.path.join(MERGED_DEFORMATION_DB_PATH, short, "combined_checkpoints.pkl")
        print(f"[FEEDBACK CONTROL] Target is Merged Pose '{target_phoneme}'. Checking merged path: {checkpoints_file}")
    else:
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

        # Load combined_checkpoints.pkl (contains accumulated deformation parameters)
        checkpoints_file = os.path.join(deformation_path, "combined_checkpoints.pkl")

    if not os.path.exists(checkpoints_file):
        print(f"[FEEDBACK CONTROL WARNING] combined_checkpoints.pkl not found at {checkpoints_file}")
        return None

    try:
        with open(checkpoints_file, 'rb') as f:
            checkpoints = pickle.load(f)

        print(f"[FEEDBACK CONTROL] Loaded {len(checkpoints)} checkpoints from {checkpoints_file}")

        # Get the LAST checkpoint (final accumulated deformation)
        final_checkpoint = checkpoints[-1]

        # Expected dimensions
        oor_num_joints = sum(len(mod.x_joint_position_1) for mod in oor_joint_modules)
        n = oor_num_joints * 3  # oor landmarks
        joint_rotate_num = 1
        joint_ooc_l_num = 300
        joint_ooc_r_num = 300
        joint_frontalis_num = 300

        # Parse the final checkpoint
        delta_b_final = np.array(final_checkpoint)

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
        'oor': sum(len(mod.x_joint_position_1) for mod in oor_joint_modules), 'ooc_l': 300,
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
    time_temp = float(keyframe_max)
    cmds.currentTime(time_temp, edit=True)

    x_joint_init_conca = x_joint_conca
    x_joint_rotation_init_conca = x_joint_rotation_conca

    x_joint_ = au22_x_joint_conca3
    x_joint_rotation_ = au22_x_joint_rotation_conca3
    x_ooc_l_joint_ = au22_x_ooc_l_joint_conca3
    x_ooc_r_joint_ = au22_x_ooc_r_joint_conca3
    x_frontalis_joint_ = au22_x_frontalis_joint_conca3

    relative_rotation_value = np.array(x_joint_rotation_) - np.array(x_joint_rotation_init_conca)

    # Check if any component is non-zero
    if np.any(relative_rotation_value != 0):
        # print(f"[DEBUG] Jaw rotation detected! relative_rotation_value: {relative_rotation_value}")
        
        # Apply relative rotation to jaw01
        cmds.rotate(
            relative_rotation_value[0], 
            relative_rotation_value[1], 
            relative_rotation_value[2], 
            'jaw01', 
            relative=True, 
            ws=True
        )
        
        # print("passed the test, now quit.")
        # sys.exit(0)
    else:
        # print(f"[DEBUG] No rotation detected for jaw01 (relative_rotation_value: {relative_rotation_value}).")
        pass

    # cmds.rotate(relative_rotation_value[0], relative_rotation_value[1], relative_rotation_value[2], 'jaw01', relative=True, ws=True)
    cmds.setKeyframe('jaw01', time=time_temp)

    if TEETH_HACK_ENABLED:
        # --- HACK: Lower Teeth & Tongue 50% Extra Rotation on RENDER Meshes ---
        # Corrects the artifact where OOR plastic deformation pulls the lips too far relative to the rigid teeth/tongue.
        # We perform the deformation on the _RENDER mesh directly and present it to the user while hiding the live Ziva mesh.
        ensure_teeth_tongue_render_meshes(at_frame_zero=False)
        
        
        for mesh_name in get_teeth_tongue_meshes():
            render_mesh = f"{mesh_name}_RENDER"
            target_to_rotate = render_mesh if cmds.objExists(render_mesh) else mesh_name
            
            try:
                # Temporarily suppress harmless wrap warnings to keep the console clean
                cmds.scriptEditorInfo(suppressWarnings=True)
                
                # We MUST rotate around the jaw's pivot, otherwise the teeth will detach from the jaw hinge!
                jaw_pivot = cmds.xform('jaw01', query=True, rotatePivot=True, ws=True)
                cmds.rotate(relative_rotation_value[0] * teeth_correction_weight, 
                            relative_rotation_value[1] * teeth_correction_weight, 
                            relative_rotation_value[2] * teeth_correction_weight, 
                            target_to_rotate, relative=True, ws=True, pivot=jaw_pivot)
                cmds.setKeyframe(target_to_rotate, time=time_temp)
                
                # Ensure the live Ziva mesh is hidden and RENDER mesh is visible
                if cmds.objExists(render_mesh):
                    if cmds.attributeQuery('visibility', node=mesh_name, exists=True):
                        cmds.setAttr(f"{mesh_name}.visibility", 0)
                    if cmds.attributeQuery('visibility', node=render_mesh, exists=True):
                        cmds.setAttr(f"{render_mesh}.visibility", 1)
                
                # Turn warnings back on immediately so we don't miss important ones
                cmds.scriptEditorInfo(suppressWarnings=False)
                
            except Exception as e:
                print(f"[WARNING] Could not apply rotation hack to {target_to_rotate}: {e}")
                cmds.scriptEditorInfo(suppressWarnings=False)

    joint_num = sum(len(mod.x_joint_position_1) for mod in oor_joint_modules)
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

def perform_the_deformation(source_path, start_frame_offset=14, continue_from_last=False, correction_vectors=None, target_phoneme=None):
    """
    Apply deformation to Maya rig with Linear Error Dissipation.
    
    Args:
        source_path: Direct path to the folder containing combined_checkpoints.pkl
        correction_vectors: Dict containing total correction needed for each joint group.
                            { 'x_joint_conca': np.array([...]), ... }
    """
    print("--- Starting Deformation Application (With Linear Error Dissipation) ---")
    
    # Use source_path directly (no more SEQUENCE_DB_PATH)
    checkpoints_path = os.path.join(source_path, "combined_checkpoints.pkl")
    if not os.path.exists(checkpoints_path):
        raise RuntimeError(f"Deformation file not found! {checkpoints_path}")

    with open(checkpoints_path, "rb") as file:
        combined_checkpoints = pickle.load(file)

    # Dimensions
    oor_num = sum(len(mod.x_joint_position_1) for mod in oor_joint_modules)
    n = oor_num * 3
    joint_rotate_num = 1
    joint_ooc_l_num = 300
    joint_ooc_r_num = 300
    joint_frontalis_num = 300
    expected_dimension = n + (3*joint_rotate_num) + (3*joint_ooc_l_num) + (3*joint_ooc_r_num) + (3*joint_frontalis_num)

    # Initialize positions
    if not continue_from_last:
        x_joint, x_joint_rotation, x_ooc_l_joint, x_ooc_r_joint, x_frontalis_joint = initial_joint_position(start_frame_offset, is_first_transition=False)
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

    # --- PRE-EMPTIVE DRIFT CORRECTION FOR JAW ROTATION ---
    if target_phoneme is not None:
        # Load the mathematical ground truth for this specific phoneme
        ideal_target_dict = load_ideal_target_positions(target_phoneme)
        
        if ideal_target_dict is not None and 'x_joint_rotation_conca' in ideal_target_dict:
            ideal_jaw_rot = np.array(ideal_target_dict['x_joint_rotation_conca'])
            
            # The Neural Network checkpoints inherently drift over long cyclic animations (e.g. V->AA->V->AA).
            # We calculate exactly how much the sequence WILL miss the true mathematical target, 
            # and inject a correction to pull it exactly into place linearly over the transition.
            delta_b_final = np.array(combined_checkpoints[-1])
            if len(delta_b_final) < expected_dimension:
                padding = np.zeros(expected_dimension - len(delta_b_final))
                delta_b_final = np.concatenate((delta_b_final, padding))
                
            expected_end_rotation = x_joint_rotation_init + delta_b_final[n:n + 3 * joint_rotate_num]
            drift_mismatch = ideal_jaw_rot - expected_end_rotation
            
            print(f"[PRE-EMPTIVE JAW DRIFT] Target Phoneme: {target_phoneme}")
            print(f"[PRE-EMPTIVE JAW DRIFT] Ideal Target Rotation: {ideal_jaw_rot}")
            print(f"[PRE-EMPTIVE JAW DRIFT] Expected Neural Net End Rotation: {expected_end_rotation}")
            print(f"[PRE-EMPTIVE JAW DRIFT] Injecting linear correction: {drift_mismatch}")
            
            if correction_vectors is None:
                correction_vectors = {}
            
            # Override the jaw rotation correction vector with our absolute pre-emptive one
            correction_vectors['x_joint_rotation_conca'] = drift_mismatch

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

        # if target_phoneme == "dataV22_AU0_neutral" and "AU1_AU2" in source_path and i == 4:
        #     script_name = __file__ if '__file__' in globals() else "h99-UI-mayascipt-V1.py"
        #     print(f"🟢 [OK] Reached line {inspect.currentframe().f_lineno} in {os.path.basename(script_name)}")
        #     print("suspicious values: delta_b0_conca[0:10] =", delta_b0_conca[0:10])
        #     print("passed the test, now quit.")
        #     sys.exit(0)


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
                
            if 'x_joint_rotation_conca' in correction_vectors:
                x_joint_rot_before = np.copy(x_joint_rotation_)
                x_joint_rotation_ = x_joint_rotation_ + (correction_vectors['x_joint_rotation_conca'] * progress)
                x_joint_rot_after = np.copy(x_joint_rotation_)
                
                # --- JAW DEBUG INTERVENTION ---
                jaw_correction = correction_vectors['x_joint_rotation_conca']

                if False and np.any(np.abs(jaw_correction) > 1e-6):
                    print("\n" + "="*50)
                    print("[JAW ERROR DEBUG] INSPECTING CORRECTION VECTOR")
                    print(f"Frame Progress: {progress}")
                    print(f"x_joint_rotation_ BEFORE correction:\n {x_joint_rot_before}")
                    print(f"correction_vectors['x_joint_rotation_conca']:\n {jaw_correction}")
                    print(f"x_joint_rotation_ AFTER correction:\n {x_joint_rot_after}")
                    print("="*50 + "\n")
                    import sys
                    sys.exit(0)
                # ------------------------------

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
                
                oor_num = sum(len(mod.x_joint_position_1) for mod in oor_joint_modules)
                joint_groups = {'oor': oor_num, 'ooc_l': 300, 'ooc_r': 300, 'frontalis': 300}
                actual_positions = retrieve_landmark_positions(joint_groups, frame, is_first_transition=False)
                actual_flat = {
                    'x_joint_conca': [item for sublist in actual_positions.get('oor', []) for item in sublist],
                    'x_joint_rotation_conca': x_joint_rotation.tolist() if hasattr(x_joint_rotation, 'tolist') else list(x_joint_rotation),
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
    playback_end = final_frame
    cmds.playbackOptions(minTime=1, maxTime=playback_end)
    
    return final_frame, per_frame_debug_data, x_joint_rotation.tolist() if hasattr(x_joint_rotation, 'tolist') else list(x_joint_rotation)


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
    oor_num = sum(len(mod.x_joint_position_1) for mod in oor_joint_modules)
    self_n = oor_num * 3
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
    output_path = os.path.join(log_path_output, "combined_checkpoints.pkl")
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
    Generates combined_checkpoints.pkl directly to the SOURCE path (from DEFORMATION_PATH_DB).
    This is a simplified version that generates checkpoints to the original training output folder.
    
    Args:
        phonemes_list: List of phonemes for this transition
        log_paths_list: List of paths (from DEFORMATION_PATH_DB)
        source_path: The source path where to save combined_checkpoints.pkl
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
    output_path = os.path.join(log_path_output, "combined_checkpoints.pkl")
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


# --- 2B. MERGED MULTI-AU DEFORMATION MODULE (FROM j75-V8) ---

def get_total_steps_convergence(log_path, phonemes):
    """Extracts phase step counts from log files for specific phases"""
    phase_files = ["log0.txt"]
    if len(phonemes) >= 3:
        if len(phonemes) >= 4:
            phase_files.extend(["log0_phase7.txt", "log0_phase8.txt"])
    
    total_steps_convergence = [0]

    for file in phase_files:
        file_path = os.path.join(log_path, file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                match = re.search(r'done at No\. (\d+) step', content)
                if match:
                    total_steps_convergence.append(int(match.group(1)))
        else:
            print(f"File not found: {file_path}")

    print(f"[{log_path}] total_steps_convergence =", total_steps_convergence)
    return total_steps_convergence


def accumulate_checkpoints(action_lists, checkpoint_indices, self_n, joint_rotate_num, joint_ooc_l_num, joint_ooc_r_num, joint_frontalis_num):
    """
    Accumulates actions at specified checkpoint indices.
    Ensures the state at index 0 is the initial, non-deformed state.
    """
    delta_b0_conca = [0] * self_n + [0] * 3 * joint_rotate_num  + [0] * 3 *joint_ooc_l_num  + [0] * 3 * joint_ooc_r_num  + [0] * 3 * joint_frontalis_num
    checkpoints = []
    global_index = 0
    found_indices = set()
    
    if 0 in checkpoint_indices:
        checkpoints.append((0, list(delta_b0_conca)))
        found_indices.add(0)

    for action_list in action_lists:
        for action in action_list:
            global_index += 1
            if len(action) == len(delta_b0_conca):
                delta_b0_conca = [delta_b0_conca[i] + action[i] for i in range(len(delta_b0_conca))]
            elif len(action) == self_n:
                for i in range(self_n):
                    delta_b0_conca[i] += action[i]
            
            if global_index in checkpoint_indices and global_index not in found_indices:
                checkpoints.append((global_index, list(delta_b0_conca)))
                found_indices.add(global_index)
                
    checkpoints.sort(key=lambda x: x[0])
    return [cp[1] for cp in checkpoints]


def process_single_deformation(config, num_checkpoints=36, overlap_ratio=0.0):
    log_path = config["log_path"]
    phonemes = config["phonemes"]

    total_steps_convergence = get_total_steps_convergence(log_path, phonemes)
    if not total_steps_convergence or len(total_steps_convergence) < 2:
        print(f"Error: Could not determine convergence steps for {config['name']} in {log_path}")
        return None

    phase_boundaries = total_steps_convergence
    timestamps = [0.0]
    if len(phonemes) == 2:
        timestamps.append(1.0)
    elif len(phonemes) == 3:
        timestamps.extend([0.3, 1.0])
    elif len(phonemes) >= 4:
        timestamps.extend([0.3, 0.6, 1.0])
    
    num_phases_available = len(phase_boundaries) - 1
    timestamps = timestamps[:num_phases_available + 1]
    if timestamps[-1] != 1.0:
        timestamps[-1] = 1.0

    phase_durations = []
    for i in range(len(timestamps) - 1):
        phase_durations.append(timestamps[i+1] - timestamps[i])
        
    total_duration = sum(phase_durations)
    phase_weights = [d / total_duration for d in phase_durations]

    phase_indices_to_load = list(range(len(phase_boundaries) - 1))
    
    # Calculate checkpoints per phase based on duration weights
    checkpoints_per_phase = []
    remaining_checkpoints = num_checkpoints
    for i, weight in enumerate(phase_weights):
        if i == len(phase_weights) - 1:
            num_cp = remaining_checkpoints
        else:
            num_cp = int(round(num_checkpoints * weight))
            remaining_checkpoints -= num_cp
        checkpoints_per_phase.append(max(1, num_cp))

    # Generate checkpoint indices for each phase
    all_checkpoints = []
    buffer_ratio = 0.05
    
    for i, num_cp in enumerate(checkpoints_per_phase):
        start_step = phase_boundaries[i]
        end_step = phase_boundaries[i+1]
        phase_len = end_step - start_step
        
        if i == 0 and num_cp > 0:
            all_checkpoints.append(0)
            num_cp -= 1
            
        if num_cp <= 0:
            continue
            
        eff_start = start_step + int(phase_len * buffer_ratio)
        eff_end = end_step - int(phase_len * buffer_ratio)
        
        if eff_end <= eff_start:
            eff_start = start_step
            eff_end = end_step

        if i > 0 and overlap_ratio > 0:
            prev_len = phase_boundaries[i] - phase_boundaries[i-1]
            overlap_steps = int(prev_len * overlap_ratio)
            eff_start = max(start_step - overlap_steps, phase_boundaries[i-1])

        cp_indices = np.linspace(eff_start, eff_end, num=num_cp, dtype=int)
        all_checkpoints.extend(cp_indices)

    all_checkpoints = sorted(list(set(all_checkpoints)))
    while len(all_checkpoints) < num_checkpoints:
        all_checkpoints.append(all_checkpoints[-1])
    if len(all_checkpoints) > num_checkpoints:
        all_checkpoints = all_checkpoints[:num_checkpoints]
    
    checkpoints = all_checkpoints

    action_lists = []
    for i in phase_indices_to_load:
        phase_num = i + 1
        file_path = os.path.join(log_path, f"final_epoch_action_list{'_phase' + str(phase_num) if phase_num > 1 else ''}.pkl")
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    if 'final_epoch_action_list' in data:
                        action_lists.append(data['final_epoch_action_list'])
                    elif 'action_list' in data:
                        action_lists.append(data['action_list'])
                    else:
                        action_lists.append([])
                elif isinstance(data, list):
                    action_lists.append(data)
                else:
                    action_lists.append([])
                print(f"Successfully loaded Phase {phase_num} with {len(action_lists[-1])} actions")
        except Exception as e:
            print(f"Warning: Could not find/load action list for Phase {phase_num} at {file_path}: {e}")
            action_lists.append([])

    self_n = sum(len(mod.x_joint_position_1) for mod in oor_joint_modules) * 3
    joint_rotate_num = 1
    joint_ooc_l_num = 300
    joint_ooc_r_num = 300
    joint_frontalis_num = 300

    accumulated = accumulate_checkpoints(action_lists, checkpoints, self_n, joint_rotate_num, joint_ooc_l_num, joint_ooc_r_num, joint_frontalis_num)
    return accumulated


def merge_multiple_deformations_to_hub(target_poses, num_checkpoints=36, is_return_to_neutral=False):
    """
    Merges multiple deformation paths from neutral into a single combined_checkpoints.pkl.
    Saves the result in MERGED_DEFORMATION_DB_PATH under a subfolder named after the combined poses.
    Returns the output directory path containing combined_checkpoints.pkl.
    """
    targets = [p for p in target_poses if p != "dataV22_AU0_neutral" and p != "Neutral"]
    if not targets:
        raise ValueError("No target poses specified for merging.")

    short_names = [t.replace('dataV22_', '').replace('phoneme_', '') for t in targets]
    subfolder_name = "_".join(short_names)
    if is_return_to_neutral:
        subfolder_name += "_to_Neutral"
    output_dir = os.path.join(MERGED_DEFORMATION_DB_PATH, subfolder_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "combined_checkpoints.pkl")

    if os.path.exists(output_path):
        if is_return_to_neutral:
            print(f"[MERGED DEFORMATION] Forcing regeneration for return-to-neutral to clear old buggy cache.")
        else:
            try:
                with open(output_path, "rb") as f:
                    existing = pickle.load(f)
                if len(existing) == num_checkpoints:
                    print(f"[MERGED DEFORMATION] Found existing combined_checkpoints.pkl in {output_dir}")
                    return output_dir
                else:
                    print(f"[MERGED DEFORMATION] Existing file has {len(existing)} frames, need {num_checkpoints}. Regenerating...")
            except Exception as e:
                print(f"[MERGED DEFORMATION] Error reading existing file: {e}. Regenerating...")

    all_accumulated_checkpoints = []
    for t_pose in targets:
        t_pose_short = t_pose.replace('dataV22_', '')
        short_name = t_pose_short.replace('phoneme_', '')
        
        if is_return_to_neutral:
            from_name = t_pose_short
            to_name = "neutral"
            config_phonemes = [short_name, "Neutral"]
        else:
            from_name = "neutral"
            to_name = t_pose_short
            config_phonemes = ["Neutral", short_name]

        user_text = f"{from_name} to {to_name}"
        phonemes, log_paths = parse_sequence_string(user_text)
        if not log_paths or not log_paths[0]:
            raise RuntimeError(f"Could not find log path in DEFORMATION_PATH_DB for {from_name} to {to_name}")
        log_path = log_paths[0]
        config = {
            "name": short_name,
            "log_path": log_path,
            "phonemes": config_phonemes
        }
        accumulated = process_single_deformation(config, num_checkpoints=num_checkpoints)
        if accumulated:
            all_accumulated_checkpoints.append(accumulated)
        else:
            raise RuntimeError(f"Failed to process checkpoints for {t_pose} from {log_path}")

    first_len = len(all_accumulated_checkpoints[0])
    if not all(len(x) == first_len for x in all_accumulated_checkpoints):
        raise RuntimeError("Not all deformation sequences produced the same number of checkpoints.")

    # Inspect Frontalis Debug Output
    # print("\n--- DEBUGGING: Inspecting Frontalis Deformation ---")
    # oor_num = sum(len(mod.x_joint_position_1) for mod in oor_joint_modules)
    # frontalis_start_index = (oor_num * 3) + (3 * 1) + (3 * 300) + (3 * 300)
    # for i, t_name in enumerate(short_names):
    #     if not all_accumulated_checkpoints[i]:
    #         continue
    #     last_vec = all_accumulated_checkpoints[i][-1]
    #     front_part = last_vec[frontalis_start_index:]
    #     print(f"\n--- {t_name} Final Checkpoint ---")
    #     if len(front_part) > 0:
    #         print(f"  Max frontalis value: {max(front_part):.4f}")
    #         print(f"  Min frontalis value: {min(front_part):.4f}")
    #         print(f"  Avg frontalis value: {np.mean(front_part):.4f}")

    # print(f"\nCombining {len(all_accumulated_checkpoints)} deformation(s) with {first_len} checkpoints each.")
    sum_checkpoints = [v.copy() for v in all_accumulated_checkpoints[0]]
    for i in range(1, len(all_accumulated_checkpoints)):
        neutral_idx = -1 if is_return_to_neutral else 0
        for j in range(first_len):
            sum_checkpoints[j] = [
                sum_checkpoints[j][k] + (all_accumulated_checkpoints[i][j][k] - all_accumulated_checkpoints[i][neutral_idx][k])
                for k in range(len(sum_checkpoints[j]))
            ]

    # print("\n--- Summed Final Checkpoint ---")
    # if sum_checkpoints:
    #     summed_front = sum_checkpoints[-1][frontalis_start_index:]
    #     if len(summed_front) > 0:
    #         print(f"  Max summed frontalis value: {max(summed_front):.4f}")
    #         print(f"  Min summed frontalis value: {min(summed_front):.4f}")
    #         print(f"  Avg summed frontalis value: {np.mean(summed_front):.4f}")

    with open(output_path, "wb") as f:
        pickle.dump(sum_checkpoints, f)

    print(f"[MERGED DEFORMATION] Successfully merged {len(targets)} deformations ({' + '.join(short_names)}) into: {output_path}")
    return output_dir


def is_combined_expression_pose(pose_name):
    """Returns True if pose_name is a combined multi-AU pose like 'dataV22_AU6_AU12_AU25'."""
    if not pose_name or pose_name == "dataV22_AU0_neutral" or pose_name == "Neutral":
        return False
    if pose_name.startswith("chord_"):
        return True
    short = pose_name.replace('dataV22_', '')
    parts = short.split('_')
    au_parts = [p for p in parts if p.upper().startswith('AU') and p.upper() != 'AU0']
    return len(au_parts) >= 2


def run_merged_transition(targets, start_frame, continue_mode=True, is_first_transition=False, expression_name=None):
    """
    Executes a merged transition from Neutral to the combined target AUs in ONE transition.
    """
    short_names = [t.replace('dataV22_', '').replace('phoneme_', '') for t in targets]
    combined_label = " + ".join(short_names)
    if expression_name:
        combined_label = f"{expression_name} ({combined_label})"

    print(f"\n{'='*60}")
    print(f"RUNNING MERGED EXPRESSION TRANSITION: Neutral -> [{combined_label}]")
    print(f"{'='*60}\n")

    num_checkpoints = 36
    merged_dir = merge_multiple_deformations_to_hub(targets, num_checkpoints=num_checkpoints, is_return_to_neutral=False)

    final_frame, per_frame_data, final_jaw_rotation = perform_the_deformation(
        merged_dir, start_frame, continue_from_last=continue_mode, correction_vectors=None, target_phoneme=f"dataV22_{'_'.join(short_names)}"
    )
    return final_frame


def run_merged_return_to_neutral(combined_pose_name, start_frame, is_first_transition=False):
    """
    Executes a merged return transition from a combined pose (e.g. AU6_AU12_AU25) back to Neutral.
    """
    if combined_pose_name.startswith("chord_"):
        targets = combined_pose_name[6:].split("___")
        short = "chord_return"
    else:
        short = combined_pose_name.replace('dataV22_', '')
        parts = short.split('_')
        targets = [f"dataV22_{p}" for p in parts if p.upper().startswith('AU') and p.upper() != 'AU0']
    if not targets:
        return start_frame

    print(f"\n{'='*60}")
    print(f"RUNNING MERGED RETURN TRANSITION: [{short}] -> Neutral")
    print(f"{'='*60}\n")

    num_checkpoints = 36
    merged_dir = merge_multiple_deformations_to_hub(targets, num_checkpoints=num_checkpoints, is_return_to_neutral=True)

    final_frame, per_frame_data, final_jaw_rotation = perform_the_deformation(
        merged_dir, start_frame, continue_from_last=True, correction_vectors=None, target_phoneme="dataV22_AU0_neutral"
    )
    return final_frame


def smart_merged_expression_transition(current_pose, target_list, start_frame, is_first_transition=False, expression_name=None):
    """
    Transitions from current_pose to a combined expression (e.g. AU6 + AU12 + AU25) in ONE single transition.
    If current_pose is not neutral, routes through neutral first.
    """
    neutral = "dataV22_AU0_neutral"
    targets = [p for p in target_list if p != neutral and p != "Neutral"]

    if not targets:
        return smart_single_transition(current_pose, neutral, start_frame, is_first_transition=is_first_transition), neutral
    if len(targets) == 1:
        final_frame = smart_single_transition(current_pose, targets[0], start_frame, is_first_transition=is_first_transition)
        return final_frame, targets[0]

    short_names = [t.replace('dataV22_', '').replace('phoneme_', '') for t in targets]
    combined_pose_name = "dataV22_" + "_".join(short_names)

    intermediate_frame = start_frame
    if current_pose != neutral and current_pose != "Neutral":
        print(f"[INFO] Routing through Neutral before merged expression: {current_pose} -> Neutral -> {combined_pose_name}")
        if is_combined_expression_pose(current_pose):
            intermediate_frame = run_merged_return_to_neutral(current_pose, start_frame, is_first_transition=is_first_transition)
        else:
            intermediate_frame = run_single_transition(current_pose, neutral, start_frame, continue_mode=True, is_first_transition=is_first_transition)
        is_first_transition = False

    final_frame = run_merged_transition(
        targets, intermediate_frame,
        continue_mode=True, is_first_transition=is_first_transition,
        expression_name=expression_name
    )
    return final_frame, combined_pose_name


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
    custom_name = ""
    if cmds.textFieldGrp('playblast_name_field', exists=True):
        custom_name = cmds.textFieldGrp('playblast_name_field', query=True, text=True).strip()
        
    if not custom_name:
        import datetime
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_name = f"unnamed_playblast_{timestamp_str}"

    playblast_script = str(a3_stochastic_relative_path / "b99-frame1to4-get-b1to3"/ "d2 helper functions"/"do_playblast.py")
    print(f"\n--- Calling Playblast Script ({custom_name}) ---")
    if os.path.exists(playblast_script):
        try:
            import sys
            import importlib
            script_dir = str(a3_stochastic_relative_path / "b99-frame1to4-get-b1to3"/ "d2 helper functions")
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            import do_playblast
            importlib.reload(do_playblast)
            do_playblast.main(custom_name=custom_name)
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
    "neutral":            ["dataV22_AU0_neutral"],
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
    "au12l + au2l":           ["dataV22_AU0_neutral", "dataV22_AU12L", "dataV22_AU2L"],
    # Compound expressions
    "smile and open mouth":   ["dataV22_AU0_neutral", "dataV22_AU12", "dataV22_AU25"],
    "surprise":               ["dataV22_AU0_neutral", "dataV22_AU1", "dataV22_AU2", "dataV22_AU25"],
    "happy":                  ["dataV22_AU0_neutral", "dataV22_AU6", "dataV22_AU12", "dataV22_AU25"],
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
    ("Neutral",                      "dataV22_AU0_neutral"),
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
    ("Neutral",                 "neutral"),
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
    ("AU12L + AU2L",           "au12l + au2l"),
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
            d1_b99_file = str(a3_stochastic_relative_path / "b99-frame1to4-get-b1to3"/ "d1-adhoc-for-b99.py")
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
        d1_b99_file = str(a3_stochastic_relative_path / "b99-frame1to4-get-b1to3"/ "d1-adhoc-for-b99.py")
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

def get_transition_frames(p_from, p_to, base_frames=35):
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
    global ACCUMULATED_ERROR, PER_FRAME_DEBUG_HISTORY, CURRENT_JAW_LOG_FILE

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
    
    if OVERLAP_RATIO > 0:
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
    
    if source_path is None:
        raise RuntimeError(f"Could not find path for transition: {phoneme_from} -> {phoneme_to}")
    
    # Check if combined_checkpoints.pkl exists in the SOURCE path (from database)
    checkpoint_file_path = os.path.join(source_path, "combined_checkpoints.pkl")
    
    # Calculate dynamic frames based on phonemes
    transition_frames = get_transition_frames(phoneme_from, phoneme_to, base_frames=35)
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
                print(f"[CHECKPOINT] Found existing combined_checkpoints.pkl in source path with correct frame count.")
        except Exception as e:
            print(f"[CHECKPOINT] Error loading existing file: {e}. Regenerating...")
            generate_new = True
    else:
        print(f"[CHECKPOINT] Not found, generating to source path...")
        generate_new = True
        
    if generate_new:
        # Generate combined_checkpoints.pkl to the SOURCE path (not sequence_combined_deformation_parameters_hub)
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
    final_frame, per_frame_data, final_jaw_rotation = perform_the_deformation(source_path, start_frame, continue_mode, correction_vectors, target_phoneme=phoneme_to)

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
            oor_num = sum(len(mod.x_joint_position_1) for mod in oor_joint_modules)
            joint_groups = {'oor': oor_num, 'ooc_l': 300, 'ooc_r': 300, 'frontalis': 300}
            actual_positions = retrieve_landmark_positions(joint_groups, final_frame, is_first_transition=False)
            
            actual_flat = {
                'x_joint_conca': [item for sublist in actual_positions.get('oor', []) for item in sublist],
                'x_joint_rotation_conca': final_jaw_rotation,
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
                    
                    # The correction offset is a flat vector of dimension N.
                    # Split it into the same tissue-group structure as actual_flat.
                    # Layout: OOR (oor_num*3) + jaw (3) + OOC_L (900) + OOC_R (900) + Frontalis (900)
                    oor_num = sum(len(mod.x_joint_position_1) for mod in oor_joint_modules)
                    n_lm_oor = oor_num * 3
                    n_lm = 300 * 3  # 900
                    idx = 0
                    
                    oor_corr = correction_offset[idx:idx + n_lm_oor]; idx += n_lm_oor
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
                n = sum(len(mod.x_joint_position_1) for mod in oor_joint_modules) * 3
                ACCUMULATED_ERROR = {
                    'x_joint_conca': np.zeros(n),
                    'x_joint_rotation_conca': np.zeros(3),
                    'x_ooc_l_joint_conca': np.zeros(n),
                    'x_ooc_r_joint_conca': np.zeros(n),
                    'x_frontalis_joint_conca': np.zeros(n)
                }
                print(f"[FEEDBACK CONTROL] Initialized ACCUMULATED_ERROR with zeros")
    # --- FORCE JAW LOG TO DISK ---
    try:
        os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
        with open(CURRENT_JAW_LOG_FILE, "a") as f:
            f.write(f"\n[TRANSITION {from_name} -> {to_name} (Frame {final_frame})]\n")
            f.write(f"  Math State (Neural Net): {final_jaw_rotation}\n")
            
            try:
                phys_rot = cmds.xform('jaw01', query=True, rotation=True, worldSpace=True)
                f.write(f"  Physics State (Maya UI): {phys_rot}\n")
            except Exception:
                f.write(f"  Physics State (Maya UI): ERROR READING JAW\n")
                
            if 'x_joint_rotation_conca' in correction_vectors if correction_vectors else False:
                f.write(f"  Correction Vector Applied: {correction_vectors['x_joint_rotation_conca']}\n")
    except Exception as e:
        print(f"[JAW LOG ERROR] Could not write per-transition jaw log: {e}")
        
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
    global LAST_ANIMATION_END_FRAME, LAST_ANIMATION_PHONEME, CONTINUE_FROM_LAST, ACCUMULATED_ERROR, PER_FRAME_DEBUG_HISTORY, LIVE_DEBUG_FRAMES, LIVE_DEBUG_POSITIONS, RENDER_AFTER_TYPEIN_ENABLED, CURRENT_JAW_LOG_FILE

    import time
    pipeline_start_time = time.time()

    print("========= STARTING FULL PIPELINE =========")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    CURRENT_JAW_LOG_FILE = os.path.join(DEBUG_OUTPUT_DIR, f"jaw_position_log_{timestamp}.txt")
    
    # --- EARLY TEST RUN: PLAYBLAST ---
    # Change 'if True:' to 'if False:' once you have verified the playblast works!
    if False:
        print("\n[TEST RUN] Triggering playblast early to verify no errors...")
        run_playblast_script()
        print("[TEST RUN] Playblast test complete. Exiting pipeline early.")
        sys.exit(0)
    # ---------------------------------

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

        # Check if user_text is an expression command that maps to multiple AUs
        lower_input = user_text.strip().lower()
        if lower_input in EXPRESSION_COMMANDS:
            sequence = EXPRESSION_COMMANDS[lower_input]
            targets = [t for t in sequence if t != "dataV22_AU0_neutral" and t != "Neutral"]
            if len(targets) > 1:
                print(f"\n{'='*60}")
                print(f"[EXPRESSION COMMAND DETECTED] '{lower_input}' -> {len(targets)} target AUs")
                print(f"Executing merged multi-AU deformation in ONE single transition...")
                print(f"{'='*60}\n")
                
                # --- AUTOMATE JUMP TO 0 IF STARTING FROM NEUTRAL ---
                if LAST_ANIMATION_PHONEME == "dataV22_AU0_neutral":
                    print("[AUTO-RESET] Starting from Neutral. Automatically resetting animation chain and jumping to frame 0 for Ziva init.")
                    # Note: Assumes reset_animation_chain is defined globally or imported
                    try:
                        cmds.currentTime(0, edit=True)
                    except Exception:
                        pass
                    start_frame = 14
                else:
                    start_frame = LAST_ANIMATION_END_FRAME
                
                final_frame, combined_pose_name = smart_merged_expression_transition(
                    LAST_ANIMATION_PHONEME, targets, start_frame,
                    is_first_transition=not CONTINUE_FROM_LAST, expression_name=lower_input
                )
                LAST_ANIMATION_END_FRAME = final_frame
                LAST_ANIMATION_PHONEME = combined_pose_name
                CONTINUE_FROM_LAST = True
                print(f"--- Completed expression '{lower_input}' (merged pose: {combined_pose_name}) ---")
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
            # We create these here so they always exist for playblasts, even if there are no snaps!
            valid_soft_tissues = [m for m in SOFT_TISSUE_MESHES if cmds.objExists(m)]
            valid_teeth_tongue = [m for m in get_teeth_tongue_meshes() if cmds.objExists(m)]
            all_proxy_targets = valid_soft_tissues + valid_teeth_tongue
            
            # Check if we need to create any proxies. If we do, we MUST be at Frame 0 
            # so the duplication captures the pure, natively evaluated Neutral pose.
            needs_proxy = any(not cmds.objExists(f"{m}_RENDER") for m in all_proxy_targets)
            if needs_proxy:
                print(f"[INITIALIZATION] Jumping to Frame 0 to safely capture perfect Neutral proxy meshes...")
                cmds.currentTime(0, edit=True)
                
                print(f"[INITIALIZATION] Creating Proxy Render Meshes for {len(all_proxy_targets)} meshes (soft tissues + teeth/tongue)...")
                for target_mesh in all_proxy_targets:
                    render_mesh = f"{target_mesh}_RENDER"
                    bs_name = f"master_bs_{target_mesh}"
                    if not cmds.objExists(render_mesh):
                        render_mesh_node = cmds.duplicate(target_mesh, name=render_mesh)[0]
                        
                        if cmds.attributeQuery('visibility', node=target_mesh, exists=True):
                            cmds.setAttr(f"{target_mesh}.visibility", 0)
                        if cmds.attributeQuery('visibility', node=render_mesh_node, exists=True):
                            cmds.setAttr(f"{render_mesh_node}.visibility", 1)
                        
                        # Link it to follow the live Ziva mesh 100% by default
                        if not cmds.objExists(bs_name):
                            cmds.blendShape([target_mesh], render_mesh_node, name=bs_name, origin='local')
                            cmds.setAttr(f"{bs_name}.{target_mesh}", 1.0)
            else:
                for target_mesh in all_proxy_targets:
                    render_mesh = f"{target_mesh}_RENDER"
                    if cmds.objExists(render_mesh):
                        if cmds.attributeQuery('visibility', node=target_mesh, exists=True):
                            cmds.setAttr(f"{target_mesh}.visibility", 0)
                        if cmds.attributeQuery('visibility', node=render_mesh, exists=True):
                            cmds.setAttr(f"{render_mesh}.visibility", 1)

            # First transition: Start from frame 14 (or current start frame if continuing)
            start_frame = LAST_ANIMATION_END_FRAME if CONTINUE_FROM_LAST else 14
            initial_start_frame = start_frame  # Save for timing metrics
            continue_mode = CONTINUE_FROM_LAST

            # --- Jaw Position Logging Start ---
            current_t = cmds.currentTime(query=True)
            cmds.currentTime(0, edit=True)
            try:
                jaw_rot_start = cmds.xform('jaw01', query=True, rotation=True, worldSpace=True)
            except Exception:
                jaw_rot_start = [0, 0, 0]
            cmds.currentTime(current_t, edit=True)
            # ------------------------------------

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
                if False and phoneme_to == "dataV22_AU0_neutral":
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

            # --- Jaw Position Logging End ---
            cmds.currentTime(LAST_ANIMATION_END_FRAME, edit=True)
            
            # Query jaw01 rotation and translation
            try:
                jaw_rot_end = cmds.xform('jaw01', query=True, rotation=True, worldSpace=True)
                jaw_pos_end = cmds.xform('jaw01', query=True, translation=True, worldSpace=True)
            except Exception:
                jaw_rot_end = [0, 0, 0]
                jaw_pos_end = [0, 0, 0]
                
            # Try to find jaw meshes and sample their vertices
            jaw_meshes = cmds.ls("*jaw*", type="transform") + cmds.ls("*Jaw*", type="transform")
            jaw_vertex_samples = []
            for jm in set(jaw_meshes):
                try:
                    # check if it has vertices
                    vtx_count = cmds.polyEvaluate(jm, vertex=True)
                    if vtx_count > 0:
                        samples = min(vtx_count, 5)
                        for v in range(samples):
                            pos = cmds.xform(f"{jm}.vtx[{v}]", query=True, translation=True, worldSpace=True)
                            jaw_vertex_samples.append(f"{jm}.vtx[{v}]: {pos}")
                except:
                    pass

            os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
            with open(CURRENT_JAW_LOG_FILE, "a") as f:
                f.write("--- Jaw Rotation & Position Log ---\n")
                f.write(f"Sequence: {user_text}\n")
                f.write(f"At frame 0: Rotation {jaw_rot_start}\n")
                f.write(f"At frame {LAST_ANIMATION_END_FRAME} (after neutral pose is done): \n")
                f.write(f"  jaw01 Rotation: {jaw_rot_end}\n")
                f.write(f"  jaw01 Position: {jaw_pos_end}\n")
                f.write(f"\nSampled Jaw Mesh Vertices at frame {LAST_ANIMATION_END_FRAME}:\n")
                if not jaw_vertex_samples:
                    f.write("  No valid jaw meshes with vertices found.\n")
                for sample in jaw_vertex_samples:
                    f.write(f"  {sample}\n")
                    
            print(f"[JAW LOG] Wrote jaw rotation log to {CURRENT_JAW_LOG_FILE}\n")
            # ----------------------------------

            # --- ZIVA RELIABLE SIMULATION DEBUG RUN ---
            # The user requested that we log and plot the vertex positions natively during the script execution.
            # Since step_timeline_forward already collected all this live data natively, we just instantly plot it!
            print("\n" + "="*60)
            print("RUNNING ZIVA RELIABLE DEBUG VISUALIZATION (Plotting Pre-collected Live Data)")
            print("="*60)
            try:
                import matplotlib.pyplot as plt
                
                target_mesh = "middleres_OrbicularisOris_RENDER"
                if not cmds.objExists(target_mesh):
                    target_mesh = "middleres_OrbicularisOris"
                vertex_idx = 0
                
                if LIVE_DEBUG_FRAMES:
                    # === PLOT THE DATA ===
                    plt.figure(figsize=(15, 6))
                    plt.plot(LIVE_DEBUG_FRAMES, LIVE_DEBUG_POSITIONS, marker='.', linestyle='-', color='b', markersize=2, label=f'Vertex Y Position ({target_mesh})')
                    
                    plt.title(f"Ziva Live Sequential Debug: {target_mesh} Vertex {vertex_idx}")
                    plt.xlabel("Frame")
                    plt.ylabel("World Y Position")
                    plt.grid(True)
                    plt.legend()
                    
                    debug_dir = str(a3_stochastic_relative_path / "b99-frame1to4-get-b1to3"/ "debug_results")
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    plot_path = os.path.join(debug_dir, f"FULL_vertex_morph_plot_{timestamp_str}.png")
                    # plt.savefig(plot_path) # DISABLED to prevent matplotlib crash
                    # print(f"\n[DEBUG] Successfully saved reliable sequential plot to: {plot_path}")
                    plt.close()
                    
                    # ALSO save the raw data to CSV!
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
                        # visualize_errors(DEBUG_DATA) # DISABLED to prevent crash
                        print(f"[DEBUG] Visualization disabled by script update")
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

            if RENDER_AFTER_TYPEIN_ENABLED:
                print("[RENDER] Type-In Render enabled! Running Arnold render and video generation...")
                render_name = None
                if cmds.textFieldGrp('render_typein_name_field', exists=True):
                    render_name = cmds.textFieldGrp('render_typein_name_field', query=True, text=True)
                    if not render_name.strip(): render_name = None
                final_video = run_arnold_render_and_video(initial_start_frame, LAST_ANIMATION_END_FRAME, render_name=render_name, camera_name="face_closeup", width=640, height=480)
            else:
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
            checkpoint_file_path = os.path.join(base_path, sequence_folder_name, "combined_checkpoints.pkl")

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

            if RENDER_AFTER_TYPEIN_ENABLED:
                print("[RENDER] Type-In Render enabled! Running Arnold render and video generation...")
                render_name = None
                if cmds.textFieldGrp('render_typein_name_field', exists=True):
                    render_name = cmds.textFieldGrp('render_typein_name_field', query=True, text=True)
                    if not render_name.strip(): render_name = None
                final_video = run_arnold_render_and_video(start_frame, final_frame, render_name=render_name, camera_name="face_closeup", width=640, height=480)
            else:
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

def reset_animation_chain(*args, **kwargs):
    """Reset animation chaining to start from neutral by deleting animation keys and initializing solvers"""
    global LAST_ANIMATION_END_FRAME, LAST_ANIMATION_PHONEME, CONTINUE_FROM_LAST, ACCUMULATED_ERROR, DEBUG_DATA, KEYBOARD_CURRENT_PHONEME

    # Reset global variables
    LAST_ANIMATION_END_FRAME = 14
    LAST_ANIMATION_PHONEME = "dataV22_AU0_neutral"
    KEYBOARD_CURRENT_PHONEME = "dataV22_AU0_neutral"
    CONTINUE_FROM_LAST = False
    ACCUMULATED_ERROR = None  # Reset feedback control error
    DEBUG_DATA.clear()  # Clear debug data
    if cmds.checkBox('continue_checkbox', exists=True):
        cmds.checkBox('continue_checkbox', edit=True, value=False)

    print("[RESET] Running delete_head_keys.py to clean hierarchy...")
    import sys
    if str(a3_stochastic_relative_path / "automation") not in sys.path:
        sys.path.insert(0, str(a3_stochastic_relative_path / "automation"))
    import delete_head_keys
    import importlib
    importlib.reload(delete_head_keys)
    
    delete_head_keys.select_head_joint_hierarchy()
    delete_head_keys.keep_only_frames_0_and_14_on_selected()
    cmds.select(clear=True)

    print("[RESET] Cleaning up Ziva solvers and blendshapes...")
    valid_soft_tissues = [m for m in SOFT_TISSUE_MESHES if cmds.objExists(m)]
    valid_teeth_tongue = [m for m in get_teeth_tongue_meshes() if cmds.objExists(m)]
    all_cleanup_meshes = valid_soft_tissues + valid_teeth_tongue
    for mesh in all_cleanup_meshes:
        bs_name = f"master_bs_{mesh}"
        if cmds.objExists(bs_name):
            cmds.cutKey(bs_name, time=(14.01, 1000000), clear=True)
            if cmds.attributeQuery(mesh, node=bs_name, exists=True):
                cmds.setAttr(f"{bs_name}.{mesh}", 1.0)

        render_mesh = f"{mesh}_RENDER"
        if cmds.objExists(render_mesh):
            cmds.cutKey(render_mesh, time=(14.01, 1000000), clear=True)
            bs_render = f"master_bs_{render_mesh}"
            if cmds.objExists(bs_render):
                cmds.cutKey(bs_render, time=(14.01, 1000000), clear=True)
            if cmds.attributeQuery("visibility", node=render_mesh, exists=True):
                cmds.setAttr(f"{render_mesh}.visibility", 1)
            if cmds.attributeQuery("visibility", node=mesh, exists=True):
                cmds.setAttr(f"{mesh}.visibility", 0)
        else:
            if cmds.attributeQuery("visibility", node=mesh, exists=True):
                cmds.setAttr(f"{mesh}.visibility", 1)
                
        snapshots = cmds.ls(f"{mesh}_sagged_snapshot_*") + cmds.ls(f"{mesh}_equil_snapshot_*")
        if snapshots:
            cmds.delete(snapshots)
                
    solvers = cmds.ls(type='zSolver')
    for solver_shape in solvers:
        solver_transform = cmds.listRelatives(solver_shape, parent=True)[0]
        
        if cmds.attributeQuery('startFrame', node=solver_transform, exists=True):
            cmds.cutKey(solver_transform, attribute='startFrame', time=(0, 1000000), clear=True)
            cmds.setAttr(f"{solver_transform}.startFrame", 1) 
            
        if cmds.attributeQuery('enable', node=solver_shape, exists=True):
            cmds.cutKey(solver_shape, attribute='enable', time=(0, 1000000), clear=True)
            cmds.setAttr(f"{solver_shape}.enable", 1)
        
    nucleus_nodes = cmds.ls(type='nucleus')
    for nucleus in nucleus_nodes:
        if cmds.attributeQuery('startFrame', node=nucleus, exists=True):
            cmds.cutKey(nucleus, attribute='startFrame', time=(0, 1000000), clear=True)
            cmds.setAttr(f"{nucleus}.startFrame", 1)
            
    # Adjust timeline minTime/maxTime and jump to frame 14
    cmds.playbackOptions(minTime=0, maxTime=50)
    cmds.currentTime(14, edit=True)
    update_status_display()
    
    try:
        update_keyboard_status_display()
    except NameError:
        pass

    print("[UI] Animation chain reset. Next animation will start from frame 14.")
    
    silent = kwargs.get('silent', False)
    if not silent:
        cmds.confirmDialog(title='Reset', message='Animation chain reset!\n\nAll animation keys >14 and physics states have been initialized.\nNext animation will start from frame 14.')

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
        # --- AUTOMATE JUMP TO 0 IF STARTING FROM NEUTRAL ---
        if KEYBOARD_CURRENT_PHONEME == "dataV22_AU0_neutral":
            print("[AUTO-RESET] Starting from Neutral. Automatically resetting animation chain and jumping to frame 0 for Ziva init.")
            reset_animation_chain(silent=True)
            try:
                import maya.cmds as cmds
                cmds.currentTime(0, edit=True)
            except Exception:
                pass
            start_frame = 14
        else:
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
        
        if cmds.checkBox('auto_playblast_checkbox', exists=True) and cmds.checkBox('auto_playblast_checkbox', query=True, value=True):
            run_playblast_script()
            
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


import threading
import maya.utils

CHORD_BUFFER = []
CHORD_TIMER = None
CHORD_WINDOW_SEC = 0.4

def process_chord_buffer():
    global CHORD_TIMER, CHORD_BUFFER, KEYBOARD_CURRENT_PHONEME
    
    keys = list(CHORD_BUFFER)
    CHORD_BUFFER.clear()
    CHORD_TIMER = None
    
    if not keys:
        return
        
    # Extract unique keys in order
    unique_keys = []
    for k in keys:
        if k not in unique_keys:
            unique_keys.append(k)
            
    # Handle neutral clear
    if '0' in unique_keys:
        unique_keys = ['0']
            
    if len(unique_keys) == 1:
        # Standard single key
        def _exec_single():
            keyboard_transition_to_phoneme(unique_keys[0])
        maya.utils.executeDeferred(_exec_single)
        return
        
    # Chord logic
    # Group by octave, pick the last pressed for each octave
    octave_picks = {}
    for k in unique_keys:
        if k in KEYBOARD_PHONEME_MAP:
            octave = KEYBOARD_PHONEME_MAP[k].get('octave', 'mid')
            octave_picks[octave] = KEYBOARD_PHONEME_MAP[k]['phoneme']
            
    targets = list(octave_picks.values())
    
    # Expand any targets that are already chords (e.g. happy=chord_...)
    expanded_targets = []
    for t in targets:
        if t.startswith("chord_"):
            expanded_targets.extend(t[6:].split("___"))
        else:
            expanded_targets.append(t)
            
    # De-duplicate expanded targets
    unique_targets = []
    for t in expanded_targets:
        if t not in unique_targets:
            unique_targets.append(t)
            
    if len(unique_targets) == 1:
        def _exec_single_after_resolve():
            target_p = unique_targets[0]
            # Find key for this phoneme
            k_char = None
            for key, info in KEYBOARD_PHONEME_MAP.items():
                if info['phoneme'] == target_p:
                    k_char = key
                    break
            if k_char:
                keyboard_transition_to_phoneme(k_char)
        maya.utils.executeDeferred(_exec_single_after_resolve)
        return
        
    # Execute merged transition
    combined_pose_name = "chord_" + "___".join(unique_targets)
    
    def _exec_chord():
        global LAST_ANIMATION_END_FRAME, LAST_ANIMATION_PHONEME, KEYBOARD_CURRENT_PHONEME
        current = LAST_ANIMATION_PHONEME
        if current == combined_pose_name:
            print("[CHORD] Already at chord pose.")
            return
            
        print(f"\n{'='*50}")
        print(f"[CHORD DETECTED] {keys} -> Resolved: {list(octave_picks.keys())}")
        print(f"[CHORD] Targets: {unique_targets}")
        print(f"{'='*50}")
        
        try:
            start_frame = LAST_ANIMATION_END_FRAME
            if current == "dataV22_AU0_neutral":
                print("[AUTO-RESET] Starting from Neutral. Reseting.")
                reset_animation_chain(silent=True)
                start_frame = 14
                try:
                    import maya.cmds as cmds
                    cmds.currentTime(0, edit=True)
                except Exception:
                    pass
                    
            final_frame = smart_single_transition(
                current, combined_pose_name, start_frame,
                is_first_transition=False
            )
            
            LAST_ANIMATION_END_FRAME = final_frame
            LAST_ANIMATION_PHONEME = combined_pose_name
            KEYBOARD_CURRENT_PHONEME = combined_pose_name
            
            # Try to show a combined label
            label = " + ".join([KEYBOARD_PHONEME_MAP[k]['label'] for k in keys if k in KEYBOARD_PHONEME_MAP])
            if cmds.text('keyboard_status_text', exists=True):
                cmds.text('keyboard_status_text', edit=True, 
                         label=f"Current: [Chord] {label}  |  Frame: {LAST_ANIMATION_END_FRAME}")
                         
            if cmds.checkBox('auto_playblast_checkbox', exists=True) and cmds.checkBox('auto_playblast_checkbox', query=True, value=True):
                run_playblast_script()
                
        except Exception as e:
            print(f"[CHORD ERROR] {e}")
            import traceback
            traceback.print_exc()
            
    maya.utils.executeDeferred(_exec_chord)


def on_keyboard_input_changed(*args):
    """
    Called when the keyboard input field changes.
    Appends typed keys to CHORD_BUFFER and starts/resets a timer to detect chords.
    """
    global KEYBOARD_CURRENT_PHONEME, CHORD_TIMER, CHORD_BUFFER
    
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
            CHORD_BUFFER.append(last_char)
            if CHORD_TIMER is not None:
                CHORD_TIMER.cancel()
            CHORD_TIMER = threading.Timer(CHORD_WINDOW_SEC, process_chord_buffer)
            CHORD_TIMER.start()
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
            if KEYBOARD_CURRENT_PHONEME.startswith("chord_"):
                current_label = "Chord"
                current_key = "*"
            else:
                current_label = PHONEME_TO_KEY.get(KEYBOARD_CURRENT_PHONEME, {}).get('label', 'Unknown')
                current_key = PHONEME_TO_KEY.get(KEYBOARD_CURRENT_PHONEME, {}).get('key', '?')
            cmds.text('keyboard_status_text', edit=True, 
                     label=f"Current: [{current_key}] {current_label}  |  Frame: {LAST_ANIMATION_END_FRAME}")
    except:
        pass


# --- SMART TRANSITION HELPER ---
def smart_single_transition(current_pose, target_pose, start_frame, is_first_transition=False):
    """
    Attempts to transition from current to target_pose.
    If the direct path doesn't exist, it falls back to current -> neutral -> target_pose.
    """
    neutral = "dataV22_AU0_neutral"

    # Handle returning to neutral from a combined expression pose
    if target_pose == neutral and is_combined_expression_pose(current_pose):
        return run_merged_return_to_neutral(current_pose, start_frame, is_first_transition=is_first_transition)

    # Handle transitioning from neutral to a combined expression pose
    if current_pose == neutral and is_combined_expression_pose(target_pose):
        if target_pose.startswith("chord_"):
            targets = target_pose[6:].split("___")
        else:
            short = target_pose.replace('dataV22_', '')
            parts = short.split('_')
            targets = [f"dataV22_{p}" for p in parts if p.upper().startswith('AU') and p.upper() != 'AU0']
        return run_merged_transition(targets, start_frame, continue_mode=True, is_first_transition=is_first_transition)

    task_string = f"{current_pose} to {target_pose}"
    normalized_key = normalize_task_key(task_string)
    db_key_map = {normalize_task_key(k): k for k in DEFORMATION_PATH_DB}
    
    # Check if path exists or if it's already a transition involving neutral
    if normalized_key in db_key_map or current_pose == neutral or target_pose == neutral:
        return run_single_transition(current_pose, target_pose, start_frame, continue_mode=True, is_first_transition=is_first_transition)
    
    # Path doesn't exist, and neither is neutral. Do fallback.
    c_short = current_pose.replace('dataV22_', '')
    t_short = target_pose.replace('dataV22_', '')
    
    print(f"\n[WARNING] Path '{c_short} -> {t_short}' does not exist!")
    print(f"[INFO] Routing through Neutral pose: {c_short} -> Neutral -> {t_short}")
    
    # Step 1: current to neutral
    if is_combined_expression_pose(current_pose):
        intermediate_frame = run_merged_return_to_neutral(
            current_pose, start_frame, is_first_transition=is_first_transition
        )
    else:
        intermediate_frame = run_single_transition(
            current_pose, neutral, start_frame,
            continue_mode=True, is_first_transition=is_first_transition
        )
    
    # Step 2: neutral to target
    if is_combined_expression_pose(target_pose):
        if target_pose.startswith("chord_"):
            targets = target_pose[6:].split("___")
        else:
            short = target_pose.replace('dataV22_', '')
            parts = short.split('_')
            targets = [f"dataV22_{p}" for p in parts if p.upper().startswith('AU') and p.upper() != 'AU0']
        final_frame = run_merged_transition(
            targets, intermediate_frame, continue_mode=True, is_first_transition=False
        )
    else:
        final_frame = run_single_transition(
            neutral, target_pose, intermediate_frame,
            continue_mode=True, is_first_transition=False
        )
    
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
        # --- AUTOMATE JUMP TO 0 IF STARTING FROM NEUTRAL ---
        if current == "dataV22_AU0_neutral":
            print("[AUTO-RESET] Starting from Neutral. Automatically resetting animation chain and jumping to frame 0 for Ziva init.")
            reset_animation_chain(silent=True)
            try:
                import maya.cmds as cmds
                cmds.currentTime(0, edit=True)
            except Exception:
                pass
            start_frame = 14
        else:
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
    """Run an expression command, automatically merging multi-AU expressions into ONE single transition."""
    global LAST_ANIMATION_END_FRAME, LAST_ANIMATION_PHONEME, KEYBOARD_CURRENT_PHONEME

    if command_key not in EXPRESSION_COMMANDS:
        print(f"[CLICK ERROR] Unknown expression: {command_key}")
        return

    sequence = EXPRESSION_COMMANDS[command_key]
    targets = [p for p in sequence if p != "dataV22_AU0_neutral" and p != "Neutral"]
    if not targets:
        if LAST_ANIMATION_PHONEME == "dataV22_AU0_neutral":
            print(f"[CLICK] Already at expression '{command_key}'")
            return
        else:
            print(f"\n{'='*50}")
            print(f"[EXPRESSION] '{command_key}' (Returning to Neutral)")
            print(f"{'='*50}")
            try:
                start_frame = LAST_ANIMATION_END_FRAME
                final_frame = smart_single_transition(
                    LAST_ANIMATION_PHONEME, "dataV22_AU0_neutral", start_frame,
                    is_first_transition=False
                )
                # Let the return animation play out. 
                # (final_frame is left exactly as returned by smart_single_transition)
                LAST_ANIMATION_END_FRAME = final_frame
                LAST_ANIMATION_PHONEME = "dataV22_AU0_neutral"
                KEYBOARD_CURRENT_PHONEME = "dataV22_AU0_neutral"
                update_status_display()
                print(f"[EXPRESSION] Returned to Neutral (frame 0)")
                if cmds.checkBox('auto_playblast_checkbox', exists=True) and cmds.checkBox('auto_playblast_checkbox', query=True, value=True):
                    run_playblast_script()
            except Exception as e:
                print(f"[EXPRESSION ERROR] {e}")
                import traceback
                traceback.print_exc()
            return

    print(f"\n{'='*50}")
    print(f"[EXPRESSION] '{command_key}' ({len(targets)} target AUs -> Merged in ONE transition)")
    print(f"{'='*50}")

    try:
        # --- AUTOMATE JUMP TO 0 IF STARTING FROM NEUTRAL ---
        if LAST_ANIMATION_PHONEME == "dataV22_AU0_neutral":
            print("[AUTO-RESET] Starting from Neutral. Automatically resetting animation chain and jumping to frame 0 for Ziva init.")
            reset_animation_chain(silent=True)
            try:
                import maya.cmds as cmds
                cmds.currentTime(0, edit=True)
            except Exception:
                pass
            start_frame = 14
        else:
            start_frame = LAST_ANIMATION_END_FRAME
        if len(targets) == 1:
            target_pose = targets[0]
            final_frame = smart_single_transition(
                LAST_ANIMATION_PHONEME, target_pose, start_frame,
                is_first_transition=False
            )
            combined_pose_name = target_pose
        else:
            final_frame, combined_pose_name = smart_merged_expression_transition(
                LAST_ANIMATION_PHONEME, targets, start_frame,
                is_first_transition=False, expression_name=command_key
            )

        LAST_ANIMATION_END_FRAME = final_frame
        LAST_ANIMATION_PHONEME = combined_pose_name
        KEYBOARD_CURRENT_PHONEME = combined_pose_name
        update_status_display()
        print(f"[EXPRESSION] Done: '{command_key}' -> Pose: '{combined_pose_name}'")
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
        focus_viewport()

def handle_facs_selection(label, internal):
    if internal:
        dropdown_immediate_transition(internal, label)
        focus_viewport()

def handle_emotion_selection(label, command_key):
    if command_key:
        dropdown_immediate_expression(command_key)
        focus_viewport()

def toggle_render_after_demo(*args):
    """Toggle the render-after-demo checkbox"""
    global RENDER_AFTER_DEMO_ENABLED
    RENDER_AFTER_DEMO_ENABLED = cmds.checkBox('render_after_demo_checkbox', query=True, value=True)
    print(f"[UI] Render after demo: {RENDER_AFTER_DEMO_ENABLED}")


def toggle_render_after_typein(*args):
    """Toggle the render-after-typein checkbox"""
    global RENDER_AFTER_TYPEIN_ENABLED
    RENDER_AFTER_TYPEIN_ENABLED = cmds.checkBox('render_after_typein_checkbox', query=True, value=True)
    print(f"[UI] Render after type-in: {RENDER_AFTER_TYPEIN_ENABLED}")


def run_arnold_render_and_video(start_frame, end_frame, render_name=None, camera_name='side view camera V2', width=680, height=680):
    """
    Renders the frame range using Arnold (already configured in the scene) with
    the specified camera, then converts the PNG sequence to an MP4
    video using ffmpeg via generate_ffmpeg_commands.py logic.

    Args:
        start_frame: First frame to render
        end_frame: Last frame to render
        render_name: Base name for the render output. If None, uses 'render_YYYYMMDD_HHMMSS'.
        camera_name: Name of the camera to use for rendering.
        width: Render width.
        height: Render height.
    """
    import subprocess
    import glob as glob_module
    import math as math_module

    # --- Determine render name ---
    if not render_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        render_name = f"render_{timestamp}"

    print(f"\n{'='*60}")
    print(f"[RENDER] Starting Arnold Render: '{render_name}'")
    print(f"[RENDER] Frames: {start_frame} -> {end_frame}")
    print(f"[RENDER] Camera: {camera_name}")
    print(f"{'='*60}\n")

    # --- Configure render settings ---
    # Try common naming variants
    camera_candidates = [camera_name, camera_name.replace('_', ' '), camera_name.replace(' ', '_')]
    found_camera = None
    for cam in camera_candidates:
        if cmds.objExists(cam):
            found_camera = cam
            break
    # Also search by partial match
    if found_camera is None:
        all_cameras = cmds.ls(type='camera')
        for cam_shape in all_cameras:
            cam_transform = cmds.listRelatives(cam_shape, parent=True)
            if cam_transform:
                cam_name = cam_transform[0]
                if camera_name.lower().replace('_', '').replace(' ', '') in cam_name.lower().replace('_', '').replace(' ', ''):
                    found_camera = cam_name
                    break

    if found_camera is None:
        print(f"[RENDER WARNING] Could not find '{camera_name}'. Using current renderable camera.")
    else:
        print(f"[RENDER] Using camera: {found_camera}")

    # Set the file name prefix for Arnold output
    # Get the default render globals
    cmds.setAttr('defaultRenderGlobals.imageFilePrefix', render_name, type='string')

    # Set frame range
    cmds.setAttr('defaultRenderGlobals.startFrame', start_frame)
    cmds.setAttr('defaultRenderGlobals.endFrame', end_frame)
    cmds.setAttr('defaultRenderGlobals.animation', 1)
    cmds.setAttr('defaultRenderGlobals.putFrameBeforeExt', 1)
    cmds.setAttr('defaultRenderGlobals.extensionPadding', 4)
    cmds.setAttr('defaultRenderGlobals.outFormatControl', 0)

    # Ensure image format is PNG (32 = PNG in Maya)
    cmds.setAttr('defaultRenderGlobals.imageFormat', 32)

    # --- Run batch render frame by frame ---
    print(f"[RENDER] Rendering {end_frame - start_frame + 1} frames...")
    for frame in range(start_frame, end_frame + 1):
        cmds.currentTime(frame, edit=True)
        if found_camera:
            cmds.render(found_camera, x=width, y=height)
        else:
            cmds.render(x=width, y=height)
        if frame % 10 == 0 or frame == end_frame:
            print(f"[RENDER] Frame {frame}/{end_frame} done.")

    print(f"[RENDER] Arnold render complete!")

    # --- Determine where Arnold wrote the PNGs ---
    # Arnold outputs to the project's images directory by default
    workspace_root = cmds.workspace(query=True, rootDirectory=True)
    images_dir = os.path.join(workspace_root, 'images')

    # Search for the rendered PNGs
    search_pattern = os.path.join(images_dir, '**', f'{render_name}*', '**', '*.png')
    rendered_files = glob_module.glob(search_pattern, recursive=True)

    # Also try flat search in images directory and subdirectories
    if not rendered_files:
        search_pattern = os.path.join(images_dir, '**', f'{render_name}*.png')
        rendered_files = glob_module.glob(search_pattern, recursive=True)

    if not rendered_files:
        # Try the temp_rendered_png_hub fallback
        fallback_dir = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub\temp_rendered_png_hub"
        search_pattern = os.path.join(fallback_dir, '**', f'{render_name}*.png')
        rendered_files = glob_module.glob(search_pattern, recursive=True)

    if not rendered_files:
        print(f"[RENDER WARNING] No rendered PNG files found. Skipping video generation.")
        print(f"[RENDER] Searched in: {images_dir}")
        return None

    # Find the directory containing the PNGs
    png_dir = os.path.dirname(rendered_files[0])
    print(f"[RENDER] Found {len(rendered_files)} rendered PNGs in: {png_dir}")

    # --- Generate MP4 from PNG sequence using ffmpeg ---
    # Determine the filename pattern (Arnold uses: renderName_frameNumber.png)
    # Find frame number range
    frame_numbers = []
    for f in rendered_files:
        basename = os.path.basename(f)
        # Try to extract frame number from filename
        parts = basename.replace('.png', '').split('.')
        for part in parts:
            try:
                frame_numbers.append(int(part))
            except ValueError:
                continue
        if not frame_numbers:
            # Try underscore-separated format
            parts2 = basename.replace('.png', '').split('_')
            for part in parts2:
                try:
                    frame_numbers.append(int(part))
                except ValueError:
                    continue

    if not frame_numbers:
        print(f"[RENDER WARNING] Could not parse frame numbers from PNG filenames. Skipping video.")
        return None

    min_frame = min(frame_numbers)
    max_frame = max(frame_numbers)
    total_frames = max_frame - min_frame + 1

    # Calculate framerate for ~5 seconds of video
    desired_video_length = 5.0
    framerate = max(1, math_module.ceil(total_frames / desired_video_length))

    # Determine the ffmpeg input pattern
    # Arnold typically outputs: renderName.NNNN.png or renderName_NNNN.png
    sample_file = os.path.basename(rendered_files[0])
    if '.' in sample_file.rsplit('.png', 1)[0]:
        # Dot-separated: renderName.0014.png
        input_pattern = os.path.join(png_dir, f"{render_name}.%04d.png")
    else:
        # Underscore-separated: renderName_0014.png
        input_pattern = os.path.join(png_dir, f"{render_name}_%04d.png")

    # Output video path
    video_output_dir = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub"
    os.makedirs(video_output_dir, exist_ok=True)
    video_output_path = os.path.join(video_output_dir, f"{render_name}.mp4")

    # Build and run ffmpeg command
    ffmpeg_cmd = (
        f'ffmpeg -y -framerate {framerate} -start_number {min_frame} '
        f'-i "{input_pattern}" '
        f'-c:v libx264 -pix_fmt yuv420p -crf 23 -preset medium '
        f'"{video_output_path}"'
    )

    print(f"\n[RENDER] Generating video with ffmpeg...")
    print(f"[RENDER] Command: {ffmpeg_cmd}")

    try:
        result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"[RENDER] Video saved to: {video_output_path}")
            return video_output_path
        else:
            print(f"[RENDER ERROR] ffmpeg failed: {result.stderr}")
            return None
    except FileNotFoundError:
        print(f"[RENDER ERROR] ffmpeg not found. Please ensure ffmpeg is in your PATH.")
        return None
    except subprocess.TimeoutExpired:
        print(f"[RENDER ERROR] ffmpeg timed out.")
        return None


def run_not_even_you_demo(*args):
    """
    Demo button: Runs the 'not even you' sentence deformation using a pre-computed
    checkpoint file from the DSAC_V2 results. If the Teeth Hack checkbox is enabled,
    the teeth/tongue rotation hack will be applied automatically during deformation.
    If the Render checkbox is enabled, runs Arnold render + ffmpeg video afterward.
    """
    global LAST_ANIMATION_END_FRAME, LAST_ANIMATION_PHONEME, KEYBOARD_CURRENT_PHONEME
    global ACCUMULATED_ERROR, LIVE_DEBUG_FRAMES, LIVE_DEBUG_POSITIONS
    import time

    # The pre-computed checkpoint path for "not even you"
    demo_checkpoint_dir = r"\\bbysvm-nfs1.its.sfu.ca\GrUViLab\Data\richardPeng\keyLabCloud\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003_25-7-18\job_250620-191133\250620-191151"
    checkpoint_file = os.path.join(demo_checkpoint_dir, "combined_checkpoints.pkl")

    if not os.path.exists(checkpoint_file):
        cmds.warning(f"Demo checkpoint not found: {checkpoint_file}")
        return

    print(f"\n{'='*60}")
    print(f"DEMO: 'Not Even You' (Teeth Hack: {TEETH_HACK_ENABLED})")
    print(f"Checkpoint: {demo_checkpoint_dir}")
    print(f"{'='*60}\n")

    pipeline_start = time.time()

    try:
        # Reset state for a fresh demo run
        ACCUMULATED_ERROR = None
        LIVE_DEBUG_FRAMES.clear()
        LIVE_DEBUG_POSITIONS.clear()

        # The phoneme sequence for "not even you":
        # Neutral -> N -> AA -> T -> IY -> V -> N -> Y -> UW
        demo_phonemes = [
            "dataV22_AU0_neutral",
            "dataV22_phoneme_N",
            "dataV22_phoneme_AA",
            "dataV22_phoneme_T",
            "dataV22_phoneme_IY",
            "dataV22_phoneme_V",
            "dataV22_phoneme_N",
            "dataV22_phoneme_Y",
            "dataV22_phoneme_UW",
        ]

        # --- CREATE PROXY RENDER MESHES IMMEDIATELY ---
        valid_soft_tissues = [m for m in SOFT_TISSUE_MESHES if cmds.objExists(m)]
        valid_teeth_tongue = [m for m in get_teeth_tongue_meshes() if cmds.objExists(m)]
        all_proxy_targets = valid_soft_tissues + valid_teeth_tongue
        needs_proxy = any(not cmds.objExists(f"{m}_RENDER") for m in all_proxy_targets)
        if needs_proxy:
            print(f"[DEMO] Creating Proxy Render Meshes at Frame 0...")
            cmds.currentTime(0, edit=True)
            for target_mesh in all_proxy_targets:
                render_mesh = f"{target_mesh}_RENDER"
                bs_name = f"master_bs_{target_mesh}"
                if not cmds.objExists(render_mesh):
                    render_mesh_node = cmds.duplicate(target_mesh, name=render_mesh)[0]
                    if cmds.attributeQuery('visibility', node=target_mesh, exists=True):
                        cmds.setAttr(f"{target_mesh}.visibility", 0)
                    if cmds.attributeQuery('visibility', node=render_mesh_node, exists=True):
                        cmds.setAttr(f"{render_mesh_node}.visibility", 1)
                    if not cmds.objExists(bs_name):
                        cmds.blendShape([target_mesh], render_mesh_node, name=bs_name, origin='local')
                        cmds.setAttr(f"{bs_name}.{target_mesh}", 1.0)
        else:
            for target_mesh in all_proxy_targets:
                render_mesh = f"{target_mesh}_RENDER"
                if cmds.objExists(render_mesh):
                    if cmds.attributeQuery('visibility', node=target_mesh, exists=True):
                        cmds.setAttr(f"{target_mesh}.visibility", 0)
                    if cmds.attributeQuery('visibility', node=render_mesh, exists=True):
                        cmds.setAttr(f"{render_mesh}.visibility", 1)

        # Run the deformation directly from the pre-computed checkpoint
        start_frame = 14
        final_frame, per_frame_data = perform_the_deformation(
            demo_checkpoint_dir, start_frame, continue_from_last=False, correction_vectors=None
        )

        # Update global state
        LAST_ANIMATION_END_FRAME = final_frame
        LAST_ANIMATION_PHONEME = demo_phonemes[-1]
        KEYBOARD_CURRENT_PHONEME = demo_phonemes[-1]
        update_status_display()

        elapsed = time.time() - pipeline_start
        print(f"\n{'='*60}")
        print(f"DEMO COMPLETE: 'Not Even You'")
        print(f"Frames: 14 -> {final_frame}  |  Time: {elapsed:.1f}s")
        print(f"Teeth Hack was {'ENABLED' if TEETH_HACK_ENABLED else 'DISABLED'}")
        print(f"{'='*60}\n")

        # Auto-playblast if enabled
        if cmds.checkBox('auto_playblast_checkbox', exists=True) and cmds.checkBox('auto_playblast_checkbox', query=True, value=True):
            run_playblast_script()

        # --- ARNOLD RENDER + VIDEO if enabled ---
        if RENDER_AFTER_DEMO_ENABLED:
            # Get custom render name from the UI field
            render_name = ""
            if cmds.textFieldGrp('render_sequence_name_field', exists=True):
                render_name = cmds.textFieldGrp('render_sequence_name_field', query=True, text=True).strip()
            # If blank, default to render_YYYYMMDD_HHMMSS
            if not render_name:
                render_name = None  # run_arnold_render_and_video will generate the timestamp name

            video_path = run_arnold_render_and_video(start_frame, final_frame, render_name=render_name)
            if video_path:
                print(f"[DEMO] Final video: {video_path}")

    except Exception as e:
        print(f"[DEMO ERROR] {e}")
        import traceback
        traceback.print_exc()


def toggle_teeth_hack(*args):
    """Toggle the teeth/tongue rotation hack based on checkbox"""
    global TEETH_HACK_ENABLED
    TEETH_HACK_ENABLED = cmds.checkBox('teeth_hack_checkbox', query=True, value=True)
    print(f"[UI] Teeth/Tongue rotation hack enabled: {TEETH_HACK_ENABLED}")
    if TEETH_HACK_ENABLED:
        try:
            ensure_teeth_tongue_render_meshes(at_frame_zero=False)
        except Exception as e:
            print(f"[WARNING] Could not initialize teeth/tongue render meshes: {e}")

def toggle_verbose_mode(*args):
    global VERBOSE_MODE
    VERBOSE_MODE = cmds.checkBox('verbose_mode_checkbox', query=True, value=True)
    _original_print(f"[UI] Verbose mode: {VERBOSE_MODE}")

def toggle_finger_clap(*args):
    global FINGER_CLAP_ENABLED
    FINGER_CLAP_ENABLED = cmds.checkBox('finger_clap_checkbox', query=True, value=True)
    print(f"[UI] Finger Clap Effect: {FINGER_CLAP_ENABLED}")

def run_finger_clap_test(*args):
    # 1. Query the UI state
    auto_playblast = cmds.checkBox('auto_playblast_checkbox', query=True, value=True)

    # 2. Add automation directory to system path if missing
    import sys
    automation_path = str(a3_stochastic_relative_path / "automation")
    if automation_path not in sys.path:
        sys.path.insert(0, automation_path)

    # 3. Import and reload simulation module
    import passive_dynamics_clap_test
    import importlib
    importlib.reload(passive_dynamics_clap_test)

    print("[UI] Triggering passive dynamics test runner...")

    # 4. Execute the simulation (which now internally handles the PBD and push operations)
    try:
        passive_dynamics_clap_test.main(auto_playblast=auto_playblast)
    except Exception as e:
        print("[PBD ERROR] Main simulation run failed:", e)

def precompute_keyboard_transitions(*args):
    """
    Precomputes all combined_checkpoints.pkl files required for keyboard transitions.
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
                
            checkpoint_file_path = os.path.join(source_path, "combined_checkpoints.pkl")
            transition_frames = get_transition_frames(p_from, p_to, base_frames=35)
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

def reset_and_run_pipeline(*args):
    """
    Developer utility: Resets the scene, waits 2 seconds, and then runs the type-in pipeline.
    """
    print("\n[DEV] Executing Reset and Run pipeline...")
    reset_animation_chain(silent=True)
    
    print("[DEV] Waiting 2 seconds for scene to settle...")
    # cmds.pause allows Maya to process internal deferred events during the sleep
    cmds.pause(sec=2)
    
    print("[DEV] Running Full Pipeline from Type-In field...")
    run_full_pipeline()

# ==========================================================
#  CAMERA CONTROLS
# ==========================================================

def focus_viewport(*args):
    """Returns focus to the 3D viewport so arrow keys don't navigate UI buttons"""
    import maya.cmds as cmds
    panel = cmds.getPanel(withFocus=True)
    if panel and 'modelPanel' in panel:
        return
        
    for p in cmds.getPanel(type='modelPanel') or []:
        cam = cmds.modelPanel(p, query=True, camera=True)
        if cam in ['persp', 'perspShape']:
            cmds.setFocus(p)
            return
            
    panels = cmds.getPanel(type='modelPanel')
    if panels:
        cmds.setFocus(panels[0])

def handle_arrow_hotkey(direction):
    """Global handler for arrow keys"""
    import maya.cmds as cmds
    
    # 1. Find correct panel
    panel = cmds.getPanel(withFocus=True)
    if not panel or 'modelPanel' not in panel:
        for p in cmds.getPanel(type='modelPanel') or []:
            cam = cmds.modelPanel(p, query=True, camera=True)
            if cam in ['persp', 'perspShape']:
                panel = p
                break
        if not panel:
            panels = cmds.getPanel(type='modelPanel')
            panel = panels[0] if panels else None

    if not panel:
        print("[CAMERA ERROR] No model panel found to rotate.")
        return
        
    cam = cmds.modelPanel(panel, query=True, camera=True)
    
    # Force focus to this panel
    cmds.setFocus(panel)
    
    # 2. Try to rotate
    amount = 15
    
    # We must operate on the transform node, not the shape node
    if cmds.objectType(cam) == 'camera':
        cam = cmds.listRelatives(cam, parent=True)[0]
        
    try:
        # Use exact camera node for orbit
        if direction == 'up':
            cmds.orbit(cam, verticalAngle=-amount)
        elif direction == 'down':
            cmds.orbit(cam, verticalAngle=amount)
        elif direction == 'left':
            cmds.orbit(cam, horizontalAngle=amount)
        elif direction == 'right':
            cmds.orbit(cam, horizontalAngle=-amount)
        cmds.refresh()
    except Exception as e:
        print(f"[CAMERA ERROR] Failed to orbit camera '{cam}': {e}")

def setup_camera_hotkeys(*args):
    """Binds arrow keys to camera tumbling"""
    import maya.cmds as cmds
    import time
    try:
        # Register handler to __main__ so hotkeys can find it
        import __main__
        __main__.handle_arrow_hotkey = handle_arrow_hotkey
        
        # We put the print statement directly in the string so we KNOW it fired, even if the function fails!
        py_up = 'import __main__; __main__.handle_arrow_hotkey("up")'
        py_dn = 'import __main__; __main__.handle_arrow_hotkey("down")'
        py_lf = 'import __main__; __main__.handle_arrow_hotkey("left")'
        py_rt = 'import __main__; __main__.handle_arrow_hotkey("right")'
        
        # Use unique names to avoid Maya's runtime command deletion bugs
        uid = int(time.time())
        rtc_up = f'camOrbitUpRTC_{uid}'
        rtc_dn = f'camOrbitDownRTC_{uid}'
        rtc_lf = f'camOrbitLeftRTC_{uid}'
        rtc_rt = f'camOrbitRightRTC_{uid}'
        
        cmds.runTimeCommand(rtc_up, annotation='Orbit Up', command=py_up, category='Custom', commandLanguage='python')
        cmds.runTimeCommand(rtc_dn, annotation='Orbit Down', command=py_dn, category='Custom', commandLanguage='python')
        cmds.runTimeCommand(rtc_lf, annotation='Orbit Left', command=py_lf, category='Custom', commandLanguage='python')
        cmds.runTimeCommand(rtc_rt, annotation='Orbit Right', command=py_rt, category='Custom', commandLanguage='python')
        
        cmds.nameCommand('camOrbitUpCmd', annotation='Orbit Up', command=rtc_up)
        cmds.nameCommand('camOrbitDownCmd', annotation='Orbit Down', command=rtc_dn)
        cmds.nameCommand('camOrbitLeftCmd', annotation='Orbit Left', command=rtc_lf)
        cmds.nameCommand('camOrbitRightCmd', annotation='Orbit Right', command=rtc_rt)
        
        cmds.hotkey(keyShortcut='Up', altModifier=True, name='camOrbitUpCmd')
        cmds.hotkey(keyShortcut='Down', altModifier=True, name='camOrbitDownCmd')
        cmds.hotkey(keyShortcut='Left', altModifier=True, name='camOrbitLeftCmd')
        cmds.hotkey(keyShortcut='Right', altModifier=True, name='camOrbitRightCmd')
        print("[CAMERA] Bound Alt+Arrow keys to camera orbit.")
    except Exception as e:
        print(f"[CAMERA SETUP ERROR] {e}")

def cleanup_camera_hotkeys(*args):
    """Restores default Maya arrow key behaviors"""
    import maya.cmds as cmds
    try:
        cmds.hotkey(keyShortcut='Up', altModifier=True, name='')
        cmds.hotkey(keyShortcut='Down', altModifier=True, name='')
        cmds.hotkey(keyShortcut='Left', altModifier=True, name='')
        cmds.hotkey(keyShortcut='Right', altModifier=True, name='')
        print("[CAMERA] Restored default Alt+Arrow keys.")
    except Exception as e:
        pass

def tumble_camera(direction):
    """Orbits the active viewport camera around its center of interest"""
    import maya.cmds as cmds
    panel = cmds.getPanel(withFocus=True)
    if not panel or 'modelPanel' not in panel:
        panels = cmds.getPanel(type='modelPanel')
        if panels: panel = panels[0]
        else: return
        
    cam = cmds.modelPanel(panel, query=True, camera=True)
    
    amount = 15  # degrees
    if direction == 'up':
        cmds.orbit(cam, verticalAngle=-amount)
    elif direction == 'down':
        cmds.orbit(cam, verticalAngle=amount)
    elif direction == 'left':
        cmds.orbit(cam, horizontalAngle=amount)
    elif direction == 'right':
        cmds.orbit(cam, horizontalAngle=-amount)
    cmds.refresh()

def set_camera_preset(preset, *args):
    """Moves the camera to a predefined preset view"""
    import maya.cmds as cmds
    
    panel = cmds.getPanel(withFocus=True)
    if not panel or 'modelPanel' not in panel:
        for p in cmds.getPanel(type='modelPanel') or []:
            cam = cmds.modelPanel(p, query=True, camera=True)
            if cam in ['persp', 'perspShape']:
                panel = p
                break
        if not panel:
            panels = cmds.getPanel(type='modelPanel')
            panel = panels[0] if panels else None
            
    if not panel:
        print(f"[CAMERA PRESET ERROR] No valid model panel found.")
        return
        
    cam = cmds.modelPanel(panel, query=True, camera=True)
    
    # Ensure we operate on the transform node, not the shape node
    if cmds.objectType(cam) == 'camera':
        cam = cmds.listRelatives(cam, parent=True)[0]
    
    # Identify what to frame. DO NOT select jaw01 (it will frame the inside of the teeth)
    target = None
    if cmds.objExists("head_grp"):
        target = "head_grp"
    elif cmds.objExists("head_mesh"):
        target = "head_mesh"
        
    if target:
        cmds.select(target)
    else:
        meshes = cmds.ls(type='mesh')
        if meshes:
            # Select transforms of meshes
            transforms = cmds.listRelatives(meshes, parent=True)
            if transforms:
                cmds.select(transforms)
        else:
            cmds.select(all=True)
            
    if preset == 'frame':
        cmds.viewFit(cam)
        cmds.select(clear=True)
        focus_viewport()
        return

    try:
        if preset == 'front':
            cmds.xform(cam, rotation=[0, 180, 0], worldSpace=True)
        elif preset == 'left_34':
            cmds.xform(cam, rotation=[0, 135, 0], worldSpace=True)
        elif preset == 'right_34':
            cmds.xform(cam, rotation=[0, 225, 0], worldSpace=True)
        elif preset == 'mouth':
            cmds.xform(cam, rotation=[0, 180, 0], worldSpace=True)
            
        cmds.viewFit(cam)
        
        if preset == 'mouth':
            cmds.dolly(cam, relative=True, distance=-20)
            cmds.track(cam, up=5)
            
    except:
        pass
    finally:
        cmds.select(clear=True)
        focus_viewport()

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

    # ==========================================================
    #  STATUS BAR
    # ==========================================================
    global LAST_ANIMATION_END_FRAME, LAST_ANIMATION_PHONEME
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
    #  CAMERA & VIEW CONTROLS
    # ==========================================================
    cmds.text(label='  Camera Controls', align='left', font='boldLabelFont',
              backgroundColor=bg_header, height=22)
    cmds.separator(height=2, style='none')
    cmds.rowLayout(numberOfColumns=5, columnWidth5=(95, 95, 95, 95, 95),
                   columnAttach=[(1, 'both', 2), (2, 'both', 2), (3, 'both', 2), (4, 'both', 2), (5, 'both', 2)])
    cmds.button(label='Front', backgroundColor=accent, command=lambda *args: set_camera_preset("front"))
    cmds.button(label='L 3/4', backgroundColor=accent, command=lambda *args: set_camera_preset("left_34"))
    cmds.button(label='R 3/4', backgroundColor=accent, command=lambda *args: set_camera_preset("right_34"))
    cmds.button(label='Mouth', backgroundColor=accent, command=lambda *args: set_camera_preset("mouth"))
    cmds.button(label='Frame', backgroundColor=btn_reset, command=lambda *args: set_camera_preset("frame"))
    cmds.setParent('..')
    
    cmds.separator(height=2, style='none')
    cmds.rowLayout(numberOfColumns=4, columnWidth4=(118, 118, 118, 118),
                   columnAttach=[(1, 'both', 2), (2, 'both', 2), (3, 'both', 2), (4, 'both', 2)])
    btn_color = [0.25, 0.3, 0.35]
    cmds.button(label='Orbit UP', backgroundColor=btn_color, command=lambda *args: handle_arrow_hotkey("up"))
    cmds.button(label='Orbit DOWN', backgroundColor=btn_color, command=lambda *args: handle_arrow_hotkey("down"))
    cmds.button(label='Orbit LEFT', backgroundColor=btn_color, command=lambda *args: handle_arrow_hotkey("left"))
    cmds.button(label='Orbit RIGHT', backgroundColor=btn_color, command=lambda *args: handle_arrow_hotkey("right"))
    cmds.setParent('..')
    
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

    cmds.checkBox('render_after_typein_checkbox',
                  label='Directly render after deformation (Arnold + Video)',
                  value=RENDER_AFTER_TYPEIN_ENABLED,
                  changeCommand=toggle_render_after_typein)

    cmds.textFieldGrp('render_typein_name_field',
                      label='Render Name:',
                      placeholderText='Leave blank for date/time',
                      columnWidth2=[80, 220])
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
    cmds.text(label='Hacks & Overrides:', align='left', font='smallBoldLabelFont')

    cmds.checkBox('teeth_hack_checkbox',
                  label='Enable teeth/tongue rotation hack (Slows UI)',
                  value=TEETH_HACK_ENABLED,
                  changeCommand=toggle_teeth_hack)

    cmds.checkBox('verbose_mode_checkbox',
                  label='Verbose Output (Development Mode)',
                  value=VERBOSE_MODE,
                  changeCommand=toggle_verbose_mode)

    cmds.button(label='Reset and Run (Dev Mode)', height=28,
                backgroundColor=[0.75, 0.45, 0.95], command=reset_and_run_pipeline)

    cmds.checkBox('finger_clap_checkbox',
                  label='Enable Finger Clap Effect (Collision)',
                  value=FINGER_CLAP_ENABLED,
                  changeCommand=toggle_finger_clap)

    cmds.separator(height=4, style='none')
    cmds.text(label='Playblast & Actions:', align='left', font='smallBoldLabelFont')

    cmds.checkBox('auto_playblast_checkbox',
                  label='Auto-run playblast after deformation',
                  value=False)
                  
    cmds.textFieldGrp('playblast_name_field', 
                      label='Name:', 
                      placeholderText='Leave blank for date/time',
                      columnWidth2=[40, 260])
                  
    cmds.button(label='Run Playblast Now', height=24,
                backgroundColor=[0.45, 0.75, 0.95], command=lambda x: run_playblast_script())
                
    cmds.button(label='Run Finger Clap Test', height=24,
                backgroundColor=[0.95, 0.45, 0.25], command=run_finger_clap_test)

    cmds.separator(height=4, style='none')
    cmds.text(label='Demo Sequences:', align='left', font='smallBoldLabelFont')

    cmds.checkBox('render_after_demo_checkbox',
                  label='Enable Render after demo (Arnold + ffmpeg video)',
                  value=RENDER_AFTER_DEMO_ENABLED,
                  changeCommand=toggle_render_after_demo)

    cmds.textFieldGrp('render_sequence_name_field',
                      label='Render Name:',
                      placeholderText='Leave blank for render_date_time',
                      columnWidth2=[80, 220])

    cmds.button(label='Demo: "Not Even You"', height=28,
                backgroundColor=[0.85, 0.55, 0.85], command=run_not_even_you_demo)

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

    bg_high = [0.8, 0.4, 0.4]
    bg_mid = [0.4, 0.8, 0.4]
    bg_low = [0.4, 0.4, 0.8]

    cmds.rowLayout(numberOfColumns=3, columnWidth3=(150, 150, 150),
                   columnAttach=[(1, 'both', 1), (2, 'both', 1), (3, 'both', 1)])
    cmds.text(label="[f] Frown", align='center', backgroundColor=bg_high)
    cmds.text(label="[b] Brow", align='center', backgroundColor=bg_high)
    cmds.text(label="[q] Squint", align='center', backgroundColor=bg_high)
    cmds.setParent('..')

    cmds.rowLayout(numberOfColumns=8, columnWidth=[(1,56), (2,56), (3,56), (4,56), (5,56), (6,56), (7,56), (8,56)],
                   columnAttach=[(1, 'both', 1), (2, 'both', 1), (3, 'both', 1), (4, 'both', 1), (5, 'both', 1), (6, 'both', 1), (7, 'both', 1), (8, 'both', 1)])
    cmds.text(label="[0] Neu", align='center', backgroundColor=bg_mid)
    cmds.text(label="[s] Smile", align='center', backgroundColor=bg_mid)
    cmds.text(label="[m] Mouth", align='center', backgroundColor=bg_mid)
    cmds.text(label="[c] Cheek", align='center', backgroundColor=bg_mid)
    cmds.text(label="[p] Pucker", align='center', backgroundColor=bg_mid)
    cmds.text(label="[1] Happy", align='center', backgroundColor=bg_mid)
    cmds.text(label="[2] Sad", align='center', backgroundColor=bg_mid)
    cmds.text(label="[3] Surpr", align='center', backgroundColor=bg_mid)
    cmds.setParent('..')

    cmds.rowLayout(numberOfColumns=8, columnWidth=[(1,56), (2,56), (3,56), (4,56), (5,56), (6,56), (7,56), (8,56)],
                   columnAttach=[(1, 'both', 1), (2, 'both', 1), (3, 'both', 1), (4, 'both', 1), (5, 'both', 1), (6, 'both', 1), (7, 'both', 1), (8, 'both', 1)])
    cmds.text(label="[n] N", align='center', backgroundColor=bg_low)
    cmds.text(label="[a] AA", align='center', backgroundColor=bg_low)
    cmds.text(label="[i] IY", align='center', backgroundColor=bg_low)
    cmds.text(label="[t] T", align='center', backgroundColor=bg_low)
    cmds.text(label="[e] EH", align='center', backgroundColor=bg_low)
    cmds.text(label="[u] UW", align='center', backgroundColor=bg_low)
    cmds.text(label="[v] V", align='center', backgroundColor=bg_low)
    cmds.text(label="[y] Y", align='center', backgroundColor=bg_low)
    cmds.setParent('..')

    cmds.button(label='Precompute All Keyboard Transitions (Fix First-Time Lag)', height=28,
                backgroundColor=[0.45, 0.75, 0.55], command=precompute_keyboard_transitions)

    cmds.textField('keyboard_input_field',
                   placeholderText='Type: n, a, i, t, e, u, v, y, s, m, c, p, 1, 2, 3, f, b, q, 0',
                   textChangedCommand=on_keyboard_input_changed,
                   backgroundColor=bg_field, height=26)

    cmds.setParent('..')
    cmds.setParent('..')

    # -- Footer --
    cmds.separator(height=4, style='none')
    cmds.setParent('..')  # end main columnLayout
    cmds.setParent('..')  # end scrollLayout
    cmds.showWindow(window_id)

    # Setup hotkeys and bind cleanup to window close
    setup_camera_hotkeys()
    cmds.scriptJob(uiDeleted=[window_id, cleanup_camera_hotkeys])


# --- Run this line in the Maya Script Editor to show the window ---
create_automation_ui()
