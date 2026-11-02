"""
========================================================================================
DEVELOPMENT SANDBOX: Overlapping Transitions with Feedback Control & Error Dissipation
========================================================================================

File: h99-UI-mayascipt-dev.py
Location: c:/keyLabLocal_D/a3-stochastic-git-V1/a3-stochastic-V1/b99-frame1to4-get-b1to3/d4 interactive interface of animation/

UPDATES IN THIS VERSION:
------------------------
1. DYNAMIC FRAME SIZING (No 50-frame squeeze):
   Uses `get_transition_frames(p_from, p_to)` directly from the 'if' branch:
   - Consonant -> Vowel (e.g. V -> AA): ~45 frames
   - Neutral -> Consonant (e.g. Neutral -> V): ~21 frames
   - Vowel -> Consonant (e.g. AA -> V): ~21 frames
   - Consonant -> Neutral (e.g. V -> Neutral): ~28 frames
   Checkpoints are allocated proportionally per phase, giving natural, multi-dozen frame durations.

2. VERTICAL JAW ROTATION ONLY (Pitch / X-axis locked):
   - In training data and Maya, vertical jaw opening/closing is pitch (X-axis).
   - Yaw (Y-axis) and Roll (Z-axis) are strictly zeroed out (`rel_rot[1] = 0.0`, `rel_rot[2] = 0.0`).
   - Prevents all sideways twisting / yawing of the jaw.
   - Restores the lower teeth & tongue rotation hack around `jaw01` pivot (`teeth_correction_weight = 0.25`).

3. TEST SEQUENCE:
   - Defaults to "0 to V to AA to V to 0" as requested for rapid testing.
========================================================================================
"""

import os
import sys
import re
import math
import json
import pickle
import datetime
from pathlib import Path
import numpy as np

# Safe fallback for Maya environment
try:
    import maya.cmds as cmds
    MAYA_AVAILABLE = True
except ImportError:
    MAYA_AVAILABLE = False
    print("[WARN] maya.cmds not available. Running in headless / dry-run mode.")

# ==============================================================================
# Dynamic Repository Root Detection
# ==============================================================================
REPO_NAME = "a3-stochastic-git-V1"
REPO_ROOT = None

if "paths" in sys.modules and hasattr(sys.modules["paths"], "__file__"):
    cached_path = Path(sys.modules["paths"].__file__).resolve().parent
    if (cached_path / "paths.py").is_file():
        REPO_ROOT = cached_path

if not REPO_ROOT and "__file__" in globals() and __file__:
    script_path = Path(__file__).resolve()
    for p in [script_path] + list(script_path.parents):
        if (p / "paths.py").is_file():
            REPO_ROOT = p
            break

if not REPO_ROOT:
    fallbacks = [
        Path(r"C:\keyLabLocal_D") / REPO_NAME,
        Path(r"D:\keyLabLocal_D") / REPO_NAME,
        Path.home() / "git" / REPO_NAME,
        Path.home() / "Documents" / "GitHub" / REPO_NAME,
    ]
    for fb in fallbacks:
        if (fb / "paths.py").is_file():
            REPO_ROOT = fb
            break

if not REPO_ROOT:
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        if (p / "paths.py").is_file():
            REPO_ROOT = p
            break

if not REPO_ROOT:
    raise RuntimeError(f"Could not locate '{REPO_NAME}' containing 'paths.py'.")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ==============================================================================
# Import Paths and Sub-modules
# ==============================================================================
from paths import (
    a3_stochastic_path,
    b98_sampling_path,
    a1_blendshape_path,
    backup_path,
    results_path
)

sys.path.insert(0, str(a3_stochastic_path / "a3-stochastic-V1-algo" / "example_train"))
sys.path.insert(0, str(a1_blendshape_path / "database"))
sys.path.insert(0, str(b98_sampling_path))
sys.path.insert(0, str(b98_sampling_path / "300joints_suite_good_old"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions" / "joint_position_ooc_l"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions" / "joint_position_ooc_r"))
sys.path.insert(0, str(b98_sampling_path / "joint_positions" / "joint_position_frontalis"))
sys.path.insert(0, str(backup_path / "b31 dataV22 backup hub"))

# Import joint position modules
try:
    import joint_position
    import joint_position_first_wave
    import joint_position_third_wave

    oor_joint_modules = [
        joint_position_first_wave,
        joint_position,
        joint_position_third_wave
    ]

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
    JOINTS_LOADED = True
except ImportError as e:
    JOINTS_LOADED = False
    print(f"[WARN] Joint modules not loaded ({e}). Headless test mode.")

# ==============================================================================
# Global Configuration & State
# ==============================================================================
ENABLE_FEEDBACK_CONTROL = True      # Enable Closed-Loop Feedback Control
ERROR_DISSIPATION_RATE = 0.5        # alpha: fraction of error corrected (0.0 to 1.0)
OVERLAP_RATIO = 0.4                 # 40% coarticulation overlap between phases
BASE_TRANSITION_FRAMES = 35         # Baseline frame duration per transition
DEBUG_ENABLED = True
TEETH_HACK_ENABLED = True
teeth_correction_weight = 0.25      # 25% jaw rotation on lower teeth/tongue

# Feedback control runtime state
ACCUMULATED_ERROR = None            # Vector dict of accumulated drift errors
DEBUG_DATA = []                     # Per-transition metrics
PER_FRAME_DEBUG_HISTORY = []        # Per-frame live tracking
LIVE_DEBUG_FRAMES = []
LIVE_DEBUG_POSITIONS = []

DEBUG_OUTPUT_DIR = str(a3_stochastic_path / "b99-frame1to4-get-b1to3" / "debug_results")
CURRENT_JAW_LOG_FILE = os.path.join(DEBUG_OUTPUT_DIR, "jaw_rotation_dissipation_log.txt")
TEETH_TONGUE_MESH_PATTERNS = ['*loTeeth01_geo*', '*tongue01_geo*']


# ==============================================================================
# Helper Functions: Transition Frames, Teeth & Tongue Hack
# ==============================================================================
def get_transition_frames(p_from, p_to, base_frames=35):
    """
    Calculate dynamic transition length based on phoneme categories.
    Directly references the 'if' branch logic from h99-UI-mayascipt-V1.py:
    - Neutral to Consonant: quick (~21 frames)
    - Consonant to Vowel: long (~45 frames)
    - Vowel to Consonant: quick (~21 frames)
    - Vowel to Vowel: long (~42 frames)
    - Consonant to Consonant: medium (~24 frames)
    """
    f = p_from.replace('dataV22_phoneme_', '').replace('dataV22_AU0_neutral', 'Neutral').replace('AU0_neutral', 'Neutral').replace('0', 'Neutral')
    t = p_to.replace('dataV22_phoneme_', '').replace('dataV22_AU0_neutral', 'Neutral').replace('AU0_neutral', 'Neutral').replace('0', 'Neutral')

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
    elif f in consonants and t == 'Neutral':
        return int(base_frames * 0.8)  # ~28 frames
    elif f in vowels and t == 'Neutral':
        return int(base_frames * 0.8)  # ~28 frames
    else:
        return base_frames


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


def get_teeth_tongue_meshes():
    """Returns unique base transform names for teeth and tongue meshes, excluding any _RENDER proxies."""
    if not MAYA_AVAILABLE:
        return []
    meshes = []
    for pattern in TEETH_TONGUE_MESH_PATTERNS:
        found = cmds.ls(pattern, type='transform') or cmds.ls(pattern) or []
        for m in found:
            if not m.endswith("_RENDER") and not m.endswith("_sagged_snapshot") and not m.endswith("_equil_snapshot"):
                if m not in meshes:
                    meshes.append(m)
    return meshes


def ensure_proxy_render_meshes_at_frame_zero():
    """
    Ensures proxy render meshes for both soft tissues and teeth/tongue exist
    by jumping to Frame 0 (pure native Neutral pose) before creating them.
    Directly learned from lines 3756-3793 in h99-UI-mayascipt-V1.py.
    """
    if not MAYA_AVAILABLE:
        return
    valid_soft_tissues = [m for m in SOFT_TISSUE_MESHES if cmds.objExists(m)]
    valid_teeth_tongue = [m for m in get_teeth_tongue_meshes() if cmds.objExists(m)]
    all_proxy_targets = valid_soft_tissues + valid_teeth_tongue

    needs_proxy = any(not cmds.objExists(f"{m.split('|')[-1].split(':')[-1]}_RENDER") for m in all_proxy_targets)
    if needs_proxy:
        print("[INITIALIZATION] Jumping to Frame 0 to safely capture perfect Neutral proxy meshes...")
        curr_time = cmds.currentTime(query=True)
        cmds.currentTime(0, edit=True)

        for target_mesh in all_proxy_targets:
            clean_name = target_mesh.split('|')[-1].split(':')[-1]
            render_mesh = f"{clean_name}_RENDER"
            bs_name = f"master_bs_{clean_name}"
            if not cmds.objExists(render_mesh):
                render_mesh_node = cmds.duplicate(target_mesh, name=render_mesh)[0]
                if not cmds.objExists(bs_name):
                    cmds.blendShape([target_mesh], render_mesh_node, name=bs_name, origin='local')
                    cmds.setAttr(f"{bs_name}.{clean_name}", 1.0)
                print(f"[RENDER PROXY] Created {render_mesh_node} for {target_mesh}")

            if cmds.attributeQuery('visibility', node=target_mesh, exists=True):
                cmds.setAttr(f"{target_mesh}.visibility", 0)
            if cmds.attributeQuery('visibility', node=render_mesh, exists=True):
                cmds.setAttr(f"{render_mesh}.visibility", 1)
        cmds.currentTime(curr_time, edit=True)
    else:
        for target_mesh in all_proxy_targets:
            clean_name = target_mesh.split('|')[-1].split(':')[-1]
            render_mesh = f"{clean_name}_RENDER"
            if cmds.objExists(render_mesh):
                if cmds.attributeQuery('visibility', node=target_mesh, exists=True):
                    cmds.setAttr(f"{target_mesh}.visibility", 0)
                if cmds.attributeQuery('visibility', node=render_mesh, exists=True):
                    cmds.setAttr(f"{render_mesh}.visibility", 1)


def reset_animation_for_dev_run(start_frame=14):
    """
    Cleans up old animation keyframes above start_frame on jaw01, teeth, and landmark joints,
    learned directly from reset_animation_chain (lines 4107-4180 in V1.py).
    """
    if not MAYA_AVAILABLE:
        return
    print(f"[RESET] Cleaning animation keyframes above frame {start_frame}...")
    if cmds.objExists('jaw01'):
        cmds.cutKey('jaw01', time=(start_frame + 0.01, 1000000), clear=True)
        cmds.currentTime(start_frame, edit=True)
        cmds.setKeyframe('jaw01', time=start_frame)

    for m in get_teeth_tongue_meshes():
        clean_name = m.split('|')[-1].split(':')[-1]
        rm = f"{clean_name}_RENDER"
        target = rm if cmds.objExists(rm) else m
        if cmds.objExists(target):
            cmds.cutKey(target, time=(start_frame + 0.01, 1000000), clear=True)
            cmds.currentTime(start_frame, edit=True)
            cmds.setKeyframe(target, time=start_frame)


# ==============================================================================
# Database & Normalization Helpers
# ==============================================================================
def normalize_task_key(task_string):
    """Converts a task string into a standardized key format for lookups."""
    key = str(task_string).lower()
    key = key.replace('_to_', ' to ')
    key = re.sub(r'\s*to\s*', '_to_', key)
    key = re.sub(r'\s+', '_', key)
    return key


def load_database():
    """Load the deformation path database and state map from JSON file."""
    base_dir = str(a3_stochastic_path)
    db_path = os.path.join(base_dir, "automation", "deformation_path_db.json")

    try:
        with open(db_path, 'r') as f:
            data = json.load(f)
            return data.get('deformation_paths', {}), data.get('state_map', {})
    except Exception as e:
        print(f"[ERROR] Failed to load database from {db_path}: {e}")
        return {}, {}


DEFORMATION_PATH_DB, STATE_MAP = load_database()


def parse_sequence_string(input_text):
    """
    Translates user text like "0 to V to AA to V to 0"
    into a list of standardized phoneme identifiers and their source paths.
    """
    cleaned_text = input_text.lower().replace(",", " ").replace("to", " ")
    parts = [p for p in cleaned_text.split() if p]

    phonemes = []
    for part in parts:
        if part in STATE_MAP:
            phonemes.append(STATE_MAP[part])
        elif f"datav22_{part}" in STATE_MAP:
            phonemes.append(STATE_MAP[f"datav22_{part}"])
        else:
            if part in ["neutral", "au0", "0"]:
                phonemes.append("dataV22_AU0_neutral")
            elif not part.startswith("dataV22_"):
                phonemes.append(f"dataV22_{part}")
            else:
                phonemes.append(part)

    if len(phonemes) < 2:
        raise ValueError(f"Need at least 2 phonemes. Got: {phonemes}")

    db_key_map = {normalize_task_key(k): k for k in DEFORMATION_PATH_DB.keys()}
    log_paths = []

    for i in range(len(phonemes) - 1):
        sa = phonemes[i]
        sb = phonemes[i + 1]

        cand_keys = [
            f"{sa}_to_{sb}",
            f"{sa} to {sb}",
            f"{sa.replace('dataV22_', '')}_to_{sb.replace('dataV22_', '')}",
            f"{sa.replace('dataV22_', '')} to {sb.replace('dataV22_', '')}",
            f"Neutral to {sb.replace('dataV22_', '')}" if "neutral" in sa.lower() else ""
        ]

        found_path = None
        for cand in cand_keys:
            if not cand:
                continue
            if cand in DEFORMATION_PATH_DB:
                found_path = DEFORMATION_PATH_DB[cand]
                break
            norm = normalize_task_key(cand)
            if norm in db_key_map:
                found_path = DEFORMATION_PATH_DB[db_key_map[norm]]
                break

        if not found_path:
            raise ValueError(f"Could not find path for transition: '{sa}' -> '{sb}' in DEFORMATION_PATH_DB.")

        log_paths.append(found_path)

    return phonemes, log_paths


def generate_short_sequence_name(user_text, phonemes):
    """Generates an informative folder name for saving deformation outputs."""
    short_names = [p.replace("dataV22_", "").replace("AU0_neutral", "Neutral") for p in phonemes]
    return "_to_".join(short_names)


# ==============================================================================
# Maya Landmark & Kinematic Helpers
# ==============================================================================
def step_timeline_forward(target_frame):
    """Steps the timeline forward sequentially to preserve solver physics caches."""
    if not MAYA_AVAILABLE:
        return
    current_time = int(cmds.currentTime(query=True))
    target = int(target_frame)
    if current_time < target:
        for f in range(current_time + 1, target + 1):
            cmds.currentTime(f, edit=True)


def retrieve_landmark_positions(joint_groups, target_frame=14, is_first_transition=True):
    """Queries world-space translations for landmark joint groups at target_frame."""
    if not MAYA_AVAILABLE:
        mock_data = {}
        for group, count in joint_groups.items():
            mock_data[group] = [[0.0, 0.0, 0.0] for _ in range(count)]
        return mock_data

    step_timeline_forward(target_frame)
    retrieved = {}
    for group_name, num_joints in joint_groups.items():
        coords = []
        for i in range(num_joints):
            joint_name = f"{group_name}_{i + 1:02d}"
            if cmds.objExists(joint_name):
                coords.append(cmds.xform(joint_name, query=True, translation=True, worldSpace=True))
            else:
                coords.append([0.0, 0.0, 0.0])
        retrieved[group_name] = coords
    return retrieved


def load_ideal_target_positions(target_phoneme):
    """Loads ground-truth neutral-to-target positions from deformation database."""
    if not JOINTS_LOADED:
        return None

    # Compute true neutral reference coordinates
    x_joint_combined = []
    for mod in oor_joint_modules:
        x_joint_combined.extend(getattr(mod, "x_joint_position_1"))
    x_neutral = [item for sublist in x_joint_combined for item in sublist]

    x_neutral_ooc_l = [item for sublist in (
        getattr(joint_position_ooc_l_base_0, "x_joint_position_1") +
        getattr(joint_position_ooc_l_base_pose11_0, "x_joint_position_1") +
        getattr(joint_position_ooc_l_base_pose11_AU1_0, "x_joint_position_1") +
        getattr(joint_position_ooc_l_base_pose11_AU1_AU2_0, "x_joint_position_1")
    ) for item in sublist]

    x_neutral_ooc_r = [item for sublist in (
        getattr(joint_position_ooc_r_base_0, "x_joint_position_1") +
        getattr(joint_position_ooc_r_base_pose11_0, "x_joint_position_1") +
        getattr(joint_position_ooc_r_base_pose11_AU1_0, "x_joint_position_1") +
        getattr(joint_position_ooc_r_base_pose11_AU1_AU2_0, "x_joint_position_1")
    ) for item in sublist]

    x_neutral_frontalis = [item for sublist in (
        getattr(joint_position_frontalis_base_0, "x_joint_position_1") +
        getattr(joint_position_frontalis_base_AU1_0, "x_joint_position_1") +
        getattr(joint_position_frontalis_base_AU1_AU2_0, "x_joint_position_1")
    ) for item in sublist]

    if target_phoneme in ["dataV22_AU0_neutral", "neutral", "AU0"]:
        return {
            'x_joint_conca': x_neutral,
            'x_joint_rotation_conca': [0.0, 0.0, 0.0],
            'x_ooc_l_joint_conca': x_neutral_ooc_l,
            'x_ooc_r_joint_conca': x_neutral_ooc_r,
            'x_frontalis_joint_conca': x_neutral_frontalis
        }

    key = normalize_task_key(f"dataV22_AU0_neutral_to_{target_phoneme}")
    db_key_map = {normalize_task_key(k): k for k in DEFORMATION_PATH_DB}
    orig_key = db_key_map.get(key)
    if not orig_key:
        print(f"[FEEDBACK CONTROL WARNING] Neutral target key '{key}' not found in DB.")
        return None

    path = DEFORMATION_PATH_DB[orig_key]
    checkpoints_file = os.path.join(path, "combined_checkpoints.pkl")
    if not os.path.exists(checkpoints_file):
        return None

    with open(checkpoints_file, 'rb') as f:
        checkpoints = pickle.load(f)
    final_checkpoint = np.array(checkpoints[-1])

    n = len(x_neutral)
    joint_rotate_num = 1
    joint_lm_num = 300 * 3

    x_ideal = np.array(x_neutral) + final_checkpoint[0:n]
    # Enforce vertical dimension only for jaw rotation (X-axis pitch, zero Y and Z)
    x_ideal_rot = np.array([final_checkpoint[n], 0.0, 0.0])

    s1 = n + 3 * joint_rotate_num
    s2 = s1 + joint_lm_num
    x_ideal_ooc_l = np.array(x_neutral_ooc_l) + final_checkpoint[s1:s2]

    s3 = s2 + joint_lm_num
    x_ideal_ooc_r = np.array(x_neutral_ooc_r) + final_checkpoint[s2:s3]

    s4 = s3 + joint_lm_num
    x_ideal_frontalis = np.array(x_neutral_frontalis) + final_checkpoint[s3:s4]

    return {
        'x_joint_conca': x_ideal.tolist(),
        'x_joint_rotation_conca': x_ideal_rot.tolist(),
        'x_ooc_l_joint_conca': x_ideal_ooc_l.tolist(),
        'x_ooc_r_joint_conca': x_ideal_ooc_r.tolist(),
        'x_frontalis_joint_conca': x_ideal_frontalis.tolist()
    }


def compute_position_error(actual_positions, ideal_positions):
    """Computes error vector: error = ideal - actual for all landmark groups."""
    if ideal_positions is None or actual_positions is None:
        return None

    error = {}
    for key in ['x_joint_conca', 'x_joint_rotation_conca', 'x_ooc_l_joint_conca',
                'x_ooc_r_joint_conca', 'x_frontalis_joint_conca']:
        act = np.array(actual_positions.get(key, []))
        idl = np.array(ideal_positions.get(key, []))
        if len(act) > 0 and len(idl) > 0 and len(act) == len(idl):
            err = idl - act
            # Jaw rotation error is strictly pitch (vertical dimension)
            if key == 'x_joint_rotation_conca':
                err[1] = 0.0
                err[2] = 0.0
            error[key] = err
        else:
            error[key] = np.zeros_like(act) if len(act) > 0 else np.array([])
    return error


def beta_function(x_joint_conca, x_joint_rotation_conca,
                  au22_x_joint_conca3, au22_x_joint_rotation_conca3,
                  au22_x_ooc_l_joint_conca3, au22_x_ooc_r_joint_conca3,
                  au22_x_frontalis_joint_conca3, start_frame, keyframe_max):
    """
    Applies rotation and joint translation keyframes in Maya.
    Enforces jaw rotation STRICTLY in the vertical dimension (pitch: X-axis only)
    to match the 'if' branch reference, preventing lateral twisting.
    """
    if not MAYA_AVAILABLE:
        return

    time_temp = float(keyframe_max)
    cmds.currentTime(time_temp, edit=True)

    # Relative jaw rotation between previous and current step
    relative_rotation_value = np.array(au22_x_joint_rotation_conca3) - np.array(x_joint_rotation_conca)

    # CRITICAL: Lock jaw rotation strictly to the vertical dimension (pitch = index 0)
    # Zero out yaw (index 1) and roll (index 2) to prevent lateral jaw twisting
    relative_rotation_value[1] = 0.0
    relative_rotation_value[2] = 0.0

    if np.any(relative_rotation_value != 0) and cmds.objExists('jaw01'):
        cmds.rotate(
            relative_rotation_value[0],
            0.0,
            0.0,
            'jaw01',
            relative=True,
            ws=True
        )

    if cmds.objExists('jaw01'):
        cmds.setKeyframe('jaw01', time=time_temp)

    # Apply lower teeth & tongue rotation around jaw pivot (learned from 'if' branch)
    if TEETH_HACK_ENABLED and cmds.objExists('jaw01'):
        jaw_pivot = cmds.xform('jaw01', query=True, rotatePivot=True, ws=True)
        for mesh_name in get_teeth_tongue_meshes():
            clean_name = mesh_name.split('|')[-1].split(':')[-1]
            render_mesh = f"{clean_name}_RENDER"
            target_to_rotate = render_mesh if cmds.objExists(render_mesh) else mesh_name
            try:
                cmds.scriptEditorInfo(suppressWarnings=True)
                cmds.rotate(relative_rotation_value[0] * teeth_correction_weight,
                            0.0,
                            0.0,
                            target_to_rotate, relative=True, ws=True, pivot=jaw_pivot)
                cmds.setKeyframe(target_to_rotate, time=time_temp)
                if cmds.objExists(render_mesh):
                    if cmds.attributeQuery('visibility', node=mesh_name, exists=True):
                        cmds.setAttr(f"{mesh_name}.visibility", 0)
                    if cmds.attributeQuery('visibility', node=render_mesh, exists=True):
                        cmds.setAttr(f"{render_mesh}.visibility", 1)
                cmds.scriptEditorInfo(suppressWarnings=False)
            except Exception:
                cmds.scriptEditorInfo(suppressWarnings=False)

    # Keyframe OOR joints
    joint_num = len(au22_x_joint_conca3) // 3
    for i in range(joint_num):
        jname = f'oor_{i + 1:02d}'
        if cmds.objExists(jname):
            cmds.move(au22_x_joint_conca3[3 * i], au22_x_joint_conca3[3 * i + 1], au22_x_joint_conca3[3 * i + 2], jname, absolute=True)
            cmds.setKeyframe(jname, time=time_temp)

    # Keyframe OOC_L, OOC_R, Frontalis joints (300 each)
    for gname, coords in [('ooc_l', au22_x_ooc_l_joint_conca3),
                          ('ooc_r', au22_x_ooc_r_joint_conca3),
                          ('frontalis', au22_x_frontalis_joint_conca3)]:
        g_num = len(coords) // 3
        for i in range(g_num):
            jname = f'{gname}_{i + 1:02d}'
            if cmds.objExists(jname):
                cmds.move(coords[3 * i], coords[3 * i + 1], coords[3 * i + 2], jname, absolute=True)
                cmds.setKeyframe(jname, time=time_temp)


def initial_joint_position(target_frame=14, is_first_transition=True):
    """Retrieves current joint positions at the start frame offset."""
    if not JOINTS_LOADED or not MAYA_AVAILABLE:
        oor_num = 300
        n = oor_num * 3
        return [0.0]*n, [0.0]*3, [0.0]*900, [0.0]*900, [0.0]*900

    oor_num = sum(len(mod.x_joint_position_1) for mod in oor_joint_modules)
    joint_groups = {'oor': oor_num, 'ooc_l': 300, 'ooc_r': 300, 'frontalis': 300}
    retrieved = retrieve_landmark_positions(joint_groups, target_frame, is_first_transition)

    x_joint = [item for sublist in retrieved.get('oor', []) for item in sublist]
    x_ooc_l = [item for sublist in retrieved.get('ooc_l', []) for item in sublist]
    x_ooc_r = [item for sublist in retrieved.get('ooc_r', []) for item in sublist]
    x_frontalis = [item for sublist in retrieved.get('frontalis', []) for item in sublist]

    # Neutral jaw rotation is relative [0, 0, 0] (like line 1179 in h99-UI-mayascipt-V1.py)
    x_rot = [0.0, 0.0, 0.0]

    return x_joint, x_rot, x_ooc_l, x_ooc_r, x_frontalis


# ==============================================================================
# CORE MODULE: Overlap Blending with Dynamic Frame Sizing & Dissipation Metadata
# ==============================================================================
def run_deformation_processing_with_dissipation(phonemes_list, log_paths_list,
                                                sequence_folder_name,
                                                overlap_ratio=0.4):
    """
    Blends multiple phoneme transitions with action-space overlap and precomputes
    the precise timing metadata for the closed-loop error dissipation schedule.

    FRAME COUNTING:
    - Calculates transition frame durations dynamically via get_transition_frames()
      (vowels ~45 frames, consonants ~21-28 frames) so the animation is NEVER squeezed.
    """
    print("\n" + "=" * 70)
    print("--- [DEFORMATION PROCESSING] Dynamic Overlap Blending & Dissipation ---")
    print(f"Phonemes: {phonemes_list}")
    print(f"Overlap Ratio: {overlap_ratio * 100:.1f}%")
    print("=" * 70)

    # Prioritize local repository results_path for reliability across all environments (Maya & headless)
    local_hub = os.path.join(str(results_path), "sequence_combined_deformation_parameters_hub")
    network_hub = r"\\bbysvm-nfs1.its.sfu.ca\GrUViLab\Data\richardPeng\SFU_PhD_cloud\2-Projects\1-physically based facial animation\3-Experiments\a3-stochastic-V1-backup-and-results\results\sequence_combined_deformation_parameters_hub"

    base_path = local_hub
    try:
        os.makedirs(local_hub, exist_ok=True)
        base_path = local_hub
    except Exception as e_loc:
        print(f"[INFO] Could not create local hub ({e_loc}). Trying network...")
        try:
            os.makedirs(network_hub, exist_ok=True)
            base_path = network_hub
        except Exception:
            fallback = os.path.join(str(a3_stochastic_path), "combined_deformation_hub")
            os.makedirs(fallback, exist_ok=True)
            base_path = fallback

    output_dir = os.path.join(base_path, sequence_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output Directory: {output_dir}")

    # 1. Load Action Lists from separate log directories
    action_lists = []
    for i, log_path in enumerate(log_paths_list):
        file_path = os.path.join(log_path, "final_epoch_action_list.pkl")
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    act = data.get('final_epoch_action_list') or data.get('action_list', [])
                else:
                    act = data
                action_lists.append(act)
                print(f"Loaded Transition {i+1} ({phonemes_list[i]} -> {phonemes_list[i+1]}): {len(act)} actions")
        except Exception as e:
            print(f"[ERROR] Failed to load actions from {file_path}: {e}")
            action_lists.append([])

    num_phases = len(action_lists)
    if num_phases == 0 or all(len(a) == 0 for a in action_lists):
        raise RuntimeError("No action lists available to process.")

    # 2. Dynamic Frame Durations based on Consonants / Vowels
    phase_target_frames = []
    for i in range(len(phonemes_list) - 1):
        dur = get_transition_frames(phonemes_list[i], phonemes_list[i + 1], base_frames=BASE_TRANSITION_FRAMES)
        phase_target_frames.append(dur)
        print(f"  Transition {i+1} ({phonemes_list[i]} -> {phonemes_list[i+1]}): Duration = {dur} frames")

    # 3. Compute Original Phase Boundaries in Action Space
    phase_lengths = [len(a) for a in action_lists]
    phase_boundaries = np.cumsum([0] + phase_lengths).tolist()
    print(f"Original Boundaries (Pre-Overlap Action Steps): {phase_boundaries}")

    # 4. Apply Overlap Blending between consecutive phases
    def modify_phase_boundaries(boundaries, front_idx, back_idx, ratio):
        p_len = boundaries[back_idx] - boundaries[front_idx]
        w_steps = int(p_len * ratio)
        for idx in range(back_idx, len(boundaries)):
            boundaries[idx] -= w_steps
        return boundaries, w_steps

    def modify_action_lists(action_lists_in, boundaries, front_idx, back_idx, w_steps):
        flat = []
        for p in action_lists_in:
            flat.extend(p)

        front_b = boundaries[front_idx]
        l1 = flat[:front_b - w_steps]
        l2 = flat[front_b - w_steps:front_b]
        l3 = flat[front_b:front_b + w_steps]
        l4 = flat[front_b + w_steps:]

        min_w = min(len(l2), len(l3))
        l2_prime = []
        for a, b in zip(l2[:min_w], l3[:min_w]):
            l2_prime.append([x + y for x, y in zip(a, b)])

        modified_flat = l1 + l2_prime + l4

        action_lists_out = []
        cur_idx = 0
        for idx in range(len(boundaries) - 1):
            if idx == front_idx:
                length = (boundaries[idx + 1] - boundaries[idx]) - w_steps
            else:
                length = (boundaries[idx + 1] - boundaries[idx])
            phase_actions = modified_flat[cur_idx:cur_idx + length]
            action_lists_out.append(phase_actions)
            cur_idx += len(phase_actions)

        return action_lists_out

    current_actions = [lst.copy() for lst in action_lists]
    current_boundaries = phase_boundaries.copy()
    overlap_windows_actions = []

    for front_phase in range(len(phase_boundaries) - 2):
        back_phase = front_phase + 1
        current_boundaries, w = modify_phase_boundaries(
            current_boundaries.copy(), front_phase, back_phase, overlap_ratio
        )
        current_actions = modify_action_lists(
            current_actions, phase_boundaries, front_phase, back_phase, w
        )
        overlap_windows_actions.append(w)
        print(f"Overlap {front_phase+1} & {back_phase+1}: {w} actions blended. New Boundary: {current_boundaries[back_phase]}")

    modified_boundaries = current_boundaries
    total_actions_len = sum(len(a) for a in current_actions)
    print(f"Modified Boundaries (Overlap Points D_k): {modified_boundaries}")
    print(f"Total Blended Actions Length: {total_actions_len}")

    # 5. Dynamic Checkpoint Allocation (Proportional to Phase Frame Durations)
    #    Each phase gets its allocated number of frames so speech does not squeeze!
    #    Overlap in frame space: W_frames[k] = int(target_frames[k] * overlap_ratio)
    phase_overlap_frames = [int(f * overlap_ratio) for f in phase_target_frames[:-1]] + [0]
    phase_net_frames = []
    for k in range(len(phase_target_frames)):
        if k < len(phase_target_frames) - 1:
            phase_net_frames.append(max(1, phase_target_frames[k] - phase_overlap_frames[k]))
        else:
            phase_net_frames.append(phase_target_frames[k])

    total_sequence_frames = sum(phase_net_frames)
    print(f"Total Animation Frames: {total_sequence_frames} frames (Dynamic multi-dozen frames per phoneme)")

    # Sample checkpoints within each phase segment proportionally
    checkpoint_indices = []
    for k in range(len(modified_boundaries) - 1):
        b_start = modified_boundaries[k]
        b_end = modified_boundaries[k + 1]
        n_pts = phase_net_frames[k]
        # Include b_start and b_end exactly
        if n_pts > 1:
            seg_pts = np.linspace(b_start, b_end, num=n_pts, endpoint=False, dtype=int).tolist()
        else:
            seg_pts = [b_start]
        checkpoint_indices.extend(seg_pts)

    # Ensure final action is included
    if total_actions_len - 1 not in checkpoint_indices:
        checkpoint_indices.append(total_actions_len - 1)
    checkpoint_indices = sorted(list(set(checkpoint_indices)))
    print(f"Total Checkpoints Sampled: {len(checkpoint_indices)} (One per frame)")

    # 6. Accumulate Blended Actions into Checkpoints
    correct_dim = len(current_actions[0][0]) if (current_actions and current_actions[0]) else 3603
    delta_accum = np.zeros(correct_dim)
    combined_checkpoints = []

    flat_blended = []
    for p in current_actions:
        flat_blended.extend(p)

    cp_set = set(checkpoint_indices)
    for step_idx, action in enumerate(flat_blended):
        delta_accum += np.array(action)
        if step_idx in cp_set:
            combined_checkpoints.append((step_idx, delta_accum.tolist()))

    if checkpoint_indices[-1] not in [cp[0] for cp in combined_checkpoints]:
        combined_checkpoints.append((total_actions_len - 1, delta_accum.tolist()))

    # 7. Build the Dissipation Schedule Metadata (Matching the Diagram)
    cp_step_list = [cp[0] for cp in combined_checkpoints]
    def find_nearest_cp_idx(step):
        diffs = [abs(s - step) for s in cp_step_list]
        return int(np.argmin(diffs))

    transitions_metadata = []
    num_transitions = len(phonemes_list) - 1
    for k in range(num_transitions):
        from_ph = phonemes_list[k]
        to_ph = phonemes_list[k + 1]

        dissipation_start_step = modified_boundaries[k + 1] if (k + 1 < len(modified_boundaries)) else total_actions_len - 1
        d_cp_idx = find_nearest_cp_idx(dissipation_start_step)

        if k == 0:
            window_start_cp = 0
            window_end_cp = d_cp_idx
            is_baseline = True
        else:
            prev_d_step = modified_boundaries[k]
            window_start_cp = find_nearest_cp_idx(prev_d_step)
            window_end_cp = d_cp_idx
            is_baseline = False

        meta_entry = {
            'transition_index': k,
            'phoneme_from': from_ph,
            'phoneme_to': to_ph,
            'is_first_transition': (k == 0),
            'is_baseline': is_baseline,
            'target_frames': phase_target_frames[k],
            'dissipation_start_step': dissipation_start_step,
            'dissipation_start_cp_idx': d_cp_idx,
            'window_start_cp_idx': window_start_cp,
            'window_end_cp_idx': window_end_cp,
            'target_phoneme': to_ph
        }
        transitions_metadata.append(meta_entry)
        print(f"[METADATA] Transition {k+1} ({from_ph}->{to_ph}): {phase_target_frames[k]} frames, Dissipation Window CP [{window_start_cp} -> {window_end_cp}]")

    # 8. Save Artifacts
    output_checkpoint_file = os.path.join(output_dir, "combined_checkpoints.pkl")
    with open(output_checkpoint_file, 'wb') as f:
        pickle.dump([cp[1] for cp in combined_checkpoints], f)
    print(f"Saved combined_checkpoints.pkl -> {output_checkpoint_file}")

    metadata_payload = {
        'phonemes': phonemes_list,
        'transitions_metadata': transitions_metadata,
        'modified_boundaries': modified_boundaries,
        'checkpoint_indices': checkpoint_indices,
        'phase_target_frames': phase_target_frames,
        'overlap_ratio': overlap_ratio
    }
    metadata_file = os.path.join(output_dir, "dissipation_metadata.pkl")
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata_payload, f)
    print(f"Saved dissipation_metadata.pkl -> {metadata_file}")

    return output_checkpoint_file, metadata_file, metadata_payload


# ==============================================================================
# CORE MODULE: Feedback-Controlled Deformation with Integrated Dissipation
# ==============================================================================
def perform_the_deformation_with_dissipation(source_path,
                                             start_frame_offset=14,
                                             continue_from_last=False):
    """
    Applies the overlapped checkpoints to the Maya facial rig with dynamic,
    closed-loop error dissipation across the exact transition intervals
    illustrated in the user design diagram.
    """
    global ACCUMULATED_ERROR, DEBUG_DATA, LIVE_DEBUG_FRAMES, LIVE_DEBUG_POSITIONS

    print("\n" + "=" * 70)
    print("--- [RIG DEFORMATION] Starting Closed-Loop Dissipating Execution ---")
    print(f"Source Folder: {source_path}")
    print(f"Start Frame Offset: {start_frame_offset}")
    print(f"Feedback Control: {'ENABLED' if ENABLE_FEEDBACK_CONTROL else 'DISABLED'}")
    print(f"Dissipation Rate (alpha): {ERROR_DISSIPATION_RATE}")
    print("=" * 70)

    checkpoints_path = os.path.join(source_path, "combined_checkpoints.pkl")
    metadata_path = os.path.join(source_path, "dissipation_metadata.pkl")

    if not os.path.exists(checkpoints_path):
        raise RuntimeError(f"Checkpoints file not found: {checkpoints_path}")

    with open(checkpoints_path, "rb") as f:
        combined_checkpoints = pickle.load(f)

    transitions_metadata = []
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            meta = pickle.load(f)
            transitions_metadata = meta.get('transitions_metadata', [])

    oor_num = 300
    n = oor_num * 3
    joint_rotate_num = 1
    joint_lm_num = 300 * 3
    expected_dim = n + 3 * joint_rotate_num + 3 * joint_lm_num

    # 1. Ensure proxy render meshes exist captured cleanly at Frame 0
    ensure_proxy_render_meshes_at_frame_zero()

    # 2. Reset timeline keys above start_frame_offset on jaw01 and teeth/tongue if starting fresh
    if not continue_from_last:
        reset_animation_for_dev_run(start_frame_offset)

    # Initial joint state
    x_joint, x_rot, x_ooc_l, x_ooc_r, x_frontalis = initial_joint_position(
        start_frame_offset, is_first_transition=not continue_from_last
    )

    x_joint_init = np.array(x_joint)
    # Jaw rotation relative to neutral is initialized strictly to [0, 0, 0]
    x_rot_init = np.array([0.0, 0.0, 0.0])
    x_ooc_l_init = np.array(x_ooc_l)
    x_ooc_r_init = np.array(x_ooc_r)
    x_frontalis_init = np.array(x_frontalis)

    total_checkpoints = len(combined_checkpoints)
    active_correction_vectors = None

    # 3. Pre-calculate pre-emptive jaw drift correction for each transition phase
    # (learned directly from lines 1404-1433 in h99-UI-mayascipt-V1.py)
    phase_jaw_corrections = []
    prev_end_cp = 0
    prev_mismatch = 0.0

    for meta_entry in transitions_metadata:
        target_ph = meta_entry['target_phoneme']
        d_cp = meta_entry['dissipation_start_cp_idx']

        ideal_dict = load_ideal_target_positions(target_ph)
        ideal_pitch = 0.0
        if ideal_dict and 'x_joint_rotation_conca' in ideal_dict:
            ideal_pitch = float(ideal_dict['x_joint_rotation_conca'][0])

        nn_pitch_at_target = 0.0
        if d_cp < len(combined_checkpoints):
            cp_data = np.array(combined_checkpoints[d_cp])
            if len(cp_data) > n:
                nn_pitch_at_target = float(cp_data[n])

        drift_mismatch = ideal_pitch - nn_pitch_at_target
        print(f"[PRE-EMPTIVE JAW DRIFT] Phase -> {target_ph}: Target Pitch={ideal_pitch:.4f}, NN Pitch={nn_pitch_at_target:.4f}, Drift Mismatch={drift_mismatch:.4f}")

        phase_jaw_corrections.append({
            'start_cp': prev_end_cp,
            'end_cp': d_cp,
            'start_mismatch': prev_mismatch,
            'end_mismatch': drift_mismatch,
            'ideal_pitch': ideal_pitch,
            'target_phoneme': target_ph
        })
        prev_end_cp = d_cp
        prev_mismatch = drift_mismatch

    print(f"\nExecuting {total_checkpoints} checkpoints from Frame {start_frame_offset + 1} to {start_frame_offset + total_checkpoints}...")

    for i, delta_conca_raw in enumerate(combined_checkpoints):
        frame = start_frame_offset + 1 + i
        delta_b0 = np.array(delta_conca_raw)

        if len(delta_b0) < expected_dim:
            delta_b0 = np.concatenate((delta_b0, np.zeros(expected_dim - len(delta_b0))))

        # Determine active phase pre-emptive jaw drift correction
        active_jaw_corr = None
        for pjc in phase_jaw_corrections:
            if pjc['start_cp'] <= i <= pjc['end_cp']:
                active_jaw_corr = pjc
                break
        if active_jaw_corr is None and phase_jaw_corrections:
            active_jaw_corr = phase_jaw_corrections[-1]

        jaw_corr = 0.0
        if active_jaw_corr is not None:
            s_cp = active_jaw_corr['start_cp']
            e_cp = active_jaw_corr['end_cp']
            s_m = active_jaw_corr['start_mismatch']
            e_m = active_jaw_corr['end_mismatch']
            span = max(1, e_cp - s_cp)
            prog = min(1.0, max(0.0, float(i - s_cp) / float(span)))
            jaw_corr = s_m + (e_m - s_m) * prog

        # Physics baseline deformation + pre-emptive vertical pitch correction
        x_joint_curr = x_joint_init + delta_b0[0:n]
        # STRICTLY vertical dimension (pitch: X-axis only, yaw=0.0, roll=0.0)
        x_rot_curr = np.array([delta_b0[n] + jaw_corr, 0.0, 0.0])

        s1 = n + 3 * joint_rotate_num
        s2 = s1 + joint_lm_num
        x_ooc_l_curr = x_ooc_l_init + delta_b0[s1:s2]

        s3 = s2 + joint_lm_num
        x_ooc_r_curr = x_ooc_r_init + delta_b0[s2:s3]

        s4 = s3 + joint_lm_num
        x_frontalis_curr = x_frontalis_init + delta_b0[s3:s4]

        # ----------------------------------------------------------------------
        # 1. Milestone Event: Check if we reached a "Start Error Dissipation" point
        # ----------------------------------------------------------------------
        for meta_entry in transitions_metadata:
            d_cp = meta_entry['dissipation_start_cp_idx']
            t_idx = meta_entry['transition_index']

            if i == d_cp and ENABLE_FEEDBACK_CONTROL:
                target_ph = meta_entry['target_phoneme']
                from_ph = meta_entry['phoneme_from']
                print(f"\n[MILESTONE: D_{t_idx+1}] Hit 'Start Error Dissipation' at Frame {frame} (Checkpoint {i})")
                print(f"  Phoneme Target: {target_ph}")

                jg = {'oor': oor_num, 'ooc_l': 300, 'ooc_r': 300, 'frontalis': 300}
                actual_pos = retrieve_landmark_positions(jg, frame, is_first_transition=False)
                actual_flat = {
                    'x_joint_conca': [item for sub in actual_pos.get('oor', []) for item in sub],
                    'x_joint_rotation_conca': [x_rot_curr[0], 0.0, 0.0],  # Strictly vertical pitch
                    'x_ooc_l_joint_conca': [item for sub in actual_pos.get('ooc_l', []) for item in sub],
                    'x_ooc_r_joint_conca': [item for sub in actual_pos.get('ooc_r', []) for item in sub],
                    'x_frontalis_joint_conca': [item for sub in actual_pos.get('frontalis', []) for item in sub]
                }

                ideal_pos = load_ideal_target_positions(target_ph)
                ideal_jaw_val = 0.0
                if ideal_pos is not None:
                    if 'x_joint_rotation_conca' in ideal_pos:
                        ideal_jaw_val = float(ideal_pos['x_joint_rotation_conca'][0])
                    new_error = compute_position_error(actual_flat, ideal_pos)
                    err_mag = np.linalg.norm(new_error['x_joint_conca']) if len(new_error.get('x_joint_conca', [])) > 0 else 0.0

                    if ACCUMULATED_ERROR is not None:
                        residual = {}
                        for k in new_error.keys():
                            p_err = np.array(ACCUMULATED_ERROR.get(k, []))
                            n_err = np.array(new_error[k])
                            if len(p_err) > 0:
                                residual[k] = (p_err * (1.0 - ERROR_DISSIPATION_RATE)) + n_err
                            else:
                                residual[k] = n_err
                        ACCUMULATED_ERROR = residual
                    else:
                        ACCUMULATED_ERROR = new_error

                    accum_mag = np.linalg.norm(ACCUMULATED_ERROR['x_joint_conca']) if len(ACCUMULATED_ERROR.get('x_joint_conca', [])) > 0 else 0.0

                    active_correction_vectors = {}
                    for k in ['x_joint_conca', 'x_ooc_l_joint_conca', 'x_ooc_r_joint_conca', 'x_frontalis_joint_conca']:
                        v = np.array(ACCUMULATED_ERROR.get(k, []))
                        if len(v) > 0:
                            active_correction_vectors[k] = v * ERROR_DISSIPATION_RATE

                    corr_mag = np.linalg.norm(active_correction_vectors['x_joint_conca']) if len(active_correction_vectors.get('x_joint_conca', [])) > 0 else 0.0

                    DEBUG_DATA.append({
                        'transition': t_idx + 1,
                        'phoneme_from': from_ph,
                        'phoneme_to': target_ph,
                        'frame': frame,
                        'checkpoint': i,
                        'error_magnitude': err_mag,
                        'accumulated_error_magnitude': accum_mag,
                        'correction_magnitude': corr_mag,
                        'dissipation_rate': ERROR_DISSIPATION_RATE,
                        'jaw_pitch': float(x_rot_curr[0]),
                        'ideal_jaw_pitch': ideal_jaw_val
                    })

                    print(f"  [STEP {t_idx+1}/{len(transitions_metadata)}] Completed {from_ph} -> {target_ph}")
                    print(f"  -> Measured Landmark Error Mag: {err_mag:.6f}")
                    print(f"  -> Accumulated Error Mag: {accum_mag:.6f}")
                    print(f"  -> Prepared Correction Mag: {corr_mag:.6f} for Next Dissipation Window")
                    print(f"  -> Vertical Jaw Rotation: {x_rot_curr[0]:.4f} deg (Ideal: {ideal_jaw_val:.4f} deg)")

        # ----------------------------------------------------------------------
        # 2. Apply Smooth Error Dissipation (Closed-Loop Component for Soft Tissues)
        # ----------------------------------------------------------------------
        if ENABLE_FEEDBACK_CONTROL and active_correction_vectors is not None:
            active_window_meta = None
            for meta_entry in transitions_metadata:
                w_start = meta_entry['window_start_cp_idx']
                w_end = meta_entry['window_end_cp_idx']
                if w_start <= i <= w_end and not meta_entry['is_baseline']:
                    active_window_meta = meta_entry
                    break

            if active_window_meta is not None:
                w_start = active_window_meta['window_start_cp_idx']
                w_end = active_window_meta['window_end_cp_idx']
                span = max(1, w_end - w_start)
                progress = min(1.0, max(0.0, float(i - w_start) / float(span)))

                if 'x_joint_conca' in active_correction_vectors:
                    x_joint_curr += (active_correction_vectors['x_joint_conca'] * progress)
                if 'x_ooc_l_joint_conca' in active_correction_vectors:
                    x_ooc_l_curr += (active_correction_vectors['x_ooc_l_joint_conca'] * progress)
                if 'x_ooc_r_joint_conca' in active_correction_vectors:
                    x_ooc_r_curr += (active_correction_vectors['x_ooc_r_joint_conca'] * progress)
                if 'x_frontalis_joint_conca' in active_correction_vectors:
                    x_frontalis_curr += (active_correction_vectors['x_frontalis_joint_conca'] * progress)

        # ----------------------------------------------------------------------
        # 3. Maya Keyframing & Live Tracking
        # ----------------------------------------------------------------------
        prev_x_joint = x_joint_init if i == 0 else x_joint
        prev_x_rot = x_rot_init if i == 0 else x_rot

        beta_function(
            prev_x_joint, prev_x_rot,
            x_joint_curr, x_rot_curr,
            x_ooc_l_curr, x_ooc_r_curr, x_frontalis_curr,
            frame - 1, frame
        )

        x_joint = x_joint_curr
        x_rot = x_rot_curr

    final_frame = start_frame_offset + total_checkpoints
    if MAYA_AVAILABLE:
        cmds.playbackOptions(minTime=1, maxTime=final_frame)

    # 4. Write Jaw Rotation Log for Verification
    try:
        os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
        with open(CURRENT_JAW_LOG_FILE, "w") as f:
            f.write("=== JAW ROTATION DISSIPATION & DRIFT VERIFICATION LOG ===\n")
            f.write(f"Source: {source_path}\n")
            f.write(f"Frames: {start_frame_offset} -> {final_frame} ({total_checkpoints} checkpoints)\n\n")
            f.write("--- Phoneme Milestones (Vertical Dimension / Pitch) ---\n")
            for entry in DEBUG_DATA:
                f.write(f"  Step {entry['transition']}: {entry['phoneme_from']} -> {entry['phoneme_to']} at Frame {entry['frame']} (CP {entry['checkpoint']})\n")
                f.write(f"    Target Jaw Pitch:  {entry.get('ideal_jaw_pitch', 0.0):.4f} deg\n")
                f.write(f"    Actual Jaw Pitch:  {entry.get('jaw_pitch', 0.0):.4f} deg\n")
                f.write(f"    Yaw (Horizontal):  0.0000 deg\n")
                f.write(f"    Roll (Lateral):    0.0000 deg\n\n")
        print(f"[JAW LOG] Successfully saved jaw verification log to: {CURRENT_JAW_LOG_FILE}")
    except Exception as e:
        print(f"[JAW LOG WARNING] Could not write jaw log: {e}")

    print("\n" + "=" * 70)
    print(f"DEFORMATION COMPLETE! Final Frame: {final_frame}")
    print(f"Transitions Logged: {len(DEBUG_DATA)}")
    print("=" * 70 + "\n")

    return final_frame, DEBUG_DATA, x_rot.tolist() if hasattr(x_rot, 'tolist') else list(x_rot)


# ==============================================================================
# UNIFIED PIPELINE RUNNER (Dev Entry Point)
# ==============================================================================
def run_dev_pipeline(sequence_text="0 to V to AA to V to 0",
                     start_frame=14,
                     overlap_ratio=0.4):
    """
    Unified entry point: Runs overlap blending followed by closed-loop feedback
    controlled deformation with smooth error dissipation.
    """
    print("\n" + "#" * 80)
    print(f"--- RUNNING DEV PIPELINE: '{sequence_text}' ---")
    print("#" * 80)

    # 1. Parse sequence
    phonemes, log_paths = parse_sequence_string(sequence_text)
    folder_name = generate_short_sequence_name(sequence_text, phonemes)

    # 2. Overlap deformation processing & dissipation schedule computation
    checkpoints_path, meta_path, metadata = run_deformation_processing_with_dissipation(
        phonemes_list=phonemes,
        log_paths_list=log_paths,
        sequence_folder_name=folder_name,
        overlap_ratio=overlap_ratio
    )

    # 3. Perform deformation with feedback control in Maya
    source_folder = os.path.dirname(checkpoints_path)
    final_frame, debug_data, final_jaw_rot = perform_the_deformation_with_dissipation(
        source_path=source_folder,
        start_frame_offset=start_frame,
        continue_from_last=False
    )

    print("\n" + "=" * 80)
    print("ALL TRANSITIONS & ERROR DISSIPATION SUCCESSFULLY EXECUTED!")
    print(f"Frames: {start_frame} -> {final_frame}")
    print(f"Transitions: {len(phonemes) - 1}")
    for entry in debug_data:
        print(f"  Step {entry['transition']}: {entry['phoneme_from']} -> {entry['phoneme_to']} (Frame {entry['frame']})")
        print(f"    Error Mag: {entry['error_magnitude']:.6f} | Accum Error: {entry['accumulated_error_magnitude']:.6f} | Dissipation Rate: {entry['dissipation_rate']}")
    print("=" * 80 + "\n")

    return final_frame, debug_data


# ==============================================================================
# Standalone Execution Hook
# ==============================================================================
if __name__ == '__main__':
    TEST_SEQ = "0 to V to AA to V to 0"
    print(f"[DEV SANDBOX] Starting execution for: {TEST_SEQ}")
    try:
        run_dev_pipeline(TEST_SEQ, start_frame=14, overlap_ratio=0.4)
    except Exception as e:
        print(f"[DEV SANDBOX ERROR] {e}")
        import traceback
        traceback.print_exc()
