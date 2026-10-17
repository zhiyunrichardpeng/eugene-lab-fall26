# ==============================================================================
# RUN MODE CONFIGURATION
# ==============================================================================
# By default, use Normal Run. Enable Scheduled Run only for special cases.
# Enable Test Scheduled Run only during testing periods (1 min delay, 1 min window).
ENABLE_SCHEDULING = False
# Supports time-only ("23:00", "8:00") OR full date-time ("2026-09-27 23:00", "2026 Sept 27 23:00", "2026/09/27 23:00")
scheduled_start_time = "23:00"
scheduled_stop_time = "8:00"
TEST_SCHEDULING = False

# Toggle to disable Pre-run Database Cleanup (removing existing DB entries for tasks we are about to run)
ENABLE_DB_CLEANUP = True

# Toggle to skip specified task states/categories from PIPELINE_TASKS
ENABLE_TASK_SKIPPING = True
SKIPPED_STATES = [
    'dataV22_AU12L',
    'dataV22_AU6',
    'dataV22_AU7',
]

manual_task = True

if manual_task == True:

    PIPELINE_TASKS = [
        'dataV22_AU0_neutral to dataV22_phoneme_N',    
        # 'dataV22_AU0_neutral to dataV22_AU7',  
        # 'dataV22_AU7 to dataV22_AU0_neutral', 
        # 'dataV22_AU12L to dataV22_AU1'    
    ]


"""
Pipeline Transition Matrix Tasks Generator and Task List.

Includes all 19 states (11 Action Units + 8 Phonemes) from the Action Unit table,
generating a full transition matrix (all pairwise transitions where src != dst).
"""

# -----------------------------------------------------------------------------
# 1. State Definitions (19 total states)
# -----------------------------------------------------------------------------

ACTION_UNITS = [
    'dataV22_AU0_neutral',     # AU 0: Neutral pose
    'dataV22_AU6',             # AU 6: Cheek raiser
    'dataV22_AU12',            # AU 12: Smile
    'dataV22_AU22',            # AU 22: WUU sound
    'dataV22_AU12L',           # AU 12L: Left asymmetric smile
    'dataV22_AU7',             # AU 7: Lid tightener
    'dataV22_AU25',            # AU 25: Lips part
    'dataV22_AU4',             # AU 4: Brow lowerer
    'dataV22_AU1',             # AU 1: Inner brow raiser
    'dataV22_AU2',             # AU 2: Outer brow raiser
    'dataV22_AU6_12_25',     # AU 6+12+25ii: Cheese (maximum)
]

PHONEMES = [
    'dataV22_phoneme_N',       # Phoneme N: net
    'dataV22_phoneme_AA',      # Phoneme AA: pot
    'dataV22_phoneme_T',       # Phoneme T: test
    'dataV22_phoneme_IY',      # Phoneme IY: beat
    'dataV22_phoneme_V',       # Phoneme V: vet
    'dataV22_phoneme_Y',       # Phoneme Y: yet
    'dataV22_phoneme_UW',      # Phoneme UW: lute
    'dataV22_phoneme_EH',      # Phoneme EH: bet
]

ALL_STATES = ACTION_UNITS + PHONEMES


# -----------------------------------------------------------------------------
# 2. Categorized Transition Task Lists
# -----------------------------------------------------------------------------

# AU to AU transitions (11 x 10 = 110)
AU_TO_AU_TASKS = [
    f"{src} to {dst}"
    for src in ACTION_UNITS
    for dst in ACTION_UNITS
    if src != dst
]

# AU to Phoneme transitions (11 x 8 = 88)
AU_TO_PHONEME_TASKS = [
    f"{src} to {dst}"
    for src in ACTION_UNITS
    for dst in PHONEMES
]

# Phoneme to AU transitions (8 x 11 = 88)
PHONEME_TO_AU_TASKS = [
    f"{src} to {dst}"
    for src in PHONEMES
    for dst in ACTION_UNITS
]

# Phoneme to Phoneme transitions (8 x 7 = 56)
PHONEME_TO_PHONEME_TASKS = [
    f"{src} to {dst}"
    for src in PHONEMES
    for dst in PHONEMES
    if src != dst
]


# -----------------------------------------------------------------------------
# 3. Full Transition Matrix Tasks (19 x 18 = 342 tasks)
# -----------------------------------------------------------------------------

if manual_task == False:
    PIPELINE_TASKS = [
        f"{src} to {dst}"
        for src in ALL_STATES
        for dst in ALL_STATES
        if src != dst
    ]



# --- Optional task skipping ---
if ENABLE_TASK_SKIPPING and SKIPPED_STATES:
    _orig_len = len(PIPELINE_TASKS)
    PIPELINE_TASKS = [
        _t for _t in PIPELINE_TASKS 
        if not any(_s in _t for _s in SKIPPED_STATES)
    ]
    print(f"[TASK FILTERING] Filtered out {_orig_len - len(PIPELINE_TASKS)} tasks containing skipped states: {SKIPPED_STATES}")

# --- Pre-run cleanup: remove existing DB entries for the tasks we are about to run ---
# This ensures re-running a single task doesn't get skipped as "already found".
if ENABLE_DB_CLEANUP:
    import json as _json, os as _os, re as _re
    _DB_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "deformation_path_db.json")
    if _os.path.exists(_DB_PATH):
        with open(_DB_PATH, 'r') as _f:
            _db_data = _json.load(_f)
        _dp = _db_data.get('deformation_paths', _db_data)
        def _normalize(s):
            k = str(s).lower().replace('_to_', ' to ')
            k = _re.sub(r'\s*to\s*', '_to_', k)
            k = _re.sub(r'\s+', '_', k)
            return k
        _keys_to_delete = []
        for _task in PIPELINE_TASKS:
            _norm = _normalize(_task)
            for _k in list(_dp.keys()):
                if _normalize(_k) == _norm:
                    _keys_to_delete.append(_k)
        for _k in _keys_to_delete:
            print(f"[PRE-RUN] Removing existing DB entry: '{_k}' -> {_dp[_k]}")
            del _dp[_k]
        if _keys_to_delete:
            with open(_DB_PATH, 'w') as _f:
                _json.dump(_db_data, _f, indent=4, sort_keys=True)
            print(f"[PRE-RUN] Cleaned {len(_keys_to_delete)} entry(ies) from database.")
else:
    print("[PRE-RUN] Database cleanup is disabled by ENABLE_DB_CLEANUP toggle.")
# --- End pre-run cleanup ---

import os
import re
import sys
import json
import subprocess
import time
import glob
import math
import pickle
import numpy as np
import datetime
import signal
from paths import a3_stochastic_relative_path


sys.path.insert(0, str(a3_stochastic_relative_path))
from automation.checkpoint_generator import run_deformation_processing

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def is_time_only(time_str):
    time_str = str(time_str).strip()
    for fmt in ["%H:%M", "%H:%M:%S"]:
        try:
            datetime.datetime.strptime(time_str, fmt)
            return True
        except ValueError:
            pass
    return False

def parse_schedule_dt(time_str, current_time, is_stop=False):
    """
    Parses time_str which can be:
    - Full date & time: 'YYYY-MM-DD HH:MM', 'YYYY/MM/DD HH:MM', 'YYYY %b %d %H:%M', 'YYYY %B %d %H:%M', etc.
    - Time only: 'HH:MM' (daily schedule)
    Returns a datetime.datetime object representing the target time.
    """
    time_str = str(time_str).strip()
    # Normalize common abbreviations like "Sept " to "Sep " for %b compatibility
    time_str_norm = time_str.replace("Sept ", "Sep ").replace("sept ", "Sep ")
    dt_formats = [
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y %b %d %H:%M",
        "%Y %B %d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y %b %d %H:%M:%S",
        "%Y %B %d %H:%M:%S"
    ]
    for fmt in dt_formats:
        try:
            return datetime.datetime.strptime(time_str_norm, fmt)
        except ValueError:
            pass
            
    # Try time-only formats
    for fmt in ["%H:%M", "%H:%M:%S"]:
        try:
            t = datetime.datetime.strptime(time_str, fmt).time()
            target = datetime.datetime.combine(current_time.date(), t)
            if is_stop:
                if t < current_time.time():
                    target += datetime.timedelta(days=1)
            else:
                if target < current_time:
                    target += datetime.timedelta(days=1)
            return target
        except ValueError:
            pass
            
    raise ValueError(f"Could not parse schedule time string: '{time_str}'. Please use format 'YYYY-MM-DD HH:MM', 'YYYY Sept 27 23:00', or 'HH:MM'.")

def is_time_in_range(start_str, stop_str, current_time):
    if is_time_only(start_str) and is_time_only(stop_str):
        start_time = datetime.datetime.strptime(str(start_str).strip(), "%H:%M").time()
        stop_time = datetime.datetime.strptime(str(stop_str).strip(), "%H:%M").time()
        curr = current_time.time()
        if start_time < stop_time:
            return start_time <= curr <= stop_time
        else: # Crosses midnight
            return curr >= start_time or curr <= stop_time
    else:
        start_dt = parse_schedule_dt(start_str, current_time, is_stop=False)
        stop_dt = parse_schedule_dt(stop_str, current_time, is_stop=True)
        return start_dt <= current_time <= stop_dt

def wait_until_start(start_str, stop_str):
    while True:
        now = datetime.datetime.now()
        if is_time_in_range(start_str, stop_str, now):
            return # Already in the window
        
        if is_time_only(start_str):
            start_time = datetime.datetime.strptime(str(start_str).strip(), "%H:%M").time()
            target = datetime.datetime.combine(now.date(), start_time)
            if target < now:
                target += datetime.timedelta(days=1)
        else:
            target = parse_schedule_dt(start_str, now, is_stop=False)
            if target < now and not is_time_in_range(start_str, stop_str, now):
                print(f"[WARNING] Scheduled start time '{start_str}' is in the past and we are not within the schedule window.")
                return
                
        wait_seconds = (target - now).total_seconds()
        print(f"Waiting {wait_seconds / 60:.2f} minutes until scheduled start time ({start_str})...")
        time.sleep(max(1.0, wait_seconds + 5.0))

def get_seconds_until_stop(stop_str, current_time):
    if is_time_only(stop_str):
        stop_time = datetime.datetime.strptime(str(stop_str).strip(), "%H:%M").time()
        target = datetime.datetime.combine(current_time.date(), stop_time)
        if stop_time < current_time.time():
            target += datetime.timedelta(days=1)
    else:
        target = parse_schedule_dt(stop_str, current_time, is_stop=True)
    return (target - current_time).total_seconds()

def replace_in_file(file_path, pattern, new_string):
    """
    Replaces occurrences of a pattern in a file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content, num_replacements = re.subn(pattern, new_string, content)
        if num_replacements > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"   [SUCCESS] Updated file: {file_path}")
        else:
            print(f"   [INFO] No changes made to file (pattern not found): {file_path}")
    except Exception as e:
        print(f"   [ERROR] Could not modify file {file_path}: {e}")

def apply_test_then_exit_mode(bat_file_path, enabled=True, target_threshold=0.99):
    """
    If enabled (test-then-exit mode), modifies threshold percentages in the bat file to target_threshold (e.g. 0.99).
    If not enabled (normal mode), restores the original parameters:
      --distance_ooc_l_threshold_percentage 0.1092
      --distance_ooc_r_threshold_percentage 0.1
      --distance_frontalis_threshold_percentage 0.1
      --distance_threshold_percentage 0.11
      --distance_jaw_threshold_percentage 0.1
    """
    if not os.path.exists(bat_file_path):
        print(f"   [WARNING] Bat file not found at: {bat_file_path}")
        return

    original_thresholds = {
        "distance_ooc_l_threshold_percentage": "0.1092",
        "distance_ooc_r_threshold_percentage": "0.1",
        "distance_frontalis_threshold_percentage": "0.1",
        "distance_threshold_percentage": "0.11",
        "distance_jaw_threshold_percentage": "0.1"
    }

    try:
        with open(bat_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        new_content = content
        if enabled:
            val_str = str(target_threshold)
            for param in original_thresholds:
                pattern = rf'(--{param}\s+)[\d\.]+'
                new_content = re.sub(pattern, rf'\g<1>{val_str}', new_content)
            
            if new_content != content:
                with open(bat_file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"   [TEST MODE] Updated all threshold percentages to {val_str} in: {bat_file_path}")
            else:
                print(f"   [TEST MODE] Bat file thresholds already set to {val_str}: {bat_file_path}")
        else:
            for param, orig_val in original_thresholds.items():
                pattern = rf'(--{param}\s+)[\d\.]+'
                new_content = re.sub(pattern, rf'\g<1>{orig_val}', new_content)
            
            if new_content != content:
                with open(bat_file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"   [NORMAL MODE] Restored original threshold percentages in: {bat_file_path}")
            else:
                print(f"   [NORMAL MODE] Bat file already using original thresholds: {bat_file_path}")
    except Exception as e:
        print(f"   [ERROR] Failed to update thresholds in bat file: {e}")

def load_deformation_db(db_path):
    """Loads the deformation path database from a JSON file."""
    if not os.path.exists(db_path):
        return {}
    try:
        with open(db_path, 'r') as f:
            data = json.load(f)
            # Check if data has nested structure with 'deformation_paths' key
            if 'deformation_paths' in data:
                return data['deformation_paths']
            # Otherwise return the whole data (backward compatibility)
            return data
    except json.JSONDecodeError:
        print(f"[WARNING] Could not decode JSON from {db_path}. Returning empty database.")
        return {}

def save_deformation_db(db_path, deformation_paths_dict):
    """Saves the deformation path database to a JSON file."""
    # Load existing data to preserve state_map
    existing_data = {}
    if os.path.exists(db_path):
        try:
            with open(db_path, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            pass

    # Normalize all paths to use forward slashes for JSON compatibility
    normalized_paths = {}
    for key, value in deformation_paths_dict.items():
        if isinstance(value, str):
            normalized_paths[key] = value.replace('\\', '/')
        else:
            normalized_paths[key] = value

    # Preserve the nested structure with state_map
    output_data = {
        "deformation_paths": normalized_paths,
        "state_map": existing_data.get("state_map", {})
    }

    with open(db_path, 'w') as f:
        json.dump(output_data, f, indent=4, sort_keys=True)

def get_latest_folder(base_path):
    """
    Finds the most recently modified directory in a given path.
    If the latest directory contains subdirectories, returns the latest subdirectory path.
    This handles the case where results are in job_XXXXXX/XXXXXX format.
    """
    list_of_dirs = glob.glob(os.path.join(base_path, '*/'))
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=os.path.getmtime)

    # Check if this directory has subdirectories
    subdirs = glob.glob(os.path.join(latest_dir, '*/'))
    if subdirs:
        # Find the latest subdirectory
        latest_subdir = max(subdirs, key=os.path.getmtime)
        return os.path.normpath(latest_subdir)

    return os.path.normpath(latest_dir)

def normalize_task_key(task_string):
    """Converts a task string into a standardized key format for lookups."""
    key = str(task_string).lower()
    key = key.replace('_to_', ' to ') # bring to a common format
    key = re.sub(r'\s*to\s*', '_to_', key)
    key = re.sub(r'\s+', '_', key)
    return key

def _create_and_run_maya_wrapper(source_file, script_to_run, save_as_path, mayapy_exe):
    """
    Creates and executes a temporary python script to run a process inside mayapy.
    """
    # Use forward slashes for paths inside the wrapper script string
    source_file_fwd = str(source_file).replace('\\', '/')
    script_to_run_fwd = str(script_to_run).replace('\\', '/')
    save_as_path_fwd = str(save_as_path).replace('\\', '/')

    wrapper_script_content = f"""
# -*- coding: utf-8 -*-
import maya.standalone
import maya.cmds as cmds
import sys
import os

SOURCE_FILE = r"{source_file_fwd}"
SCRIPT_TO_RUN = r"{script_to_run_fwd}"
SAVE_AS_PATH = r"{save_as_path_fwd}"

print("--- Maya Wrapper Script Started ---")
print(f"Source file: {{SOURCE_FILE}}")
print(f"Script to run: {{SCRIPT_TO_RUN}}")
print(f"Save path: {{SAVE_AS_PATH}}")

try:
    maya.standalone.initialize()
    print("Maya standalone initialized.")

    cmds.file(SOURCE_FILE, open=True, force=True)
    print(f"Opened file: {{SOURCE_FILE}}")

    script_dir = os.path.dirname(SCRIPT_TO_RUN)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    print(f"Executing script: {{SCRIPT_TO_RUN}}")
    with open(SCRIPT_TO_RUN, 'r') as f:
        exec(f.read())
    print("Execution complete.")

    cmds.file(rename=SAVE_AS_PATH)
    cmds.file(save=True, force=True, type='mayaBinary')
    print(f"Saved deformed scene to: {{SAVE_AS_PATH}}")

except Exception as e:
    print(f"An error occurred in Maya wrapper: {{e}}")
    sys.exit(1)
finally:
    if 'maya.standalone' in sys.modules:
        print("Uninitializing Maya standalone...")
        maya.standalone.uninitialize()
    print("--- Maya Wrapper Script Finished ---")
"""
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_script_path = os.path.join(project_root, "automation", "tmp_maya_wrapper.py")
    with open(temp_script_path, "w") as f:
        f.write(wrapper_script_content)
        
    print("   Executing temporary Maya wrapper script...")
    
    try:
        subprocess.run([mayapy_exe, temp_script_path], check=True)
    except FileNotFoundError:
        print(f"FATAL ERROR: 'mayapy.exe' not found at the specified path: {mayapy_exe}")
        print("Please correct the MAYAPY_EXE_PATH variable in this script.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error during execution of the Maya wrapper script: {e}")
        sys.exit(1)

    os.remove(temp_script_path)
    print("   Temporary wrapper script removed.")

def main():
    """
    Main function to run the automation pipeline for a sequence of tasks.
    """
    # === CONFIGURATION PANEL ===
    TEST_THEN_EXIT_MODE = False  # Enable "test then exit" mode (sets thresholds to 0.99 for fast testing)

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH = os.path.join(os.path.dirname(__file__), "deformation_path_db.json")
    

    MAYAPY_EXE_PATH = "C:/Program Files/Autodesk/Maya2023/bin/mayapy.exe"
    SOURCE_REST_MODEL_MB_FILE = "C:/keyLabLocal_D/c99-maya-ziva-workshop/experiment_files/g10-10 rest pose 1200 landmarks v9.mb"
    RESULTS_BASE_PATH = "C:/keyLabLocal_D/a3-stochastic-V1-backup-and-results/results/DSAC_V2_m1_vertexMove_Va003"
    
    EXPECTED_STEPS_OVER_30K = False
    TARGET_TISSUE_OBJECT_NAME = "oor"

    # --- New Scheduling and Timeout Config ---
    TASK_TIMEOUT_SECONDS = 1200  # 20 minutes
    
    # Scheduling configuration is now controlled globally at the top of this file
    if TEST_SCHEDULING:
        now = datetime.datetime.now()
        SCHEDULED_START = (now + datetime.timedelta(minutes=1)).strftime("%H:%M")
        SCHEDULED_STOP = (now + datetime.timedelta(minutes=2)).strftime("%H:%M")
        TASK_TIMEOUT_SECONDS = 60  # 1 min timeout for testing
        print(f"[TEST MODE] Scheduled to start at {SCHEDULED_START} and stop at {SCHEDULED_STOP}")
    else:
        SCHEDULED_START = scheduled_start_time # "23:00"
        SCHEDULED_STOP = scheduled_stop_time # "08:00"

    # Set up report file
    report_filename = f"pipeline_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    # report_filepath = os.path.join(os.path.dirname(__file__), report_filename)
    report_dir = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files\tasks pipeline report"
    os.makedirs(report_dir, exist_ok=True)
    report_filepath = os.path.join(report_dir, report_filename)
        
    # Initialize report
    with open(report_filepath, 'w', encoding='utf-8') as rf:
        rf.write(f"Pipeline Execution Report\nStarted at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        rf.write(f"Schedule: {SCHEDULED_START} to {SCHEDULED_STOP}\n")
        rf.write("-" * 50 + "\n")
    # === END OF CONFIGURATION ===

    print("--- Starting Automation Pipeline ---")
    
    deformation_db = load_deformation_db(DB_PATH)
    print("--- DEBUG: Loaded Database Content ---")
    print(json.dumps(deformation_db, indent=2))

    db_key_map = {normalize_task_key(k): k for k in deformation_db}
    print("--- DEBUG: Normalized DB Key Map ---")
    print(json.dumps(db_key_map, indent=2))

    if ENABLE_SCHEDULING:
        wait_until_start(SCHEDULED_START, SCHEDULED_STOP)

    batch_start_time = datetime.datetime.now()
    tasks_succeeded = 0
    tasks_failed = 0
    tasks_timed_out = 0
    total_task_duration = 0

    for task_index, task_string in enumerate(PIPELINE_TASKS):
        task_start_time = datetime.datetime.now()
        
        if ENABLE_SCHEDULING:
            now = datetime.datetime.now()
            if not is_time_in_range(SCHEDULED_START, SCHEDULED_STOP, now):
                print(f"[INFO] Current time {now.strftime('%Y-%m-%d %H:%M:%S')} is outside the scheduled window ({SCHEDULED_START} - {SCHEDULED_STOP}). Stopping task execution.")
                with open(report_filepath, 'a', encoding='utf-8') as rf:
                    rf.write(f"\n--- Batch stopped at {now.strftime('%Y-%m-%d %H:%M:%S')} due to schedule window closing ---\n")
                break

        print(f"\n\n{'='*20} STARTING TASK {task_index + 1}/{len(PIPELINE_TASKS)}: {task_string} {'='*20}")

        normalized_task_key = normalize_task_key(task_string)
        
        if normalized_task_key in db_key_map:
            print(f"[INFO] Task '{task_string}' already found in database. Skipping.")
            continue

        try:
            from_pose, to_pose = [part.strip() for part in re.split(r'\s+to\s+|_to_', task_string, flags=re.IGNORECASE)]
        except ValueError:
            print(f"[ERROR] Invalid task format: '{task_string}'. Skipping.")
            continue
        
        if normalize_task_key(from_pose) == normalize_task_key(to_pose):
            print(f"[INFO] Skipping 'identity' task: '{task_string}'")
            continue

        IS_NEUTRAL_TO_OTHER_POSE = (normalize_task_key(from_pose) == 'datav22_au0_neutral')
        IS_NON_NEUTRAL_REST_POSE = not IS_NEUTRAL_TO_OTHER_POSE

        GENERATED_RESULTS_PATH = ""
        if IS_NON_NEUTRAL_REST_POSE:
            source_task_string = f"dataV22_AU0_neutral to {from_pose}"
            normalized_source_key = normalize_task_key(source_task_string)
            
            print(f"[INFO] Task starts from non-neutral pose '{from_pose}'. Looking for prerequisite task '{source_task_string}'...")
            print(f"--- DEBUG: Prerequisite Lookup ---")
            print(f"  Normalized key to find: '{normalized_source_key}'")

            original_source_key = db_key_map.get(normalized_source_key)
            
            if not original_source_key:
                error_msg = f"[ERROR] Cannot perform task. Prerequisite task '{source_task_string}' not found in database. Please run that task first."
                print(error_msg)
                
                with open(report_filepath, 'a', encoding='utf-8') as rf:
                    rf.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Task {task_index + 1}: {task_string} -> FAILED (Missing Prerequisite)\n")
                    rf.write(f"   Error Details: {error_msg}\n")
                
                tasks_failed += 1
                task_finish_time = datetime.datetime.now()
                total_task_duration += (task_finish_time - task_start_time).total_seconds()
                continue
            
            GENERATED_RESULTS_PATH = deformation_db[original_source_key]
            print(f"[INFO] Found prerequisite path: {GENERATED_RESULTS_PATH}")
        
        j77_file = os.path.join(PROJECT_ROOT, "a3-stochastic-V1-algo", "example_train", "j77_V3_estimate_r_limit_newest.py")
        d2_helper_file = os.path.join(PROJECT_ROOT, "b99-frame1to4-get-b1to3", "d2 helper functions", "d2-adhoc-for-b99-part2-generate-mb-use-V3.py")
        d1_b99_file = os.path.join(PROJECT_ROOT, "b99-frame1to4-get-b1to3", "d1-adhoc-for-b99.py")

        if IS_NON_NEUTRAL_REST_POSE:
            print(f"Automating generation of rest pose for '{from_pose}'...")
            rest_pose_mb_path = f"C:/keyLabLocal_D/c99-maya-ziva-workshop/different pose as rest pose_auto/{from_pose}_as_rest_pose.mb"

            if os.path.exists(rest_pose_mb_path):
                print(f"[INFO] Rest pose file already exists at '{rest_pose_mb_path}'. Skipping generation.")
            else:
                # Check for and generate combined_checkpoints.pkl if needed
                checkpoint_file_path = os.path.join(GENERATED_RESULTS_PATH, "combined_checkpoints.pkl")
                if not os.path.exists(checkpoint_file_path):
                    print(f"[INFO] 'combined_checkpoints.pkl' not found in '{GENERATED_RESULTS_PATH}'. Generating it now...")
                    
                    prereq_from_pose = 'dataV22_AU0_neutral' # The prerequisite always starts from neutral
                    
                    phonemes = [prereq_from_pose, from_pose]
                    log_paths = [GENERATED_RESULTS_PATH]
                    
                    try:
                        run_deformation_processing(phonemes, log_paths, checkpoint_file_path)
                        print("[SUCCESS] Successfully generated 'combined_checkpoints.pkl'.")
                    except Exception as e:
                        error_msg = f"[ERROR] Failed to generate 'combined_checkpoints.pkl': {e}"
                        print(error_msg)
                        print("[ERROR] Cannot proceed without combined_checkpoints.pkl. Skipping current task.")
                        
                        with open(report_filepath, 'a', encoding='utf-8') as rf:
                            rf.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Task {task_index + 1}: {task_string} -> FAILED (Checkpoint Error)\n")
                            rf.write(f"   Error Details: {error_msg}\n")
                        
                        tasks_failed += 1
                        task_finish_time = datetime.datetime.now()
                        total_task_duration += (task_finish_time - task_start_time).total_seconds()
                        continue
                else:
                    print(f"[INFO] Found 'combined_checkpoints.pkl' in '{GENERATED_RESULTS_PATH}'.")

                replace_in_file(d2_helper_file, r'path2_combined_checkpoints\s*=\s*.+', f'path2_combined_checkpoints = r"{GENERATED_RESULTS_PATH}"')
                _create_and_run_maya_wrapper(SOURCE_REST_MODEL_MB_FILE, d2_helper_file, rest_pose_mb_path, MAYAPY_EXE_PATH)
            
            replace_in_file(j77_file, r'maya_file_to_open\s*=\s*.+', f'maya_file_to_open = r"{rest_pose_mb_path}"')
        else:
             replace_in_file(j77_file, r'maya_file_to_open\s*=\s*.+', f'maya_file_to_open = r"{SOURCE_REST_MODEL_MB_FILE}"')

        print("Configuring j77 task...")
        replace_in_file(j77_file, r"this_task\s*=\s*'.*?'", f"this_task = '{from_pose} to {to_pose}'")
        replace_in_file(j77_file, r'from_pose\s*=\s*".*?"', f'from_pose = "{from_pose}"')
        replace_in_file(j77_file, r'to_pose\s*=\s*".*?"', f'to_pose = "{to_pose}"')

        # Update copy_from to point to the correct dataV22 file based on to_pose
        copy_from_path = f"C:/keyLabLocal_D/a3-stochastic-V1-backup-and-results/backup/b31 dataV22 backup hub/{to_pose}.py"
        replace_in_file(j77_file, r'copy_from\s*=\s*r"[^"]*"', f'copy_from = r"{copy_from_path}"')
        # copy_from_path = f"C:/keyLabLocal_D/a3-stochastic-V1-backup-and-results/backup/b31 dataV22 backup hub/{from_pose}.py"
        # replace_in_file(j77_file, r'copy_from\s*=\s*r"[^"]*"', f'copy_from = r"{copy_from_path}"')        

        print("Configuring d1-adhoc task...")
        # replace_in_file(d1_b99_file, r'neutral_to_other_pose_flag\s*=\s*.*', f"neutral_to_other_pose_flag = {1 if IS_NEUTRAL_TO_OTHER_POSE else 0}")
        replace_in_file(d1_b99_file, r'neutral_to_other_pose_flag\s*=\s*[01]', f"neutral_to_other_pose_flag = {1 if IS_NEUTRAL_TO_OTHER_POSE else 0}")
        if not IS_NEUTRAL_TO_OTHER_POSE:
            replace_in_file(d1_b99_file, r'previous_pose_folder\s*=\s*.+', f'previous_pose_folder = r"{GENERATED_RESULTS_PATH}"')

        print("Executing main pipeline batch file...")
        bat_file_path = os.path.join(PROJECT_ROOT, "b99-frame1to4-get-b1to3", "run_d1_b99.bat")
        if not os.path.exists(bat_file_path):
            bat_file_path = os.path.join(os.path.dirname(__file__), "run_d1_b99.bat")

        # Apply test then exit mode threshold modifications if enabled
        apply_test_then_exit_mode(bat_file_path, enabled=TEST_THEN_EXIT_MODE, target_threshold=0.9)

        # Dynamically shorten timeout if schedule stop time is approaching
        dynamic_timeout = TASK_TIMEOUT_SECONDS
        if ENABLE_SCHEDULING:
            secs_until_stop = get_seconds_until_stop(SCHEDULED_STOP, datetime.datetime.now())
            if secs_until_stop < dynamic_timeout:
                dynamic_timeout = max(1, int(secs_until_stop))
                print(f"[INFO] Adjusting task timeout to {dynamic_timeout}s to prevent running past the scheduled stop time.")

        try:
            print(f"Executing main pipeline batch file with timeout {dynamic_timeout}s...")
            # Use Popen with CREATE_NEW_PROCESS_GROUP to ensure all child processes (Maya) are cleanly killed
            process = subprocess.Popen(
                bat_file_path, 
                shell=True, 
                text=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            
            try:
                stdout, stderr = process.communicate(timeout=dynamic_timeout)
                if stdout: print(stdout)
                
                if process.returncode == 0:
                    print("Batch file execution complete.")
                    task_status = "SUCCESS"
                    task_error = ""
                else:
                    print(f"[ERROR] Task failed with return code {process.returncode}!")
                    if stdout: print("STDOUT:\n", stdout)
                    if stderr: print("STDERR:\n", stderr)
                    task_status = "FAILED"
                    out_str = (stderr or stdout or "No output available.")
                    task_error = out_str.strip()[-500:]
            except subprocess.TimeoutExpired:
                print(f"[ERROR] Task timed out after {dynamic_timeout} seconds! Force killing process tree with taskkill...")
                # Hard force stop using Windows taskkill
                subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True, capture_output=True)
                stdout, stderr = process.communicate()  # Collect any remaining output
                if stdout: print(stdout)
                task_status = "TIMEOUT"
                task_error = f"TimeoutExpired after {dynamic_timeout}s"
        except Exception as e:
            print(f"[ERROR] Unexpected execution failure: {e}")
            task_status = "FAILED"
            task_error = str(e)[-500:]

        task_finish_time = datetime.datetime.now()
        task_duration = (task_finish_time - task_start_time).total_seconds()
        total_task_duration += task_duration
        
        if task_status == "SUCCESS":
            tasks_succeeded += 1
        elif task_status == "TIMEOUT":
            tasks_timed_out += 1
        else:
            tasks_failed += 1

        # Append to report immediately
        with open(report_filepath, 'a', encoding='utf-8') as rf:
            rf.write(f"[{task_finish_time.strftime('%H:%M:%S')}] Task {task_index + 1}: {task_string} -> {task_status} (took {task_duration:.2f}s)\n")
            if task_status != "SUCCESS":
                rf.write(f"   Error Details: {task_error}\n")

        if task_status != "SUCCESS":
            # If we failed/timed out, skip to the next task
            continue

        # If we reach here, the batch file succeeded (check=True would have raised exception otherwise)
        # Check for new result folder
        time.sleep(5)
        new_result_path = get_latest_folder(RESULTS_BASE_PATH)

        if new_result_path:
            clean_path = new_result_path.replace('\\', '/')
            print(f"Found new result path: {clean_path}")
            deformation_db[task_string] = clean_path
            save_deformation_db(DB_PATH, deformation_db)
            db_key_map[normalize_task_key(task_string)] = task_string
            print(f"Saved new path to database for task '{task_string}'.")
        else:
            print("[ERROR] Could not find the new result path after execution. Database not updated.")

    batch_finish_time = datetime.datetime.now()
    batch_duration = (batch_finish_time - batch_start_time).total_seconds()
    total_processed = tasks_succeeded + tasks_failed + tasks_timed_out
    avg_duration = (total_task_duration / total_processed) if total_processed > 0 else 0

    summary = f"""
==================================================
BATCH EXECUTION SUMMARY
==================================================
Finished at: {batch_finish_time.strftime('%Y-%m-%d %H:%M:%S')}
Total Batch Duration: {batch_duration / 60:.2f} minutes

Tasks Processed: {total_processed}
  - Succeeded: {tasks_succeeded}
  - Failed: {tasks_failed}
  - Timed Out: {tasks_timed_out}

Average Time per Task: {avg_duration:.2f} seconds
==================================================
"""
    print(summary)
    with open(report_filepath, 'a', encoding='utf-8') as rf:
        rf.write(summary)

    print(f"\n\n{'='*20} ALL PIPELINE TASKS COMPLETE {'='*20}")

if __name__ == "__main__":
    main()




# PIPELINE_TASKS = [
#     'dataV22_AU6 to dataV22_AU7',
#     'dataV22_AU7 to dataV22_AU6'    
# ]    


# PIPELINE_TASKS = [
#     'dataV22_AU0_neutral to dataV22_phoneme_EH',

#     'dataV22_phoneme_Y to dataV22_phoneme_EH',
#     'dataV22_phoneme_EH to dataV22_phoneme_T',    

#     # AU0 (Neutral) transitions (Can go to everything except itself)
	
#     'dataV22_AU0_neutral to dataV22_AU6',
#     'dataV22_AU0_neutral to dataV22_AU12',
#     'dataV22_AU0_neutral to dataV22_AU22',
#     'dataV22_AU0_neutral to dataV22_AU12L',
#     'dataV22_AU0_neutral to dataV22_AU7',
#     'dataV22_AU0_neutral to dataV22_AU25',
#     'dataV22_AU0_neutral to dataV22_AU4',
#     'dataV22_AU0_neutral to dataV22_AU1',
#     'dataV22_AU0_neutral to dataV22_AU2',
#     'dataV22_AU0_neutral to dataV22_phoneme_N',
#     'dataV22_AU0_neutral to dataV22_phoneme_AA',
#     'dataV22_AU0_neutral to dataV22_phoneme_T',
#     'dataV22_AU0_neutral to dataV22_phoneme_IY',
#     'dataV22_AU0_neutral to dataV22_phoneme_V',
#     'dataV22_AU0_neutral to dataV22_phoneme_Y',
#     'dataV22_AU0_neutral to dataV22_phoneme_UW',
#     'dataV22_AU0_neutral to dataV22_phoneme_EH',

#     # AU Transitions (Can go to other AUs, but blocked from Phonemes by '\')
#     'dataV22_AU6 to dataV22_AU0_neutral',
#     'dataV22_AU6 to dataV22_AU12',
#     'dataV22_AU6 to dataV22_AU22',
#     # 'dataV22_AU6 to dataV22_AU12L',
#     'dataV22_AU6 to dataV22_AU7',
#     'dataV22_AU6 to dataV22_AU25',
#     'dataV22_AU6 to dataV22_AU4',
#     'dataV22_AU6 to dataV22_AU1',
#     'dataV22_AU6 to dataV22_AU2',

#     'dataV22_AU12 to dataV22_AU0_neutral',
#     'dataV22_AU12 to dataV22_AU6',
#     'dataV22_AU12 to dataV22_AU22',
#     # 'dataV22_AU12 to dataV22_AU12L',
#     'dataV22_AU12 to dataV22_AU7',
#     'dataV22_AU12 to dataV22_AU25',
#     'dataV22_AU12 to dataV22_AU4',
#     'dataV22_AU12 to dataV22_AU1',
#     'dataV22_AU12 to dataV22_AU2',

#     'dataV22_AU22 to dataV22_AU0_neutral',
#     'dataV22_AU22 to dataV22_AU6',
#     'dataV22_AU22 to dataV22_AU12',
#     # 'dataV22_AU22 to dataV22_AU12L',
#     'dataV22_AU22 to dataV22_AU7',
#     'dataV22_AU22 to dataV22_AU25',
#     'dataV22_AU22 to dataV22_AU4',
#     'dataV22_AU22 to dataV22_AU1',
#     'dataV22_AU22 to dataV22_AU2',

#     # 'dataV22_AU12L to dataV22_AU0_neutral',
#     # 'dataV22_AU12L to dataV22_AU6',
#     # 'dataV22_AU12L to dataV22_AU12',
#     # 'dataV22_AU12L to dataV22_AU22',
#     # 'dataV22_AU12L to dataV22_AU7',
#     # 'dataV22_AU12L to dataV22_AU25',
#     # 'dataV22_AU12L to dataV22_AU4',
#     # 'dataV22_AU12L to dataV22_AU1',
#     # 'dataV22_AU12L to dataV22_AU2',

#     'dataV22_AU7 to dataV22_AU0_neutral',
#     'dataV22_AU7 to dataV22_AU6',
#     'dataV22_AU7 to dataV22_AU12',
#     'dataV22_AU7 to dataV22_AU22',
#     # 'dataV22_AU7 to dataV22_AU12L',
#     'dataV22_AU7 to dataV22_AU25',
#     'dataV22_AU7 to dataV22_AU4',
#     'dataV22_AU7 to dataV22_AU1',
#     'dataV22_AU7 to dataV22_AU2',

#     'dataV22_AU25 to dataV22_AU0_neutral',
#     'dataV22_AU25 to dataV22_AU6',
#     'dataV22_AU25 to dataV22_AU12',
#     'dataV22_AU25 to dataV22_AU22',
#     # 'dataV22_AU25 to dataV22_AU12L',
#     'dataV22_AU25 to dataV22_AU7',
#     'dataV22_AU25 to dataV22_AU4',
#     'dataV22_AU25 to dataV22_AU1',
#     'dataV22_AU25 to dataV22_AU2',

#     'dataV22_AU4 to dataV22_AU0_neutral',
#     'dataV22_AU4 to dataV22_AU6',
#     'dataV22_AU4 to dataV22_AU12',
#     'dataV22_AU4 to dataV22_AU22',
#     # 'dataV22_AU4 to dataV22_AU12L',
#     'dataV22_AU4 to dataV22_AU7',
#     'dataV22_AU4 to dataV22_AU25',
#     'dataV22_AU4 to dataV22_AU1',
#     'dataV22_AU4 to dataV22_AU2',

#     'dataV22_AU1 to dataV22_AU0_neutral',
#     'dataV22_AU1 to dataV22_AU6',
#     'dataV22_AU1 to dataV22_AU12',
#     'dataV22_AU1 to dataV22_AU22',
#     # 'dataV22_AU1 to dataV22_AU12L',
#     'dataV22_AU1 to dataV22_AU7',
#     'dataV22_AU1 to dataV22_AU25',
#     'dataV22_AU1 to dataV22_AU4',
#     'dataV22_AU1 to dataV22_AU2',

#     'dataV22_AU2 to dataV22_AU0_neutral',
#     'dataV22_AU2 to dataV22_AU6',
#     'dataV22_AU2 to dataV22_AU12',
#     'dataV22_AU2 to dataV22_AU22',
#     # 'dataV22_AU2 to dataV22_AU12L',
#     'dataV22_AU2 to dataV22_AU7',
#     'dataV22_AU2 to dataV22_AU25',
#     'dataV22_AU2 to dataV22_AU4',
#     'dataV22_AU2 to dataV22_AU1',

#     # Phoneme Transitions (Can go to AU0 and other Phonemes, but blocked from other AUs)
#     'dataV22_phoneme_N to dataV22_AU0_neutral',
#     'dataV22_phoneme_N to dataV22_phoneme_AA',
#     'dataV22_phoneme_N to dataV22_phoneme_T',
#     'dataV22_phoneme_N to dataV22_phoneme_IY',
#     'dataV22_phoneme_N to dataV22_phoneme_V',
#     'dataV22_phoneme_N to dataV22_phoneme_Y',
#     'dataV22_phoneme_N to dataV22_phoneme_UW',
#     'dataV22_phoneme_N to dataV22_phoneme_EH',

#     'dataV22_phoneme_AA to dataV22_AU0_neutral',
#     'dataV22_phoneme_AA to dataV22_phoneme_N',
#     'dataV22_phoneme_AA to dataV22_phoneme_T',
#     'dataV22_phoneme_AA to dataV22_phoneme_IY',
#     'dataV22_phoneme_AA to dataV22_phoneme_V',
#     'dataV22_phoneme_AA to dataV22_phoneme_Y',
#     'dataV22_phoneme_AA to dataV22_phoneme_UW',
#     'dataV22_phoneme_AA to dataV22_phoneme_EH',

#     'dataV22_phoneme_T to dataV22_AU0_neutral',
#     'dataV22_phoneme_T to dataV22_phoneme_N',
#     'dataV22_phoneme_T to dataV22_phoneme_AA',
#     'dataV22_phoneme_T to dataV22_phoneme_IY',
#     'dataV22_phoneme_T to dataV22_phoneme_V',
#     'dataV22_phoneme_T to dataV22_phoneme_Y',
#     'dataV22_phoneme_T to dataV22_phoneme_UW',
#     'dataV22_phoneme_T to dataV22_phoneme_EH',

#     'dataV22_phoneme_IY to dataV22_AU0_neutral',
#     'dataV22_phoneme_IY to dataV22_phoneme_N',
#     'dataV22_phoneme_IY to dataV22_phoneme_AA',
#     'dataV22_phoneme_IY to dataV22_phoneme_T',
#     'dataV22_phoneme_IY to dataV22_phoneme_V',
#     'dataV22_phoneme_IY to dataV22_phoneme_Y',
#     'dataV22_phoneme_IY to dataV22_phoneme_UW',
#     'dataV22_phoneme_IY to dataV22_phoneme_EH',

#     'dataV22_phoneme_V to dataV22_AU0_neutral',
#     'dataV22_phoneme_V to dataV22_phoneme_N',
#     'dataV22_phoneme_V to dataV22_phoneme_AA',
#     'dataV22_phoneme_V to dataV22_phoneme_T',
#     'dataV22_phoneme_V to dataV22_phoneme_IY',
#     'dataV22_phoneme_V to dataV22_phoneme_Y',
#     'dataV22_phoneme_V to dataV22_phoneme_UW',
#     'dataV22_phoneme_V to dataV22_phoneme_EH',

#     'dataV22_phoneme_Y to dataV22_AU0_neutral',
#     'dataV22_phoneme_Y to dataV22_phoneme_N',
#     'dataV22_phoneme_Y to dataV22_phoneme_AA',
#     'dataV22_phoneme_Y to dataV22_phoneme_T',
#     'dataV22_phoneme_Y to dataV22_phoneme_IY',
#     'dataV22_phoneme_Y to dataV22_phoneme_V',
#     'dataV22_phoneme_Y to dataV22_phoneme_UW',
#     'dataV22_phoneme_Y to dataV22_phoneme_EH',

#     'dataV22_phoneme_UW to dataV22_AU0_neutral',
#     'dataV22_phoneme_UW to dataV22_phoneme_N',
#     'dataV22_phoneme_UW to dataV22_phoneme_AA',
#     'dataV22_phoneme_UW to dataV22_phoneme_T',
#     'dataV22_phoneme_UW to dataV22_phoneme_IY',
#     'dataV22_phoneme_UW to dataV22_phoneme_V',
#     'dataV22_phoneme_UW to dataV22_phoneme_Y',
#     'dataV22_phoneme_UW to dataV22_phoneme_EH',

#     # New Phoneme EH Transitions
#     'dataV22_phoneme_EH to dataV22_AU0_neutral',
#     'dataV22_phoneme_EH to dataV22_phoneme_N',
#     'dataV22_phoneme_EH to dataV22_phoneme_AA',
#     'dataV22_phoneme_EH to dataV22_phoneme_T',
#     'dataV22_phoneme_EH to dataV22_phoneme_IY',
#     'dataV22_phoneme_EH to dataV22_phoneme_V',
#     'dataV22_phoneme_EH to dataV22_phoneme_Y',
#     'dataV22_phoneme_EH to dataV22_phoneme_UW'

#     # # From AU7 to AU12/AU22/AU25
#     # 'dataV22_AU7 to dataV22_AU12',
#     # 'dataV22_AU7 to dataV22_AU22',
#     # 'dataV22_AU7 to dataV22_AU25',
    
#     # # From AU7 to AU0_neutral/AU1/AU2/AU4
#     # 'dataV22_AU7 to dataV22_AU0_neutral',
#     # 'dataV22_AU7 to dataV22_AU1',
#     # 'dataV22_AU7 to dataV22_AU2',
#     # 'dataV22_AU7 to dataV22_AU4',
#     # # 'dataV22_AU7 to dataV22_AU6',
#     # 'dataV22_AU7 to dataV22_AU12',
#     # 'dataV22_AU7 to dataV22_AU22',
#     # 'dataV22_AU7 to dataV22_AU25'
# ]

# PIPELINE_TASKS = [
#     # To AU7
#     # 'dataV22_AU0_neutral to dataV22_AU7',
#     # 'dataV22_AU1 to dataV22_AU7',
#     # 'dataV22_AU2 to dataV22_AU7',
#     # 'dataV22_AU4 to dataV22_AU7',
#     # 'dataV22_AU6 to dataV22_AU7',

# ]