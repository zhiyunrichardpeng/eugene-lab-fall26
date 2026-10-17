"""
Precompute Helper for combined_checkpoints.pkl

This script iterates through all transition paths in 'automation/deformation_path_db.json'
and pre-generates 'combined_checkpoints.pkl' for any transitions that don't have one yet.

This allows the animation pipeline (h99-UI-mayascipt-V1.py) to run without waiting
for processing when generating new animations.

Usage:
    Run this script standalone (not in Maya) to precompute all missing checkpoints.
"""

import os
import sys
import json
import re
import pickle
import math
import numpy as np
from paths import a3_stochastic_relative_path

# --- PATH CONFIGURATION ---
BASE_DIR = str(a3_stochastic_relative_path)
DB_PATH = os.path.join(BASE_DIR, "automation", "deformation_path_db.json")
OUTPUT_BASE_PATH = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\sequence_combined_deformation_parameters_hub"


def load_database():
    """Load the deformation path database and state map from the JSON file"""
    try:
        with open(DB_PATH, 'r') as f:
            data = json.load(f)
            return data.get('deformation_paths', {}), data.get('state_map', {})
    except FileNotFoundError:
        print(f"ERROR: Database file not found at {DB_PATH}")
        return {}, {}
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON database: {e}")
        return {}, {}


def normalize_task_key(task_string):
    """Converts a task string into a standardized key format for lookups."""
    key = str(task_string).lower()
    key = key.replace('_to_', ' to ')  # bring to a common format
    key = re.sub(r'\s*to\s*', '_to_', key)
    key = re.sub(r'\s+', '_', key)
    return key


def generate_short_sequence_name(transition_key, from_state, to_state):
    """
    Generate a short folder name for the checkpoint output.
    """
    # Simplify the state names
    from_short = from_state.replace("dataV22_", "").replace("AU0_neutral", "neutral").replace("phoneme_", "")
    to_short = to_state.replace("dataV22_", "").replace("AU0_neutral", "neutral").replace("phoneme_", "")
    
    return f"seq_{from_short}_to_{to_short}"


def extract_phonemes_from_key(transition_key, state_map):
    """
    Extract from/to phoneme names from a transition key.
    
    Examples:
        "dataV22_AU0_neutral to dataV22_phoneme_N" -> ("dataV22_AU0_neutral", "dataV22_phoneme_N")
        "dataV22_phoneme_N to dataV22_phoneme_IY" -> ("dataV22_phoneme_N", "dataV22_phoneme_IY")
    """
    # Try different separators
    if ' to ' in transition_key:
        parts = transition_key.split(' to ')
    elif '_to_' in transition_key:
        parts = transition_key.split('_to_')
    else:
        return None, None
    
    if len(parts) != 2:
        return None, None
    
    from_state = parts[0].strip()
    to_state = parts[1].strip()
    
    return from_state, to_state


def run_deformation_processing_standalone(log_paths_list, sequence_folder_name):
    """
    Standalone version of run_deformation_processing that works outside Maya.
    This generates the combined_checkpoints.pkl file.
    """
    print(f"\n--- Starting Deformation Processing ---")
    print(f"Log Paths: {log_paths_list}")
    print(f"Output Folder: {sequence_folder_name}")
    
    overlap_ratio = 0
    log_paths = log_paths_list
    
    # Create output directory
    log_path_output = os.path.join(OUTPUT_BASE_PATH, sequence_folder_name)
    if not os.path.exists(log_path_output):
        os.makedirs(log_path_output)
        print(f"Created directory: {log_path_output}")
    else:
        print(f"Directory already exists: {log_path_output}")
    
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
                print(f"WARNING: File not found: {file_path}")
        
        print(f"total_steps_convergence = {total_steps_convergence}")
        return total_steps_convergence
    
    total_steps_convergence = get_total_steps_from_separate_logs()
    
    if len(total_steps_convergence) <= 1:
        print("ERROR: No step counts found. Cannot generate checkpoints.")
        return False
    
    # Calculate phase boundaries
    phase_boundaries = np.cumsum(total_steps_convergence).tolist()
    print(f"Phase boundaries: {phase_boundaries}")
    
    # Load action lists from separate paths
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
            print(f"WARNING: Could not find action list at {file_path}")
            action_lists.append([])
        except Exception as e:
            print(f"ERROR loading {file_path}: {e}")
            action_lists.append([])
    
    # Flatten all action lists
    all_actions = []
    for phase_actions in action_lists:
        all_actions.extend(phase_actions)
    
    if len(all_actions) == 0:
        print("ERROR: No actions found. Cannot generate checkpoints.")
        return False
    
    print(f"Total actions loaded: {len(all_actions)}")
    
    # Calculate checkpoint selection (36 checkpoints)
    num_checkpoints = 36
    total_steps = len(all_actions)
    
    if total_steps <= num_checkpoints:
        # Use all steps
        checkpoint_indices = list(range(total_steps))
    else:
        # Evenly distribute
        step_size = total_steps / num_checkpoints
        checkpoint_indices = [int(i * step_size) for i in range(num_checkpoints)]
        # Make sure last checkpoint is the final step
        if checkpoint_indices[-1] != total_steps - 1:
            checkpoint_indices[-1] = total_steps - 1
    
    print(f"Selected {len(checkpoint_indices)} checkpoint indices")
    
    # Calculate cumulative deformation at each checkpoint
    combined_checkpoints = []
    cumulative = None
    
    for i, checkpoint_idx in enumerate(checkpoint_indices):
        # Sum all actions up to this checkpoint
        if cumulative is None:
            cumulative = np.array(all_actions[0])
        
        # Accumulate from previous checkpoint to this one
        prev_idx = checkpoint_indices[i-1] if i > 0 else 0
        for j in range(prev_idx + 1, checkpoint_idx + 1):
            if j < len(all_actions):
                cumulative = cumulative + np.array(all_actions[j])
        
        combined_checkpoints.append((checkpoint_idx, cumulative.tolist()))
    
    # Save results
    output_path = os.path.join(log_path_output, "combined_checkpoints.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump([cp[1] for cp in combined_checkpoints], f)
    
    print(f"Results saved to: {output_path}")
    
    # Verification
    try:
        with open(output_path, "rb") as f:
            loaded_checkpoints = pickle.load(f)
        print(f"Verification: Loaded {len(loaded_checkpoints)} checkpoints")
        return True
    except Exception as e:
        print(f"ERROR verifying output: {e}")
        return False


def precompute_all_checkpoints(dry_run=False):
    """
    Main function to precompute all missing combined_checkpoints.pkl files.
    
    Args:
        dry_run: If True, only report what would be done without actually processing
    """
    print("=" * 70)
    print("PRECOMPUTE HELPER: combined_checkpoints.pkl Generator")
    print("=" * 70)
    
    deformation_paths, state_map = load_database()
    
    if not deformation_paths:
        print("ERROR: No deformation paths found in database")
        return
    
    print(f"\nFound {len(deformation_paths)} transitions in database\n")
    
    # Track statistics
    already_exists = []
    needs_processing = []
    processing_failed = []
    processing_succeeded = []
    
    for transition_key, source_path in deformation_paths.items():
        # Extract from/to states
        from_state, to_state = extract_phonemes_from_key(transition_key, state_map)
        
        if from_state is None or to_state is None:
            print(f"SKIP: Could not parse transition key: {transition_key}")
            continue
        
        # Generate output folder name
        sequence_folder_name = generate_short_sequence_name(transition_key, from_state, to_state)
        
        # Check if combined_checkpoints.pkl already exists
        checkpoint_path = os.path.join(OUTPUT_BASE_PATH, sequence_folder_name, "combined_checkpoints.pkl")
        
        # Also check in the source path (some might already have it there)
        source_checkpoint_path = os.path.join(source_path, "combined_checkpoints.pkl")
        
        if os.path.exists(checkpoint_path):
            already_exists.append(transition_key)
            print(f"EXISTS: {transition_key}")
            print(f"        -> {checkpoint_path}")
            continue
        
        if os.path.exists(source_checkpoint_path):
            # Copy from source to output location
            print(f"FOUND IN SOURCE: {transition_key}")
            print(f"        -> {source_checkpoint_path}")
            
            # Create output directory and copy
            if not dry_run:
                output_dir = os.path.join(OUTPUT_BASE_PATH, sequence_folder_name)
                os.makedirs(output_dir, exist_ok=True)
                
                import shutil
                shutil.copy2(source_checkpoint_path, checkpoint_path)
                print(f"        COPIED to: {checkpoint_path}")
                already_exists.append(transition_key)
            continue
        
        # Needs processing
        needs_processing.append((transition_key, source_path, sequence_folder_name))
        print(f"NEEDS PROCESSING: {transition_key}")
        print(f"        source: {source_path}")
        print(f"        output: {sequence_folder_name}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Already exists: {len(already_exists)}")
    print(f"Needs processing: {len(needs_processing)}")
    
    if dry_run:
        print("\n[DRY RUN] No actual processing performed")
        return
    
    if not needs_processing:
        print("\nAll checkpoints already exist! Nothing to do.")
        return
    
    # Process missing checkpoints
    print(f"\n{'=' * 70}")
    print(f"PROCESSING {len(needs_processing)} MISSING CHECKPOINTS")
    print(f"{'=' * 70}")
    
    for i, (transition_key, source_path, sequence_folder_name) in enumerate(needs_processing):
        print(f"\n[{i+1}/{len(needs_processing)}] Processing: {transition_key}")
        
        # The log_paths for a single transition is just the source_path
        log_paths = [source_path]
        
        try:
            success = run_deformation_processing_standalone(log_paths, sequence_folder_name)
            if success:
                processing_succeeded.append(transition_key)
            else:
                processing_failed.append(transition_key)
        except Exception as e:
            print(f"ERROR processing {transition_key}: {e}")
            processing_failed.append(transition_key)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Already existed: {len(already_exists)}")
    print(f"Successfully processed: {len(processing_succeeded)}")
    print(f"Failed to process: {len(processing_failed)}")
    
    if processing_failed:
        print("\nFailed transitions:")
        for key in processing_failed:
            print(f"  - {key}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Precompute combined_checkpoints.pkl for all transitions')
    parser.add_argument('--dry-run', action='store_true', help='Only show what would be done, without processing')
    
    args = parser.parse_args()
    
    precompute_all_checkpoints(dry_run=args.dry_run)
