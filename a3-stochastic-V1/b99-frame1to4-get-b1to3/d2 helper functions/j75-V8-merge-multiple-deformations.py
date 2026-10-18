import numpy as np
import re
import os
import math
import pickle
from paths import results_path

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
                match = re.search(r'done at No. (\d+) step', content)
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
    
    total_frames = sum(len(lst) for lst in action_lists)
    if not checkpoint_indices:
        print("Warning: Checkpoint indices list is empty.")
        return []

    # Adjust last checkpoint if it's out of bounds
    if checkpoint_indices and checkpoint_indices[-1] >= total_frames:
        checkpoint_indices[-1] = total_frames - 1 if total_frames > 0 else 0
    
    print(f"Starting accumulation. Checkpoint indices: {checkpoint_indices}")

    for phase_idx, action_list in enumerate(action_lists):
        if not action_list:
            continue
            
        for action_idx, action in enumerate(action_list):
            # If the current step is a checkpoint, save the state *before* this action is applied.
            if global_index in checkpoint_indices and global_index not in found_indices:
                checkpoints.append(delta_b0_conca.copy())
                found_indices.add(global_index)
                print(f"  - Checkpoint captured at step {global_index}")

            # Apply the action to get the state for the *next* step
            delta_b0_conca = [delta_b0_conca[j] + action[j] for j in range(len(delta_b0_conca))]
            global_index += 1
    
    # After the loop, check if the very last state needs to be captured.
    # This handles the case where a checkpoint is at the total number of frames.
    if global_index in checkpoint_indices and global_index not in found_indices:
        checkpoints.append(delta_b0_conca.copy())
        found_indices.add(global_index)
        print(f"  - Checkpoint captured at final step {global_index}")

    missing = set(checkpoint_indices) - found_indices
    if missing:
        print(f"Warning: Missing {len(missing)} checkpoints: {sorted(missing)}")
    
    print(f"Accumulation complete. Captured {len(checkpoints)} checkpoints.")
    return checkpoints

def process_single_deformation(config, num_checkpoints=36, overlap_ratio=0.0):
    """
    Processes a single deformation sequence, from loading data to generating accumulated checkpoints.
    """
    log_path = config["log_path"]
    phonemes = config["phonemes"]
    
    print(f"\n--- Processing deformation for {phonemes} from {log_path} ---")

    total_steps_convergence = get_total_steps_convergence(log_path, phonemes)
    phase_boundaries = np.cumsum(total_steps_convergence).tolist()

    number_of_timestamps = len(phonemes)
    timestamps = np.linspace(0, 1.68, number_of_timestamps).tolist()

    phase_indices_to_load = [0]
    if len(phonemes) >= 4:
        phase_indices_to_load.extend([6, 7])

    total_duration = timestamps[-1] if timestamps else 0
    phase_durations = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    phase_weights = [duration/total_duration for duration in phase_durations] if total_duration > 0 else []

    must_include_indices = phase_boundaries.copy()
    
    remaining_checkpoints = num_checkpoints - len(must_include_indices)
    
    if sum(phase_weights) == 0:
        phase_counts = [0] * len(phase_weights)
    else:
        total_weights = sum(phase_weights)
        phase_counts = [max(0, int(round(weight * remaining_checkpoints / total_weights))) for weight in phase_weights]

    while sum(phase_counts) != remaining_checkpoints:
        if sum(phase_counts) < remaining_checkpoints:
            if not phase_weights: break
            total_weights = sum(phase_weights)
            phase_idx = np.argmax([(weight * remaining_checkpoints / total_weights) - count for weight, count in zip(phase_weights, phase_counts)])
            phase_counts[phase_idx] += 1
        else:
            phase_idx = np.argmin([(weight * remaining_checkpoints / total_weights) - count for weight, count in zip(phase_weights, phase_counts)])
            phase_counts[phase_idx] -= 1

    checkpoints = []
    buffer_ratio = 0.1
    phase_ranges = [(phase_boundaries[i], phase_boundaries[i+1]) for i in range(len(phonemes)-1)]

    for i, (start, end) in enumerate(phase_ranges):
        if i >= len(phase_counts) or phase_counts[i] == 0: continue
        phase_length = end - start
        if phase_length <= 0: continue
        
        buffer = int(buffer_ratio * phase_length)
        valid_start = start + buffer
        valid_end = end - buffer

        if valid_start >= valid_end:
            valid_start = start
            valid_end = end -1
        
        phase_indices = np.linspace(valid_start, valid_end, num=phase_counts[i], dtype=int).tolist()
        checkpoints.extend(phase_indices)

    checkpoints = sorted(set(checkpoints + must_include_indices))

    while len(checkpoints) > num_checkpoints:
        checkpoints.pop(1)
    while len(checkpoints) < num_checkpoints:
        if len(checkpoints) > 1:
            checkpoints.insert(1, checkpoints[0] + (checkpoints[1] - checkpoints[0]) // 2)
        elif len(checkpoints) == 1:
            checkpoints.append(checkpoints[0] + 1)
        else: # If checkpoints is empty, add from must_include
            if must_include_indices:
                checkpoints.append(must_include_indices[0])
            else:
                checkpoints.append(0)


    action_lists = []
    for i in phase_indices_to_load:
        phase_num = i + 1
        file_path = os.path.join(log_path, f"final_epoch_action_list{'_phase' + str(phase_num) if phase_num > 1 else ''}.pkl")
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                action_lists.append(data['final_epoch_action_list'])
                print(f"Successfully loaded Phase {phase_num} with {len(data['final_epoch_action_list'])} actions")
        except FileNotFoundError:
            print(f"Warning: Could not find action list for Phase {phase_num} at {file_path}")
            action_lists.append([])

    self_n = 300 * 3
    joint_rotate_num = 1
    joint_ooc_l_num = 300
    joint_ooc_r_num = 300
    joint_frontalis_num = 300

    accumulated = accumulate_checkpoints(action_lists, checkpoints, self_n, joint_rotate_num, joint_ooc_l_num, joint_ooc_r_num, joint_frontalis_num)
    
    print(f"--- Finished processing for {phonemes}. Generated {len(accumulated)} checkpoints. ---")
    return accumulated

def main():
    # --- CONFIGURATION ---
    # TODO: Replace the placeholder paths with the correct paths for each deformation.
    deformation_configs = [
        {
            "name": "AU1",
            "log_path":str(results_path / "job_251112-114911/251112-114930"), 
            # Placeholder for AU1
            "phonemes": ["Neutral", "AU1"],
        },
        {
            "name": "AU2",
            "log_path":str(results_path / "job_251113-113629/251113-113645"), 
            # This is the path for AU2
            "phonemes": ["Neutral", "AU2"],
        },
        {
            "name": "AU25",
            "log_path":str(results_path / "job_251120-135844/251120-135852"),
            "phonemes": ["Neutral", "AU25"],
        },
    ]
    
    num_checkpoints = 36
    output_filename = "combined_checkpoints.pkl"
    
    # --- Define Output Path ---
    base_output_dir =str(results_path / "merged_deformation_parameters_hub")
    subfolder_name = "_".join(config["name"] for config in deformation_configs)
    output_dir = os.path.join(base_output_dir, subfolder_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    all_accumulated_checkpoints = []
    for config in deformation_configs:
        accumulated = process_single_deformation(config, num_checkpoints)
        if accumulated:
            all_accumulated_checkpoints.append(accumulated)

    if not all_accumulated_checkpoints:
        print("No checkpoints were generated. Exiting.")
        return

    first_len = len(all_accumulated_checkpoints[0])
    if not all(len(x) == first_len for x in all_accumulated_checkpoints):
        print("Error: Not all deformation sequences produced the same number of checkpoints.")
        for i, config in enumerate(deformation_configs):
            print(f"  - {config['name']}: {len(all_accumulated_checkpoints[i])} checkpoints")
        return

    # --- DEBUGGING: Inspect Frontalis Deformation ---
    print("\n--- DEBUGGING: Inspecting Frontalis Deformation ---")
    frontalis_start_index = (300 * 3) + (3 * 1) + (3 * 300) + (3 * 300)

    for i, config in enumerate(deformation_configs):
        name = config["name"]
        if not all_accumulated_checkpoints[i]:
            print(f"\n--- {name} has no checkpoints. Skipping debug output. ---")
            continue
        
        last_checkpoint_vector = all_accumulated_checkpoints[i][-1]
        frontalis_part = last_checkpoint_vector[frontalis_start_index:]
        
        print(f"\n--- {name} Final Checkpoint ---")
        if frontalis_part:
            print(f"  Max frontalis value: {max(frontalis_part):.4f}")
            print(f"  Min frontalis value: {min(frontalis_part):.4f}")
            print(f"  Avg frontalis value: {np.mean(frontalis_part):.4f}")
        else:
            print("  Frontalis part is empty.")

    print(f"\nCombining {len(all_accumulated_checkpoints)} deformation(s) with {first_len} checkpoints each.")
    
    sum_checkpoints = [v.copy() for v in all_accumulated_checkpoints[0]]

    for i in range(1, len(all_accumulated_checkpoints)):
        for j in range(first_len):
            sum_checkpoints[j] = [sum_checkpoints[j][k] + all_accumulated_checkpoints[i][j][k] for k in range(len(sum_checkpoints[j]))]

    # --- DEBUGGING: Inspect Summed Frontalis ---
    print("\n--- Summed Final Checkpoint ---")
    if sum_checkpoints:
        summed_frontalis_part = sum_checkpoints[-1][frontalis_start_index:]
        if summed_frontalis_part:
            print(f"  Max summed frontalis value: {max(summed_frontalis_part):.4f}")
            print(f"  Min summed frontalis value: {min(summed_frontalis_part):.4f}")
            print(f"  Avg summed frontalis value: {np.mean(summed_frontalis_part):.4f}")
        else:
            print("  Summed frontalis part is empty.")
    else:
        print("  Summed checkpoints list is empty.")
    print("--- END DEBUGGING ---")

    with open(output_path, 'wb') as f:
        pickle.dump(sum_checkpoints, f)

    print(f"\nSuccessfully combined checkpoints and saved to: {output_path}")

    try:
        with open(output_path, "rb") as f:
            loaded_checkpoints = pickle.load(f)
        print(f"Verification: Loaded {len(loaded_checkpoints)} combined checkpoints from file.")
        if loaded_checkpoints:
            print(f"Dimension of first checkpoint vector: {len(loaded_checkpoints[0])}")
            # Verify the first checkpoint is all zeros
            first_is_zero = all(v == 0 for v in loaded_checkpoints[0])
            print(f"Verification: First checkpoint is zero vector: {first_is_zero}")
    except Exception as e:
        print(f"Error verifying output file: {e}")

if __name__ == "__main__":
    main()