"""
Debug script for run_deformation_processing function
Extracted from h99-UI-mayascipt-V1.py for standalone debugging

Purpose: Debug why the first action appears to be 0 (very little deformation from frame 14 to 15)
"""

import numpy as np
import re
import os
import math
import pickle
from paths import results_path

def run_deformation_processing(phonemes_list, log_paths_list, sequence_folder_name):
    """
    Runs the full deformation parameter processing pipeline.
    """
    print("=" * 80)
    print("--- Starting Deformation Processing ---")
    print(f"Phonemes: {phonemes_list}")
    print(f"Log Paths: {log_paths_list}")
    print(f"Output Folder: {sequence_folder_name}")
    print("=" * 80)

    # --- Your script starts here ---

    overlap_ratio = 0 # 0.1

    # Configuration for the 3 separate paths
    # (THIS IS NOW AN ARGUMENT)
    log_paths = log_paths_list

    # 1. Separate the folder name
    base_path =str(results_path / "sequence_combined_deformation_parameters_hub")
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

    # For "You've not eaten it yet", using natural female speech timing
    if len(phonemes) == 15:  # Specific case for "You've not eaten it yet"
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

        if len(original_durations) < 2:
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

    time_reduced_percentage = (timestamps[-1] - adjusted_timestamps[-1])/timestamps[-1] if timestamps[-1] > 0 else 0
    num_checkpoints = 36
    updated_checkpoints_num = math.floor(num_checkpoints * (1 - time_reduced_percentage))
    num_checkpoints = updated_checkpoints_num

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
    print("\n" + "=" * 80)
    print("LOADING ACTION LISTS")
    print("=" * 80)
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
                print(f"Successfully loaded phase {i+1} from {log_path}")
                print(f"  Number of actions: {len(action_lists[-1])}")

                # DEBUG: Check the first action in this phase
                if len(action_lists[-1]) > 0:
                    first_action = action_lists[-1][0]
                    non_zero_count = sum(1 for x in first_action if x != 0)
                    action_sum = sum(abs(x) for x in first_action)
                    print(f"  [DEBUG] First action in phase {i+1}:")
                    print(f"    Dimension: {len(first_action)}")
                    print(f"    Non-zero elements: {non_zero_count}/{len(first_action)}")
                    print(f"    Sum of absolute values: {action_sum}")
                    print(f"    First 10 elements: {first_action[:10]}")

        except FileNotFoundError:
            print(f"Warning: Could not find action list at {file_path}")
            action_lists.append([])
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            action_lists.append([])

    action_lists_old = action_lists.copy()

    # Apply overlap between phases
    def modify_phase_boundaries(phase_boundaries, front_phase, back_phase):
        phase_length = phase_boundaries[back_phase] - phase_boundaries[front_phase]
        co_expression_steps = int(phase_length * overlap_ratio)

        for i in range(back_phase, len(phase_boundaries)):
            phase_boundaries[i] -= co_expression_steps

        return phase_boundaries, co_expression_steps

    def modify_action_lists(action_lists, front_phase, back_phase, co_expression_steps):
        flat_actions = []
        for phase in action_lists:
            flat_actions.extend(phase)

        front_boundary = phase_boundaries[front_phase]

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

        for i in range(len(phase_boundaries)-1):
            if i == front_phase:
                phase_length = (phase_boundaries[i+1] - phase_boundaries[i]) - co_expression_steps
            else:
                phase_length = (phase_boundaries[i+1] - phase_boundaries[i])

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
            action_lists_current, front_phase, back_phase, co_expression_steps
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
    if total_weights == 0: total_weights = 1

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
    checkpoints = []
    for i, (start, end, _) in enumerate(phase_ranges):
        phase_length = end - start
        if phase_length <= 0:
            continue

        # Always include start and end
        phase_checkpoints = [start, end]

        # Add intermediate points if we have enough checkpoints allocated
        if phase_counts[i] > 2:
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
            phase_checkpoints = [end]

        checkpoints.extend(phase_checkpoints)
        print(f"Phase {i+1} ({phonemes[i]} to {phonemes[i+1]}): checkpoints at {phase_checkpoints[:5]}{'...' if len(phase_checkpoints) > 5 else ''}")

    # Combine with must-include points and sort
    checkpoints = sorted(set(checkpoints + must_include_indices))

    # Ensure the LAST checkpoint is the final frame
    final_phase_end = phase_boundaries[-1] if phase_boundaries else 0
    if final_phase_end not in checkpoints:
        checkpoints.append(final_phase_end)
        checkpoints = sorted(set(checkpoints))
        print(f"[FORCED] Added final checkpoint at {final_phase_end} to ensure final phoneme is reached")

    # Final adjustment to ensure exactly num_checkpoints
    while len(checkpoints) > num_checkpoints:
        if len(checkpoints) <= 2:
            break

        distances = []
        for i in range(1, len(checkpoints)-1):
            distances.append((i, checkpoints[i+1] - checkpoints[i-1]))
        if distances:
            min_dist_idx = min(distances, key=lambda x: x[1])[0]
            if checkpoints[min_dist_idx] not in phase_boundaries:
                checkpoints.pop(min_dist_idx)
            else:
                for idx, _ in sorted(distances, key=lambda x: x[1]):
                    if checkpoints[idx] not in phase_boundaries:
                        checkpoints.pop(idx)
                        break
                else:
                    checkpoints.pop(min_dist_idx)
        else:
            checkpoints.pop(-2)

    while len(checkpoints) < num_checkpoints:
        max_gap = 0
        insert_idx = 1
        for i in range(1, len(checkpoints)):
            gap = checkpoints[i] - checkpoints[i-1]
            if gap > max_gap:
                max_gap = gap
                insert_idx = i
        if max_gap > 1:
            new_point = checkpoints[insert_idx-1] + max_gap//2
            checkpoints.insert(insert_idx, new_point)
        else:
            break
        checkpoints = sorted(checkpoints)

    print("\nOptimized checkpoints:", checkpoints)
    print(f"First checkpoint: {checkpoints[0]} (should be 0 - neutral start)")
    print(f"Last checkpoint: {checkpoints[-1]} (should be {phase_boundaries[-1]} - final phoneme)")

    print("\nCount per phase:")
    for i, (start, end, _) in enumerate(phase_ranges):
        count = sum(1 for x in checkpoints if start <= x <= end)
        has_start = start in checkpoints
        has_end = end in checkpoints
        print(f"Phase {i+1} ({phonemes[i]} → {phonemes[i+1]}): {count} checkpoints [start: {'✓' if has_start else '✗'}, end: {'✓' if has_end else '✗'}]")
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
    valid_checkpoints = [cp for cp in checkpoints if cp < total_length_all or (cp == phase_boundaries[-1] and cp <= total_length_all)]
    missing_count = len(checkpoints) - len(valid_checkpoints)

    if missing_count > 0:
        print(f"Found {missing_count} checkpoints beyond available frames (max: {total_length_all - 1})")

        while len(valid_checkpoints) < num_checkpoints:
            max_gap = 0
            insert_idx = 1
            for i in range(1, len(valid_checkpoints)):
                gap = valid_checkpoints[i] - valid_checkpoints[i-1]
                if gap > max_gap:
                    max_gap = gap
                    insert_idx = i

            if max_gap > 1:
                new_point = valid_checkpoints[insert_idx-1] + max_gap//2
                valid_checkpoints.insert(insert_idx, new_point)
                print(f"Adding replacement checkpoint at {new_point}")
            else:
                new_point = valid_checkpoints[-1] - 1 if len(valid_checkpoints) > 0 else 0
                if new_point < 0: new_point = 0
                valid_checkpoints.insert(len(valid_checkpoints)-1, new_point)
                print(f"Adding replacement checkpoint at {new_point} (near end)")

            valid_checkpoints = sorted(list(set(valid_checkpoints)))

    checkpoint_indices = sorted(list(set(valid_checkpoints)))

    while len(checkpoint_indices) < num_checkpoints:
        checkpoint_indices.append(checkpoint_indices[-1] if checkpoint_indices else 0)
    while len(checkpoint_indices) > num_checkpoints:
        checkpoint_indices.pop()

    print(f"\nFinal checkpoint indices ({len(checkpoint_indices)} points):")
    print(checkpoint_indices)

    # Initialize parameters
    self_n = 300 * 3
    joint_rotate_num = 1
    joint_ooc_l_num = 300
    joint_ooc_r_num = 300
    joint_frontalis_num = 300

    correct_dimension = self_n + 3 * joint_rotate_num + 3 * joint_ooc_l_num + 3 * joint_ooc_r_num + 3 * joint_frontalis_num

    # Dynamically determine the correct dimension size
    if action_lists and action_lists[0]:
        actual_dimension = len(action_lists[0][0])
        print(f"\nDetected actual dimension size: {actual_dimension}")
        print(f"Expected dimension: {correct_dimension}")

        if actual_dimension != correct_dimension:
            print(f"ADJUSTING: Using actual dimension {actual_dimension} instead of calculated {correct_dimension}")
            correct_dimension = actual_dimension
    else:
        print(f"Using default dimension: {correct_dimension}")

    def accumulate_checkpoints(action_lists, checkpoint_indices):
        """
        Accumulates actions and captures checkpoints.
        Added extensive debugging to investigate the first action issue.
        """
        delta_b0_conca = [0] * correct_dimension
        checkpoints = []
        global_index = 0
        found_indices = set()

        print("\n" + "=" * 80)
        print("CHECKPOINT ACCUMULATION WITH DEBUGGING")
        print("=" * 80)
        print(f"Using dimension size: {correct_dimension}")

        total_frames = sum(len(lst) for lst in action_lists)
        print(f"Total frames available: {total_frames}")

        # DEBUG: Inspect first 5 actions across all phases
        print("\n" + "-" * 80)
        print("DEBUGGING: Inspecting first 5 actions from each phase")
        print("-" * 80)
        for phase_idx, action_list in enumerate(action_lists):
            if not action_list:
                print(f"Phase {phase_idx+1}: EMPTY")
                continue

            print(f"\nPhase {phase_idx+1}: {len(action_list)} actions total")
            for action_idx in range(min(5, len(action_list))):
                action = action_list[action_idx]
                non_zero_count = sum(1 for x in action if x != 0)
                action_sum = sum(abs(x) for x in action)
                max_abs_val = max(abs(x) for x in action) if len(action) > 0 else 0

                print(f"  Action {action_idx}:")
                print(f"    Non-zero: {non_zero_count}/{len(action)}")
                print(f"    Sum of |values|: {action_sum:.6f}")
                print(f"    Max |value|: {max_abs_val:.6f}")
                if action_idx == 0:
                    print(f"    First 20 elements: {action[:20]}")

        # Adjust last checkpoint if needed
        if checkpoint_indices[-1] == total_frames:
            print(f"\nNote: Last checkpoint index {checkpoint_indices[-1]} equals total frames")
            checkpoint_indices[-1] = total_frames - 1
            print(f"Adjusted last checkpoint to index {checkpoint_indices[-1]}")

        print(f"\nCheckpoint indices to capture: {checkpoint_indices}")

        checkpoint_data = []

        print("\n" + "-" * 80)
        print("ACCUMULATING ACTIONS")
        print("-" * 80)

        for phase_idx, action_list in enumerate(action_lists):
            if not action_list:
                continue

            for action_idx, action in enumerate(action_list):
                # Store previous state for debugging
                prev_delta = delta_b0_conca.copy()
                prev_non_zero = sum(1 for x in prev_delta if x != 0)

                # Update the accumulated state
                delta_b0_conca = [delta_b0_conca[j] + action[j] for j in range(len(delta_b0_conca))]

                # Check if current global index matches any checkpoint
                if global_index in checkpoint_indices:
                    non_zero_count = sum(1 for x in delta_b0_conca if x != 0)
                    delta_sum = sum(abs(x) for x in delta_b0_conca)
                    action_non_zero = sum(1 for x in action if x != 0)
                    action_sum = sum(abs(x) for x in action)

                    checkpoint_data.append({
                        'global_index': global_index,
                        'phase': phase_idx + 1,
                        'action_idx': action_idx,
                        'non_zero_count': non_zero_count,
                        'delta_sum': delta_sum,
                        'action_non_zero': action_non_zero,
                        'action_sum': action_sum,
                        'prev_non_zero': prev_non_zero
                    })

                    checkpoints.append((global_index, delta_b0_conca.copy()))
                    found_indices.add(global_index)

                    print(f"\n*** CHECKPOINT CAPTURED at global_index={global_index} ***")
                    print(f"    Phase {phase_idx+1}, Action {action_idx}")
                    print(f"    Accumulated state: {non_zero_count}/{len(delta_b0_conca)} non-zero, sum=|values|={delta_sum:.6f}")
                    print(f"    Current action: {action_non_zero}/{len(action)} non-zero, sum=|values|={action_sum:.6f}")
                    print(f"    Previous state: {prev_non_zero} non-zero elements")

                    # Special debugging for the first checkpoint (which corresponds to frame 14->15 transition)
                    if global_index == checkpoint_indices[0]:
                        print(f"\n{'!' * 80}")
                        print(f"!!! FIRST CHECKPOINT - THIS IS THE FRAME 14->15 TRANSITION !!!")
                        print(f"{'!' * 80}")
                        print(f"    This checkpoint should show REAL deformation")
                        print(f"    Current action sample (first 30 elements): {action[:30]}")
                        print(f"    Accumulated delta sample (first 30 elements): {delta_b0_conca[:30]}")
                        if action_sum < 0.001:
                            print(f"    *** WARNING: Action sum is very small ({action_sum}) - THIS IS THE PROBLEM! ***")
                        else:
                            print(f"    Action sum looks reasonable: {action_sum}")

                global_index += 1

        # Print checkpoint data verification
        print("\n" + "=" * 80)
        print("CHECKPOINT DATA VERIFICATION")
        print("=" * 80)
        for i, data in enumerate(checkpoint_data):
            print(f"\nCheckpoint {i}:")
            print(f"  Global Index: {data['global_index']}")
            print(f"  Phase {data['phase']}, Action {data['action_idx']}")
            print(f"  Accumulated: {data['non_zero_count']} non-zero, sum={data['delta_sum']:.6f}")
            print(f"  Action: {data['action_non_zero']} non-zero, sum={data['action_sum']:.6f}")

        # Final verification
        missing = set(checkpoint_indices) - found_indices
        if missing:
            print(f"\nWarning: Missing {len(missing)} checkpoints: {sorted(missing)}")
            print(f"Maximum available index was {global_index-1}")

        print(f"\nSuccessfully captured {len(checkpoints)}/{len(checkpoint_indices)} checkpoints")
        return checkpoints

    # Process all phases
    combined_checkpoints = accumulate_checkpoints(action_lists, checkpoint_indices)

    # Save results
    output_path = os.path.join(log_path_output, "combined_checkpoints.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump([cp[1] for cp in combined_checkpoints], f)

    print(f"\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Final checkpoint indices captured:")
    for idx, _ in combined_checkpoints:
        print(f"  {idx}")

    print(f"\nResults saved to: {output_path}")

    # Verification
    try:
        with open(output_path, "rb") as f:
            loaded_checkpoints = pickle.load(f)
        print(f"Verification: Loaded {len(loaded_checkpoints)} checkpoints")
    except Exception as e:
        print(f"Error verifying output: {e}")

    print("\n" + "=" * 80)
    print("DEFORMATION PROCESSING COMPLETED")
    print("=" * 80)


# Example usage for debugging
if __name__ == "__main__":
    # Example configuration - modify these for your specific case
    phonemes_list = ["phoneme V", "phoneme IY"]
    log_paths_list = [
        str(results_path / "job_251207-034523" / "251207-034531"),
        # r"C:/keyLabLocal_D/a3-stochastic-V1-backup-and-results/results/DSAC_V2_m1_vertexMove_Va003/job_ANOTHER_PATH/TIMESTAMP"
    ]
    sequence_folder_name = "debug_sequence_test"

    print("=" * 80)
    print("DEBUG SCRIPT FOR DEFORMATION PROCESSING")
    print("=" * 80)
    print("\nTo use this script:")
    print("1. Update the phonemes_list variable")
    print("2. Update the log_paths_list variable with your actual log paths")
    print("3. Update the sequence_folder_name variable")
    print("4. Run the script: python h97-debug-deformation-processing.py")
    print("\nThe script will print detailed debugging information about:")
    print("- Action list loading")
    print("- First actions in each phase")
    print("- Checkpoint accumulation")
    print("- The first checkpoint (frame 14->15 transition)")
    print("=" * 80)

    # Uncomment the line below to run the function
    run_deformation_processing(phonemes_list, log_paths_list, sequence_folder_name)
