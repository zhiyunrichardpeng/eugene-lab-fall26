import os
import re
import math
import pickle
import numpy as np

def run_deformation_processing(phonemes_list, log_paths_list, output_path_pkl):
    """
    Runs the full deformation parameter processing pipeline.
    This function is adapted from 'b99-frame1to4-get-b1to3\d4 interactive interface of animation\h99-UI-mayascipt-V1.py'
    """
    print("--- Starting Deformation Processing ---")
    print(f"Phonemes: {phonemes_list}")
    print(f"Log Paths: {log_paths_list}")
    print(f"Output File: {output_path_pkl}")

    # --- Your script starts here ---
    
    overlap_ratio = 0.1

    # Configuration for the 3 separate paths
    # (THIS IS NOW AN ARGUMENT)
    log_paths = log_paths_list

    # 2. Create the full path
    log_path_output = os.path.dirname(output_path_pkl)

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
                    match = re.search(r'done at No. (\d+) step', content)
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

    # generate equal interval's timestemps.
    if len(phonemes) <= 1:
        print("Error: Need at least 2 phonemes (a start and an end) to generate timestamps.")
        return # Exit if not enough phonemes
        
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

    time_reduced_percentage = (timestamps[-1] - adjusted_timestamps[-1])/timestamps[-1] if timestamps[-1] > 0 else 0
    num_checkpoints = 36
    updated_checkpoints_num = math.floor(num_checkpoints * (1 - time_reduced_percentage))
    num_checkpoints = updated_checkpoints_num

    buffer_ratio = 0.1

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
    checkpoints = []
    for i, (start, end, _) in enumerate(phase_ranges):
        if phase_counts[i] == 0:
            continue
            
        phase_length = end - start
        if phase_length <= 0: continue # Skip empty phases

        buffer = int(buffer_ratio * phase_length)
        valid_start = start + buffer
        valid_end = end - buffer
        
        if valid_start >= valid_end: # Handle very short phases
            valid_start = start
            valid_end = end - 1
            if valid_start >= valid_end:
                phase_indices = [start] * phase_counts[i] # Just grab the start
                checkpoints.extend(phase_indices)
                continue

        phase_indices = np.linspace(
            valid_start, 
            valid_end, 
            num=phase_counts[i], 
            dtype=int
        ).tolist()
        
        checkpoints.extend(phase_indices)

    # Combine with must-include points and sort
    checkpoints = sorted(set(checkpoints + must_include_indices))

    # Final adjustment to ensure exactly num_checkpoints
    while len(checkpoints) > num_checkpoints:
        distances = []
        for i in range(1, len(checkpoints)-1):
            distances.append((i, checkpoints[i+1] - checkpoints[i-1]))
        if distances:
            min_dist_idx = min(distances, key=lambda x: x[1])[0]
            checkpoints.pop(min_dist_idx)
        else:
            checkpoints.pop()

    while len(checkpoints) < num_checkpoints:
        max_gap = 0
        insert_idx = 1
        for i in range(1, len(checkpoints)):
            gap = checkpoints[i] - checkpoints[i-1]
            if gap > max_gap:
                max_gap = gap
                insert_idx = i
        if max_gap > 0: # Only add if gap is larger than 1
            new_point = checkpoints[insert_idx-1] + max_gap//2
            checkpoints.insert(insert_idx, new_point)
        else: # If no gaps, just add to the end
            checkpoints.append(checkpoints[-1] + 1 if checkpoints else 0)
        checkpoints = sorted(checkpoints)

    print("\nOptimized checkpoints:", checkpoints)
    print("Count per phase:")
    for i, (start, end, _) in enumerate(phase_ranges):
        count = sum(1 for x in checkpoints if start <= x < end)
        print(f"Phase {i+1}: {count} checkpoints")
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
    valid_checkpoints = [cp for cp in checkpoints if cp < total_length_all]
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

    if action_lists and action_lists[0]:
        actual_dimension = len(action_lists[0][0])
        print(f"\nDetected actual dimension size: {actual_dimension}")
        print(f"Current self_n + 3*joint_rotate_num = {self_n} + {3*joint_rotate_num} = {self_n + 3*joint_rotate_num}")
        
        if actual_dimension != (self_n + 3 * joint_rotate_num):
            print(f"ADJUSTING: Using actual dimension {actual_dimension} instead of calculated {self_n + 3 * joint_rotate_num}")
            correct_dimension = actual_dimension
        else:
            correct_dimension = self_n + 3 * joint_rotate_num
    else:
        correct_dimension = self_n + 3 * joint_rotate_num
        print(f"Using default dimension: {correct_dimension}")

    def accumulate_checkpoints(action_lists, checkpoint_indices):
        delta_b0_conca = [0] * correct_dimension
        checkpoints = []
        global_index = 0
        found_indices = set()
        
        print(f"\nStarting checkpoint accumulation...")
        print(f"Using dimension size: {correct_dimension}")
        
        total_frames = sum(len(lst) for lst in action_lists)
        print(f"Total frames available: {total_frames}")
        
        print(f"\nChecking first 3 actions across all phases:")
        action_count = 0
        for phase_idx, action_list in enumerate(action_lists):
            if not action_list:
                continue
            for action_idx, action in enumerate(action_list[:3]):  # Check first 3 actions
                non_zero_count = sum(1 for x in action if x != 0)
                print(f"Phase {phase_idx+1}, Action {action_idx}: {non_zero_count}/{len(action)} non-zero elements")
                action_count += 1
                if action_count >= 9:
                    break
            if action_count >= 9:
                break
        
        if checkpoint_indices and checkpoint_indices[-1] == total_frames:
            print(f"Note: Last checkpoint index {checkpoint_indices[-1]} equals total frames")
            checkpoint_indices[-1] = total_frames - 1
            print(f"Adjusted last checkpoint to index {checkpoint_indices[-1]}")
        
        print(f"Checkpoint indices: {checkpoint_indices}")
        
        checkpoint_data = []
        
        for phase_idx, action_list in enumerate(action_lists):
            if not action_list:
                continue
                
            for action_idx, action in enumerate(action_list):
                delta_b0_conca = [delta_b0_conca[j] + action[j] for j in range(len(delta_b0_conca))]
                
                if global_index in checkpoint_indices:
                    non_zero_count = sum(1 for x in delta_b0_conca if x != 0)
                    checkpoint_data.append({
                        'index': global_index,
                        'non_zero_count': non_zero_count,
                        'data_sample': delta_b0_conca[:5]
                    })
                    
                    checkpoints.append((global_index, delta_b0_conca.copy()))
                    found_indices.add(global_index)
                    print(f"Checkpoint at {global_index} (Phase {phase_idx+1}, Action {action_idx}) - {non_zero_count}/{len(delta_b0_conca)} non-zero values")
                
                global_index += 1
        
        print(f"\n=== CHECKPOINT DATA VERIFICATION ===")
        for i, data in enumerate(checkpoint_data[:5]):
            print(f"Checkpoint {i}: Index {data['index']}, {data['non_zero_count']} non-zero, sample: {data['data_sample']}")
        
        missing = set(checkpoint_indices) - found_indices
        if missing:
            print(f"\nWarning: Missing {len(missing)} checkpoints: {sorted(missing)}")
            print(f"Maximum available index was {global_index-1}")
        
        print(f"\nSuccessfully captured {len(checkpoints)}/{len(checkpoint_indices)} checkpoints")
        return checkpoints

    combined_checkpoints = accumulate_checkpoints(action_lists, checkpoint_indices)

    print(f"\n=== FINAL VERIFICATION ===")
    print(f"Total checkpoints captured: {len(combined_checkpoints)}")

    if combined_checkpoints:
        all_zero_checkpoints = 0
        for idx, (checkpoint_idx, checkpoint_data) in enumerate(combined_checkpoints):
            non_zero_count = sum(1 for x in checkpoint_data if x != 0)
            if non_zero_count == 0:
                all_zero_checkpoints += 1
                if all_zero_checkpoints <= 3:
                    print(f"WARNING: Checkpoint {idx} (index {checkpoint_idx}) contains ALL ZEROS!")
        
        if all_zero_checkpoints > 0:
            print(f"TOTAL: {all_zero_checkpoints}/{len(combined_checkpoints)} checkpoints contain all zeros")
        else:
            print("SUCCESS: All checkpoints contain non-zero data")
            
        print(f"\nFirst checkpoint sample (first 10 elements): {combined_checkpoints[0][1][:10]}")
    else:
        print("ERROR: No checkpoints were captured!")


    # Save results
    with open(output_path_pkl, 'wb') as f:
        pickle.dump([cp[1] for cp in combined_checkpoints], f)

    print(f"\nFinal checkpoint indices captured:")
    for idx, _ in combined_checkpoints:
        print(f"- {idx}")

    print(f"\nResults saved to: {output_path_pkl}")

    # Verification
    try:
        with open(output_path_pkl, "rb") as f:
            loaded_checkpoints = pickle.load(f)
        print(f"Verification: Loaded {len(loaded_checkpoints)} checkpoints")
        print("len of combined_checkpoints", len(loaded_checkpoints))
    except Exception as e:
        print(f"Error verifying output: {e}")

    print("\nCode execution completed successfully!")
