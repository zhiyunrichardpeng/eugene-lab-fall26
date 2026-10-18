
import numpy as np
import re
import os
import math

# import numpy as np
import pickle
from paths import results_path
# import os
log_path =str(results_path / "job_250623-144458/250623-144522")
log_path_output =str(results_path / "job_250623-144458_experiment/job_250623-144458_experimentV5")

overlap_ratio = 0


# def get_total_steps_convergence():
#     """Extracts phase step counts from log files (non-accumulated values)"""
#     log_path =str(results_path / "job_250623-144458/250623-144522")
#     phase_files = [
#         "log0.txt",
#         # "log0_phase2.txt",
#         # "log0_phase3.txt",
#         # "log0_phase4.txt",
#         # "log0_phase5.txt",
#         # "log0_phase6.txt",
#         "log0_phase7.txt",
#         "log0_phase8.txt"
#     ]

#     total_steps_convergence = [0]  # Starting with 0 for the initial phase

phonemes = ["Neutral", "N", "Y", "UW"]

def get_total_steps_convergence(phonemes):
    """Extracts phase step counts from log files for specific phases"""
    log_path =str(results_path / "job_250623-144458/250623-144522")
    
    # Map phoneme transitions to phase files
    phase_files = ["log0.txt"]  # Always include Phase 1
    
    if len(phonemes) > 2:  # If we have more than just Neutral and UW
        # Add Phase 7 and 8 files only if they correspond to phoneme transitions
        if len(phonemes) >= 4:  # For ["Neutral", "N", "Y", "UW"]
            phase_files.extend(["log0_phase7.txt", "log0_phase8.txt"])
    
    total_steps_convergence = [0]  # Starting with 0 for the initial phase

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

    print("total_steps_convergence =", total_steps_convergence)
    return total_steps_convergence

# Get the non-accumulated phase steps
total_steps_convergence = get_total_steps_convergence(phonemes)

# Calculate accumulated phase boundaries
phase_boundaries = np.cumsum(total_steps_convergence).tolist()
phase_boundaries_old = phase_boundaries

# Timestamp data (in seconds)
# timestamps = [0, 0.23, 0.52, 0.67, 0.89, 0.98, 1.14, 1.16, 1.68]
# phonemes = ["Neutral", "N", "AA", "T", "IY", "V", "N", "Y", "UW"]

# timestamps = [0, 0.23, 1.14, 1.16, 1.68]
# Create evenly spaced timestamps
even_timestamps = np.linspace(0, 1.68, 5).tolist()
timestamps = even_timestamps
print("Evenly spaced timestamps:", even_timestamps)

# use neu to N, and N to Y, and see if the Y or UW will seem wrong.

# Phase loading - generalized for any phoneme count
phase_indices_to_load = [0]  # Always load Phase 1
if len(phonemes) >= 4:  # For ["Neutral", "N", "Y", "UW"]
    phase_indices_to_load.extend([6, 7])  # Add Phase 7 and 8 (0-based would be 6 and 7)


def adjust_durations_with_overlap(timestamps, overlap_ratio):
    # Calculate original durations between phonemes
    original_durations = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    
    # Calculate adjusted durations with 40% overlap
    adjusted_durations = []
    for i in range(len(original_durations)-2):
        # Each duration is reduced by 40% of the next duration
        adjusted_duration = original_durations[i] - original_durations[i+1] * overlap_ratio
        adjusted_durations.append(max(0, adjusted_duration))  # Ensure non-negative
    
    # Keep the last duration unchanged
    adjusted_durations.append(original_durations[-2])
    adjusted_durations.append(original_durations[-1])
    
    # Reconstruct absolute timestamps from adjusted durations
    adjusted_timestamps = [timestamps[0]]
    for duration in adjusted_durations:
        adjusted_timestamps.append(adjusted_timestamps[-1] + duration)
    
    # adjusted_timestamps.append(timestamps[-1])
    return adjusted_timestamps, adjusted_durations

# Apply the duration adjustment
adjusted_timestamps, adjusted_durations = adjust_durations_with_overlap(timestamps, overlap_ratio)
print("Original timestamps:", timestamps)
print("Adjusted timestamps:", adjusted_timestamps)

# Verify the changes
print("\nOriginal durations:", [round(timestamps[i+1]-timestamps[i], 3) for i in range(len(timestamps)-1)])
print("Adjusted durations:", [adjusted_durations[i] for i in range(len(adjusted_durations))])


total_duration = timestamps[-1]  # 1.68 seconds

# Calculate phase durations and weights
# phase_durations = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
# phase_weights = [duration/total_duration for duration in phase_durations]



# Calculate new phase durations and weights
adjusted_phase_durations = [adjusted_timestamps[i+1] - adjusted_timestamps[i] for i in range(len(adjusted_timestamps)-1)]
adjusted_phase_weights = [duration/total_duration for duration in adjusted_phase_durations]

phase_durations = adjusted_phase_durations
phase_weights = adjusted_phase_weights

# Must-include indices are exactly the accumulated phase boundaries
must_include_indices = phase_boundaries.copy()
must_include_indices_old = must_include_indices


time_reduced_percentage = (timestamps[-1] - adjusted_timestamps[-1])/timestamps[-1]
num_checkpoints = 36

if overlap_ratio == 0:
    updated_checkpoints_num = num_checkpoints
else:
    updated_checkpoints_num = math.floor(num_checkpoints * (1 - time_reduced_percentage))

num_checkpoints = updated_checkpoints_num

# print("passed the test, now quit. updated_checkpoints_num", updated_checkpoints_num)
# import sys
# sys.exit(0)  # Exit with status code 0  

buffer_ratio = 0.1  # 10% buffer around must-include points

# Phase ranges with weights using accumulated phase boundaries
# phase_ranges = [
#     # (start_idx, end_idx, weight)
#     (phase_boundaries[0], phase_boundaries[1], phase_weights[0]),  # Neutral to N (0-0.23s)
#     (phase_boundaries[1], phase_boundaries[2], phase_weights[1]),  # N to AA (0.23-0.52s)
#     (phase_boundaries[2], phase_boundaries[3], phase_weights[2]),  # AA to T (0.52-0.67s)
#     (phase_boundaries[3], phase_boundaries[4], phase_weights[3]),  # T to IY (0.67-0.89s)
#     (phase_boundaries[4], phase_boundaries[5], phase_weights[4]),  # IY to V (0.89-0.98s)
#     (phase_boundaries[5], phase_boundaries[6], phase_weights[5]),  # V to N (0.98-1.14s)
#     (phase_boundaries[6], phase_boundaries[7], phase_weights[6]),  # N to Y (1.14-1.16s)
#     (phase_boundaries[7], phase_boundaries[8], phase_weights[7])   # Y to UW (1.16-1.68s)
# ]

# Generalized phase ranges calculation
phase_ranges = [
    (phase_boundaries[i], phase_boundaries[i+1], phase_weights[i]) 
    for i in range(len(phonemes)-1)
]

# Print verification
print("\nAccumulated phase_boundaries =", phase_boundaries)
print("\nMust Include Indices:", must_include_indices)
print("\nPhase Ranges:")
for i, (start, end, weight) in enumerate(phase_ranges):
    print(f"Phase {i+1}: {start}-{end} (weight: {weight:.3f})")

# Print phase information for verification
print("Phoneme Transition Durations and Weights:")
for i in range(len(phonemes)-1):
    print(f"{phonemes[i]} to {phonemes[i+1]}: {phase_durations[i]:.2f}s ({phase_weights[i]*100:.1f}%)")

# Calculate checkpoint counts per phase (minus must-include points)
total_weights = sum(weight for _, _, weight in phase_ranges)
remaining_checkpoints = num_checkpoints - len(must_include_indices)

# Initial distribution
phase_counts = [max(1, int(round(weight * remaining_checkpoints / total_weights))) for _, _, weight in phase_ranges]

# Adjust counts to sum exactly to remaining_checkpoints
while sum(phase_counts) != remaining_checkpoints:
    if sum(phase_counts) < remaining_checkpoints:
        # Find phase with largest weight that's underrepresented
        phase_idx = np.argmax([
            (weight * remaining_checkpoints / total_weights) - count 
            for (_, _, weight), count in zip(phase_ranges, phase_counts)
        ])
        phase_counts[phase_idx] += 1
    else:
        # Find phase with smallest weight that's overrepresented
        phase_idx = np.argmin([
            (weight * remaining_checkpoints / total_weights) - count 
            for (_, _, weight), count in zip(phase_ranges, phase_counts)
        ])
        phase_counts[phase_idx] -= 1

# Generate samples with dynamic spacing
checkpoints = []
for i, (start, end, _) in enumerate(phase_ranges):
    if phase_counts[i] == 0:
        continue
        
    # Calculate ideal spacing for this phase
    phase_length = end - start
    dynamic_min_distance = phase_length / (phase_counts[i] + 1)  # +1 for end buffer
    
    # Create sampling range with dynamic buffer
    buffer = int(buffer_ratio * phase_length)
    valid_start = start + buffer
    valid_end = end - buffer
    
    # Evenly sample with phase-appropriate spacing
    phase_indices = np.linspace(
        valid_start, 
        valid_end, 
        num=phase_counts[i], 
        dtype=int
    ).tolist()
    
    # Ensure minimum spacing from must-include points
    phase_indices = [
        x for x in phase_indices 
        if all(abs(x - m) > dynamic_min_distance/3 for m in must_include_indices)  # Relaxed spacing
    ]
    
    # If we lost points, add them back at ideal positions
    if len(phase_indices) < phase_counts[i]:
        ideal_positions = np.linspace(valid_start, valid_end, phase_counts[i]+2)[1:-1]  # Exclude endpoints
        phase_indices = [int(x) for x in ideal_positions]
    
    checkpoints.extend(phase_indices)

# Combine with must-include points and sort
checkpoints = sorted(set(checkpoints + must_include_indices))

# Final adjustment to ensure exactly num_checkpoints
while len(checkpoints) > num_checkpoints:
    # Remove point with smallest distance to neighbors
    distances = []
    for i in range(1, len(checkpoints)-1):
        distances.append((i, checkpoints[i+1] - checkpoints[i-1]))
    if distances:
        min_dist_idx = min(distances, key=lambda x: x[1])[0]
        checkpoints.pop(min_dist_idx)
    else:
        checkpoints.pop()  # fallback

while len(checkpoints) < num_checkpoints:
    # Find largest gap and add a point
    max_gap = 0
    insert_idx = 1
    for i in range(1, len(checkpoints)):
        gap = checkpoints[i] - checkpoints[i-1]
        if gap > max_gap:
            max_gap = gap
            insert_idx = i
    new_point = checkpoints[insert_idx-1] + max_gap//2
    checkpoints.insert(insert_idx, new_point)
    checkpoints = sorted(checkpoints)  # Re-sort to be safe

print("\old nOptimized checkpoints:", checkpoints)
print("\nCount per phase:")
for i, (start, end, _) in enumerate(phase_ranges):
    count = sum(1 for x in checkpoints if start <= x < end)
    print(f"{phonemes[i]} to {phonemes[i+1]}: {count} checkpoints")
print(f"\nTotal checkpoints: {len(checkpoints)} (target: {num_checkpoints})")

checkpoints_old = checkpoints



## second part of the code



# Configuration
base_path =str(results_path / "job_250623-144458/250623-144522")
only_run_phase1_flag = 0  # Set to 1 to only run phase 1

# Initialize parameters
self_n = 300 * 3  # Replace with actual value of `self.n` (see config file.)
joint_rotate_num = 1  # Replace with actual value of `self.joint_rotate_num`
# num_checkpoints = 36  # Total number of checkpoints across all phases

# Define phase information (modify these ranges according to your actual phase boundaries)

# phase_names = ["Phase1", "Phase2", "Phase3", "Phase4", "Phase5", "Phase6", "Phase7", "Phase8"]

phase_names = [f"Phase{i+1}" for i in range(len(phonemes)-1)]

# Load action lists for all phases
# action_lists = []
# for i, phase_name in enumerate(phase_names):
#     if only_run_phase1_flag and i > 0:
#         continue
        
#     if i == 0:
#         file_path = os.path.join(base_path, f"final_epoch_action_list.pkl")
#         r"\final_epoch_action_list.pkl",
#     else:
#         file_path = os.path.join(base_path, f"final_epoch_action_list_phase{i+1}.pkl")
    
#     try:
#         with open(file_path, "rb") as f:
#             data = pickle.load(f)
#             action_lists.append(data[f'final_epoch_action_list'])
#             # print(f"Successfully loaded phase {i} with {len(data[f'final_epoch_action_list'])} actions")
#             print(f"Successfully loaded {phase_name} with {len(data['final_epoch_action_list'])} actions")
#     except FileNotFoundError:
#         if i == 0:  # Phase 1 is required
#             raise
#         print(f"Warning: Could not find action list for phase {i}")
#         action_lists.append([])

action_lists = []
for i in phase_indices_to_load:
    phase_num = i + 1  # Convert to 1-based numbering
    phase_name = f"Phase{phase_num}"
    
    if phase_num == 1:
        file_path = os.path.join(base_path, "final_epoch_action_list.pkl")
    else:
        file_path = os.path.join(base_path, f"final_epoch_action_list_phase{phase_num}.pkl")
    
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            action_lists.append(data['final_epoch_action_list'])
            print(f"Successfully loaded {phase_name} with {len(data['final_epoch_action_list'])} actions")
    except FileNotFoundError:
        if phase_num == 1:
            raise FileNotFoundError(f"Required phase file not found: {file_path}")
        print(f"Warning: Could not find action list for {phase_name}")
        action_lists.append([])

# Verify we have the expected number of action lists
expected_phases = 1  # At least Phase 1
if len(phonemes) >= 4:
    expected_phases = 3  # Phase 1, 7, and 8

if len(action_lists) != expected_phases:
    print(f"Warning: Expected {expected_phases} phases but loaded {len(action_lists)}")

# input poses to overlap:   T, and IY, respecting to phase 3 and 4
# front_phase = 3 # number 3 is phase 3.
# back_phase = 4 # number 4 is phase 3.


def modify_phase_boundaries(phase_boundaries, front_phase, back_phase):
    if overlap_ratio == 0:
        
        co_expression_steps = 0
        return phase_boundaries, co_expression_steps    
    # Calculate overlap steps (e.g., 10% of phase length)
    phase_length = phase_boundaries[back_phase] - phase_boundaries[front_phase]
    co_expression_steps = int(phase_length * overlap_ratio)  # Adjust this as needed
    
    # Reduce subsequent boundaries by overlap steps
    for i in range(back_phase, len(phase_boundaries)):
        phase_boundaries[i] -= co_expression_steps
    
    return phase_boundaries, co_expression_steps


def modify_action_lists(action_lists, front_phase, back_phase, co_expression_steps):

    if overlap_ratio == 0:
        return action_lists
    
    # Flatten all actions
    flat_actions = []
    for phase in action_lists:
        flat_actions.extend(phase)
    
    front_boundary = phase_boundaries[front_phase]
    
    # Split into segments
    l1 = flat_actions[:front_boundary - co_expression_steps]
    l2 = flat_actions[front_boundary - co_expression_steps:front_boundary]
    l3 = flat_actions[front_boundary:front_boundary + co_expression_steps]
    l4 = flat_actions[front_boundary + co_expression_steps:]
    
    # Verify segment lengths
    # print(f"\nSegment lengths:")
    # print(f"l1: {len(l1)}")
    # print(f"l2: {len(l2)} (should be {co_expression_steps})")
    # print(f"l3: {len(l3)} (should be {co_expression_steps})")
    # print(f"l4: {len(l4)}")
    
    # Element-wise addition (blending)
    l2_prime = []
    for a, b in zip(l2, l3):
        blended = [x + y for x, y in zip(a, b)]
        l2_prime.append(blended)
    
    # Combined list (should be shorter by co_expression_steps)
    modified_flat = l1 + l2_prime + l4
    
    # Verify length reduction
    original_length = len(flat_actions)
    new_length = len(modified_flat)
    expected_length = original_length - co_expression_steps
    
    if new_length != expected_length:
        print(f"ERROR: Length mismatch. Expected {expected_length}, got {new_length}")
    
    # Reconstruct phase structure
    action_lists_new = []
    current_idx = 0
    
    for i in range(len(phase_boundaries)-1):
        # Calculate adjusted phase lengths
        if i == front_phase:
            phase_length = (phase_boundaries[i+1] - phase_boundaries[i]) - co_expression_steps
        elif i == back_phase:
            phase_length = (phase_boundaries[i+1] - phase_boundaries[i])
        else:
            phase_length = (phase_boundaries[i+1] - phase_boundaries[i])
        
        phase_actions = modified_flat[current_idx:current_idx + phase_length]
        action_lists_new.append(phase_actions)
        current_idx += len(phase_actions)
    
    return action_lists_new

# action_lists_new = modify_action_lists(action_lists, front_phase, back_phase, co_expression_steps)


action_lists_old = action_lists.copy()



def debug_print_lengths(action_lists, phase_boundaries, message=""):
    total_lengths = [len(lst) for lst in action_lists]
    cumulative_lengths = np.cumsum([0] + total_lengths)
    total_length_all = cumulative_lengths[-1]
    # print(f"\n{message}")
    # print(f"Phase lengths: {total_lengths}")
    # print(f"Phase boundaries: {phase_boundaries}")
    # print(f"Total length: {total_length_all}")
    return total_length_all

# Initialize with the original data
action_lists_current = [lst.copy() for lst in action_lists]  # Deep copy
phase_boundaries_current = phase_boundaries.copy()

# Print initial state
initial_length = debug_print_lengths(action_lists_current, phase_boundaries_current, "Initial state")

# Process phases in descending order
for front_phase in range(1, len(phase_boundaries) - 3):
    back_phase = front_phase + 1
    
    print(f"\n=== Processing overlap between phase {front_phase} and {back_phase} ===")
    
    # Get current lengths before modification
    pre_length = debug_print_lengths(action_lists_current, phase_boundaries_current, "Before modification")
    
    # Modify phase boundaries
    phase_boundaries_new, co_expression_steps = modify_phase_boundaries(
        phase_boundaries_current.copy(), 
        front_phase, 
        back_phase
    )
    
    # Modify action lists
    action_lists_new = modify_action_lists(
        action_lists_current, 
        front_phase, 
        back_phase, 
        co_expression_steps
    )
    
    # Verify the reduction
    post_length = debug_print_lengths(action_lists_new, phase_boundaries_new, "After modification")
    
    expected_reduction = co_expression_steps
    actual_reduction = pre_length - post_length
    
    if actual_reduction != expected_reduction:
        print(f"\nWARNING: Expected reduction of {expected_reduction} frames, got {actual_reduction}")
        print("Possible issues:")
        print("1. modify_phase_boundaries() isn't reducing boundaries correctly")
        print("2. modify_action_lists() isn't removing the overlapped frames")
        print("3. Phase boundary calculations are incorrect")
    else:
        print(f"\nCorrectly reduced by {expected_reduction} frames")
    
    # Update for next iteration
    action_lists_current = action_lists_new
    phase_boundaries_current = phase_boundaries_new

# Final verification
final_length = debug_print_lengths(action_lists_current, phase_boundaries_current, "Final state")
total_reduction = initial_length - final_length
print(f"\nTotal frames removed through overlapping: {total_reduction}")

# Update the final results
action_lists = action_lists_current
phase_boundaries = phase_boundaries_current

# Final update after all overlaps
# action_lists = action_lists_current
# phase_boundaries = phase_boundaries_current
must_include_indices = phase_boundaries

# print("\nFinal phase boundaries after all overlaps:")
# for i, boundary in enumerate(phase_boundaries):
#     print(f"Phase {i} boundary: {boundary}")



# Phase ranges with weights using accumulated phase boundaries
# phase_ranges = [
#     # (start_idx, end_idx, weight)
#     (phase_boundaries[0], phase_boundaries[1], phase_weights[0]),  # Neutral to N (0-0.23s)
#     (phase_boundaries[1], phase_boundaries[2], phase_weights[1]),  # N to AA (0.23-0.52s)
#     (phase_boundaries[2], phase_boundaries[3], phase_weights[2]),  # AA to T (0.52-0.67s)
#     (phase_boundaries[3], phase_boundaries[4], phase_weights[3]),  # T to IY (0.67-0.89s)
#     (phase_boundaries[4], phase_boundaries[5], phase_weights[4]),  # IY to V (0.89-0.98s)
#     (phase_boundaries[5], phase_boundaries[6], phase_weights[5]),  # V to N (0.98-1.14s)
#     (phase_boundaries[6], phase_boundaries[7], phase_weights[6]),  # N to Y (1.14-1.16s)
#     (phase_boundaries[7], phase_boundaries[8], phase_weights[7])   # Y to UW (1.16-1.68s)
# ]

phase_ranges = [
    (phase_boundaries[i], phase_boundaries[i+1], phase_weights[i]) 
    for i in range(len(phonemes)-1)
]

# Print verification
# print("\nAccumulated phase_boundaries =", phase_boundaries)
# print("\nMust Include Indices:", must_include_indices)
# print("\nPhase Ranges:")
# for i, (start, end, weight) in enumerate(phase_ranges):
#     print(f"Phase {i+1}: {start}-{end} (weight: {weight:.3f})")

# Print phase information for verification
# print("Phoneme Transition Durations and Weights:")
# for i in range(len(phonemes)-1):
#     print(f"{phonemes[i]} to {phonemes[i+1]}: {phase_durations[i]:.2f}s ({phase_weights[i]*100:.1f}%)")

# Calculate checkpoint counts per phase (minus must-include points)
total_weights = sum(weight for _, _, weight in phase_ranges)
remaining_checkpoints = num_checkpoints - len(must_include_indices)

# Initial distribution
phase_counts = [max(1, int(round(weight * remaining_checkpoints / total_weights))) for _, _, weight in phase_ranges]

# Adjust counts to sum exactly to remaining_checkpoints
while sum(phase_counts) != remaining_checkpoints:
    if sum(phase_counts) < remaining_checkpoints:
        # Find phase with largest weight that's underrepresented
        phase_idx = np.argmax([
            (weight * remaining_checkpoints / total_weights) - count 
            for (_, _, weight), count in zip(phase_ranges, phase_counts)
        ])
        phase_counts[phase_idx] += 1
    else:
        # Find phase with smallest weight that's overrepresented
        phase_idx = np.argmin([
            (weight * remaining_checkpoints / total_weights) - count 
            for (_, _, weight), count in zip(phase_ranges, phase_counts)
        ])
        phase_counts[phase_idx] -= 1

# Generate samples with dynamic spacing
checkpoints = []
for i, (start, end, _) in enumerate(phase_ranges):
    if phase_counts[i] == 0:
        continue
        
    # Calculate ideal spacing for this phase
    phase_length = end - start
    dynamic_min_distance = phase_length / (phase_counts[i] + 1)  # +1 for end buffer
    
    # Create sampling range with dynamic buffer
    buffer = int(buffer_ratio * phase_length)
    valid_start = start + buffer
    valid_end = end - buffer
    
    # Evenly sample with phase-appropriate spacing
    phase_indices = np.linspace(
        valid_start, 
        valid_end, 
        num=phase_counts[i], 
        dtype=int
    ).tolist()
    
    # Ensure minimum spacing from must-include points
    phase_indices = [
        x for x in phase_indices 
        if all(abs(x - m) > dynamic_min_distance/3 for m in must_include_indices)  # Relaxed spacing
    ]
    
    # If we lost points, add them back at ideal positions
    if len(phase_indices) < phase_counts[i]:
        ideal_positions = np.linspace(valid_start, valid_end, phase_counts[i]+2)[1:-1]  # Exclude endpoints
        phase_indices = [int(x) for x in ideal_positions]
    
    checkpoints.extend(phase_indices)

# Combine with must-include points and sort
checkpoints = sorted(set(checkpoints + must_include_indices))

# Final adjustment to ensure exactly num_checkpoints
while len(checkpoints) > num_checkpoints:
    # Remove point with smallest distance to neighbors
    distances = []
    for i in range(1, len(checkpoints)-1):
        distances.append((i, checkpoints[i+1] - checkpoints[i-1]))
    if distances:
        min_dist_idx = min(distances, key=lambda x: x[1])[0]
        checkpoints.pop(min_dist_idx)
    else:
        checkpoints.pop()  # fallback

while len(checkpoints) < num_checkpoints:
    # Find largest gap and add a point
    max_gap = 0
    insert_idx = 1
    for i in range(1, len(checkpoints)):
        gap = checkpoints[i] - checkpoints[i-1]
        if gap > max_gap:
            max_gap = gap
            insert_idx = i
    new_point = checkpoints[insert_idx-1] + max_gap//2
    checkpoints.insert(insert_idx, new_point)
    checkpoints = sorted(checkpoints)  # Re-sort to be safe

print("\nnew Optimized checkpoints:", checkpoints)
print("\nCount per phase:")
for i, (start, end, _) in enumerate(phase_ranges):
    count = sum(1 for x in checkpoints if start <= x < end)
    print(f"{phonemes[i]} to {phonemes[i+1]}: {count} checkpoints")
print(f"\nTotal checkpoints: {len(checkpoints)} (target: {num_checkpoints})")




# from here, cover the old list with new list:



print("before:")
print("phase_boundaries", phase_boundaries_old)
print("must_include_indices", must_include_indices_old)
print("len action_lists", len(action_lists_old))
print("check point", checkpoints_old)

print("after")
print("phase_boundaries", phase_boundaries)
print("must_include_indices", must_include_indices)
print("len action_lists", len(action_lists))
print("check point", checkpoints)


# Calculate total length across all phases
total_lengths = [len(lst) for lst in action_lists]
cumulative_lengths = np.cumsum([0] + total_lengths)
total_length_all = cumulative_lengths[-1]
print(f"Total length across all phases: {total_length_all}")



checkpoint_indices = checkpoints


def accumulate_checkpoints(action_lists, checkpoint_indices):
    # Initialize state
    delta_b0_conca = [0] * (self_n + 3 * joint_rotate_num)
    checkpoints = []
    global_index = 0
    found_indices = set()
    
    print(f"\nStarting checkpoint accumulation...")
    
    # Calculate total number of frames
    total_frames = sum(len(lst) for lst in action_lists)
    print(f"Total frames available: {total_frames}")
    
    # Adjust last checkpoint if it's exactly one beyond available frames
    if checkpoint_indices[-1] == total_frames:
        print(f"Note: Last checkpoint index {checkpoint_indices[-1]} equals total frames")
        checkpoint_indices[-1] = total_frames - 1
        print(f"Adjusted last checkpoint to index {checkpoint_indices[-1]}")
    
    print(f"Checkpoint indices: {checkpoint_indices}")
    
    for phase_idx, action_list in enumerate(action_lists):
        if not action_list:
            continue
            
        for action_idx, action in enumerate(action_list):
            # Update the accumulated state
            delta_b0_conca = [delta_b0_conca[j] + action[j] for j in range(len(delta_b0_conca))]
            
            # Check if current global index matches any checkpoint
            if global_index in checkpoint_indices:
                checkpoints.append((global_index, delta_b0_conca.copy()))
                found_indices.add(global_index)
                print(f"Checkpoint at {global_index} (Phase {phase_idx+1}, Action {action_idx})")
            
            global_index += 1
    
    # Final verification
    missing = set(checkpoint_indices) - found_indices
    if missing:
        print(f"\nWarning: Missing {len(missing)} checkpoints: {sorted(missing)}")
        print(f"Maximum available index was {global_index-1}")
        
        # Special case: if only missing the last checkpoint and we're at end
        if len(missing) == 1 and list(missing)[0] == global_index:
            print("Capturing final frame as last checkpoint")
            checkpoints.append((global_index-1, delta_b0_conca.copy()))
    
    print(f"\nSuccessfully captured {len(checkpoints)}/{len(checkpoint_indices)} checkpoints")
    return checkpoints

# Process all phases
combined_checkpoints = accumulate_checkpoints(action_lists, checkpoint_indices)

# Save results
output_path = os.path.join(log_path_output, "combined_checkpoints.pkl")
with open(output_path, 'wb') as f:
    pickle.dump([cp[1] for cp in combined_checkpoints], f)  # Save only the states

print(f"\nFinal checkpoint indices captured:")
for idx, _ in combined_checkpoints:
    print(f"- {idx}")

print(f"\nResults saved to: {output_path}")



print(f"\nSuccessfully processed {len(combined_checkpoints)} checkpoints across {len(action_lists)} phases")
print(f"Results saved to: {output_path}")


# print(f"\nSuccessfully processed {len(combined_checkpoints)} checkpoints across 8 phases")
# print(f"Results saved to: {output_path}")

# Verification
try:
    with open(output_path, "rb") as f:
        loaded_checkpoints = pickle.load(f)
    print(f"Verification: Loaded {len(loaded_checkpoints)} checkpoints")
except Exception as e:
    print(f"Error verifying output: {e}")

# Print phase distribution
print("\nCheckpoint distribution by phase:")
for i in range(len(phase_boundaries)-1):
    start = phase_boundaries[i]
    end = phase_boundaries[i+1]
    count = sum(1 for idx in checkpoint_indices if start <= idx < end)
    print(f"{phase_names[i]}: {count} checkpoints ({start}-{end})")