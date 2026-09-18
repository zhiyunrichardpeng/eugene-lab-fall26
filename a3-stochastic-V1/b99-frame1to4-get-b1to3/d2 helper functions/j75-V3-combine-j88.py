# import numpy as np

# import re
# import os

# def getPhase_boundaries():

#     log_path = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\job_250623-144458\250623-144522"
#     phase_files = [
#         "log0.txt",
#         "log0_phase2.txt",
#         "log0_phase3.txt",
#         "log0_phase4.txt",
#         "log0_phase5.txt",
#         "log0_phase6.txt",
#         "log0_phase7.txt",
#         "log0_phase8.txt"
#     ]

#     phase_boundaries = [0]

#     for file in phase_files:
#         file_path = os.path.join(log_path, file)
#         if os.path.exists(file_path):
#             with open(file_path, 'r') as f:
#                 content = f.read()
#                 match = re.search(r'done at No\. (\d+) step', content)
#                 if match:
#                     phase_boundaries.append(int(match.group(1)))
#         else:
#             print(f"File not found: {file_path}")

#     print("phase_boundaries =", phase_boundaries)
#     return phase_boundaries

# phase_boundaries = getPhase_boundaries()

# # Timestamp data (in seconds)
# timestamps = [0, 0.23, 0.52, 0.67, 0.89, 0.98, 1.14, 1.16, 1.68]
# phonemes = ["Neutral", "N", "AA", "T", "IY", "V", "N", "Y", "UW"]
# total_duration = timestamps[-1]  # 1.68 seconds

# # Calculate phase durations and weights
# phase_durations = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
# phase_weights = [duration/total_duration for duration in phase_durations]


# # Input parameters (accumulated indices)
# must_include_indices = [
#     0,                          # Start
#     36782,                      # N
#     36782 + 21999,              # AA
#     36782 + 21999 + 29932,      # T
#     36782 + 21999 + 29932 + 19384,   # IY
#     36782 + 21999 + 29932 + 19384 + 54552,  # V
#     36782 + 21999 + 29932 + 19384 + 54552 + 53856,  # N
#     36782 + 21999 + 29932 + 19384 + 54552 + 53856 + 20822,  # Y
#     36782 + 21999 + 29932 + 19384 + 54552 + 53856 + 20822 + 76816  # UW
# ]

# num_checkpoints = 36
# buffer_ratio = 0.1  # 10% buffer around must-include points

# # Phase ranges with weights based on actual timestamps
# phase_ranges = [
#     # (start_idx, end_idx, weight)
#     (0, 36782, phase_weights[0]),               # Neutral to N (0-0.23s)
#     (36782, 36782+21999, phase_weights[1]),     # N to AA (0.23-0.52s)
#     (36782+21999, 36782+21999+29932, phase_weights[2]),      # AA to T (0.52-0.67s)
#     (36782+21999+29932, 36782+21999+29932+19384, phase_weights[3]),  # T to IY (0.67-0.89s)
#     (36782+21999+29932+19384, 36782+21999+29932+19384+54552, phase_weights[4]),  # IY to V (0.89-0.98s)
#     (36782+21999+29932+19384+54552, 36782+21999+29932+19384+54552+53856, phase_weights[5]),  # V to N (0.98-1.14s)
#     (36782+21999+29932+19384+54552+53856, 36782+21999+29932+19384+54552+53856+20822, phase_weights[6]),  # N to Y (1.14-1.16s)
#     (36782+21999+29932+19384+54552+53856+20822, 36782+21999+29932+19384+54552+53856+20822+76816, phase_weights[7])  # Y to UW (1.16-1.68s)
# ]

import numpy as np
import re
import os

# import numpy as np
import pickle
# import os
log_path = r"\\bbysvm-nfs1.its.sfu.ca\GrUViLab\Data\richardPeng\keyLabCloud\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003_25-7-18\job_250623-144458\250623-144522"


def get_total_steps_convergence():
    """Extracts phase step counts from log files (non-accumulated values)"""
    log_path = r"\\bbysvm-nfs1.its.sfu.ca\GrUViLab\Data\richardPeng\keyLabCloud\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003_25-7-18\job_250623-144458\250623-144522"
    phase_files = [
        "log0.txt",
        "log0_phase2.txt",
        "log0_phase3.txt",
        "log0_phase4.txt",
        "log0_phase5.txt",
        "log0_phase6.txt",
        "log0_phase7.txt",
        "log0_phase8.txt"
    ]

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
total_steps_convergence = get_total_steps_convergence()

# Calculate accumulated phase boundaries
phase_boundaries = np.cumsum(total_steps_convergence).tolist()

# Timestamp data (in seconds)
timestamps = [0, 0.23, 0.52, 0.67, 0.89, 0.98, 1.14, 1.16, 1.68]
phonemes = ["Neutral", "N", "AA", "T", "IY", "V", "N", "Y", "UW"]
total_duration = timestamps[-1]  # 1.68 seconds

# Calculate phase durations and weights
phase_durations = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
phase_weights = [duration/total_duration for duration in phase_durations]

# Must-include indices are exactly the accumulated phase boundaries
must_include_indices = phase_boundaries.copy()

num_checkpoints = 288
buffer_ratio = 0.1  # 10% buffer around must-include points

# Phase ranges with weights using accumulated phase boundaries
phase_ranges = [
    # (start_idx, end_idx, weight)
    (phase_boundaries[0], phase_boundaries[1], phase_weights[0]),  # Neutral to N (0-0.23s)
    (phase_boundaries[1], phase_boundaries[2], phase_weights[1]),  # N to AA (0.23-0.52s)
    (phase_boundaries[2], phase_boundaries[3], phase_weights[2]),  # AA to T (0.52-0.67s)
    (phase_boundaries[3], phase_boundaries[4], phase_weights[3]),  # T to IY (0.67-0.89s)
    (phase_boundaries[4], phase_boundaries[5], phase_weights[4]),  # IY to V (0.89-0.98s)
    (phase_boundaries[5], phase_boundaries[6], phase_weights[5]),  # V to N (0.98-1.14s)
    (phase_boundaries[6], phase_boundaries[7], phase_weights[6]),  # N to Y (1.14-1.16s)
    (phase_boundaries[7], phase_boundaries[8], phase_weights[7])   # Y to UW (1.16-1.68s)
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

print("\nOptimized checkpoints:", checkpoints)
print("\nCount per phase:")
for i, (start, end, _) in enumerate(phase_ranges):
    count = sum(1 for x in checkpoints if start <= x < end)
    print(f"{phonemes[i]} to {phonemes[i+1]}: {count} checkpoints")
print(f"\nTotal checkpoints: {len(checkpoints)} (target: {num_checkpoints})")





## second part of the code



# Configuration
base_path = r"\\bbysvm-nfs1.its.sfu.ca\GrUViLab\Data\richardPeng\keyLabCloud\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003_25-7-18\job_250623-144458\250623-144522"
only_run_phase1_flag = 0  # Set to 1 to only run phase 1

# Initialize parameters
self_n = 300 * 3  # Replace with actual value of `self.n` (see config file.)
joint_rotate_num = 1  # Replace with actual value of `self.joint_rotate_num`
num_checkpoints = 288  # Total number of checkpoints across all phases

# Define phase information (modify these ranges according to your actual phase boundaries)
# phase_boundaries = [0, 36782, 58781, 88713, 108097, 162649, 216505, 237327, 314143]
phase_boundaries, total_steps_convergence
# phase_boundaries = 
phase_names = ["Phase1", "Phase2", "Phase3", "Phase4", "Phase5", "Phase6", "Phase7", "Phase8"]

# Load action lists for all phases
action_lists = []
for i in range(1, 9):
    if only_run_phase1_flag and i > 1:
        continue
        
    if i == 1:
        file_path = os.path.join(base_path, f"final_epoch_action_list.pkl")
        r"\final_epoch_action_list.pkl",
    else:
        file_path = os.path.join(base_path, f"final_epoch_action_list_phase{i}.pkl")
    
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            action_lists.append(data[f'final_epoch_action_list'])
            print(f"Successfully loaded phase {i} with {len(data[f'final_epoch_action_list'])} actions")
    except FileNotFoundError:
        if i == 1:  # Phase 1 is required
            raise
        print(f"Warning: Could not find action list for phase {i}")
        action_lists.append([])

# Calculate total length across all phases
total_lengths = [len(lst) for lst in action_lists]
cumulative_lengths = np.cumsum([0] + total_lengths)
total_length_all = cumulative_lengths[-1]
print(f"Total length across all phases: {total_length_all}")

# Define checkpoint indices (pre-calculated based on phase boundaries)
# checkpoint_indices = [0, 3678, 13486, 23295, 33104, 36782, 38981, 43381, 47781, 
#                       52181, 56582, 58781, 69756, 77738, 88713, 90651, 95820, 
#                       100989, 106159, 108097, 135373, 162649, 168034, 189577, 
#                       211120, 216505, 237327, 245008, 253787, 262566, 271345, 
#                       280124, 288903, 297682, 306462, 314143]

checkpoint_indices = checkpoints

# def accumulate_checkpoints(action_lists, checkpoint_indices):
#     delta_b0_conca = [0] * self_n + [0] * 3 * joint_rotate_num
#     checkpoints = []
#     global_index = 0
    
#     for phase_idx, action_list in enumerate(action_lists):
#         if not action_list:  # Skip empty phases
#             continue
            
#         print(f"Processing phase {phase_idx+1} with {len(action_list)} actions")
        
#         for action in action_list:
#             # Update the accumulated state
#             delta_b0_conca = [delta_b0_conca[j] + action[j] for j in range(self_n + 3 * joint_rotate_num)]
            
#             # Check if current global index matches any checkpoint
#             if global_index in checkpoint_indices:
#                 checkpoints.append(delta_b0_conca.copy())
#                 print(f"Checkpoint at index {global_index} (phase {phase_idx+1})")
            
#             global_index += 1
    
#     # Verify we got all checkpoints
#     if len(checkpoints) != len(checkpoint_indices):
#         print(f"Warning: Expected {len(checkpoint_indices)} checkpoints, got {len(checkpoints)}")
    
#     return checkpoints

# # Process all phases
# combined_checkpoints = accumulate_checkpoints(action_lists, checkpoint_indices)

# def accumulate_checkpoints(action_lists, checkpoint_indices):
#     # Initialize state
#     delta_b0_conca = [0] * (self_n + 3 * joint_rotate_num)
#     checkpoints = []
#     global_index = 0
#     found_indices = set()
    
#     print(f"\nStarting checkpoint accumulation...")
#     print(f"Total checkpoints expected: {len(checkpoint_indices)}")
#     print(f"Checkpoint indices: {sorted(checkpoint_indices)}")
    
#     for phase_idx, action_list in enumerate(action_lists):
#         if not action_list:  # Skip empty phases
#             print(f"Phase {phase_idx+1} is empty, skipping")
#             continue
            
#         print(f"\nProcessing phase {phase_idx+1} with {len(action_list)} actions")
#         print(f"Global index range: {global_index} to {global_index + len(action_list) - 1}")
        
#         for action_idx, action in enumerate(action_list):
#             # Update the accumulated state
#             delta_b0_conca = [delta_b0_conca[j] + action[j] for j in range(len(delta_b0_conca))]
            
#             # Check if current global index matches any checkpoint
#             if global_index in checkpoint_indices:
#                 checkpoints.append((global_index, delta_b0_conca.copy()))
#                 found_indices.add(global_index)
#                 print(f"  Checkpoint found at:")
#                 print(f"    Global index: {global_index}")
#                 print(f"    Phase: {phase_idx+1}")
#                 print(f"    Action index: {action_idx}")
#                 print(f"    Current state: {delta_b0_conca[:5]}...")  # First 5 elements for brevity
            
#             global_index += 1
    
#     # Verification and error reporting
#     print("\nCheckpoint accumulation complete")
#     print(f"Found {len(checkpoints)} checkpoints")
    
#     # Check for missing checkpoints
#     missing = set(checkpoint_indices) - found_indices
#     if missing:
#         print(f"\nERROR: Missing {len(missing)} checkpoints:")
#         for idx in sorted(missing):
#             print(f"  - Index {idx} not found")
        
#         # Find the maximum available index
#         max_index = global_index - 1
#         print(f"\nMaximum available index was {max_index}")
        
#         # Suggest adjustments for missing checkpoints beyond the data
#         out_of_range = [idx for idx in missing if idx > max_index]
#         if out_of_range:
#             print("\nSome checkpoints are beyond the available data range:")
#             for idx in out_of_range:
#                 print(f"  - Index {idx} is beyond maximum index {max_index}")
    
#     return checkpoints

# # Process all phases with enhanced debugging
# print(f"\nStarting checkpoint processing...")
# print(f"Number of action lists: {len(action_lists)}")
# print(f"Total actions: {sum(len(lst) for lst in action_lists)}")

# combined_checkpoints = accumulate_checkpoints(action_lists, checkpoint_indices)

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
output_path = os.path.join(log_path, "combined_checkpoints.pkl")
with open(output_path, 'wb') as f:
    pickle.dump([cp[1] for cp in combined_checkpoints], f)  # Save only the states

print(f"\nFinal checkpoint indices captured:")
for idx, _ in combined_checkpoints:
    print(f"- {idx}")

print(f"\nResults saved to: {output_path}")


# Save results
# output_path = os.path.join(log_path, "combined_checkpoints.pkl")
# with open(output_path, 'wb') as f:
#     pickle.dump(combined_checkpoints, f)

print(f"\nSuccessfully processed {len(combined_checkpoints)} checkpoints across {len(action_lists)} phases")
print(f"Results saved to: {output_path}")

# Verification load
# with open(output_path, 'rb') as f:
#     loaded_checkpoints = pickle.load(f)
# print(f"Verification: Loaded {len(loaded_checkpoints)} checkpoints")

# Save results
# output_path = os.path.join(base_path, "combined_checkpoints_8phases.pkl")
# output_path = os.path.join(base_path, "combined_checkpoints.pkl")
# with open(output_path, "wb") as f:
#     pickle.dump(combined_checkpoints, f)

print(f"\nSuccessfully processed {len(combined_checkpoints)} checkpoints across 8 phases")
print(f"Results saved to: {output_path}")

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