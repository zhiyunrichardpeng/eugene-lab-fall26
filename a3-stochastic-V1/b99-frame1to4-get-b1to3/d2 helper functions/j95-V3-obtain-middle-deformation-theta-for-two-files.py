import numpy as np
import pickle
import os
from paths import results_path

# Configuration
base_path =str(results_path / "job_250623-144458/250623-144522")
only_run_phase1_flag = 0  # Set to 1 to only run phase 1

# Initialize parameters
self_n = 300 * 3  # Replace with actual value of `self.n` (see config file.)
joint_rotate_num = 1  # Replace with actual value of `self.joint_rotate_num`
num_checkpoints = 36  # Total number of checkpoints across all phases

# Define phase information (modify these ranges according to your actual phase boundaries)
phase_boundaries = [0, 36782, 58781, 88713, 108097, 162649, 216505, 237327, 314143]
phase_names = ["Phase1", "Phase2", "Phase3", "Phase4", "Phase5", "Phase6", "Phase7", "Phase8"]

# Load action lists for all phases
action_lists = []
for i in range(1, 9):
    if only_run_phase1_flag and i > 1:
        continue
        
    if i == 1:
        file_path = os.path.join(base_path, f"final_epoch_action_list.pkl")
        # r"\final_epoch_action_list.pkl",
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
checkpoint_indices = [0, 3678, 13486, 23295, 33104, 36782, 38981, 43381, 47781, 
                      52181, 56582, 58781, 69756, 77738, 88713, 90651, 95820, 
                      100989, 106159, 108097, 135373, 162649, 168034, 189577, 
                      211120, 216505, 237327, 245008, 253787, 262566, 271345, 
                      280124, 288903, 297682, 306462, 314143]

def accumulate_checkpoints(action_lists, checkpoint_indices):
    delta_b0_conca = [0] * self_n + [0] * 3 * joint_rotate_num
    checkpoints = []
    global_index = 0
    
    for phase_idx, action_list in enumerate(action_lists):
        if not action_list:  # Skip empty phases
            continue
            
        print(f"Processing phase {phase_idx+1} with {len(action_list)} actions")
        
        for action in action_list:
            # Update the accumulated state
            delta_b0_conca = [delta_b0_conca[j] + action[j] for j in range(self_n + 3 * joint_rotate_num)]
            
            # Check if current global index matches any checkpoint
            if global_index in checkpoint_indices:
                checkpoints.append(delta_b0_conca.copy())
                print(f"Checkpoint at index {global_index} (phase {phase_idx+1})")
            
            global_index += 1
    
    # Verify we got all checkpoints
    if len(checkpoints) != len(checkpoint_indices):
        print(f"Warning: Expected {len(checkpoint_indices)} checkpoints, got {len(checkpoints)}")
    
    return checkpoints

# Process all phases
combined_checkpoints = accumulate_checkpoints(action_lists, checkpoint_indices)

# Save results
# output_path = os.path.join(base_path, "combined_checkpoints_8phases.pkl")
output_path = os.path.join(base_path, "combined_checkpoints.pkl")
with open(output_path, "wb") as f:
    pickle.dump(combined_checkpoints, f)

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