import numpy as np
import re
import os
import math
import pickle
import json

overlap_ratio = 0.4

# Configuration for the 3 separate paths
log_paths = [
    r"\\bbysvm-nfs1.its.sfu.ca/GrUViLab/Data/richardPeng/SFU_PhD_cloud/2-Projects/1-physically based facial animation/3-Experiments/a3-stochastic-V1-backup-and-results/results/job_251206-021708/251206-021716", # from f83, neutral to V
    r"\\bbysvm-nfs1.its.sfu.ca/GrUViLab/Data/richardPeng/SFU_PhD_cloud/2-Projects/1-physically based facial animation/3-Experiments/a3-stochastic-V1-backup-and-results/results/job_251224-150822/251224-150830"  # from f70,  v to aa
]




# 1. Separate the folder name
base_path = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\sequence_combined_deformation_parameters_hub"
folder_name = "0-V-AA"

# 2. Create the full path
log_path_output = os.path.join(base_path, folder_name)

# 3. Create the folder if it doesn't exist
if not os.path.exists(log_path_output):
    os.makedirs(log_path_output)
    print(f"Created directory: {log_path_output}")
else:
    print(f"Directory already exists: {log_path_output}")

# log_path_output = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\sequence_combined_deformation_parameters_hub\seq2_0-AU2L-0-AU2R-0"

def get_total_steps_from_separate_logs():
    """Extracts step counts from 3 separate log files"""
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

# Calculate accumulated phase boundaries (now only 3 phases + start)
phase_boundaries = np.cumsum(total_steps_convergence).tolist()
phase_boundaries_old = phase_boundaries.copy()

print(f"Phase boundaries: {phase_boundaries}")

# Timestamp data (in seconds) - adjust for 3 phases

# phonemes = ["Neutral", "AU2L", "Neutral", "AU2R", "Neutral"]  # Adjust phonemes for your 3 phases
# timestamps = [0, 0.52, 1.14, 1.68]  # Adjust timestamps for your 3 phases

phonemes = ["Neutral", "V", "AA"]
total_duration = 10 # This is dummy time. only for the percentage.

# generate equal interval's timestemps.
timestamps = [i * total_duration / (len(phonemes) - 1) for i in range(len(phonemes))]

print(f"expressions: {phonemes}")
print(f"Timestamps: {[round(t, 2) for t in timestamps]}")
# Output: [0.0, 0.42, 0.84, 1.26, 1.68]


def adjust_durations_with_overlap(timestamps, overlap_ratio):
    # Calculate original durations between phonemes
    original_durations = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    
    # Calculate adjusted durations with 40% overlap
    adjusted_durations = []
    for i in range(len(original_durations)-1):
        # Each duration is reduced by 40% of the next duration
        adjusted_duration = original_durations[i] - original_durations[i+1] * overlap_ratio
        adjusted_durations.append(max(0, adjusted_duration))
    
    # Keep the last duration unchanged
    adjusted_durations.append(original_durations[-1])
    
    # Reconstruct absolute timestamps from adjusted durations
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
adjusted_phase_durations = [adjusted_timestamps[i+1] - adjusted_timestamps[i] for i in range(len(adjusted_timestamps)-1)]
adjusted_phase_weights = [duration/total_duration for duration in adjusted_phase_durations]

phase_durations = adjusted_phase_durations
phase_weights = adjusted_phase_weights

# Must-include indices are exactly the accumulated phase boundaries
must_include_indices = phase_boundaries.copy()
must_include_indices_old = must_include_indices.copy()

# Time-preserving overlap: keep the same number of checkpoints (and thus
# the same number of animation frames) regardless of overlap_ratio.
# The blended action list is shorter, but we sample more densely from it.
# OLD (duration-shrinking):
#   time_reduced_percentage = (timestamps[-1] - adjusted_timestamps[-1])/timestamps[-1]
#   num_checkpoints = math.floor(36 * (len(phonemes) - 1) * (1 - time_reduced_percentage))
num_checkpoints = 36 * (len(phonemes) - 1)

buffer_ratio = 0.1

# Adaptive phase ranges based on phonemes length
phase_ranges = []
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
            # Handle different possible key names
            if 'final_epoch_action_list' in data:
                action_lists.append(data['final_epoch_action_list'])
            elif 'action_list' in data:
                action_lists.append(data['action_list'])
            else:
                # Assume the data is the action list itself
                action_lists.append(data)
            print(f"Successfully loaded phase {i+1} from {log_path} with {len(action_lists[-1])} actions")
    except FileNotFoundError:
        print(f"Warning: Could not find action list at {file_path}")
        action_lists.append([])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        action_lists.append([])

action_lists_old = action_lists.copy()

# ===================================================================
# OVERLAP BLENDING WITH MIDPOINT SPLIT
# ===================================================================
#
# After blending L2 + L3 -> L2', we use the MIDPOINT of L2' as the
# new boundary between the two transitions. This lets us save each
# transition (Neutral->V, V->AA) as its own combined_checkpoints.pkl,
# with the overlap baked in. h99-UI-mayascipt-V1.py can then call
# each transition individually via the updated deformation_path_db.json.
#
# Diagram:
#   Original:  [...Phase1... | ...Phase2...]
#                         ^B (original boundary)
#   With overlap window W around B:
#     L1 = Phase1 actions before overlap
#     L2 = last W actions of Phase1  (tail)
#     L3 = first W actions of Phase2 (head)
#     L4 = Phase2 actions after overlap
#
#   Blended:  L2'[k] = L2[k] + L3[k]   for k=1..W
#
#   Midpoint split of L2':
#     L2'_first = L2'[0 : W//2]    -> appended to Phase1
#     L2'_second = L2'[W//2 : W]   -> prepended to Phase2
#
#   Final Phase1 actions = L1 + L2'_first
#   Final Phase2 actions = L2'_second + L4
#
# ===================================================================

def blend_with_midpoint_split(action_lists, phase_boundaries, overlap_ratio):
    """
    Blend overlapping actions between adjacent phases and split at the
    midpoint of the blended region.
    
    Also computes the OVERLAP CORRECTION OFFSET for each boundary.
    
    At the midpoint of L2', the landmark position differs from the ideal
    target because:
      - The second half of original L2 (outgoing transition) was NOT applied
      - The first half of original L3 (incoming transition) WAS prematurely absorbed
    
    Correction offset = cumsum(L2[W//2:W]) - cumsum(L3[0:W//2])
    
    Adding this offset to the actual boundary position gives the "true"
    target position for error computation in h99.
    
    Returns:
        split_action_lists: list of action lists, one per transition,
                            with overlap baked in and split at midpoint.
        new_boundaries: updated phase boundaries after midpoint split.
        overlap_corrections: list of correction offset vectors (one per boundary),
                             to be saved and used by h99 for error dissipation.
    """
    # Flatten all actions into a single list
    flat_actions = []
    for phase in action_lists:
        flat_actions.extend(phase)
    
    total_len = len(flat_actions)
    dim = len(flat_actions[0]) if flat_actions else 0
    print(f"\n[MIDPOINT SPLIT] Total flat actions: {total_len}, dimension: {dim}")
    print(f"[MIDPOINT SPLIT] Original phase boundaries: {phase_boundaries}")
    print(f"[MIDPOINT SPLIT] Overlap ratio: {overlap_ratio}")
    
    num_transitions = len(phase_boundaries) - 1
    
    new_flat_actions = list(flat_actions)
    cumulative_offset = 0
    
    blend_infos = []
    overlap_corrections = []  # One correction vector per boundary
    
    for b_idx in range(1, len(phase_boundaries) - 1):
        B = phase_boundaries[b_idx] - cumulative_offset
        
        back_phase_len = phase_boundaries[b_idx + 1] - phase_boundaries[b_idx]
        front_phase_len = phase_boundaries[b_idx] - phase_boundaries[b_idx - 1]
        W = int(min(front_phase_len, back_phase_len) * overlap_ratio)
        
        if W == 0:
            print(f"[MIDPOINT SPLIT] Boundary {b_idx}: W=0, no overlap to apply")
            blend_infos.append({'boundary_idx': b_idx, 'W': 0, 'midpoint_offset': 0})
            overlap_corrections.append(np.zeros(dim).tolist())
            continue
        
        print(f"\n[MIDPOINT SPLIT] Processing boundary {b_idx} at adjusted position {B}")
        print(f"  W = {W} (overlap window)")
        
        # Extract the four regions from the current flat list
        l1 = new_flat_actions[:B - W]
        l2 = new_flat_actions[B - W:B]
        l3 = new_flat_actions[B:B + W]
        l4 = new_flat_actions[B + W:]
        
        print(f"  L1: {len(l1)} actions, L2: {len(l2)} actions, L3: {len(l3)} actions, L4: {len(l4)} actions")
        
        # ---------------------------------------------------------------
        # COMPUTE OVERLAP CORRECTION OFFSET (before blending modifies l2/l3)
        # ---------------------------------------------------------------
        # At the midpoint boundary, to recover the "true target" position:
        #   true_target = actual_at_boundary + correction
        # where:
        #   correction = sum(L2[W//2 : W]) - sum(L3[0 : W//2])
        #
        # L2[W//2:W] = remaining (unfinished) outgoing movement
        # L3[0:W//2] = prematurely absorbed incoming movement
        mid = len(l2) // 2
        
        # Cumulative sum of L2 second half (the unfinished part)
        l2_second_half = l2[mid:]  # original L2 actions, indices [W//2 : W]
        remaining_L2 = np.zeros(dim)
        for action in l2_second_half:
            remaining_L2 += np.array(action)
        
        # Cumulative sum of L3 first half (the prematurely absorbed part)
        l3_first_half = l3[:mid]   # original L3 actions, indices [0 : W//2]
        absorbed_L3 = np.zeros(dim)
        for action in l3_first_half:
            absorbed_L3 += np.array(action)
        
        # Correction offset
        correction = remaining_L2 - absorbed_L3
        
        print(f"  Correction offset magnitude: {np.linalg.norm(correction):.6f}")
        print(f"    remaining_L2 magnitude: {np.linalg.norm(remaining_L2):.6f}")
        print(f"    absorbed_L3 magnitude:  {np.linalg.norm(absorbed_L3):.6f}")
        
        overlap_corrections.append(correction.tolist())
        
        # ---------------------------------------------------------------
        # BLEND L2 + L3 -> L2'
        # ---------------------------------------------------------------
        l2_prime = []
        for a, b in zip(l2, l3):
            blended = [x + y for x, y in zip(a, b)]
            l2_prime.append(blended)
        
        # Midpoint split of L2'
        l2_first = l2_prime[:mid]
        l2_second = l2_prime[mid:]
        
        print(f"  L2' total: {len(l2_prime)}, first half: {len(l2_first)}, second half: {len(l2_second)}")
        
        # Rebuild flat actions
        new_flat_actions = l1 + l2_first + l2_second + l4
        
        new_boundary = len(l1) + len(l2_first)
        
        blend_infos.append({
            'boundary_idx': b_idx,
            'W': W,
            'mid': mid,
            'new_boundary': new_boundary,
            'removed_steps': W
        })
        
        cumulative_offset += W
        
        print(f"  New boundary position: {new_boundary}")
        print(f"  New flat actions length: {len(new_flat_actions)} (removed {W} steps)")
    
    # Split into per-transition action lists
    new_split_points = [0]
    for info in blend_infos:
        new_split_points.append(info['new_boundary'])
    new_split_points.append(len(new_flat_actions))
    
    print(f"\n[MIDPOINT SPLIT] New split points: {new_split_points}")
    
    split_action_lists = []
    for i in range(len(new_split_points) - 1):
        start = new_split_points[i]
        end = new_split_points[i + 1]
        split_action_lists.append(new_flat_actions[start:end])
        print(f"  Transition {i+1}: actions[{start}:{end}] = {end - start} steps")
    
    return split_action_lists, new_split_points, overlap_corrections


# Run the midpoint-split overlap blending
split_action_lists, new_split_points, overlap_corrections = blend_with_midpoint_split(
    action_lists, phase_boundaries, overlap_ratio
)

print(f"\n{'='*60}")
print(f"MIDPOINT SPLIT COMPLETE")
print(f"{'='*60}")
for i, actions in enumerate(split_action_lists):
    print(f"  Transition {i+1} ({phonemes[i]} -> {phonemes[i+1]}): {len(actions)} actions")

# ===================================================================
# GENERATE PER-TRANSITION combined_checkpoints.pkl
# ===================================================================
#
# For each transition, we subsample 36 checkpoints from its action list,
# accumulate the deformation, and save as combined_checkpoints.pkl.
# This matches the format used by h99-UI-mayascipt-V1.py's
# perform_the_deformation().
# ===================================================================

# Detect correct dimension from the action lists
if action_lists and action_lists[0]:
    correct_dimension = len(action_lists[0][0])
    print(f"\nDetected dimension: {correct_dimension}")
else:
    correct_dimension = 300 * 3 + 3  # fallback
    print(f"\nUsing default dimension: {correct_dimension}")

num_checkpoints_per_transition = 36


def generate_checkpoints_for_transition(actions, num_checkpoints):
    """
    Subsample and accumulate deformation checkpoints for a single transition.
    
    Args:
        actions: list of incremental action vectors for this transition
        num_checkpoints: how many checkpoints to produce (e.g., 36)
    
    Returns:
        List of cumulative deformation vectors (one per checkpoint),
        same format as existing combined_checkpoints.pkl
    """
    total_steps = len(actions)
    
    if total_steps == 0:
        print("  WARNING: No actions for this transition!")
        return []
    
    # Select checkpoint indices (evenly spaced, always including last)
    if total_steps <= num_checkpoints:
        checkpoint_indices = list(range(total_steps))
    else:
        step_size = total_steps / num_checkpoints
        checkpoint_indices = [int(i * step_size) for i in range(num_checkpoints)]
        # Ensure last step is included
        if checkpoint_indices[-1] != total_steps - 1:
            checkpoint_indices[-1] = total_steps - 1
    
    checkpoint_indices = sorted(set(checkpoint_indices))
    
    print(f"  Selected {len(checkpoint_indices)} checkpoint indices from {total_steps} steps")
    print(f"  First few: {checkpoint_indices[:5]}, Last few: {checkpoint_indices[-5:]}")
    
    # Accumulate and snapshot at checkpoints
    cumulative = np.zeros(len(actions[0]))
    checkpoints_data = []
    checkpoint_set = set(checkpoint_indices)
    
    for step_idx, action in enumerate(actions):
        cumulative = cumulative + np.array(action)
        
        if step_idx in checkpoint_set:
            checkpoints_data.append(cumulative.tolist())
    
    print(f"  Generated {len(checkpoints_data)} checkpoints")
    
    # Verify non-zero data
    non_zero_count = sum(1 for cp in checkpoints_data if any(x != 0 for x in cp))
    print(f"  Non-zero checkpoints: {non_zero_count}/{len(checkpoints_data)}")
    
    return checkpoints_data


# Create output directories and generate checkpoints for each transition
output_dirs = []
for i, actions in enumerate(split_action_lists):
    from_name = phonemes[i].replace("Neutral", "neutral")
    to_name = phonemes[i + 1].replace("Neutral", "neutral")
    
    # Create subfolder for this transition
    transition_folder = f"overlap{int(overlap_ratio*100)}pct_{from_name}_to_{to_name}"
    transition_path = os.path.join(log_path_output, transition_folder)
    
    if not os.path.exists(transition_path):
        os.makedirs(transition_path)
        print(f"\nCreated: {transition_path}")
    
    output_dirs.append(transition_path)
    
    print(f"\n--- Generating checkpoints for transition {i+1}: {from_name} -> {to_name} ---")
    print(f"  {len(actions)} total actions")
    
    checkpoints_data = generate_checkpoints_for_transition(actions, num_checkpoints_per_transition)
    
    # Save combined_checkpoints.pkl
    output_path = os.path.join(transition_path, "combined_checkpoints.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(checkpoints_data, f)
    
    print(f"  Saved: {output_path}")
    
    # Save overlap_correction.pkl
    # For transitions 0..K-3 (all except the last), there is a correction at the
    # END boundary. The correction tells h99 how to recover the "true" target
    # position from the actual boundary position:
    #   true_target = actual + correction
    # where correction = sum(L2_second_half) - sum(L3_first_half)
    if i < len(overlap_corrections):
        correction_data = {
            'overlap_correction_offset': overlap_corrections[i],
            'overlap_ratio': overlap_ratio,
            'description': (
                'At the end of this transition (midpoint of blended L2\'), '
                'the actual position differs from the true target. '
                'Add this offset to the actual position to get the true target. '
                'correction = sum(L2[W//2:W]) - sum(L3[0:W//2])'
            )
        }
        correction_path = os.path.join(transition_path, "overlap_correction.pkl")
        with open(correction_path, 'wb') as f:
            pickle.dump(correction_data, f)
        
        corr_mag = np.linalg.norm(overlap_corrections[i])
        print(f"  Saved overlap_correction.pkl (magnitude: {corr_mag:.6f})")
    else:
        # Last transition has no correction (it ends at the true target)
        print(f"  No overlap correction needed (last transition)")
    
    # Verification
    with open(output_path, 'rb') as f:
        loaded = pickle.load(f)
    print(f"  Verification: {len(loaded)} checkpoints loaded, first has {len(loaded[0])} elements")


# ===================================================================
# ALSO SAVE THE COMBINED (ALL-TRANSITIONS) CHECKPOINT
# ===================================================================
# This is the original full-sequence combined_checkpoints.pkl that the
# existing pipeline can also use.

print(f"\n--- Also generating combined (full sequence) checkpoints ---")

# Use the blended flat actions (all transitions concatenated)
all_blended_actions = []
for actions in split_action_lists:
    all_blended_actions.extend(actions)

combined_checkpoints_data = generate_checkpoints_for_transition(
    all_blended_actions, num_checkpoints
)

combined_output_path = os.path.join(log_path_output, "combined_checkpoints.pkl")
with open(combined_output_path, 'wb') as f:
    pickle.dump(combined_checkpoints_data, f)
print(f"  Saved combined: {combined_output_path}")


# ===================================================================
# UPDATE deformation_path_db.json
# ===================================================================
# Add new overlap-aware entries to the database so that h99-UI-mayascipt-V1.py
# can look up each transition individually with the overlap baked in.

db_path = r"C:\Users\zpa18\a3-stochastic-V1\automation\deformation_path_db.json"

print(f"\n--- Updating deformation_path_db.json ---")

# Load existing db
with open(db_path, 'r') as f:
    db = json.load(f)

# Phoneme name mapping for db keys
phoneme_db_names = {
    "Neutral": "dataV22_AU0_neutral",
    "V": "dataV22_phoneme_V",
    "AA": "dataV22_phoneme_AA",
    "N": "dataV22_phoneme_N",
    "T": "dataV22_phoneme_T",
    "IY": "dataV22_phoneme_IY",
    "UW": "dataV22_phoneme_UW",
    "EH": "dataV22_phoneme_EH",
    "Y": "dataV22_phoneme_Y",
}

overlap_pct = int(overlap_ratio * 100)
new_entries = {}

for i in range(len(phonemes) - 1):
    from_phoneme = phonemes[i]
    to_phoneme = phonemes[i + 1]
    
    from_db = phoneme_db_names.get(from_phoneme, from_phoneme)
    to_db = phoneme_db_names.get(to_phoneme, to_phoneme)
    
    # Key format: "dataV22_X to dataV22_Y [overlap40]"
    key = f"{from_db} to {to_db} [overlap{overlap_pct}]"
    
    new_entries[key] = output_dirs[i]
    print(f"  Added: {key}")
    print(f"    -> {output_dirs[i]}")

# Add to database
db["deformation_paths"].update(new_entries)

# Save updated db
with open(db_path, 'w') as f:
    json.dump(db, f, indent=4, sort_keys=True)

print(f"\nDatabase updated: {db_path}")
print(f"Added {len(new_entries)} new overlap-aware entries")


# ===================================================================
# SUMMARY
# ===================================================================
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Overlap ratio: {overlap_ratio} ({overlap_pct}%)")
print(f"Phoneme sequence: {' -> '.join(phonemes)}")
print(f"")
print(f"Per-transition outputs:")
for i, d in enumerate(output_dirs):
    print(f"  {i+1}. {phonemes[i]} -> {phonemes[i+1]}: {d}")
print(f"")
print(f"Combined output: {combined_output_path}")
print(f"Database: {db_path}")
print(f"\nThese overlap-aware transitions can now be used by h99-UI-mayascipt-V1.py")
print(f"via the [overlap{overlap_pct}] keys in deformation_path_db.json.")

# ===================================================================
# VISUALIZATION: Inspect the overlap correction offset
# ===================================================================
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 11

print(f"\n--- Generating visualization ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# ---- PANEL 1: Cumulative deformation norm (original vs blended) ----
ax1 = axes[0]

# Original (no overlap): accumulate per-step deformation norm
original_flat = []
for phase in action_lists_old:
    original_flat.extend(phase)

# Subsample for plotting (every 100th step)
subsample = max(1, len(original_flat) // 500)

# Original cumulative
cum_orig = np.zeros(len(original_flat[0]))
norms_orig = []
steps_orig = []
for s, action in enumerate(original_flat):
    cum_orig += np.array(action)
    if s % subsample == 0:
        norms_orig.append(np.linalg.norm(cum_orig))
        steps_orig.append(s)

# Blended cumulative
blended_flat = []
for phase in split_action_lists:
    blended_flat.extend(phase)

cum_blend = np.zeros(len(blended_flat[0]))
norms_blend = []
steps_blend = []
for s, action in enumerate(blended_flat):
    cum_blend += np.array(action)
    if s % subsample == 0:
        norms_blend.append(np.linalg.norm(cum_blend))
        steps_blend.append(s)

ax1.plot(steps_orig, norms_orig, color='#4A90D9', linewidth=1.2, label='Original (no overlap)', alpha=0.8)
ax1.plot(steps_blend, norms_blend, color='#E85D75', linewidth=1.2, label=f'Blended ({overlap_pct}% overlap)', alpha=0.8)

# Mark original phase boundaries
for b_idx, b_pos in enumerate(phase_boundaries_old):
    if b_pos > 0 and b_pos < len(original_flat):
        ax1.axvline(x=b_pos, color='#4A90D9', linestyle='--', alpha=0.4, linewidth=0.8)

# Mark blended split points
for b_idx, b_pos in enumerate(new_split_points):
    if b_pos > 0 and b_pos < len(blended_flat):
        ax1.axvline(x=b_pos, color='#E85D75', linestyle='--', alpha=0.4, linewidth=0.8)

# Annotate correction offset magnitudes at boundaries
for i, corr in enumerate(overlap_corrections):
    corr_mag = np.linalg.norm(corr)
    if corr_mag > 0:
        b_pos = new_split_points[i + 1]
        ax1.annotate(
            f'correction\n|c|={corr_mag:.1f}',
            xy=(b_pos, norms_blend[min(len(norms_blend)-1, b_pos // subsample)]),
            xytext=(b_pos + len(blended_flat)*0.05, max(norms_blend)*0.85),
            fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFACD', edgecolor='#333333', alpha=0.9)
        )

ax1.set_xlabel('Optimization Step')
ax1.set_ylabel('||Δb|| (Cumulative Deformation Norm)')
ax1.set_title('Cumulative Deformation: Original vs Overlapped')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.2)

# ---- PANEL 2: Per-tissue correction offset breakdown ----
ax2 = axes[1]

n_lm = 300 * 3  # 900
tissue_names = ['OOR', 'Jaw', 'OOC-L', 'OOC-R', 'Frontalis']
tissue_slices = [(0, n_lm), (n_lm, n_lm+3), (n_lm+3, 2*n_lm+3), (2*n_lm+3, 3*n_lm+3), (3*n_lm+3, 4*n_lm+3)]
colors = ['#4A90D9', '#50C878', '#E85D75', '#F5A623', '#9B59B6']

x_positions = np.arange(len(tissue_names))
bar_width = 0.25

for corr_idx, corr in enumerate(overlap_corrections):
    corr_arr = np.array(corr)
    tissue_mags = []
    for start, end in tissue_slices:
        if end <= len(corr_arr):
            tissue_mags.append(np.linalg.norm(corr_arr[start:end]))
        else:
            tissue_mags.append(0)
    
    offset = (corr_idx - len(overlap_corrections)/2 + 0.5) * bar_width
    bars = ax2.bar(x_positions + offset, tissue_mags, bar_width,
                   label=f'Boundary {corr_idx+1}', color=colors[corr_idx % len(colors)], alpha=0.8,
                   edgecolor='#333333', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, tissue_mags):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)

ax2.set_xticks(x_positions)
ax2.set_xticklabels(tissue_names)
ax2.set_ylabel('||correction|| per tissue group')
ax2.set_title('Correction Offset by Tissue Group')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2, axis='y')

# ---- PANEL 3: Remaining L2 vs Absorbed L3 decomposition ----
ax3 = axes[2]

# Re-compute the L2/L3 halves for visualization
# We need to re-extract from original data
orig_flat = list(original_flat)
cum_offset_viz = 0

boundary_labels = []
remaining_mags = []
absorbed_mags = []
correction_mags = []

for b_idx in range(1, len(phase_boundaries_old) - 1):
    B = phase_boundaries_old[b_idx] - cum_offset_viz
    back_len = phase_boundaries_old[b_idx + 1] - phase_boundaries_old[b_idx]
    front_len = phase_boundaries_old[b_idx] - phase_boundaries_old[b_idx - 1]
    W = int(min(front_len, back_len) * overlap_ratio)
    mid = W // 2
    
    if W == 0:
        continue
    
    l2_viz = orig_flat[B - W:B]
    l3_viz = orig_flat[B:B + W]
    
    # Remaining L2 (second half)
    rem = np.zeros(len(l2_viz[0]))
    for a in l2_viz[mid:]:
        rem += np.array(a)
    
    # Absorbed L3 (first half)
    ab = np.zeros(len(l3_viz[0]))
    for a in l3_viz[:mid]:
        ab += np.array(a)
    
    corr = rem - ab
    
    boundary_labels.append(f'B{b_idx}\n({phonemes[b_idx-1]}→{phonemes[b_idx]}|{phonemes[b_idx]}→{phonemes[b_idx+1]})')
    remaining_mags.append(np.linalg.norm(rem))
    absorbed_mags.append(np.linalg.norm(ab))
    correction_mags.append(np.linalg.norm(corr))
    
    # Shift for next boundary
    # Blend removes W steps
    blended_viz = []
    for a, b in zip(l2_viz, l3_viz):
        blended_viz.append([x+y for x,y in zip(a,b)])
    orig_flat = orig_flat[:B-W] + blended_viz + orig_flat[B+W:]
    cum_offset_viz += W

x_pos = np.arange(len(boundary_labels))
bar_w = 0.22

bars1 = ax3.bar(x_pos - bar_w, remaining_mags, bar_w, label='||remaining L2|| (unfinished)',
                color='#4A90D9', edgecolor='#333333', linewidth=0.5, alpha=0.85)
bars2 = ax3.bar(x_pos, absorbed_mags, bar_w, label='||absorbed L3|| (premature)',
                color='#E85D75', edgecolor='#333333', linewidth=0.5, alpha=0.85)
bars3 = ax3.bar(x_pos + bar_w, correction_mags, bar_w, label='||correction|| = ||L2 − L3||',
                color='#50C878', edgecolor='#333333', linewidth=0.5, alpha=0.85)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8)

ax3.set_xticks(x_pos)
ax3.set_xticklabels(boundary_labels, fontsize=8)
ax3.set_ylabel('Magnitude')
ax3.set_title('Overlap Correction Decomposition')
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.2, axis='y')

plt.suptitle(f'Overlap Correction Inspection — {" → ".join(phonemes)} — ρ={overlap_ratio}',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

# Save figure
fig_path = os.path.join(log_path_output, "overlap_correction_visualization.png")
plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nVisualization saved: {fig_path}")

plt.show()

print(f"\nCode execution completed successfully!")