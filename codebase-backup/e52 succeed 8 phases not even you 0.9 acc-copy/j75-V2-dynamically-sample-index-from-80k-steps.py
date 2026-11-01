import numpy as np

# Timestamp data (in seconds)
timestamps = [0, 0.23, 0.52, 0.67, 0.89, 0.98, 1.14, 1.16, 1.68]
phonemes = ["Neutral", "N", "AA", "T", "IY", "V", "N", "Y", "UW"]
total_duration = timestamps[-1]  # 1.68 seconds

# Calculate phase durations and weights
phase_durations = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
phase_weights = [duration/total_duration for duration in phase_durations]

# Input parameters (accumulated indices)
must_include_indices = [
    0,                          # Start
    36782,                      # N
    36782 + 21999,              # AA
    36782 + 21999 + 29932,      # T
    36782 + 21999 + 29932 + 19384,   # IY
    36782 + 21999 + 29932 + 19384 + 54552,  # V
    36782 + 21999 + 29932 + 19384 + 54552 + 53856,  # N
    36782 + 21999 + 29932 + 19384 + 54552 + 53856 + 20822,  # Y
    36782 + 21999 + 29932 + 19384 + 54552 + 53856 + 20822 + 76816  # UW
]

num_checkpoints = 36
buffer_ratio = 0.1  # 10% buffer around must-include points

# Phase ranges with weights based on actual timestamps
phase_ranges = [
    # (start_idx, end_idx, weight)
    (0, 36782, phase_weights[0]),               # Neutral to N (0-0.23s)
    (36782, 36782+21999, phase_weights[1]),     # N to AA (0.23-0.52s)
    (36782+21999, 36782+21999+29932, phase_weights[2]),      # AA to T (0.52-0.67s)
    (36782+21999+29932, 36782+21999+29932+19384, phase_weights[3]),  # T to IY (0.67-0.89s)
    (36782+21999+29932+19384, 36782+21999+29932+19384+54552, phase_weights[4]),  # IY to V (0.89-0.98s)
    (36782+21999+29932+19384+54552, 36782+21999+29932+19384+54552+53856, phase_weights[5]),  # V to N (0.98-1.14s)
    (36782+21999+29932+19384+54552+53856, 36782+21999+29932+19384+54552+53856+20822, phase_weights[6]),  # N to Y (1.14-1.16s)
    (36782+21999+29932+19384+54552+53856+20822, 36782+21999+29932+19384+54552+53856+20822+76816, phase_weights[7])  # Y to UW (1.16-1.68s)
]

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