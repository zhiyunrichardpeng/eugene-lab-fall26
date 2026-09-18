# Debug System for Dissipating Error Offset

## Overview

This debugging system tracks and visualizes the errors in the feedback control loop to help diagnose why drift persists even with `ERROR_DISSIPATION_RATE = 1.0`.

## What is Measured

For each transition (e.g., Neutral → N, N → IY, IY → T, etc.):

### 1. Landmark Distances (Actual vs Ideal)

**What it measures**: Distance between actual landmark positions (B_hat, C_hat, D_hat) and ideal positions (B_tilde, C_tilde, D_tilde) from precomputed neutral-to-target deformations.

- **Mean Distance**: Average Euclidean distance across all 300 oor landmarks
- **Max Distance**: Maximum distance among all landmarks
- **Total Error**: L2 norm of all position differences

**Purpose**: Shows how far the actual positions deviate from where they should be if starting from perfect neutral.

### 2. Vertex Distances (Actual vs Ideal) - OOR Tissue

**What it measures**: Estimated distance between actual and ideal vertex positions for the OOR (oral orbicularis) tissue mesh.

- Currently **estimated** from landmark distances (×1.2 factor)
- Full implementation would require loading ground truth scenes

**Purpose**: Provides tissue-level error measurement to complement landmark-level errors.

### 3. Vertex Distances (Actual vs Ground Truth)

**What it measures**: Distance between actual vertex positions and ground truth from original training scenes.

- Currently **placeholder** (requires loading original Maya scenes)
- Would show true positional accuracy vs training data

**Purpose**: Ultimate validation - how close are we to the original target poses?

### 4. Error Magnitude

**What it measures**: L2 norm of the error vector between actual and ideal landmark positions.

**Formula**: `error_magnitude = ||ideal_positions - actual_positions||`

**Purpose**: Single scalar value representing total error for this transition.

### 5. Accumulated Error Magnitude

**What it measures**: L2 norm of the accumulated error that will be applied in the next transition.

**Formula**: `accumulated_error = (1 - dissipation_rate) × previous_error + new_error`

**Purpose**: Shows how error propagates across transitions with dissipation.

## Output Files

All files are saved to:
```
C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\d4 interactive interface of animation\h96-debug-dissipating-error\
```

### CSV File: `debug_errors_<timestamp>.csv`

Contains detailed measurements for each transition:

| Column | Description |
|--------|-------------|
| transition | Transition number (1, 2, 3, ...) |
| phoneme_from | Starting phoneme |
| phoneme_to | Target phoneme |
| frame | Final frame number after transition |
| landmark_mean_hat_tilde | Mean landmark distance (actual vs ideal) |
| landmark_max_hat_tilde | Max landmark distance (actual vs ideal) |
| landmark_total_hat_tilde | Total landmark error magnitude |
| vertex_mean_hat_tilde | Mean vertex distance (estimated) |
| vertex_max_hat_tilde | Max vertex distance (estimated) |
| vertex_total_hat_tilde | Total vertex error (estimated) |
| vertex_mean_hat_ground | Mean vertex distance vs ground truth (placeholder) |
| vertex_max_hat_ground | Max vertex distance vs ground truth (placeholder) |
| vertex_total_hat_ground | Total vertex error vs ground truth (placeholder) |
| error_magnitude | Current error magnitude |
| accumulated_error_magnitude | Accumulated error for next transition |
| correction_applied | Whether error correction was applied (True/False) |
| dissipation_rate | Dissipation rate used (0.0 to 1.0) |

### Visualization: `error_analysis_<timestamp>.png`

A 3×2 grid of plots:

1. **Landmark Distances (Actual vs Ideal)**
   - Mean and max distances across transitions
   - Shows if landmarks are drifting from ideal positions

2. **Vertex Distances (Actual vs Ideal) - OOR Tissue**
   - Mean and max vertex distances
   - Tissue-level drift measurement

3. **Error Magnitude per Transition**
   - New error introduced at each step
   - Should decrease if feedback control is working

4. **Accumulated Error Over Time**
   - Total accumulated error across sequence
   - Should stabilize or decrease with dissipation

5. **All Metrics Comparison**
   - Overlay of landmark mean, vertex mean, and error magnitude
   - Shows relative scales and trends

6. **Vertex Distances (Actual vs Ground Truth)**
   - Ultimate accuracy measurement
   - Placeholder until ground truth loading is implemented

## How to Use

### Running the Debug System

1. Open Maya with your rest pose scene
2. Run `h99-UI-mayascipt-V1.py` in Maya Script Editor
3. In the UI:
   - Enter your test sequence (e.g., `neutral to phoneme_N to phoneme_IY to phoneme_T to phoneme_Y to phoneme_EH to phoneme_T`)
   - Enable feedback control ✓
   - Set dissipation rate (e.g., 1.0)
   - Click **Generate Animation**

4. After completion, check the debug output folder for:
   - CSV file with detailed measurements
   - PNG visualization with 6 plots

### Recommended Test Sequence

```
neutral to phoneme_N to phoneme_IY to phoneme_T to phoneme_Y to phoneme_EH to phoneme_T
```

This creates 6 transitions:
1. Neutral → N (baseline, no correction)
2. N → IY (first correction applied)
3. IY → T
4. T → Y
5. Y → EH
6. EH → T

## Interpreting the Results

### Expected Behavior with Dissipation Rate = 1.0

With `ERROR_DISSIPATION_RATE = 1.0`, the system should apply **100% correction** each step:

**Expected**:
- **Error magnitude**: Should be non-zero but should NOT grow unbounded
- **Accumulated error**: Should equal current error (no residual since 1.0 means full correction)
- **Landmark distances**: Should stabilize or improve over transitions

**If drift still occurs**, possible causes:

1. **Correction is applied incorrectly**
   - Check that landmarks are actually moved before the transition
   - Verify correction magnitude matches error magnitude

2. **New error exceeds corrected error**
   - Each transition introduces new error larger than what was corrected
   - Net result: error still accumulates despite 100% correction

3. **Ideal positions are incorrect**
   - The "tilde" (ideal) positions from neutral-to-target may not match chained reality
   - Pre-computed transitions assume perfect neutral start, but corrections change dynamics

4. **Correction timing is wrong**
   - Correction should be applied **before** delta_BC, not after
   - Check order of operations in `run_single_transition`

### Diagnostic Checks

#### Check 1: Is correction actually applied?

Look at CSV column `correction_applied`:
- Transition 1: Should be `False` (no previous error)
- Transitions 2+: Should be `True`

If all `False`, the correction logic isn't triggering.

#### Check 2: Does error magnitude increase?

Plot the `error_magnitude` column:
- **Constant**: New error ~equals corrected error (steady state)
- **Increasing**: New error > corrected error (system diverging)
- **Decreasing**: Correction is working (system converging)

#### Check 3: Accumulated vs Current Error

Compare `error_magnitude` vs `accumulated_error_magnitude`:
- With rate=1.0: `accumulated_error ≈ current_error` (no residual)
- If `accumulated_error >> current_error`: Bug in accumulation formula

#### Check 4: Landmark distances vs Error magnitude

Both should show similar trends:
- If `landmark_mean_hat_tilde` increases but `error_magnitude` is constant: Landmarks drifting from ideal
- If both increase: Fundamental drift issue in the transition chain

## Implementation Notes

### Current Limitations

1. **Vertex distances are estimated**
   - Multiplied from landmark distances (×1.2 factor)
   - Full implementation requires loading ground truth scenes at each phoneme

2. **Ground truth comparison not implemented**
   - Requires opening original training Maya scenes
   - Would need `maya.standalone` to load scenes in batch

3. **Only OOR tissue measured**
   - Other tissues (ooc_l, ooc_r, frontalis) not included in vertex tracking
   - Could be added if needed

### Future Enhancements

1. **Load ground truth vertices**
   ```python
   # Pseudocode
   ground_truth_scene = load_scene(phoneme_to)
   ground_truth_vertices = get_vertices(ground_truth_scene, 'oor')
   vertex_dist = compute_distance(actual_vertices, ground_truth_vertices)
   ```

2. **Per-landmark error heatmap**
   - Save per-landmark errors to identify problematic regions
   - Visualize as 3D scatter plot with error magnitude as color

3. **Velocity and acceleration tracking**
   - Track rate of error change
   - Identify sudden jumps vs gradual drift

4. **Correction effectiveness metric**
   - `effectiveness = (error_before - error_after) / error_before`
   - Should be ~1.0 with dissipation_rate=1.0

## Troubleshooting

### "Mesh 'oor' not found"

**Cause**: OOR tissue mesh doesn't exist in scene.

**Solution**: Ensure Maya scene has `oor` mesh object. Check mesh name with `cmds.ls(type='mesh')`.

### CSV file empty or has NaN values

**Cause**: Distance computation failed (e.g., length mismatch).

**Solution**: Check console for `[DEBUG WARNING]` messages showing which computation failed.

### Visualization shows all zeros

**Cause**: Debug data wasn't collected (DEBUG_ENABLED=False or no transitions ran).

**Solution**:
1. Check `DEBUG_ENABLED = True` in script
2. Ensure multi-phoneme sequence ran (3+ phonemes)

### "Could not load ideal positions"

**Cause**: Precomputed neutral-to-phoneme file doesn't exist in database.

**Solution**: Ensure `automation/deformation_path_db.json` has entry for each phoneme in your sequence.

## Example Console Output

```
[DEBUG] === Detailed Measurements for dataV22_phoneme_N → dataV22_phoneme_IY ===
[DEBUG] Landmark Distance (Actual vs Ideal):
  Mean: 0.042356, Max: 0.135789, Total: 2.316547
[DEBUG] Vertex Distance (Actual vs Ideal): (Estimated from landmarks)
  Mean: 0.050827, Max: 0.162947, Total: 2.779856
[FEEDBACK CONTROL] Current error magnitude (oor): 2.316547
[FEEDBACK CONTROL] Accumulated error magnitude (oor): 2.316547
[FEEDBACK CONTROL] Error will be applied in next transition with dissipation rate 1.0
[DEBUG] Debug entry saved (transition #2)
```

## Key Questions the Debug System Answers

1. **Is the error increasing or decreasing over transitions?**
   → Check error_magnitude plot

2. **Is the correction being applied?**
   → Check correction_applied column in CSV

3. **How much do landmarks deviate from ideal positions?**
   → Check landmark_mean_hat_tilde values

4. **Does accumulated error stabilize or grow?**
   → Check accumulated_error_magnitude plot

5. **Which transition introduces the most error?**
   → Find peak in error_magnitude plot

## Related Files

- **Main script**: `h99-UI-mayascipt-V1.py` (lines 174-419: debug functions, 1868-1989: debug integration)
- **Feedback control docs**: `FEEDBACK_CONTROL_README.md`
- **Database**: `automation/deformation_path_db.json`

## Version

Debug system version 1.0 - Added 2025-12-16
