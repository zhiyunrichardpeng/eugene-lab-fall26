# Dissipating Error Offset (Feedback Control Loop) Implementation

## Overview

This document describes the implementation of a **Feedback Control Loop** (Dissipating Error Offset) in the facial animation system to correct accumulated drift when chaining multiple phoneme transitions.

## The Problem: Open Loop Drift

Previously, the system operated as an **"Open Loop"** system:
- Pre-computed transitions were applied blindly without checking actual positions
- Each transition added ~10% error
- With 15 transitions, accumulated drift became significant
- Example: Neutral → Y → UW → V → N → ... (15 transitions) resulted in large positional errors

## The Solution: Feedback Control Loop

The new system implements a **Closed Loop** with error feedback:
- After each transition, we measure the error between:
  - **Actual position** (where we ended up)
  - **Ideal position** (where we should be if starting from perfect neutral)
- This error is **dissipated** (partially corrected) in the next transition
- Maintains the physical characteristics (velocity, momentum, collisions) while correcting drift

## How It Works

### Experiment Design (3 Transitions)

The implementation follows this experimental design:

**Transition 1: A → B (Neutral → phoneme_Y)**
- Use current code (no correction needed, this is the baseline)
- A = Neutral pose
- delta_AB applied → arrives at B_hat
- B_tilde = ideal B position (from neutral-to-Y deformation)
- In theory: B_hat should equal B_tilde
- error_AB = B_tilde - B_hat (measured but not corrected yet)

**Transition 2: B → C (phoneme_Y → phoneme_UW)**
- Start from B_hat (actual position after first transition)
- Apply delta_BC → arrives at C_hat (without correction)
- BUT: Before applying delta_BC, we correct part of error_AB
- Corrected start position = B_hat + (error_AB × dissipation_rate)
- C_tilde = ideal C position (from neutral-to-UW deformation)
- error_BC = C_tilde - C_hat (new error)
- accumulated_error_BC = (1 - dissipation_rate) × error_AB + error_BC

**Transition 3: C → D (phoneme_UW → phoneme_V)**
- Start from C_hat
- Before applying delta_CD, correct part of accumulated_error_BC
- Corrected start = C_hat + (accumulated_error_BC × dissipation_rate)
- Apply delta_CD → arrives at D_prime
- D_tilde = ideal D position (from neutral-to-V deformation)
- error_CD = D_tilde - D_prime
- accumulated_error_CD = (1 - dissipation_rate) × accumulated_error_BC + error_CD

This error continues to be dissipated in subsequent transitions.

## Key Components

### 1. Configuration Parameters

Located at the top of `h99-UI-mayascipt-V1.py`:

```python
# --- FEEDBACK CONTROL CONFIGURATION ---
ENABLE_FEEDBACK_CONTROL = True  # Enable/disable feedback control loop
ERROR_DISSIPATION_RATE = 0.5  # How much of the error to apply (0.0 to 1.0)

ACCUMULATED_ERROR = None  # Global error accumulator
```

**ERROR_DISSIPATION_RATE** controls the feedback strength:
- **0.5** (default): 50% of error corrected per transition, 50% persists
- **1.0**: 100% correction (aggressive, may cause instability)
- **0.0**: No correction (disabled)
- **0.3-0.7**: Recommended range for smooth, stable correction

### 2. Core Functions

#### `load_ideal_target_positions(target_phoneme)`
- Loads "ground truth" positions from precomputed neutral-to-target deformations
- Reads from `deltaB_temp.pkl` files in the deformation database
- Returns ideal positions for: oor, ooc_l, ooc_r, frontalis landmarks, and jaw rotation

#### `compute_position_error(actual_positions, ideal_positions)`
- Computes the error vector: `error = ideal - actual`
- Returns error for all landmark groups
- Error is a vector (same dimensionality as positions)

#### `apply_dissipating_error(positions, accumulated_error, dissipation_rate)`
- Applies error correction to positions
- Formula: `corrected_position = current_position + (accumulated_error × dissipation_rate)`
- Returns corrected positions

#### `run_single_transition()` (Modified)
- **Before transition**: Applies accumulated error from previous transition
  - Reads current landmark positions
  - Applies error correction
  - Updates Maya scene with corrected positions
- **Performs transition**: Runs normal deformation
- **After transition**: Computes new error for next transition
  - Reads actual final positions
  - Loads ideal positions for target phoneme
  - Computes new error
  - Updates accumulated error: `(1 - rate) × old_error + new_error`

### 3. UI Controls

The updated UI (Expression Automator v4) includes:

1. **Enable feedback control loop** checkbox
   - Toggle feedback control on/off
   - Default: ON

2. **Error dissipation rate** slider (0.0 to 1.0)
   - Adjust correction strength
   - Default: 0.5
   - Live update during configuration

3. **Status display**
   - Shows current feedback control state
   - Displays error magnitude in console logs

## Usage

### Running the Example Sequence

To test with "Neutral to phoneme_Y to phoneme_UW to phoneme_V":

1. Open Maya with the rest pose scene
2. Run `h99-UI-mayascipt-V1.py` in Maya Script Editor
3. In the UI:
   - Enter: `neutral to phoneme_Y to phoneme_UW to phoneme_V`
   - Check **Enable feedback control loop** ✓
   - Set **Error dissipation rate** to 0.5
   - Click **Generate Animation**

### Understanding the Console Output

The system logs detailed information:

```
[FEEDBACK CONTROL] Loading ideal target positions for dataV22_phoneme_Y
[FEEDBACK CONTROL] Loaded 900 oor landmark coordinates

[FEEDBACK CONTROL] Applying accumulated error from previous transition
  Dissipation rate: 0.5
[FEEDBACK CONTROL] Applying error correction to landmarks at frame 50
[FEEDBACK CONTROL] Error correction applied

[FEEDBACK CONTROL] Computing error for next transition
[FEEDBACK CONTROL] Accumulated error magnitude (oor): 0.123456
[FEEDBACK CONTROL] Error will be applied in next transition with dissipation rate 0.5
```

### Comparing With and Without Feedback Control

To compare results:

1. **Without feedback control**:
   - Uncheck "Enable feedback control loop"
   - Run sequence
   - Note final positions

2. **With feedback control**:
   - Check "Enable feedback control loop"
   - Reset (to start from neutral)
   - Run same sequence
   - Compare final positions

The feedback control version should show significantly less drift, especially visible in the final phoneme pose.

## Mathematical Formulation

### Error Accumulation

At each transition `i`:

```
error[i] = ideal_position[i] - actual_position[i]

accumulated_error[i] = (1 - α) × accumulated_error[i-1] + error[i]
```

Where `α = ERROR_DISSIPATION_RATE`

### Error Correction

Before applying transition `i`:

```
corrected_start[i] = actual_position[i-1] + α × accumulated_error[i-1]
```

Then apply delta:

```
actual_position[i] = corrected_start[i] + delta[i]
```

### Dissipation Over Time

The residual error from transition 0 after N transitions:

```
residual_error[N] = error[0] × (1 - α)^N
```

Example with α = 0.5:
- After 1 transition: 50% remains
- After 2 transitions: 25% remains
- After 3 transitions: 12.5% remains
- After 4 transitions: 6.25% remains

This shows the error **exponentially decays** while still maintaining some influence for smooth correction.

## Implementation Details

### Data Flow

1. **Initialization** (first transition):
   - `ACCUMULATED_ERROR = None`
   - No correction applied (is_first_transition=True)

2. **After first transition**:
   - Compute error_AB = ideal_B - actual_B
   - Store in `ACCUMULATED_ERROR`

3. **Before second transition**:
   - Read current position (B_hat)
   - Apply correction: B_corrected = B_hat + error_AB × 0.5
   - Update Maya landmarks to B_corrected
   - Apply delta_BC from B_corrected

4. **After second transition**:
   - Compute error_BC = ideal_C - actual_C
   - Update: `ACCUMULATED_ERROR = 0.5 × error_AB + error_BC`

5. **Repeat** for all subsequent transitions

### Landmark Groups Corrected

The system corrects all facial landmark groups:
- **oor**: 300 landmarks (oral orbicularis)
- **ooc_l**: 300 landmarks (left eye orbicularis)
- **ooc_r**: 300 landmarks (right eye orbicularis)
- **frontalis**: 300 landmarks (forehead)
- **jaw rotation**: 3 rotation values

Total: 3603 parameters corrected per transition

### File Dependencies

The system requires precomputed deformations in the database:
- `automation/deformation_path_db.json`: Maps phoneme transitions to file paths
- Each deformation folder contains:
  - `deltaB_temp.pkl`: Final state with ideal positions
  - `final_epoch_action_list.pkl`: Transition deltas

Example path structure:
```
results/DSAC_V2_m1_vertexMove_Va003/
├── job_251206-053234/251206-053242/  # Neutral → phoneme_Y
│   └── deltaB_temp.pkl
├── job_251205-230610/251205-230619/  # Neutral → phoneme_UW
│   └── deltaB_temp.pkl
└── job_251206-021708/251206-021716/  # Neutral → phoneme_V
    └── deltaB_temp.pkl
```

## Troubleshooting

### "Could not find neutral-to-X in database"

**Cause**: The ideal position file for a phoneme is not in the database.

**Solution**: Ensure precomputed deformations exist for all target phonemes:
- Check `deformation_path_db.json` contains entries like:
  - `"dataV22_AU0_neutral to dataV22_phoneme_Y"`
  - `"dataV22_AU0_neutral to dataV22_phoneme_UW"`
  - etc.

### "deltaB_temp.pkl not found"

**Cause**: The deformation folder exists but doesn't contain the final state file.

**Solution**: Re-run the deformation training for that phoneme to generate the file.

### Large error magnitude (> 10.0)

**Possible causes**:
1. Dissipation rate too low (errors accumulating)
2. Initial transitions had very large errors
3. Ideal positions don't match current rig configuration

**Solutions**:
- Increase `ERROR_DISSIPATION_RATE` to 0.6-0.8 for more aggressive correction
- Check that neutral pose in Maya matches the trained neutral pose
- Verify ideal position files are from the same rig version

### Unstable oscillation

**Cause**: Dissipation rate too high (overcorrection).

**Solution**: Reduce `ERROR_DISSIPATION_RATE` to 0.3-0.4 for smoother convergence.

## Future Enhancements

Potential improvements:

1. **Adaptive dissipation rate**: Adjust based on error magnitude
2. **Per-landmark weights**: Correct facial regions differently
3. **Error visualization**: Real-time error heatmap display
4. **Error logging**: Save error history to file for analysis
5. **Predictive correction**: Anticipate drift based on transition history

## Technical Notes

### Why Not 100% Correction?

Setting `ERROR_DISSIPATION_RATE = 1.0` would apply full correction each step. However:
- May cause sudden jerky movements
- Doesn't preserve momentum/dynamics
- Can lead to oscillation if ideal positions have small errors
- The residual error provides smooth blending

### Robotics Context

This technique is common in robotics:
- **PID Control**: Proportional-Integral-Derivative feedback
- **Kalman Filtering**: Optimal error estimation
- **Model Predictive Control**: Future error prediction

Our implementation is a simplified **Proportional Controller** with exponential decay.

## References

- Original implementation: `h99-UI-mayascipt-V1.py` (lines 30-39, 168-242, 296-351, 1508-1666)
- Database configuration: `automation/deformation_path_db.json`
- Deformation training: `a3-stochastic-V1-algo/training/`

## Version History

- **v4.0**: Added feedback control loop (current)
- **v3.0**: Multi-phoneme chaining
- **v2.0**: Single transition automation
- **v1.0**: Manual deformation application
