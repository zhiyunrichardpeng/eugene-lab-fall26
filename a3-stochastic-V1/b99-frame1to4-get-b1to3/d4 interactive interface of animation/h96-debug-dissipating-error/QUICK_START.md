# Quick Start Guide - Debug System

## Running the Test

### 1. Set the Test Sequence

In Maya Script Editor, run the modified `h99-UI-mayascipt-V1.py` with:

```
Test Sequence: neutral to phoneme_N to phoneme_IY to phoneme_T to phoneme_Y to phoneme_EH to phoneme_T
Dissipation Rate: 1.0 (already set in code)
Debug Enabled: True (already set)
```

### 2. Execute

1. Open Maya with rest pose scene
2. Run the script
3. Enter the sequence in the UI
4. Click "Generate Animation"

### 3. Check Results

After completion, navigate to:
```
C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\d4 interactive interface of animation\h96-debug-dissipating-error\
```

You'll find:
- `debug_errors_<timestamp>.csv` - Detailed measurements
- `error_analysis_<timestamp>.png` - 6 visualization plots

## What to Look For

### CSV Analysis

Open the CSV file and look at these columns:

| What to Check | Column Name | Expected with Rate=1.0 | If Wrong |
|---------------|-------------|------------------------|----------|
| Correction applied? | correction_applied | TRUE for transitions 2+ | Check logic in line 1807 |
| Error growing? | error_magnitude | Stable or decreasing | New error > correction |
| Accumulated error? | accumulated_error_magnitude | ≈ error_magnitude | Bug in line 1964 formula |
| Landmark drift? | landmark_mean_hat_tilde | < 0.1 units | Large drift from ideal |

### Visualization Analysis

**Plot 1 (Top Left) - Landmark Distances**:
- Should show actual vs ideal landmark distances
- If increasing: Landmarks drifting away from ideal positions

**Plot 3 (Middle Left) - Error Magnitude**:
- Most critical plot!
- Should be flat or decreasing
- If increasing: System is diverging

**Plot 4 (Middle Right) - Accumulated Error**:
- With rate=1.0, should match Plot 3
- If much larger: Residual error not dissipating

## Quick Diagnosis

### Scenario 1: Error Increases Despite 100% Correction

**Symptoms**:
- `error_magnitude` increases over transitions
- `correction_applied` = TRUE
- `accumulated_error_magnitude` ≈ `error_magnitude`

**Diagnosis**: **New error introduced by each transition exceeds correction**

**Why**: Pre-computed transitions (delta_BC, delta_CD) were generated assuming perfect neutral start. When you apply correction before transition, you change the starting conditions, which makes the pre-computed delta inappropriate.

**Solution**: Need to either:
1. Re-compute transitions on-the-fly from corrected positions (expensive)
2. Use predictive correction that anticipates the mismatch
3. Accept that chained transitions will always drift from pre-computed paths

### Scenario 2: Correction Not Applied

**Symptoms**:
- `correction_applied` = FALSE for all transitions
- `error_magnitude` continuously increases

**Diagnosis**: **Correction logic not triggering**

**Check**:
- Line 1807: `if ENABLE_FEEDBACK_CONTROL and ACCUMULATED_ERROR is not None and not is_first_transition:`
- Is `ACCUMULATED_ERROR` actually set after first transition?

### Scenario 3: Correction Applied But Ineffective

**Symptoms**:
- `correction_applied` = TRUE
- Landmarks moved (check console for "Applying error correction")
- But `error_magnitude` stays constant or increases

**Diagnosis**: **Correction magnitude too small OR new error too large**

**Check**:
- Print the correction magnitude: `np.linalg.norm(correction)`
- Compare to `error_magnitude`
- If correction >> error but error still increases: New error is being introduced by the transition itself

## Console Output to Watch

Look for these in Maya console:

```
[FEEDBACK CONTROL] Applying accumulated error from previous transition
  Dissipation rate: 1.0
[FEEDBACK CONTROL] Applying error correction to landmarks at frame 50
[FEEDBACK CONTROL] Error correction applied

[FEEDBACK CONTROL] Current error magnitude (oor): 2.316547
[FEEDBACK CONTROL] Accumulated error magnitude (oor): 2.316547
```

**Key numbers**:
1. Current error should be measured AFTER applying correction
2. If current error ≈ previous accumulated error: Correction had no effect!
3. If current error < previous accumulated error: Correction is working partially

## Root Cause Investigation

### Hypothesis 1: Pre-computed Deltas Don't Match Corrected State

**Test**:
1. Look at `landmark_mean_hat_tilde` for transition 1 (no correction)
2. Compare to transition 2 (with correction)
3. If transition 2 error > transition 1 error: Correction is making things worse!

**Why**: Delta_BC was computed assuming start from B_tilde (ideal B), but after correction we start from B_hat + correction, which is different.

### Hypothesis 2: Ideal Positions (B_tilde) Are Wrong

**Test**:
1. Manually load neutral scene
2. Apply neutral→N deformation
3. Compare final positions to N_tilde from database
4. If they don't match: Database is stale or from different rig version

### Hypothesis 3: Error Computation Is Wrong

**Test**:
1. Print both `actual_flat['x_joint_conca']` and `ideal_positions['x_joint_conca']`
2. Manually compute: `error = ideal - actual`
3. Check if signs are correct (should point FROM actual TO ideal)

## Next Steps After Analysis

### If CSV shows error is increasing:

1. **Check console logs** for correction magnitude
2. **Verify** that landmarks actually move during correction (add print statements)
3. **Compare** error before and after applying delta
4. **Hypothesis**: Pre-computed deltas don't work with corrected start positions

### If CSV shows correction not applied:

1. **Check** `ACCUMULATED_ERROR` is set after first transition
2. **Verify** `is_first_transition` flag is correct
3. **Debug** line 1807 condition

### If CSV shows correction applied but ineffective:

1. **Measure** correction magnitude vs error magnitude
2. **Check** if new error is larger than correction
3. **Consider** that chained pre-computed transitions fundamentally cannot match ideal paths

## Files Modified

- `h99-UI-mayascipt-V1.py`:
  - Lines 41-44: Debug configuration
  - Lines 174-263: Distance computation functions
  - Lines 268-418: CSV and visualization functions
  - Lines 1868-1989: Debug integration in run_single_transition
  - Lines 2066-2075: Save debug data after completion

## Support

For detailed documentation, see:
- `README_DEBUG.md` - Full debug system documentation
- `FEEDBACK_CONTROL_README.md` - Feedback control explanation

## Expected Output Example

```
Transition 1: Neutral → N
  landmark_mean: 0.023, error_magnitude: 1.234, correction_applied: FALSE

Transition 2: N → IY
  landmark_mean: 0.045, error_magnitude: 2.456, correction_applied: TRUE
  (Error increased despite correction!)

Transition 3: IY → T
  landmark_mean: 0.067, error_magnitude: 3.789, correction_applied: TRUE
  (Error continues to increase)
```

This pattern suggests: **New error from each transition > correction from previous transition**
