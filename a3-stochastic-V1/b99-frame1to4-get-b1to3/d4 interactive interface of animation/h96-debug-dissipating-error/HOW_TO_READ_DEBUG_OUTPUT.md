# How to Read Debug Output - Complete Guide

## Table of Contents
1. [Understanding the Notation System](#understanding-the-notation-system)
2. [CSV Column Definitions](#csv-column-definitions)
3. [How to Interpret the Data](#how-to-interpret-the-data)
4. [Visualization Plots Explained](#visualization-plots-explained)
5. [Common Analysis Scenarios](#common-analysis-scenarios)
6. [Step-by-Step Analysis Workflow](#step-by-step-analysis-workflow)

---

## Understanding the Notation System

### The Three Key Symbols: Hat (^), Tilde (~), Prime (')

The debug system uses standard mathematical notation to distinguish between different types of positions:

#### **Hat (^)** - Actual Simulated Positions
- **Symbol**: `B_hat`, `C_hat`, `D_hat`
- **Meaning**: The **actual** positions that result from the physics simulation
- **Example**: After simulating neutral→N transition, the landmarks end up at `N_hat`
- **In CSV**: Columns with `_hat_` in the name (e.g., `landmark_mean_hat_tilde`)

**Why we care**: These are what the simulation actually produced, which may differ from what we wanted.

#### **Tilde (~)** - Ideal Target Positions
- **Symbol**: `B_tilde`, `C_tilde`, `D_tilde`
- **Meaning**: The **ideal** positions from the pre-computed neutral→target deformation
- **Source**: Computed from `x_ideal = x_neutral + delta_b_final` (from combined_checkpoints.pkl)
- **In CSV**: Columns with `_tilde` in the name (e.g., `landmark_mean_hat_tilde`)

**Why we care**: These represent the "correct" target positions that should have been reached if the simulation was perfect.

#### **Prime (')** - Corrected Positions
- **Symbol**: `D_prime`
- **Meaning**: Positions **after applying feedback correction**
- **Formula**: `D_prime = D_hat + correction`
- **In CSV**: Correction magnitude is logged in `correction_magnitude`

**Why we care**: This shows how much we're adjusting the positions to compensate for accumulated error.

### Visual Example: Transition Chain

```
Transition Sequence: Neutral → N → IY → T

Neutral (Start)
    ↓ [Simulate neutral→N]
N_hat (Actual result)         ←→  N_tilde (Ideal target)
    ↑                             ↑
    |-- Error_N = N_tilde - N_hat--|
    ↓ [Apply correction]
N_prime (Corrected position)
    ↓ [Simulate N→IY]
IY_hat (Actual result)        ←→  IY_tilde (Ideal target)
    ↑                             ↑
    |-- Error_IY = IY_tilde - IY_hat --|
```

**Key Insight**:
- If simulation was perfect: `N_hat = N_tilde` (zero error)
- In reality: `N_hat ≠ N_tilde` (drift occurs)
- Feedback control: Apply correction to get `N_prime ≈ N_tilde`

---

## CSV Column Definitions

### Basic Information
| Column | Description | Example Value |
|--------|-------------|---------------|
| `transition` | Sequential transition number (1, 2, 3...) | 1 |
| `phoneme_from` | Starting phoneme | dataV22_AU0_neutral |
| `phoneme_to` | Target phoneme | dataV22_phoneme_N |
| `frame` | Maya frame number where transition ends | 50 |

### Landmark Distances (Control Points)

**What are landmarks?**
300 control joints per muscle group (oor, ooc_l, ooc_r, frontalis) that drive the tissue simulation.

#### Actual vs Ideal (hat vs tilde)
| Column | Formula | Meaning |
|--------|---------|---------|
| `landmark_mean_hat_tilde` | `mean(\|\|hat - tilde\|\|)` | Average distance per landmark |
| `landmark_max_hat_tilde` | `max(\|\|hat - tilde\|\|)` | Worst-case landmark deviation |
| `landmark_total_hat_tilde` | `\|\|hat - tilde\|\|` (L2 norm) | Total error magnitude across all landmarks |

**How to interpret**:
- **Mean < 0.01**: Excellent alignment
- **Mean 0.01-0.05**: Good, minor drift
- **Mean 0.05-0.1**: Moderate drift, may be visible
- **Mean > 0.1**: Significant drift, definitely visible

### Vertex Distances (Tissue Surface)

**What are vertices?**
Mesh surface points of the tissue (e.g., OOR has ~3000+ vertices). These represent the visible skin surface.

#### Actual vs Ideal (hat vs tilde)
| Column | Formula | Meaning |
|--------|---------|---------|
| `vertex_mean_hat_tilde` | `mean(\|\|vertex_hat - vertex_tilde\|\|)` | Average vertex deviation |
| `vertex_max_hat_tilde` | `max(\|\|vertex_hat - vertex_tilde\|\|)` | Worst-case vertex deviation |
| `vertex_total_hat_tilde` | `\|\|vertex_hat - vertex_tilde\|\|` | Total vertex error magnitude |

**How to interpret**:
- Vertices show the **visible** impact of landmark errors
- Usually smaller than landmark distances (tissue smoothing effect)
- If vertex_mean > 0.02, the drift will be noticeable to viewers

#### Actual vs Ground Truth (hat vs ground)
| Column | Meaning |
|--------|---------|
| `vertex_mean_hat_ground` | Same as `vertex_mean_hat_tilde` (ideal = ground truth) |
| `vertex_max_hat_ground` | Same as `vertex_max_hat_tilde` |
| `vertex_total_hat_ground` | Same as `vertex_total_hat_tilde` |

**Note**: In this system, "ideal" and "ground truth" are the same (both from pre-computed neutral→target deformations).

### Error Tracking

#### Error Magnitudes
| Column | Formula | Meaning |
|--------|---------|---------|
| `error_magnitude` | `\|\|new_error\|\|` | Current error after this transition (same as `new_error_magnitude`) |
| `new_error_magnitude` | `\|\|tilde - hat\|\|` | Fresh error introduced by this transition |
| `prev_error_magnitude` | `\|\|ACCUMULATED_ERROR[previous]\|\|` | Error carried forward from previous transition |
| `residual_error_magnitude` | `\|\|(1-α)·prev_error + new_error\|\|` | Total error after dissipation formula |
| `accumulated_error_magnitude` | `\|\|ACCUMULATED_ERROR[current]\|\|` | Error that will be corrected in next transition |

**Key Relationships**:
```
new_error = ideal_positions - actual_positions  (computed AFTER transition)
residual_error = (1 - dissipation_rate) × prev_error + new_error
accumulated_error = residual_error  (stored for next transition)
```

**Critical insight**:
- If `residual_error_magnitude > prev_error_magnitude`: Error is **growing**
- If `residual_error_magnitude < prev_error_magnitude`: Error is **shrinking**
- If `residual_error_magnitude ≈ prev_error_magnitude`: Error is **stable** (equilibrium)

#### Correction Information
| Column | Meaning | Values |
|--------|---------|--------|
| `correction_applied` | Was correction applied before this transition? | TRUE / FALSE |
| `correction_magnitude` | How much correction was applied (L2 norm) | 0.0 if not applied, >0 otherwise |
| `dissipation_rate` | Error dissipation rate used | 0.0 to 1.0 |

**Dissipation Rate Explained**:
- **α = 1.0 (100%)**: Full correction, no residual error carried forward
  - `correction = accumulated_error × 1.0 = accumulated_error`
  - `residual_error = (1 - 1.0) × prev_error + new_error = new_error`
- **α = 0.5 (50%)**: Half correction, half residual
  - `correction = accumulated_error × 0.5`
  - `residual_error = 0.5 × prev_error + new_error`
- **α = 0.0 (0%)**: No correction (open loop)
  - `correction = 0`
  - `residual_error = prev_error + new_error`

---

## How to Interpret the Data

### The Feedback Control Flow

For each transition `A → B`:

**Step 1: Before Transition (Correction Phase)**
- Load accumulated error from previous transition (`A_error`)
- Apply correction: `A_prime = A_hat + (dissipation_rate × A_error)`
- Log `correction_magnitude = ||(dissipation_rate × A_error)||`

**Step 2: During Transition (Simulation Phase)**
- Simulate transition starting from `A_prime` (corrected positions)
- Simulation applies pre-computed `delta_AB` deformation
- Result: New positions `B_hat`

**Step 3: After Transition (Error Measurement Phase)**
- Load ideal target: `B_tilde = B_neutral + delta_B_final`
- Compute new error: `B_error_new = B_tilde - B_hat`
- Compute residual: `B_error_residual = (1 - α) × A_error + B_error_new`
- Store for next transition: `ACCUMULATED_ERROR = B_error_residual`

**CSV Logging**:
- `correction_applied`: TRUE (for transition 2+)
- `correction_magnitude`: From Step 1
- `new_error_magnitude`: From Step 3 (`||B_error_new||`)
- `prev_error_magnitude`: From Step 3 (`||A_error||`)
- `residual_error_magnitude`: From Step 3 (`||B_error_residual||`)
- `accumulated_error_magnitude`: From Step 3 (`||ACCUMULATED_ERROR||`)

### Reading a Single Row

**Example CSV Row**:
```csv
transition,phoneme_from,phoneme_to,frame,landmark_mean_hat_tilde,correction_magnitude,prev_error_magnitude,new_error_magnitude,residual_error_magnitude,accumulated_error_magnitude,correction_applied,dissipation_rate
3,dataV22_phoneme_IY,dataV22_phoneme_T,150,0.045,2.156,2.314,2.789,2.789,2.789,TRUE,1.0
```

**What this tells us**:

1. **Basic Info**:
   - This is the 3rd transition (IY → T)
   - Ended at frame 150

2. **Landmark Accuracy**:
   - `landmark_mean_hat_tilde = 0.045`: Average landmark is 0.045 units off target
   - **Assessment**: Moderate drift (visible but not extreme)

3. **Correction Applied**:
   - `correction_applied = TRUE`: Correction was applied before this transition
   - `correction_magnitude = 2.156`: Total correction magnitude (across 900 coordinates)
   - With 300 landmarks (900 coords), average correction per landmark = 2.156/300 ≈ 0.007 units

4. **Error Evolution**:
   - `prev_error_magnitude = 2.314`: Error from previous transition (IY)
   - **Correction removed** this error (dissipation_rate = 1.0)
   - `new_error_magnitude = 2.789`: Fresh error introduced by IY→T transition
   - `residual_error_magnitude = 2.789`: Total error after dissipation formula
   - With α=1.0: `residual = (1-1.0)×2.314 + 2.789 = 0 + 2.789 = 2.789` ✓

5. **Diagnosis**:
   - **Problem**: New error (2.789) > previous error (2.314)
   - **Meaning**: Even with 100% correction, error is **increasing**
   - **Root cause**: Pre-computed delta_IY_T assumes starting from ideal_IY, but we're starting from corrected_IY ≠ ideal_IY

### Tracking Error Over Time

**Load CSV and plot `accumulated_error_magnitude`**:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('debug_errors_20251216_143022.csv')
plt.plot(df['transition'], df['accumulated_error_magnitude'], marker='o')
plt.xlabel('Transition Number')
plt.ylabel('Accumulated Error Magnitude')
plt.title('Error Evolution Across Transitions')
plt.grid(True)
plt.show()
```

**Interpretation**:
- **Flat line**: Error is stable (new_error ≈ correction)
- **Increasing**: System is diverging (new_error > correction)
- **Decreasing**: System is converging (correction working effectively)

---

## Visualization Plots Explained

The system generates 6 plots in `error_analysis_<timestamp>.png`:

### Plot 1: Landmark Distances (Top Left)
**Title**: "Landmark Distance (Actual vs Ideal)"

**What it shows**:
- Blue line: `landmark_mean_hat_tilde` (average per landmark)
- Orange line: `landmark_max_hat_tilde` (worst-case landmark)

**How to read**:
- Y-axis: Distance in Maya units
- X-axis: Transition number
- **Increasing trend**: Landmarks drifting away from ideal positions
- **Stable trend**: Drift is controlled

**What to look for**:
- Mean should stay < 0.05 for good quality
- Max should stay < 0.15 (outlier landmarks)
- If both increase: Systematic drift across all landmarks

### Plot 2: Vertex Distances (Top Right)
**Title**: "Vertex Distance (Actual vs Ideal)"

**What it shows**:
- Blue line: `vertex_mean_hat_tilde` (average per vertex)
- Orange line: `vertex_max_hat_tilde` (worst-case vertex)

**How to read**:
- Shows the **visible** surface deviation
- Usually smaller than landmark distances (tissue smoothing)
- This is what the audience will see!

**What to look for**:
- Mean < 0.02: Not noticeable
- Mean 0.02-0.05: Subtle, may be acceptable
- Mean > 0.05: Clearly visible drift

### Plot 3: Error Magnitude (Middle Left) - **MOST IMPORTANT**
**Title**: "Error Magnitude Over Transitions"

**What it shows**:
- Blue line: `new_error_magnitude` (fresh error per transition)
- Orange line: `accumulated_error_magnitude` (error for next correction)

**How to read**:
- With dissipation_rate = 1.0, both lines should overlap (no residual)
- **Increasing**: Feedback control is failing (new error > correction)
- **Decreasing**: Feedback control is working
- **Flat**: Equilibrium (new error ≈ correction, but still drifting)

**Critical diagnostic**:
- If this plot increases → **fundamental problem**: Pre-computed deltas incompatible with corrected start positions

### Plot 4: Accumulated Error (Middle Right)
**Title**: "Accumulated Error Magnitude"

**What it shows**:
- Single line: `accumulated_error_magnitude`
- Same as orange line in Plot 3, but isolated for clarity

**How to read**:
- This is the error that will be corrected in the next transition
- Should match `new_error_magnitude` when dissipation_rate = 1.0
- If much larger → residual error is accumulating

### Plot 5: Correction Applied (Bottom Left)
**Title**: "Correction Magnitude Over Transitions"

**What it shows**:
- Bar chart: `correction_magnitude` per transition
- Height = how much correction was applied

**How to read**:
- First transition: No bar (no previous error to correct)
- Subsequent transitions: Should have bars if `correction_applied = TRUE`
- **Compare to Plot 3**: Is correction magnitude comparable to error magnitude?

**Diagnostic**:
- If correction_magnitude << error_magnitude: Not correcting enough
- If correction_magnitude ≈ error_magnitude: Correcting appropriately
- If correction_magnitude >> error_magnitude: Over-correcting (shouldn't happen with dissipation_rate ≤ 1.0)

### Plot 6: Dissipation Rate (Bottom Right)
**Title**: "Dissipation Rate Per Transition"

**What it shows**:
- Line plot: `dissipation_rate` (should be constant if not changed)

**How to read**:
- Usually a flat line at 1.0 (for testing)
- If varying, shows where dissipation rate changed

**Why it matters**:
- Helps identify if error changes are due to rate changes vs. actual drift

---

## Common Analysis Scenarios

### Scenario 1: Error Increases Despite 100% Correction

**Symptoms**:
- `dissipation_rate = 1.0`
- `correction_applied = TRUE`
- `accumulated_error_magnitude` increases over transitions
- Plot 3 shows increasing trend

**CSV Evidence**:
```csv
transition,prev_error_magnitude,new_error_magnitude,residual_error_magnitude,correction_magnitude
2,2.156,2.314,2.314,2.156
3,2.314,2.789,2.789,2.314
4,2.789,3.102,3.102,2.789
```

**Analysis**:
- Transition 2: Started with 2.156 error, corrected it fully
- After simulation: New error = 2.314 (grew by +0.158)
- Transition 3: Corrected 2.314, got 2.789 new error (grew by +0.475)
- **Pattern**: `new_error > prev_error` → net accumulation

**Diagnosis**: **Pre-computed deltas are incompatible with corrected start positions**

**Why it happens**:
1. `delta_BC` was computed assuming start from `ideal_B` (B_tilde)
2. After correction, we start from `corrected_B` (B_prime) ≠ ideal_B
3. Applying delta_BC from wrong starting position introduces new error
4. New error > correction applied → net drift

**Solution options**:
1. **Accept limitation**: Pre-computed transitions will drift when chained with corrections
2. **Re-compute on-the-fly**: Compute delta_BC starting from corrected_B (expensive)
3. **Predictive correction**: Predict where you'll end up and correct toward that
4. **Reduce dissipation rate**: Use α < 1.0 to under-correct and find equilibrium

### Scenario 2: Correction Not Applied

**Symptoms**:
- `correction_applied = FALSE` for all transitions
- `correction_magnitude = 0.0`
- Error continuously accumulates (open loop behavior)

**CSV Evidence**:
```csv
transition,correction_applied,correction_magnitude,accumulated_error_magnitude
1,FALSE,0.0,2.156
2,FALSE,0.0,4.312
3,FALSE,0.0,6.897
```

**Diagnosis**: **Feedback control logic not triggering**

**Check**:
1. Is `ENABLE_FEEDBACK_CONTROL = True`?
2. Is `ACCUMULATED_ERROR` being set after first transition?
3. Is `is_first_transition` flag correct?

**Fix**: Debug the condition at [h99-UI-mayascipt-V1.py:1910](h99-UI-mayascipt-V1.py#L1910)

### Scenario 3: Correction Applied But Residual Error Persists

**Symptoms**:
- `dissipation_rate = 1.0` (should have no residual)
- `residual_error_magnitude ≠ new_error_magnitude` (should be equal with α=1.0)
- `accumulated_error_magnitude` much larger than `new_error_magnitude`

**CSV Evidence**:
```csv
transition,dissipation_rate,prev_error_magnitude,new_error_magnitude,residual_error_magnitude
2,1.0,2.156,2.314,4.470
```

**Expected** (with α=1.0):
```
residual = (1 - 1.0) × 2.156 + 2.314 = 0 + 2.314 = 2.314
```

**Actual**:
```
residual = 4.470  ← WRONG!
```

**Diagnosis**: **Bug in residual error calculation**

**Check**: Formula at [h99-UI-mayascipt-V1.py:2091](h99-UI-mayascipt-V1.py#L2091)
```python
residual_error[key] = (1 - ERROR_DISSIPATION_RATE) * prev_err + new_err
```

**Fix**: Verify that `ERROR_DISSIPATION_RATE` has correct value (should be 1.0)

### Scenario 4: Landmarks OK But Vertices Drifting

**Symptoms**:
- `landmark_mean_hat_tilde < 0.05` (landmarks are close to ideal)
- `vertex_mean_hat_tilde > 0.1` (vertices are far from ideal)

**CSV Evidence**:
```csv
transition,landmark_mean_hat_tilde,vertex_mean_hat_tilde
3,0.034,0.156
```

**Diagnosis**: **Tissue simulation is not following landmark corrections**

**Possible causes**:
1. Ziva tissue settings have high stiffness (resisting landmark movement)
2. Tissue cache is stale (not re-simulating after correction)
3. Attachment weights are weak (tissue not following landmarks)

**Fix**:
- Clear Ziva cache after applying corrections
- Check tissue-to-landmark attachment weights
- Verify tissue stiffness parameters

### Scenario 5: Error Oscillates

**Symptoms**:
- Error magnitude alternates up/down across transitions
- Plot 3 shows zigzag pattern

**CSV Evidence**:
```csv
transition,new_error_magnitude
1,2.156
2,3.421
3,1.897
4,3.102
5,2.034
```

**Diagnosis**: **Over-correction causing oscillation**

**Why it happens**:
- Correction overshoots ideal position
- Next transition starts from other side of target
- Creates oscillating behavior

**Fix**: Reduce `ERROR_DISSIPATION_RATE` to < 1.0 (e.g., 0.8 or 0.5)

---

## Step-by-Step Analysis Workflow

### Step 1: Load and Inspect CSV

```python
import pandas as pd
import numpy as np

# Load the CSV
df = pd.read_csv('h96-debug-dissipating-error/debug_errors_<timestamp>.csv')

# Display basic info
print(f"Total transitions: {len(df)}")
print(f"Phoneme sequence: {' → '.join(df['phoneme_to'].values)}")
print(f"\nFirst few rows:")
print(df.head())
```

### Step 2: Check if Correction is Working

```python
# Check correction application
print(f"\nCorrection applied: {df['correction_applied'].sum()} / {len(df)} transitions")
print(f"Average correction magnitude: {df[df['correction_applied']]['correction_magnitude'].mean():.4f}")

# Compare to error magnitude
print(f"Average error magnitude: {df['new_error_magnitude'].mean():.4f}")
```

**Expected**: Correction should be applied for all transitions except the first.

### Step 3: Analyze Error Trend

```python
# Check if error is growing, shrinking, or stable
error_trend = np.polyfit(df['transition'], df['accumulated_error_magnitude'], 1)
slope = error_trend[0]

if slope > 0.1:
    print(f"\n⚠️ ERROR INCREASING: Slope = {slope:.4f}")
    print("   → Feedback control is not preventing drift")
elif slope < -0.1:
    print(f"\n✅ ERROR DECREASING: Slope = {slope:.4f}")
    print("   → Feedback control is working!")
else:
    print(f"\n⚡ ERROR STABLE: Slope = {slope:.4f}")
    print("   → System is in equilibrium (but may still be drifting)")
```

### Step 4: Check Residual Error Formula

```python
# Verify residual error calculation
df['expected_residual'] = (1 - df['dissipation_rate']) * df['prev_error_magnitude'] + df['new_error_magnitude']
df['residual_error_diff'] = np.abs(df['residual_error_magnitude'] - df['expected_residual'])

max_diff = df['residual_error_diff'].max()
if max_diff > 0.01:
    print(f"\n⚠️ RESIDUAL ERROR MISMATCH: Max difference = {max_diff:.4f}")
    print("   → Check residual error calculation in code")
else:
    print(f"\n✅ RESIDUAL ERROR FORMULA CORRECT: Max difference = {max_diff:.6f}")
```

### Step 5: Compare New Error vs Correction

```python
# For transitions with correction, compare new error to previous correction
df_corrected = df[df['correction_applied']].copy()
df_corrected['prev_correction'] = df_corrected['correction_magnitude'].shift(1)
df_corrected['error_growth'] = df_corrected['new_error_magnitude'] - df_corrected['prev_correction']

avg_growth = df_corrected['error_growth'].mean()
if avg_growth > 0:
    print(f"\n⚠️ ERROR GROWING: Average new_error exceeds previous correction by {avg_growth:.4f}")
    print("   → Pre-computed deltas are incompatible with corrected start positions")
else:
    print(f"\n✅ ERROR CONTROLLED: Average new_error is {abs(avg_growth):.4f} less than correction")
```

### Step 6: Identify Worst Transitions

```python
# Find transitions with largest error increase
df['error_increase'] = df['new_error_magnitude'] - df['prev_error_magnitude']
worst = df.nlargest(3, 'error_increase')[['transition', 'phoneme_from', 'phoneme_to', 'error_increase']]
print("\n🔴 Worst transitions (largest error increase):")
print(worst)
```

### Step 7: Check Landmark vs Vertex Consistency

```python
# Compare landmark and vertex drift
df['landmark_to_vertex_ratio'] = df['landmark_mean_hat_tilde'] / df['vertex_mean_hat_tilde']
avg_ratio = df['landmark_to_vertex_ratio'].mean()

print(f"\n📊 Landmark/Vertex ratio: {avg_ratio:.2f}")
if avg_ratio > 2.0:
    print("   → Landmarks drifting more than vertices (tissue smoothing effect)")
elif avg_ratio < 0.5:
    print("   ⚠️ Vertices drifting more than landmarks (tissue simulation issue)")
else:
    print("   → Normal relationship between landmarks and vertices")
```

### Step 8: Generate Summary Report

```python
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)
print(f"Total transitions: {len(df)}")
print(f"Dissipation rate: {df['dissipation_rate'].iloc[0]}")
print(f"\nError Statistics:")
print(f"  Initial error: {df['new_error_magnitude'].iloc[0]:.4f}")
print(f"  Final error: {df['new_error_magnitude'].iloc[-1]:.4f}")
print(f"  Average error: {df['new_error_magnitude'].mean():.4f}")
print(f"  Error trend: {'INCREASING' if slope > 0.1 else 'DECREASING' if slope < -0.1 else 'STABLE'}")
print(f"\nLandmark Drift:")
print(f"  Average: {df['landmark_mean_hat_tilde'].mean():.4f}")
print(f"  Maximum: {df['landmark_mean_hat_tilde'].max():.4f}")
print(f"\nVertex Drift:")
print(f"  Average: {df['vertex_mean_hat_tilde'].mean():.4f}")
print(f"  Maximum: {df['vertex_mean_hat_tilde'].max():.4f}")
print("="*60)
```

---

## Quick Reference

### Key Formulas

| Concept | Formula |
|---------|---------|
| New error | `new_error = ideal_positions - actual_positions` |
| Correction | `correction = dissipation_rate × accumulated_error` |
| Corrected positions | `corrected_pos = actual_pos + correction` |
| Residual error | `residual = (1 - α) × prev_error + new_error` |
| Accumulated error | `accumulated_error = residual_error` (for next transition) |

### Color Code for Interpretation

| Value Range | Color | Meaning |
|-------------|-------|---------|
| landmark_mean < 0.01 | 🟢 Green | Excellent |
| landmark_mean 0.01-0.05 | 🟡 Yellow | Good |
| landmark_mean 0.05-0.1 | 🟠 Orange | Moderate drift |
| landmark_mean > 0.1 | 🔴 Red | Significant drift |

### Common CSV Patterns

| Pattern | Diagnosis |
|---------|-----------|
| `new_error` increasing | System diverging |
| `new_error` decreasing | System converging |
| `new_error ≈ constant` | Equilibrium (stable drift) |
| `residual ≠ expected` | Calculation bug |
| `correction_magnitude = 0` | Correction not applied |
| `vertex >> landmark` | Tissue simulation issue |

---

## Example Analysis

Given this CSV data:
```csv
transition,phoneme_to,landmark_mean_hat_tilde,correction_magnitude,prev_error_magnitude,new_error_magnitude,residual_error_magnitude,accumulated_error_magnitude,correction_applied,dissipation_rate
1,dataV22_phoneme_N,0.042,0.0,0.0,2.316,2.316,2.316,FALSE,1.0
2,dataV22_phoneme_IY,0.058,2.316,2.316,3.124,3.124,3.124,TRUE,1.0
3,dataV22_phoneme_T,0.071,3.124,3.124,3.892,3.892,3.892,TRUE,1.0
```

**Analysis**:

1. **Transition 1** (Neutral → N):
   - No correction (first transition)
   - New error: 2.316
   - Landmark mean: 0.042 (good)

2. **Transition 2** (N → IY):
   - Correction applied: 2.316 (100% of previous error)
   - New error: 3.124 (grew by +0.808)
   - Landmark mean: 0.058 (still acceptable)
   - **Problem**: New error > previous error despite correction

3. **Transition 3** (IY → T):
   - Correction applied: 3.124
   - New error: 3.892 (grew by +0.768)
   - Landmark mean: 0.071 (getting worse)
   - **Pattern confirmed**: Error increasing at ~0.8 per transition

**Conclusion**: Feedback control is applying corrections correctly, but each transition introduces more error than was corrected. This indicates that pre-computed deltas are incompatible with corrected starting positions.

**Recommendation**:
- Consider reducing dissipation_rate to 0.6-0.8 to find equilibrium
- Or re-compute transitions on-the-fly from corrected positions
- Or accept that chained pre-computed transitions will drift

---

## Further Reading

- [FEEDBACK_CONTROL_README.md](FEEDBACK_CONTROL_README.md) - Complete feedback control documentation
- [README_DEBUG.md](README_DEBUG.md) - Debug system technical details
- [QUICK_START.md](QUICK_START.md) - Quick diagnostic guide
- [FINAL_FIXES.md](FINAL_FIXES.md) - Summary of all fixes applied

---

**Last Updated**: 2025-12-16
**Author**: Claude Code Debug System
**Version**: 1.0
