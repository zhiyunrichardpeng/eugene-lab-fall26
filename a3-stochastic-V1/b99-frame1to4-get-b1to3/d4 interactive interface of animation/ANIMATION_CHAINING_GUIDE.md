# Animation Chaining Feature - User Guide

## Overview

The animation chaining feature allows you to create sequential phoneme animations where each new animation continues from where the previous one ended. This enables you to build complex speech sequences incrementally.

## How It Works

### Traditional Mode (No Chaining)
- Each animation starts at **frame 14** (rest pose)
- Animation runs from frame 14 to frame 50 (for 36 checkpoints)
- Example: "Neutral → Y" creates frames 14-50

### Chaining Mode (NEW!)
- First animation starts at frame 14
- Second animation continues from the last frame of the first animation
- Each animation appends to the timeline
- Example workflow:
  1. "Neutral → Y" creates frames 14-50 (37 frames)
  2. Enable "Continue from last"
  3. "phoneme_Y → phoneme_UW" creates frames 50-86 (36 more frames)
  4. "phoneme_UW → phoneme_V" creates frames 86-122 (36 more frames)

## UI Components

### New UI Elements

1. **"Continue from last animation" Checkbox**
   - Unchecked (default): Start from frame 14 (neutral pose)
   - Checked: Continue from the last animation's end frame

2. **Status Display**
   - Shows: "Last animation: Frame X, Phoneme: Y"
   - Updates after each animation completes

3. **"Reset Chain" Button**
   - Resets the chain back to frame 14 (neutral pose)
   - Unchecks the "Continue from last" checkbox
   - Use this to start a new animation sequence

4. **"Generate Animation" Button**
   - Same as before, but now respects chaining mode

## Step-by-Step Usage

### Example: Building "You've" (Y → UW → V)

#### Step 1: Create First Phoneme (Neutral → Y)
1. Open Maya with the rig
2. Run the script to open the UI
3. **Input**: `Neutral to phoneme_Y`
4. **Checkbox**: UNCHECKED
5. Click "Generate Animation"
6. **Result**: Animation from frame 14 to frame 50
7. UI now shows: "Last animation: Frame 50, Phoneme: phoneme_Y"

#### Step 2: Continue with Second Phoneme (Y → UW)
1. **Input**: `phoneme_Y to phoneme_UW`
2. **Checkbox**: CHECK "Continue from last animation"
3. Click "Generate Animation"
4. **Result**: Animation from frame 50 to frame 86 (appended!)
5. UI now shows: "Last animation: Frame 86, Phoneme: phoneme_UW"

#### Step 3: Continue with Third Phoneme (UW → V)
1. **Input**: `phoneme_UW to phoneme_V`
2. **Checkbox**: STILL CHECKED
3. Click "Generate Animation"
4. **Result**: Animation from frame 86 to frame 122
5. UI now shows: "Last animation: Frame 122, Phoneme: phoneme_V"

#### Final Result
- Complete timeline: frames 14-122 showing "You've" (Y → UW → V)
- Total: ~108 frames of continuous animation

## Important Notes

### ✅ DO's
- ✅ Check "Continue from last" AFTER the first animation completes
- ✅ Start your next input with the ending phoneme (e.g., if last was "Y", start with "phoneme_Y to ...")
- ✅ Use "Reset Chain" when starting a completely new sequence
- ✅ Watch the status display to confirm the last phoneme

### ❌ DON'Ts
- ❌ Don't check "Continue from last" for the very first animation
- ❌ Don't start from "Neutral" when chaining (it will use the last phoneme instead)
- ❌ Don't forget to uncheck or reset when starting a new unrelated sequence

## Technical Details

### Global Variables (Auto-managed)
- `LAST_ANIMATION_END_FRAME`: Tracks the last frame number (default: 14)
- `LAST_ANIMATION_PHONEME`: Stores the last phoneme (default: "dataV22_AU0_neutral")
- `CONTINUE_FROM_LAST`: Boolean flag for chaining mode (default: False)

### Frame Calculation
```python
# Traditional mode:
frames = 14 to (14 + num_checkpoints)

# Chaining mode:
frames = LAST_ANIMATION_END_FRAME to (LAST_ANIMATION_END_FRAME + num_checkpoints)
```

### Initial Keyframe Logic
- **Traditional**: Sets keyframe at frame 14 for rest pose
- **Chaining**: Skips initial keyframe, continues from existing pose

## Use Cases

### 1. Word-by-Word Animation
Build "Hello" phoneme by phoneme:
1. H (Neutral → H)
2. EH (H → EH) [chain]
3. L (EH → L) [chain]
4. OW (L → OW) [chain]

### 2. Sentence Building
Create "You've not eaten it yet" incrementally:
1. "You've": Neutral → Y → UW → V
2. "not": V → N → AA → T (continue)
3. "eaten": T → IY → T → N (continue)
4. "it": N → IY → T (continue)
5. "yet": T → Y → EH → T (continue)

### 3. Testing Transitions
Test specific phoneme transitions:
1. Create Neutral → A
2. Chain A → B
3. Chain B → C
4. Reset and try different path: Neutral → A → D → E

## Troubleshooting

### Issue: Animation starts from wrong frame
**Solution**: Click "Reset Chain" to start fresh from frame 14

### Issue: Phonemes don't connect smoothly
**Solution**: Ensure the starting phoneme of the new sequence matches the ending phoneme of the last sequence

### Issue: Timeline gets too long
**Solution**:
1. Reset chain
2. Render the current sequence
3. Start a new sequence

## Console Output Example

```
=== ANIMATION CHAINING INFO ===
Continue from last: True
Start frame: 50
Last phoneme: dataV22_phoneme_Y
New sequence: dataV22_phoneme_Y → dataV22_phoneme_UW
===============================

--- Starting Deformation Application ---
Start frame offset: 50
Continue from last: True
Continuing from last animation at frame 50
[...]
--- Deformation complete. Final frame: 86 ---

[CHAINING] Animation complete. Final frame: 86
[CHAINING] Last phoneme: dataV22_phoneme_UW
[CHAINING] Ready for next animation from frame 86
```

## Benefits

✨ **Incremental Building**: Create complex sequences step-by-step
✨ **Easy Testing**: Test individual transitions before committing
✨ **Flexible Editing**: Insert or replace phoneme segments
✨ **Timeline Management**: Build animations without frame conflicts
✨ **Realistic Speech**: Create natural co-articulation between phonemes

## Future Enhancements

Potential improvements:
- Save/load animation chains
- Visual timeline preview
- Undo last segment
- Auto-suggest next phoneme based on word dictionary
- Blend strength control between chained segments

---

**Version**: 3.0 - Chaining Edition
**Date**: 2025-12-09
**Feature**: Animation Chaining for Sequential Phoneme Animations
