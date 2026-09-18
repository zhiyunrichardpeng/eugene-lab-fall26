# Auto-Breakdown Multi-Phoneme Feature

## Overview

The animation system now automatically breaks down multi-phoneme sequences into individual transitions and chains them together using **existing transition files** from the database. This means you don't need to manually type each transition or click "continue" mode repeatedly.

## How It Works

### **Input Detection**

When you enter a sequence with **3 or more phonemes** (e.g., "Neutral to phoneme_Y to phoneme_UW to phoneme_V"):

1. The system detects this is a multi-phoneme sequence
2. It automatically breaks it down into individual transitions
3. Each transition uses the **existing transition file** if available
4. All transitions are chained together automatically

### **Example: "Neutral to phoneme_Y to phoneme_UW to phoneme_V to phoneme_N to phoneme_AA"**

**What happens behind the scenes:**

```
STEP 1: Neutral → phoneme_Y
  - Uses existing file: "dataV22_AU0_neutral to dataV22_phoneme_Y"
  - Frames: 14 → 50 (36 frames)
  - Continue mode: FALSE (starting fresh)

STEP 2: phoneme_Y → phoneme_UW
  - Uses existing file: "dataV22_phoneme_Y to dataV22_phoneme_UW"
  - Frames: 50 → 86 (36 frames)
  - Continue mode: TRUE (chaining from previous)
  - Initial pose: Retrieved from frame 50 (not frame 14!)

STEP 3: phoneme_UW → phoneme_V
  - Uses existing file: "dataV22_phoneme_UW to dataV22_phoneme_V"
  - Frames: 86 → 122 (36 frames)
  - Continue mode: TRUE

STEP 4: phoneme_V → phoneme_N
  - Uses existing file: "dataV22_phoneme_V to dataV22_phoneme_N"
  - Frames: 122 → 158 (36 frames)
  - Continue mode: TRUE

STEP 5: phoneme_N → phoneme_AA
  - Uses existing file: "dataV22_phoneme_N to dataV22_phoneme_AA"
  - Frames: 158 → 194 (36 frames)
  - Continue mode: TRUE

RESULT:
  - Total frames: 14 → 194 (180 frames)
  - Timeline automatically extends to frame 230 (194 + 36)
  - All transitions smoothly connected
```

## Key Features

### ✅ **Reuses Existing Transition Files**

- Each transition (e.g., "phoneme_Y to phoneme_UW") looks for the pre-generated file
- If the file exists, it's loaded directly (fast!)
- If the file doesn't exist, it's generated on-the-fly

### ✅ **Automatic Chaining**

- First transition starts from frame 14 (or current LAST_ANIMATION_END_FRAME if continuing)
- Each subsequent transition:
  - Starts from the **last frame** of the previous transition
  - Uses the **actual joint positions** from that frame (not frame 14!)
  - Automatically enables "continue mode"

### ✅ **No Frame Jumps**

- Frame 51 smoothly continues from frame 50's pose
- Frame 87 smoothly continues from frame 86's pose
- No more sudden jumps back to neutral pose

### ✅ **Timeline Auto-Extension**

- Timeline automatically extends 36 frames beyond the final frame
- You can see all frames in the Maya timeline

## Usage Examples

### **Example 1: Simple Multi-Phoneme Sequence**

**User Input:**
```
Neutral to phoneme_Y to phoneme_UW to phoneme_V
```

**What happens:**
1. System detects 4 phonemes (3 transitions)
2. Runs:
   - Neutral → Y (frames 14-50)
   - Y → UW (frames 50-86)
   - UW → V (frames 86-122)
3. Final result: Frames 14-122, timeline shows up to frame 158

### **Example 2: Full Sentence ("You've not eaten it yet")**

**User Input:**
```
Neutral to phoneme_Y to phoneme_UW to phoneme_V to phoneme_N to phoneme_AA to phoneme_T to phoneme_IY to phoneme_T to phoneme_N to phoneme_IY to phoneme_T to phoneme_Y to phoneme_EH to phoneme_T
```

**What happens:**
1. System detects 15 phonemes (14 transitions!)
2. Each transition uses existing file:
   - "dataV22_AU0_neutral to dataV22_phoneme_Y"
   - "dataV22_phoneme_Y to dataV22_phoneme_UW"
   - "dataV22_phoneme_UW to dataV22_phoneme_V"
   - ... and so on
3. All 14 transitions are chained together automatically
4. Final result: ~504 frames (14 × 36 frames)

### **Example 3: Continue from Previous Animation**

**Scenario:**
- You already have frames 14-50 with "Neutral → Y"
- Check "Continue from last animation"
- Enter: `phoneme_Y to phoneme_UW to phoneme_V to phoneme_N`

**What happens:**
1. System detects 4 phonemes (3 transitions)
2. Starts from frame 50 (not frame 14!)
3. Runs:
   - Y → UW (frames 50-86)
   - UW → V (frames 86-122)
   - V → N (frames 122-158)
4. Final result: Frames 14-158 (complete sequence)

## Console Output Example

```
========= STARTING FULL PIPELINE =========

============================================================
MULTI-PHONEME SEQUENCE DETECTED: 6 phonemes
Breaking down into 5 individual transitions...
============================================================

[STEP 1/5] Transition: dataV22_AU0_neutral → dataV22_phoneme_Y

============================================================
RUNNING TRANSITION: dataV22_AU0_neutral → dataV22_phoneme_Y
Start frame: 14, Continue mode: False
============================================================

[REUSE] Transition file already exists: C:\...\seq_neutral_to_phoneme_Y\combined_checkpoints.pkl
--- Starting Deformation Application ---
Start frame offset: 14
Continue from last: False
Getting initial joint positions from frame 14
...
[TRANSITION COMPLETE] dataV22_AU0_neutral → dataV22_phoneme_Y
Final frame: 50

[STATE UPDATE] Next transition will start from frame 50

[STEP 2/5] Transition: dataV22_phoneme_Y → dataV22_phoneme_UW

============================================================
RUNNING TRANSITION: dataV22_phoneme_Y → dataV22_phoneme_UW
Start frame: 50, Continue mode: True
============================================================

[REUSE] Transition file already exists: C:\...\seq_phoneme_Y_to_phoneme_UW\combined_checkpoints.pkl
--- Starting Deformation Application ---
Start frame offset: 50
Continue from last: True
Retrieving joint positions from the LAST FRAME (frame 50) instead of frame 14
...
[TRANSITION COMPLETE] dataV22_phoneme_Y → dataV22_phoneme_UW
Final frame: 86

[STATE UPDATE] Next transition will start from frame 86

... (continues for all transitions)

============================================================
ALL TRANSITIONS COMPLETE!
Total frames: 14 → 194
Final phoneme: dataV22_phoneme_AA
============================================================
```

## Technical Details

### **Single Transition Mode (2 phonemes)**

If you enter a sequence with **only 2 phonemes**:
- Uses the **original behavior**
- Generates/loads the transition file normally
- Respects the "Continue from last" checkbox

### **Multi-Transition Mode (3+ phonemes)**

If you enter a sequence with **3 or more phonemes**:
- **Automatically** breaks down into individual transitions
- **First transition**: Uses current continue mode setting
- **Subsequent transitions**: Automatically set to continue mode
- Each transition loads its own existing file

### **File Lookup Logic**

For each transition, the system:
1. Creates a user-friendly name (e.g., "phoneme_Y to phoneme_UW")
2. Looks up the path in the database
3. Checks if the checkpoint file exists
4. If exists: Loads it directly
5. If not: Generates it using the deformation processing pipeline

### **State Management**

Global variables are updated after each transition:
- `LAST_ANIMATION_END_FRAME`: Updated to the last frame of the current transition
- `LAST_ANIMATION_PHONEME`: Updated to the ending phoneme of the current transition
- `CONTINUE_FROM_LAST`: Automatically set to TRUE for subsequent transitions

## Benefits

✨ **Faster Workflow**: No need to manually type each transition
✨ **Reuses Existing Data**: Leverages pre-generated transition files
✨ **Smooth Chaining**: No frame jumps between transitions
✨ **Automatic Continue Mode**: Handles chaining logic internally
✨ **Extensible Timeline**: Automatically extends to show all frames

## Backward Compatibility

- **Single transitions still work**: "Neutral to phoneme_Y" behaves exactly as before
- **Manual chaining still works**: You can still check "Continue from last" for single transitions
- **No breaking changes**: Existing workflows remain unchanged

---

**Version**: 3.1 - Auto-Breakdown Edition
**Date**: 2025-12-09
**Feature**: Automatic Multi-Phoneme Breakdown with Transition File Reuse
