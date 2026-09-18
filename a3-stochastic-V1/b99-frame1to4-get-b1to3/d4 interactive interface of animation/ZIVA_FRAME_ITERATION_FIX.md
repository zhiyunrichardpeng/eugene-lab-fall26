# Ziva Frame Iteration Fix

## Overview

Added intelligent frame iteration logic to ensure Ziva simulation proceeds correctly for the **first transition** while allowing subsequent transitions to jump directly to their target frame for efficiency.

## Problem

When retrieving landmark positions:
- **First transition (Neutral → phoneme_Y)**: Needs to iterate through frames 0-14 one-by-one to ensure Ziva simulation calculates properly
- **Subsequent transitions (phoneme_Y → phoneme_UW, etc.)**: Can jump directly to the target frame since the simulation state is already established

Without this, jumping directly to frame 14 on the first transition could cause incorrect Ziva simulation states.

## Solution

Added an `is_first_transition` parameter that controls whether the function iterates through frames or jumps directly.

### **Modified Functions**

#### 1. **retrieve_landmark_positions()** ([h99-UI-mayascipt-V1.py:122-154](h99-UI-mayascipt-V1.py#L122-L154))

```python
def retrieve_landmark_positions(joint_groups_to_query, target_frame=14, is_first_transition=True):
    """
    Retrieve landmark positions at a specific frame.

    Args:
        joint_groups_to_query: Dictionary of joint groups and their landmark counts
        target_frame: Frame number to retrieve positions from (default 14 for neutral pose)
        is_first_transition: If True, iterate through frames 0-14 to ensure Ziva simulation proceeds frame-by-frame
    """
    # For the first transition, iterate through frames to ensure Ziva simulation proceeds correctly
    if is_first_transition:
        print(f"  [FIRST TRANSITION] Iterating through frames 0-{target_frame} for Ziva simulation...")
        for frame_number in range(target_frame + 1):
            cmds.currentTime(str(frame_number), edit=True)
        print(f"  Reached target frame {target_frame}")
    else:
        # For subsequent transitions, jump directly to the target frame
        print(f"  Retrieving landmark positions at frame {target_frame}")
        cmds.currentTime(target_frame, edit=True)
```

**Behavior:**
- **is_first_transition=True**: Iterates through frames 0, 1, 2, ..., 14 (or target_frame)
- **is_first_transition=False**: Jumps directly to target_frame (e.g., frame 50)

#### 2. **initial_joint_position()** ([h99-UI-mayascipt-V1.py:156-173](h99-UI-mayascipt-V1.py#L156-L173))

```python
def initial_joint_position(target_frame=14, is_first_transition=True):
    """
    Gets initial joint positions for the deformation at a specific frame.

    Args:
        target_frame: Frame number to retrieve positions from (default 14 for neutral pose)
        is_first_transition: If True, iterate through frames 0-14 to ensure Ziva simulation proceeds frame-by-frame
    """
    retrieved_positions = retrieve_landmark_positions(joint_groups_to_query, target_frame, is_first_transition)
```

**Behavior:**
- Passes the `is_first_transition` flag down to `retrieve_landmark_positions()`

#### 3. **perform_the_deformation()** ([h99-UI-mayascipt-V1.py:287-307](h99-UI-mayascipt-V1.py#L287-L307))

```python
# Get initial joint positions based on continue mode
if not continue_from_last:
    # Starting fresh from frame 14 (neutral pose) - this is the FIRST transition
    print(f"Getting initial joint positions from frame {start_frame_offset}")
    # is_first_transition=True: Iterate through frames 0-14 for Ziva simulation
    x_joint, x_joint_rotation, x_ooc_l_joint, x_ooc_r_joint, x_frontalis_joint = initial_joint_position(start_frame_offset, is_first_transition=True)
else:
    # Continuing from last animation - use the LAST FRAME's joint positions as the initial state
    print(f"Continuing from last animation at frame {start_frame_offset}")
    # is_first_transition=False: Jump directly to target frame (no need to iterate)
    x_joint, x_joint_rotation, x_ooc_l_joint, x_ooc_r_joint, x_frontalis_joint = initial_joint_position(start_frame_offset, is_first_transition=False)
```

**Logic:**
- **continue_from_last=False** (first transition): `is_first_transition=True` → Iterate through frames
- **continue_from_last=True** (subsequent transitions): `is_first_transition=False` → Jump to target frame

## How It Works

### **Example: "Neutral to phoneme_Y to phoneme_UW to phoneme_V"**

#### **Transition 1: Neutral → phoneme_Y (Frames 14-50)**

```
[FIRST TRANSITION] Iterating through frames 0-14 for Ziva simulation...
  Frame 0...
  Frame 1...
  Frame 2...
  ...
  Frame 14...
Reached target frame 14
Retrieving landmark positions...
```

**Why:** Ziva simulation needs to compute each frame sequentially from the start to establish the correct simulation state at frame 14.

#### **Transition 2: phoneme_Y → phoneme_UW (Frames 50-86)**

```
Continuing from last animation at frame 50
Retrieving landmark positions at frame 50
cmds.currentTime(50, edit=True)  # Jump directly to frame 50
Retrieving landmark positions...
```

**Why:** The simulation state is already established from the previous animation. We can jump directly to frame 50 to read the joint positions without re-simulating.

#### **Transition 3: phoneme_UW → phoneme_V (Frames 86-122)**

```
Continuing from last animation at frame 86
Retrieving landmark positions at frame 86
cmds.currentTime(86, edit=True)  # Jump directly to frame 86
Retrieving landmark positions...
```

**Why:** Same as above - jump directly to frame 86.

## Console Output Example

### **First Transition (Neutral → Y)**
```
--- Getting Initial Joint Positions at frame 14 ---
  [FIRST TRANSITION] Iterating through frames 0-14 for Ziva simulation...
  Reached target frame 14
  Retrieving landmarks for group: oor
  Retrieving landmarks for group: ooc_l
  Retrieving landmarks for group: ooc_r
  Retrieving landmarks for group: frontalis
```

### **Second Transition (Y → UW)**
```
--- Getting Initial Joint Positions at frame 50 ---
  Retrieving landmark positions at frame 50
  Retrieving landmarks for group: oor
  Retrieving landmarks for group: ooc_l
  Retrieving landmarks for group: ooc_r
  Retrieving landmarks for group: frontalis
```

## Benefits

✅ **Correct Ziva Simulation**: First transition ensures Ziva computes frames 0-14 sequentially
✅ **Performance Optimization**: Subsequent transitions skip unnecessary frame iteration
✅ **Smooth Chaining**: All transitions work correctly regardless of their position in the chain
✅ **Automatic Detection**: The system automatically knows which transitions are first vs. subsequent based on `continue_from_last` flag

## Technical Details

### **Frame Iteration Logic**

```python
if is_first_transition:
    # Iterate through all frames up to target
    for frame_number in range(target_frame + 1):  # 0, 1, 2, ..., target_frame
        cmds.currentTime(str(frame_number), edit=True)
else:
    # Jump directly to target
    cmds.currentTime(target_frame, edit=True)
```

### **State Determination**

The `is_first_transition` flag is set based on the `continue_from_last` parameter:
- `continue_from_last=False` → `is_first_transition=True` (first transition in chain)
- `continue_from_last=True` → `is_first_transition=False` (subsequent transition)

This works because:
1. The **first transition** in any sequence always has `continue_from_last=False`
2. All **subsequent transitions** automatically have `continue_from_last=True` (set by the auto-breakdown logic)

## Use Cases

### **Manual Single Transition**
```
User types: "Neutral to phoneme_Y"
Result: Iterates frames 0-14 (is_first_transition=True)
```

### **Manual Chained Transitions**
```
1. User types: "Neutral to phoneme_Y" → Iterates frames 0-14
2. User checks "Continue from last"
3. User types: "phoneme_Y to phoneme_UW" → Jumps to frame 50 (is_first_transition=False)
```

### **Auto-Breakdown Multi-Phoneme**
```
User types: "Neutral to phoneme_Y to phoneme_UW to phoneme_V"

Step 1: Neutral → Y
  - continue_from_last=False → is_first_transition=True
  - Iterates frames 0-14

Step 2: Y → UW
  - continue_from_last=True → is_first_transition=False
  - Jumps to frame 50

Step 3: UW → V
  - continue_from_last=True → is_first_transition=False
  - Jumps to frame 86
```

---

**Version**: 3.2
**Date**: 2025-12-09
**Feature**: Intelligent Ziva Frame Iteration for First vs. Subsequent Transitions
