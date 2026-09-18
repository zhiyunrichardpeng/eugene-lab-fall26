# commandPort -name "localhost:7001" -sourceType "mel";

import maya.cmds as cmds

# --- 1. CONFIGURATION ---
# Animation: "Neutral to N to IY to T to Y to EH to T"
# Frame 15 = start (at Neutral), Frame 230 = end (at T)
# Each transition takes ~36 frames
# The label shows WHICH PHONEME the face is deforming TO

# Format: (start_frame, target_phoneme_label)
# Shows the TARGET phoneme during the transition
PHONEME_FRAME_TIMELINE = [
    (15, "to N"),       # Frame 15-50: Deforming to N
    (50, "to IY"),      # Frame 50-86: Deforming to IY
    (86, "to T"),       # Frame 86-122: Deforming to T
    (122, "to Y"),      # Frame 122-158: Deforming to Y
    (158, "to EH"),     # Frame 158-194: Deforming to EH
    (194, "to T"),      # Frame 194-230: Deforming to T (final)
    (230, "T finished"),      # Frame 230+: Animation complete
]

HUD_NAME = "PhonemeDisplayHUD"

# --- 2. THE HUD FUNCTION ---
def phoneme_hud_logic():
    curr_frame = cmds.currentTime(query=True)
    
    active_label = "..."
    for frame_start, label in PHONEME_FRAME_TIMELINE:
        if curr_frame >= frame_start:
            # We add extra spaces around the text to fake a larger "block" visual
            active_label = f"   {label}   "
        else:
            break
            
    return active_label

# --- 4. CREATE THE HUD ---
def create_phoneme_hud():
    if cmds.headsUpDisplay(HUD_NAME, exists=True):
        cmds.headsUpDisplay(HUD_NAME, remove=True)
        
    cmds.headsUpDisplay(
        HUD_NAME,
        section=1,   # Bottom
        block=0,     # Center
        blockSize="large",  # CORRECTED: Changed 'huge' back to 'large'
        label="",           # Keep label empty so data is centered
        labelFontSize="large", 
        dataFontSize="large", 
        command=phoneme_hud_logic,
        attachToRefresh=True
    )
    print(f"[SUCCESS] Phoneme HUD updated for 'Neutral to N to IY to T to Y to EH to T' (frames 15-230).")

create_phoneme_hud()