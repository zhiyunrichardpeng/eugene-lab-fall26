if debug_mode_vsCode == 0:
    # import maya.standalone
    # maya.standalone.initialize()    
    import maya.cmds as cmds
    import maya.OpenMaya as OpenMaya    
    import maya.mel as mel

def step_forward_one_frame():
    current_frame = cmds.currentTime(query=True)
    cmds.currentTime(current_frame + 1)

# Loop from i = 14 to i = 49
for i in range(14, 50):  # range(14, 50) includes 14 to 49
    start_frame = i + 0
    keframe_max = i + 1

    # Step forward one frame in Maya
    step_forward_one_frame()

    # Calculate distance (replace with your actual logic)
    distance_1 = calculate_distance_adhoc(dataV22_phoneme_UW, keframe_max)
    distance_log1.append(distance_1)
    time_log1.append(keframe_max)

    # Print results
    print("distance_1", distance_1)
    print("No.", keframe_max, "Frame distance done")

# Final output (optional)
print("Distance log:", distance_log1)
print("Time log:", time_log1)