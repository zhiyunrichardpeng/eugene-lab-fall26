import maya.cmds as cmds
import time
import os
import subprocess
from paths import a3_stochastic_relative_path, results_path

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Easily name the image series here
BASE_NAME = "v40_try_fix_numerical_drift_V2"

# Output directory for the PNG images
INPUT_DIR =str(results_path / "maya_render_output_hub/temp_rendered_png_hub/playblust_results")

# Resolution
WIDTH = 256
HEIGHT = 256

# Video generation script
VIDEO_SCRIPT = os.path.join(a3_stochastic_relative_path, "b99-frame1to4-get-b1to3", "d2 helper functions", "make_playblast_video.py")

def main(custom_name=None):
    global BASE_NAME
    if custom_name:
        BASE_NAME = custom_name
        
    # Ensure output directory exists
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)

    # Get animation frame ranges
    start_frame = cmds.playbackOptions(q=True, minTime=True)
    end_frame = cmds.playbackOptions(q=True, maxTime=True)
    num_frames = max(1, end_frame - start_frame + 1)

    # File path prefix for playblast. Maya will append '.XXXX.png'
    file_path_base = os.path.join(INPUT_DIR, BASE_NAME)

    print("-" * 50)
    print(f"--- Starting Playblast ---")
    print(f"Sequence: {BASE_NAME}")
    print(f"Frames: {start_frame} to {end_frame} ({num_frames} frames)")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Output Directory: {INPUT_DIR}")

    # Start timing
    start_time = time.time()
    
    # Run playblast
    # framePadding=4 ensures numbers like .0386.png
    cmds.playblast(
        format="image",
        compression="png",
        filename=file_path_base,
        widthHeight=[WIDTH, HEIGHT],
        viewer=False,
        showOrnaments=False,
        clearCache=True,
        framePadding=4,
        forceOverwrite=True,
        offScreen=True,
        percent=100
    )

    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    time_per_frame = total_time / num_frames

    print(f"\n--- Playblast Complete ---")
    print(f"Total time taken: {total_time:.4f} seconds")
    print(f"Average time per frame: {time_per_frame:.4f} seconds")
    print("-" * 50)

    # Run the video generation script
    print(f"\nRunning video generation script: \n{VIDEO_SCRIPT}")
    try:
        # Update the BASE_NAME in the video script to match our current BASE_NAME
        with open(VIDEO_SCRIPT, "r") as f:
            lines = f.readlines()
        
        with open(VIDEO_SCRIPT, "w") as f:
            for line in lines:
                if line.startswith("BASE_NAME ="):
                    f.write(f'BASE_NAME = "{BASE_NAME}"\n')
                else:
                    f.write(line)

        # Use python to execute the external script
        cmd = f'python "{VIDEO_SCRIPT}"'
        print(f"Executing: {cmd}")
        
        # subprocess.run waits for it to finish and check=True raises exception on failure
        subprocess.run(cmd, shell=True, check=True)
        print("\nVideo generation script executed successfully.")
    except Exception as e:
        print(f"\nFailed to run the video script. Error: {e}")

if __name__ == "__main__":
    main()
