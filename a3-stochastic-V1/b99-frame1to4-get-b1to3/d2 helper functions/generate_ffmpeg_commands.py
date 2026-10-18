import os
from paths import results_path
# --- MAIN CONFIGURATION ---

# 1. Add your base names to this list.
#    The script will automatically detect 'skin' or 'muscle' to choose the right folder.
BASE_NAMES = [
    "v95_test_30_frames",
    # "q64_muscle_seq3_n_sad",
    # Add other base names here, for example:
    # "q65_seq2_skin",
]

# 2. Set the name of the output sub-folder.
#    (In your example, this was 'o59-25-10-8')
OUTPUT_SUBFOLDER = "o59-25-10-8" 

# 3. Define the base paths (you probably won't need to change these).
PNG_BASE_PATH =str(results_path / "maya_render_output_hub/temp_rendered_png_hub")
MP4_OUTPUT_BASE_PATH =str(results_path / "maya_render_output_hub")

# 4. Name of the batch file to be created.
BATCH_FILENAME = "run_all.bat"

# --- FFMPEG PARAMETERS (Advanced) ---
# --- VIDEO LENGTH TUNING ---
# Start processing from this frame number
START_FRAME = 14

# Set the desired length of the final video in seconds
# e.g., for 518 frames (15 idle + 14 phoneme transitions * ~36 frames),
# a natural speaking time for "You've not eat neat yet" is ~1.5 to 2.0 seconds.
DESIRED_VIDEO_LENGTH_SECONDS = 5 # Tune this to change how long the video plays!

import math
import glob

VIDEO_CODEC = "libx264"
PIXEL_FORMAT = "yuv420p"
CRF = 23 # Constant Rate Factor (lower is higher quality)
PRESET = "medium"
SCALE = "1024:1024"
SCALE_FILTER = "lanczos"
# --- END CONFIGURATION ---


def generate_ffmpeg_command(base_name, output_subfolder):
    """Generates a single ffmpeg command based on the base_name."""
    
    # Determine the source folder ('face_closeup' or 'muscle_closeup')
    if "skin" in base_name.lower() or "lips" in base_name.lower():
        image_type_folder = "face_closeup"
    elif "muscle" in base_name.lower() or "youvenot" in base_name.lower():
        image_type_folder = "muscle_closeup"
    else:
        print(f"Warning: Could not determine type for '{base_name}'. Defaulting to 'muscle_closeup'.")
        image_type_folder = "muscle_closeup"
        
    # manually set:
    # image_type_folder = "fig15_view_sideV2"
    image_type_folder = "face_closeup"

    # Construct paths
    # input_dir = os.path.join(PNG_BASE_PATH, image_type_folder)
    input_dir = os.path.join(PNG_BASE_PATH, image_type_folder, "beauty")
    input_filename_pattern = f"{base_name}_1_%%04d.png"
    input_path = os.path.join(input_dir, input_filename_pattern)
    
    output_dir = os.path.join(MP4_OUTPUT_BASE_PATH, output_subfolder)
    output_filename = f"{base_name}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # Ensure the output directory exists. Ffmpeg might error if it doesn't.
    # Note: This check happens when the python script is run, not when the bat is run.
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Adaptive Framerate Calculation ---
    search_pattern = os.path.join(input_dir, f"{base_name}_1_*.png")
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        print(f"Warning: No files found for {base_name} at {search_pattern}")
        framerate = 30
    else:
        # Find maximum frame index
        max_frame = -1
        for f in matching_files:
            try:
                frame_idx = int(f.split('_')[-1].split('.')[0])
                if frame_idx > max_frame:
                    max_frame = frame_idx
            except ValueError:
                pass
                
        if max_frame >= START_FRAME:
            total_frames = (max_frame - START_FRAME) + 1
        else:
            total_frames = 0
            
        if DESIRED_VIDEO_LENGTH_SECONDS > 0 and total_frames > 0:
            framerate = math.ceil(total_frames / DESIRED_VIDEO_LENGTH_SECONDS)
        else:
            framerate = 30
            
    print(f"[{base_name}] Detected {total_frames} frames (from {START_FRAME} to {max_frame}). Calculated framerate: {framerate}")

    # Assemble the command
    command = (
        f'ffmpeg -framerate {framerate} -start_number {START_FRAME} -i "{input_path}" ' 
        f'-c:v {VIDEO_CODEC} -pix_fmt {PIXEL_FORMAT} -crf {CRF} -preset {PRESET} ' 
        f'-vf "scale={SCALE}:flags={SCALE_FILTER}" "{output_path}"'
    )
    
    return command

def main():
    """Main function to generate commands and write the batch file."""
    
    if not BASE_NAMES:
        print("The BASE_NAMES list is empty. Please add base names to generate commands.")
        return
        
    print("--- Generated FFMPEG Commands ---")
    
    all_commands = []
    for name in BASE_NAMES:
        if not name.strip(): # Skip empty names
            continue
        command = generate_ffmpeg_command(name, OUTPUT_SUBFOLDER)
        print(command)
        all_commands.append(command)
        
    try:
        with open(BATCH_FILENAME, "w") as f:
            f.write("@echo off\n")
            f.write("echo Starting FFMPEG processing...\n\n")
            for i, command in enumerate(all_commands):
                f.write("echo.\n")
                f.write(f'echo "--- [{i+1}/{len(all_commands)}] Processing: {BASE_NAMES[i]} ---"\n')
                f.write(command + "\n")
            f.write("\n@echo on\n")
            f.write("echo.\n")
            f.write("echo All tasks complete.\n")
            # f.write("pause\n") # Keeps the window open to see the result
            
        print(f"\nSuccessfully created batch file: {BATCH_FILENAME}")
        print("You can now double-click this file to run all the ffmpeg commands.")

    except Exception as e:
        print(f"\nAn error occurred while writing the batch file: {e}")

if __name__ == "__main__":
    main()
