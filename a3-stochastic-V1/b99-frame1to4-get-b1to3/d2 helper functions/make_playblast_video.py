import os
import subprocess
import glob

# ==========================================
# 1. MAIN CONFIGURATION
# ==========================================
# The base name of your playblast sequence without the numbers/extension
BASE_NAME = "unnamed_playblast_20260710_111139"

# TODO: Update this path to where your playblast PNG files are actually saved!
# For example: r"C:\Users\zpa18\Documents\maya\projects\default\images"
INPUT_DIR = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub\temp_rendered_png_hub\playblust_results" 

# Where to save the final MP4 video
OUTPUT_DIR = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub\composited_videos"

# ==========================================
# 2. FFMPEG PARAMETERS
# ==========================================
START_FRAME = 0   # Your sequence starts at .0000.png
FPS = 30          # Assuming standard 24 fps for your Maya animation
VIDEO_CODEC = "libx264"
PIXEL_FORMAT = "yuv420p"
CRF = 23          # Quality (lower is better, 23 is default visually lossless)
PRESET = "medium"

def generate_playblast_video():
    """Runs ffmpeg automatically on the playblast frames."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_path = os.path.join(OUTPUT_DIR, f"{BASE_NAME}_playblast.mp4")
    
    # Matches files like: v87_add_snap_V2.0000.png, v87_add_snap_V2.0548.png
    input_pattern = os.path.join(INPUT_DIR, f"{BASE_NAME}.%04d.png")
    
    # Verify files exist before running FFMPEG
    sample_files = glob.glob(os.path.join(INPUT_DIR, f"{BASE_NAME}.*.png"))
    if not sample_files:
        print(f"[ERROR] Could not find any PNG files matching '{BASE_NAME}.*.png'")
        print(f"[ERROR] Please check that INPUT_DIR is correct: {INPUT_DIR}")
        return

    print(f"Found {len(sample_files)} frames for {BASE_NAME}. Generating video at {FPS} FPS...")

    # We use a filter to ensure width/height are divisible by 2 (required for yuv420p mp4 format)
    ffmpeg_cmd = (
        f'ffmpeg -y -framerate {FPS} -start_number {START_FRAME} '
        f'-i "{input_pattern}" '
        f'-c:v {VIDEO_CODEC} -pix_fmt {PIXEL_FORMAT} -crf {CRF} -preset {PRESET} '
        f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
        f'"{output_path}"'
    )
    
    print(f"\nRunning FFMPEG Command:\n{ffmpeg_cmd}\n")
    
    try:
        subprocess.run(ffmpeg_cmd, shell=True, check=True)
        print(f"\n[{BASE_NAME}] SUCCESS! Playblast video saved to:\n -> {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n[{BASE_NAME}] ERROR: FFMPEG failed with error: {e}")

if __name__ == "__main__":
    generate_playblast_video()
