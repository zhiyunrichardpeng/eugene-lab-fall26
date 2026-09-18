import cv2
import glob
import os
import numpy as np
import subprocess
import math

# ==========================================
# 1. MAIN CONFIGURATION
# ==========================================
# Add your base names to this list.
BASE_NAMES = [
    "v45_reproduce_notevenyou_seqV1", 
    # "another_sequence_name",
]

# Set your shadow intensity (1.0 is pure multiply, 0.0 is no AO)
AO_OPACITY = 0.65  

# ==========================================
# 2. PATH DEFINITIONS
# ==========================================
# PNG_BASE_PATH = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub\temp_rendered_png_hub\masterLayer\face_closeup"
PNG_BASE_PATH = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub\temp_rendered_png_hub\masterLayer\fig15_view_sideV2"

AO_DIR = os.path.join(PNG_BASE_PATH, "A0_Pass")
COLOR_DIR = os.path.join(PNG_BASE_PATH, "beauty")
OUTPUT_DIR = os.path.join(PNG_BASE_PATH, "composited_output")

# FFMPEG Video Output Path
MP4_OUTPUT_BASE_PATH = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub"
OUTPUT_SUBFOLDER = "composited_videos"

# ==========================================
# 3. FFMPEG PARAMETERS
# ==========================================
START_FRAME = 15
DESIRED_VIDEO_LENGTH_SECONDS = 5
VIDEO_CODEC = "libx264"
PIXEL_FORMAT = "yuv420p"
CRF = 23
PRESET = "medium"
SCALE = "1024:-2" # Keeps the original aspect ratio intact
SCALE_FILTER = "lanczos"


def process_sequence(base_name):
    """Composites color and AO for a single sequence base name."""
    seq_output_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(seq_output_dir, exist_ok=True)
    
    # Find files based on base_name
    ao_files = sorted(glob.glob(os.path.join(AO_DIR, f"{base_name}*.png")))
    color_files = sorted(glob.glob(os.path.join(COLOR_DIR, f"{base_name}*.png")))
    
    if not ao_files or not color_files:
        print(f"[{base_name}] Skipping: Missing AO or Color files in folders.")
        return None
        
    frame_count = min(len(color_files), len(ao_files))
    print(f"\n[{base_name}] Found {len(color_files)} Color frames and {len(ao_files)} AO frames.")
    print(f"[{base_name}] Beginning composite for {frame_count} frames...")
    
    for i in range(frame_count):
        color_path = color_files[i]
        ao_path = ao_files[i]
        
        # Read the images. Keep alpha channel for color if it exists.
        color_img = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
        ao_img = cv2.imread(ao_path, cv2.IMREAD_GRAYSCALE)
        
        if color_img is None or ao_img is None:
            continue
            
        color_img = color_img.astype(np.float32) / 255.0
        ao_img = ao_img.astype(np.float32) / 255.0
        
        blended = color_img.copy()
        
        # Isolate the RGB channels (ignoring Alpha for the math)
        rgb = color_img[:,:,:3]
        
        # --- LUMINANCE-ONLY MULTIPLY ---
        # Convert to HSV color space so we can separate brightness from color
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Multiply AO against the Value (brightness) channel only
        v_darkened = v * ao_img
        
        # Apply the Opacity slider (Linear Interpolation) to the Value channel
        v_blended = (v_darkened * AO_OPACITY) + (v * (1.0 - AO_OPACITY))
        
        # Merge back together and convert back to RGB
        hsv_blended = cv2.merge([h, s, v_blended])
        rgb_blended = cv2.cvtColor(hsv_blended, cv2.COLOR_HSV2BGR)
        
        # Put the new color data back into the final image (Alpha remains untouched)
        blended[:,:,:3] = rgb_blended
        
        # Save output uniformly so FFMPEG can easily parse it
        out_name = f"{base_name}_{i+START_FRAME:04d}.png"
        cv2.imwrite(os.path.join(seq_output_dir, out_name), (blended * 255).astype(np.uint8))
        
    print(f"[{base_name}] Compositing complete!")
    return seq_output_dir, frame_count


def generate_ffmpeg_video(base_name, input_dir, total_frames):
    """Runs ffmpeg automatically on the composited frames."""
    output_folder = os.path.join(MP4_OUTPUT_BASE_PATH, OUTPUT_SUBFOLDER)
    os.makedirs(output_folder, exist_ok=True)
    
    output_path = os.path.join(output_folder, f"{base_name}_V3_luminance_only.mp4")
    
    # Calculate FPS based on desired video length
    framerate = max(1, math.ceil(total_frames / DESIRED_VIDEO_LENGTH_SECONDS))
    
    # Note: We outputted the composited files as base_name_0001.png
    input_pattern = os.path.join(input_dir, f"{base_name}_%04d.png")
    
    ffmpeg_cmd = (
        f'ffmpeg -y -framerate {framerate} -start_number {START_FRAME} '
        f'-i "{input_pattern}" '
        f'-c:v {VIDEO_CODEC} -pix_fmt {PIXEL_FORMAT} -crf {CRF} -preset {PRESET} '
        f'-vf "scale={SCALE}:flags={SCALE_FILTER}" "{output_path}"'
    )
    
    print(f"[{base_name}] Running FFMPEG Video Generation...")
    try:
        subprocess.run(ffmpeg_cmd, shell=True, check=True)
        print(f"[{base_name}] SUCCESS! Video saved to:\n -> {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[{base_name}] ERROR: FFMPEG failed with error: {e}")


def main():
    if not BASE_NAMES:
        print("The BASE_NAMES list is empty. Please add base names to begin.")
        return
        
    for name in BASE_NAMES:
        if not name.strip():
            continue
            
        result = process_sequence(name)
        if result:
            seq_output_dir, frame_count = result
            generate_ffmpeg_video(name, seq_output_dir, frame_count)
            
    print("\n--- ALL TASKS COMPLETE ---")

if __name__ == "__main__":
    main()