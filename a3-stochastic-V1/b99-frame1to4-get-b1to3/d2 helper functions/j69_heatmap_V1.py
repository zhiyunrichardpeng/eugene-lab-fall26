import cv2
import numpy as np
import glob
import os
from paths import results_path, a3_stochastic_relative_path

# Set paths
input_folder = str(results_path / "maya_render_output_hub" / "temp_rendered_png_hub" / "face_closeup" / "beauty")
output_folder = str(a3_stochastic_relative_path / "visualization")
os.makedirs(output_folder, exist_ok=True)
output_video_path = os.path.join(output_folder, 'heatmap_video_surprise_v3.mp4')

# Load the base frame (neutral expression) in grayscale
# base_img_path = os.path.join(input_folder, 't10_two_eyes_wires_done_skin_1_0000.png')
base_img_path = os.path.join(input_folder, 'u78_test_suprise_v2_1_0000.png')
base_img = cv2.imread(base_img_path, cv2.IMREAD_GRAYSCALE)

if base_img is None:
    print(f"Error: Could not load base image at {base_img_path}")
    exit(1)

# Initialize video writer
height, width = base_img.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 24  # Assuming 24 fps
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

for i in range(1, 51):
    filename = f'u78_test_suprise_v2_1_{i:04d}.png'
    img_path = os.path.join(input_folder, filename)
    
    # Read current frame
    current_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if current_img is None:
        print(f"Warning: Could not load {img_path}")
        continue
    
    # Calculate the absolute difference between the base frame and current frame
    diff = cv2.absdiff(base_img, current_img)
    
    # Optional: Apply a slight blur to reduce rendering noise artifacts
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    
    # To prevent rendering noise from blowing up when there is no movement,
    # we use a fixed scale instead of NORM_MINMAX. 
    # Adjust `max_expected_diff` to change the heatmap sensitivity.
    max_expected_diff = 10.0
    
    # Optional: threshold out tiny noise (e.g. differences of 1 or 2 due to anti-aliasing)
    diff[diff < 3] = 0
    
    diff_normalized = np.clip(diff * (255.0 / max_expected_diff), 0, 255).astype(np.uint8)
    
    # Apply a colormap (JET goes from blue for no movement, to red for high movement)
    heatmap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
    
    # Write the frame to the video
    out_video.write(heatmap)
    
    # Save intermediate frames to PNG
    if i in [28, 38, 50]:
        intermediate_png_path = os.path.join(output_folder, f'heatmap_frame_{i:04d}.png')
        cv2.imwrite(intermediate_png_path, heatmap)
        print(f"Saved intermediate frame: {intermediate_png_path}")

out_video.release()
print(f"Heatmap video generated successfully: {output_video_path}")