import cv2
import numpy as np
import os
import math

def create_comparison_video():
    seq1_dir = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub\temp_rendered_png_hub\muscle_closeup\beauty"
    seq2_dir = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub\temp_rendered_png_hub\muscle_closeup\beauty"
    
    seq1_prefix = "u54_test_V2_1_"
    seq2_prefix = "u55_test_V2_1_"
    
    output_dir = r"C:\Users\zpa18\a3-stochastic-V1\visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "overlap_comparison_heatmap_2.58s_v2.mp4")
    
    start_frame = 14
    end_frame = 734
    total_frames = end_frame - start_frame + 1
    
    # --- VIDEO LENGTH TUNING ---
    # Set the desired length of the final video in seconds
    desired_video_length_seconds = 2.58  # Tune this to change how long the video plays!
    
    if desired_video_length_seconds > 0:
        fps = math.ceil(total_frames / desired_video_length_seconds)
    else:
        fps = 30
    # ---------------------------
    
    # Check first frame to get dimensions
    first_frame_path = os.path.join(seq1_dir, f"{seq1_prefix}{start_frame:04d}.png")
    if not os.path.exists(first_frame_path):
        print(f"Error: Could not find {first_frame_path}")
        return
        
    sample_img = cv2.imread(first_frame_path)
    h, w, _ = sample_img.shape
    
    out_w = w * 3
    out_h = h
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    print(f"Generating comparison video to {output_path}...")
    
    for i in range(start_frame, end_frame + 1):
        if i % 50 == 0:
            print(f"Processing frame {i}/{end_frame}")
            
        img1_path = os.path.join(seq1_dir, f"{seq1_prefix}{i:04d}.png")
        img2_path = os.path.join(seq2_dir, f"{seq2_prefix}{i:04d}.png")
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"Warning: Missing frame {i}, skipping.")
            out.write(np.zeros((out_h, out_w, 3), dtype=np.uint8))
            continue
            
        diff = cv2.absdiff(img1, img2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
        
        scale_factor = 5.0
        enhanced_diff = cv2.convertScaleAbs(gray_diff, alpha=scale_factor, beta=0)
        
        heatmap = cv2.applyColorMap(enhanced_diff, cv2.COLORMAP_HOT)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        cv2.putText(img1, "0% Overlap", (30, 50), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(heatmap, "Difference Heatmap", (30, 50), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(img2, "40% Overlap", (30, 50), font, font_scale, (255, 255, 255), thickness)
        
        cv2.putText(heatmap, f"Frame: {i}", (30, 100), font, font_scale, (255, 255, 255), thickness)
        
        mean_err = np.mean(gray_diff)
        cv2.putText(heatmap, f"Mean Diff: {mean_err:.2f}", (30, 150), font, font_scale, (255, 255, 255), thickness)

        combined = np.hstack((img1, heatmap, img2))
        
        out.write(combined)
        
    out.release()
    print("Done! Video saved to:", output_path)

if __name__ == "__main__":
    create_comparison_video()
