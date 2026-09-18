"""
Compare two sets of rendered animation frames (e.g., 10% overlap vs 40% overlap).
Produces a visual side-by-side comparison, difference heatmap, and summary text file.

How to use:
1. Update SET_A_PATTERN and SET_B_PATTERN with the paths to your two image sets.
2. Ensure the frame numbering `{f_idx:04d}` aligns with your files.
3. Update OUTPUT_DIR to where you want the results saved.
4. Run this script in Python.
"""

import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import csv

# ============================================================
# CONFIGURATION - CHANGE THESE PATHS
# ============================================================
BASE_DIR = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub\temp_rendered_png_hub\muscle_closeup\beauty"

# Set A path pattern (e.g., your 10% overlap results)
# The "{f_idx:04d}" part will be replaced by the frame number like 0000, 0001, etc.
SET_A_PATTERN = os.path.join(BASE_DIR, "t48_for_fig1_arnold_1_{f_idx:04d}.png")
# SET_A_PATTERN = os.path.join(BASE_DIR, "t48_for_fig1_arnold_1_0194.png")
SET_A_LABEL = "Set A (0% Overlap)"

# Set B path pattern (e.g., your 40% overlap results)
# SET_B_PATTERN = os.path.join(BASE_DIR, "40pct_overlap_1.png")
SET_B_PATTERN = os.path.join(BASE_DIR, "40pct_overlap_animation_1_{f_idx:04d}.png")
SET_B_LABEL = "Set B (40% Overlap)"

FRAME_START = 0
FRAME_END = 194  # inclusive
DIFF_AMPLIFICATION = 10  # multiplier for difference heatmap visibility

# Where to save the comparison results
OUTPUT_DIR = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub\comparison_results"

# ============================================================
# METRICS FUNCTIONS
# ============================================================

def compute_mse(img_a, img_b):
    """Mean Squared Error between two images."""
    return np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)

def compute_mae(img_a, img_b):
    """Mean Absolute Error."""
    return np.mean(np.abs(img_a.astype(np.float64) - img_b.astype(np.float64)))

def compute_diff_map(img_a, img_b, amplify=1):
    """Absolute difference map, optionally amplified."""
    diff = np.abs(img_a.astype(np.float64) - img_b.astype(np.float64))
    diff_amplified = np.clip(diff * amplify, 0, 255).astype(np.uint8)
    return diff_amplified

# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 70)
    print(f"  COMPARING: {SET_A_LABEL} vs {SET_B_LABEL}")
    print("=" * 70)
    print(f" Output folder: {OUTPUT_DIR}\n")

    # Auto-detect if we are processing a sequence or a single file
    is_sequence = "{f_idx" in SET_A_PATTERN and "{f_idx" in SET_B_PATTERN
    frames = list(range(FRAME_START, FRAME_END + 1)) if is_sequence else [0]
    
    metrics = {'frame': [], 'mse': [], 'mae': [], 'max_diff': []}
    acc_diff_map = None
    processed_count = 0

    print("Analyzing frames...")
    
    for i, f_idx in enumerate(frames):
        if is_sequence:
            path_a = SET_A_PATTERN.format(f_idx=f_idx)
            path_b = SET_B_PATTERN.format(f_idx=f_idx)
            disp_frame = f_idx
        else:
            path_a = SET_A_PATTERN
            path_b = SET_B_PATTERN
            disp_frame = "Single_Image"

        if not os.path.exists(path_a) or not os.path.exists(path_b):
            continue

        # Load images
        img_a = np.array(Image.open(path_a).convert('RGB'))
        img_b = np.array(Image.open(path_b).convert('RGB'))

        # Resize if dimensions differ
        if img_a.shape != img_b.shape:
            h = min(img_a.shape[0], img_b.shape[0])
            w = min(img_a.shape[1], img_b.shape[1])
            img_a = np.array(Image.fromarray(img_a).resize((w, h)))
            img_b = np.array(Image.fromarray(img_b).resize((w, h)))

        # Calculate difference metrics
        mse = compute_mse(img_a, img_b)
        mae = compute_mae(img_a, img_b)
        max_diff = np.max(np.abs(img_a.astype(int) - img_b.astype(int)))
        
        metrics['frame'].append(disp_frame)
        metrics['mse'].append(mse)
        metrics['mae'].append(mae)
        metrics['max_diff'].append(max_diff)

        # Accumulate spatial difference for the heatmap
        grayscale_diff = np.mean(np.abs(img_a.astype(float) - img_b.astype(float)), axis=2)
        if acc_diff_map is None:
            acc_diff_map = grayscale_diff
        else:
            acc_diff_map += grayscale_diff

        processed_count += 1
        if (i + 1) % 20 == 0 or i == 0 or not is_sequence:
            print(f"  Frame {disp_frame}: MSE={mse:.2f}, Max Pixel Difference={max_diff}")

    if processed_count == 0:
        print("ERROR: No matching frames found to compare. Check your paths!")
        return

    # --- 1. Save Text Summary ---
    mean_mse = np.mean(metrics['mse'])
    mean_mae = np.mean(metrics['mae'])
    
    summary_path = os.path.join(OUTPUT_DIR, f"summary_report_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write(" ANIMATION COMPARISON SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Set A: {SET_A_LABEL}\n")
        f.write(f"Set B: {SET_B_LABEL}\n")
        f.write(f"Frames compared: {processed_count}\n\n")
        
        f.write("--- DIFFERENCE STATISTICS ---\n")
        f.write(f"Average Mean Squared Error (MSE): {mean_mse:.4f}\n")
        f.write(f"Average Mean Absolute Error (MAE): {mean_mae:.4f} (out of 255)\n")
        f.write(f"Highest single pixel difference: {np.max(metrics['max_diff'])} (out of 255)\n\n")
        
        if mean_mse == 0:
            f.write("CONCLUSION: The two animations are PIXEL-FOR-PIXEL IDENTICAL.\n")
        elif mean_mse < 10:
            f.write("CONCLUSION: The animations have extremely minor differences (likely imperceptible).\n")
        else:
            f.write("CONCLUSION: The animations have noticeable physical differences.\n")
            
    print(f"\nSaved summary to: {os.path.basename(summary_path)}")

    # --- 2. Save Time-Series Plot (Sequence Only) ---
    if is_sequence and processed_count > 1:
        plt.figure(figsize=(10, 4))
        plt.plot(metrics['frame'], metrics['mse'], color='red')
        plt.title('Difference Over Time (Mean Squared Error)')
        plt.xlabel('Frame Number')
        plt.ylabel('MSE (Higher = More Different)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(OUTPUT_DIR, f"difference_over_time_{timestamp}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved graph to:   {os.path.basename(plot_path)}")

    # --- 3. Save Overall Average Heatmap ---
    avg_diff_map = acc_diff_map / processed_count
    plt.figure(figsize=(8, 6))
    plt.imshow(np.clip(avg_diff_map * DIFF_AMPLIFICATION, 0, 255), cmap='hot', vmin=0, vmax=255)
    plt.title(f'Average Pixel Change (Amplified {DIFF_AMPLIFICATION}x)\nHotter color = bigger difference')
    plt.axis('off')
    plt.colorbar()
    
    heatmap_path = os.path.join(OUTPUT_DIR, f"heatmap_average_diff_{timestamp}.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {os.path.basename(heatmap_path)}")
    
    print("\nCompare script finished successfully!")

if __name__ == "__main__":
    main()
