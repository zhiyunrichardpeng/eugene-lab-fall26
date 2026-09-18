"""
Compare two sets of rendered animation frames (10% vs 40% overlap).
Produces:
  1. Per-frame SSIM & MSE time-series plot
  2. Amplified difference heatmap video (side-by-side montage)
  3. Per-frame pixel-difference histogram
  4. Summary statistics text file
  5. Key-frame montage (frames with largest differences)

Output goes to: <OUTPUT_DIR>/comparison_results/
"""

import os
import sys
import glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import csv
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub\temp_rendered_png_hub\muscle_closeup\beauty"

# 10% overlap
SET_A_PATTERN = os.path.join(BASE_DIR, "t48_for_fig1_arnold_1_{:04d}.png")
SET_A_LABEL = "10% Overlap"

# 40% overlap
SET_B_PATTERN = os.path.join(BASE_DIR, "t47_get_last_frame194_1_{:04d}.png")
SET_B_LABEL = "40% Overlap"

FRAME_START = 0
FRAME_END = 194  # inclusive
DIFF_AMPLIFICATION = 10  # multiplier for difference heatmap visibility

OUTPUT_DIR = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub\comparison_10pct_vs_40pct"

# ============================================================
# METRICS
# ============================================================

def compute_mse(img_a, img_b):
    """Mean Squared Error between two images (float arrays, 0-255)."""
    return np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)


def compute_mae(img_a, img_b):
    """Mean Absolute Error."""
    return np.mean(np.abs(img_a.astype(np.float64) - img_b.astype(np.float64)))


def compute_ssim_simple(img_a, img_b, C1=6.5025, C2=58.5225):
    """
    Simplified SSIM (no windowing, computed globally per channel then averaged).
    C1 = (0.01*255)^2, C2 = (0.03*255)^2
    """
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)

    mu_a = np.mean(a)
    mu_b = np.mean(b)
    sigma_a_sq = np.var(a)
    sigma_b_sq = np.var(b)
    sigma_ab = np.mean((a - mu_a) * (b - mu_b))

    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a_sq + sigma_b_sq + C2)

    return numerator / denominator


def compute_psnr(mse_val):
    """Peak Signal-to-Noise Ratio from MSE."""
    if mse_val < 1e-10:
        return 100.0  # identical
    return 10 * np.log10(255.0 ** 2 / mse_val)


def compute_diff_map(img_a, img_b, amplify=1):
    """Absolute difference map, optionally amplified."""
    diff = np.abs(img_a.astype(np.float64) - img_b.astype(np.float64))
    diff_amplified = np.clip(diff * amplify, 0, 255).astype(np.uint8)
    return diff_amplified


def compute_per_channel_stats(img_a, img_b):
    """Per-channel (R,G,B) mean absolute difference."""
    diff = np.abs(img_a.astype(np.float64) - img_b.astype(np.float64))
    return {
        'R': np.mean(diff[:, :, 0]),
        'G': np.mean(diff[:, :, 1]),
        'B': np.mean(diff[:, :, 2]),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("  ANIMATION COMPARISON: 10% Overlap vs 40% Overlap")
    print("=" * 70)
    print(f"  Frames: {FRAME_START} to {FRAME_END}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    # --- Pass 1: Compute per-frame metrics ---
    frames = list(range(FRAME_START, FRAME_END + 1))
    metrics = {
        'frame': [],
        'mse': [],
        'mae': [],
        'ssim': [],
        'psnr': [],
        'max_pixel_diff': [],
        'pct_pixels_changed': [],
        'R_mae': [],
        'G_mae': [],
        'B_mae': [],
    }

    print("Pass 1: Computing per-frame metrics...")
    for i, f_idx in enumerate(frames):
        path_a = SET_A_PATTERN.format(f_idx)
        path_b = SET_B_PATTERN.format(f_idx)

        if not os.path.exists(path_a):
            print(f"  [SKIP] Missing: {os.path.basename(path_a)}")
            continue
        if not os.path.exists(path_b):
            print(f"  [SKIP] Missing: {os.path.basename(path_b)}")
            continue

        img_a = np.array(Image.open(path_a).convert('RGB'))
        img_b = np.array(Image.open(path_b).convert('RGB'))

        # Resize if dimensions differ
        if img_a.shape != img_b.shape:
            h = min(img_a.shape[0], img_b.shape[0])
            w = min(img_a.shape[1], img_b.shape[1])
            img_a = np.array(Image.fromarray(img_a).resize((w, h)))
            img_b = np.array(Image.fromarray(img_b).resize((w, h)))

        mse = compute_mse(img_a, img_b)
        mae = compute_mae(img_a, img_b)
        ssim = compute_ssim_simple(img_a, img_b)
        psnr = compute_psnr(mse)
        max_diff = np.max(np.abs(img_a.astype(int) - img_b.astype(int)))
        pct_changed = np.mean(np.any(img_a != img_b, axis=2)) * 100

        ch_stats = compute_per_channel_stats(img_a, img_b)

        metrics['frame'].append(f_idx)
        metrics['mse'].append(mse)
        metrics['mae'].append(mae)
        metrics['ssim'].append(ssim)
        metrics['psnr'].append(psnr)
        metrics['max_pixel_diff'].append(max_diff)
        metrics['pct_pixels_changed'].append(pct_changed)
        metrics['R_mae'].append(ch_stats['R'])
        metrics['G_mae'].append(ch_stats['G'])
        metrics['B_mae'].append(ch_stats['B'])

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Frame {f_idx:4d}: MSE={mse:.2f}, MAE={mae:.2f}, SSIM={ssim:.6f}, Changed={pct_changed:.1f}%")

    if not metrics['frame']:
        print("ERROR: No frames were processed!")
        return

    print(f"\n  Processed {len(metrics['frame'])} frames.\n")

    # --- Save CSV ---
    csv_path = os.path.join(OUTPUT_DIR, f"per_frame_metrics_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(metrics.keys())
        for row in zip(*metrics.values()):
            writer.writerow(row)
    print(f"  Saved CSV: {csv_path}")

    # --- Plot 1: Time-series of all metrics (multi-panel) ---
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
    fig.suptitle('Animation Comparison: 10% Overlap vs 40% Overlap\n(Per-Frame Metrics Over Time)',
                 fontsize=16, fontweight='bold', y=0.98)

    x = metrics['frame']

    # Panel 1: MSE
    ax = axes[0]
    ax.plot(x, metrics['mse'], color='#e74c3c', linewidth=1.5, label='MSE')
    ax.fill_between(x, metrics['mse'], alpha=0.15, color='#e74c3c')
    ax.set_ylabel('Mean Squared Error', fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Pixel-Level Difference (MSE)', fontsize=12)

    # Panel 2: MAE
    ax = axes[1]
    ax.plot(x, metrics['mae'], color='#e67e22', linewidth=1.5, label='MAE (overall)')
    ax.plot(x, metrics['R_mae'], color='red', linewidth=0.8, alpha=0.6, label='R channel')
    ax.plot(x, metrics['G_mae'], color='green', linewidth=0.8, alpha=0.6, label='G channel')
    ax.plot(x, metrics['B_mae'], color='blue', linewidth=0.8, alpha=0.6, label='B channel')
    ax.set_ylabel('Mean Abs Error (0-255)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Mean Absolute Error (Overall + Per Channel)', fontsize=12)

    # Panel 3: SSIM
    ax = axes[2]
    ax.plot(x, metrics['ssim'], color='#2ecc71', linewidth=1.5, label='SSIM')
    ax.fill_between(x, metrics['ssim'], alpha=0.15, color='#2ecc71')
    ax.set_ylabel('SSIM (0=different, 1=same)', fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Structural Similarity Index (SSIM)', fontsize=12)

    # Panel 4: % pixels changed
    ax = axes[3]
    ax.plot(x, metrics['pct_pixels_changed'], color='#9b59b6', linewidth=1.5, label='% Pixels Changed')
    ax.fill_between(x, metrics['pct_pixels_changed'], alpha=0.15, color='#9b59b6')
    ax.set_ylabel('% Pixels Different', fontsize=11)
    ax.set_xlabel('Frame Number', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Percentage of Pixels That Differ', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot1_path = os.path.join(OUTPUT_DIR, f"metrics_timeseries_{timestamp}.png")
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved time-series plot: {plot1_path}")

    # --- Plot 2: Key-frame montage (top-N most different frames) ---
    N_KEY = 6
    sorted_indices = np.argsort(metrics['mse'])[::-1]  # highest MSE first
    top_indices = sorted_indices[:N_KEY]

    fig, axes = plt.subplots(N_KEY, 3, figsize=(18, N_KEY * 4))
    fig.suptitle(f'Top {N_KEY} Most Different Frames (Sorted by MSE)\n'
                 f'Left: {SET_A_LABEL} | Center: {SET_B_LABEL} | Right: Amplified Difference (×{DIFF_AMPLIFICATION})',
                 fontsize=14, fontweight='bold', y=1.01)

    for row, idx in enumerate(top_indices):
        f_idx = metrics['frame'][idx]
        img_a = np.array(Image.open(SET_A_PATTERN.format(f_idx)).convert('RGB'))
        img_b = np.array(Image.open(SET_B_PATTERN.format(f_idx)).convert('RGB'))

        if img_a.shape != img_b.shape:
            h = min(img_a.shape[0], img_b.shape[0])
            w = min(img_a.shape[1], img_b.shape[1])
            img_a = np.array(Image.fromarray(img_a).resize((w, h)))
            img_b = np.array(Image.fromarray(img_b).resize((w, h)))

        diff_map = compute_diff_map(img_a, img_b, amplify=DIFF_AMPLIFICATION)

        axes[row, 0].imshow(img_a)
        axes[row, 0].set_title(f'{SET_A_LABEL} — Frame {f_idx}', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(img_b)
        axes[row, 1].set_title(f'{SET_B_LABEL} — Frame {f_idx}', fontsize=10)
        axes[row, 1].axis('off')

        # Show diff as hot colormap for better visibility
        diff_gray = np.mean(diff_map.astype(float), axis=2)
        axes[row, 2].imshow(diff_gray, cmap='hot', vmin=0, vmax=255)
        axes[row, 2].set_title(
            f'Diff ×{DIFF_AMPLIFICATION} — MSE={metrics["mse"][idx]:.2f}, SSIM={metrics["ssim"][idx]:.4f}',
            fontsize=10)
        axes[row, 2].axis('off')

    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, f"key_frames_montage_{timestamp}.png")
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved key-frames montage: {plot2_path}")

    # --- Plot 3: Difference distribution histogram (aggregate) ---
    print("\nPass 2: Computing aggregate difference distribution...")
    all_diffs = []
    sample_frames = np.linspace(FRAME_START, FRAME_END, min(50, len(frames)), dtype=int)
    for f_idx in sample_frames:
        path_a = SET_A_PATTERN.format(f_idx)
        path_b = SET_B_PATTERN.format(f_idx)
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            continue
        img_a = np.array(Image.open(path_a).convert('RGB'))
        img_b = np.array(Image.open(path_b).convert('RGB'))
        if img_a.shape != img_b.shape:
            h = min(img_a.shape[0], img_b.shape[0])
            w = min(img_a.shape[1], img_b.shape[1])
            img_a = np.array(Image.fromarray(img_a).resize((w, h)))
            img_b = np.array(Image.fromarray(img_b).resize((w, h)))
        pixel_diffs = np.abs(img_a.astype(int) - img_b.astype(int)).flatten()
        all_diffs.extend(pixel_diffs[pixel_diffs > 0].tolist())  # only non-zero diffs

    fig, ax = plt.subplots(figsize=(12, 5))
    if all_diffs:
        ax.hist(all_diffs, bins=100, color='#3498db', alpha=0.8, edgecolor='#2c3e50', linewidth=0.3)
        ax.set_xlabel('Pixel Intensity Difference (0-255)', fontsize=12)
        ax.set_ylabel('Count (non-zero differences only)', fontsize=12)
        ax.set_title(f'Distribution of Pixel Differences (sampled from {len(sample_frames)} frames)',
                     fontsize=14, fontweight='bold')
        ax.axvline(np.median(all_diffs), color='red', linestyle='--', label=f'Median: {np.median(all_diffs):.1f}')
        ax.axvline(np.mean(all_diffs), color='orange', linestyle='--', label=f'Mean: {np.mean(all_diffs):.1f}')
        ax.legend(fontsize=11)
    else:
        ax.text(0.5, 0.5, 'No pixel differences found!\nThe two sets are identical.',
                transform=ax.transAxes, ha='center', va='center', fontsize=16)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot3_path = os.path.join(OUTPUT_DIR, f"diff_histogram_{timestamp}.png")
    plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved histogram: {plot3_path}")

    # --- Plot 4: Spatial heatmap of average difference ---
    print("\nPass 3: Computing spatial average difference heatmap...")
    acc_diff = None
    count = 0
    for f_idx in frames:
        path_a = SET_A_PATTERN.format(f_idx)
        path_b = SET_B_PATTERN.format(f_idx)
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            continue
        img_a = np.array(Image.open(path_a).convert('RGB')).astype(np.float64)
        img_b = np.array(Image.open(path_b).convert('RGB')).astype(np.float64)
        if img_a.shape != img_b.shape:
            h = min(img_a.shape[0], img_b.shape[0])
            w = min(img_a.shape[1], img_b.shape[1])
            img_a = np.array(Image.fromarray(img_a.astype(np.uint8)).resize((w, h))).astype(np.float64)
            img_b = np.array(Image.fromarray(img_b.astype(np.uint8)).resize((w, h))).astype(np.float64)

        diff = np.mean(np.abs(img_a - img_b), axis=2)  # grayscale diff
        if acc_diff is None:
            acc_diff = diff
        else:
            acc_diff += diff
        count += 1

    if acc_diff is not None and count > 0:
        avg_diff = acc_diff / count

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Spatial Distribution of Differences (Averaged Over All Frames)',
                     fontsize=14, fontweight='bold')

        # Raw average difference
        im1 = axes[0].imshow(avg_diff, cmap='hot', vmin=0)
        axes[0].set_title(f'Average Pixel Difference\n(max={avg_diff.max():.2f}, mean={avg_diff.mean():.2f})',
                         fontsize=11)
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Amplified for visibility
        im2 = axes[1].imshow(np.clip(avg_diff * DIFF_AMPLIFICATION, 0, 255), cmap='hot', vmin=0, vmax=255)
        axes[1].set_title(f'Amplified ×{DIFF_AMPLIFICATION}\n(highlights subtle differences)', fontsize=11)
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plot4_path = os.path.join(OUTPUT_DIR, f"spatial_heatmap_{timestamp}.png")
        plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved spatial heatmap: {plot4_path}")

    # --- Summary Report ---
    summary_path = os.path.join(OUTPUT_DIR, f"summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  ANIMATION COMPARISON SUMMARY\n")
        f.write(f"  {SET_A_LABEL} vs {SET_B_LABEL}\n")
        f.write(f"  Generated: {timestamp}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Frames compared: {len(metrics['frame'])}\n")
        f.write(f"Frame range: {FRAME_START} to {FRAME_END}\n\n")

        f.write("--- OVERALL STATISTICS ---\n")
        f.write(f"Mean MSE:              {np.mean(metrics['mse']):.4f}\n")
        f.write(f"Max  MSE:              {np.max(metrics['mse']):.4f} (frame {metrics['frame'][np.argmax(metrics['mse'])]})\n")
        f.write(f"Min  MSE:              {np.min(metrics['mse']):.4f} (frame {metrics['frame'][np.argmin(metrics['mse'])]})\n")
        f.write(f"Std  MSE:              {np.std(metrics['mse']):.4f}\n\n")

        f.write(f"Mean MAE:              {np.mean(metrics['mae']):.4f}\n")
        f.write(f"Max  MAE:              {np.max(metrics['mae']):.4f}\n\n")

        f.write(f"Mean SSIM:             {np.mean(metrics['ssim']):.6f}\n")
        f.write(f"Min  SSIM:             {np.min(metrics['ssim']):.6f} (frame {metrics['frame'][np.argmin(metrics['ssim'])]})\n")
        f.write(f"Max  SSIM:             {np.max(metrics['ssim']):.6f}\n\n")

        f.write(f"Mean PSNR:             {np.mean(metrics['psnr']):.2f} dB\n")
        f.write(f"Min  PSNR:             {np.min(metrics['psnr']):.2f} dB\n\n")

        f.write(f"Mean % pixels changed: {np.mean(metrics['pct_pixels_changed']):.2f}%\n")
        f.write(f"Max  % pixels changed: {np.max(metrics['pct_pixels_changed']):.2f}%\n\n")

        f.write(f"Max single-pixel diff: {np.max(metrics['max_pixel_diff'])} (out of 255)\n\n")

        # Interpretation
        mean_ssim = np.mean(metrics['ssim'])
        mean_mse = np.mean(metrics['mse'])
        f.write("--- INTERPRETATION ---\n")
        if mean_ssim > 0.999:
            f.write("The two animations are VIRTUALLY IDENTICAL.\n")
            f.write("Differences are likely due to rendering noise/anti-aliasing.\n")
        elif mean_ssim > 0.99:
            f.write("The two animations are VERY SIMILAR.\n")
            f.write("Differences are subtle and likely imperceptible to the human eye.\n")
        elif mean_ssim > 0.95:
            f.write("The two animations have MINOR DIFFERENCES.\n")
            f.write("Some differences may be noticeable upon close inspection.\n")
        elif mean_ssim > 0.90:
            f.write("The two animations have MODERATE DIFFERENCES.\n")
        else:
            f.write("The two animations have SIGNIFICANT DIFFERENCES.\n")

        f.write(f"\nMean MAE of {np.mean(metrics['mae']):.2f}/255 means on average each pixel differs\n")
        f.write(f"by {np.mean(metrics['mae'])/255*100:.3f}% of the full intensity range.\n")

        # Top different frames
        f.write(f"\n--- TOP {N_KEY} MOST DIFFERENT FRAMES ---\n")
        for rank, idx in enumerate(top_indices):
            f_idx = metrics['frame'][idx]
            f.write(f"  #{rank+1}: Frame {f_idx:4d} — MSE={metrics['mse'][idx]:.2f}, "
                    f"SSIM={metrics['ssim'][idx]:.6f}, "
                    f"Changed={metrics['pct_pixels_changed'][idx]:.1f}%\n")

    print(f"  Saved summary: {summary_path}")

    # --- Print summary to console ---
    with open(summary_path, 'r') as f:
        print("\n" + f.read())

    print("\n" + "=" * 70)
    print(f"  ALL RESULTS SAVED TO: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
