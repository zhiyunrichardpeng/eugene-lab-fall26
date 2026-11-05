"""
Compare two sets of rendered animation frames (0% Overlap vs 10% Overlap).
Handles differing frame counts across the same time duration using
time-normalized frame mapping (proportional nearest-neighbor indexing).

Produces:
  1. Per-frame SSIM & MSE time-series plot
  2. Side-by-side comparison video (0% vs 10% + amplified difference heatmap)
  3. Per-frame pixel-difference histogram
  4. Summary statistics text file
  5. Key-frame montage (frames with largest differences)

Output goes to: <OUTPUT_DIR>/
"""

import os
import sys
import glob
import re
import math
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import csv
from datetime import datetime
from pathlib import Path

# Add repository root containing paths.py to sys.path
for parent in Path(__file__).resolve().parents:
    if (parent / "paths.py").is_file():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from paths import results_path

# ============================================================
# CONFIGURATION
# ============================================================
RESULTS_DIR = Path(r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003")
if not RESULTS_DIR.exists():
    RESULTS_DIR = results_path

COMPOSITED_BASE = RESULTS_DIR / "maya_render_output_hub/temp_rendered_png_hub/masterLayer/muscle_closeup/composited_output"

# 0% overlap (Baseline / timekeeper sequence: 204 timeline frames / 190 rendered frames on disk)
BASE_DIR_A = str(COMPOSITED_BASE / "test_0pct_render_250_V1")
SET_A_PATTERN = os.path.join(BASE_DIR_A, "test_0pct_render_250_V1_{:04d}.png")
SET_A_LABEL = "0% Overlap"

# 10% overlap (Mapped sequence: 186 frames)
BASE_DIR_B = str(COMPOSITED_BASE / "test_10pct_render_250_V1")
SET_B_PATTERN = os.path.join(BASE_DIR_B, "test_10pct_render_250_V1_{:04d}.png")
SET_B_LABEL = "10% Overlap"

# --- Frame Range / Mapping Configuration ---
# MAPPING_MODE:
#   "direct": Compares exact same frame numbers 1:1 (Frame N vs Frame N).
#             Isolates localized physical muscle differences (e.g. lips) as bright red
#             against a pitch-black background, eliminating motion-phase blur.
#   "time_normalized": Maps sequences across different frame counts using proportional indexing.
MAPPING_MODE = "direct"

# If AUTO_DETECT is True: discovers all available frames on disk matching each pattern.
# If AUTO_DETECT is False: uses the explicit frame ranges below.
AUTO_DETECT = False

# Sequence A frames (0% overlap)
FRAME_START_A = 15
FRAME_END_A = 204

# Sequence B frames (10% overlap)
FRAME_START_B = 15
FRAME_END_B = 186

DIFF_AMPLIFICATION = 10  # multiplier for difference heatmap visibility

OUTPUT_DIR = str(RESULTS_DIR / "maya_render_output_hub/comparison_0pct_vs_10pct")

# Comparison Video Generation
GENERATE_VIDEO = True
DESIRED_VIDEO_LENGTH_SECONDS = 0.516  # Both sequences represent 0.516 seconds of simulation
INSPECTION_FPS = 30                   # Slower framerate for human visual inspection

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

    return float(numerator / denominator)


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
        'R': float(np.mean(diff[:, :, 0])),
        'G': float(np.mean(diff[:, :, 1])),
        'B': float(np.mean(diff[:, :, 2])),
    }


# ============================================================
# FRAME PAIR BUILDER
# ============================================================

def build_frame_pairs():
    """
    Builds synchronized frame pairs between Sequence A and Sequence B.
    - 'direct': 1:1 frame-by-frame comparison (Frame N vs Frame N).
      Recommended for physical simulation comparisons to isolate localized
      muscle deformation (e.g. lips) on a clean black background.
    - 'time_normalized': Nearest-neighbor proportional indexing across
      differing duration frame counts.
    """
    if AUTO_DETECT:
        def extract_nums(pattern, folder):
            base_pat = os.path.basename(pattern).replace("{:04d}", r"(\d+)")
            regex = re.compile(base_pat)
            nums = []
            if os.path.isdir(folder):
                for fname in os.listdir(folder):
                    m = regex.match(fname)
                    if m:
                        nums.append(int(m.group(1)))
            return sorted(nums)
        frames_a = extract_nums(SET_A_PATTERN, BASE_DIR_A)
        frames_b = extract_nums(SET_B_PATTERN, BASE_DIR_B)
    else:
        frames_a = list(range(FRAME_START_A, FRAME_END_A + 1))
        frames_b = list(range(FRAME_START_B, FRAME_END_B + 1))

    if not frames_a or not frames_b:
        raise ValueError(
            f"Frame list is empty! Sequence A count: {len(frames_a)}, Sequence B count: {len(frames_b)}"
        )

    pairs = []
    if MAPPING_MODE == "direct":
        common_frames = sorted(list(set(frames_a).intersection(set(frames_b))))
        total = len(common_frames)
        for i, f in enumerate(common_frames):
            progress = i / float(total - 1) if total > 1 else 0.0
            path_a = SET_A_PATTERN.format(f)
            path_b = SET_B_PATTERN.format(f)
            pairs.append({
                'index': i,
                'progress': progress,
                'frame_a': f,
                'frame_b': f,
                'path_a': path_a,
                'path_b': path_b,
            })
        return common_frames, common_frames, pairs
    else:
        total_a = len(frames_a)
        total_b = len(frames_b)
        for i in range(total_a):
            progress = i / float(total_a - 1) if total_a > 1 else 0.0
            j = int(round(progress * (total_b - 1))) if total_b > 1 else 0
            f_a = frames_a[i]
            f_b = frames_b[j]
            path_a = SET_A_PATTERN.format(f_a)
            path_b = SET_B_PATTERN.format(f_b)
            pairs.append({
                'index': i,
                'progress': progress,
                'frame_a': f_a,
                'frame_b': f_b,
                'path_a': path_a,
                'path_b': path_b,
            })
        return frames_a, frames_b, pairs


# ============================================================
# VIDEO GENERATION HELPER
# ============================================================

def generate_comparison_videos(valid_pairs, output_dir, timestamp):
    """
    Generates side-by-side comparison videos:
    [ Set A (0%) | Set B (10%) | Amplified Difference Heatmap ]
    Outputs:
      1. Realtime playback (~0.516s) matching simulation speed
      2. Inspection playback (30 FPS) for slow-motion frame-by-frame review
    """
    if not valid_pairs:
        return

    # Read first pair to get dimensions
    sample_a = cv2.imread(valid_pairs[0]['path_a'])
    if sample_a is None:
        return
    h, w = sample_a.shape[:2]

    # Combined frame will be 3 panels wide
    panel_w = w
    panel_h = h
    montage_w = panel_w * 3
    montage_h = panel_h

    # Framerate for 0.516s simulation duration
    total_frames = len(valid_pairs)
    realtime_fps = max(1, math.ceil(total_frames / DESIRED_VIDEO_LENGTH_SECONDS))

    video_configs = [
        (f"comparison_side_by_side_realtime_{timestamp}.mp4", realtime_fps, "Simulation Speed (~0.516s)"),
        (f"comparison_side_by_side_30fps_{timestamp}.mp4", INSPECTION_FPS, "Slow-Motion Inspection (30 FPS)"),
    ]

    print("\nPass 4: Rendering side-by-side comparison videos...")

    # We write frames once to a temporary image sequence or directly with cv2.VideoWriter
    writers = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for fname, fps, desc in video_configs:
        vpath = os.path.join(output_dir, fname)
        writer = cv2.VideoWriter(vpath, fourcc, fps, (montage_w, montage_h))
        writers.append((writer, vpath, fps, desc))

    for pair in valid_pairs:
        img_a = cv2.imread(pair['path_a'])
        img_b = cv2.imread(pair['path_b'])

        if img_a is None or img_b is None:
            continue

        if img_a.shape != (panel_h, panel_w, 3):
            img_a = cv2.resize(img_a, (panel_w, panel_h))
        if img_b.shape != (panel_h, panel_w, 3):
            img_b = cv2.resize(img_b, (panel_w, panel_h))

        # Compute amplified difference map in color
        diff = np.abs(img_a.astype(np.float32) - img_b.astype(np.float32))
        diff_amplified = np.clip(diff * DIFF_AMPLIFICATION, 0, 255).astype(np.uint8)

        # Apply colormap to grayscale diff for visual clarity
        diff_gray = cv2.cvtColor(diff_amplified, cv2.COLOR_BGR2GRAY)
        diff_color = cv2.applyColorMap(diff_gray, cv2.COLORMAP_HOT)

        # Annotate panels
        txt_a = f"{SET_A_LABEL} - Frame {pair['frame_a']:04d}"
        if pair['frame_a'] == pair['frame_b']:
            txt_b = f"{SET_B_LABEL} - Frame {pair['frame_b']:04d}"
            txt_diff = f"Diff x{DIFF_AMPLIFICATION} (1:1 Frame Match)"
        else:
            txt_b = f"{SET_B_LABEL} - Frame {pair['frame_b']:04d} (mapped)"
            txt_diff = f"Diff x{DIFF_AMPLIFICATION} | Progress: {pair['progress']*100:5.1f}%"

        cv2.putText(img_a, txt_a, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_b, txt_b, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(diff_color, txt_diff, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Montage: Left (0%) | Center (10%) | Right (Difference)
        montage = np.hstack([img_a, img_b, diff_color])

        for writer, _, _, _ in writers:
            writer.write(montage)

    for writer, vpath, fps, desc in writers:
        writer.release()
        print(f"  Saved video [{desc}]:\n   -> {vpath}")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    frames_a, frames_b, pairs = build_frame_pairs()
    total_a = len(frames_a)
    total_b = len(frames_b)

    print("=" * 70)
    print(f"  ANIMATION COMPARISON: {SET_A_LABEL} vs {SET_B_LABEL}")
    print("=" * 70)
    print(f"  Set A ({SET_A_LABEL}): {total_a} frames (Range: {frames_a[0]} to {frames_a[-1]})")
    print(f"  Set B ({SET_B_LABEL}): {total_b} frames (Range: {frames_b[0]} to {frames_b[-1]})")
    print(f"  Timekeeper Baseline:   {SET_A_LABEL} ({total_a} steps, duration: {DESIRED_VIDEO_LENGTH_SECONDS}s)")
    print(f"  Synchronization:       Time-normalized proportional mapping")
    print(f"  Output Directory:      {OUTPUT_DIR}")
    print()

    # --- Pass 1: Compute per-frame metrics ---
    metrics = {
        'index': [],
        'progress': [],
        'frame_a': [],
        'frame_b': [],
        'frame': [],  # For backwards compatibility with older plot scripts
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

    valid_pairs = []
    print("Pass 1: Computing per-frame metrics with time-normalized mapping...")
    for pair in pairs:
        path_a = pair['path_a']
        path_b = pair['path_b']

        if not os.path.exists(path_a):
            print(f"  [SKIP] Missing Set A: {os.path.basename(path_a)}")
            continue
        if not os.path.exists(path_b):
            print(f"  [SKIP] Missing Set B: {os.path.basename(path_b)}")
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
        max_diff = int(np.max(np.abs(img_a.astype(int) - img_b.astype(int))))
        pct_changed = float(np.mean(np.any(img_a != img_b, axis=2)) * 100)

        ch_stats = compute_per_channel_stats(img_a, img_b)

        metrics['index'].append(pair['index'])
        metrics['progress'].append(pair['progress'])
        metrics['frame_a'].append(pair['frame_a'])
        metrics['frame_b'].append(pair['frame_b'])
        metrics['frame'].append(pair['frame_a'])
        metrics['mse'].append(mse)
        metrics['mae'].append(mae)
        metrics['ssim'].append(ssim)
        metrics['psnr'].append(psnr)
        metrics['max_pixel_diff'].append(max_diff)
        metrics['pct_pixels_changed'].append(pct_changed)
        metrics['R_mae'].append(ch_stats['R'])
        metrics['G_mae'].append(ch_stats['G'])
        metrics['B_mae'].append(ch_stats['B'])
        valid_pairs.append(pair)

        i = pair['index']
        if (i + 1) % 20 == 0 or i == 0 or i == total_a - 1:
            print(f"  Step {i:3d} (A:{pair['frame_a']:04d} -> B:{pair['frame_b']:04d}, {pair['progress']*100:5.1f}%): "
                  f"MSE={mse:.2f}, MAE={mae:.2f}, SSIM={ssim:.6f}, Changed={pct_changed:.1f}%")

    if not metrics['index']:
        print("ERROR: No valid frames were processed!")
        return

    print(f"\n  Successfully processed {len(valid_pairs)} synchronized frame pairs.\n")

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
    fig.suptitle(f'Animation Comparison: {SET_A_LABEL} ({total_a}f) vs {SET_B_LABEL} ({total_b}f)\n'
                 f'(Time-Normalized Metrics Over 0.516s Simulation Timeline)',
                 fontsize=16, fontweight='bold', y=0.98)

    x = metrics['frame_a']

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
    ax.set_xlabel(f'Baseline Frame Index ({SET_A_LABEL}) [Normalized Progress: 0% -> 100%]', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Percentage of Pixels That Differ', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot1_path = os.path.join(OUTPUT_DIR, f"metrics_timeseries_{timestamp}.png")
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved time-series plot: {plot1_path}")

    # --- Plot 2: Key-frame montage (top-N most different frames) ---
    N_KEY = min(6, len(valid_pairs))
    sorted_indices = np.argsort(metrics['mse'])[::-1]  # highest MSE first
    top_indices = sorted_indices[:N_KEY]

    fig, axes = plt.subplots(N_KEY, 3, figsize=(18, N_KEY * 4))
    if N_KEY == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(f'Top {N_KEY} Most Different Frames (Sorted by MSE)\n'
                 f'Left: {SET_A_LABEL} | Center: {SET_B_LABEL} (Time-Mapped) | Right: Amplified Difference (×{DIFF_AMPLIFICATION})',
                 fontsize=14, fontweight='bold', y=1.01)

    for row, idx in enumerate(top_indices):
        pair = valid_pairs[idx]
        img_a = np.array(Image.open(pair['path_a']).convert('RGB'))
        img_b = np.array(Image.open(pair['path_b']).convert('RGB'))

        if img_a.shape != img_b.shape:
            h = min(img_a.shape[0], img_b.shape[0])
            w = min(img_a.shape[1], img_b.shape[1])
            img_a = np.array(Image.fromarray(img_a).resize((w, h)))
            img_b = np.array(Image.fromarray(img_b).resize((w, h)))

        diff_map = compute_diff_map(img_a, img_b, amplify=DIFF_AMPLIFICATION)

        axes[row, 0].imshow(img_a)
        axes[row, 0].set_title(f'{SET_A_LABEL} — Frame {pair["frame_a"]}', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(img_b)
        b_title = f'{SET_B_LABEL} — Frame {pair["frame_b"]}' + (' (mapped)' if pair['frame_a'] != pair['frame_b'] else '')
        axes[row, 1].set_title(b_title, fontsize=10)
        axes[row, 1].axis('off')

        diff_gray = np.mean(diff_map.astype(float), axis=2)
        axes[row, 2].imshow(diff_gray, cmap='hot', vmin=0, vmax=255)
        axes[row, 2].set_title(
            f'Diff ×{DIFF_AMPLIFICATION} — MSE={metrics["mse"][idx]:.2f}, SSIM={metrics["ssim"][idx]:.4f} ({pair["progress"]*100:.1f}%)',
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
    sample_indices = np.linspace(0, len(valid_pairs) - 1, min(50, len(valid_pairs)), dtype=int)
    for idx in sample_indices:
        pair = valid_pairs[idx]
        img_a = np.array(Image.open(pair['path_a']).convert('RGB'))
        img_b = np.array(Image.open(pair['path_b']).convert('RGB'))
        if img_a.shape != img_b.shape:
            h = min(img_a.shape[0], img_b.shape[0])
            w = min(img_a.shape[1], img_b.shape[1])
            img_a = np.array(Image.fromarray(img_a).resize((w, h)))
            img_b = np.array(Image.fromarray(img_b).resize((w, h)))
        pixel_diffs = np.abs(img_a.astype(int) - img_b.astype(int)).flatten()
        all_diffs.extend(pixel_diffs[pixel_diffs > 0].tolist())

    fig, ax = plt.subplots(figsize=(12, 5))
    if all_diffs:
        ax.hist(all_diffs, bins=100, color='#3498db', alpha=0.8, edgecolor='#2c3e50', linewidth=0.3)
        ax.set_xlabel('Pixel Intensity Difference (0-255)', fontsize=12)
        ax.set_ylabel('Count (non-zero differences only)', fontsize=12)
        ax.set_title(f'Distribution of Pixel Differences (sampled from {len(sample_indices)} aligned frames)',
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
    for pair in valid_pairs:
        img_a = np.array(Image.open(pair['path_a']).convert('RGB')).astype(np.float64)
        img_b = np.array(Image.open(pair['path_b']).convert('RGB')).astype(np.float64)
        if img_a.shape != img_b.shape:
            h = min(img_a.shape[0], img_b.shape[0])
            w = min(img_a.shape[1], img_b.shape[1])
            img_a = np.array(Image.fromarray(img_a.astype(np.uint8)).resize((w, h))).astype(np.float64)
            img_b = np.array(Image.fromarray(img_b.astype(np.uint8)).resize((w, h))).astype(np.float64)

        diff = np.mean(np.abs(img_a - img_b), axis=2)
        if acc_diff is None:
            acc_diff = diff
        else:
            acc_diff += diff
        count += 1

    if acc_diff is not None and count > 0:
        avg_diff = acc_diff / count

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Spatial Distribution of Differences (Averaged Over All Aligned Frames)',
                     fontsize=14, fontweight='bold')

        im1 = axes[0].imshow(avg_diff, cmap='hot', vmin=0)
        axes[0].set_title(f'Average Pixel Difference\n(max={avg_diff.max():.2f}, mean={avg_diff.mean():.2f})',
                         fontsize=11)
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(np.clip(avg_diff * DIFF_AMPLIFICATION, 0, 255), cmap='hot', vmin=0, vmax=255)
        axes[1].set_title(f'Amplified ×{DIFF_AMPLIFICATION}\n(highlights subtle simulation differences)', fontsize=11)
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
        f.write(f"  Baseline Sequence:   {SET_A_LABEL} ({total_a} frames, {frames_a[0]}..{frames_a[-1]})\n")
        f.write(f"  Mapped Sequence:     {SET_B_LABEL} ({total_b} frames, {frames_b[0]}..{frames_b[-1]})\n")
        f.write(f"  Time Alignment:      Nearest-neighbor proportional indexing across {DESIRED_VIDEO_LENGTH_SECONDS}s\n")
        f.write(f"  Generated:           {timestamp}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Synchronized Pairs Compared: {len(metrics['index'])}\n\n")

        f.write("--- OVERALL STATISTICS ---\n")
        f.write(f"Mean MSE:              {np.mean(metrics['mse']):.4f}\n")
        max_idx = int(np.argmax(metrics['mse']))
        min_idx = int(np.argmin(metrics['mse']))
        f.write(f"Max  MSE:              {np.max(metrics['mse']):.4f} (Step {metrics['index'][max_idx]}: SetA F{metrics['frame_a'][max_idx]} vs SetB F{metrics['frame_b'][max_idx]})\n")
        f.write(f"Min  MSE:              {np.min(metrics['mse']):.4f} (Step {metrics['index'][min_idx]}: SetA F{metrics['frame_a'][min_idx]} vs SetB F{metrics['frame_b'][min_idx]})\n")
        f.write(f"Std  MSE:              {np.std(metrics['mse']):.4f}\n\n")

        f.write(f"Mean MAE:              {np.mean(metrics['mae']):.4f}\n")
        f.write(f"Max  MAE:              {np.max(metrics['mae']):.4f}\n\n")

        min_ssim_idx = int(np.argmin(metrics['ssim']))
        f.write(f"Mean SSIM:             {np.mean(metrics['ssim']):.6f}\n")
        f.write(f"Min  SSIM:             {np.min(metrics['ssim']):.6f} (Step {metrics['index'][min_ssim_idx]}: SetA F{metrics['frame_a'][min_ssim_idx]} vs SetB F{metrics['frame_b'][min_ssim_idx]})\n")
        f.write(f"Max  SSIM:             {np.max(metrics['ssim']):.6f}\n\n")

        f.write(f"Mean PSNR:             {np.mean(metrics['psnr']):.2f} dB\n")
        f.write(f"Min  PSNR:             {np.min(metrics['psnr']):.2f} dB\n\n")

        f.write(f"Mean % pixels changed: {np.mean(metrics['pct_pixels_changed']):.2f}%\n")
        f.write(f"Max  % pixels changed: {np.max(metrics['pct_pixels_changed']):.2f}%\n\n")

        f.write(f"Max single-pixel diff: {np.max(metrics['max_pixel_diff'])} (out of 255)\n\n")

        # Interpretation
        mean_ssim = float(np.mean(metrics['ssim']))
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
            p = valid_pairs[idx]
            f.write(f"  #{rank+1}: SetA Frame {p['frame_a']:4d} vs SetB Frame {p['frame_b']:4d} "
                    f"({p['progress']*100:5.1f}%) — MSE={metrics['mse'][idx]:.2f}, "
                    f"SSIM={metrics['ssim'][idx]:.6f}, "
                    f"Changed={metrics['pct_pixels_changed'][idx]:.1f}%\n")

    print(f"  Saved summary: {summary_path}")

    # --- Video Generation ---
    if GENERATE_VIDEO:
        generate_comparison_videos(valid_pairs, OUTPUT_DIR, timestamp)

    # --- Print summary to console ---
    with open(summary_path, 'r') as f:
        print("\n" + f.read())

    print("\n" + "=" * 70)
    print(f"  ALL RESULTS SAVED TO: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
