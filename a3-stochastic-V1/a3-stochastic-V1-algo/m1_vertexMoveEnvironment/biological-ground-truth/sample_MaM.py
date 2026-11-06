"""
=============================================================================
Plot: 
=============================================================================
SIGGRAPH TOG submission style — Times New Roman, thick lines, clean axes.

Usage:
    python plotMaM.py

Output:
    C:/Users/zpa18/a3-stochastic-V1/visualization/skin_distance_to_target.png
    C:/Users/zpa18/a3-stochastic-V1/visualization/skin_distance_to_target.pdf
=============================================================================
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random
from collections import defaultdict


# ===================== CONFIGURATION =====================
INPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "MaM_Public", "MaM_sample.csv"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "visualization"
)
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "skin_distance_to_target.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "skin_distance_to_target.pdf")
DPI = 600
# =========================================================
"""
AVAILABLE EXPRESSIONS
+---------------+--------------+
| Maya          |   MaM        | 
| 1. Neutral    | Face-at-Rest |
| 2. Smile      | Smile-Open   |
| 3. Happy      | happy        |
| 4. Sad        | sad          | 
| 5. Frown      | angry
| 6. Surprise   | surprised
| 7. Squint     | Eye-Gentle
| 8. Raise-brow | Forehead
+---------------+----------------

emg/*.csv contain:
1 row per frame [Clips are 6 s (180 frames at 30 fps)]
emg/{pid}_{rec}/{identifier}.csv holds an unnamed integer frame index starting at 0, 
then 22 float columns at three decimals, one row per video frame at 30 Hz. 
The channels are 11 muscles on two sides (_l and _r), named from the recording protocol and partly German. 
Column order is the same in every file: alphabetical, with _l before _r.

sEMG row i matches video frame i, aligned at the clip start. 
The sEMG can be shorter than the video, by up to about 14 frames in schaede clips, and the video is never shorter. 
Clamp both to min(len(emg), n_video_frames)
"""
random.seed(42)
expressions = ["Smile-Open", "sad", "angry", "forehead", "happy"]
# ---------- Load data ----------
participants = defaultdict(lambda: defaultdict(list))
REC = 2
TASK = 4
PARTICIPANTS = 10
"""
x-axis: time
y-axis: muscle activation inensity in microvolts
We're interested in the muscles:
    - orbicularis oculi (OOC) check
    - orbicularis oris (OOR) check
    - (Lateraler) frontalis check
    - zygomaticus major check
    - zygomaticus minor check
"""
with open(INPUT_CSV, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        path = row["emg"]
        expression = path.split("/")[REC].split("_")[TASK].split(".")[0]
        if expression not in expressions:
            continue
        folder = path.split("/")[1]
        pid, session = folder.split("_")
        participants[pid][session].append(row)
valid = {
    pid: sessions for pid, sessions in participants.items() if len(sessions) >= 2
}

selected_pids = random.sample(list(valid.keys()), PARTICIPANTS)

sampled_rows = []
for pid in selected_pids:
    sessions = valid[pid]
    selected_sessions = random.sample(
        list(sessions.keys()),
        min(2, len(sessions))
    )

    for session in selected_sessions:
        sampled_rows.extend(sessions[session])

# Display selected files
print(f"Total csvs recovered: {len(sampled_rows)}")
for row in sampled_rows:
    print(row["expression"], row["emg"])
"""
write(row["expression])
"""


# sad = [expr for expr in sampled_rows if "sad" in expr["expression"]]
# happy = []
# angry = []
# smile = []
# forehead = []

# print(f"Loaded {len(frames)} data points from {INPUT_CSV}")
# print(f"  Frame range : {frames[0]} -> {frames[-1]}")
# print(f"  Distance range : {distances.min():.4f} - {distances.max():.4f} cm")

# # ================================================================
# #  SIGGRAPH / TOG  style  –  Times New Roman, thick lines, clean
# # ================================================================
# # Force matplotlib to use Type-1 / TrueType fonts (no bitmap / Type-3)
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType in PDF
matplotlib.rcParams["ps.fonttype"] = 42    # TrueType in PS/EPS

plt.rcParams.update({
    # ---- Font ----
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "stix",          # math symbols match Times
    "font.size":          18,              # base font size

    # ---- Axes ----
    "axes.linewidth":     1.8,
    "axes.labelsize":     20,
    "axes.titlesize":     20,
    "axes.labelweight":   "bold",
    "axes.labelpad":      8,

    # ---- Ticks ----
    "xtick.major.size":   7,
    "xtick.major.width":  1.8,
    "xtick.minor.size":   4,
    "xtick.minor.width":  1.2,
    "ytick.major.size":   7,
    "ytick.major.width":  1.8,
    "ytick.minor.size":   4,
    "ytick.minor.width":  1.2,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.labelsize":    16,
    "ytick.labelsize":    16,
    "xtick.top":          True,            # ticks on all four sides (TOG)
    "ytick.right":        True,

    # ---- Legend ----
    "legend.fontsize":    14,
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "black",
    "legend.fancybox":    False,           # square corners = academic style

    # ---- Figure ----
    "figure.dpi":         150,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})
INPUT_EMG_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "MaM_Public", "emg"
)
path = os.path.join(INPUT_EMG_CSV, "04_2S", "04_2S_emotion_02_sad.csv")
# skipinitialspace: this file (and 04_2S_emotion_10_sad.csv) was re-saved with space-aligned columns
df = pd.read_csv(path, index_col=0, skipinitialspace=True)
df["time"] = df.index / 30
# ---------- Create figure ----------
fig, ax = plt.subplots(figsize=(6, 6))

# Main curve – solid black, thick (classic TOG single-series style)
ax.plot(
    df["time"],
    df["Orbicularis Oris_r"],
    color="#1f77b4",
    # linewidth=1.5,
    solid_capstyle="round",
    solid_joinstyle="round",
    zorder=3,
    label="Orbicularis Oris_r",
)
# other_path = os.path.join(INPUT_EMG_CSV, "04_2S", "04_2S_emotion_10_sad.csv")
# df = pd.read_csv(other_path, index_col=0, skipinitialspace=True)
# df["time"] = df.index / 30
# ax.plot(
#     df["time"],
#     df["Orbicularis Oris_r"],
#     color="#f2c45f",
#     # linewidth=1.5,
#     solid_capstyle="round",
#     solid_joinstyle="round",
#     zorder=3,
#     label="Orbicularis Oris_r",
# )

# # No fill, no grid — keeps it clean for TOG print

# # Mark the target frame with a thin dashed vertical line
# ax.axvline(
#     x=frames[-1],
#     color="grey",
#     linestyle="--",
#     linewidth=1.4,
#     alpha=0.6,
#     zorder=1,
# )
# # Small text annotation for the target frame
# ax.annotate(
#     f"target\n(frame {frames[-1]})",
#     xy=(frames[-1], distances.min()),
#     xytext=(-50, 30),
#     textcoords="offset points",
#     fontsize=13,
#     fontstyle="italic",
#     ha="center",
#     arrowprops=dict(arrowstyle="->", color="grey", lw=1.2),
# )

# # ---- Axis labels (bold, Times New Roman via rcParams) ----
ax.set_xlabel("Time (s)", fontweight="bold")
ax.set_ylabel(r"Activation ($\mu$V)", fontweight="bold")

# # ---- Spines: keep all four, black, thick (TOG style) ----
# for spine in ax.spines.values():
#     spine.set_visible(True)
#     spine.set_color("black")
#     spine.set_linewidth(1.8)

# # ---- Limits ----
# ax.set_ylim(bottom=0)
# ax.set_xlim(frames[0], frames[-1])

# # ---- Minor ticks for a polished look ----
ax.minorticks_on()

# # ---- Legend ----
# ax.legend(loc="upper right")

plt.tight_layout()
plt.show()
# # Save both PNG (for preview) and PDF (for LaTeX / TOG submission)
# plt.savefig(OUTPUT_PNG, dpi=DPI)
# plt.savefig(OUTPUT_PDF)
# print(f"\nPNG saved to: {OUTPUT_PNG}")
# print(f"PDF saved to: {OUTPUT_PDF}")
# plt.show()
