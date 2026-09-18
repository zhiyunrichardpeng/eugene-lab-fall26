"""
=============================================================================
Plot: Skin Euclidean Distance to Target Pose (N → Y)
=============================================================================
SIGGRAPH TOG submission style — Times New Roman, thick lines, clean axes.

Run this script *outside* Maya with a standard Python environment that has
matplotlib and numpy installed.

Usage:
    python plot_skin_distance.py

Output:
    C:/Users/zpa18/a3-stochastic-V1/visualization/skin_distance_to_target.png
    C:/Users/zpa18/a3-stochastic-V1/visualization/skin_distance_to_target.pdf
=============================================================================
"""

import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ===================== CONFIGURATION =====================
INPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "skin_distance_data.csv"
)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "skin_distance_to_target.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "skin_distance_to_target.pdf")
DPI = 600
# =========================================================

# ---------- Load data ----------
frames = []
distances = []

with open(INPUT_CSV, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        frames.append(int(row["frame"]))
        distances.append(float(row["mean_euclidean_distance_cm"]))

frames = np.array(frames)
distances = np.array(distances)

print(f"Loaded {len(frames)} data points from {INPUT_CSV}")
print(f"  Frame range : {frames[0]} -> {frames[-1]}")
print(f"  Distance range : {distances.min():.4f} - {distances.max():.4f} cm")

# ================================================================
#  SIGGRAPH / TOG  style  –  Times New Roman, thick lines, clean
# ================================================================
# Force matplotlib to use Type-1 / TrueType fonts (no bitmap / Type-3)
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

# ---------- Create figure ----------
fig, ax = plt.subplots(figsize=(6, 6))

# Main curve – solid black, thick (classic TOG single-series style)
ax.plot(
    frames,
    distances,
    color="#1f77b4",
    linewidth=3.5,
    solid_capstyle="round",
    solid_joinstyle="round",
    zorder=3,
    label="Mean vertex distance",
)

# No fill, no grid — keeps it clean for TOG print

# Mark the target frame with a thin dashed vertical line
ax.axvline(
    x=frames[-1],
    color="grey",
    linestyle="--",
    linewidth=1.4,
    alpha=0.6,
    zorder=1,
)
# Small text annotation for the target frame
ax.annotate(
    f"target\n(frame {frames[-1]})",
    xy=(frames[-1], distances.min()),
    xytext=(-50, 30),
    textcoords="offset points",
    fontsize=13,
    fontstyle="italic",
    ha="center",
    arrowprops=dict(arrowstyle="->", color="grey", lw=1.2),
)

# ---- Axis labels (bold, Times New Roman via rcParams) ----
ax.set_xlabel("Frames", fontweight="bold")
ax.set_ylabel("Distance (cm)", fontweight="bold")

# ---- Spines: keep all four, black, thick (TOG style) ----
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(1.8)

# ---- Limits ----
ax.set_ylim(bottom=0)
ax.set_xlim(frames[0], frames[-1])

# ---- Minor ticks for a polished look ----
ax.minorticks_on()

# ---- Legend ----
ax.legend(loc="upper right")

plt.tight_layout()

# Save both PNG (for preview) and PDF (for LaTeX / TOG submission)
plt.savefig(OUTPUT_PNG, dpi=DPI)
plt.savefig(OUTPUT_PDF)
print(f"\nPNG saved to: {OUTPUT_PNG}")
print(f"PDF saved to: {OUTPUT_PDF}")
plt.show()
