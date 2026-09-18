"""
=============================================================================
Plot: Skin Elastic Strain Energy (V → N)
=============================================================================
SIGGRAPH TOG submission style — Times New Roman, thick blue line, square.

Run this script *outside* Maya with a standard Python environment that has
matplotlib and numpy installed.

Usage:
    python plot_skin_strain_energy.py

Output:
    C:/Users/zpa18/a3-stochastic-V1/visualization/skin_strain_energy.png
=============================================================================
"""

import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ===================== CONFIGURATION =====================
INPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "skin_strain_energy_data.csv"
)
OUTPUT_PNG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "skin_strain_energy.png"
)
DPI = 600
# =========================================================

# ---------- Load data ----------
frames = []
energies = []

with open(INPUT_CSV, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        frames.append(int(row["frame"]))
        energies.append(float(row["elastic_strain_energy_J"]))

frames = np.array(frames)
energies = np.array(energies)

print(f"Loaded {len(frames)} data points from {INPUT_CSV}")
print(f"  Frame range : {frames[0]} -> {frames[-1]}")
print(f"  Energy range : {energies.min():.6e} - {energies.max():.6e} J")

# ================================================================
#  SIGGRAPH / TOG  style  –  Times New Roman, thick lines, clean
# ================================================================
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

plt.rcParams.update({
    # ---- Font ----
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          18,

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
    "xtick.top":          True,
    "ytick.right":        True,

    # ---- Legend ----
    "legend.fontsize":    14,
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "black",
    "legend.fancybox":    False,

    # ---- Figure ----
    "figure.dpi":         150,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# ---------- Create figure (1:1 ratio) ----------
fig, ax = plt.subplots(figsize=(6, 6))

# Main curve — blue, thick
ax.plot(
    frames,
    energies,
    color="#1f77b4",
    linewidth=3.5,
    solid_capstyle="round",
    solid_joinstyle="round",
    zorder=3,
    label="Skin elastic strain energy",
)

# ---- Axis labels ----
ax.set_xlabel("Frames", fontweight="bold")
ax.set_ylabel("Elastic Strain Energy (J)", fontweight="bold")

# ---- Spines: all four visible, black, thick (TOG style) ----
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(1.8)

# ---- Limits ----
ax.set_ylim(bottom=0)
ax.set_xlim(frames[0], frames[-1])

# ---- Minor ticks ----
ax.minorticks_on()

# ---- Legend ----
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=DPI)
print(f"\nPNG saved to: {OUTPUT_PNG}")
plt.show()
