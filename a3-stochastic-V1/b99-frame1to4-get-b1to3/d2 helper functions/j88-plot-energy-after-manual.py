import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
from matplotlib.colors import Normalize  # Import Normalize from matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation 
# from IPython import display
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
import csv
import os

from os import path
from paths import backup_path

def plot_energy_vs_time_plot(save_folder, self_time_for_energy_log, self_strain_energy):
    # Set up the plot with a professional style
    plt.figure(figsize=(8, 6), dpi=300)  # Larger figure size and high DPI
    plt.rcParams['font.family'] = 'serif'  # Use serif font
    plt.rcParams['font.size'] = 12  # Increase font size

    # Plot the data with a thicker line and professional color
    plt.plot(self_time_for_energy_log, self_strain_energy, 
             label='Strain energy over time', 
             color='blue', linewidth=2, linestyle='-')

    # Add labels with increased font size
    plt.xlabel('Animation time (s)', fontsize=14)
    plt.ylabel('Strain energy (mJ)', fontsize=14)

    # Remove the upper and right bounds of the box
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Set axis limits (optional, adjust based on your data)
    plt.xlim(min(self_time_for_energy_log), max(self_time_for_energy_log))
    plt.ylim(min(self_strain_energy), max(self_strain_energy))

    # Add a legend with a shadow and place it in the best location
    plt.legend(frameon=True, shadow=True, loc='best', fontsize=12)

    # Save the plot with high resolution
    plt.savefig(os.path.join(save_folder, 'energy_vs_time_plot.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Save the data to a CSV file for use in Origin
    with open(os.path.join(save_folder, "energy_data.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Animation time (s)", "Strain energy (mJ)"])  # Write header
        writer.writerows(zip(self_time_for_energy_log, self_strain_energy))  # Write data rows


save_folder = str(backup_path / "e46 adjustable overlap")

scaling_factor = 9.26 * 10**-7 * 10**3
scaled_self_strain_energy = [t * scaling_factor for t in self_strain_energy]




scaling_factor_frame_to_seconds = 0  # (frame - 15) * 1 / 30
scaled_self_time_for_energy_log = [(t -15)*1/30 for t in self_time_for_energy_log]  # unit: seconds.

plot_energy_vs_time_plot(save_folder, scaled_self_time_for_energy_log, scaled_self_strain_energy)

print("energy curve plotted")