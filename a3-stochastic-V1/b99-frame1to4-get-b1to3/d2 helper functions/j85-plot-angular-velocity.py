import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
filename = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\job_250516-154657\250516-154711\adhoc.txt"
df = pd.read_csv(filename)

# Clean data - replace empty strings or invalid values with NaN and drop them
df.replace('', np.nan, inplace=True)
df.dropna(subset=['angular velocity (degree/second)'], inplace=True)

# Convert columns to numeric (in case they were read as strings)
df['internal steps'] = pd.to_numeric(df['internal steps'])
df['angular velocity (degree/second)'] = pd.to_numeric(df['angular velocity (degree/second)'])

# Create the plot
plt.figure(figsize=(12, 6))

# Plot angular velocity vs internal steps
plt.plot(df['internal steps'], -df['angular velocity (degree/second)'], 
         'b-', linewidth=1.5, label='Angular Velocity')

# Highlight zero crossings if needed
zero_crossings = np.where(np.diff(np.sign(df['angular velocity (degree/second)'])))[0]
if len(zero_crossings) > 0:
    plt.scatter(df['internal steps'].iloc[zero_crossings], 
                -df['angular velocity (degree/second)'].iloc[zero_crossings],
                color='red', zorder=5, label='Zero Crossings')

# Add labels and title
plt.xlabel('Internal Steps', fontsize=12)
plt.ylabel('Angular Velocity (degree/second)', fontsize=12)
plt.title('Angular Velocity vs. Simulation Steps', fontsize=14)

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)

# Adjust x-axis to show major steps
max_steps = df['internal steps'].max()
x_ticks = np.arange(0, max_steps + 1, 2000)  # Show tick every 2000 steps
plt.xticks(x_ticks)

# # Highlight different phases if needed
# phases = {
#     'Phase 1': (0, 3000),
#     'Zero Velocity': (3000, 5600),
#     'Phase 2': (5600, 10000),
#     'Zero Velocity 2': (10000, max_steps)
# }

# for phase, (start, end) in phases.items():
#     if start < max_steps:
#         plt.axvspan(start, min(end, max_steps), alpha=0.1, label=phase)

plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()