import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

csv_path = r"C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\d4 interactive interface of animation\h96-debug-dissipating-error\debug_errors_20260430_173610.csv"
output_dir = os.path.dirname(csv_path)

# SIGGRAPH style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

unit_to_mm = 0.962

transitions = []
transition_labels = []
landmark_mean = []

with open(csv_path, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        transitions.append(float(row['transition']))
        
        pf = str(row['phoneme_from']).split('_')[-1]
        if pf == 'neutral': pf = 'AU0'
        
        pt = str(row['phoneme_to']).split('_')[-1]
        if pt == 'neutral': pt = 'AU0'
        
        transition_labels.append(f"{pf} to {pt}")
        
        landmark_mean.append(float(row['landmark_mean_hat_tilde']) * unit_to_mm)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

ax.plot(transitions, landmark_mean, 'b-o', label='Landmark Mean', linewidth=2, markersize=8)

ax.set_xlabel('Transition')
ax.set_ylabel('Distance (mm)')
ax.legend()
ax.grid(True, alpha=0.3, linestyle='--')

ax.set_xticks(transitions)
ax.set_xticklabels(transition_labels, rotation=45, ha='right')

plt.tight_layout()

output_path = os.path.join(output_dir, "replotted_metrics_only_landmark.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved replotted graph to: {output_path}")
