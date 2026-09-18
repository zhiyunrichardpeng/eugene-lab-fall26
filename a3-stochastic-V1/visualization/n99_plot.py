import matplotlib.pyplot as plt

# Your extracted data
transitions = [
    "Neutral to N", "N to AA", "AA to T", "T to IY",
    "IY to V", "V to N", "N to Y", "Y to UW"
]
durations = [0.23, 0.29, 0.15, 0.22, 0.09, 0.16, 0.02, 0.52]
percentages = [13.7, 17.3, 8.9, 13.1, 5.4, 9.5, 1.2, 31.0]

# Calculate cumulative times for the x-axis nodes
cumulative_time = [0]
for d in durations:
    cumulative_time.append(cumulative_time[-1] + d)

# 1. TIGHTER HEIGHT: Reduced figsize from (15, 5) to (15, 3)
fig, ax = plt.subplots(figsize=(15, 3))

# Draw the main timeline wire and nodes
ax.plot(cumulative_time, [0] * len(cumulative_time), "-o", 
        color="#333333", markerfacecolor="white", markersize=8, linewidth=2)

# Annotate each transition segment
for i in range(len(transitions)):
    midpoint = cumulative_time[i] + (durations[i] / 2)
    
    # 2. COMPRESSED SPACING: Reduced offset so boxes hug the line closer
    y_offset = 0.12 if i % 2 == 0 else -0.12
    va = 'bottom' if i % 2 == 0 else 'top'
    
    label = f"{transitions[i]}\n{durations[i]}s ({percentages[i]}%)"
    
    ax.annotate(label, 
                xy=(midpoint, 0), 
                xytext=(midpoint, y_offset),
                textcoords="data",
                ha="center", 
                va=va, 
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#cccccc", lw=1),
                arrowprops=dict(arrowstyle="-", color="#999999", shrinkA=0, shrinkB=0))

# Format the axes for a clean look
ax.set_yticks([])  
# 3. COMPRESSED BOUNDS: Reduced from 0.4 to 0.25 to remove dead space
ax.set_ylim(-0.25, 0.25)  
ax.set_xticks(cumulative_time)
ax.set_xticklabels([f"{t:.2f}s" for t in cumulative_time], fontsize=10)
ax.set_xlabel("Cumulative Time (seconds)", fontsize=12, labelpad=10, fontweight='bold')
ax.set_title("Sequence of Lip-Corner Deformation Transitions", fontsize=14, pad=15, fontweight='bold')

# Clean up the bounding box (spines)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False) 
ax.tick_params(axis='x', length=5, color='#333333') 

plt.tight_layout()

# 4. INCREASED RESOLUTION: dpi=600 for sharp PPT graphics, bbox_inches='tight' crops whitespace
plt.savefig("transition_timeline_compact_highres.png", dpi=600, bbox_inches='tight') 
plt.show()