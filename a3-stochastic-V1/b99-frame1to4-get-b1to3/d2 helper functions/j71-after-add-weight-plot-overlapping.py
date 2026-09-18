import numpy as np
import matplotlib.pyplot as plt

# Configuration
OVERLAPPING_LENGTH = 2000
UNIT_PRIME = 1.0

# Weighted Smooth-Out Function
def weighted_smooth_out(current_step, total_steps=OVERLAPPING_LENGTH):
    t = min(current_step / total_steps, 1.0)
    eased_t = t**2 * (3 - 2*t)  # Cubic easing
    
    # Weights (w1 decreasing, w2 increasing)
    w1 = 1 - eased_t
    w2 = eased_t
    
    # Combined control (c1=UNIT_PRIME, c2=0)
    value = w1 * UNIT_PRIME + w2 * 0.0
    return value, w1, w2

# Weighted Smooth-In Function
def weighted_smooth_in(current_step, total_steps=OVERLAPPING_LENGTH):
    t = min(current_step / total_steps, 1.0)
    eased_t = t**2 * (3 - 2*t)  # Cubic easing
    
    # Weights (w1 increasing from 0.25, w2 decreasing)
    w1 = 0.25 + 0.75 * eased_t
    w2 = 1 - w1
    
    # Combined control (c1=0, c2=UNIT_PRIME)
    value = w1 * UNIT_PRIME + w2 * 0.0
    return value, w1, w2

# Generate data
steps = np.arange(OVERLAPPING_LENGTH)
out_values, out_w1, out_w2 = zip(*[weighted_smooth_out(s) for s in steps])
in_values, in_w1, in_w2 = zip(*[weighted_smooth_in(s) for s in steps])

# Create the plot
plt.figure(figsize=(14, 8))

# Plot the control values
plt.subplot(2, 1, 1)
plt.plot(steps, out_values, 'r-', linewidth=2, label='Smooth-Out Curve')
plt.plot(steps, in_values, 'b-', linewidth=2, label='Smooth-In Curve')
plt.title('Weighted Transition Curves\n(Overlapping Length = 2000)', fontsize=14)
plt.ylabel('Control Value', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=UNIT_PRIME, color='gray', linestyle='--')

# Plot the weight functions
plt.subplot(2, 1, 2)
plt.plot(steps, out_w1, 'r--', linewidth=1.5, label='Out W1 (decreasing)')
plt.plot(steps, out_w2, 'r:', linewidth=1.5, label='Out W2 (increasing)')
plt.plot(steps, in_w1, 'b--', linewidth=1.5, label='In W1 (increasing)')
plt.plot(steps, in_w2, 'b:', linewidth=1.5, label='In W2 (decreasing)')
plt.title('Weight Functions', fontsize=14)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Weight Value', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('weighted_transition_curves.png', dpi=300)
plt.show()