import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from paths import backup_path

# Constants
self = lambda: None  # Dummy self object


def unitprime():
    self.expected_total_steps = 10000
    
    # translation_anatomy_limit = 35/30/(self.expected_total_steps/300)*0.208

    # translation_anatomy_limit = 0.012133
    # old:  0.0208 * (20/args['keyframe_num']) # for 20 frames. if 10 frames, 0.0208 * 2 ---
    rotation_anatomy_limit = 35/30/self.expected_total_steps*31.24

    adjust_magnitude_max = 0.6
    synchronizing_adjustment_parameter = 8/10 #* 2 # 1.0/5 # make the law rotates slower 5 times than before.  
    # times 2 because the r limit changed to sigmoid. the intergral is around 1/2 of the original curve.
    # 0.9 is because we want it to scale to slower to further align with translation.  0/9 CAN WORK.  0.6 also passed.  
    # new code has moved the  0.6 part to line 859 in m1: cap_max = self.UNIT_PRIME_jaw_translate * 3000/min((self.expected_total_steps - 500), 3000)

    rotation_anatomy_limit = rotation_anatomy_limit * synchronizing_adjustment_parameter * adjust_magnitude_max
    # old: 0.00843 * (20/args['keyframe_num']) # for 20 frames. if 10 frames, 0.0208 * 2 ---
    # degrees_min = initial_angle_degrees * args['distance_threshold_percentage']

    # UNIT_PRIME = translation_anatomy_limit
    # # old distance_min / args['unit_denominator'] # 100000.0 # 70.0 # round(distance_min / 70.0,4)

    # if UNIT_PRIME > translation_anatomy_limit:  # 0.1/args['unit_denominator']:# 0.0001:
    #     UNIT_PRIME = translation_anatomy_limit # 0.1/args['unit_denominator']

    UNIT_PRIME_jaw_translate = rotation_anatomy_limit

    return UNIT_PRIME_jaw_translate


self.UNIT_PRIME_jaw_translate = unitprime()

adjust_factor_adhoc = 1.0 # 40/55
cap_max = self.UNIT_PRIME_jaw_translate # * 3000/max((self.expected_total_steps - 200), 3000) *  adjust_factor_adhoc
omega_max = cap_max


self.step_counter_internal = np.arange(0, 9900)  # t from 0 to 4200

# Parameters for omega(t)
omega_max = self.UNIT_PRIME_jaw_translate
T = self.expected_total_steps
t1 = 0.07 * T
t2 = 0.93 * T
k = 0.004  # Sigmoid steepness

# Define the omega(t) function
def omega(t):
    def safe_exp(x, limit=1e6):
        """
        Safely compute exp(x) by clipping x to avoid overflow.
        """
        return np.exp(np.clip(x, -limit, limit))

    # Sigmoid calculations
    sigmoid1 = 1 / (1 + safe_exp(-k * (t - t1)))  # Sigmoid for acceleration
    sigmoid2 = 1 / (1 + safe_exp(-k * (t - t2)))  # Sigmoid for deceleration

    # Envelope to ensure omega(0) = 0 and omega(T) = 0
    envelope = (1 - np.exp(-k * t)) * (1 - np.exp(-k * (T - t)))

    return omega_max * (sigmoid1 - sigmoid2) * envelope

# Compute r_limit_sigmoid
velocity = omega(self.step_counter_internal)
# Apply minimum threshold to the velocity cap
MINIMUM_THRESHOLD = 1.2174288e-06
r_limit_sigmoid = np.maximum(velocity, MINIMUM_THRESHOLD)
# r_limit_sigmoid = max(velocity, MINIMUM_THRESHOLD)

# r_limit_sigmoid = self.UNIT_PRIME_jaw_translate * (3*(self.step_counter_internal/self.expected_total_steps)**2 - 2*(self.step_counter_internal/self.expected_total_steps)**3)
# if r_limit_sigmoid >= cap_max:
#     r_limit_sigmoid = cap_max

cap_max = self.UNIT_PRIME_jaw_translate  # Your upper limit
r_limit_sigmoid = np.minimum(r_limit_sigmoid, cap_max)

# Reshape into (N, 3) and set columns 1 & 2 (0-based index) to 0
reshaped = r_limit_sigmoid.reshape(-1, 3)  # -1 = auto-compute rows
reshaped[:, 1:3] = 0  # Set 2nd and 3rd elements to 0

# Flatten back to 1D
r_limit_sigmoid_modified = reshaped.flatten()

# angular_velocity_rad = np.deg2rad(r_limit_sigmoid)
# Compute cumulative angle
dt = 1.0
total_angle_rad = np.cumsum(r_limit_sigmoid_modified * dt)

# Find the first index where cumulative angle >= 4.8
stop_index = np.argmax(total_angle_rad >= 4.8)

# If 4.8 is never reached, keep all velocities
if stop_index == 0 and total_angle_rad[0] < 4.8:
    stop_index = len(total_angle_rad)

# Set all velocities after `stop_index` to 0
# r_limit_sigmoid_modified[stop_index:] = 0

# Recompute cumulative angle (optional, for verification)
total_angle_rad = np.cumsum(r_limit_sigmoid_modified * dt)
print("Final cumulative angle:", total_angle_rad[-1])  # Should be ~4.8

# Select first elements of every 3 for plotting
first_elements = r_limit_sigmoid_modified[::3]
time_steps = self.step_counter_internal[::3]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time_steps, first_elements, label='r_limit_sigmoid (1st of every 3)', color='blue', linewidth=2)
# plt.axvline(x=time_steps[stop_index // 3], color='red', linestyle='--', label='Angle reaches 4.8 degrees')
plt.xlabel('Time (t)', fontsize=14)
plt.ylabel('Angular Velocity (degrees/s)', fontsize=14)
plt.title('Velocity Zeroed After 4.8 degrees', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
# plt.show()


object_name = "jaw"

# Save the plot to a file
output_folder = str(backup_path / "d46")
output_file = os.path.join(output_folder, f'ideal_velocity_{object_name}_no_cut_off.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()        
