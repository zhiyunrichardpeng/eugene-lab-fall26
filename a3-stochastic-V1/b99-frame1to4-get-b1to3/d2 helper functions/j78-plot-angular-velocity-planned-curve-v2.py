import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


target_rotation_deg = 0.04044802306349184
target_rotation_deg = target_rotation_deg * 0.95
target_steps = 750
magnitude = 0.000238

def calculate_rotation_characteristics(UNIT_PRIME_jaw_translate, target_steps, target_rotation_deg):
    """Calculate the total rotation for given parameters"""
    omega_max = UNIT_PRIME_jaw_translate
    k = 0.004
    # total_time = (35/30)  # Total simulation time in seconds
    
    print(f"\n=== DEBUG: Input Parameters ===")
    print(f"UNIT_PRIME_jaw_translate: {omega_max}")
    print(f"target_steps: {target_steps}")
    print(f"target_rotation_deg: {target_rotation_deg}")
    

    def omega(t, T):
        t1 = 0.07 * T
        t2 = 0.93 * T
        sigmoid1 = 1 / (1 + np.exp(-k * (t - t1)))
        sigmoid2 = 1 / (1 + np.exp(-k * (t - t2)))
        envelope = (1 - np.exp(-k * t)) * (1 - np.exp(-k * (T - t)))
        return omega_max * (sigmoid1 - sigmoid2) * envelope
    
    # Create proper time array for integration
    time_points = np.arange(0, target_steps)
    velocities = omega(time_points, target_steps)
    
    # Calculate rotation per step (degrees)
    rotation_per_step = velocities #  * (total_time/target_steps)
    print("rotation_per_step", rotation_per_step)


    
    # Total rotation via cumulative sum (more accurate for discrete steps)
    cumulative_rotation = np.cumsum(rotation_per_step)
    total_rotation = cumulative_rotation[-1]  # Final cumulative rotation
    
    print(f"\n=== DEBUG: Rotation Calculation ===")
    print(f"Total rotation achieved: {total_rotation:.6f}°")
    print(f"First 5 rotation steps: {rotation_per_step[:5]}")
    print(f"First 5 cumulative: {cumulative_rotation[:5]}")

    # Find when 99% of rotation is complete
    # completed_steps = np.argmax(cumulative_rotation >= 0.99 * target_rotation_deg)
    
    # Find when 99% of rotation is complete
    threshold = 0.99 * target_rotation_deg
    condition = cumulative_rotation >= threshold
    completed_steps = np.argmax(condition)

    print(f"\n=== DEBUG: Completion Detection ===")
    print(f"99% threshold: {threshold:.6f}°")
    print(f"Condition array: {condition[:10]}...")  # First 10 elements
    print(f"Index where condition becomes True: {completed_steps}")
    print(f"Cumulative rotation at completion: {cumulative_rotation[completed_steps]:.6f}°")
    print(f"Rotation at step 46: {cumulative_rotation[46]:.6f}°")
    print(f"Rotation at step 3700: {cumulative_rotation[-1]:.6f}°")

    return total_rotation, completed_steps, time_points, velocities, cumulative_rotation

def plot_results(time_points, velocities, cumulative_rotation, target_rotation_deg, target_steps, optimal_unit_prime):
    """Visualize the velocity profile and cumulative rotation"""
    plt.figure(figsize=(12, 6))
    
    # Convert time points to steps for plotting
    steps = np.linspace(0, target_steps, len(time_points))
    
    # Velocity profile
    plt.subplot(1, 2, 1)
    plt.plot(steps, velocities, 'b-', label='Velocity (deg/s)')
    plt.xlabel('Steps')
    plt.ylabel('Angular Velocity (deg/s)')
    plt.title(f'Velocity Profile\nUNIT_PRIME = {optimal_unit_prime:.6f}')
    plt.grid(True)
    
    # Cumulative rotation
    plt.subplot(1, 2, 2)
    plt.plot(steps, cumulative_rotation, 'r-', label='Cumulative Rotation')
    plt.axhline(y=target_rotation_deg, color='g', linestyle='--', label='Target Rotation')
    plt.axhline(y=0.99*target_rotation_deg, color='m', linestyle=':', label='99% Target')
    plt.xlabel('Steps')
    plt.ylabel('Rotation (deg)')
    plt.title('Cumulative Rotation')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def objective_function(params, target_steps, target_rotation_deg):
    UNIT_PRIME_jaw_translate = params[0]
    total_rotation, completed_steps, _, _, _ = calculate_rotation_characteristics(
        UNIT_PRIME_jaw_translate, target_steps, target_rotation_deg)
    
    rotation_error = abs(total_rotation - target_rotation_deg)/target_rotation_deg
    steps_error = abs(completed_steps - target_steps)/target_steps
    
    return rotation_error + 0.3 * steps_error

def find_optimal_parameters(target_steps, target_rotation_deg):
    bounds = [(0.0001, 0.01)]
    # result = minimize(
    #     objective_function,
    #     x0=[0.001],
    #     args=(target_steps, target_rotation_deg),
    #     bounds=bounds,
    #     method='L-BFGS-B',
    #     options={'maxiter': 100}
    # )
    # result = magnitude
    
    optimal_unit_prime = magnitude # result.x[0]

    final_rotation, final_steps, time_points, velocities, cum_rotation = calculate_rotation_characteristics(
        optimal_unit_prime, target_steps, target_rotation_deg)
    
    print(f"\nOptimization Results:")
    print(f"Optimal UNIT_PRIME_jaw_translate: {optimal_unit_prime:.8f}")
    print(f"Achieved rotation: {final_rotation:.6f}° (target: {target_rotation_deg}°)")
    print(f"99% rotation completed at step: {final_steps} (target: {target_steps})")
    print(f"Time per step: {(35/30)/target_steps:.6f} seconds")
    # print('cum_rotation', cum_rotation)
    
    plot_results(time_points, velocities, cum_rotation, target_rotation_deg, target_steps, optimal_unit_prime)
    return optimal_unit_prime

print("Starting optimization...")
optimal_value = find_optimal_parameters(target_steps, target_rotation_deg)
print(f"\nRecommended value: {optimal_value:.8f}")