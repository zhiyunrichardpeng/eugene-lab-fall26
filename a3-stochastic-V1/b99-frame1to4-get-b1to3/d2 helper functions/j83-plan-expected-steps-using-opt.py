import numpy as np
from scipy.optimize import minimize_scalar

def calculate_effective_duration(UNIT_PRIME_jaw_translate, target_steps=3700):
    """Calculate how many steps are needed to complete rotation with given parameters"""
    omega_max = UNIT_PRIME_jaw_translate
    k = 0.004
    T_guess = 3700  # Initial guess
    
    def omega(t, T):
        t1 = 0.07 * T
        t2 = 0.93 * T
        sigmoid1 = 1 / (1 + np.exp(-k * (t - t1)))
        sigmoid2 = 1 / (1 + np.exp(-k * (t - t2)))
        envelope = (1 - np.exp(-k * t)) * (1 - np.exp(-k * (T - t)))
        return omega_max * (sigmoid1 - sigmoid2) * envelope
    
    # Find T where cumulative rotation reaches target
    def objective(T):
        steps = np.linspace(0, T, 100)
        velocities = [omega(t, T) for t in steps]
        total_rotation = np.trapz(velocities, steps)
        return abs(total_rotation - omega_max * target_steps)
    
    result = minimize_scalar(objective, bounds=(3000, 5000), method='bounded')
    return result.x

# Optimization function to find UNIT_PRIME_jaw_translate
def find_optimal_unit_prime(target_steps=3700):
    def objective(unit_prime):
        effective_steps = calculate_effective_duration(unit_prime)
        return abs(effective_steps - target_steps)
    
    # Search in reasonable range (adjust based on your system)
    result = minimize_scalar(objective, bounds=(0.001, 0.1), method='bounded')
    return result.x

# Run optimization
optimal_unit_prime = find_optimal_unit_prime()
print(f"Optimal UNIT_PRIME_jaw_translate: {optimal_unit_prime:.6f}")