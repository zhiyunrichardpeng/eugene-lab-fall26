import numpy as np
import matplotlib.pyplot as plt

class SmoothCurvePlotter:
    def __init__(self, overlapping_length=2000, unit_prime=1.0):
        self.overlapping_length = overlapping_length
        self.UNIT_PRIME = unit_prime
        self.step_counter_internal = np.arange(overlapping_length)
        
    def get_smooth_limit(self):
        """Your original smooth-in function"""
        t = np.minimum(self.step_counter_internal / self.overlapping_length, 1.0)
        eased_t = t**2 * (3 - 2*t)
        t_limit = self.UNIT_PRIME * (0.25 + 0.75 * eased_t)
        return np.minimum(t_limit, self.UNIT_PRIME)
    
    def obtain_t_limit_sigmoid_endList(self):
        """Your original smooth-out function adjusted for plotting"""
        t = np.linspace(0, 1, self.overlapping_length)
        eased_t = t**2 * (3 - 2*t)
        inverted_eased_t = 1 - eased_t
        return np.maximum(self.UNIT_PRIME * inverted_eased_t, 0.0)
    
    def plot_curves(self):
        """Plot both curves on the same graph"""
        plt.figure(figsize=(10, 6))
        
        # Generate smooth-in curve (ramp up)
        smooth_in = self.get_smooth_limit()
        
        # Generate smooth-out curve (ramp down)
        smooth_out = self.obtain_t_limit_sigmoid_endList()
        
        # Plot both curves
        plt.plot(self.step_counter_internal, smooth_in, label='Smooth-In Curve', color='blue', linewidth=2)
        plt.plot(self.step_counter_internal, smooth_out, label='Smooth-Out Curve', color='red', linewidth=2)
        
        # Add annotations for key points
        plt.axhline(y=self.UNIT_PRIME, color='gray', linestyle='--', alpha=0.5)
        plt.text(self.overlapping_length*0.8, self.UNIT_PRIME*1.05, f'UNIT_PRIME = {self.UNIT_PRIME}', ha='center')
        
        # Customize plot
        plt.title('Smooth Transition Curves Comparison\n(Overlapping Length = {self.overlapping_length})')
        plt.xlabel('Step')
        plt.ylabel('Control Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save and show
        plt.savefig('smooth_curves_comparison.png', dpi=300)
        plt.show()

# Usage:
plotter = SmoothCurvePlotter(overlapping_length=2000, unit_prime=1.0)
plotter.plot_curves()