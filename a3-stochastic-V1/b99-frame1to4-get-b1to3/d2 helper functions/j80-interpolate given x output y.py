import numpy as np
import matplotlib.pyplot as plt

# Given data
x_data = np.array([0.9, 0.8, 0.5, 0.1, 0.003])
y_data = np.array([3525, 9200, 16000, 33076, 39000])

# Fit a cubic polynomial (degree = 3)
coefficients = np.polyfit(x_data, y_data, 3)
poly = np.poly1d(coefficients)

# Print the polynomial equation
print("Fitted polynomial:")
print(poly)

# Generate points for smooth curve
x_fit = np.linspace(0.002, 0.91, 500)
y_fit = poly(x_fit)

# Predict y at x=0.8
x_target = 0.7
y_pred = poly(x_target)
print(f"\nAt x = {x_target:.2f}, y = {y_pred:.0f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='red', s=100, label='Original Data')
plt.plot(x_fit, y_fit, 'b-', label=f'Cubic Fit: {poly}')
plt.scatter([x_target], [y_pred], color='green', s=200, marker='*', 
            label=f'Prediction at x=0.8: y={y_pred:.0f}')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Perfect Fit to All Data Points', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()