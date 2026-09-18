import cv2
import numpy as np
import os

def adjust_gamma(image, gamma=1.5):
    # Build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# Resolve paths relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, 'real character UW 26-6-26.jpg')
output_path = os.path.join(script_dir, 'real character UW 26-6-26_lighter.jpg')

# Load your screenshot
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input image not found: {input_path}")

img = cv2.imread(input_path)
if img is None:
    raise ValueError(f"Failed to read image: {input_path}")

# Increase gamma (values > 1.0 brighten the image)
# Try 1.5 to 2.2 for very dark renders
brightened_img = adjust_gamma(img, gamma=1.8)

cv2.imwrite(output_path, brightened_img)
print(f"Brightened image saved to: {output_path}")