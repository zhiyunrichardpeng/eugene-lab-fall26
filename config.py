from pathlib import Path
import os

a3_stochastic_relative_path = Path(__file__).resolve().parent / "a3-stochastic-V1"
b98_sampling_path = a3_stochastic_relative_path / "b98-sampling-point"
a1_blendshape_path = a3_stochastic_relative_path / "a1-blendshape-guided-animation-pipeline"
print(f"Project root path: {a3_stochastic_relative_path}")
print(f"b98 sampling joints path: {b98_sampling_path}")
print(f"Blendshape guided pipeline path: {a1_blendshape_path}")