from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
a3_stochastic_relative_path = ROOT / "a3-stochastic-V1"
b98_sampling_path = a3_stochastic_relative_path / "b98-sampling-joint"
b99_frame1to4_path = a3_stochastic_relative_path / "b99-frame1to4-get-b1to3"
a1_blendshape_path = a3_stochastic_relative_path / "a1-blendshape-guided-animation-pipeline"

results_path = ROOT / "results" / "DSAC_V2_m1_vertexMove_Va003"
backup_path = ROOT / "backup"

print(f"======== Important paths ========")
print(f"Repo root path: {ROOT}")
print(f"a3-stochastic path: {a3_stochastic_relative_path}")
print(f"b98 sampling joints path: {b98_sampling_path}")
print(f"Blendshape guided pipeline path: {a1_blendshape_path}")
print(f"b99-frame1to4-get-b1to3: {b99_frame1to4_path}")
print(f"Results path: {results_path}")
print(f"Backup path: {backup_path}")
