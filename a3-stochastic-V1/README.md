# Physically Based Facial Deformation & Piano Interface

This project provides a physically-based muscle solving pipeline to generate facial deformation trajectories between different poses and a custom interface to interact with these animations in Autodesk Maya using a "piano-style" interaction methodology.

## Prerequisites & Required Assets

Before running the pipeline or the UI, you need to have the following foundational assets available in your environment:

1. **The Rest Pose**: The base/neutral starting pose of the facial geometry (e.g., `dataV22_AU0_neutral`).
2. **The Target Pose(s)**: The specific facial deformation pose you want to transition towards (e.g., `dataV22_phoneme_N`).
3. **Deformation Trajectory**: The sequence of step-by-step deformations connecting the rest pose to the target pose. 
   - The Piano Interface needs this trajectory data to function properly. 
   - If you do not already have pre-computed trajectories, you can generate them yourself using the Optimization Pipeline (see below).

**Example of a generated trajectory path map:**
```json
"dataV22_AU0_neutral to dataV22_phoneme_N": "C:/keyLabLocal_D/a3-stochastic-V1-backup-and-results/results/DSAC_V2_m1_vertexMove_Va003/job_260717-174842/260717-174850"
```

## Main Components

### 1. Optimization Pipeline (`run_pipeline_update.py`)
This is the core execution script that runs the physically-based optimization solver.

- **Function**: It runs the analytical gradient optimization to obtain the physical trajectory of the deformation. 
- **Methodology**: It calculates the gradient step-by-step for various facial tissues (OOR, OOC, Frontalis, Jaw) using a multi-component objective function that balances:
  1. Euclidean Distance
  2. Strain Energy
  3. Muscle Work

### 2. Maya Piano Interface (`h99-UI-mayascipt-V1.py`)
This is the front-end user script that executes inside Autodesk Maya.

- **Function**: It reads and consumes the trajectories obtained by the Optimization Pipeline.
- **Methodology**: By linking the calculated deformation steps to the interface, it drives the mesh deformation directly in the Maya viewport, allowing for smooth, real-time "piano-style" interaction and blending between complex poses.

## Environment & Installation

This project utilizes advanced scientific libraries and PyTorch to calculate analytical gradients, as well as PySide/PyQt5 to render the UI.

To install the Python dependencies outside of Maya, run:
```bash
pip install -r requirements.txt
```

*Note: Since the Piano Interface script operates inside Maya, ensure that your Maya Python interpreter (`mayapy`) has access to the necessary packages or run the UI script directly from within the Maya Script Editor.*
