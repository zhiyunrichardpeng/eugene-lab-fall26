# eugene-lab-fall26

## Quick Start for New Users
When you clone this repository to your local machine, the codebase uses relative path resolution so you can run scripts directly without hardcoding user folders or machine-specific directories:

```bash
git clone https://github.com/zhiyunrichardpeng/eugene-lab-fall26.git
cd eugene-lab-fall26
```

### Running Pipeline Scripts
You can run automation scripts directly from terminal or VS Code:
```bash
python a3-stochastic-V1/automation/run_pipeline_update.py
```

> **Note on imports**: Top-level path definitions are stored in `paths.py` at the repository root. Automation scripts automatically resolve and locate the repository root via `pathlib.Path`. If you are creating a new standalone script in a subdirectory that imports `paths.py`, ensure the repository root is added to `sys.path`:
> ```python
> import sys
> from pathlib import Path
> 
> # Add repository root to sys.path (adjust parents depth based on folder level)
> REPO_ROOT = Path(__file__).resolve().parents[2]
> if str(REPO_ROOT) not in sys.path:
>     sys.path.insert(0, str(REPO_ROOT))
> 
> from paths import a3_stochastic_relative_path
> ```
> Alternatively, you can set `PYTHONPATH` to the repository root directory in your environment.

---

## Code Setup
All of the core codebase is located under the `a3-stochastic-V1` directory. As such, it is the base for the rest of the paths defined in `paths.py`: a convenient file to import necessary paths from. It includes:
```
a3_stochastic_relative_path: resolves to <repo_root>\a3-stochastic-V1
b98_sampling_path 
b99_frame1to4_path
a1_blendshape_path
```
The first path serves as the project root dir, while the other 3 are used extensively throughout many other directories (`b99-frame1to4-get-b1to3` and its subdirectories). All variables are of type `Path` (see [pathlib](https://docs.python.org/3/library/pathlib.html) for details), so when passing to APIs requiring strings, cast them with `str()`.

**Note**: I (Viktor) am considering making them const.

## Updates
As of today (10/17/2026) the paths containing `KeyLabLocal` are being updated to relative paths.