# Update Deprecated First Run (11/05/26)
## The issue 
The previous version of the codebase took around 2 hours to run a single transition (phase) in `d1-adhoc-for-b999.py` because it was using a numerical gradient descent method and also not using the `deformation-path-db.json`.

This is what I understood from my talk with Richard last week. Maybe I'm missing some things.
## What changed
I was supposed to modify `evaluator.py` to use the new analytical gradient and also the database to hopefully speed up the process of running a single phase.

I did this in an earlier commit this week `#e82aad6` but just today had the chance to run the code.
## What happened
I fixed some bugs and ra the updated code with `run_d1_b99.bat` and this time it didn't crash but also took around 1h 29m for a single phase. Slightly better than 2 hours but that might just be an average over phases.

The results are in this same folder:
- `d1-adhoc-time[2].txt`
- `d1-adhoc-time-1-phase.txt`
- `d1-output-log.txt`
## What to do next?
Probably ask Richard and LLM what might be missing or went wrong.
# Update Deprecated Second Run (11/06/26)
## The issue 
I realized that I imported the wrong `a3_stochastic` path file in `evaluator.py`. While I imported the analytical gradient module and initialized the evaluator's `analytical_calculator` in `__init__`,
I never set the `calculator` paramater of `compute_analytical_gradient` as the analytical gradient. I was using the old `distance_calculator` still. I believe that's one reason the d1-ahdoc code was still taking around 1.5 hours for a single phase. 

Second of all, when I ran `run_d1_b99.bat` it was still running the old local version of `a3-stochastic-V1` so none of my changes were even being reflected...

I also had to do more path resolution just copying the code from `h99-UI-mayascript-V1.py` into various files included in the commit along with this file.
## What changed
Here's the path resolution thing:
```python
# ==============================================================================
# Dynamic Repository Root Detection
# ==============================================================================
REPO_NAME = "a3-stochastic-Rich-local-V1"
REPO_ROOT = None

if "paths" in sys.modules and hasattr(sys.modules["paths"], "__file__"):
    cached_path = Path(sys.modules["paths"].__file__).resolve().parent
    if (cached_path / "paths.py").is_file():
        REPO_ROOT = cached_path

if not REPO_ROOT and "__file__" in globals() and __file__:
    script_path = Path(__file__).resolve()
    for p in [script_path] + list(script_path.parents):
        if (p / "paths.py").is_file():
            REPO_ROOT = p
            break

if not REPO_ROOT:
    fallbacks = [
        Path(r"C:\keyLabLocal_D") / REPO_NAME,
        Path(r"D:\keyLabLocal_D") / REPO_NAME,
        Path.home() / "git" / REPO_NAME,
        Path.home() / "Documents" / "GitHub" / REPO_NAME,
    ]
    for fb in fallbacks:
        if (fb / "paths.py").is_file():
            REPO_ROOT = fb
            break

if REPO_ROOT and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

And the analytical calculator change:
```python
    def compute_single_gradient(self, position, target, positions_lip_joint, step_t1, object_name,batch_size=1, epsilon=1e-3, threshold=1e-5, clip_value=1.0):
        # TEST: early return with analytical gradient
        if object_name == "oor":
            calculator = self.analytical_calculator
```
## What happened
This time I got this result:
```powershell
Step: 36600:   0%|          | 36600/100000000 [56:59<2594:33:48, 10.70it/s]
in env, hit one success. should early stop now. success_flag: 1 distance: 3.4371030301023935 min dis: 3.8165741610845667 distance_ooc_l: 0.0 min distance_ooc_l: 0.0 distance_ooc_r: 0.0 min distance_ooc_r: 0.0 distance_frontalis: 0.0 min distance_frontalis: 0.0    
final distance_jaw is:  1.2785456835879812
final distance_jaw is: 1.2785456835879812
in env l646, hit one success. should early stop now. self.info['success_flag']: -1
No gradient data available for object oor in phase 1
success in evaluator, distance is: 3.4371030301023935
distance_in_MayaFile:  3.4371030301023935
Game over since hit one success action sequence. Exiting the program early.
```
Compared to last run's time of `[1:28:53<4046:09:35,  6.86it/s]`, today's run was **35% faster** using the same parameters.

Maybe it can be even faster but I'll need some more help to figure out how.


**The results are in this same folder:**
- `d1-adhoc-run2.txt`
## What to do next?
- Check the cProfile output for 1 run. I forgot to do it this time.
- Ask Richard and LLM how we could make this faster or if it's good enough move on to another task.
# Update Deprecated Third and Fourth Run (11/08/26, 11/11/26)
Embarrasing 
# Useful Windows commands
Check process run time:
```powershell
New-TimeSpan -Start (get-process mayapy).StartTime
```
Check ALL processes:
```powershell
Tasklist
```
Check SPECIFIC processes by name:
```powershell
Tasklist | findstr /i "pattern"
```