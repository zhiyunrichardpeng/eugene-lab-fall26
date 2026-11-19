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
# Update Deprecated Fifth Run (11/19/26)
I did this one together with Claude  (through Claude Code). Claude read the code, the commit history and my logs, found the causes below, wrote the code changes and tested them on my laptop against a fake Maya. I ran everything in real Maya on the lab machine and sent the output back.
## The issue
Even with the analytical gradient, one phase still took 57 to 85 minutes, and `ooc_l`, `ooc_r` and `frontalis` always printed a distance of `0.0`. Richard's guess was that `compute_single_gradient` was still calling `get_full_gradient` somewhere.
## What we found
- `get_full_gradient` is **not** being called. `oor` and `ooc_l` return from the analytical branch first. In the Maya profile the analytical gradient is about 2.5 ms per step.
- Every run ended at step 36,600 no matter which gradient I used. The step count comes from the jaw: the env limits how fast it can rotate, using `expected_steps: 33076` for jaw threshold 0.1 in `JawRotationConfig` (`a2_fromFrameGetDeltaB.py`).
- The lips finish at about step 950. The other ~35,000 steps are the lips sitting still while the jaw catches up.
- The `0.0` distances are in the data, not the code. No phoneme pose (AA, EH, IY, N, T, UW, V, Y) moves `ooc_l`, `ooc_r` or `frontalis` at all. Only the AU poses do (AU1, AU2, AU6). So changing the target to T or IY would not have helped.
- On top of that, this branch could not have moved those tissues anyway: the evaluator sent zeros for `ooc_r` and `frontalis`, the env zeroed them again, and the env never re-read those three meshes from Maya.
- The failed run from 11/11 (almost 3 hours, no success) was because the jaw gradient had been replaced with zeros, so the jaw never moved. That was already fixed in `d91d611`.
- The time per step was almost all inside `env.step`, not the gradient. The profile showed 7.2 million `np.array_equal` calls (every joint coordinate compared one at a time) and the lip mesh read as the two biggest costs.
## What changed
- `0c50b52` evaluator: the numerical jaw gradient is recomputed every 10 steps instead of every step (and skipped when the jaw is frozen), the gradient plots are saved once at the end of a phase instead of every 100 iterations, and a `[timing]` line is printed every 1000 steps and saved to `step_timing.txt`.
- `7bc1e55` env: the per-joint loops in `step()` are vectorized. Same Maya calls, far less Python.
- `6994247` evaluator + env: `ooc_r` and `frontalis` are enabled next to `ooc_l` (calculators, observation slices, gradients, clipping, mesh read-back). Tissue thresholds in `run_d1_b99.bat` went from 0.9999 to 0.1.
- `33070aa` merge with origin. The `print` / `sys.exit(0)` debug lines from "print quit debugging" were left out, because with them the evaluator quits on the first gradient call.

New settings in the args dict of `a2_fromFrameGetDeltaB.py`: `jaw_gradient_every` (10), `gradient_plot_every` (0), `timing_report_every` (1000). Setting them to 1 / 100 / 0 gives the old behavior.
## What happened
All of these are 2,000-step capped runs on the lab machine (with cProfile on), not full phases.

| | ms/step while lips move | ms/step while waiting on jaw |
|---|---|---|
| Before today (11/12 full run, average) | 140 | 140 |
| After jaw caching + plots at end | 96.0 | 72.3 |
| After vectorizing `env.step` | 73.3 | 52.3 |

- Neutral to N gives the same numbers digit for digit after the changes (`3.4371030301023935` at step 1000), so the speedups did not change the result.
- Estimated full Neutral to N phase: about 32 minutes instead of 85. **Not measured yet.**
- AU6 target (lips + `ooc_l` + `ooc_r`): phase 1 hit success at step 890. Lips 17.55 to 1.59, `ooc_l` 21.64 to 1.95, `ooc_r` 32.69 to 3.27. This is the first time `ooc_l` or `ooc_r` actually moved on this branch.
- My first AU6 attempt was really another N run, because I edited `overwrite_dataV22_script.py` but the copy never happened. Check the banner: `initial_distance` should be 17.547 for AU6, not 38.166.
## What to do next?
- Run one full Neutral to N phase without the cap to confirm the ~32 minutes and that it still ends at the same distances as before.
- Test `frontalis` with AU1 or AU2 (AU6 does not move it). Those have zero lip and jaw motion, which has never been run.
- Look at the AU6 result in Maya. Success only means the sampled vertices got within 10% of the target.
- For AU tests set `PhaseManager(max_phases=1)` in `a2_fromFrameGetDeltaB.py`, otherwise it goes on to the phoneme chain (phase 2 = "N to AA"). `only_run_phase1_flag` is never read.
- Ask Richard:
  - Should the jaw schedule (33,076 steps) be re-fit now that the lips only need about 950 steps? This is what decides the run time now.
  - What per-step limit should `ooc_l`, `ooc_r` and `frontalis` use? Right now they use the lip limit as a placeholder.
  - Is this branch meant to reproduce the old results faster, or to do what main does? It is still 179 files away from main.
  - There is no phoneme O file, so what is the "O to T to IY" chain supposed to be?
- Not done yet: skipping the lip mesh read while the lips are frozen. It is the biggest cost left per step.
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