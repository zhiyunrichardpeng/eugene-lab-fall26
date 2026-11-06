# Update Deprecated First Run (11/05/26)
## The issue with this version
The previous version of the codebase took around 2 hours to run a single transition (phase) in `d1-adhoc-for-b999.py` because it was using a numerical gradient descent method and also not using the `deformation-path-db.json`.

This is what I understood from my talk with Richard last week. Maybe I'm missing some things.
## What was going to change
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