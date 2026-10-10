@setlocal
@REM Paths are worked out from where this file sits, so it runs from any user's copy of the repo.
@REM A3_ROOT = the a3-stochastic-V1 folder (one level above this file)
@for %%I in ("%~dp0..") do @set "A3_ROOT=%%~fI"
@REM REPO_ROOT = the folder holding config.py (two levels above this file).
@REM Adding it to PYTHONPATH lets mayapy find config.py when run from Command Prompt or the pipeline.
@for %%I in ("%~dp0..\..") do @set "REPO_ROOT=%%~fI"
@if defined PYTHONPATH (set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%") else (set "PYTHONPATH=%REPO_ROOT%")

@REM "C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe" "%A3_ROOT%\b98-sampling-joint\d1-adhoc-for-b99.py" --max_iteration 20 --step_counter_internal_max 20 --num_eval_episode 1 --eval_interval 1 --keyframe_num 2 --keyframe_translation_num 2 --distance_threshold_percentage 0.9999 --distance_jaw_threshold_percentage 0.9999 --unit_denominator 1000.0 --sample_batch_size 2 2>&1 | findstr /V /C:"Frame rate mismatch" /C:"Cannot find procedure" /C:"File read in" /C:"Imported animation keys may not match" /C:"00:00:" /C:"Error: line 1:" /C:"qt.svg: Cannot open file" /C:"in gym make" /C:"seed " /C:" closing " /C:"Error: ziva "
"C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe" "%A3_ROOT%\b98-sampling-joint\d1-adhoc-for-b98.py" --max_iteration 20 --step_counter_internal_max 20 --num_eval_episode 1 --eval_interval 1 --keyframe_num 2 --keyframe_translation_num 2 --distance_threshold_percentage 0.9999 --distance_jaw_threshold_percentage 0.9999 --unit_denominator 1000.0 --sample_batch_size 2
