@setlocal
@REM Paths are worked out from where this file sits, so it runs from any user's copy of the repo.
@REM A3_ROOT = the a3-stochastic-V1 folder (one level above this file)
@for %%I in ("%~dp0..") do @set "A3_ROOT=%%~fI"
@REM REPO_ROOT = the folder holding config.py (two levels above this file).
@REM Adding it to PYTHONPATH lets mayapy find config.py when run from Command Prompt or the pipeline.
@for %%I in ("%~dp0..\..") do @set "REPO_ROOT=%%~fI"
@if defined PYTHONPATH (set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%") else (set "PYTHONPATH=%REPO_ROOT%")

"C:\Program Files\Autodesk\Maya2023\bin\mayapy.exe" "%A3_ROOT%\b95-debug-use\a99-debug-frame16-issue.py"
