@echo off
REM Quick fix script for JSON path formatting issues

python "%~dp0fix_json_paths.py" "%~dp0deformation_path_db.json"

pause
