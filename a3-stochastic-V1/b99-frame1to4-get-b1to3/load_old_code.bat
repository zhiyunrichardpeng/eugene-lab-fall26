@echo off
@REM copy "C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo\example_train\a2-ad-hoc.py" %BACKUP_PATH%%BACKUP_FOLDER_NAME%
@REM have to comment it, to avoid wrongly overwrite. wasted 10 min today. 24-6-10. 25-12-5 wasted 10 min again. have to comment it out.
set BACKUP_FOLDER_NAME="g91 this is the weekend package3 25-12-7\"
@REM set BACKUP_FOLDER_NAME="prevent from wrong loading\"
set BACKUP_PATH="C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\"

@REM copy "C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files\deltaB_temp.pkl" %BACKUP_PATH%%BACKUP_FOLDER_NAME%
@REM copy "C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\240423-150653 log files\cp26-a001 after-deform-task b2.mb" %BACKUP_PATH%%BACKUP_FOLDER_NAME%

set FILE_NAME="m1_vertexMove_Va001_env.py"
set SOURCE=%BACKUP_PATH%%BACKUP_FOLDER_NAME%%FILE_NAME%
set DESTINATION="C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo\m1_vertexMoveEnvironment\m1_vertexMove\m1_vertexMove_Va001\envs"
copy %SOURCE% %DESTINATION%

@REM copy "C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\run_d1_b99.bat" %BACKUP_PATH%%BACKUP_FOLDER_NAME%
set FILE_NAME="run_d1_b99.bat"
set SOURCE=%BACKUP_PATH%%BACKUP_FOLDER_NAME%%FILE_NAME%
set DESTINATION="C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3"
copy %SOURCE% %DESTINATION%

@REM copy "C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo\training\evaluator.py" %BACKUP_PATH%%BACKUP_FOLDER_NAME%
set FILE_NAME="evaluator.py"
set SOURCE=%BACKUP_PATH%%BACKUP_FOLDER_NAME%%FILE_NAME%
set DESTINATION="C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo\training"
copy %SOURCE% %DESTINATION%


@REM @REM copy "C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo\training\off_sampler.py" %BACKUP_PATH%%BACKUP_FOLDER_NAME%
@REM set FILE_NAME="off_sampler.py"
@REM set SOURCE=%BACKUP_PATH%%BACKUP_FOLDER_NAME%%FILE_NAME%
@REM set DESTINATION="C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo\training"
@REM copy %SOURCE% %DESTINATION%

@REM copy "C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\d1-adhoc-for-b99.py" %BACKUP_PATH%%BACKUP_FOLDER_NAME%
set FILE_NAME="d1-adhoc-for-b99.py"
set SOURCE=%BACKUP_PATH%%BACKUP_FOLDER_NAME%%FILE_NAME%
set DESTINATION="C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3"
copy %SOURCE% %DESTINATION%

@REM copy "C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\d2-adhoc-for-b99-part2.py" %BACKUP_PATH%%BACKUP_FOLDER_NAME%
set FILE_NAME="d2-adhoc-for-b99-part2.py"
set SOURCE=%BACKUP_PATH%%BACKUP_FOLDER_NAME%%FILE_NAME%
set DESTINATION="C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3"
copy %SOURCE% %DESTINATION%

@REM copy "C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo\example_train\a2_fromFrameGetDeltaB.py" %BACKUP_PATH%%BACKUP_FOLDER_NAME%
set FILE_NAME="a2_fromFrameGetDeltaB.py"
set SOURCE=%BACKUP_PATH%%BACKUP_FOLDER_NAME%%FILE_NAME%
set DESTINATION="C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo\example_train"
copy %SOURCE% %DESTINATION%

@REM copy "C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo\example_train\j77_V3_estimate_r_limit_newest.py" %BACKUP_PATH%%BACKUP_FOLDER_NAME%

set FILE_NAME="j77_V3_estimate_r_limit_newest.py"
set SOURCE=%BACKUP_PATH%%BACKUP_FOLDER_NAME%%FILE_NAME%
set DESTINATION="C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo\example_train"
copy %SOURCE% %DESTINATION%

@REM @REM copy "C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\c96_finalized_datav21generating.py" %BACKUP_PATH%%BACKUP_FOLDER_NAME%
@REM set FILE_NAME="c96_finalized_datav21generating.py"
@REM set SOURCE=%BACKUP_PATH%%BACKUP_FOLDER_NAME%%FILE_NAME%
@REM set DESTINATION="C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3"
@REM copy %SOURCE% %DESTINATION%

@REM copy "C:\Users\zpa18\a3-stochastic-V1\a1-blendshape-guided-animation-pipeline\database\dataV22.py" %BACKUP_PATH%%BACKUP_FOLDER_NAME%
set FILE_NAME="dataV22.py"
set SOURCE=%BACKUP_PATH%%BACKUP_FOLDER_NAME%%FILE_NAME%
set DESTINATION="C:\Users\zpa18\a3-stochastic-V1\a1-blendshape-guided-animation-pipeline\database"
copy %SOURCE% %DESTINATION%

@REM copy "C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo\training\trainer.py" %BACKUP_PATH%%BACKUP_FOLDER_NAME%
set FILE_NAME="trainer.py"
set SOURCE=%BACKUP_PATH%%BACKUP_FOLDER_NAME%%FILE_NAME%
set DESTINATION="C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo\training"
copy %SOURCE% %DESTINATION%


set FILE_NAME="run_pipeline_update.py"
set SOURCE=%BACKUP_PATH%%BACKUP_FOLDER_NAME%%FILE_NAME%
set DESTINATION="C:\Users\zpa18\a3-stochastic-V1\automation"
copy %SOURCE% %DESTINATION%



@REM copy "C:\Users\zpa18\a3-stochastic-V1\a3-stochastic-V1-algo\training\trainer.py" %BACKUP_PATH%%BACKUP_FOLDER_NAME%
@REM set FILE_NAME="c96_finalized_datav21generating_testZiva.py"
@REM set SOURCE=%BACKUP_PATH%%BACKUP_FOLDER_NAME%%FILE_NAME%
@REM set DESTINATION="C:\Users\zpa18\a3-stochastic-V1\b98-sampling-joint"
@REM copy %SOURCE% %DESTINATION%







@REM set BACKUP_FOLDER_NAME="a62 rotation succeed though many errors"
@REM set BACKUP_PATH="C:\Users\zpa18\a3-stochastic-V1\backup\"
@REM copy "C:\Users\zpa18\a3-stochastic-V1\b99-frame1to4-get-b1to3\d2-adhoc-for-b99-part2.py" %BACKUP_PATH%%BACKUP_FOLDER_NAME%