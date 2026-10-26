@echo off
setlocal
cd /d "%~dp0\..\.."
set /p DAVISROOT=Input extracted DAVIS root directory: 
python experiments/08_davis2017_generalization/prepare_davis.py --davis-root "%DAVISROOT%"
if errorlevel 1 goto :err
python experiments/08_davis2017_generalization/run_davis_frozen.py --device auto
if errorlevel 1 goto :err
python experiments/08_davis2017_generalization/summarize_davis.py --input results/davis2017_new
if errorlevel 1 goto :err
 echo DONE. See results\davis2017_new\AGGREGATE.csv
 pause
 exit /b 0
:err
 echo ERROR: see console output
 pause
 exit /b 1
