@echo off
setlocal
cd /d "%~dp0\..\..\benchmark_suite"
python color_structured9\run_all_color_structured.py --videos Coastguard Container --masks block grid
pause
