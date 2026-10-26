@echo off
setlocal
cd /d "%~dp0\..\..\benchmark_suite"
python workflow\run_formal23_all.py
pause
