@echo off
setlocal
cd /d "%~dp0"
python replot_fig1_from_csv.py ^
 --csv results_fig1_3dev\ALL_FIG1_POINTS.csv ^
 --output results_fig1_3dev\figures
pause
