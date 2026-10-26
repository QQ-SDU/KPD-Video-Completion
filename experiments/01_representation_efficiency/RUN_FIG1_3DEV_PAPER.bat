@echo off
setlocal
cd /d "%~dp0"

python run_fig1_3dev.py ^
 --inputs data\akiyo_qcif.y4m data\foreman_qcif.y4m data\mobile_qcif.y4m ^
 --names Akiyo Foreman Mobile ^
 --output results_fig1_3dev ^
 --max-frames 100 ^
 --kpd-max-rank 36 ^
 --kpd-sweeps 40 ^
 --kpd-restarts 2 ^
 --tsvd-max-rank 32 ^
 --tucker-max-frac 0.40 ^
 --points 16 ^
 --max-budget-pct 30 ^

echo.
echo Finished.
echo See results_fig1_3dev\figures
pause
