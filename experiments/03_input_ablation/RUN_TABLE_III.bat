@echo off
setlocal
cd /d "%~dp0\..\.."
python experiments\common_ablation\run_paper_ablation.py ^
 --study inputs --candidates B,BR,BRM,FULL ^
 --root benchmark_suite ^
 --manifest data\formal23\development_manifest.csv ^
 --videos Akiyo,Foreman,Mobile ^
 --sampling-rate 0.20 --train-fraction 0.80 ^
 --rank 4 --sweeps 12 --lam-space 0.005 --lam-time 0.05 --ridge 1e-5 ^
 --folds 5 --steps 1200 --patch-t 16 --patch-h 48 --patch-w 48 ^
 --batch-size 2 --hidden 12 --lr 0.001 --detail-weight 2.0 --max-residual 0.25 ^
 --device auto ^
 --output ..\results\table3_input_ablation
pause
