@echo off
setlocal
cd /d "%~dp0\..\.."
set COMMON=--root benchmark_suite --manifest data\formal23\development_manifest.csv --videos Akiyo,Foreman,Mobile --sampling-rate 0.20 --train-fraction 0.80 --rank 4 --sweeps 12 --lam-space 0.005 --lam-time 0.05 --ridge 1e-5 --folds 5 --steps 1200 --patch-t 16 --patch-h 48 --patch-w 48 --batch-size 2 --hidden 12 --lr 0.001 --detail-weight 2.0 --max-residual 0.25 --input-mode FULL --device auto
python experiments\common_ablation\run_paper_ablation.py --study rank      --candidates 4,8,12,16,20 %COMMON% --output ..\results\fig4_rank
python experiments\common_ablation\run_paper_ablation.py --study folds     --candidates 2,3,5,8      %COMMON% --output ..\results\fig4_folds
python experiments\common_ablation\run_paper_ablation.py --study steps     --candidates 300,600,900,1200 %COMMON% --output ..\results\fig4_steps
python experiments\common_ablation\run_paper_ablation.py --study lam_space --candidates 0.001,0.005,0.01,0.05 %COMMON% --output ..\results\fig4_lambda_s
python experiments\common_ablation\run_paper_ablation.py --study lam_time  --candidates 0.01,0.05,0.1,0.5 %COMMON% --output ..\results\fig4_lambda_t
pause
