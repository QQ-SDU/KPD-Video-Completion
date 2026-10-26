@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist results\fig3_stagewise mkdir results\fig3_stagewise
python src\ours_v7\run_v51_one_case.py --input benchmark_suite\data\formal23\bridge_close_qcif.y4m --video-name "Bridge Close" --output results\fig3_stagewise\bridge_close --sampling-rate 0.30 --resize 144x176 --max-frames 100 --rank 4 --masked-sweeps 12 --lam-space 0.005 --lam-time 0.05 --ridge 1e-5 --protocol final --crossfit-folds 5 --steps 1200 --patch-t 16 --patch-h 48 --patch-w 48 --batch-size 2 --hidden 12 --lr 0.001 --detail-weight 2.0 --max-residual 0.25 --device auto --show-frames 50
python src\ours_v7\run_v51_one_case.py --input benchmark_suite\data\formal23\foreman_qcif.y4m --video-name Foreman --output results\fig3_stagewise\foreman --sampling-rate 0.30 --resize 144x176 --max-frames 100 --rank 4 --masked-sweeps 12 --lam-space 0.005 --lam-time 0.05 --ridge 1e-5 --protocol final --crossfit-folds 5 --steps 1200 --patch-t 16 --patch-h 48 --patch-w 48 --batch-size 2 --hidden 12 --lr 0.001 --detail-weight 2.0 --max-residual 0.25 --device auto --show-frames 50
python src\ours_v7\run_v51_one_case.py --input benchmark_suite\data\formal23\mobile_cif.y4m --video-name Mobile --output results\fig3_stagewise\mobile --sampling-rate 0.30 --resize 288x352 --max-frames 50 --rank 4 --masked-sweeps 12 --lam-space 0.005 --lam-time 0.05 --ridge 1e-5 --protocol final --crossfit-folds 5 --steps 1200 --patch-t 16 --patch-h 48 --patch-w 48 --batch-size 2 --hidden 12 --lr 0.001 --detail-weight 2.0 --max-residual 0.25 --device auto --show-frames 25
pause
