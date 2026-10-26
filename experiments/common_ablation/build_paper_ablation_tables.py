from pathlib import Path
import argparse
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--table2', default='results/table2_supervision_ablation/RESULTS.csv')
ap.add_argument('--table3', default='results/table3_input_ablation/RESULTS.csv')
ap.add_argument('--out', default='results/paper_tables')
a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

p=Path(a.table2)
if p.exists():
    d=pd.read_csv(p)
    rows=[]
    for cand,g in d.groupby('candidate',sort=False):
        row={'Method':cand}
        for _,r in g.iterrows():
            row[str(r['video'])]=f"{r['test_psnr_full']:.3f}/{r['test_ssim_full']:.4f}"
        row['Val_MAE']=g['strict_val_mae'].mean()
        row['Avg_PSNR']=g['test_psnr_full'].mean()
        row['Avg_SSIM']=g['test_ssim_full'].mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'TableII_reproduced.csv',index=False,encoding='utf-8-sig')

p=Path(a.table3)
if p.exists():
    d=pd.read_csv(p)
    rows=[]
    base=None
    for cand,g in d.groupby('candidate',sort=False):
        pmean=g['test_psnr_full'].mean()
        if base is None: base=pmean
        rows.append({'Input':cand,'Val_MAE':g['strict_val_mae'].mean(),'PSNR':pmean,'SSIM':g['test_ssim_full'].mean(),'Delta_PSNR':pmean-base})
    pd.DataFrame(rows).to_csv(out/'TableIII_reproduced.csv',index=False,encoding='utf-8-sig')
print('Saved to',out)
