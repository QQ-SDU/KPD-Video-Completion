#!/usr/bin/env python3
"""Make a publication-style LaTeX table only from a COMPLETE nine-method run."""
import argparse,csv
from pathlib import Path
import numpy as np
from davis9 import ALL,RATES,read_csv

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True,help='davis2017_all9 results directory')
    a=ap.parse_args();root=Path(a.output)
    src=root/'SUMMARY_COMPLETE.csv'
    if not src.exists():raise SystemExit('SUMMARY_COMPLETE.csv missing: all 9 methods must finish first')
    rows=read_csv(src);table={(float(r['sampling_rate']),r['method']):r for r in rows}
    rates=sorted({r for r,_ in table})
    for sr in rates:
        for m in ALL:
            if (sr,m) not in table or int(table[sr,m]['completed'])!=int(table[sr,m]['expected']):
                raise SystemExit(f'Incomplete for SR={sr}, {m}')
    keys=('psnr_full_mean','ssim_full_mean')
    wins={}; seconds={}
    for sr in rates:
        for key in keys:
            scores={m:float(table[sr,m][key]) for m in ALL}
            order=sorted(scores,key=scores.get,reverse=True)
            wins[(sr,key)]=order[0];seconds[(sr,key)]=order[1]
    names=sorted(set(r['name'] for r in read_csv(root/'per_run_metrics.csv')))
    lines=[r'\begin{table*}[t]',r'\centering',r'\caption{Comparison of nine video completion methods on '+str(len(names))+' selected DAVIS 2017 video sequence(s) under random tube-wise sampling using one fixed seed. All methods use identical observation masks.}',r'\label{tab:davis9}',r'\setlength{\tabcolsep}{5pt}',r'\begin{tabular}{l'+'cc'*len(rates)+'}',r'\hline']
    lines.append('Method & '+' & '.join(r'\multicolumn{2}{c}{SR='+f'{sr:.1f}'+'}' for sr in rates)+r' \\')
    lines.append(' & '+' & '.join(['PSNR','SSIM']*len(rates))+r' \\')
    lines.append(r'\hline')
    for m in ALL:
        cells=[]
        for sr in rates:
            r=table[sr,m]
            for key in keys:
                val=float(r[key]);s=f'{val:.3f}' if key=='psnr_full_mean' else f'{val:.4f}'
                if wins[(sr,key)]==m:s=r'\textbf{'+s+'}'
                elif seconds[(sr,key)]==m:s=r'\underline{'+s+'}'
                cells.append(s)
        name=r'S$^2$NTNN' if m=='S2NTNN' else m
        lines.append(name+' & '+' & '.join(cells)+r' \\')
    lines += [r'\hline',r'\end{tabular}',r'\end{table*}']
    (root/'DAVIS9_TABLE.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote',root/'DAVIS9_TABLE.tex')
if __name__=='__main__':main()
