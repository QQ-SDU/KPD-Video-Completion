#!/usr/bin/env python3
"""Write a LaTeX subsection only after complete FROZEN DAVIS runs.
No placeholder numbers are ever inserted into a manuscript.
"""
import argparse,csv,json,statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',required=True)
    p.add_argument('--out',default=None)
    args=p.parse_args()
    root=Path(args.results).resolve()
    cfg=json.loads((root/'RUN_CONFIG.json').read_text(encoding='utf-8'))
    if cfg['smoke_test']:
        raise SystemExit('Refusing to create paper LaTeX from smoke-test settings')
    files=list((root/'per_case').glob('*.json'))
    rows=[json.loads(x.read_text(encoding='utf-8')) for x in files]
    keys={(r['name'],round(r['rate_requested'],6),r['seed_requested'],r['method']) for r in rows}
    expected={(v,round(s,6),seed,method) for v in cfg['video_names'] for s in cfg['rates'] for seed in cfg['seeds'] for method in cfg['methods']}
    missing=expected-keys
    if missing:raise SystemExit(f'Refusing incomplete paper table: {len(missing)} cases missing')
    if not {'KPD','DIRECT','CROSSFIT'}<=set(cfg['methods']):
        raise SystemExit('Need all three KPD / Direct / CrossFit arms')
    L=[r'\subsection{Generalization to Unseen Video Sequences}',
       r'We further evaluate the fixed configuration on eight unseen video sequences from the DAVIS 2017 validation set. The first 50 frames of each sequence are converted to grayscale and resized to $144\times256$. We apply the same tube-wise random sampling protocol at $\mathrm{SR}\in\{0.2,0.4\}$, using three predetermined random seeds per video. All hyperparameters and CNN training budgets are unchanged from Table~\ref{tab:reference-config}. No clean-video frames are used for model fitting or parameter selection.',
       r'\begin{table}[t]',r'\centering',
       r'\caption{Cross-dataset completion results on eight DAVIS sequences (three seeds per sequence). All reported values are averages over videos and seeds. PSNR is in dB.}',
       r'\label{tab:davis-generalization}',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{ccccc}',r'\hline',
       r'SR & Method & PSNR & PSNR$_{\rm miss}$ & SSIM\\',r'\hline']
    for sr in cfg['rates']:
        for name in ['KPD','DIRECT','CROSSFIT']:
            rr=[x for x in rows if abs(x['rate_requested']-sr)<1e-9 and x['method']==name]
            avg=lambda field:statistics.mean(float(x[field]) for x in rr)
            L.append(f"{sr:.1f} & {('CrossFit' if name=='CROSSFIT' else ('Direct' if name=='DIRECT' else name))} & {avg('psnr_full'):.3f} & {avg('psnr_missing'):.3f} & {avg('ssim_full'):.4f}\\\\")
        L.append(r'\hline')
    L += [r'\end{tabular}',r'\end{table}']
    for sr in cfg['rates']:
        paired=[]
        for n in cfg['video_names']:
            for seed in cfg['seeds']:
                rr={r['method']:r for r in rows if r['name']==n and abs(r['rate_requested']-sr)<1e-9 and r['seed_requested']==seed}
                paired.append(rr['CROSSFIT']['psnr_missing']-rr['DIRECT']['psnr_missing'])
        pos=sum(x>0 for x in paired)
        L.append(f'At $\\mathrm{{SR}}={sr:.1f}$, CrossFit changes the average missing-region PSNR by {statistics.mean(paired):+.3f} dB relative to Direct and improves {pos}/{len(paired)} paired video–seed cases.')
    L.append(r'These comparisons assess the held-out residual-supervision mechanism on a separate video collection while retaining the same spatial tube-wise observation model.')
    path=Path(args.out).resolve() if args.out else root/'DAVIS_SECTION.tex'
    path.write_text('\n\n'.join(L)+'\n',encoding='utf-8')
    print('Generated LaTeX from fully completed runs:',path)
    print('NOTE: verify the Table I label in your manuscript; replace tab:reference-config if needed.')

if __name__=='__main__':main()
