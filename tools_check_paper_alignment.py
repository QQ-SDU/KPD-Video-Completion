from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/'IMPLEMENTATION_CONFIG.json').read_text(encoding='utf-8'))
bench=json.loads((ROOT/'benchmark_suite/config/method_parameters.json').read_text(encoding='utf-8'))
exp=json.loads((ROOT/'benchmark_suite/config/experiment.json').read_text(encoding='utf-8'))
ours=bench['Ours']
checks={
 'no_fixed_seed': exp['sampling'].get('seeds', []) == [],
 'K=4': ours['rank']==cfg['backbone']['rank_K']==4,
 'sweeps=12': ours['masked_sweeps']==cfg['backbone']['masked_sweeps_S']==12,
 'lambda_s': ours['lambda_s']==cfg['backbone']['lambda_s'],
 'lambda_t': ours['lambda_t']==cfg['backbone']['lambda_t'],
 'lambda_0': ours['lambda_0']==cfg['backbone']['lambda_0'],
 'F=5': ours['crossfit_folds']==cfg['crossfit']['folds_F']==5,
 'hidden=12': ours['hidden']==cfg['cnn']['hidden_channels']==12,
 'steps=1200': ours['steps']==cfg['cnn']['steps']==1200,
 'five SRs': exp['sampling']['sampling_rates']==[0.1,0.2,0.3,0.4,0.5],
 '9 methods': exp['methods']['all']==cfg['benchmark23']['methods'],
}
for k,v in checks.items(): print(('PASS' if v else 'FAIL'),k)
if not all(checks.values()): sys.exit(2)
print('All frozen paper-configuration checks passed.')
