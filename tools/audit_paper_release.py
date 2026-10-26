#!/usr/bin/env python3
"""Static release-scope checks; do not substitute for numerical reproduction."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
MAPPING=ROOT/'PAPER_CODE_MAPPING.md'
assert MAPPING.is_file()
assert not (ROOT/'benchmark_suite/Methods/Ours/reference_v7').exists()
for rel in ('src/ours_v7/masked_kpd.py','src/ours_v7/detail_refiner_v41.py',
            'benchmark_suite/Methods/Ours/run_same_mask.py',
            'experiments/09_davis2017_all9/davis9.py',
            'scripts/reproduce_rgb_ours.py',
            'experiments/19_final_heterogeneous_360p/video_supplement.py'):
    assert (ROOT/rel).is_file(),f'Missing critical paper file: {rel}'
for rel in ('benchmark_suite/Methods/Ours/run_same_mask.py',
            'src/ours_v7/run_same_mask_benchmark_adapter.py'):
    s=(ROOT/rel).read_text(encoding='utf-8')
    assert 'reference_v7' not in s and 'spatial_kernel' in s
s=(ROOT/'experiments/19_final_heterogeneous_360p/final_davis.py').read_text(encoding='utf-8')
module=ast.parse(s)
eight=[a.value for n in module.body if isinstance(n,ast.Assign) for target in n.targets if isinstance(target,ast.Name) and target.id=='EIGHT' for a in n.value.elts]
assert set(eight)=={'bear','drift-chicane','cows','dog','camel','kite-surf','parkour','breakdance'}
assert 'blackswan' not in eight
print('PASS: paper release paths, canonical core, RGB sequence selection and kernel option')
