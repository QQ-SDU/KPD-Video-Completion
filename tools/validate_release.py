#!/usr/bin/env python3
"""Dependency-free static parser. Not a scientific experiment check."""
from pathlib import Path
import ast, json
ROOT=Path(__file__).resolve().parents[1]
py=list(ROOT.rglob('*.py')); js=list(ROOT.rglob('*.json'))
for f in py:
    if any(k in f.parts for k in ('.git','__pycache__')):continue
    ast.parse(f.read_text(encoding='utf-8-sig'),filename=str(f))
for f in js:json.loads(f.read_text(encoding='utf-8-sig'))
print(f'PASS: parsed {len(py)} Python files and {len(js)} JSON files.')
print('NOTE: no imports, datasets, GPU, MATLAB or paper results were verified.')
