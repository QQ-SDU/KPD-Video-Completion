from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data_color_structured9'
INPUT_DIR = DATA_DIR / 'inputs'
MANIFEST = DATA_DIR / 'manifest.csv'
ART = ROOT / 'artifacts_color_structured9'
CASES_DIR = ART / 'cases_channels'
CASES_RGB_DIR = ART / 'cases_rgb'
RESULTS_CH_DIR = ART / 'results_channels'
RESULTS_RGB_DIR = ART / 'results_rgb'
METRICS_DIR = ART / 'metrics'
FIGURES_DIR = ART / 'figures'
LOGS_DIR = ART / 'logs'
ASSETS_DIR = ART / 'image_assets'
PREVIEW_DIR = ART / 'case_previews'

MASK_TYPES = ['grid', 'text', 'block']
CHANNELS = [('R', 0), ('G', 1), ('B', 2)]
METHODS_ORDER = ['DRO-TFF', 'HLRTF', 'KBR', 'LRTFR', 'OTLRM', 'TNN', 'SALTS', 'S2NTNN', 'Ours']
DEFAULT_FRAME_INDEX = 14
