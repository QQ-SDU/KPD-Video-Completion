from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def _get_font(size: int):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf',
        'arial.ttf',
    ]
    for c in candidates:
        if Path(c).exists() or c == 'arial.ttf':
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def observed_grid_mask(h: int, w: int, period: int = 64, line_width: int = 2,
                       offset_y: int = 18, offset_x: int = 22) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    missing = (((yy - offset_y) % period) < line_width) | (((xx - offset_x) % period) < line_width)
    return ~missing


def observed_text_mask(h: int, w: int, font_size: int | None = None, stroke: int = 1) -> np.ndarray:
    img = Image.new('L', (w, h), color=0)
    draw = ImageDraw.Draw(img)
    if font_size is None:
        font_size = max(18, int(min(h, w) * 0.155))
    font = _get_font(font_size)
    lines = ['KPD', 'VIDEO']
    y_positions = [int(h * 0.34), int(h * 0.58)]
    for text, y in zip(lines, y_positions):
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
        tw = bbox[2] - bbox[0]
        x = (w - tw) // 2
        draw.text((x, y), text, fill=255, font=font, stroke_width=stroke, stroke_fill=255)
    missing = np.asarray(img, dtype=np.uint8) > 0
    return ~missing


def observed_block_mask(h: int, w: int, missing_ratio: float = 0.06,
                        center_x_frac: float = 0.62, center_y_frac: float = 0.50) -> np.ndarray:
    side = int(round(np.sqrt(missing_ratio * h * w)))
    side = max(8, min(side, min(h, w)))
    missing = np.zeros((h, w), dtype=bool)
    x0 = int(w * center_x_frac - side / 2)
    y0 = int(h * center_y_frac - side / 2)
    x0 = max(0, min(w - side, x0))
    y0 = max(0, min(h - side, y0))
    missing[y0:y0 + side, x0:x0 + side] = True
    return ~missing


def build_observed_mask(mask_type: str, h: int, w: int, seed: int = 1) -> np.ndarray:
    mt = mask_type.lower()
    if mt == 'grid':
        return observed_grid_mask(h, w)
    if mt == 'text':
        return observed_text_mask(h, w)
    if mt == 'block':
        return observed_block_mask(h, w)
    raise KeyError(f'Unknown mask type: {mask_type}')
