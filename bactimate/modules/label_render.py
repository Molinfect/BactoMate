"""High-quality text label rendering helpers for overlays."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable


def _font_candidates() -> list[str]:
    return [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "arial.ttf",
        "DejaVuSans.ttf",
    ]


@lru_cache(maxsize=16)
def _load_font(size: int):
    from PIL import ImageFont

    px = max(8, int(size))
    for path in _font_candidates():
        try:
            return ImageFont.truetype(path, px)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_hq_labels(
    pil_img,
    labels: Iterable[tuple[str, int, int]],
    font_size: int = 12,
    scale: int = 2,
    fill_rgb: tuple[int, int, int] = (255, 255, 255),
    stroke_rgb: tuple[int, int, int] = (0, 0, 0),
    stroke_width: int = 2,
):
    """Draw anti-aliased labels with outline onto an RGB PIL image."""
    from PIL import Image, ImageDraw

    labels = list(labels)
    if not labels:
        return pil_img

    w, h = pil_img.size
    scale = max(1, int(scale))
    up = Image.new("RGBA", (w * scale, h * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(up)
    font = _load_font(max(8, int(font_size)) * scale)
    sw = max(1, int(stroke_width) * scale)

    fill_rgba = (*fill_rgb, 255)
    stroke_rgba = (*stroke_rgb, 255)
    for text, x, y in labels:
        draw.text(
            (int(x) * scale, int(y) * scale),
            str(text),
            fill=fill_rgba,
            font=font,
            stroke_width=sw,
            stroke_fill=stroke_rgba,
        )

    # Pillow compatibility: Image.Resampling in newer versions, constants in older versions.
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    up = up.resize((w, h), resample=resample)

    base = pil_img.convert("RGBA")
    base.alpha_composite(up)
    return base.convert("RGB")

