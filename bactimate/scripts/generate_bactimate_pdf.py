#!/usr/bin/env python3
"""
Generate BactiMate_Overview.pdf — high-end marketing brochure.
Run from project root: python -m bactimate.scripts.generate_bactimate_pdf

Uses matplotlib. Launches BactiMate briefly to capture each tab, then builds
a premium brochure. High DPI; hero screenshots; body copy and bullet points per tab (from user manual).
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import bactimate.gui.patches  # noqa: F401

try:
    from bactimate.gui.constants import LOGS_DIR
    from bactimate.gui.log_config import setup_logging
    import logging
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(logging.WARNING, logs_dir=LOGS_DIR)
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

# PDF and image quality
PDF_DPI = 180

# Palette
DARK = "#0d1b2a"
MID = "#1b263b"
SLATE = "#415a77"
ACCENT = "#c9a227"
LIGHT = "#e0e1dd"
WHITE = "#ffffff"
TEXT = "#1a1a1a"
TEXT_LIGHT = "#4a5568"
BG_SOFT = "#f8f9fa"

# Tab name, tagline, body copy, bullets (key things you can do — from user manual)
# Order matches app tabs: 1 TIFF, 2 Omnipose, 3 U-NET, 4 Analyze, 5 Tracking, 6 Post-processing
TAB_INFO = [
    ("TIFF Conversion", "Convert once. Analyze everywhere.",
     "Turn ND2, LIF, and CZI files into TIFF with one workflow. Your pipeline starts here.",
     ("Set frame count and channel handling (separate or combined)",
      "Optional drift correction (X,Y registration) and photobleaching correction",
      "Preview before convert; batch conversion for folders",
      "Mark channels as FI or Foci for Analyze Masks")),
    ("Omnipose Segment", "State-of-the-art cell segmentation.",
     "Segment bacteria with built-in or your own trained models. Live preview, batch, or train.",
     ("Choose built-in model (e.g. bact_phase_omni) or your trained .pth model",
      "Diameter (0=auto), flow & mask threshold; GPU, DBSCAN, overlay",
      "Preview with overlay and frame navigation; Process file or folder",
      "Train custom model from image/mask pairs; transfer learning supported")),
    ("U-NET Segment", "Your model. Your workflow.",
     "Prefer U-Net? Train in BactiMate or use Building U-NET. Preview and batch.",
     ("Model dropdown: trained .keras from BactiMate or train new",
      "Training: image/mask pairs, filters, tile & prepare; optional Building U-NET path",
      "Preview U-NET Segmentation; Process file or folder",
      "Output to _segmentation_unet/ or next to input")),
    ("Analyze Masks", "From masks to metrics in one click.",
     "Filter by morphology and size, export to Excel. Optional FI and foci detection.",
     ("Morphological filters: aspect ratio, eccentricity, solidity; size filters (area, width, length)",
      "Export to Excel: Summary + Per-Cell Data; filter visualization TIFF",
      "Analyze FI (fluorescence intensity); match by name or one file for all",
      "Run foci analysis: Manual, Auto-detect, or Spotiflow; combined summary across folder")),
    ("Tracking", "Follow every cell. Every frame.",
     "Trackastra delivers cell lineage and persistent IDs. Link to images and Excel.",
     ("Mask TIFF (time series) required; optional original TIFF and Excel from Analyze Masks",
      "Model: general_2d or general_2d_w_SAM2_features; Mode: greedy, greedy_nodiv, or ilp",
      "Run Tracking → tracked masks, lineage, persistent cell IDs",
      "Device: CPU, CUDA, or Apple MPS")),
    ("Post-processing", "Publication-ready in minutes.",
     "Scale bars, montages, growth-curve videos. From data to figure in one place.",
     ("Add scale bar: length, position, color; pixel size from metadata",
      "Montage: folder or files → grid (phase, FI channels, merged); labels, spacing",
      "Growth curve video: TIFF sequence + Excel → side-by-side graph and video",
      "Output: TIFF, AVI, GIF, MP4; time per frame and FPS settings")),
]


def _find_tab_widget(main_window):
    from PyQt6.QtWidgets import QTabWidget
    cw = main_window.centralWidget()
    if not cw:
        return None
    layout = cw.layout()
    if not layout or layout.count() < 2:
        return None
    w = layout.itemAt(1).widget()
    return w if isinstance(w, QTabWidget) else None


def _capture_screenshots(app, window, out_dir: Path) -> list[Path]:
    paths = []
    tabs = _find_tab_widget(window)
    if not tabs:
        return paths
    for i in range(tabs.count()):
        tabs.setCurrentIndex(i)
        for _ in range(20):
            app.processEvents()
            time.sleep(0.05)
        pixmap = window.grab()
        if pixmap.isNull():
            continue
        path = out_dir / f"tab_{i:02d}.png"
        if pixmap.save(str(path), "PNG"):
            paths.append(path)
    return paths


def _draw_accent_line(ax, y: float, color: str = ACCENT, lw: float = 2) -> None:
    ax.plot([0, 1], [y, y], color=color, lw=lw, transform=ax.transAxes, solid_capstyle="round")


def _cover_page(pdf: PdfPages, logo_path: Path | None) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()

    # Top accent line
    _draw_accent_line(ax, 0.92, ACCENT, 3)
    # Logo in upper third; title and tagline below with clear spacing (no overlap)
    if logo_path and logo_path.exists():
        try:
            img = mimg.imread(str(logo_path))
            ax_logo = fig.add_axes([0.34, 0.58, 0.32, 0.18])
            ax_logo.imshow(img)
            ax_logo.axis("off")
        except Exception:
            pass
    # Title and tagline clearly below logo (no overlap)
    ax.text(0.5, 0.38, "BactiMate", ha="center", va="top", fontsize=44, fontweight="bold", color=WHITE, fontfamily="sans-serif")
    ax.text(0.5, 0.46, "Your mate for bacteria microscopy", ha="center", va="top", fontsize=15, color=LIGHT, fontfamily="sans-serif")
    ax.text(0.5, 0.54, "From raw data to publication — one workflow.", ha="center", va="top", fontsize=13, color=SLATE, fontfamily="sans-serif")
    ax.plot([0.25, 0.75], [0.06, 0.06], color=ACCENT, lw=1.5, transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight", facecolor=DARK, edgecolor="none", dpi=PDF_DPI)
    plt.close(fig)


def _value_proposition_page(pdf: PdfPages) -> None:
    """Overview: 2-column grid. 1 at top-left → 6 at bottom-right. Accent line below header, not through title."""
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(WHITE)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_position([0.05, 0.05, 0.9, 0.9])

    # Header at top of page (transAxes: high y = top)
    ax.text(0.5, 0.96, "Why BactiMate", ha="center", va="top", fontsize=24, fontweight="bold", color=DARK, fontfamily="sans-serif", transform=ax.transAxes)
    ax.text(0.5, 0.91, "One application. Every step of your pipeline.", ha="center", va="top", fontsize=11, color=TEXT_LIGHT, fontfamily="sans-serif", transform=ax.transAxes)
    # Accent line below subtitle as a divider
    _draw_accent_line(ax, 0.84, ACCENT, 2)
    # Footer at bottom of page
    ax.text(0.5, 0.03, "Overview", ha="center", va="bottom", fontsize=10, color=SLATE, fontfamily="sans-serif", transform=ax.transAxes)

    # Grid: row_y from top to bottom so 1,2 are top row and 5,6 are bottom row
    items = [
        ("1", "TIFF Conversion", "ND2, LIF, CZI → TIFF. Metadata, drift & photobleaching."),
        ("2", "Omnipose Segment", "Built-in & custom models. Preview, batch, train your own."),
        ("3", "U-NET Segment", "U-Net or Building U-NET. Preview and batch prediction."),
        ("4", "Analyze Masks", "Morphology filters, Excel, fluorescence & foci detection."),
        ("5", "Tracking", "Trackastra lineage. Persistent IDs. Link to analysis."),
        ("6", "Post-processing", "Scale bars, montages, growth-curve videos."),
    ]
    col_x = [0.08, 0.52]
    # First row at top (high y), last row at bottom (low y): 0.74, 0.46, 0.18
    row_y = [0.74, 0.46, 0.18]
    for i, (num, title, desc) in enumerate(items):
        col = i % 2
        row = i // 2
        x, y = col_x[col], row_y[row]
        ax.add_patch(FancyBboxPatch((x, y - 0.01), 0.08, 0.04, boxstyle="round,pad=0.006", facecolor=MID, edgecolor="none", transform=ax.transAxes))
        ax.text(x + 0.02, y, num, ha="left", va="center", fontsize=10, fontweight="bold", color=WHITE, transform=ax.transAxes)
        ax.text(x + 0.12, y, title, ha="left", va="center", fontsize=10, fontweight="bold", color=DARK, transform=ax.transAxes)
        # Two lines so text stays in column (no overflow)
        lines = textwrap.wrap(desc, width=32)
        desc_str = "\n".join(lines[:2]) if len(lines) >= 2 else (lines[0] if lines else "")
        ax.text(x, y - 0.058, desc_str, ha="left", va="top", fontsize=9, color=TEXT, transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight", facecolor=WHITE, edgecolor="none", dpi=PDF_DPI)
    plt.close(fig)


def _tab_page(
    pdf: PdfPages,
    tab_name: str,
    tagline: str,
    body: str,
    bullets: tuple[str, ...],
    screenshot_path: Path | None,
    page_num: int,
) -> None:
    """Hero screenshot + headline + tagline + body + bullet list (what you can do on this tab)."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(WHITE)

    # Slim top bar
    ax_bar = fig.add_axes([0, 0.96, 1, 0.04])
    ax_bar.set_facecolor(DARK)
    ax_bar.set_axis_off()
    fig.text(0.94, 0.978, f"{page_num:02d}", ha="center", va="center", fontsize=9, color=ACCENT, fontweight="bold")
    fig.text(0.06, 0.978, tab_name.upper(), ha="left", va="center", fontsize=9, color=LIGHT, fontfamily="sans-serif")

    # Hero image: slightly reduced height to leave room for body + bullets
    if screenshot_path and screenshot_path.exists():
        try:
            img = mimg.imread(str(screenshot_path))
            ax_main = fig.add_axes([0.04, 0.36, 0.92, 0.58])
            ax_main.imshow(img)
            ax_main.axis("off")
        except Exception:
            pass

    # Headline and tagline above image
    fig.text(0.06, 0.94, tab_name, ha="left", va="top", fontsize=26, fontweight="bold", color=DARK, fontfamily="sans-serif")
    fig.text(0.06, 0.895, tagline, ha="left", va="top", fontsize=12, color=SLATE, fontstyle="italic", fontfamily="sans-serif")

    # Content box: body + bullets (what you can do on this tab)
    box_bottom, box_height = 0.06, 0.28
    ax_patch = fig.add_axes([0, 0, 1, 1])
    ax_patch.set_axis_off()
    ax_patch.set_zorder(0)
    ax_patch.add_patch(FancyBboxPatch(
        (0.04, box_bottom), 0.92, box_height,
        boxstyle="round,pad=0.02", facecolor=BG_SOFT, edgecolor="#dee2e6", linewidth=0.5,
        transform=fig.transFigure, alpha=0.98,
    ))
    # Body paragraph at top of box
    wrapped_body = "\n".join(textwrap.wrap(body, width=78))
    fig.text(0.08, box_bottom + box_height - 0.02, wrapped_body, ha="left", va="top", fontsize=10, color=TEXT, fontfamily="sans-serif")
    # Bullets below body: "What you can do" — key points from user manual
    body_lines = max(1, wrapped_body.count("\n") + 1)
    bullet_y = box_bottom + box_height - 0.02 - body_lines * 0.024 - 0.012
    line_height = 0.021
    for b in bullets:
        b_wrapped = "\n".join(textwrap.wrap(b, width=72))
        fig.text(0.08, bullet_y, "• " + b_wrapped.replace("\n", "\n  "), ha="left", va="top", fontsize=9, color=TEXT_LIGHT, fontfamily="sans-serif")
        n_lines = max(1, b_wrapped.count("\n") + 1)
        bullet_y -= n_lines * line_height

    pdf.savefig(fig, bbox_inches="tight", facecolor=WHITE, edgecolor="none", dpi=PDF_DPI)
    plt.close(fig)


def _build_pdf(screenshot_paths: list[Path], output_pdf: Path, logo_path: Path | None) -> None:
    with PdfPages(str(output_pdf)) as pdf:
        _cover_page(pdf, logo_path)
        _value_proposition_page(pdf)
        for idx, (tab_name, tagline, body, bullets) in enumerate(TAB_INFO):
            path = screenshot_paths[idx] if idx < len(screenshot_paths) else None
            _tab_page(pdf, tab_name, tagline, body, bullets, path, page_num=idx + 1)
        d = pdf.infodict()
        d["Title"] = "BactiMate — Your mate for bacteria microscopy"
        d["Author"] = "BactiMate"


def main() -> None:
    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication
    from bactimate.gui.main_window import MainWindow
    from bactimate.gui.styles import get_modern_stylesheet
    from bactimate.gui.paths import get_app_root

    output_pdf = get_app_root() / "docs" / "BactiMate_Overview.pdf"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    logo_path = get_app_root() / "img" / "logo2.png"

    app = QApplication(sys.argv)
    app.setApplicationName("BactiMate")
    app.setStyle("Fusion")
    app.setStyleSheet(get_modern_stylesheet())
    try:
        from bactimate.gui.main_window import _get_app_icon_path
        icon_path = _get_app_icon_path()
        if icon_path:
            app.setWindowIcon(QIcon(str(icon_path)))
    except Exception:
        pass

    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()

    def do_capture_and_pdf() -> None:
        try:
            with tempfile.TemporaryDirectory(prefix="bactimate_pdf_") as tmp:
                temp_dir = Path(tmp)
                screenshot_paths = _capture_screenshots(app, window, temp_dir)
                _build_pdf(screenshot_paths, output_pdf, logo_path)
            print(f"PDF written: {output_pdf}")
        except Exception as e:
            print(f"Error building PDF: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            window.close()
            app.quit()

    QTimer.singleShot(2500, do_capture_and_pdf)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
