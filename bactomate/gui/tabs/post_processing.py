"""
Post-processing tab: growth curve video, scale bar on image, and montage.
"""
from __future__ import annotations

import math
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QColorDialog,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from bactomate.gui.image_canvas import ImageCanvas
from bactomate.gui.module_shell import PreviewSurface, mount_embedded_preview
from bactomate.gui.preview.preview_window import (
    CollapsibleSection,
    LUT_HIST_CANVAS_HEIGHT_PX,
    LUT_HIST_FIGSIZE_INCHES,
    apply_lut_histogram_figure_margins,
)
from bactomate.gui.image_utils import to_qpixmap, scale_for_preview
from bactomate.gui.preferences import (
    delete_montage_preset,
    get_default_browse_dir,
    list_montage_preset_names,
    load_montage_preset,
    save_montage_preset,
)
from bactomate.modules.tiff_metadata import (
    normalize_tiff_array_to_tcyx,
    read_calibration_from_path,
    read_metadata_from_path,
)
from bactomate.gui.session_params import apply_to_widgets, get_from_widgets
from bactomate.gui.utils import safe_imread
from bactomate.gui.widgets import (
    DragDropGroupBox,
    create_info_icon,
    show_file_selection_dialog,
    show_task_complete_dialog,
)

from bactomate.gui.layout_helpers import (
    ResponsivePanel,
    build_path_drop_group,
    stacked_path_block,
    style_compact_action_button,
    style_compact_browse_button,
    style_module_primary_button,
    vbox_from_rows,
)
from bactomate.gui.styles import FLAT_GROUPBOX_FRAME_STYLESHEET
from bactomate.gui.workers import (
    GrowthCurveVideoWorker,
    apply_video_montage_overlays,
    compose_growth_curve_combined_frame,
    load_growth_curve_summary_data,
    load_growth_curve_video_frames_rgb,
)

class PostProcessingTab(QWidget):
    """Tab for post-processing: growth curve video, scale bar on image, and montage."""

    MONTAGE_AUTO_LUT_LOW = 0.5
    MONTAGE_AUTO_LUT_HIGH = 99.8

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self._module_preview_mount: Optional[QWidget] = None
        self._preview_surface: Optional[PreviewSurface] = None
        self._embedded_post_preview: Optional[QWidget] = None
        self._active_montage_preview: Optional[QWidget] = None
        self._post_preview_mode = "montage"
        self._video_preview_cache: Optional[dict[str, Any]] = None
        self._video_preview_refresh_timer: Optional[QTimer] = None
        self._video_montage_widgets_ready = False
        self.montage_channel_color_overrides: dict = {}  # (path_stem, ch_idx) -> "auto"|"gray"|"red"|"green"|...
        self.montage_tile_color_overrides: Dict[Any, Tuple[float, float, float]] = {}  # tile label or idx -> (r,g,b)
        self.montage_tile_percentile_overrides: Dict[Any, Tuple[float, float]] = {}  # tile label or idx -> (p_low, p_high)
        self.montage_excluded_tile_keys: set[str] = set()
        self.montage_tile_order: list[str] = []
        self.montage_tile_z_modes: Dict[str, str] = {}  # tile label -> global|first|max
        self._montage_z_projection_max = False
        self._montage_border_color = "Gray"
        self._last_montage_layout: dict[str, Any] = {}
        self.init_ui()

    @staticmethod
    def _qt_widget_alive(widget: Any) -> bool:
        if widget is None:
            return False
        try:
            widget.objectName()
            return True
        except RuntimeError:
            return False

    @classmethod
    def _safe_spin_value(cls, widget: Any, default: int = 0) -> int:
        if not cls._qt_widget_alive(widget):
            return default
        try:
            return int(widget.value())
        except Exception:
            return default

    @classmethod
    def _safe_combo_text(cls, widget: Any, default: str = "") -> str:
        if not cls._qt_widget_alive(widget):
            return default
        try:
            return str(widget.currentText())
        except RuntimeError:
            return default

    @classmethod
    def _safe_checked(cls, widget: Any, *, default: bool = False) -> bool:
        if not cls._qt_widget_alive(widget):
            return default
        try:
            return bool(widget.isChecked())
        except RuntimeError:
            return default

    @staticmethod
    def _compact_preview_control(widget: QWidget, width: int) -> None:
        """Keep preview-panel controls from expanding across compact cards."""
        widget.setMinimumWidth(width)
        widget.setMaximumWidth(width)
        widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def _montage_use_z_max_projection(self) -> bool:
        combo = getattr(self, "montage_z_projection", None)
        if self._qt_widget_alive(combo):
            use_max = "Max" in self._safe_combo_text(combo, "First Z plane")
            self._montage_z_projection_max = use_max
            return use_max
        return bool(getattr(self, "_montage_z_projection_max", False))

    def _sync_montage_z_projection_mode(self, text: str = "") -> None:
        combo = getattr(self, "montage_z_projection", None)
        if self._qt_widget_alive(combo) and not text:
            text = combo.currentText()
        self._montage_z_projection_max = "Max" in str(text or "")

    def _montage_border_color_name(self) -> str:
        combo = getattr(self, "montage_border_color", None)
        if self._qt_widget_alive(combo):
            self._montage_border_color = self._safe_combo_text(combo, "Gray") or "Gray"
        return str(getattr(self, "_montage_border_color", "Gray") or "Gray")

    def _sync_montage_border_color(self, text: str = "") -> None:
        combo = getattr(self, "montage_border_color", None)
        if self._qt_widget_alive(combo) and not text:
            text = combo.currentText()
        self._montage_border_color = str(text or "Gray")

    def _montage_border_fill(self) -> tuple[str, tuple[int, ...]]:
        name = self._montage_border_color_name().strip().lower()
        if name == "white":
            return "RGB", (255, 255, 255)
        if name == "black":
            return "RGB", (0, 0, 0)
        if name == "transparent":
            return "RGBA", (0, 0, 0, 0)
        return "RGB", (150, 150, 150)

    def _montage_export_mode(self) -> str:
        combo = getattr(self, "montage_export_mode", None)
        return self._safe_combo_text(combo, "Full montage") or "Full montage"

    def _montage_export_is_merged_only(self) -> bool:
        return "merged" in self._montage_export_mode().lower()

    def _montage_label_settings(self) -> dict[str, Any]:
        show = self._safe_checked(getattr(self, "montage_show_labels", None), default=False)
        position = self._safe_combo_text(getattr(self, "montage_label_position", None), "Below") or "Below"
        color_name = self._safe_combo_text(getattr(self, "montage_label_color", None), "Black") or "Black"
        font_size = self._safe_spin_value(getattr(self, "montage_label_font_size", None), 12)
        return {
            "show": show,
            "position": position,
            "color_name": color_name,
            "font_size": max(8, font_size),
        }

    @staticmethod
    def _montage_label_fill(color_name: str, *, rgba: bool = False) -> tuple[int, ...]:
        name = str(color_name or "Black").strip().lower()
        if name == "white":
            rgb = (255, 255, 255)
        else:
            rgb = (0, 0, 0)
        return (*rgb, 255) if rgba else rgb

    @staticmethod
    def _montage_short_label(text: str) -> str:
        stem = str(text or "").strip()
        if stem.endswith(" Merged"):
            return "Merged"
        if " " in stem:
            return stem.split(" ", 1)[-1][:32]
        return stem[:32] + ("…" if len(stem) > 32 else "")

    @staticmethod
    def _nice_scalebar_length_um(pixel_size: float, tile_width_px: int, *, target_fraction: float = 0.2) -> float:
        if pixel_size <= 0 or tile_width_px <= 0:
            return 10.0
        target_um = float(tile_width_px) * float(pixel_size) * float(target_fraction)
        if target_um <= 0:
            return 1.0
        exp = math.floor(math.log10(target_um))
        base = 10.0 ** exp
        for mult in (1.0, 2.0, 5.0, 10.0):
            candidate = mult * base
            if candidate >= target_um * 0.85:
                return float(candidate)
        return float(10.0 * base)

    def _auto_montage_scalebar_length(self, layout: Optional[dict] = None) -> None:
        params = self._montage_scalebar_params()
        pixel_size = float(params.get("pixel_size") or 0.108)
        target_rect = self._montage_scalebar_target_rect(layout)
        tile_w = int(target_rect[2]) if target_rect is not None else 512
        length_um = self._nice_scalebar_length_um(pixel_size, tile_w)
        widget = getattr(self, "montage_sb_length_um", None)
        if self._qt_widget_alive(widget):
            widget.setValue(length_um)

    def _montage_export_extension(self, fmt_text: str) -> str:
        return {
            "PNG": ".png",
            "TIFF": ".tif",
            "JPEG": ".jpg",
        }.get(str(fmt_text or "PNG"), ".png")

    def _montage_export_dpi(self, dlg: Optional[Any] = None) -> int:
        combo = getattr(dlg, "_export_dpi_combo", None) if dlg is not None else getattr(self, "montage_export_dpi", None)
        text = self._safe_combo_text(combo, "300 DPI") or "300 DPI"
        try:
            return int(str(text).split()[0])
        except Exception:
            return 300

    def _montage_export_format(self, dlg: Optional[Any] = None) -> str:
        combo = getattr(dlg, "_export_format_combo", None) if dlg is not None else getattr(self, "montage_export_format", None)
        return self._safe_combo_text(combo, "PNG") or "PNG"

    def _save_montage_array_to_path(self, arr: np.ndarray, out_path: Path, *, dpi: int = 300) -> None:
        from PIL import Image

        img = Image.fromarray(arr)
        suffix = out_path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.getchannel("A"))
                img = background
            img.convert("RGB").save(out_path, quality=95, subsampling=0, dpi=(dpi, dpi))
        elif suffix in (".png", ".tif", ".tiff"):
            img.save(out_path, dpi=(dpi, dpi))
        else:
            img.save(out_path)

    def _default_montage_output_stem(self, paths: Optional[List[Path]] = None, *, merged_only: bool = False) -> str:
        """Default export stem: source filename/base name plus _montage or _merged."""
        suffix = "_merged" if merged_only else "_montage"
        paths = list(paths or self._get_montage_image_paths() or [])
        if len(paths) == 1:
            return f"{paths[0].stem}{suffix}"
        inferred_bases = []
        for p in paths:
            inferred = self._infer_channel_from_stem(p.stem)
            if inferred and inferred[0]:
                inferred_bases.append(inferred[0].strip("_"))
        unique_bases = {b for b in inferred_bases if b}
        if len(unique_bases) == 1:
            return f"{next(iter(unique_bases))}{suffix}"
        if paths:
            parent_name = paths[0].parent.name or paths[0].stem
            return f"{parent_name}{suffix}"
        path_widget = getattr(self, "montage_path", None)
        raw = path_widget.text().strip() if self._qt_widget_alive(path_widget) else ""
        if raw:
            p = Path(raw.split("|")[0].strip())
            return f"{p.stem or p.name or 'montage'}{suffix}"
        return "merged" if merged_only else "montage"

    def _capture_montage_preset_state(self, dlg: Optional[Any] = None) -> dict[str, Any]:
        state = dict(self.get_session_params())
        state["montage_tile_color_overrides"] = {
            str(k): [float(v[0]), float(v[1]), float(v[2])]
            for k, v in (getattr(self, "montage_tile_color_overrides", None) or {}).items()
        }
        state["montage_export_mode"] = self._montage_export_mode()
        if dlg is not None and self._qt_widget_alive(getattr(dlg, "_export_format_combo", None)):
            state["montage_export_format"] = dlg._export_format_combo.currentText()
        else:
            state["montage_export_format"] = self._montage_export_format(dlg)
        if dlg is not None and self._qt_widget_alive(getattr(dlg, "_export_dpi_combo", None)):
            state["montage_export_dpi"] = dlg._export_dpi_combo.currentText()
        else:
            state["montage_export_dpi"] = f"{self._montage_export_dpi(dlg)} DPI"
        label_settings = self._montage_label_settings()
        state.update(
            {
                "montage_show_labels": label_settings["show"],
                "montage_label_position": label_settings["position"],
                "montage_label_color": label_settings["color_name"],
                "montage_label_font_size": label_settings["font_size"],
            }
        )
        if dlg is not None:
            per_tile_lut = getattr(dlg, "_per_tile_lut", None) or {}
            serial_lut: dict[str, list[float]] = {}
            for key, value in per_tile_lut.items():
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    serial_lut[str(key)] = [float(value[0]), float(value[1])]
            state["montage_per_tile_lut"] = serial_lut
            focus_roi = getattr(dlg, "_focus_roi", None)
            if focus_roi is not None and len(focus_roi) == 4:
                state["montage_focus_roi"] = [int(v) for v in focus_roi]
        return state

    def _apply_montage_preset_state(self, data: dict[str, Any], dlg: Optional[Any] = None) -> None:
        self.set_session_params(data)
        colors = data.get("montage_tile_color_overrides")
        if isinstance(colors, dict):
            self.montage_tile_color_overrides = {}
            for key, value in colors.items():
                try:
                    if isinstance(value, (list, tuple)) and len(value) == 3:
                        self.montage_tile_color_overrides[str(key)] = (
                            float(value[0]),
                            float(value[1]),
                            float(value[2]),
                        )
                except Exception:
                    continue
        export_mode = getattr(self, "montage_export_mode", None)
        if self._qt_widget_alive(export_mode) and "montage_export_mode" in data:
            idx = export_mode.findText(str(data["montage_export_mode"]))
            if idx >= 0:
                export_mode.setCurrentIndex(idx)
        export_format = getattr(dlg, "_export_format_combo", None) if dlg is not None else getattr(self, "montage_export_format", None)
        if self._qt_widget_alive(export_format) and "montage_export_format" in data:
            idx = export_format.findText(str(data["montage_export_format"]))
            if idx >= 0:
                export_format.setCurrentIndex(idx)
        export_dpi = getattr(dlg, "_export_dpi_combo", None) if dlg is not None else getattr(self, "montage_export_dpi", None)
        if self._qt_widget_alive(export_dpi) and "montage_export_dpi" in data:
            idx = export_dpi.findText(str(data["montage_export_dpi"]))
            if idx >= 0:
                export_dpi.setCurrentIndex(idx)
        if dlg is not None and isinstance(data.get("montage_per_tile_lut"), dict):
            dlg._per_tile_lut = {}
            for key, value in data["montage_per_tile_lut"].items():
                try:
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        dlg._per_tile_lut[str(key)] = (float(value[0]), float(value[1]))
                except Exception:
                    continue
        if dlg is not None and isinstance(data.get("montage_focus_roi"), (list, tuple)) and len(data["montage_focus_roi"]) == 4:
            dlg._focus_roi = tuple(int(v) for v in data["montage_focus_roi"])

    @staticmethod
    def _montage_figure_preset_tooltips() -> dict[str, str]:
        """Hover text for quick figure-style buttons (does not change tile LUT/order)."""
        return {
            "Nature": (
                "Journal-style export for print.\n"
                "• TIFF, 600 DPI\n"
                "• White background, 4 px spacing\n"
                "• Scale bar on (white, 4 px thick, 16 pt label)\n"
                "• Tile labels on (black, 12 pt)\n"
                "• Focus inset: white box, 5 px line"
            ),
            "Presentation": (
                "Slides / screen viewing.\n"
                "• PNG, 300 DPI\n"
                "• Black background, 8 px spacing\n"
                "• Scale bar on (white, thicker 6 px, 24 pt label)\n"
                "• Tile labels on (white, 16 pt)\n"
                "• Does not change your LUTs or tile order"
            ),
            "Supplement": (
                "Minimal supplementary figure.\n"
                "• PNG, 300 DPI\n"
                "• Transparent background, 2 px spacing\n"
                "• No scale bar, no tile labels\n"
                "• Good for overlaying in other figures"
            ),
        }

    def _apply_montage_figure_preset(self, name: str, dlg: Optional[Any] = None) -> None:
        presets = {
            "Nature": {
                "montage_export_format": "TIFF",
                "montage_export_dpi": "600 DPI",
                "montage_sb_enable": True,
                "montage_sb_thickness": 4,
                "montage_sb_font_size": 16,
                "montage_sb_color": "White",
                "montage_sb_show_label": True,
                "montage_border": 0,
                "montage_border_color": "White",
                "montage_spacing": 4,
                "montage_show_labels": True,
                "montage_label_font_size": 12,
                "montage_label_color": "Black",
                "montage_focus_line_width": 5,
                "montage_focus_color": "White",
            },
            "Presentation": {
                "montage_export_format": "PNG",
                "montage_export_dpi": "300 DPI",
                "montage_sb_enable": True,
                "montage_sb_thickness": 6,
                "montage_sb_font_size": 24,
                "montage_sb_color": "White",
                "montage_border": 0,
                "montage_border_color": "Black",
                "montage_spacing": 8,
                "montage_show_labels": True,
                "montage_label_font_size": 16,
                "montage_label_color": "White",
            },
            "Supplement": {
                "montage_export_format": "PNG",
                "montage_export_dpi": "300 DPI",
                "montage_sb_enable": False,
                "montage_border": 0,
                "montage_border_color": "Transparent",
                "montage_spacing": 2,
                "montage_show_labels": False,
            },
        }
        payload = presets.get(str(name or "").strip())
        if payload is None:
            return
        self._apply_montage_preset_state(payload, dlg)

    def _map_focus_roi_to_export_layout(
        self,
        roi: Optional[tuple[int, int, int, int]],
        *,
        source_layout: Optional[dict],
        export_layout: Optional[dict],
    ) -> Optional[tuple[int, int, int, int]]:
        """Map a focus ROI from the full montage layout onto a merged-only export layout."""
        if roi is None or not source_layout or not export_layout:
            return roi
        source_labels = list(source_layout.get("tile_labels", []) or [])
        source_boxes = list(source_layout.get("tile_boxes", []) or [])
        export_boxes = list(export_layout.get("tile_boxes", []) or [])
        if not source_boxes or not export_boxes:
            return roi
        merged_source = [
            (i, source_boxes[i])
            for i, label in enumerate(source_labels)
            if i < len(source_boxes) and str(label).endswith(" Merged")
        ]
        if not merged_source:
            return roi

        rx0, ry0, rx1, ry1 = [int(v) for v in roi]
        cx = (rx0 + rx1) / 2.0
        cy = (ry0 + ry1) / 2.0

        def score(box: tuple[int, int, int, int]) -> float:
            bx, by, bw, bh = [int(v) for v in box]
            ix0 = max(rx0, bx)
            iy0 = max(ry0, by)
            ix1 = min(rx1, bx + bw)
            iy1 = min(ry1, by + bh)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            contains_center = bx <= cx <= bx + bw and by <= cy <= by + bh
            return float(inter) + (1e12 if contains_center else 0.0)

        source_rank, source_box = max(merged_source, key=lambda item: score(item[1]))
        merged_order = [i for i, _box in merged_source]
        export_idx = merged_order.index(source_rank) if source_rank in merged_order else 0
        export_idx = max(0, min(export_idx, len(export_boxes) - 1))

        sx, sy, sw, sh = [int(v) for v in source_box]
        ex, ey, ew, eh = [int(v) for v in export_boxes[export_idx]]
        if sw <= 0 or sh <= 0 or ew <= 0 or eh <= 0:
            return roi

        mapped = (
            int(round(ex + (rx0 - sx) * ew / sw)),
            int(round(ey + (ry0 - sy) * eh / sh)),
            int(round(ex + (rx1 - sx) * ew / sw)),
            int(round(ey + (ry1 - sy) * eh / sh)),
        )
        width = int(export_layout.get("out_w", 0) or 0)
        height = int(export_layout.get("out_h", 0) or 0)
        if width <= 0 or height <= 0:
            return mapped
        return self._normalize_montage_roi(*mapped, width, height)

    def _compose_montage_for_export(
        self,
        *,
        per_tile_lut: Optional[Dict[Any, Tuple[float, float]]] = None,
        per_tile_color_override: Optional[Dict[Any, Tuple[float, float, float]]] = None,
        per_tile_percentile_override: Optional[Dict[Any, Tuple[float, float]]] = None,
        focus_roi: Optional[tuple[int, int, int, int]] = None,
        merged_only: bool = False,
        source_layout: Optional[dict] = None,
    ) -> Optional[tuple[np.ndarray, dict]]:
        result = self._build_montage_array(
            per_tile_lut=per_tile_lut,
            per_tile_color_override=per_tile_color_override or getattr(self, "montage_tile_color_overrides", None) or {},
            per_tile_percentile_override=per_tile_percentile_override or getattr(self, "montage_tile_percentile_overrides", None) or {},
            return_layout=True,
            merged_only=merged_only,
        )
        if result is None:
            return None
        out, export_layout = result
        displayed = self._apply_montage_scalebar_if_enabled(out, layout=export_layout)
        export_focus_roi = (
            self._map_focus_roi_to_export_layout(
                focus_roi,
                source_layout=source_layout,
                export_layout=export_layout,
            )
            if merged_only
            else focus_roi
        )
        displayed = self._apply_montage_focus_inset_if_enabled(displayed, roi=export_focus_roi)
        return displayed, export_layout

    def _montage_batch_signature(self, path: Path) -> Optional[dict[str, Any]]:
        try:
            if path.suffix.lower() == ".nd2":
                meta = read_metadata_from_path(path)
                channel_count = int(meta.get("channel_count") or 1)
                channel_names = [str(x) for x in (meta.get("channel_names") or [])[:channel_count]]
                return {"channel_count": channel_count, "channel_names": channel_names, "has_merged": channel_count >= 2}
            datasets = self._get_montage_datasets()
            if not datasets:
                return None
            # Lightweight peek: count expected tiles from one dataset pattern.
            return {"channel_count": None, "channel_names": [], "has_merged": None}
        except Exception:
            return None

    def _summarize_montage_batch_export(
        self,
        selected_files: List[Path],
        *,
        export_mode: str,
        export_format: str,
        export_dpi: int,
    ) -> tuple[str, list[str]]:
        warnings: list[str] = []
        current_labels: list[str] = []
        preview_dlg = getattr(self, "_active_montage_preview", None) or getattr(self, "_embedded_post_preview", None)
        if preview_dlg is not None and hasattr(preview_dlg, "_layout"):
            current_labels = list(getattr(preview_dlg, "_layout", {}).get("tile_labels", []) or [])
        else:
            current_labels = list(getattr(self, "_last_montage_layout", {}).get("tile_labels", []) or [])
        current_merged = sum(1 for lb in current_labels if str(lb).endswith(" Merged"))
        if "merged" in export_mode.lower() and current_merged == 0:
            warnings.append("Current preview has no merged tile; merged-only export may fail for some files.")
        lines = [
            f"Export {len(selected_files)} file(s)",
            f"Mode: {export_mode}",
            f"Format: {export_format} at {export_dpi} DPI",
            f"Output suffix: {'_merged' if 'merged' in export_mode.lower() else '_montage'}",
        ]
        if warnings:
            lines.append("")
            lines.extend(warnings)
        return "\n".join(lines), warnings

    @staticmethod
    def _montage_tile_key(label: str) -> str:
        return str(label or "").strip()

    def _montage_tile_included(self, tile_key: str) -> bool:
        return tile_key not in getattr(self, "montage_excluded_tile_keys", set())

    def _montage_z_max_for_tile(self, tile_key: str) -> bool:
        mode = str(self.montage_tile_z_modes.get(tile_key, "global")).lower()
        if mode == "first":
            return False
        if mode == "max":
            return True
        return self._montage_use_z_max_projection()

    def _montage_lookup_color_override(
        self, tile_key: str, tile_idx: int, default: Optional[Tuple[float, float, float]]
    ) -> Optional[Tuple[float, float, float]]:
        overrides = getattr(self, "montage_tile_color_overrides", None) or {}
        if tile_key in overrides:
            return overrides[tile_key]
        if tile_idx in overrides:
            return overrides[tile_idx]
        return default

    def _montage_lookup_color_override_for_meta(
        self,
        tile_key: str,
        tile_idx: int,
        meta: Optional[dict],
        default: Optional[Tuple[float, float, float]],
        overrides: Optional[Dict[Any, Tuple[float, float, float]]] = None,
    ) -> Optional[Tuple[float, float, float]]:
        overrides = overrides if overrides is not None else (getattr(self, "montage_tile_color_overrides", None) or {})
        if tile_key in overrides:
            return overrides[tile_key]
        if tile_idx in overrides:
            return overrides[tile_idx]
        channel_norm = (meta or {}).get("channel_norm")
        if channel_norm and channel_norm in overrides:
            return overrides[channel_norm]
        return default

    def _montage_lookup_percentile_override(
        self, tile_key: str, tile_idx: int
    ) -> Optional[Tuple[float, float]]:
        overrides = getattr(self, "montage_tile_percentile_overrides", None) or {}
        if tile_key in overrides:
            return overrides[tile_key]
        if tile_idx in overrides:
            return overrides[tile_idx]
        return None

    def _montage_lookup_percentile_override_for_meta(
        self,
        tile_key: str,
        tile_idx: int,
        meta: Optional[dict],
        overrides: Optional[Dict[Any, Tuple[float, float]]] = None,
    ) -> Optional[Tuple[float, float]]:
        overrides = overrides if overrides is not None else (getattr(self, "montage_tile_percentile_overrides", None) or {})
        if tile_key in overrides:
            return overrides[tile_key]
        if tile_idx in overrides:
            return overrides[tile_idx]
        channel_norm = (meta or {}).get("channel_norm")
        if channel_norm and channel_norm in overrides:
            return overrides[channel_norm]
        return None

    def _montage_lookup_lut(
        self, per_tile_lut: Optional[Dict[Any, Tuple[float, float]]], tile_key: str, tile_idx: int
    ) -> Optional[Tuple[float, float]]:
        if not per_tile_lut:
            return None
        if tile_key in per_tile_lut:
            return per_tile_lut[tile_key]
        if tile_idx in per_tile_lut:
            return per_tile_lut[tile_idx]
        return None

    def _montage_lookup_lut_for_meta(
        self,
        per_tile_lut: Optional[Dict[Any, Tuple[float, float]]],
        tile_key: str,
        tile_idx: int,
        meta: Optional[dict],
    ) -> Optional[Tuple[float, float]]:
        lut_pair = self._montage_lookup_lut(per_tile_lut, tile_key, tile_idx)
        if lut_pair is not None:
            return lut_pair
        channel_norm = (meta or {}).get("channel_norm")
        if per_tile_lut and channel_norm in per_tile_lut:
            return per_tile_lut[channel_norm]
        return None

    def _try_update_montage_pixel_size_from_paths(self) -> None:
        """Read pixel size from the first montage input file for embedded scale bar."""
        if not getattr(self, "montage_sb_pixel_size", None):
            return
        paths = self._get_montage_image_paths()
        if not paths:
            return
        try:
            ps, _ = read_calibration_from_path(paths[0])
            if ps > 0:
                self.montage_sb_pixel_size.setValue(ps)
                if getattr(self, "montage_sb_pixel_size_source", None) is not None:
                    self.montage_sb_pixel_size_source.setText(f"(from {paths[0].name})")
        except Exception:
            pass

    def attach_module_preview_mount(self, mount: Optional[QWidget]) -> None:
        """Wire the module-shell right pane for post-processing previews."""
        self._module_preview_mount = mount
        self._preview_surface = None
        if mount is not None:
            parent = mount.parent()
            if isinstance(parent, PreviewSurface):
                self._preview_surface = parent
                parent.set_preview_context("post_processing")
                parent.set_pick_callback(self._on_pick_post_preview_file)
        self.refresh_post_candidate_list()

    def _on_pick_post_preview_file(self, path: str) -> None:
        try:
            mode = str(getattr(self, "_post_preview_mode", "montage"))
            if mode == "video":
                self.video_path.setText(str(Path(path)))
            else:
                self.montage_path.setText(str(Path(path)))
        except Exception:
            pass

    def _set_post_preview_mode(self, mode: str) -> None:
        mode = str(mode or "montage")
        if mode not in {"montage", "video"}:
            mode = "montage"
        self._post_preview_mode = mode
        surf = self._preview_surface
        if surf is not None:
            if mode == "video":
                surf.set_preview_context("post_processing_video")
            else:
                surf.set_preview_context("post_processing")
        self.refresh_post_candidate_list()

    def _on_post_section_toggled(self, section: str, expanded: bool) -> None:
        if not expanded:
            return
        sections = {
            "montage": getattr(self, "montage_section", None),
            "video": getattr(self, "video_montage_section", None),
        }
        for key, widget in sections.items():
            if key != section and widget is not None:
                try:
                    widget.set_expanded(False)
                except Exception:
                    pass
        if str(getattr(self, "_post_preview_mode", "montage")) != section:
            self._close_embedded_post_preview()
        self._set_post_preview_mode(section)

    def refresh_post_candidate_list(self, *, selected: Optional[str] = None) -> None:
        """Populate the right-hand picker from the active post-processing section."""
        surf = self._preview_surface
        if surf is None:
            return
        try:
            mode = str(getattr(self, "_post_preview_mode", "montage"))
            if mode == "video":
                raw = (self.video_path.text() or "").strip() if hasattr(self, "video_path") else ""
                video_paths = self._parse_growth_curve_video_paths(raw)
                paths = [str(p.resolve()) for p in video_paths if p.exists()]
            else:
                paths = [str(p.resolve()) for p in self._get_montage_image_paths()]
        except Exception:
            paths = []
        surf.set_candidate_files(
            paths,
            selected=selected,
            advance_placeholder=bool(getattr(self, "_embedded_post_preview", None) is not None),
        )

    def _reparent_video_montage_controls(self, parent: Optional[QWidget] = None) -> None:
        """Keep shared spinboxes/buttons alive when the embedded video preview closes."""
        host = parent or self
        for widget in (
            self.time_interval,
            self.yaxis_min,
            self.yaxis_max,
            self.fps,
            self.format_all,
            self.format_tif,
            self.format_avi,
            self.format_gif,
            self.format_mp4,
            self.video_mask_contours,
            self.video_mask_path,
            self.video_foci_overlay,
            self.video_foci_path,
            self.process_btn,
        ):
            if self._qt_widget_alive(widget):
                widget.setParent(host)

    def _close_embedded_post_preview(self) -> None:
        prev = getattr(self, "_embedded_post_preview", None)
        if prev is not None:
            if prev.objectName() == "EmbeddedVideoMontagePreview":
                self._reparent_video_montage_controls(self)
            try:
                prev.close()
                prev.deleteLater()
            except Exception:
                pass
        self._embedded_post_preview = None
        self._active_montage_preview = None
        self._video_preview_cache = None
        surf = self._preview_surface
        if surf is not None:
            try:
                surf.clear_mount()
            except Exception:
                pass

    def init_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setMinimumWidth(0)
        scroll_layout = QVBoxLayout()
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        scroll_layout.setSpacing(12)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._init_video_montage_widgets()

        # Order: Image montage → Video montage (scale bar lives in image montage preview)
        montage_group = self._build_montage_group()
        self.montage_section = CollapsibleSection("Image montage", montage_group, parent=self, expanded=True)
        scroll_layout.addWidget(self.montage_section)

        growth_curve_group = self._build_growth_curve_video_group()
        self.video_montage_section = CollapsibleSection("Video montage", growth_curve_group, parent=self, expanded=False)
        scroll_layout.addWidget(self.video_montage_section)

        self.video_path.textChanged.connect(self._update_button_state)
        self.excel_path.textChanged.connect(self._update_button_state)
        self.excel_path.textChanged.connect(lambda _text="": self._schedule_video_preview_refresh())
        self.video_path.textChanged.connect(lambda _text="": self.refresh_post_candidate_list())
        self.video_path.textChanged.connect(lambda _text="": self._invalidate_video_preview_cache())

        self.montage_section.toggled.connect(lambda expanded: self._on_post_section_toggled("montage", expanded))
        self.video_montage_section.toggled.connect(lambda expanded: self._on_post_section_toggled("video", expanded))

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        self.setLayout(layout)
        self._update_button_state()

    def _build_video_input_group(self) -> QWidget:
        """Build the video input (TIFF sequence) group."""
        video_group = DragDropGroupBox(
            "Video Input (TIFF Sequence)",
            parent=self,
            on_drop_callback=self.handle_dropped_video_path
        )
        video_group.setMinimumWidth(0)
        video_layout = QVBoxLayout()
        self.video_path = QLineEdit()
        self.video_path.setPlaceholderText("TIFF file, folder, or channel stacks (|)…")
        self.video_path.setMinimumWidth(0)
        file_font = QFont()
        file_font.setPointSize(11)
        self.video_path.setFont(file_font)
        file_btn = QPushButton("Browse File...")
        file_btn.clicked.connect(self.select_video_file)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.clicked.connect(self.select_video_folder)
        channels_btn = QPushButton("Channels…")
        channels_btn.setToolTip(
            "Select one TIFF stack per fluorescence channel (same frame count and image size). "
            "Channels are merged with color lookup tables (green, magenta, cyan, …)."
        )
        channels_btn.clicked.connect(self.select_video_multi_channel)
        autofill_btn = QPushButton("Auto-fill")
        autofill_btn.setToolTip(
            "Auto-detect phase/FI channels, analysis workbook, masks, and foci files "
            "from the selected dataset folder."
        )
        autofill_btn.clicked.connect(self._auto_fill_video_inputs)
        for btn in (file_btn, folder_btn):
            style_compact_browse_button(btn)
        style_compact_action_button(channels_btn, horizontal_padding=18)
        style_compact_action_button(autofill_btn, horizontal_padding=16)

        input_block = QWidget(self)
        input_block.setMinimumWidth(0)
        input_block.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        input_block_layout = QVBoxLayout(input_block)
        input_block_layout.setContentsMargins(0, 0, 0, 0)
        input_block_layout.setSpacing(4)

        input_path_row = QHBoxLayout()
        input_path_row.setSpacing(4)
        input_label = QLabel("Input:")
        input_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        input_path_row.addWidget(input_label, 0)
        input_path_row.addWidget(
            create_info_icon(
                "Video input options:\n\n"
                "• Single TIFF: grayscale time series, RGB (T,H,W,3), or multi-channel (TCYX / T,C,H,W). "
                "Multiple channels are composited with additive color LUTs (green, magenta, cyan, yellow, …).\n\n"
                "• Folder: sorted *.tif sequence (grayscale).\n\n"
                "• Channels: pick several single-channel TIFF stacks (one file per channel); "
                "paths are joined with |. Same T×H×W required.\n\n"
                "Tip: You can also paste multiple paths separated by | or one path per line.",
                parent_widget=self,
            ),
            0,
        )
        input_path_row.addWidget(self.video_path, 1)
        input_block_layout.addLayout(input_path_row)

        input_button_panel = ResponsivePanel(self, breakpoint=520)
        input_button_panel.set_defer_reflow(True)
        input_button_panel.add_widget(file_btn)
        input_button_panel.add_widget(folder_btn)
        input_button_panel.add_widget(channels_btn)
        input_button_panel.add_widget(autofill_btn)
        input_button_panel.set_defer_reflow(False)
        input_button_panel.reflow()
        input_block_layout.addWidget(input_button_panel)
        video_layout.addWidget(input_block)
        video_group.setLayout(video_layout)
        return video_group

    def _init_video_montage_widgets(self) -> None:
        """Create video montage parameter widgets once (mounted in the right preview pane)."""
        if self._video_montage_widgets_ready:
            return
        self._video_montage_widgets_ready = True

        self.time_interval = QDoubleSpinBox()
        self.time_interval.setMinimum(0.0)
        self.time_interval.setMaximum(10000.0)
        self.time_interval.setValue(10.0)
        self.time_interval.setDecimals(2)
        self.time_interval.setSuffix(" min")
        self.time_interval.setFixedWidth(100)

        self.yaxis_min = QDoubleSpinBox()
        self.yaxis_min.setMinimum(0.0)
        self.yaxis_min.setMaximum(100.0)
        self.yaxis_min.setValue(0.0)
        self.yaxis_min.setDecimals(2)
        self.yaxis_min.setSpecialValueText("Auto")
        self._compact_preview_control(self.yaxis_min, 90)

        self.yaxis_max = QDoubleSpinBox()
        self.yaxis_max.setMinimum(0.0)
        self.yaxis_max.setMaximum(100.0)
        self.yaxis_max.setValue(0.0)
        self.yaxis_max.setDecimals(2)
        self.yaxis_max.setSpecialValueText("Auto")
        self._compact_preview_control(self.yaxis_max, 90)

        self.fps = QDoubleSpinBox()
        self.fps.setMinimum(0.1)
        self.fps.setMaximum(60.0)
        self.fps.setValue(5.0)
        self.fps.setDecimals(1)
        self.fps.setSuffix(" fps")
        self._compact_preview_control(self.fps, 95)

        self.format_all = QCheckBox("Select All")
        self.format_all.stateChanged.connect(self._on_select_all_changed)
        self.format_tif = QCheckBox("TIFF")
        self.format_tif.setChecked(True)
        self.format_avi = QCheckBox("AVI")
        self.format_avi.setChecked(True)
        self.format_gif = QCheckBox("GIF")
        self.format_mp4 = QCheckBox("MP4")
        self.format_tif.stateChanged.connect(self._on_format_changed)
        self.format_avi.stateChanged.connect(self._on_format_changed)
        self.format_gif.stateChanged.connect(self._on_format_changed)
        self.format_mp4.stateChanged.connect(self._on_format_changed)

        self.video_mask_contours = QCheckBox("Mask contours")
        self.video_mask_contours.setToolTip("Draw segmentation masks as thin contours over the real video frames.")
        self.video_mask_path = QLineEdit()
        self.video_mask_path.setPlaceholderText("Optional mask TIFF for contour overlay")
        self.video_mask_path.setMinimumWidth(0)

        self.video_foci_overlay = QCheckBox("Foci markers")
        self.video_foci_overlay.setToolTip("Draw foci markers from a CSV/XLSX with frame, x, and y columns.")
        self.video_foci_path = QLineEdit()
        self.video_foci_path.setPlaceholderText("Optional foci CSV/XLSX with frame, x, y columns")
        self.video_foci_path.setMinimumWidth(0)

        self.process_btn = QPushButton("Create video montage")
        style_module_primary_button(self.process_btn)
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.create_video)

        self.time_interval.valueChanged.connect(self._invalidate_video_preview_cache)
        self.time_interval.valueChanged.connect(self._schedule_video_preview_refresh)
        self.yaxis_min.valueChanged.connect(self._schedule_video_preview_refresh)
        self.yaxis_max.valueChanged.connect(self._schedule_video_preview_refresh)
        self.video_mask_contours.toggled.connect(self._invalidate_video_preview_cache)
        self.video_mask_contours.toggled.connect(self._schedule_video_preview_refresh)
        self.video_mask_path.textChanged.connect(lambda _text="": self._invalidate_video_preview_cache())
        self.video_mask_path.textChanged.connect(lambda _text="": self._schedule_video_preview_refresh())
        self.video_foci_overlay.toggled.connect(self._invalidate_video_preview_cache)
        self.video_foci_overlay.toggled.connect(self._schedule_video_preview_refresh)
        self.video_foci_path.textChanged.connect(lambda _text="": self._invalidate_video_preview_cache())
        self.video_foci_path.textChanged.connect(lambda _text="": self._schedule_video_preview_refresh())

    def _build_excel_section(self) -> QWidget:
        """Build Excel analysis file input for video montage."""
        file_font = QFont()
        file_font.setPointSize(11)
        path_field_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        excel_group = DragDropGroupBox(
            "Excel Analysis File",
            parent=self,
            on_drop_callback=self.handle_dropped_excel_path
        )
        excel_group.setMinimumWidth(0)
        self.excel_path = QLineEdit()
        self.excel_path.setPlaceholderText("Optional Excel file for growth curve (preview works without it)")
        self.excel_path.setMinimumWidth(0)
        self.excel_path.setFont(file_font)
        self.excel_path.setSizePolicy(path_field_policy)
        excel_btn = QPushButton("Browse File...")
        excel_btn.clicked.connect(self.select_excel_file)
        excel_group_lay = QVBoxLayout(excel_group)
        excel_group_lay.addWidget(
            stacked_path_block(
                path_label="Excel:",
                line_edit=self.excel_path,
                buttons=(excel_btn,),
                info_icon=create_info_icon(
                    "Optional Excel file from the Analyze Masks tab containing growth data.\n"
                    "If omitted, the preview/export shows the real video only without a growth graph.\n\n"
                    "Required columns:\n"
                    "- Frame: Frame number (1, 2, 3, ...)\n"
                    "- Total Cell Area (Relative): Normalized cell area (first frame = 1.0)",
                    parent_widget=self,
                ),
                parent=self,
            )
        )
        excel_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        return excel_group

    def _build_graph_and_video_settings(self, *, connect_live_refresh: bool = False) -> QWidget:
        """Build time, graph, FPS, and output-format controls for the video montage preview pane."""
        settings_container_layout = QVBoxLayout()
        settings_container_layout.setSpacing(8)

        time_group = QGroupBox("Time Settings")
        time_group.setMinimumWidth(0)
        time_layout = QHBoxLayout()
        time_label = QLabel("Time per frame (min.):")
        time_info_icon = create_info_icon(
            "Time difference between consecutive frames.\n\n"
            "This is used to convert frame numbers to time (minutes) for the growth-curve x-axis.\n\n"
            "Example: If images were taken every 10 minutes, enter 10.0",
            parent_widget=self,
        )
        time_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        time_layout.addWidget(time_label)
        time_layout.addWidget(time_info_icon)
        time_layout.addStretch()
        time_layout.addWidget(self.time_interval)
        time_group.setLayout(time_layout)
        settings_container_layout.addWidget(time_group)
        
        # Graph settings
        graph_group = QGroupBox("Graph Settings")
        graph_group.setMinimumWidth(0)
        graph_layout = QVBoxLayout()
        
        # Y-axis min/max
        yaxis_layout = QHBoxLayout()
        ymin_label = QLabel("Min:")
        ymax_label = QLabel("Max:")
        graph_info_icon = create_info_icon(
            "Configure the Y-axis range for the growth curve graph.\n\n"
            "Y-axis represents: Relative cell area (normalized to first frame = 1.0)\n\n"
            "Y-axis min/max:\n"
            "- Set to 'Auto' (0.0) to automatically scale based on data\n"
            "- Set custom values to fix the Y-axis range\n"
            "- Useful for comparing multiple videos with consistent scaling\n\n"
            "Example: Set min=0.0, max=3.0 to show growth from 1x to 3x initial area",
            parent_widget=self
        )
        graph_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        
        yaxis_row = QHBoxLayout()
        yaxis_row.setSpacing(8)
        yaxis_row.addWidget(ymin_label)
        yaxis_row.addWidget(self.yaxis_min)
        yaxis_row.addWidget(ymax_label)
        yaxis_row.addWidget(self.yaxis_max)
        yaxis_row.addWidget(graph_info_icon)
        yaxis_row.addStretch(1)
        graph_layout.addLayout(yaxis_row)
        
        graph_group.setLayout(graph_layout)
        graph_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        settings_container_layout.addWidget(graph_group)
        
        # Video settings
        video_settings_group = QGroupBox("Video Settings")
        video_settings_group.setMinimumWidth(0)
        video_settings_layout = QHBoxLayout()
        fps_label = QLabel("Frames per second (FPS):")
        fps_info_icon = create_info_icon(
            "Playback speed for the output video.\n\n"
            "Higher values = faster playback\n"
            "Lower values = slower playback\n\n"
            "Default: 5.0 fps\n"
            "Typical range: 1-10 fps for time-lapse videos",
            parent_widget=self
        )
        fps_info_icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        video_settings_layout.addWidget(fps_label)
        video_settings_layout.addWidget(self.fps)
        video_settings_layout.addWidget(fps_info_icon)
        video_settings_layout.addStretch()
        video_settings_group.setLayout(video_settings_layout)
        video_settings_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        video_settings_group.setMinimumWidth(0)
        settings_container_layout.addWidget(video_settings_group)

        overlay_group = QGroupBox("Video overlays")
        overlay_group.setMinimumWidth(0)
        overlay_layout = QVBoxLayout()
        overlay_toggle_row = QHBoxLayout()
        overlay_toggle_row.setSpacing(8)
        overlay_toggle_row.addWidget(self.video_mask_contours)
        overlay_toggle_row.addWidget(self.video_foci_overlay)
        overlay_toggle_row.addWidget(
            create_info_icon(
                "FI channels: use the Channels button on the left to choose one TIFF stack per FI channel; "
                "they are merged as real video data with LUT colors.\n\n"
                "Mask contours: draws segmentation masks as outlines only, so the microscopy signal stays visible.\n\n"
                "Foci markers: draws simple rings from CSV/XLSX columns named frame, x, y (or common variants).",
                parent_widget=self,
            )
        )
        mask_btn = QPushButton("Mask…")
        style_compact_action_button(mask_btn, horizontal_padding=14)
        mask_btn.clicked.connect(self._select_video_mask_file)
        foci_btn = QPushButton("Foci…")
        style_compact_action_button(foci_btn, horizontal_padding=14)
        foci_btn.clicked.connect(self._select_video_foci_file)
        overlay_toggle_row.addStretch(1)
        overlay_toggle_row.addWidget(mask_btn)
        overlay_toggle_row.addWidget(foci_btn)
        overlay_layout.addLayout(overlay_toggle_row)

        mask_row = QHBoxLayout()
        mask_row.setSpacing(6)
        mask_label = QLabel("Mask:")
        mask_label.setMinimumWidth(44)
        mask_row.addWidget(mask_label)
        mask_row.addWidget(self.video_mask_path, 1)
        overlay_layout.addLayout(mask_row)

        foci_row = QHBoxLayout()
        foci_row.setSpacing(6)
        foci_label = QLabel("Foci:")
        foci_label.setMinimumWidth(44)
        foci_row.addWidget(foci_label)
        foci_row.addWidget(self.video_foci_path, 1)
        overlay_layout.addLayout(foci_row)
        overlay_group.setLayout(overlay_layout)
        overlay_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        settings_container_layout.addWidget(overlay_group)
        
        # Output format selection (in same row)
        format_group = QGroupBox("Output Formats")
        format_group.setMinimumWidth(0)
        format_layout = QVBoxLayout()
        
        format_layout.addWidget(self.format_all)
        format_checkbox_row = QHBoxLayout()
        format_checkbox_row.setSpacing(12)
        for cb in (self.format_tif, self.format_avi, self.format_gif, self.format_mp4):
            format_checkbox_row.addWidget(cb)
        format_checkbox_row.addStretch(1)
        format_layout.addLayout(format_checkbox_row)
        
        format_group.setLayout(format_layout)
        format_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        settings_container_layout.addWidget(format_group)
        
        settings_container = QWidget()
        settings_container.setLayout(settings_container_layout)
        return settings_container

    def _build_growth_curve_video_group(self) -> QWidget:
        """Build video montage inputs on the left; preview/settings/export live on the right."""
        group = QGroupBox("")
        group.setFlat(True)
        group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        group.setMinimumWidth(0)
        group_layout = QVBoxLayout()
        video_group = self._build_video_input_group()
        group_layout.addWidget(video_group)
        group_layout.addWidget(self._build_excel_section())
        self.preview_video_btn = QPushButton("Preview video montage")
        style_module_primary_button(self.preview_video_btn)
        self.preview_video_btn.setEnabled(False)
        self.preview_video_btn.clicked.connect(self._preview_video_montage)
        group_layout.addWidget(self.preview_video_btn)
        group.setLayout(group_layout)
        return group

    def _build_montage_group(self) -> QWidget:
        """Build the Montage group: one image or folder → Phase first, then FI channels, then merged."""
        group = QGroupBox("")
        group.setFlat(True)
        group.setStyleSheet(FLAT_GROUPBOX_FRAME_STYLESHEET)
        group.setMinimumWidth(0)
        layout = QVBoxLayout()

        # —— Input (own drag-and-drop zone) ———
        self.montage_path = QLineEdit()
        self.montage_path.setPlaceholderText("One ND2 or TIFF file, or a folder of images")
        self.montage_path.setMinimumWidth(0)
        self.montage_path.textChanged.connect(lambda _text="": self.refresh_post_candidate_list())
        folder_btn = QPushButton("Browse folder…")
        folder_btn.clicked.connect(self._select_montage_folder)
        files_btn = QPushButton("Add files…")
        files_btn.clicked.connect(self._select_montage_files)
        layout.addWidget(
            build_path_drop_group(
                self,
                lambda p: self.montage_path.setText(p) if getattr(self, "montage_path", None) else None,
                line_edit=self.montage_path,
                buttons=(folder_btn, files_btn),
                path_label="Input:",
            )
        )

        overlay_group = QGroupBox("Optional overlay: segmented cells")
        overlay_group.setMinimumWidth(0)
        overlay_layout = QVBoxLayout()
        overlay_toggle_row = QHBoxLayout()
        self.montage_overlay_enable = QCheckBox("Enable cell boundary overlay")
        self.montage_overlay_enable.setChecked(False)
        overlay_toggle_row.addWidget(self.montage_overlay_enable)
        overlay_toggle_row.addWidget(create_info_icon(
            "When enabled, cell boundaries from a mask TIFF are drawn on montage tiles (yellow contours).\n"
            "Leave Mask empty to auto-find ``*_masks.tif`` next to the montage input.",
            parent_widget=self,
        ))
        overlay_toggle_row.addStretch()
        overlay_layout.addLayout(overlay_toggle_row)

        self.montage_overlay_mask_path = QLineEdit()
        self.montage_overlay_mask_path.setPlaceholderText("Optional mask TIFF override (auto-find base_masks.tif if empty)")
        self.montage_overlay_mask_path.setMinimumWidth(0)
        overlay_mask_btn = QPushButton("Mask…")
        style_compact_action_button(overlay_mask_btn, horizontal_padding=18)

        def _select_overlay_mask() -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select mask TIFF",
                get_default_browse_dir(),
                "TIFF Files (*.tif *.tiff *.TIF *.TIFF);;All Files (*)",
            )
            if path:
                self.montage_overlay_mask_path.setText(path)

        overlay_mask_btn.clicked.connect(_select_overlay_mask)
        overlay_layout.addWidget(
            build_path_drop_group(
                self,
                lambda p: self.montage_overlay_mask_path.setText(p)
                if getattr(self, "montage_overlay_mask_path", None)
                else None,
                line_edit=self.montage_overlay_mask_path,
                buttons=(overlay_mask_btn,),
                path_label="Mask:",
            )
        )

        overlay_params_row = QHBoxLayout()
        overlay_params_row.addWidget(QLabel("Frame (0-based):"))
        self.montage_overlay_frame = QSpinBox()
        self.montage_overlay_frame.setRange(0, 100000)
        self.montage_overlay_frame.setValue(0)
        overlay_params_row.addWidget(self.montage_overlay_frame)
        overlay_params_row.addStretch()
        overlay_layout.addLayout(overlay_params_row)

        overlay_group.setLayout(overlay_layout)
        layout.addWidget(overlay_group)

        # Layout/export controls live in the preview, where their effect is visible immediately.
        btn_row = QHBoxLayout()
        preview_montage_btn = QPushButton("Preview image montage")
        style_module_primary_button(preview_montage_btn)
        preview_montage_btn.clicked.connect(self._preview_montage)
        btn_row.addWidget(preview_montage_btn, 1)
        layout.addLayout(btn_row)
        group.setLayout(layout)
        return group

    def _build_montage_layout_controls(self, parent: QWidget) -> QGroupBox:
        """Create compact montage layout controls for the preview pane."""
        group = QGroupBox("")
        group.setObjectName("MontageLayoutControls")
        group.setMinimumWidth(0)
        group.setStyleSheet(
            """
            QGroupBox#MontageLayoutControls {
                border: 1px solid #303844;
                border-radius: 8px;
                margin-top: 8px;
                padding: 10px 10px 8px 10px;
                background-color: #232830;
            }
            QGroupBox#MontageLayoutControls::title {
                color: #4a9eff;
                padding: 0 6px;
            }
            """
        )
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)

        self.montage_rows = QSpinBox(parent)
        self.montage_rows.setRange(1, 50)
        self.montage_rows.setValue(1)
        self.montage_rows.setToolTip("Number of rows. Default 1 = single row of tiles.")
        self._compact_preview_control(self.montage_rows, 70)
        self.montage_cols = QSpinBox(parent)
        self.montage_cols.setRange(0, 50)
        self.montage_cols.setValue(0)
        self.montage_cols.setSpecialValueText("Auto")
        self.montage_cols.setToolTip("Auto = derived from number of tiles and rows.")
        self._compact_preview_control(self.montage_cols, 80)

        self.montage_spacing = QSpinBox(parent)
        self.montage_spacing.setRange(0, 100)
        self.montage_spacing.setValue(4)
        self.montage_spacing.setSuffix(" px")
        self._compact_preview_control(self.montage_spacing, 80)
        self.montage_border = QSpinBox(parent)
        self.montage_border.setRange(0, 100)
        self.montage_border.setValue(0)
        self.montage_border.setSuffix(" px")
        self._compact_preview_control(self.montage_border, 80)
        self.montage_border_color = QComboBox(parent)
        self.montage_border_color.addItems(["Gray", "White", "Black", "Transparent"])
        self.montage_border_color.setToolTip("Fill color for montage spacing and outer border. Transparent is preserved for PNG/TIFF exports.")
        self._compact_preview_control(self.montage_border_color, 120)
        saved_border_color = str(getattr(self, "_montage_border_color", "White") or "White")
        border_color_idx = self.montage_border_color.findText(saved_border_color)
        if border_color_idx >= 0:
            self.montage_border_color.setCurrentIndex(border_color_idx)
        self.montage_border_color.currentTextChanged.connect(self._sync_montage_border_color)

        self.montage_order = QComboBox(parent)
        self.montage_order.addItems(["Row-major", "Column-major"])
        self.montage_order.setToolTip("Row-major: left to right, then top to bottom. Column-major: top to bottom, then left to right.")
        self.montage_order.setMinimumWidth(0)
        self._compact_preview_control(self.montage_order, 160)

        self.montage_z_projection = QComboBox(parent)
        self.montage_z_projection.addItems(["First Z", "Max projection"])
        self.montage_z_projection.setToolTip(
            "For z-stacks, choose the first z plane or collapse each channel with a max-intensity projection."
        )
        self.montage_z_projection.setMinimumWidth(0)
        self._compact_preview_control(self.montage_z_projection, 160)
        if getattr(self, "_montage_z_projection_max", False):
            self.montage_z_projection.setCurrentIndex(1)
        self.montage_z_projection.currentTextChanged.connect(self._sync_montage_z_projection_mode)

        row_a = QHBoxLayout()
        row_a.setSpacing(8)
        row_a.addWidget(QLabel("Rows", group))
        row_a.addWidget(self.montage_rows)
        row_a.addWidget(QLabel("Cols", group))
        row_a.addWidget(self.montage_cols)
        row_a.addWidget(QLabel("Spacing", group))
        row_a.addWidget(self.montage_spacing)
        row_a.addWidget(QLabel("Border", group))
        row_a.addWidget(self.montage_border)
        row_a.addWidget(QLabel("Color", group))
        row_a.addWidget(self.montage_border_color)
        row_a.addStretch(1)

        row_b = QHBoxLayout()
        row_b.setSpacing(8)
        row_b.addWidget(QLabel("Order", group))
        row_b.addWidget(self.montage_order)
        row_b.addWidget(QLabel("Z-stack", group))
        row_b.addWidget(self.montage_z_projection)
        row_b.addStretch(1)

        layout.addWidget(vbox_from_rows(row_a, row_b, parent=group))
        return group

    def _build_montage_label_controls(self, parent: QWidget) -> QGroupBox:
        """Per-tile label options for montage export."""
        group = QGroupBox("")
        group.setObjectName("MontageLabelControls")
        group.setStyleSheet(
            """
            QGroupBox#MontageLabelControls {
                border: 1px solid #303844;
                border-radius: 8px;
                margin-top: 8px;
                padding: 10px 10px 8px 10px;
                background-color: #232830;
            }
            QGroupBox#MontageLabelControls::title { color: #4a9eff; padding: 0 6px; }
            """
        )
        outer = QVBoxLayout(group)
        outer.setContentsMargins(8, 12, 8, 8)
        outer.setSpacing(6)

        self.montage_show_labels = QCheckBox("Show tile labels", parent)
        self.montage_show_labels.setToolTip("Draw channel/tile names on the montage (e.g. Phase, GFP, Merged).")

        row_a = QHBoxLayout()
        row_a.setSpacing(8)
        row_a.addWidget(self.montage_show_labels)
        row_a.addWidget(QLabel("Position", group))
        self.montage_label_position = QComboBox(parent)
        self.montage_label_position.addItems(["Below", "Inside top-left", "Inside bottom-left"])
        self._compact_preview_control(self.montage_label_position, 145)
        row_a.addWidget(self.montage_label_position)
        row_a.addWidget(QLabel("Color", group))
        self.montage_label_color = QComboBox(parent)
        self.montage_label_color.addItems(["Black", "White"])
        self._compact_preview_control(self.montage_label_color, 90)
        row_a.addWidget(self.montage_label_color)
        row_a.addWidget(QLabel("Font", group))
        self.montage_label_font_size = QSpinBox(parent)
        self.montage_label_font_size.setRange(8, 48)
        self.montage_label_font_size.setValue(12)
        self._compact_preview_control(self.montage_label_font_size, 70)
        row_a.addWidget(self.montage_label_font_size)
        row_a.addStretch(1)

        outer.addWidget(vbox_from_rows(row_a, parent=group))
        return group

    def _build_montage_scalebar_controls(self, parent: QWidget) -> QGroupBox:
        """Scale bar options inside montage preview."""
        group = QGroupBox("")
        group.setObjectName("MontageScaleBarControls")
        group.setStyleSheet(
            """
            QGroupBox#MontageScaleBarControls {
                border: 1px solid #303844;
                border-radius: 8px;
                margin-top: 8px;
                padding: 10px 10px 8px 10px;
                background-color: #232830;
            }
            QGroupBox#MontageScaleBarControls::title { color: #4a9eff; padding: 0 6px; }
            """
        )
        outer = QVBoxLayout(group)
        outer.setContentsMargins(8, 12, 8, 8)
        outer.setSpacing(6)

        self.montage_sb_enable = QCheckBox("Add scale bar to montage", parent)
        self.montage_sb_enable.setChecked(False)

        row_a = QHBoxLayout()
        row_a.setSpacing(8)
        row_a.addWidget(self.montage_sb_enable)
        row_a.addWidget(QLabel("Pixel size (µm/px)", group))
        self.montage_sb_pixel_size = QDoubleSpinBox(parent)
        self.montage_sb_pixel_size.setRange(0.001, 100.0)
        self.montage_sb_pixel_size.setDecimals(3)
        self.montage_sb_pixel_size.setValue(0.108)
        self._compact_preview_control(self.montage_sb_pixel_size, 90)
        row_a.addWidget(self.montage_sb_pixel_size)
        self.montage_sb_pixel_size_source = QLabel("", parent)
        self.montage_sb_pixel_size_source.setStyleSheet("color: #2ecc71; font-size: 9pt;")
        row_a.addWidget(self.montage_sb_pixel_size_source)
        row_a.addWidget(QLabel("Length (µm)", group))
        self.montage_sb_length_um = QDoubleSpinBox(parent)
        self.montage_sb_length_um.setRange(0.1, 1000.0)
        self.montage_sb_length_um.setDecimals(1)
        self.montage_sb_length_um.setValue(10.0)
        self._compact_preview_control(self.montage_sb_length_um, 90)
        row_a.addWidget(self.montage_sb_length_um)
        auto_length_btn = QPushButton("Auto length", parent)
        style_compact_action_button(auto_length_btn, horizontal_padding=12)
        auto_length_btn.setToolTip("Pick a visually nice scale-bar length from tile width and pixel size.")
        row_a.addWidget(auto_length_btn)
        row_a.addWidget(QLabel("Position", group))
        self.montage_sb_position = QComboBox(parent)
        self.montage_sb_position.addItems(["Bottom-right", "Bottom-left", "Top-right", "Top-left"])
        self._compact_preview_control(self.montage_sb_position, 135)
        row_a.addWidget(self.montage_sb_position)
        row_a.addStretch(1)

        row_b = QHBoxLayout()
        row_b.setSpacing(8)
        self.montage_sb_show_label = QCheckBox("Show label (e.g. 10 µm)", parent)
        self.montage_sb_show_label.setChecked(True)
        row_b.addWidget(self.montage_sb_show_label)
        row_b.addWidget(QLabel("Color", group))
        self.montage_sb_color = QComboBox(parent)
        self.montage_sb_color.addItems(["White", "Black"])
        self._compact_preview_control(self.montage_sb_color, 90)
        row_b.addWidget(self.montage_sb_color)
        row_b.addWidget(QLabel("Thickness", group))
        self.montage_sb_thickness = QSpinBox(parent)
        self.montage_sb_thickness.setRange(1, 20)
        self.montage_sb_thickness.setValue(5)
        self._compact_preview_control(self.montage_sb_thickness, 70)
        row_b.addWidget(self.montage_sb_thickness)
        row_b.addWidget(QLabel("Font", group))
        self.montage_sb_font_size = QSpinBox(parent)
        self.montage_sb_font_size.setRange(8, 72)
        self.montage_sb_font_size.setValue(20)
        self._compact_preview_control(self.montage_sb_font_size, 70)
        row_b.addWidget(self.montage_sb_font_size)
        row_b.addStretch(1)

        outer.addWidget(vbox_from_rows(row_a, row_b, parent=group))
        self._try_update_montage_pixel_size_from_paths()
        auto_length_btn.clicked.connect(lambda *_args: self._auto_montage_scalebar_length(getattr(self, "_last_montage_layout", None)))
        return group

    def _montage_scalebar_params(self) -> dict[str, Any]:
        return {
            "pixel_size": self._safe_value(getattr(self, "montage_sb_pixel_size", None), default=0.108),
            "length_um": self._safe_value(getattr(self, "montage_sb_length_um", None), default=10.0),
            "thickness": self._safe_spin_value(getattr(self, "montage_sb_thickness", None), 5),
            "position": self._safe_combo_text(getattr(self, "montage_sb_position", None), "Bottom-right"),
            "color": self._safe_combo_text(getattr(self, "montage_sb_color", None), "White"),
            "font_size": self._safe_spin_value(getattr(self, "montage_sb_font_size", None), 20),
            "show_label": self._safe_checked(getattr(self, "montage_sb_show_label", None), default=True),
        }

    def _build_montage_focus_controls(self, parent: QWidget) -> QGroupBox:
        """Focus inset options inside montage preview."""
        group = QGroupBox("")
        group.setObjectName("MontageFocusControls")
        group.setStyleSheet(
            """
            QGroupBox#MontageFocusControls {
                border: 1px solid #303844;
                border-radius: 8px;
                margin-top: 8px;
                padding: 10px 10px 8px 10px;
                background-color: #232830;
            }
            QGroupBox#MontageFocusControls::title { color: #4a9eff; padding: 0 6px; }
            """
        )
        outer = QVBoxLayout(group)
        outer.setContentsMargins(8, 12, 8, 8)
        outer.setSpacing(6)

        self.montage_focus_enable = QCheckBox("Add focus inset / zoom box", parent)
        self.montage_focus_enable.setToolTip(
            "Right-drag a box on the preview. The selected region is magnified into the chosen corner and exported with the montage."
        )

        self.montage_focus_zoom = QComboBox(parent)
        self.montage_focus_zoom.addItems(["2x", "4x", "8x", "12x", "16x"])
        self.montage_focus_zoom.setCurrentText("4x")
        self._compact_preview_control(self.montage_focus_zoom, 75)
        self.montage_focus_position = QComboBox(parent)
        self.montage_focus_position.addItems(["Top-right", "Top-left", "Bottom-right", "Bottom-left"])
        self._compact_preview_control(self.montage_focus_position, 135)
        self.montage_focus_color = QComboBox(parent)
        self.montage_focus_color.addItems(["Yellow", "White", "Black", "Cyan", "Magenta", "Green", "Red"])
        self.montage_focus_color.setCurrentText("White")
        self._compact_preview_control(self.montage_focus_color, 110)
        self.montage_focus_line_width = QSpinBox(parent)
        self.montage_focus_line_width.setRange(1, 20)
        self.montage_focus_line_width.setValue(5)
        self._compact_preview_control(self.montage_focus_line_width, 70)
        self.montage_focus_clear_btn = QPushButton("Clear ROI", parent)
        self.montage_focus_clear_btn.setToolTip("Remove the current focus inset selection.")
        style_compact_action_button(self.montage_focus_clear_btn, horizontal_padding=16)

        row_a = QHBoxLayout()
        row_a.setSpacing(8)
        row_a.addWidget(self.montage_focus_enable)
        row_a.addWidget(QLabel("Zoom", group))
        row_a.addWidget(self.montage_focus_zoom)
        row_a.addWidget(QLabel("Position", group))
        row_a.addWidget(self.montage_focus_position)
        row_a.addWidget(QLabel("Box color", group))
        row_a.addWidget(self.montage_focus_color)
        row_a.addWidget(QLabel("Line width", group))
        row_a.addWidget(self.montage_focus_line_width)
        row_a.addWidget(self.montage_focus_clear_btn)
        row_a.addStretch(1)

        outer.addWidget(vbox_from_rows(row_a, parent=group))
        return group

    def _montage_focus_params(self) -> dict[str, Any]:
        return {
            "enabled": self._safe_checked(getattr(self, "montage_focus_enable", None), default=False),
            "zoom": self._safe_combo_text(getattr(self, "montage_focus_zoom", None), "4x"),
            "position": self._safe_combo_text(getattr(self, "montage_focus_position", None), "Top-right"),
            "color": self._safe_combo_text(getattr(self, "montage_focus_color", None), "White"),
            "line_width": self._safe_spin_value(getattr(self, "montage_focus_line_width", None), 5),
        }

    @staticmethod
    def _montage_focus_color_rgb(name: str) -> tuple[int, int, int]:
        colors = {
            "black": (0, 0, 0),
            "cyan": (0, 255, 255),
            "green": (0, 220, 80),
            "magenta": (255, 0, 255),
            "red": (255, 40, 40),
            "white": (255, 255, 255),
            "yellow": (255, 230, 0),
        }
        return colors.get(str(name or "").strip().lower(), (255, 230, 0))

    @classmethod
    def _safe_value(cls, widget: Any, *, default: float = 0.0) -> float:
        if not cls._qt_widget_alive(widget):
            return default
        try:
            return float(widget.value())
        except Exception:
            return default

    def _montage_scalebar_target_rect(self, layout: Optional[dict]) -> Optional[tuple[int, int, int, int]]:
        if not isinstance(layout, dict):
            return None
        boxes = layout.get("tile_boxes") or []
        if not boxes:
            return None
        # Anchor in the bottom-right occupied tile, not in the gray montage background.
        return max(
            ((int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes if len(b) >= 4),
            key=lambda b: (b[1], b[0]),
            default=None,
        )

    def _apply_montage_scalebar_if_enabled(self, img: np.ndarray, *, layout: Optional[dict] = None) -> np.ndarray:
        if not self._safe_checked(getattr(self, "montage_sb_enable", None), default=False):
            return img
        params = self._montage_scalebar_params()
        target_rect = self._montage_scalebar_target_rect(layout)
        if target_rect is not None:
            params["target_rect"] = target_rect
        out = self._draw_scalebar_on_image(img, params=params)
        return out if out is not None else img

    @staticmethod
    def _normalize_montage_roi(
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        width: int,
        height: int,
        *,
        min_size: int = 8,
    ) -> Optional[tuple[int, int, int, int]]:
        left = max(0, min(int(x0), int(x1), max(0, width - 1)))
        right = min(width, max(int(x0), int(x1)))
        top = max(0, min(int(y0), int(y1), max(0, height - 1)))
        bottom = min(height, max(int(y0), int(y1)))
        if right - left < min_size or bottom - top < min_size:
            return None
        return (left, top, right, bottom)

    def _apply_montage_focus_inset_if_enabled(
        self,
        img: np.ndarray,
        *,
        roi: Optional[tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        params = self._montage_focus_params()
        if not params.get("enabled") or roi is None:
            return img
        out = self._draw_montage_focus_inset(img, roi=roi, params=params)
        return out if out is not None else img

    def _draw_montage_focus_inset(
        self,
        img: np.ndarray,
        *,
        roi: tuple[int, int, int, int],
        params: Optional[dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """Draw a source ROI, connector, and magnified inset on a copy of the montage."""
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            QMessageBox.warning(self, "Focus inset", "Pillow (PIL) is required for drawing the focus inset.")
            return None
        if params is None:
            params = self._montage_focus_params()

        h, w = img.shape[:2]
        norm_roi = self._normalize_montage_roi(*roi, w, h)
        if norm_roi is None:
            return img
        x0, y0, x1, y1 = norm_roi
        source_w = x1 - x0
        source_h = y1 - y0
        zoom_text = str(params.get("zoom") or "4x").lower().replace("x", "").strip()
        try:
            zoom = max(1.0, float(zoom_text))
        except ValueError:
            zoom = 4.0
        inset_w = max(20, int(round(source_w * zoom)))
        inset_h = max(20, int(round(source_h * zoom)))
        max_inset_w = max(40, int(w * 0.45))
        max_inset_h = max(40, int(h * 0.45))
        fit_scale = min(1.0, max_inset_w / max(1, inset_w), max_inset_h / max(1, inset_h))
        if fit_scale < 1.0:
            inset_w = max(20, int(round(inset_w * fit_scale)))
            inset_h = max(20, int(round(inset_h * fit_scale)))

        line_width = max(1, int(params.get("line_width") or 3))
        margin = max(12, line_width * 4)
        position = str(params.get("position") or "Top-right").lower()
        if "left" in position:
            ix0 = margin
        else:
            ix0 = max(margin, w - margin - inset_w)
        if "bottom" in position:
            iy0 = max(margin, h - margin - inset_h)
        else:
            iy0 = margin
        ix1 = min(w - margin, ix0 + inset_w)
        iy1 = min(h - margin, iy0 + inset_h)
        ix0 = max(0, ix1 - inset_w)
        iy0 = max(0, iy1 - inset_h)

        pil_img = Image.fromarray(img)
        inset = pil_img.crop((x0, y0, x1, y1)).resize((max(1, ix1 - ix0), max(1, iy1 - iy0)), Image.Resampling.LANCZOS)
        if pil_img.mode == "RGBA" and inset.mode != "RGBA":
            inset = inset.convert("RGBA")
        pil_img.paste(inset, (ix0, iy0), inset if inset.mode == "RGBA" else None)

        draw = ImageDraw.ImageDraw(pil_img)
        color_rgb = self._montage_focus_color_rgb(str(params.get("color") or "Yellow"))
        color = (*color_rgb, 255) if pil_img.mode == "RGBA" else color_rgb
        for offset in range(line_width):
            draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)
            draw.rectangle([ix0 - offset, iy0 - offset, ix1 + offset, iy1 + offset], outline=color)

        # Connect nearest corners so the inset reads like a magnified callout.
        src_anchor = (x1, y0) if ix0 >= x0 else (x0, y0)
        if iy0 > y0:
            src_anchor = (src_anchor[0], y1)
        inset_anchor = (ix0, iy1) if ix0 >= x0 else (ix1, iy1)
        if iy0 > y0:
            inset_anchor = (inset_anchor[0], iy0)
        draw.line([src_anchor, inset_anchor], fill=color, width=line_width)
        return np.array(pil_img)

    def _edit_montage_channel_colors(self) -> None:
        """Legacy: Open dialog for ND2 channel colors. Prefer _edit_montage_tile_colors when in montage preview."""
        paths = self._get_montage_image_paths()
        nd2_path = None
        channel_names: List[str] = []
        for p in paths:
            if p.suffix.lower() == ".nd2":
                meta = read_metadata_from_path(p)
                nc = meta.get("channel_count") or 1
                names = meta.get("channel_names") or [f"Ch{i+1}" for i in range(nc)]
                if nc >= 1:
                    nd2_path = p
                    channel_names = names[:nc]
                    break
        if not nd2_path or not channel_names:
            QMessageBox.information(
                self,
                "Channel colors",
                "No multi-channel ND2 in the current montage paths. Add an ND2 file and try again, or use \"Edit channel colors\" in the montage preview to set color per tile.",
            )
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Channel display colors")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(f"File: {nd2_path.name}\nSet color per channel (Auto = from metadata):"))
        form = QFormLayout()
        combos: List[tuple] = []
        choices = ["Auto", "Grayscale", "Red", "Orange", "Green", "Blue", "Yellow", "Cyan", "Magenta"]
        for i, name in enumerate(channel_names):
            cb = QComboBox()
            cb.addItems(choices)
            key = (nd2_path.stem, i)
            current = self.montage_channel_color_overrides.get(key, "auto")
            idx = next((j for j, c in enumerate(choices) if c.lower() == current), 0)
            cb.setCurrentIndex(idx)
            form.addRow(f"Ch{i+1} ({name}):", cb)
            combos.append((key, cb))
        layout.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        layout.addWidget(bb)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            for key, cb in combos:
                val = cb.currentText().lower()
                if val == "auto":
                    self.montage_channel_color_overrides.pop(key, None)
                else:
                    self.montage_channel_color_overrides[key] = val

    def _edit_montage_tile_colors(
        self, layout: dict, refresh_callback: Optional[Callable[[], None]] = None
    ) -> None:
        """Open dialog to include/exclude tiles, z projection, color, and percentile per tile."""
        tiles = layout.get("all_tiles") or layout.get("tiles", [])
        labels = layout.get("all_tile_labels") or layout.get("tile_labels", []) or [t[1] for t in tiles]
        z_choices = ["Use global", "First Z", "Max intensity"]
        z_to_mode = {"Use global": "global", "First Z": "first", "Max intensity": "max"}
        mode_to_z = {v: k for k, v in z_to_mode.items()}
        preset_names = ["Default", "Grayscale", "Red", "Orange", "Green", "Blue", "Yellow", "Cyan", "Magenta", "Pick…"]
        preset_to_rgb = {
            "Default": None,
            "Grayscale": (1.0, 1.0, 1.0),
            "Red": (204 / 255.0, 0.0, 0.0),
            "Orange": (242 / 255.0, 137 / 255.0, 26 / 255.0),
            "Green": (86 / 255.0, 207 / 255.0, 33 / 255.0),
            "Blue": (0 / 255.0, 0 / 255.0, 225 / 255.0),
            "Yellow": (231 / 255.0, 213 / 255.0, 32 / 255.0),
            "Cyan": (0.0, 1.0, 1.0),
            "Magenta": (1.0, 0.0, 1.0),
        }

        rows: List[dict] = []
        saved_order = list(getattr(self, "montage_tile_order", []) or [])
        order_rank = {key: i + 1 for i, key in enumerate(saved_order)}
        for idx in range(len(tiles)):
            label = labels[idx] if idx < len(labels) else (tiles[idx][1] if idx < len(tiles) else f"Tile {idx}")
            tile_key = self._montage_tile_key(label)
            is_merged = str(label).endswith(" Merged")
            default_color = tiles[idx][2] if idx < len(tiles) else None
            color_override = self._montage_lookup_color_override(tile_key, idx, default_color)
            current_rgb = color_override if color_override is not None else default_color
            pct = self._montage_lookup_percentile_override(tile_key, idx) or (
                self.MONTAGE_AUTO_LUT_LOW,
                self.MONTAGE_AUTO_LUT_HIGH,
            )
            include_cb = QCheckBox()
            include_cb.setChecked(self._montage_tile_included(tile_key))
            order_spin = QSpinBox()
            order_spin.setRange(1, max(1, len(tiles)))
            order_spin.setValue(order_rank.get(tile_key, idx + 1))
            order_spin.setFixedWidth(64)
            z_combo = QComboBox()
            z_combo.addItems(z_choices)
            z_mode = str(self.montage_tile_z_modes.get(tile_key, "global"))
            z_combo.setCurrentText(mode_to_z.get(z_mode, "Use global"))
            z_combo.setEnabled(not is_merged)
            color_combo = QComboBox()
            color_combo.addItems(preset_names)
            color_combo.setEnabled(not is_merged and default_color is not None)
            if default_color is None:
                color_combo.setCurrentIndex(0)
            elif color_override is None:
                color_combo.setCurrentIndex(0)
            else:
                key = next((k for k, v in preset_to_rgb.items() if v == color_override), None)
                if key and key != "Pick…":
                    color_combo.setCurrentIndex(preset_names.index(key))
                else:
                    color_combo.addItem("Custom")
                    color_combo.setCurrentText("Custom")
            p_low_spin = QDoubleSpinBox()
            p_low_spin.setRange(0.0, 99.9)
            p_low_spin.setDecimals(1)
            p_low_spin.setSingleStep(0.5)
            p_low_spin.setValue(float(pct[0]))
            p_low_spin.setFixedWidth(72)
            p_low_spin.setEnabled(not is_merged)
            p_high_spin = QDoubleSpinBox()
            p_high_spin.setRange(0.1, 100.0)
            p_high_spin.setDecimals(1)
            p_high_spin.setSingleStep(0.5)
            p_high_spin.setValue(float(pct[1]))
            p_high_spin.setFixedWidth(72)
            p_high_spin.setEnabled(not is_merged)
            rows.append(
                {
                    "tile_idx": idx,
                    "tile_key": tile_key,
                    "label": label,
                    "is_merged": is_merged,
                    "default_color": default_color,
                    "include_cb": include_cb,
                    "order_spin": order_spin,
                    "z_combo": z_combo,
                    "color_combo": color_combo,
                    "current_rgb": current_rgb,
                    "p_low_spin": p_low_spin,
                    "p_high_spin": p_high_spin,
                }
            )

        if not rows:
            QMessageBox.information(self, "Edit tiles", "No tiles in the current montage layout.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Edit tiles")
        dlg.setMinimumWidth(720)
        vlayout = QVBoxLayout(dlg)
        vlayout.addWidget(
            QLabel(
                "Include or exclude tiles, set z-stack handling per channel, and adjust color / percentile normalization. "
                "Unchecked tiles stay in this list so you can bring them back. Use Order to rearrange the montage."
            )
        )
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_body = QWidget()
        form = QFormLayout(scroll_body)
        for row in rows:
            line = QWidget()
            line_layout = QHBoxLayout(line)
            line_layout.setContentsMargins(0, 0, 0, 0)
            line_layout.setSpacing(6)
            line_layout.addWidget(row["include_cb"])
            line_layout.addWidget(QLabel("Order:"))
            line_layout.addWidget(row["order_spin"])
            line_layout.addWidget(QLabel("Z:"))
            line_layout.addWidget(row["z_combo"])
            line_layout.addWidget(row["color_combo"])
            line_layout.addWidget(QLabel("P"))
            line_layout.addWidget(row["p_low_spin"])
            line_layout.addWidget(QLabel("–"))
            line_layout.addWidget(row["p_high_spin"])
            line_layout.addWidget(QLabel("%"))
            line_layout.addStretch()
            short = row["label"][:48] + ("…" if len(row["label"]) > 48 else "")
            form.addRow(short, line)
        scroll.setWidget(scroll_body)
        vlayout.addWidget(scroll, 1)

        def on_combo_change(row_idx: int) -> None:
            r = rows[row_idx]
            if r["is_merged"] or r["default_color"] is None:
                return
            default_color = r["default_color"]
            combo = r["color_combo"]
            current_rgb = r["current_rgb"]
            text = combo.currentText()
            if text == "Default":
                new_rgb = default_color
            elif text == "Pick…":
                base = current_rgb or default_color or (1.0, 1.0, 1.0)
                qc = QColorDialog.getColor(
                    QColor(int(base[0] * 255), int(base[1] * 255), int(base[2] * 255)),
                    self,
                    "Pick color for tile",
                )
                if qc.isValid():
                    new_rgb = (qc.red() / 255.0, qc.green() / 255.0, qc.blue() / 255.0)
                else:
                    new_rgb = current_rgb
                if combo.findText("Custom") < 0:
                    combo.addItem("Custom")
                combo.setCurrentText("Custom")
            elif text == "Custom":
                new_rgb = current_rgb
            else:
                new_rgb = preset_to_rgb.get(text) or default_color
            rows[row_idx]["current_rgb"] = new_rgb

        for i, row in enumerate(rows):
            row["color_combo"].currentIndexChanged.connect(lambda *_, ri=i: on_combo_change(ri))

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        vlayout.addWidget(bb)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            excluded: set[str] = set()
            ordered_rows = sorted(
                enumerate(rows),
                key=lambda item: (int(item[1]["order_spin"].value()), item[0]),
            )
            self.montage_tile_order = [row["tile_key"] for _i, row in ordered_rows]
            for row in rows:
                tile_key = row["tile_key"]
                if not row["include_cb"].isChecked():
                    excluded.add(tile_key)
                z_text = row["z_combo"].currentText()
                self.montage_tile_z_modes[tile_key] = z_to_mode.get(z_text, "global")
                if row["is_merged"] or row["default_color"] is None:
                    continue
                default_rgb = row["default_color"]
                chosen_rgb = row["current_rgb"]
                if chosen_rgb is not None and chosen_rgb != default_rgb:
                    self.montage_tile_color_overrides[tile_key] = chosen_rgb
                else:
                    self.montage_tile_color_overrides.pop(tile_key, None)
                p_low = float(row["p_low_spin"].value())
                p_high = float(row["p_high_spin"].value())
                if p_high <= p_low:
                    p_high = min(100.0, p_low + 0.1)
                if abs(p_low - self.MONTAGE_AUTO_LUT_LOW) > 1e-9 or abs(p_high - self.MONTAGE_AUTO_LUT_HIGH) > 1e-9:
                    self.montage_tile_percentile_overrides[tile_key] = (p_low, p_high)
                else:
                    self.montage_tile_percentile_overrides.pop(tile_key, None)
            self.montage_excluded_tile_keys = excluded
            if refresh_callback:
                refresh_callback()

    def _select_montage_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder with images", get_default_browse_dir())
        if path:
            self.montage_path.setText(path)

    def _select_montage_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select images", get_default_browse_dir(),
            "Images (*.tif *.tiff *.nd2 *.png *.jpg *.jpeg);;ND2 (*.nd2);;TIFF (*.tif *.tiff);;All (*)"
        )
        if paths:
            self.montage_path.setText("|".join(paths))

    def _draw_scalebar_on_image(self, img: np.ndarray, *, params: Optional[dict[str, Any]] = None) -> Optional[np.ndarray]:
        """Draw scale bar on a copy of img (H, W, 3) uint8. Position auto-adjusts so bar + label stay inside image."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            QMessageBox.warning(self, "Scale bar", "Pillow (PIL) is required for drawing the scale bar.")
            return None
        if params is None:
            params = self._montage_scalebar_params()
        h, w = img.shape[:2]
        target_rect = params.get("target_rect")
        if target_rect is not None:
            try:
                tx, ty, tw, th = [int(v) for v in target_rect]
                tx = max(0, min(tx, max(0, w - 1)))
                ty = max(0, min(ty, max(0, h - 1)))
                tw = max(1, min(tw, w - tx))
                th = max(1, min(th, h - ty))
            except Exception:
                tx, ty, tw, th = 0, 0, w, h
        else:
            tx, ty, tw, th = 0, 0, w, h
        pixel_size = float(params.get("pixel_size") or 0.108)
        length_um = float(params.get("length_um") or 10.0)
        length_px = max(1, int(round(length_um / max(pixel_size, 1e-9))))
        thickness = int(params.get("thickness") or 5)
        margin = max(12, thickness + 6)
        length_px = min(length_px, max(1, tw - 2 * margin))
        gap = 4
        pos_name = str(params.get("position") or "Bottom-right").lower()
        color_name = str(params.get("color") or "White").lower()
        color = (255, 255, 255) if color_name == "white" else (0, 0, 0)
        if img.ndim == 3 and img.shape[-1] == 4:
            color = (*color, 255)
        font = None
        font_size = int(params.get("font_size") or 20)
        for path in [
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "arial.ttf",
        ]:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        show_label = bool(params.get("show_label", True))
        label = ""
        label_w, label_h = 0, 0
        if show_label:
            label = f"{length_um:.0f}" if length_um >= 1 else f"{length_um}"
            label += " µm"
            try:
                bbox = font.getbbox(label)
                label_w = bbox[2] - bbox[0]
                label_h = bbox[3] - bbox[1]
            except AttributeError:
                label_w = len(label) * (font_size // 2)
                label_h = font_size

        # Vertical position so bar + label fit (label above bar for bottom, below bar for top)
        if "bottom" in pos_name:
            y0_max = ty + th - margin - thickness
            y0 = (y0_max - (label_h + gap)) if show_label else y0_max
            # Keep label inside image: label top at ly = y0 - gap - label_h must be >= margin
            if show_label:
                y0 = max(y0, ty + margin + gap + label_h)
            y0 = max(ty + margin, y0)
        else:
            y0 = ty + margin
            if show_label and (y0 + thickness + gap + label_h > ty + th - margin):
                y0 = max(ty + margin, ty + th - margin - thickness - gap - label_h)
        y1 = y0 + thickness

        # Horizontal: prefer bar at corner, clamp so label stays in [margin, w-margin]
        half_bar = length_px // 2
        half_label = (label_w + 1) // 2
        lx_min = tx + margin + half_label
        lx_max = (tx + tw - margin - half_label) if tw > 2 * margin + label_w else lx_min
        if "right" in pos_name:
            lx = min(tx + tw - margin - half_bar, lx_max)
            lx = max(lx, lx_min)
        else:
            lx = max(tx + margin + half_bar, lx_min)
            lx = min(lx, lx_max)
        x0 = int(lx - half_bar)
        x0 = max(tx + margin, min(x0, tx + tw - margin - length_px))
        x1 = x0 + length_px
        lx = x0 + half_bar

        pil_img = Image.fromarray(img)
        draw = ImageDraw.ImageDraw(pil_img)
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=color)
        if show_label:
            if "top" in pos_name:
                ly = y1 + gap  # label top just below bar
            else:
                ly = y0 - gap - label_h  # label bottom just above bar (anchor "mt" = top of text at ly)
            draw.text((lx, ly), label, fill=color, font=font, anchor="mt")
        return np.array(pil_img)

    _MONTAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".nd2")

    # Channel name -> (R, G, B) in 0-1 for false color. Phase → gray; common fluorophores → semantic colors.
    _CHANNEL_COLOR_MAP = {
        "gray": (1.0, 1.0, 1.0),
        "grayscale": (1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0),
        "orange": (1.0, 86 / 255.0, 0.0),  # RGB (255, 86, 0) — dsRED / mScarlet / 561 nm
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
    }

    @staticmethod
    def _channel_name_to_rgb(name: str) -> tuple[float, float, float]:
        """Map channel name (from metadata) to display color (R,G,B) 0-1. Phase→gray; mScarlet/RFP→red; GFP→green; YFP→yellow; etc."""
        if not name:
            return (0.0, 1.0, 0.0)  # default green
        n = name.strip().lower()
        if "phase" in n or "bf" == n or "brightfield" in n or "trans" in n or "transmission" in n:
            return (1.0, 1.0, 1.0)  # grayscale
        if "mscarlet" in n or "dsred" in n or "561" in n or "orange" in n:
            return (1.0, 86 / 255.0, 0.0)  # orange RGB (255, 86, 0) — dsRED / mScarlet / 561 nm
        if "mcherry" in n or "rfp" in n or "red" in n or "cy3" in n or "texas" in n:
            return (1.0, 0.0, 0.0)
        if "gfp" in n or "egfp" in n or "fitc" in n or "green" in n:
            return (0.0, 1.0, 0.0)
        if "yfp" in n or "mvenus" in n or "venus" in n or "yellow" in n:
            return (1.0, 1.0, 0.0)
        if "cfp" in n or "cerulean" in n or "cyan" in n:
            return (0.0, 1.0, 1.0)
        if "cy5" in n or "magenta" in n:
            return (1.0, 0.0, 1.0)
        return (0.0, 1.0, 0.0)  # default green

    # Default colors for multiple single-channel (non-ND2) images: cycle so each file gets a different color.
    _CYCLE_COLORS = ((1.0, 0.0, 0.0), (1.0, 86 / 255.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0))  # R, orange (255,86,0), G, B, Y, C, M

    def _get_channel_colors_for_path(
        self, path: Path, n_channels: int, entry_index: Optional[int] = None
    ) -> List[tuple]:
        """Return list of (r,g,b) 0-1 per channel. ND2: from metadata + overrides. Non-ND2 single-channel: cycle by entry_index."""
        if path.suffix.lower() != ".nd2" and n_channels == 1 and entry_index is not None:
            return [self._CYCLE_COLORS[entry_index % len(self._CYCLE_COLORS)]]
        overrides = getattr(self, "montage_channel_color_overrides", None) or {}
        meta = read_metadata_from_path(path) if path.suffix.lower() == ".nd2" else {}
        names = meta.get("channel_names") or [f"Ch{i+1}" for i in range(n_channels)]
        wavelengths = meta.get("channel_wavelengths") or []
        result = []
        for i in range(n_channels):
            key = (path.stem, i)
            if key in overrides:
                choice = overrides[key].lower()
                if choice == "auto":
                    pass  # fall through to name-based
                else:
                    result.append(self._CHANNEL_COLOR_MAP.get(choice, (0.0, 1.0, 0.0)))
                    continue
            # Auto: use phase flag or name
            if i < len(wavelengths) and wavelengths[i].get("phase_contrast"):
                result.append((1.0, 1.0, 1.0))
            else:
                result.append(self._channel_name_to_rgb(names[i] if i < len(names) else ""))
        return result

    # TIFF filename suffixes → channel name for grouping (stem must end with one of these).
    _CHANNEL_SUFFIXES = [
        ("_phase", "phase"), ("_Phase", "phase"),
        ("_phase60x", "phase"), ("_Phase60X", "phase"), ("_phase40x", "phase"), ("_phase100x", "phase"),  # e.g. EM13748_AnTC_004_Phase60X
        ("_fi", "FI"), ("_FI", "FI"),  # generic fluorescence (e.g. untitled_fi.tif + untitled_phase.tif → one merged tile)
        ("_gfp", "GFP"), ("_GFP", "GFP"), ("_egfp", "GFP"),
        ("_mscarlet", "mScarlet"), ("_mScarlet", "mScarlet"), ("_dsred", "dsRED"), ("_dsRed", "dsRED"),
        ("_mcherry", "mCherry"), ("_rfp", "RFP"),
        ("_561", "561nm"), ("_orange", "Orange"),
        ("_yfp", "YFP"), ("_YFP", "YFP"), ("_venus", "Venus"),
        ("_cfp", "CFP"), ("_cyan", "Cyan"),
        ("_red", "red"), ("_green", "green"), ("_blue", "blue"),
    ]

    @staticmethod
    def _looks_non_channel_token(token: str) -> bool:
        t = (token or "").strip().lower()
        if not t:
            return True
        if t in {"fi", "foci", "phase", "mask", "masks", "seg", "segmentation", "overlay", "bf", "dic"}:
            return True
        if t.endswith("x") and t[:-1].isdigit():
            return True
        if t.startswith("t=") or t.startswith("z="):
            return True
        return False

    @classmethod
    def _infer_channel_from_stem(cls, stem: str) -> Optional[tuple[str, str]]:
        """If stem looks like 'base_channel', return (base, channel_name). E.g. exp_phase -> (exp, phase); exp_fi_GFP -> (exp, GFP)."""
        if "_fi_" in stem:
            base, _, ch = stem.partition("_fi_")
            return (base.strip("_"), ch.strip("_") or "FI")
        if "_foci_" in stem:
            base, _, ch = stem.partition("_foci_")
            return (base.strip("_"), ch.strip("_") or "FI")
        if stem.lower().endswith("_foci"):
            prefix = stem[:-len("_foci")].strip("_")
            if "_" in prefix:
                base, token = prefix.rsplit("_", 1)
                if token and not cls._looks_non_channel_token(token):
                    return (base.strip("_"), token.strip("_"))
            return (prefix.strip("_"), "FI")
        stem_lower = stem.lower()
        for suffix, ch_name in cls._CHANNEL_SUFFIXES:
            if stem_lower.endswith(suffix.lower()):
                base = stem[: -len(suffix)].strip("_")
                return (base, ch_name)
        return None

    @staticmethod
    def _channel_sort_key(channel_name: str) -> tuple:
        """Sort key: phase first (0), then red(1), orange(2), green(3), blue(4), yellow(5), cyan(6), magenta(7), else (8, name)."""
        n = channel_name.strip().lower()
        if "phase" in n or n in ("bf", "brightfield", "trans"):
            return (0, n)
        if "mcherry" in n or "rfp" in n or "red" in n or "cy3" in n:
            return (1, n)
        if "mscarlet" in n or "dsred" in n or "561" in n or "orange" in n:
            return (2, n)
        if "green" in n or "gfp" in n or "fitc" in n:
            return (3, n)
        if "blue" in n:
            return (4, n)
        if "yellow" in n or "yfp" in n or "venus" in n:
            return (5, n)
        if "cyan" in n or "cfp" in n:
            return (6, n)
        if "magenta" in n or "cy5" in n:
            return (7, n)
        return (8, n)

    @staticmethod
    def _normalize_channel_token(channel_name: str) -> str:
        """Normalize channel names for cross-source matching (Excel vs filename/metadata)."""
        t = "".join(ch for ch in (channel_name or "").strip().lower() if ch.isalnum())
        if not t:
            return ""
        aliases = {
            "egfp": "gfp",
            "yfp": "venus",
            "mvenus": "venus",
            "rfp": "mcherry",
            "dsred": "mscarlet",
            "fluorescence": "fi",
            "fluor": "fi",
            "foci": "fi",
            "brightfield": "phase",
            "transmission": "phase",
        }
        return aliases.get(t, t)

    @classmethod
    def _is_phase_channel_token(cls, token: str) -> bool:
        t = cls._normalize_channel_token(token)
        return t in {"phase", "bf", "trans", "brightfield"}

    def _prepare_montage_mask_overlay(
        self,
        base_name: str,
        channel_paths: List[Path],
        frame_index: int,
        mask_override_path: Optional[Path],
    ) -> dict:
        """Build montage overlay context: labeled mask frame and shape (cell boundaries only)."""
        mask_path = self._resolve_montage_mask_path(base_name, channel_paths, mask_override_path)
        mask_frame = self._load_mask_frame(mask_path, frame_index) if mask_path is not None else None
        src_shape: Optional[tuple[int, int]] = None
        if mask_frame is not None and mask_frame.ndim == 2:
            src_shape = (int(mask_frame.shape[0]), int(mask_frame.shape[1]))
        return {
            "base_name": base_name,
            "mask_frame": mask_frame,
            "mask_path": mask_path,
            "source_shape": src_shape,
            "boundary_cache": {},
        }

    def _resolve_montage_mask_path(
        self,
        base_name: str,
        channel_paths: List[Path],
        mask_override: Optional[Path],
    ) -> Optional[Path]:
        """Resolve mask path for montage overlays, preferring manual override."""
        if mask_override is not None and mask_override.exists() and mask_override.is_file():
            return mask_override

        base_candidates = {base_name.strip("_")}
        if base_name.lower().endswith("_phase"):
            base_candidates.add(base_name[:-6].strip("_"))
        if base_name.lower().endswith("_fi"):
            base_candidates.add(base_name[:-3].strip("_"))
        for p in channel_paths:
            base_candidates.add(p.stem.strip("_"))
            inferred = self._infer_channel_from_stem(p.stem)
            if inferred and inferred[0]:
                base_candidates.add(inferred[0].strip("_"))

        search_dirs: List[Path] = []
        for p in channel_paths:
            for d in (p.parent, p.parent.parent):
                if d.exists() and d.is_dir() and d not in search_dirs:
                    search_dirs.append(d)

        seg_dirs: List[Path] = []
        for d in list(search_dirs):
            try:
                for sd in d.glob("_segmentation_*"):
                    if sd.is_dir() and sd not in seg_dirs:
                        seg_dirs.append(sd)
            except Exception:
                continue
        for sd in seg_dirs:
            if sd not in search_dirs:
                search_dirs.append(sd)

        for d in search_dirs:
            for base in sorted(b for b in base_candidates if b):
                for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
                    candidate = d / f"{base}_masks{ext}"
                    if candidate.exists() and candidate.is_file():
                        return candidate
        return None

    @staticmethod
    def _load_mask_frame(mask_path: Path, frame_index: int) -> Optional[np.ndarray]:
        """Load one mask frame as int32 labels."""
        try:
            arr = safe_imread(str(mask_path))
        except Exception:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 2:
            return arr.astype(np.int32, copy=False)
        if arr.ndim == 3:
            # Prefer treating as T,Y,X for masks.
            if arr.shape[-1] in (1, 2, 3, 4) and arr.shape[0] > 8 and arr.shape[1] > 8:
                arr = arr[..., 0]
            else:
                idx = int(max(0, min(frame_index, arr.shape[0] - 1)))
                arr = arr[idx]
            if arr.ndim == 2:
                return arr.astype(np.int32, copy=False)
        if arr.ndim == 4:
            t_idx = int(max(0, min(frame_index, arr.shape[0] - 1)))
            arr = arr[t_idx]
            if arr.ndim == 3:
                if arr.shape[0] in (1, 2, 3, 4):
                    arr = arr[0]
                elif arr.shape[-1] in (1, 2, 3, 4):
                    arr = arr[..., 0]
                else:
                    arr = arr[0]
            if arr.ndim == 2:
                return arr.astype(np.int32, copy=False)
        return None

    @staticmethod
    def _mask_boundaries(mask_labels: np.ndarray) -> np.ndarray:
        """Return binary boundaries for labeled mask image."""
        if mask_labels is None or mask_labels.ndim != 2:
            return np.zeros((1, 1), dtype=bool)
        try:
            from skimage.segmentation import find_boundaries

            return find_boundaries(mask_labels, mode="inner") & (mask_labels > 0)
        except Exception:
            m = np.asarray(mask_labels)
            edges = np.zeros_like(m, dtype=bool)
            edges[1:, :] |= (m[1:, :] != m[:-1, :])
            edges[:-1, :] |= (m[:-1, :] != m[1:, :])
            edges[:, 1:] |= (m[:, 1:] != m[:, :-1])
            edges[:, :-1] |= (m[:, :-1] != m[:, 1:])
            return edges & (m > 0)

    def _apply_overlay_to_tile(
        self,
        tile_rgb: np.ndarray,
        tile_meta: dict,
        overlay_ctx: dict,
        tile_index: int,
        tiles: List[tuple],
        per_tile_color_override: Optional[Dict[int, Tuple[float, float, float]]],
    ) -> np.ndarray:
        """Apply cell boundary overlay from a labeled mask (yellow contours)."""
        if tile_rgb is None or tile_rgb.ndim != 3 or tile_rgb.shape[-1] != 3:
            return tile_rgb
        from PIL import Image

        arr = np.ascontiguousarray(tile_rgb.copy())
        h, w = arr.shape[:2]

        mask_frame = overlay_ctx.get("mask_frame")
        if mask_frame is not None and mask_frame.ndim == 2:
            cache = overlay_ctx.setdefault("boundary_cache", {})
            key = (h, w)
            edges = cache.get(key)
            if edges is None:
                edges = self._mask_boundaries(mask_frame)
                if edges.shape != (h, w):
                    img = Image.fromarray((edges.astype(np.uint8) * 255))
                    img = img.resize((w, h), Image.Resampling.NEAREST)
                    edges = np.asarray(img) > 0
                cache[key] = edges
            arr[edges] = np.array([255, 255, 0], dtype=np.uint8)

        return arr

    def _get_montage_datasets(self) -> List[dict]:
        """
        Return list of datasets. Each dataset yields tiles in order: Phase, FI channels (with colors), Merged.
        - Single ND2 or single multi-channel TIFF: one dataset (channels from metadata, frame 0).
        - Folder of ND2s: one dataset per ND2.
        - Folder of TIFFs: group by base name (e.g. exp_phase.tif + exp_GFP.tif → one dataset); channels from filenames.
        """
        paths = self._get_montage_image_paths()
        if not paths:
            return []
        nd2_paths = [p for p in paths if p.suffix.lower() == ".nd2"]
        tiff_paths = [p for p in paths if p.suffix.lower() in (".tif", ".tiff")]

        def _detect_tiff_channel_count(path: Path, meta: Optional[dict[str, Any]] = None) -> int:
            """Detect channel count from TIFF axes first, then metadata, then common stack shapes."""
            meta = meta or read_metadata_from_path(path)
            try:
                nc_meta = int(meta.get("channel_count") or 1)
            except Exception:
                nc_meta = 1
            best = max(1, nc_meta)
            try:
                import tifffile

                with tifffile.TiffFile(str(path)) as tif:
                    if getattr(tif, "series", None) and len(tif.series) > 0:
                        s0 = tif.series[0]
                        axes = str(getattr(s0, "axes", "") or "").upper()
                        shape = tuple(int(x) for x in (getattr(s0, "shape", None) or ()))
                        if axes and shape and len(axes) == len(shape):
                            if "C" in axes:
                                best = max(best, int(shape[axes.index("C")]))
                            elif "S" in axes and axes.endswith("S"):
                                # Samples axis is RGB/RGBA display data, not microscopy channels.
                                pass
                            elif len(shape) >= 3:
                                leading = int(shape[0])
                                second = int(shape[1]) if len(shape) >= 4 else 1
                                if 1 < leading <= 12 and axes[-2:] == "YX":
                                    best = max(best, leading)
                                elif leading == 1 and 1 < second <= 12 and axes[-2:] == "YX":
                                    best = max(best, second)
                        elif shape and len(shape) >= 3:
                            leading = int(shape[0])
                            second = int(shape[1]) if len(shape) >= 4 else 1
                            if 1 < leading <= 12 and shape[-1] not in (3, 4):
                                best = max(best, leading)
                            elif leading == 1 and 1 < second <= 12 and shape[-1] not in (3, 4):
                                best = max(best, second)
            except Exception:
                pass
            return max(1, best)

        # Single file: one dataset
        if len(paths) == 1:
            p = paths[0]
            if p.suffix.lower() == ".nd2":
                return [{"type": "nd2", "path": p}]
            # Single TIFF: treat as one channel unless multi-channel metadata
            meta = read_metadata_from_path(p)
            nc = _detect_tiff_channel_count(p, meta)
            if nc > 1:
                return [{"type": "nd2_like", "path": p, "channel_count": nc}]  # same load path as ND2
            return [{"type": "single_tiff", "path": p}]

        # Multiple files: ND2s each one dataset; TIFFs grouped by base
        datasets: List[dict] = []
        for p in nd2_paths:
            datasets.append({"type": "nd2", "path": p})

        if tiff_paths:
            # Two or more TIFFs: one group → tile per file + one Merged tile (colors from filename when detected, else user can set in montage)
            if len(tiff_paths) >= 2:
                channel_entries: List[tuple[Path, str, Optional[tuple[str, str]]]] = []
                inferred_bases: List[str] = []
                for p in tiff_paths:
                    inferred = self._infer_channel_from_stem(p.stem)
                    ch_name = inferred[1] if inferred else p.stem
                    if inferred and inferred[0]:
                        inferred_bases.append(inferred[0])
                    channel_entries.append((p, ch_name, inferred))
                # Drop generic helper files (_fi / _foci) when real channel-specific files are present.
                has_specific = any(
                    inf is not None
                    and (inf[1] or "").strip().lower() not in {"fi"}
                    for _p, _ch, inf in channel_entries
                )
                if has_specific:
                    filtered_entries = []
                    for p, ch_name, inf in channel_entries:
                        stem_low = p.stem.lower()
                        is_generic = (
                            (inf is not None and (inf[1] or "").strip().lower() == "fi")
                            and (stem_low.endswith("_fi") or stem_low.endswith("_foci"))
                        )
                        if not is_generic:
                            filtered_entries.append((p, ch_name, inf))
                    if filtered_entries:
                        channel_entries = filtered_entries
                channel_list: List[tuple[Path, str]] = [(p, ch_name) for p, ch_name, _inf in channel_entries]
                channel_list.sort(key=lambda x: self._channel_sort_key(x[1]))
                if inferred_bases and len(set(inferred_bases)) == 1:
                    base = inferred_bases[0]
                elif inferred_bases:
                    base = inferred_bases[0]
                else:
                    base = channel_list[0][0].stem if channel_list else "montage"
                datasets.append({"type": "tiff_group", "base": base, "channels": channel_list})
            else:
                # Single TIFF: one tile only (no merge)
                datasets.append({"type": "single_tiff", "path": tiff_paths[0]})

        return datasets

    def _get_montage_image_paths(self) -> List[Path]:
        """Resolve montage input to a sorted list of image paths. Folder: filter by dropdown (All / ND2 only / TIFF only)."""
        raw = self.montage_path.text().strip()
        if not raw:
            return []
        if "|" in raw:
            return [Path(p.strip()) for p in raw.split("|") if Path(p.strip()).exists()]
        p = Path(raw)
        if not p.exists():
            return []
        if p.is_file():
            return [p] if p.suffix.lower() in self._MONTAGE_EXTS else []
        # Folder: include all ND2 and TIFF (deduplicate by resolved path so each file appears once)
        paths = []
        seen = set()
        for ext in self._MONTAGE_EXTS:
            for x in p.glob(f"*{ext}"):
                if x.is_file():
                    r = x.resolve()
                    if r not in seen:
                        seen.add(r)
                        paths.append(x)
        return sorted(paths, key=lambda x: x.name)

    def _montage_start_folder(self) -> Path:
        paths = self._get_montage_image_paths()
        if paths:
            return paths[0].parent
        path_widget = getattr(self, "montage_path", None)
        raw = path_widget.text().strip() if self._qt_widget_alive(path_widget) else ""
        if raw:
            p = Path(raw.split("|")[0].strip())
            if p.is_dir():
                return p
            if p.parent.exists():
                return p.parent
        return Path(get_default_browse_dir())

    def _montage_batch_candidates(self, folder: Path) -> List[Path]:
        """ND2/TIFF files in a folder, sorted once by filename."""
        if not folder.exists() or not folder.is_dir():
            return []
        exts = (".nd2", ".tif", ".tiff")
        seen = set()
        files: List[Path] = []
        for p in folder.iterdir():
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            try:
                key = p.resolve()
            except Exception:
                key = p
            if key in seen:
                continue
            seen.add(key)
            files.append(p)
        return sorted(files, key=lambda p: p.name.lower())

    def _select_montage_batch_files(self, parent: QWidget) -> Optional[List[Path]]:
        """Pick ND2/TIFF files for applying the current montage preview settings."""
        dialog = QDialog(parent)
        dialog.setWindowTitle("Use montage settings on files")
        dialog.setMinimumSize(620, 520)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 18)
        layout.setSpacing(10)

        message = QLabel(
            "Select ND2/TIFF files. Each selected file will be exported with the current montage preview settings."
        )
        message.setWordWrap(True)
        layout.addWidget(message)

        current_dir = [self._montage_start_folder()]
        folder_row = QHBoxLayout()
        folder_label = QLabel(str(current_dir[0]))
        folder_label.setWordWrap(True)
        browse_btn = QPushButton("Choose folder...")
        style_compact_action_button(browse_btn, horizontal_padding=18)
        folder_row.addWidget(QLabel("Folder:"))
        folder_row.addWidget(folder_label, 1)
        folder_row.addWidget(browse_btn)
        layout.addLayout(folder_row)

        action_row = QHBoxLayout()
        select_all_btn = QPushButton("Select all")
        deselect_all_btn = QPushButton("Deselect all")
        style_compact_action_button(select_all_btn, horizontal_padding=18)
        style_compact_action_button(deselect_all_btn, horizontal_padding=18)
        action_row.addWidget(select_all_btn)
        action_row.addWidget(deselect_all_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        file_list = QListWidget()
        layout.addWidget(file_list, 1)
        status_label = QLabel("")
        status_label.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(status_label)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        for std_btn in (
            button_box.button(QDialogButtonBox.StandardButton.Ok),
            button_box.button(QDialogButtonBox.StandardButton.Cancel),
        ):
            if std_btn is not None:
                style_compact_action_button(std_btn, horizontal_padding=20)
                std_btn.setFixedHeight(36)
        button_box.setMinimumHeight(44)

        footer = QWidget(dialog)
        footer.setMinimumHeight(54)
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(0, 8, 0, 2)
        footer_layout.setSpacing(0)
        footer_layout.addWidget(button_box)
        layout.addWidget(footer, 0)

        def populate(folder: Path) -> None:
            file_list.clear()
            files = self._montage_batch_candidates(folder)
            current_paths = set()
            for p in self._get_montage_image_paths():
                try:
                    current_paths.add(p.resolve())
                except Exception:
                    current_paths.add(p)
            for p in files:
                item = QListWidgetItem(p.name)
                item.setData(Qt.ItemDataRole.UserRole, p)
                try:
                    is_current = p.resolve() in current_paths
                except Exception:
                    is_current = p in current_paths
                item.setCheckState(Qt.CheckState.Checked if is_current or not current_paths else Qt.CheckState.Unchecked)
                file_list.addItem(item)
            folder_label.setText(str(folder))
            status_label.setText(f"{len(files)} ND2/TIFF file(s) found.")
            ok_btn = button_box.button(QDialogButtonBox.StandardButton.Ok)
            if ok_btn is not None:
                ok_btn.setEnabled(bool(files))

        def choose_folder() -> None:
            selected = QFileDialog.getExistingDirectory(dialog, "Select folder with montage files", str(current_dir[0]))
            if not selected:
                return
            current_dir[0] = Path(selected)
            populate(current_dir[0])

        def set_all(checked: bool) -> None:
            state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            for i in range(file_list.count()):
                file_list.item(i).setCheckState(state)

        browse_btn.clicked.connect(choose_folder)
        select_all_btn.clicked.connect(lambda: set_all(True))
        deselect_all_btn.clicked.connect(lambda: set_all(False))
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        populate(current_dir[0])
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        selected_files: List[Path] = []
        for i in range(file_list.count()):
            item = file_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_files.append(Path(item.data(Qt.ItemDataRole.UserRole)))
        return selected_files

    def _build_montage_array(
        self,
        per_tile_lut: Optional[Dict[int, Tuple[float, float]]] = None,
        per_tile_color_override: Optional[Dict[int, Tuple[float, float, float]]] = None,
        per_tile_percentile_override: Optional[Dict[int, Tuple[float, float]]] = None,
        return_layout: bool = False,
        merged_only: bool = False,
    ) -> Optional[np.ndarray]:
        """Build montage: Phase first, then FI channels (with colors), then merged. One frame only (first timepoint; z-stack projection optional).
        If per_tile_lut is set, use (vmin, vmax) per tile index; else use per-tile percentile.
        If per_tile_color_override is set, use (r,g,b) per tile index for colored tiles.
        If per_tile_percentile_override is set, use (low, high) percentiles per tile when LUT is not manually set.
        If return_layout is True, return (array, layout_dict) with layout_dict containing rows, cols, tiles, etc. for preview."""
        datasets = self._get_montage_datasets()
        if not datasets:
            return None
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            QMessageBox.warning(self, "Montage", "Pillow (PIL) is required for montage.")
            return None
        use_color_merge = True  # Always: phase + FI channels + merged
        use_global_lut = False  # Per-tile LUT (or per_tile_lut from preview)
        order_col_major = "Column" in self._safe_combo_text(
            getattr(self, "montage_order", None),
            "Row-major (left→right, top→bottom)",
        )
        rows = self._safe_spin_value(getattr(self, "montage_rows", None), 1)
        cols = self._safe_spin_value(getattr(self, "montage_cols", None), 0)
        spacing = self._safe_spin_value(getattr(self, "montage_spacing", None), 4)
        border = self._safe_spin_value(getattr(self, "montage_border", None), 8)
        z_project_max = self._montage_use_z_max_projection()
        label_settings = self._montage_label_settings()
        show_labels = bool(label_settings["show"])
        font_size = int(label_settings["font_size"])
        label_position = str(label_settings["position"])
        label_color_name = str(label_settings["color_name"])
        overlay_enabled = self._safe_checked(getattr(self, "montage_overlay_enable", None), default=False)
        overlay_frame = self._safe_spin_value(getattr(self, "montage_overlay_frame", None), 0)
        overlay_mask_override = None
        mask_edit = getattr(self, "montage_overlay_mask_path", None)
        if self._qt_widget_alive(mask_edit):
            mask_text = mask_edit.text().strip()
            if mask_text:
                overlay_mask_override = Path(mask_text)

        def _normalize_plane(
            plane: np.ndarray,
            p_low: float = self.MONTAGE_AUTO_LUT_LOW,
            p_high: float = self.MONTAGE_AUTO_LUT_HIGH,
        ) -> np.ndarray:
            """Robust 0-1 normalization using percentiles so weak channels don't go black."""
            plane = np.asarray(plane, dtype=np.float64)
            valid = plane[np.isfinite(plane)]
            if valid.size == 0:
                return np.zeros_like(plane)
            vmin = float(np.percentile(valid, p_low))
            vmax = float(np.percentile(valid, p_high))
            if vmax <= vmin:
                vmax = vmin + 1.0
            out = (plane - vmin) / (vmax - vmin)
            return np.clip(out, 0.0, 1.0)

        def _normalize_plane_with_lut(plane: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> np.ndarray:
            """Normalize plane to 0-1 using (vmin, vmax) if provided, else percentile."""
            plane = np.asarray(plane, dtype=np.float64)
            valid = plane[np.isfinite(plane)]
            if valid.size == 0:
                return np.zeros_like(plane)
            if vmin is not None and vmax is not None and vmax > vmin:
                out = (plane - vmin) / (vmax - vmin)
            else:
                vmin = float(np.percentile(valid, self.MONTAGE_AUTO_LUT_LOW))
                vmax = float(np.percentile(valid, self.MONTAGE_AUTO_LUT_HIGH))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                out = (plane - vmin) / (vmax - vmin)
            return np.clip(out, 0.0, 1.0)

        def _compose_merged(arr_hwc: np.ndarray, colors: List[tuple]) -> np.ndarray:
            """Merge: each channel in its assigned color (including the first). Uses percentile LUT so weak channels aren't black."""
            return _compose_merged_with_lut(arr_hwc, colors, {})

        def _compose_merged_with_lut(arr_hwc: np.ndarray, colors: List[tuple], per_ch_lut: dict) -> np.ndarray:
            """Merge: each channel contributes in its color (colors[0] for channel 0, etc.). Phase/first channel no longer forced to gray."""
            h, w = arr_hwc.shape[0], arr_hwc.shape[1]
            C = arr_hwc.shape[-1]
            comp_rgb = np.zeros((h, w, 3), dtype=np.float64)
            for c in range(min(C, len(colors))):
                vmin, vmax = per_ch_lut.get(c) or (None, None)
                plane = _normalize_plane_with_lut(arr_hwc[..., c], vmin, vmax)
                r, g, b = colors[c]
                comp_rgb[..., 0] += plane * r
                comp_rgb[..., 1] += plane * g
                comp_rgb[..., 2] += plane * b
            if comp_rgb.max() > 1e-9:
                comp_rgb = comp_rgb / comp_rgb.max()
            out = comp_rgb / max(comp_rgb.max(), 1e-9)
            return (out * 255).clip(0, 255).astype(np.uint8)

        def _select_z_plane_or_project(arr: np.ndarray, z_axis: int, *, z_max: bool) -> np.ndarray:
            """Take first z plane or max-project over z for montage display."""
            if arr.ndim <= z_axis:
                return arr
            if z_max:
                return np.nanmax(arr, axis=z_axis)
            return np.take(arr, 0, axis=z_axis)

        def _collapse_tile_axes(tile_arr: np.ndarray) -> np.ndarray:
            """Reduce stack-like singleton/leading axes to a PIL-compatible 2D or RGB tile."""
            a = np.asarray(tile_arr)
            while a.ndim > 3 and a.shape[0] == 1:
                a = a[0]
            while a.ndim > 3:
                a = a[0]
            if a.ndim == 3:
                if a.shape[-1] in (1, 2, 3, 4):
                    return a
                if a.shape[0] in (1, 2, 3, 4):
                    return a[0] if a.shape[0] == 1 else np.nanmax(a, axis=0)
                return a[0]
            return a

        def _hwc_from_channel_stack(
            stack_czyx: np.ndarray, ch_names: List[str], label_prefix: str, path: Path
        ) -> tuple[np.ndarray, List[str], List[tuple]]:
            """Project/sort channels from (C,Z,Y,X) or (C,Y,X) to (H,W,C)."""
            n_ch = int(stack_czyx.shape[0])
            order = sorted(
                range(n_ch),
                key=lambda i: (
                    0 if self._channel_sort_key(ch_names[i] if i < len(ch_names) else f"Ch{i+1}")[0] == 0 else 1,
                    i,
                ),
            )
            planes: List[np.ndarray] = []
            sorted_names: List[str] = []
            for i in order:
                ch = np.asarray(stack_czyx[i])
                nm = ch_names[i] if i < len(ch_names) else f"Ch{i+1}"
                sorted_names.append(nm)
                tile_lbl = self._montage_tile_key(f"{label_prefix} {nm}")
                zmax = self._montage_z_max_for_tile(tile_lbl)
                if ch.ndim == 3 and ch.shape[0] > 1:
                    plane = np.nanmax(ch, axis=0) if zmax else ch[0]
                elif ch.ndim == 3:
                    plane = ch[0]
                else:
                    plane = ch
                planes.append(np.asarray(plane))
            arr_hwc = np.stack(planes, axis=-1)
            colors = [self._channel_name_to_rgb(name) for name in sorted_names]
            return arr_hwc, sorted_names, colors

        def _nd2_axis_index(
            axis_keys: tuple[str, ...],
            *,
            t: int = 0,
            c: int = 0,
            z: int = 0,
            position: int = 0,
        ) -> tuple:
            idx: list[Any] = []
            for key in axis_keys:
                if key == "T":
                    idx.append(int(t))
                elif key == "C":
                    idx.append(int(c))
                elif key == "Z":
                    idx.append(int(z))
                elif key == "P":
                    idx.append(int(position))
                elif key in {"Y", "X", "S"}:
                    idx.append(slice(None))
                else:
                    idx.append(0)
            return tuple(idx)

        def _nd2_to_czyx(
            arr: np.ndarray,
            axis_keys: tuple[str, ...],
            *,
            frame_index: int,
            n_channels: int,
            n_z: int,
        ) -> np.ndarray:
            """Convert arbitrary ND2 axis order to (C, Z, Y, X)."""
            if not axis_keys or len(axis_keys) != arr.ndim or "Y" not in axis_keys or "X" not in axis_keys:
                raise ValueError("ND2 axis keys do not match array shape")
            t_idx = min(max(0, int(frame_index)), int(arr.shape[axis_keys.index("T")]) - 1) if "T" in axis_keys else 0
            c_count = int(arr.shape[axis_keys.index("C")]) if "C" in axis_keys else max(1, int(n_channels))
            z_count = int(arr.shape[axis_keys.index("Z")]) if "Z" in axis_keys else max(1, int(n_z))
            c_count = max(1, min(c_count, max(1, int(n_channels))))
            z_count = max(1, z_count)
            channel_stacks: list[np.ndarray] = []
            for c_idx in range(c_count):
                z_planes: list[np.ndarray] = []
                for z_idx in range(z_count):
                    plane = np.asarray(
                        arr[
                            _nd2_axis_index(
                                axis_keys,
                                t=t_idx,
                                c=c_idx,
                                z=z_idx,
                            )
                        ]
                    )
                    plane = np.squeeze(plane)
                    if plane.ndim > 2:
                        plane = plane.reshape(-1, plane.shape[-2], plane.shape[-1])[0]
                    z_planes.append(plane)
                channel_stacks.append(np.stack(z_planes, axis=0))
            return np.stack(channel_stacks, axis=0)

        def load_nd2_or_multichannel_tiff(
            path: Path,
            frame_index: int = 0,
            *,
            label_prefix: str = "",
            channel_count_hint: Optional[int] = None,
        ) -> Optional[tuple[np.ndarray, List[str], List[tuple], int]]:
            """Load first frame from ND2 or multi-channel TIFF. Return (arr (H,W,C), channel_names, colors, n_timepoints) with channels sorted phase first. n_timepoints is 1 for TIFF; for ND2 it is sizes['T']."""
            try:
                suf = path.suffix.lower()
                meta = read_metadata_from_path(path)
                nC = int(channel_count_hint or meta.get("channel_count") or 1)
                names = meta.get("channel_names") or [f"Ch{i+1}" for i in range(nC)]
                nT, nZ = 1, 1
                nC_file = int(nC)
                if suf == ".nd2":
                    import nd2
                    with nd2.ND2File(path) as r:
                        arr = np.ascontiguousarray(r.asarray())
                        sizes = getattr(r, "sizes", None) or {}
                        axis_keys = tuple(sizes.keys()) if sizes else ()
                    nT = int(sizes.get("T", 1))
                    nZ = int(sizes.get("Z", 1))
                    nC_file = int(sizes.get("C", 1))
                    prefix = label_prefix or path.stem
                    if axis_keys and len(axis_keys) == arr.ndim and "C" in axis_keys:
                        stack = _nd2_to_czyx(
                            arr,
                            axis_keys,
                            frame_index=frame_index,
                            n_channels=nC_file,
                            n_z=nZ,
                        )
                        arr_hwc, names, colors = _hwc_from_channel_stack(
                            stack, names[: int(stack.shape[0])], prefix, path
                        )
                        return np.asarray(arr_hwc), names, colors, nT

                    # Legacy fallback for older nd2 builds without axis keys.
                    if arr.ndim == 5:
                        stack = arr[frame_index] if nT > 1 else arr[0]  # (C, Z, Y, X)
                        arr_hwc, names, colors = _hwc_from_channel_stack(
                            stack, names[: int(stack.shape[0])], prefix, path
                        )
                        return np.asarray(arr_hwc), names, colors, nT
                    elif arr.ndim == 4:
                        if nT > 1:
                            stack = arr[frame_index]
                            if stack.ndim == 2:
                                stack = stack[np.newaxis, ...]
                            elif nC_file <= 1 and nZ > 1:
                                tile_lbl = self._montage_tile_key(prefix)
                                zmax = self._montage_z_max_for_tile(tile_lbl)
                                plane = _select_z_plane_or_project(stack, 0, z_max=zmax)
                                stack = plane[np.newaxis, ...]
                        elif nC_file > 1 and nZ > 1 and arr.shape[0] == nC_file:
                            arr_hwc, names, colors = _hwc_from_channel_stack(
                                arr, names[: nC_file], prefix, path
                            )
                            return np.asarray(arr_hwc), names, colors, nT
                        elif nC_file <= 1 and nZ > 1:
                            tile_lbl = self._montage_tile_key(prefix)
                            zmax = self._montage_z_max_for_tile(tile_lbl)
                            plane = _select_z_plane_or_project(arr, 0, z_max=zmax)
                            arr = plane[..., np.newaxis]
                        elif nC_file > 1 and arr.shape[0] == nC_file:
                            arr_hwc, names, colors = _hwc_from_channel_stack(
                                arr, names[: nC_file], prefix, path
                            )
                            return np.asarray(arr_hwc), names, colors, nT
                        else:
                            arr = arr[0] if arr.shape[0] == 1 else arr
                    elif arr.ndim == 3 and nT > 1:
                        arr = arr[frame_index]
                        arr = arr[..., np.newaxis]
                    elif arr.ndim == 3 and nC_file <= 1 and nZ > 1:
                        tile_lbl = self._montage_tile_key(prefix)
                        zmax = self._montage_z_max_for_tile(tile_lbl)
                        plane = _select_z_plane_or_project(arr, 0, z_max=zmax)
                        arr = plane[..., np.newaxis]
                    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4):
                        arr = np.transpose(arr, (1, 2, 0))
                    elif arr.ndim == 3 and arr.shape[-1] not in (1, 2, 3, 4):
                        arr = np.transpose(arr, (1, 2, 0))
                else:
                    import tifffile
                    with tifffile.TiffFile(str(path)) as tif:
                        arr = tif.asarray()
                        series_axes = ""
                        try:
                            if getattr(tif, "series", None) and len(tif.series) > 0:
                                series_axes = str(getattr(tif.series[0], "axes", None) or "")
                        except Exception:
                            series_axes = ""
                    arr = normalize_tiff_array_to_tcyx(
                        np.asarray(arr),
                        num_time=int(meta.get("frame_count") or 1),
                        num_channels=int(channel_count_hint or meta.get("channel_count") or nC),
                        num_z=int(meta.get("z_count") or 1),
                        series_axes=series_axes,
                    )
                    prefix = label_prefix or path.stem
                    if arr.ndim == 5:
                        t_idx = min(max(0, int(frame_index)), arr.shape[0] - 1)
                        stack = arr[t_idx]
                        arr_hwc, names, colors = _hwc_from_channel_stack(
                            stack, names[: int(stack.shape[0])], prefix, path
                        )
                        return np.asarray(arr_hwc), names, colors, nT
                    elif arr.ndim == 4:
                        t_idx = min(max(0, int(frame_index)), arr.shape[0] - 1)
                        stack = arr[t_idx]
                        if stack.ndim == 3 and stack.shape[0] in (1, 2, 3, 4, 5, 6):
                            arr_hwc, names, colors = _hwc_from_channel_stack(
                                stack, names[: int(stack.shape[0])], prefix, path
                            )
                            return np.asarray(arr_hwc), names, colors, nT
                        arr = stack
                    elif arr.ndim == 2:
                        arr = arr[:, :, np.newaxis]
                    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4):
                        arr = np.transpose(arr, (1, 2, 0))
                if arr.shape[-1] == 2:
                    arr = np.concatenate([arr, np.zeros((*arr.shape[:2], 1), dtype=arr.dtype)], axis=-1)
                nCh = arr.shape[-1]
                # Use file channel count when we have multi-channel ND2 so names/colors match
                if suf == ".nd2" and nC_file > 1 and len(names) < nCh:
                    names = (names + [f"Ch{i+1}" for i in range(len(names), nCh)])[:nCh]
                order = sorted(range(nCh), key=lambda i: self._channel_sort_key(names[i] if i < len(names) else f"Ch{i+1}"))
                arr = np.asarray(arr[..., order])
                names = [names[i] if i < len(names) else f"Ch{i+1}" for i in order]
                colors = self._get_channel_colors_for_path(path, nCh)
                colors = [colors[i] for i in order]
                return np.asarray(arr), names, colors, nT
            except Exception:
                return None

        def load_single_plane(path: Path, frame_index: int = 0, *, z_max: Optional[bool] = None) -> Optional[np.ndarray]:
            """Load one 2D image (first frame/page)."""
            if z_max is None:
                z_max = z_project_max
            try:
                if path.suffix.lower() in (".tif", ".tiff"):
                    import tifffile
                    meta = read_metadata_from_path(path)
                    with tifffile.TiffFile(str(path)) as tif:
                        arr = tif.asarray()
                        series_axes = ""
                        try:
                            if getattr(tif, "series", None) and len(tif.series) > 0:
                                series_axes = str(getattr(tif.series[0], "axes", None) or "")
                        except Exception:
                            series_axes = ""
                    arr = np.asarray(arr)
                    if arr.ndim == 2:
                        pass
                    elif arr.ndim == 3 and arr.shape[-1] in (1, 2, 3, 4):
                        arr = arr[..., 0]
                    else:
                        tcyx = normalize_tiff_array_to_tcyx(
                            arr,
                            num_time=int(meta.get("frame_count") or 1),
                            num_channels=int(meta.get("channel_count") or 1),
                            num_z=int(meta.get("z_count") or 1),
                            series_axes=series_axes,
                        )
                        if tcyx.ndim == 5:
                            t_idx = min(max(0, int(frame_index)), tcyx.shape[0] - 1)
                            arr = _select_z_plane_or_project(tcyx[t_idx, 0], 0, z_max=bool(z_max))
                        elif tcyx.ndim == 4:
                            t_idx = min(max(0, int(frame_index)), tcyx.shape[0] - 1)
                            arr = tcyx[t_idx, 0]
                        elif tcyx.ndim == 3:
                            arr = tcyx[0]
                else:
                    arr = safe_imread(str(path))
                    if arr.ndim >= 3 and arr.shape[0] <= 4:
                        arr = arr[0]
                    arr = np.asarray(arr)
                if arr.ndim > 2:
                    arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
                return np.asarray(arr)
            except Exception:
                return None

        tiles: List[tuple] = []  # (arr or None, label, color_or_None)
        tile_meta: List[dict] = []
        all_arrays_for_global: List[np.ndarray] = []
        overlay_groups: Dict[str, dict] = {}

        for ds in datasets:
            ds_type = ds["type"]
            base_label = ds.get("base", Path(ds["path"]).stem if "path" in ds else "?")

            if ds_type in ("nd2", "nd2_like"):
                path = ds["path"]
                overlay_group_key = f"{path.stem}|{ds_type}"
                if overlay_enabled and overlay_group_key not in overlay_groups:
                    overlay_groups[overlay_group_key] = self._prepare_montage_mask_overlay(
                        base_name=path.stem,
                        channel_paths=[path],
                        frame_index=overlay_frame,
                        mask_override_path=overlay_mask_override,
                    )
                result = load_nd2_or_multichannel_tiff(
                    path,
                    0,
                    label_prefix=base_label,
                    channel_count_hint=ds.get("channel_count"),
                )
                if result is None:
                    tiles.append((None, base_label, None))
                    tile_meta.append({})
                    continue
                arr_hwc, ch_names, colors, n_timepoints = result
                nCh = arr_hwc.shape[-1] if arr_hwc.ndim == 3 else 1
                # Single channel but multiple timepoints → one tile per frame (real montage)
                if nCh == 1 and n_timepoints > 1 and path.suffix.lower() == ".nd2":
                    try:
                        import nd2
                        with nd2.ND2File(path) as r:
                            arr_full = np.ascontiguousarray(r.asarray())
                        # arr_full is (T, Y, X) or (T, C, Y, X)
                        if arr_full.ndim == 3:
                            frames = arr_full  # (T, Y, X)
                        else:
                            frames = arr_full[:, 0, :, :]  # (T, Y, X)
                        n_show = min(frames.shape[0], 30)  # cap at 30 frames
                        gray = (1.0, 1.0, 1.0)
                        for t in range(n_show):
                            plane = np.asarray(frames[t])
                            tiles.append((plane, f"{base_label} t{t}", gray))
                            tile_meta.append({})
                            if use_global_lut:
                                all_arrays_for_global.append(np.asarray(plane, dtype=np.float64))
                    except Exception:
                        single = arr_hwc[..., 0] if arr_hwc.ndim == 3 else arr_hwc
                        col = self._get_channel_colors_for_path(path, 1, entry_index=0)[0]
                        tiles.append((single, base_label, col))
                        ch_norm = self._normalize_channel_token(ch_names[0] if ch_names else "FI")
                        if overlay_enabled:
                            tile_meta.append(
                                {
                                    "kind": "channel",
                                    "overlay_group": overlay_group_key,
                                    "channel_norm": ch_norm,
                                }
                            )
                        else:
                            tile_meta.append({"kind": "channel", "channel_norm": ch_norm})
                        if use_global_lut:
                            all_arrays_for_global.append(np.asarray(single, dtype=np.float64))
                    continue
                if not use_color_merge or nCh < 2:
                    single = arr_hwc[..., 0] if arr_hwc.ndim == 3 else arr_hwc
                    col = self._get_channel_colors_for_path(path, 1, entry_index=0)[0]
                    tiles.append((single, base_label, col))
                    ch_norm = self._normalize_channel_token(ch_names[0] if ch_names else "FI")
                    if overlay_enabled:
                        tile_meta.append(
                            {
                                "kind": "channel",
                                "overlay_group": overlay_group_key,
                                "channel_norm": ch_norm,
                            }
                        )
                    else:
                        tile_meta.append({"kind": "channel", "channel_norm": ch_norm})
                    if use_global_lut:
                        all_arrays_for_global.append(np.asarray(single, dtype=np.float64))
                    continue
                group_channel_tile_indices: List[int] = []
                group_channel_norms: List[str] = []
                for c in range(arr_hwc.shape[-1]):
                    plane = np.asarray(arr_hwc[..., c])
                    label = ch_names[c] if c < len(ch_names) else f"Ch{c+1}"
                    tile_idx = len(tiles)
                    tiles.append((plane, f"{base_label} {label}", colors[c]))
                    ch_norm = self._normalize_channel_token(label)
                    if overlay_enabled:
                        group_channel_tile_indices.append(tile_idx)
                        group_channel_norms.append(ch_norm)
                        tile_meta.append(
                            {
                                "kind": "channel",
                                "overlay_group": overlay_group_key,
                                "channel_norm": ch_norm,
                            }
                        )
                    else:
                        group_channel_tile_indices.append(tile_idx)
                        group_channel_norms.append(ch_norm)
                        tile_meta.append({"kind": "channel", "channel_norm": ch_norm})
                    if use_global_lut:
                        all_arrays_for_global.append(np.asarray(plane, dtype=np.float64))
                merged = _compose_merged(arr_hwc, colors)
                tiles.append((merged, f"{base_label} Merged", None))
                if overlay_enabled:
                    tile_meta.append(
                        {
                            "kind": "merged",
                            "overlay_group": overlay_group_key,
                            "channel_tile_indices": group_channel_tile_indices,
                            "channel_norms": group_channel_norms,
                        }
                    )
                else:
                    tile_meta.append({})
                continue

            if ds_type == "tiff_group":
                base = ds["base"]
                channel_list: List[tuple[Path, str]] = ds["channels"]
                overlay_group_key = f"{base}|tiff_group"
                channel_paths_for_group = [p for p, _ in channel_list]
                if overlay_enabled and overlay_group_key not in overlay_groups:
                    overlay_groups[overlay_group_key] = self._prepare_montage_mask_overlay(
                        base_name=base,
                        channel_paths=channel_paths_for_group,
                        frame_index=overlay_frame,
                        mask_override_path=overlay_mask_override,
                    )
                loaded_channels: List[tuple[np.ndarray, str, tuple]] = []
                for path, ch_name in channel_list:
                    tile_lbl = self._montage_tile_key(f"{base} {ch_name}")
                    plane = load_single_plane(
                        path, 0, z_max=self._montage_z_max_for_tile(tile_lbl)
                    )
                    if plane is not None:
                        color = self._channel_name_to_rgb(ch_name)
                        loaded_channels.append((plane, ch_name, color))
                        if use_global_lut:
                            all_arrays_for_global.append(np.asarray(plane, dtype=np.float64))
                group_channel_tile_indices: List[int] = []
                group_channel_norms: List[str] = []
                for plane, ch_name, color in loaded_channels:
                    tile_idx = len(tiles)
                    tiles.append((plane, f"{base} {ch_name}", color))
                    ch_norm = self._normalize_channel_token(ch_name)
                    if overlay_enabled:
                        group_channel_tile_indices.append(tile_idx)
                        group_channel_norms.append(ch_norm)
                        tile_meta.append(
                            {
                                "kind": "channel",
                                "overlay_group": overlay_group_key,
                                "channel_norm": ch_norm,
                            }
                        )
                    else:
                        group_channel_tile_indices.append(tile_idx)
                        group_channel_norms.append(ch_norm)
                        tile_meta.append({"kind": "channel", "channel_norm": ch_norm})
                if len(loaded_channels) > 1:
                    planes = [p for p, _ch, _c in loaded_channels]
                    color_list = [c for _p, _ch, c in loaded_channels]
                    h = max(p.shape[0] for p in planes)
                    w = max(p.shape[1] for p in planes)
                    resized = []
                    for p in planes:
                        if p.shape[0] == h and p.shape[1] == w:
                            resized.append(np.asarray(p, dtype=np.float64))
                        else:
                            # Resize to common size
                            p_u8 = ((p.astype(np.float64) - np.nanmin(p)) / max(np.nanmax(p) - np.nanmin(p), 1e-9) * 255).clip(0, 255).astype(np.uint8)
                            img = Image.fromarray(p_u8).resize((w, h), Image.Resampling.LANCZOS)
                            resized.append(np.array(img).astype(np.float64))
                    arr_hwc = np.stack(resized, axis=-1)
                    merged = _compose_merged(arr_hwc, color_list)
                    tiles.append((merged, f"{base} Merged", None))
                    if overlay_enabled:
                        tile_meta.append(
                            {
                                "kind": "merged",
                                "overlay_group": overlay_group_key,
                                "channel_tile_indices": group_channel_tile_indices,
                                "channel_norms": group_channel_norms,
                            }
                        )
                    else:
                        tile_meta.append({})
                continue

            if ds_type == "single_tiff":
                path = ds["path"]
                arr = load_single_plane(
                    path, 0, z_max=self._montage_z_max_for_tile(self._montage_tile_key(base_label))
                )
                if arr is not None:
                    tiles.append((arr, base_label, (1.0, 1.0, 1.0)))
                    tile_meta.append({})
                    if use_global_lut:
                        all_arrays_for_global.append(np.asarray(arr, dtype=np.float64))
                else:
                    tiles.append((None, base_label, None))
                    tile_meta.append({})
                continue

        def _apply_tile_order(tile_list: List[tuple], meta_list: List[dict]) -> tuple[List[tuple], List[dict]]:
            order = list(getattr(self, "montage_tile_order", []) or [])
            if not order:
                return tile_list, meta_list
            rank = {key: i for i, key in enumerate(order)}
            indexed = list(enumerate(zip(tile_list, meta_list)))
            indexed.sort(
                key=lambda item: (
                    rank.get(self._montage_tile_key(item[1][0][1]), len(rank) + item[0]),
                    item[0],
                )
            )
            return [pair[0] for _i, pair in indexed], [pair[1] for _i, pair in indexed]

        all_tiles = list(tiles)
        all_tile_meta = list(tile_meta)
        all_tiles, all_tile_meta = _apply_tile_order(all_tiles, all_tile_meta)

        excluded_keys = set(getattr(self, "montage_excluded_tile_keys", set()) or set())
        render_tiles: List[tuple] = []
        render_meta: List[dict] = []
        for i, tile in enumerate(all_tiles):
            lbl = tile[1] if len(tile) > 1 else ""
            if self._montage_tile_key(lbl) in excluded_keys:
                continue
            render_tiles.append(tile)
            render_meta.append(all_tile_meta[i] if i < len(all_tile_meta) else {})
        tiles = render_tiles
        tile_meta = render_meta

        # Global LUT (only when no per_tile_lut): use percentiles across all tiles
        gmin, gmax = None, None
        if use_global_lut and all_arrays_for_global and not per_tile_lut:
            all_pixels = np.concatenate([a.ravel() for a in all_arrays_for_global])
            all_pixels = all_pixels[np.isfinite(all_pixels)]
            if all_pixels.size > 0:
                gmin = float(np.percentile(all_pixels, self.MONTAGE_AUTO_LUT_LOW))
                gmax = float(np.percentile(all_pixels, self.MONTAGE_AUTO_LUT_HIGH))
            if gmin is None:
                gmin, gmax = 0.0, 255.0
            if gmax <= gmin:
                gmax = gmin + 1.0

        n_tiles = len(tiles)
        if n_tiles == 0:
            return None
        if rows <= 0 and cols <= 0:
            cols = int(np.ceil(np.sqrt(n_tiles)))
            rows = max(1, (n_tiles + cols - 1) // cols)
        elif rows <= 0:
            rows = max(1, (n_tiles + cols - 1) // cols)
        elif cols <= 0:
            cols = max(1, (n_tiles + rows - 1) // rows)

        pil_images = []
        rendered_tiles: List[tuple] = []
        rendered_meta: List[dict] = []
        max_w, max_h = 0, 0
        placeholder_size = 100

        def _lut_from_percentile(tile_arr: np.ndarray, tile_key: str, tile_idx: int, tile_label: str) -> tuple[Optional[float], Optional[float]]:
            if tile_label.endswith(" Merged"):
                return (None, None)
            meta = tile_meta[tile_idx] if tile_idx < len(tile_meta) else {}
            pct = self._montage_lookup_percentile_override_for_meta(
                tile_key,
                tile_idx,
                meta,
                overrides=per_tile_percentile_override,
            )
            if pct is None:
                return (None, None)
            p_low, p_high = pct
            p_low = float(max(0.0, min(99.9, p_low)))
            p_high = float(max(p_low + 0.1, min(100.0, p_high)))
            plane = np.asarray(tile_arr[..., 0] if tile_arr.ndim == 3 and tile_arr.shape[-1] == 1 else tile_arr, dtype=np.float64)
            valid = plane[np.isfinite(plane)]
            if valid.size == 0:
                return (None, None)
            vmin = float(np.percentile(valid, p_low))
            vmax = float(np.percentile(valid, p_high))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                return (None, None)
            return (vmin, vmax)

        for idx, (arr, label, color_or_none) in enumerate(tiles):
            tile_key = self._montage_tile_key(label)
            meta = tile_meta[idx] if idx < len(tile_meta) else {}
            if arr is None:
                placeholder = np.full((placeholder_size, placeholder_size, 3), 128, dtype=np.uint8)
                pil_images.append((Image.fromarray(placeholder), label + "?"))
                rendered_tiles.append((placeholder, label + "?", None))
                rendered_meta.append(meta)
                max_w = max(max_w, placeholder_size)
                max_h = max(max_h, placeholder_size)
                continue
            arr = _collapse_tile_axes(np.asarray(arr))
            color_ov = self._montage_lookup_color_override_for_meta(
                tile_key,
                idx,
                meta,
                None,
                overrides=per_tile_color_override,
            )
            if color_ov is not None and color_or_none is not None:
                color_or_none = color_ov
            # Rebuild merged tile from preceding channel tiles when user adjusted color and/or LUT on any of them
            has_lut = bool(per_tile_lut) and any(
                self._montage_lookup_lut_for_meta(
                    per_tile_lut,
                    self._montage_tile_key(tiles[i][1]),
                    i,
                    tile_meta[i] if i < len(tile_meta) else {},
                ) is not None
                for i in range(idx)
            )
            has_color_override = any(
                self._montage_lookup_color_override_for_meta(
                    self._montage_tile_key(tiles[i][1]),
                    i,
                    tile_meta[i] if i < len(tile_meta) else {},
                    None,
                    overrides=per_tile_color_override,
                ) is not None
                for i in range(idx)
            )
            has_pct_override = any(
                self._montage_lookup_percentile_override_for_meta(
                    self._montage_tile_key(tiles[i][1]),
                    i,
                    tile_meta[i] if i < len(tile_meta) else {},
                    overrides=per_tile_percentile_override,
                ) is not None
                for i in range(idx)
            )
            if label.endswith(" Merged") and color_or_none is None and idx > 0 and (has_lut or has_color_override or has_pct_override):
                try:
                    ch_arrays = [np.asarray(tiles[i][0]).squeeze() for i in range(idx)]
                    if all(a.ndim == 2 for a in ch_arrays) and len(ch_arrays) >= 1:
                        h_max = max(a.shape[0] for a in ch_arrays)
                        w_max = max(a.shape[1] for a in ch_arrays)
                        resized = []
                        for a in ch_arrays:
                            if a.shape[0] == h_max and a.shape[1] == w_max:
                                resized.append(np.asarray(a, dtype=np.float64))
                            else:
                                a_u8 = ((a.astype(np.float64) - np.nanmin(a)) / max(np.nanmax(a) - np.nanmin(a), 1e-9) * 255).clip(0, 255).astype(np.uint8)
                                resized.append(np.array(Image.fromarray(a_u8).resize((w_max, h_max), Image.Resampling.LANCZOS)).astype(np.float64))
                        arr_hwc = np.stack(resized, axis=-1)
                        colors_ch = []
                        for i in range(idx):
                            ch_lbl = self._montage_tile_key(tiles[i][1])
                            ch_meta = tile_meta[i] if i < len(tile_meta) else {}
                            ov = self._montage_lookup_color_override_for_meta(
                                ch_lbl,
                                i,
                                ch_meta,
                                tiles[i][2],
                                overrides=per_tile_color_override,
                            )
                            colors_ch.append(ov if ov is not None else tiles[i][2])
                        per_ch_lut = {}
                        for i in range(idx):
                            ch_lbl = self._montage_tile_key(tiles[i][1])
                            ch_meta = tile_meta[i] if i < len(tile_meta) else {}
                            lut_pair = self._montage_lookup_lut_for_meta(per_tile_lut, ch_lbl, i, ch_meta)
                            if lut_pair is not None:
                                per_ch_lut[i] = lut_pair
                        for i in range(min(idx, arr_hwc.shape[-1])):
                            ch_lbl = self._montage_tile_key(tiles[i][1])
                            ch_meta = tile_meta[i] if i < len(tile_meta) else {}
                            if i in per_ch_lut or self._montage_lookup_percentile_override_for_meta(
                                ch_lbl,
                                i,
                                ch_meta,
                                overrides=per_tile_percentile_override,
                            ) is None:
                                continue
                            pp_low, pp_high = self._montage_lookup_percentile_override_for_meta(
                                ch_lbl,
                                i,
                                ch_meta,
                                overrides=per_tile_percentile_override,
                            ) or (self.MONTAGE_AUTO_LUT_LOW, self.MONTAGE_AUTO_LUT_HIGH)
                            plane = np.asarray(arr_hwc[..., i], dtype=np.float64)
                            valid = plane[np.isfinite(plane)]
                            if valid.size == 0:
                                continue
                            vvmin = float(np.percentile(valid, max(0.0, min(99.9, float(pp_low)))))
                            vvhigh = max(float(pp_low) + 0.1, min(100.0, float(pp_high)))
                            vvmax = float(np.percentile(valid, vvhigh))
                            if np.isfinite(vvmin) and np.isfinite(vvmax) and vvmax > vvmin:
                                per_ch_lut[i] = (vvmin, vvmax)
                        arr = _compose_merged_with_lut(arr_hwc, colors_ch, per_ch_lut)
                except Exception:
                    pass
            # Resolve (vmin, vmax) for this tile: per_tile_lut override, else global, else percentile.
            # Merged tile has no LUT: it is fully determined by the channel tiles' LUTs and colors.
            vmin_override, vmax_override = None, None
            lut_pair = self._montage_lookup_lut_for_meta(per_tile_lut, tile_key, idx, meta)
            if lut_pair is not None and not label.endswith(" Merged"):
                vmin_override, vmax_override = lut_pair
                if vmax_override <= vmin_override:
                    vmax_override = vmin_override + 1e-9
            elif not label.endswith(" Merged"):
                vmin_override, vmax_override = _lut_from_percentile(np.asarray(arr), tile_key, idx, label)
            # Single-channel with semantic color (r,g,b)
            if color_or_none is not None and (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 1)):
                ch = arr[..., 0] if arr.ndim == 3 else arr
                ch = ch.astype(np.float64)
                if vmin_override is not None and vmax_override is not None:
                    ch = (ch - vmin_override) / max(vmax_override - vmin_override, 1e-9)
                elif use_global_lut and gmin is not None and gmax is not None:
                    ch = (ch - gmin) / max(gmax - gmin, 1e-9)
                else:
                    valid = ch[np.isfinite(ch)]
                    if valid.size > 0:
                        vmin = float(np.percentile(valid, self.MONTAGE_AUTO_LUT_LOW))
                        vmax = float(np.percentile(valid, self.MONTAGE_AUTO_LUT_HIGH))
                        if vmax <= vmin:
                            vmax = vmin + 1.0
                        ch = (ch - vmin) / (vmax - vmin)
                    ch = np.clip(ch, 0.0, 1.0)
                r, g, b = color_or_none
                arr = np.stack([ch * r, ch * g, ch * b], axis=-1)
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            # Pre-composed merged RGB (from _compose_merged)
            elif color_or_none is None and arr.ndim == 3 and arr.shape[-1] == 3 and arr.dtype == np.uint8:
                if vmin_override is not None and vmax_override is not None:
                    arr = ((arr.astype(np.float64) - vmin_override) / max(vmax_override - vmin_override, 1e-9) * 255).clip(0, 255).astype(np.uint8)
            # Multi-channel RGB from load_one (not expanded)
            elif use_color_merge and arr.ndim == 3 and arr.shape[-1] >= 2:
                if arr.shape[-1] == 2:
                    arr = np.concatenate([arr, np.zeros((*arr.shape[:2], 1), dtype=arr.dtype)], axis=-1)
                if arr.shape[-1] >= 3:
                    arr = arr[..., :3]
                if arr.dtype != np.uint8:
                    if vmin_override is not None and vmax_override is not None:
                        arr = ((arr.astype(np.float64) - vmin_override) / max(vmax_override - vmin_override, 1e-9) * 255).clip(0, 255).astype(np.uint8)
                    else:
                        per_ch = [(arr[..., c].astype(np.float64) - arr[..., c].min()) / max(arr[..., c].max() - arr[..., c].min(), 1e-9) for c in range(3)]
                        arr = (np.stack(per_ch, axis=-1) * 255).clip(0, 255).astype(np.uint8)
            else:
                if arr.ndim == 3 and arr.shape[-1] >= 3:
                    arr = arr.mean(axis=-1, keepdims=True)
                    arr = np.concatenate([arr, arr, arr], axis=-1)
                if arr.dtype != np.uint8:
                    if vmin_override is not None and vmax_override is not None:
                        arr = ((arr.astype(np.float64) - vmin_override) / max(vmax_override - vmin_override, 1e-9) * 255).clip(0, 255).astype(np.uint8)
                    elif use_global_lut and gmin is not None and gmax is not None:
                        arr = ((arr.astype(np.float64) - gmin) / max(gmax - gmin, 1e-9) * 255).clip(0, 255).astype(np.uint8)
                    else:
                        arr = ((arr.astype(np.float64) - arr.min()) / max(arr.max() - arr.min(), 1e-9) * 255).clip(0, 255).astype(np.uint8)
            if overlay_enabled and meta.get("overlay_group") in overlay_groups:
                arr = _collapse_tile_axes(np.asarray(arr))
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                elif arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=-1)
                arr = self._apply_overlay_to_tile(
                    tile_rgb=np.asarray(arr, dtype=np.uint8),
                    tile_meta=meta,
                    overlay_ctx=overlay_groups[meta["overlay_group"]],
                    tile_index=idx,
                    tiles=tiles,
                    per_tile_color_override=per_tile_color_override,
                )
            arr = _collapse_tile_axes(np.asarray(arr))
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            if arr.ndim != 2 and not (arr.ndim == 3 and arr.shape[-1] in (3, 4)):
                arr = np.squeeze(arr)
                arr = _collapse_tile_axes(np.asarray(arr))
            if arr.ndim != 2 and not (arr.ndim == 3 and arr.shape[-1] in (3, 4)):
                continue
            pil_images.append((Image.fromarray(arr), label))
            rendered_tiles.append((arr, label, color_or_none))
            rendered_meta.append(meta)
            max_w = max(max_w, arr.shape[1])
            max_h = max(max_h, arr.shape[0])
        if merged_only:
            merged_items = [
                (pil_item, tile, meta)
                for pil_item, tile, meta in zip(pil_images, rendered_tiles, rendered_meta)
                if str(pil_item[1]).endswith(" Merged")
            ]
            if not merged_items:
                return None
            pil_images = [item[0] for item in merged_items]
            rendered_tiles = [item[1] for item in merged_items]
            rendered_meta = [item[2] for item in merged_items]
            max_w = max(int(pil_img.size[0]) for pil_img, _label in pil_images)
            max_h = max(int(pil_img.size[1]) for pil_img, _label in pil_images)
            n_tiles = len(pil_images)
            if self._safe_spin_value(getattr(self, "montage_cols", None), 0) <= 0:
                cols = n_tiles
            else:
                cols = max(1, min(cols, n_tiles))
            rows = max(1, (n_tiles + cols - 1) // cols)
        if not pil_images:
            return None
        cell_w = max_w
        labels_below = show_labels and "below" in label_position.lower()
        cell_h = max_h + (font_size + 4) if labels_below else max_h
        out_w = border * 2 + cols * cell_w + (cols - 1) * spacing
        out_h = border * 2 + rows * cell_h + (rows - 1) * spacing
        font = None
        for path in ["/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"]:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        out_mode, out_fill = self._montage_border_fill()
        out_pil = Image.new(out_mode, (out_w, out_h), out_fill)
        draw = ImageDraw.ImageDraw(out_pil)
        tile_boxes: list[tuple[int, int, int, int]] = []
        for idx, (pil_img, stem) in enumerate(pil_images):
            r, c = idx // cols, idx % cols
            x0 = border + c * (cell_w + spacing)
            y0 = border + r * (cell_h + spacing)
            # Resize to fit cell (max_w x max_h)
            pil_resized = pil_img.resize((max_w, max_h), Image.Resampling.LANCZOS)
            if out_mode == "RGBA" and pil_resized.mode != "RGBA":
                pil_resized = pil_resized.convert("RGBA")
            out_pil.paste(pil_resized, (x0, y0))
            tile_boxes.append((int(x0), int(y0), int(max_w), int(max_h)))
            if show_labels:
                label_text = self._montage_short_label(stem)
                label_fill = self._montage_label_fill(label_color_name, rgba=(out_mode == "RGBA"))
                pos_lower = label_position.lower()
                if "inside" in pos_lower:
                    pad = 4
                    if "bottom" in pos_lower:
                        draw.text(
                            (x0 + pad, y0 + max_h - pad),
                            label_text,
                            fill=label_fill,
                            font=font,
                            anchor="lb",
                        )
                    else:
                        draw.text(
                            (x0 + pad, y0 + pad),
                            label_text,
                            fill=label_fill,
                            font=font,
                            anchor="lt",
                        )
                else:
                    ly = y0 + max_h + 2
                    draw.text((x0 + max_w // 2, ly), label_text, fill=label_fill, font=font, anchor="mt")
        out_arr = np.array(out_pil)
        if return_layout:
            layout = {
                "rows": rows,
                "cols": cols,
                "border": border,
                "spacing": spacing,
                "cell_w": cell_w,
                "cell_h": cell_h,
                "max_w": max_w,
                "max_h": max_h,
                "out_w": out_w,
                "out_h": out_h,
                "n_tiles": n_tiles,
                "tiles": rendered_tiles,
                "tile_meta": rendered_meta,
                "tile_labels": [label for _, label, _ in rendered_tiles],
                "tile_boxes": tile_boxes,
                "all_tiles": all_tiles,
                "all_tile_meta": all_tile_meta,
                "all_tile_labels": [label for _, label, _ in all_tiles],
            }
            return (out_arr, layout)
        return out_arr

    def _preview_montage(self) -> None:
        paths = self._get_montage_image_paths()
        if not paths:
            QMessageBox.warning(
                self, "Montage",
                "No images found. Enter one ND2 or TIFF file, or a folder."
            )
            return
        datasets = self._get_montage_datasets()
        if not datasets:
            QMessageBox.warning(
                self, "Montage",
                "No datasets to show. For a folder of TIFFs, use names like sample_phase.tif and sample_GFP.tif so channels can be grouped."
            )
            return
        result = self._build_montage_array(
            return_layout=True,
            per_tile_color_override=getattr(self, "montage_tile_color_overrides", None) or {},
            per_tile_percentile_override=getattr(self, "montage_tile_percentile_overrides", None) or {},
        )
        if result is None:
            QMessageBox.warning(self, "Montage", "Could not build montage (check image formats and that Pillow is installed).")
            return
        out, layout = result
        self._last_montage_layout = layout
        self._show_montage_preview_dialog(
            out,
            layout,
            embedded=getattr(self, "_module_preview_mount", None) is not None,
        )

    def _save_montage(self) -> None:
        paths = self._get_montage_image_paths()
        if not paths:
            QMessageBox.warning(
                self, "Montage",
                "No images found. Enter one ND2 or TIFF file, or a folder."
            )
            return
        if not self._get_montage_datasets():
            QMessageBox.warning(
                self, "Montage",
                "No datasets to include. For a folder of TIFFs, use names like sample_phase.tif and sample_GFP.tif."
            )
            return
        merged_only = self._montage_export_is_merged_only()
        result = self._compose_montage_for_export(merged_only=merged_only)
        if result is None:
            if merged_only:
                QMessageBox.warning(self, "Montage", "No merged tile available to export.")
            else:
                QMessageBox.warning(self, "Montage", "Could not build montage (check formats and Pillow).")
            return
        out, _layout = result
        default_dir = paths[0].parent
        fmt_text = self._montage_export_format()
        ext = self._montage_export_extension(fmt_text)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save montage", str(default_dir / f"{self._default_montage_output_stem(paths, merged_only=merged_only)}{ext}"),
            "PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg);;All (*)"
        )
        if not path:
            return
        try:
            out_path = Path(path)
            if not out_path.suffix:
                out_path = out_path.with_suffix(ext)
            self._save_montage_array_to_path(out, out_path, dpi=self._montage_export_dpi())
            self.parent_window.log_output(f"Montage saved: {out_path}\n")
            show_task_complete_dialog(
                self,
                title="Montage saved",
                message="Montage image saved.",
                output_dir=out_path.parent,
            )
        except Exception as e:
            QMessageBox.warning(self, "Montage", f"Save failed: {e}")

    def _show_montage_preview_dialog(
        self,
        initial_montage: np.ndarray,
        layout: dict,
        *,
        embedded: bool = False,
    ) -> None:
        """Non-modal montage preview: click a tile to select it, adjust LUT via histogram; Update, Edit colors, Save, Close."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        if embedded and getattr(self, "_module_preview_mount", None) is not None:
            self._close_embedded_post_preview()
            dlg = QWidget(self)
            dlg.setObjectName("EmbeddedMontagePreview")
            dlg.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        else:
            dlg = QDialog(self)
            dlg.setWindowTitle("Montage preview")
            dlg.setModal(False)
            dlg.setMinimumSize(900, 800)
        main_layout = QVBoxLayout(dlg)
        dlg._layout = layout
        self._last_montage_layout = layout
        dlg._per_tile_lut = {}
        dlg._focus_roi = None
        dlg._focus_drag_start = None
        dlg._current_displayed = self._apply_montage_scalebar_if_enabled(initial_montage.copy(), layout=layout)

        selected_label = QLabel("Click a tile to adjust LUT. Enable Focus inset, then right-drag a box to add a zoom callout. Middle-drag to pan, wheel to zoom.")
        selected_label.setWordWrap(True)
        selected_label.setStyleSheet("color: #aaa; font-size: 11px;")
        main_layout.addWidget(selected_label)

        image_canvas = ImageCanvas()
        image_canvas.setToolTip("Zoom: mouse wheel. Pan: left-click and drag (or middle-click and drag).")
        image_canvas.setMinimumHeight(320)
        image_canvas.setStyleSheet("background-color: #777777; border: 1px solid #555555;")
        image_canvas.enable_drag_selection = True
        image_canvas.click_button = Qt.MouseButton.LeftButton
        image_canvas.selection_button = Qt.MouseButton.RightButton  # so left-click selects tile, not drag
        image_canvas.pan_button = Qt.MouseButton.MiddleButton
        main_layout.addWidget(image_canvas, 1)

        layout_group = self._build_montage_layout_controls(dlg)
        layout_section = CollapsibleSection("Layout", layout_group, parent=dlg, expanded=False, compact=True)
        main_layout.addWidget(layout_section)
        scalebar_group = self._build_montage_scalebar_controls(dlg)
        scalebar_section = CollapsibleSection("Scale bar", scalebar_group, parent=dlg, expanded=False, compact=True)
        main_layout.addWidget(scalebar_section)
        focus_group = self._build_montage_focus_controls(dlg)
        focus_section = CollapsibleSection("Focus inset", focus_group, parent=dlg, expanded=False, compact=True)
        main_layout.addWidget(focus_section)
        labels_group = self._build_montage_label_controls(dlg)
        labels_section = CollapsibleSection("Labels", labels_group, parent=dlg, expanded=False, compact=True)
        main_layout.addWidget(labels_section)

        lut_group = QGroupBox("")
        lut_group.setMinimumHeight(0)
        lut_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lut_layout = QVBoxLayout()
        lut_layout.setContentsMargins(8, 8, 8, 6)
        lut_layout.setSpacing(4)
        montage_lut_canvas_height = 92 if embedded else LUT_HIST_CANVAS_HEIGHT_PX
        hist_fig = Figure(figsize=(LUT_HIST_FIGSIZE_INCHES[0], 1.35 if embedded else LUT_HIST_FIGSIZE_INCHES[1]))
        hist_canvas = FigureCanvas(hist_fig)
        hist_canvas.setMinimumHeight(montage_lut_canvas_height)
        hist_canvas.setMaximumHeight(montage_lut_canvas_height)
        hist_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        hist_ax = hist_fig.add_subplot(111)
        hist_fig.patch.set_facecolor("#2b2b2b")
        hist_ax.set_facecolor("#2b2b2b")
        hist_ax.tick_params(colors="white", labelsize=8 if embedded else 10)
        for spine in ["bottom", "top", "left", "right"]:
            hist_ax.spines[spine].set_color("white")
        hist_ax.xaxis.label.set_color("white")
        hist_ax.yaxis.label.set_color("white")
        apply_lut_histogram_figure_margins(hist_fig)
        if embedded:
            hist_fig.subplots_adjust(left=0.15, right=0.98, top=0.86, bottom=0.28)
        hist_canvas.setStyleSheet("background-color: #2b2b2b;")
        hist_canvas.setCursor(Qt.CursorShape.PointingHandCursor)
        lut_layout.addWidget(hist_canvas)
        min_max_row = QHBoxLayout()
        min_max_row.setSpacing(6)
        min_max_row.addWidget(QLabel("Min:", lut_group))
        min_spin = QDoubleSpinBox()
        min_spin.setRange(0, 65535)
        min_spin.setValue(0)
        min_spin.setDecimals(0)
        self._compact_preview_control(min_spin, 78)
        min_max_row.addWidget(min_spin)
        min_max_row.addWidget(QLabel("Max:", lut_group))
        max_spin = QDoubleSpinBox()
        max_spin.setRange(0, 65535)
        max_spin.setValue(65535)
        max_spin.setDecimals(0)
        self._compact_preview_control(max_spin, 78)
        min_max_row.addWidget(max_spin)
        auto_selected_btn = QPushButton("Auto selected")
        style_compact_action_button(auto_selected_btn, horizontal_padding=12)
        auto_selected_btn.setToolTip("Set the selected tile to the automatic robust LUT range.")
        min_max_row.addWidget(auto_selected_btn)
        auto_all_btn = QPushButton("Auto all")
        style_compact_action_button(auto_all_btn, horizontal_padding=12)
        auto_all_btn.setToolTip("Set automatic robust LUTs for all non-merged tiles.")
        min_max_row.addWidget(auto_all_btn)
        min_max_row.addStretch()
        lut_layout.addLayout(min_max_row)
        lut_group.setLayout(lut_layout)
        lut_section = CollapsibleSection("LUTs", lut_group, parent=dlg, expanded=False, compact=True)
        main_layout.addWidget(lut_section)

        selected_index: List[Optional[int]] = [None]
        current_vmin: List[Optional[float]] = [None]
        current_vmax: List[Optional[float]] = [None]
        hist_vmin_line: List[Any] = [None]
        hist_vmax_line: List[Any] = [None]
        hist_vspan: List[Any] = [None]
        dragging_line: List[Optional[str]] = [None]
        is_updating_lut: List[bool] = [False]

        def tile_at_xy(x: int, y: int) -> Optional[int]:
            layout = dlg._layout
            rows, cols = layout["rows"], layout["cols"]
            border = layout["border"]
            spacing = layout["spacing"]
            cell_w, cell_h = layout["cell_w"], layout["cell_h"]
            if x < border or y < border:
                return None
            c = (x - border) // (cell_w + spacing)
            r = (y - border) // (cell_h + spacing)
            if c < 0 or c >= cols or r < 0 or r >= rows:
                return None
            idx = r * cols + c
            if idx >= layout["n_tiles"]:
                return None
            return idx

        def get_tile_raw_for_hist(tile_idx: int) -> Optional[tuple[np.ndarray, np.dtype]]:
            """Return (flat float array, original dtype) for histogram. dtype used for full intensity range."""
            layout = dlg._layout
            tiles_list = layout["tiles"]
            if tile_idx < 0 or tile_idx >= len(tiles_list):
                return None
            arr, _label, _color = tiles_list[tile_idx]
            if arr is None:
                return None
            if arr.ndim == 2:
                return arr.astype(np.float64).ravel(), arr.dtype
            if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                if arr.shape[-1] == 1:
                    return arr[..., 0].astype(np.float64).ravel(), arr.dtype
                return arr.mean(axis=-1).astype(np.float64).ravel(), arr.dtype
            return None

        def full_range_for_dtype(dtype: np.dtype, data_min: float, data_max: float) -> tuple[float, float]:
            """Full possible intensity range for consistent histogram across images (e.g. 0–65535 for 16-bit).
            When dtype is uint8/uint16 use that range. When unknown (e.g. float), infer from data: if values
            fit in 0–255 or 0–65535 use that full range so montages can share the same scale; else use data range.
            """
            if dtype == np.uint8:
                return (0.0, 255.0)
            if dtype == np.uint16:
                return (0.0, 65535.0)
            # Unknown or float: infer from data so we still get a stable range when data is really 8/16-bit
            if data_max <= 255 and data_min >= 0:
                return (0.0, 255.0)
            if data_max <= 65535 and data_min >= 0:
                return (0.0, 65535.0)
            return (data_min, data_max)

        def auto_lut_for_tile(tile_idx: int) -> Optional[tuple[float, float]]:
            result = get_tile_raw_for_hist(tile_idx)
            if result is None:
                return None
            flat, _dtype = result
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                return None
            vmin = float(np.percentile(flat, self.MONTAGE_AUTO_LUT_LOW))
            vmax = float(np.percentile(flat, self.MONTAGE_AUTO_LUT_HIGH))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                return None
            return (vmin, vmax)

        hist_range: List[tuple[float, float]] = [(0.0, 1.0)]  # (hmin, hmax) for current tile; used by spinboxes and drag

        def update_histogram_for_tile(tile_idx: Optional[int]) -> None:
            hist_ax.clear()
            if tile_idx is None:
                lut_group.setTitle("")
                hist_ax.text(0.5, 0.5, "Click a tile to select", ha="center", va="center", color="white", fontsize=13, transform=hist_ax.transAxes)
                hist_ax.set_xlim(0, 1)
                hist_ax.set_ylim(0, 1)
                min_spin.setEnabled(False)
                max_spin.setEnabled(False)
                hist_canvas.draw()
                return
            labels = dlg._layout.get("tile_labels", [])
            title = labels[tile_idx] if tile_idx < len(labels) else f"Tile {tile_idx}"
            result = get_tile_raw_for_hist(tile_idx)
            if result is None:
                hist_ax.text(0.5, 0.5, "No data", ha="center", va="center", color="white", transform=hist_ax.transAxes)
                hist_canvas.draw()
                return
            flat, dtype = result
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                hist_canvas.draw()
                return
            fmin, fmax = float(np.min(flat)), float(np.max(flat))
            if fmax <= fmin:
                fmax = fmin + 1.0
            hmin, hmax = full_range_for_dtype(dtype, fmin, fmax)
            hist_range[0] = (hmin, hmax)
            n_bins = 256 if hmax - hmin <= 256 else min(1024, max(256, int(hmax - hmin + 1)))
            hist, bin_edges = np.histogram(flat, bins=n_bins, range=(hmin, hmax))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            log_counts = np.log1p(hist)
            if current_vmin[0] is not None and current_vmax[0] is not None:
                vmin = current_vmin[0]
                vmax = current_vmax[0]
            else:
                auto_pair = auto_lut_for_tile(tile_idx)
                vmin, vmax = auto_pair if auto_pair is not None else (
                    float(np.percentile(flat, self.MONTAGE_AUTO_LUT_LOW)),
                    float(np.percentile(flat, self.MONTAGE_AUTO_LUT_HIGH)),
                )
            vmin = max(hmin, min(vmin, hmax - 1e-9))
            vmax = min(hmax, max(vmax, hmin + 1e-9))
            if vmax <= vmin:
                vmax = vmin + 1.0
            current_vmin[0], current_vmax[0] = vmin, vmax
            hist_vspan[0] = hist_ax.axvspan(vmin, vmax, alpha=0.12, color="yellow", zorder=0)
            hist_ax.bar(bin_centers, log_counts, width=bin_edges[1] - bin_edges[0], color="#4A90E2", alpha=0.85, edgecolor="none", zorder=1)
            hist_vmin_line[0] = hist_ax.axvline(vmin, color="#FF6B6B", linewidth=2, linestyle="--", label=f"Min: {vmin:.1f}", zorder=2)
            hist_vmax_line[0] = hist_ax.axvline(vmax, color="#4ECDC4", linewidth=2, linestyle="--", label=f"Max: {vmax:.1f}", zorder=2)
            hist_ax.set_xlim(hmin, hmax)
            y_max = float(np.max(log_counts)) if log_counts.size > 0 else 1.0
            hist_ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)
            hist_ax.set_xlabel("Intensity", color="white", fontsize=10)
            hist_ax.set_ylabel("Log(Count+1)", color="white", fontsize=10)
            hist_ax.set_title(f"Selected: tile {tile_idx} — {title}", color="white", fontsize=10, pad=4)
            hist_ax.legend(loc="upper right", facecolor="#2b2b2b", edgecolor="white", labelcolor="white", fontsize=9, framealpha=0.9)
            hist_ax.grid(True, alpha=0.25, color="white")
            hist_ax.margins(x=0.02, y=0.08)
            apply_lut_histogram_figure_margins(hist_fig)
            hist_canvas.draw()
            # Sync min/max spinboxes to current range and limits (block signals to avoid feedback)
            min_spin.blockSignals(True)
            max_spin.blockSignals(True)
            min_spin.setRange(hmin, hmax)
            max_spin.setRange(hmin, hmax)
            min_spin.setValue(vmin)
            max_spin.setValue(vmax)
            min_spin.setDecimals(0 if dtype in (np.uint8, np.uint16) else 2)
            max_spin.setDecimals(0 if dtype in (np.uint8, np.uint16) else 2)
            min_spin.blockSignals(False)
            max_spin.blockSignals(False)
            min_spin.setEnabled(True)
            max_spin.setEnabled(True)

        def refresh_preview(reset_view: bool = True) -> None:
            result = self._build_montage_array(
                per_tile_lut=dlg._per_tile_lut,
                per_tile_color_override=getattr(self, "montage_tile_color_overrides", None) or {},
                per_tile_percentile_override=getattr(self, "montage_tile_percentile_overrides", None) or {},
                return_layout=True,
                merged_only=False,
            )
            if result is None:
                return
            out, new_layout = result
            dlg._layout = new_layout
            self._last_montage_layout = new_layout
            displayed = self._apply_montage_scalebar_if_enabled(out, layout=dlg._layout)
            displayed = self._apply_montage_focus_inset_if_enabled(displayed, roi=getattr(dlg, "_focus_roi", None))
            dlg._current_displayed = displayed
            dummy_mask = np.zeros(out.shape[:2], dtype=np.int32)
            img_scaled, _ = scale_for_preview(dlg._current_displayed, dummy_mask)
            pixmap = to_qpixmap(img_scaled)
            w, h = img_scaled.shape[1], img_scaled.shape[0]
            dlg._display_w, dlg._display_h = w, h
            image_canvas.set_pixmap(pixmap, (w, h), reset_view=reset_view)
            sync_focus_overlay()
            image_canvas.update()
            image_canvas.repaint()

        def refresh_preview_same_view(*_args) -> None:
            refresh_preview(reset_view=False)

        def compose_current_displayed() -> Optional[np.ndarray]:
            merged_only = self._montage_export_is_merged_only()
            result = self._compose_montage_for_export(
                per_tile_lut=dlg._per_tile_lut,
                focus_roi=getattr(dlg, "_focus_roi", None),
                merged_only=merged_only,
                source_layout=dlg._layout,
            )
            if result is None:
                return None
            displayed, export_layout = result
            if merged_only:
                dlg._export_layout = export_layout
            return displayed

        def sync_focus_overlay() -> None:
            if not focus_inset_enabled():
                image_canvas.enable_roi_editing = False
                image_canvas.clear_roi_overlays()
                return
            roi = getattr(dlg, "_focus_roi", None)
            if roi is None:
                image_canvas.enable_roi_editing = False
                image_canvas.clear_roi_overlays()
                return
            color_rgb = self._montage_focus_color_rgb(
                self._safe_combo_text(getattr(self, "montage_focus_color", None), "White")
            )
            line_width = self._safe_spin_value(getattr(self, "montage_focus_line_width", None), 5)
            image_canvas.enable_roi_editing = True
            image_canvas.set_editable_roi(
                tuple(int(v) for v in roi),
                color=QColor(*color_rgb),
                line_width=line_width,
                show_handles=True,
            )

        for widget in (
            getattr(self, "montage_rows", None),
            getattr(self, "montage_cols", None),
            getattr(self, "montage_spacing", None),
            getattr(self, "montage_border", None),
            getattr(self, "montage_label_font_size", None),
        ):
            if self._qt_widget_alive(widget):
                widget.valueChanged.connect(refresh_preview_same_view)
        show_labels_widget = getattr(self, "montage_show_labels", None)
        if self._qt_widget_alive(show_labels_widget):
            show_labels_widget.toggled.connect(refresh_preview_same_view)
        for widget in (
            getattr(self, "montage_order", None),
            getattr(self, "montage_z_projection", None),
            getattr(self, "montage_label_position", None),
            getattr(self, "montage_label_color", None),
        ):
            if self._qt_widget_alive(widget):
                widget.currentTextChanged.connect(refresh_preview_same_view)
        for widget in (
            getattr(self, "montage_sb_enable", None),
            getattr(self, "montage_sb_show_label", None),
            getattr(self, "montage_focus_enable", None),
        ):
            if self._qt_widget_alive(widget):
                widget.toggled.connect(refresh_preview_same_view)
        for widget in (
            getattr(self, "montage_sb_pixel_size", None),
            getattr(self, "montage_sb_length_um", None),
            getattr(self, "montage_sb_thickness", None),
            getattr(self, "montage_sb_font_size", None),
            getattr(self, "montage_focus_line_width", None),
        ):
            if self._qt_widget_alive(widget):
                widget.valueChanged.connect(refresh_preview_same_view)
        for widget in (
            getattr(self, "montage_border_color", None),
            getattr(self, "montage_sb_position", None),
            getattr(self, "montage_sb_color", None),
            getattr(self, "montage_focus_zoom", None),
            getattr(self, "montage_focus_position", None),
            getattr(self, "montage_focus_color", None),
        ):
            if self._qt_widget_alive(widget):
                widget.currentTextChanged.connect(refresh_preview_same_view)
        clear_focus_btn = getattr(self, "montage_focus_clear_btn", None)
        if self._qt_widget_alive(clear_focus_btn):
            clear_focus_btn.clicked.connect(lambda *_args: (setattr(dlg, "_focus_roi", None), sync_focus_overlay(), refresh_preview_same_view()))

        def on_roi_changed(x0: int, y0: int, x1: int, y1: int) -> None:
            dlg._focus_roi = (int(x0), int(y0), int(x1), int(y1))

        def on_roi_edit_finished(x0: int, y0: int, x1: int, y1: int) -> None:
            dlg._focus_roi = (int(x0), int(y0), int(x1), int(y1))
            sync_focus_overlay()
            refresh_preview_same_view()

        image_canvas.roi_changed.connect(on_roi_changed)
        image_canvas.roi_edit_finished.connect(on_roi_edit_finished)

        def display_to_montage_xy(x: int, y: int) -> tuple[int, int]:
            layout = dlg._layout
            ow, oh = layout.get("out_w"), layout.get("out_h")
            dw, dh = getattr(dlg, "_display_w", None), getattr(dlg, "_display_h", None)
            if ow is not None and oh is not None and dw and dh:
                x = int(round(x * ow / dw))
                y = int(round(y * oh / dh))
            return (
                max(0, min(int(x), int((ow or x + 1) - 1))),
                max(0, min(int(y), int((oh or y + 1) - 1))),
            )

        def on_canvas_click(x: int, y: int) -> None:
            layout = dlg._layout
            x, y = display_to_montage_xy(x, y)
            idx = tile_at_xy(x, y)
            if idx is None:
                return
            selected_index[0] = idx
            labels = layout.get("tile_labels", [])
            label = labels[idx] if idx < len(labels) else f"Tile {idx}"
            is_merged_tile = label.endswith(" Merged")
            if is_merged_tile:
                lut_group.setEnabled(False)
                selected_label.setText(f"Selected: tile {idx} — {label}. The merged image has no separate LUT; it is derived from the channel tiles above. Adjust those tiles to change the merge.")
                current_vmin[0], current_vmax[0] = None, None
            else:
                lut_group.setEnabled(True)
                tile_key = self._montage_tile_key(label)
                lut_pair = dlg._per_tile_lut.get(tile_key)
                if lut_pair is not None:
                    current_vmin[0], current_vmax[0] = lut_pair
                else:
                    current_vmin[0], current_vmax[0] = None, None
                selected_label.setText(f"Selected: tile {idx} — {label}. Drag the histogram lines to set display range.")
            update_histogram_for_tile(idx)

        def focus_inset_enabled() -> bool:
            return self._safe_checked(getattr(self, "montage_focus_enable", None), default=False)

        def on_focus_drag_started(x: int, y: int) -> None:
            if not focus_inset_enabled():
                return
            dlg._focus_drag_start = display_to_montage_xy(x, y)

        def on_focus_drag_updated(x: int, y: int) -> None:
            if not focus_inset_enabled() or getattr(dlg, "_focus_drag_start", None) is None:
                return
            x0, y0 = dlg._focus_drag_start
            x1, y1 = display_to_montage_xy(x, y)
            roi = self._normalize_montage_roi(
                x0, y0, x1, y1, int(dlg._layout.get("out_w", 0)), int(dlg._layout.get("out_h", 0))
            )
            if roi is not None:
                image_canvas.set_pending_roi(roi)

        def on_focus_drag_finished(x: int, y: int) -> None:
            if not focus_inset_enabled() or getattr(dlg, "_focus_drag_start", None) is None:
                dlg._focus_drag_start = None
                image_canvas.set_pending_roi(None)
                return
            x0, y0 = dlg._focus_drag_start
            x1, y1 = display_to_montage_xy(x, y)
            dlg._focus_drag_start = None
            image_canvas.set_pending_roi(None)
            roi = self._normalize_montage_roi(
                x0, y0, x1, y1, int(dlg._layout.get("out_w", 0)), int(dlg._layout.get("out_h", 0))
            )
            if roi is None:
                dlg._focus_roi = None
                selected_label.setText("Focus inset selection was too small. Right-drag a larger box on the preview.")
            else:
                dlg._focus_roi = roi
                selected_label.setText("Focus inset set. Drag handles to resize, drag inside to move, or right-drag to replace.")
            sync_focus_overlay()
            refresh_preview_same_view()

        def apply_min_max_from_spinboxes() -> None:
            if selected_index[0] is None:
                return
            hmin, hmax = hist_range[0]
            vmin = min_spin.value()
            vmax = max_spin.value()
            vmin = max(hmin, min(vmin, hmax - 1e-9))
            vmax = min(hmax, max(vmax, hmin + 1e-9))
            if vmax <= vmin:
                vmax = vmin + 1e-9
            current_vmin[0], current_vmax[0] = vmin, vmax
            sel = selected_index[0]
            if sel is not None and sel < len(dlg._layout.get("tile_labels", [])):
                dlg._per_tile_lut[self._montage_tile_key(dlg._layout["tile_labels"][sel])] = (vmin, vmax)
            min_spin.setValue(vmin)
            max_spin.setValue(vmax)
            if hist_vmin_line[0] is not None:
                hist_vmin_line[0].set_xdata([vmin, vmin])
            if hist_vmax_line[0] is not None:
                hist_vmax_line[0].set_xdata([vmax, vmax])
            if hist_vspan[0] is not None:
                hist_vspan[0].remove()
            hist_vspan[0] = hist_ax.axvspan(vmin, vmax, alpha=0.15, color="yellow")
            hist_canvas.draw()
            refresh_preview(reset_view=False)

        def apply_auto_lut_to_tile(tile_idx: int) -> bool:
            labels = dlg._layout.get("tile_labels", [])
            if tile_idx < 0 or tile_idx >= len(labels):
                return False
            label = labels[tile_idx]
            if str(label).endswith(" Merged"):
                return False
            lut_pair = auto_lut_for_tile(tile_idx)
            if lut_pair is None:
                return False
            tile_key = self._montage_tile_key(label)
            dlg._per_tile_lut[tile_key] = lut_pair
            tile_meta = dlg._layout.get("tile_meta", [])
            channel_norm = (tile_meta[tile_idx] if tile_idx < len(tile_meta) else {}).get("channel_norm")
            if channel_norm:
                dlg._per_tile_lut[channel_norm] = lut_pair
            return True

        def on_auto_selected() -> None:
            sel = selected_index[0]
            if sel is None or not apply_auto_lut_to_tile(sel):
                return
            current_vmin[0], current_vmax[0] = dlg._per_tile_lut[self._montage_tile_key(dlg._layout["tile_labels"][sel])]
            update_histogram_for_tile(sel)
            refresh_preview(reset_view=False)

        def on_auto_all() -> None:
            labels = dlg._layout.get("tile_labels", [])
            changed = False
            for i, label in enumerate(labels):
                if str(label).endswith(" Merged"):
                    continue
                changed = apply_auto_lut_to_tile(i) or changed
            if not changed:
                return
            sel = selected_index[0]
            if sel is not None:
                tile_key = self._montage_tile_key(labels[sel]) if sel < len(labels) else ""
                current_vmin[0], current_vmax[0] = dlg._per_tile_lut.get(tile_key, (None, None))
                update_histogram_for_tile(sel)
            refresh_preview(reset_view=False)

        def on_spin_editing_finished() -> None:
            """Apply LUT when user finishes editing (Enter or focus leave). Lets you type full number without preview jumping on each digit."""
            if selected_index[0] is None or is_updating_lut[0]:
                return
            is_updating_lut[0] = True
            try:
                apply_min_max_from_spinboxes()
            finally:
                is_updating_lut[0] = False

        min_spin.editingFinished.connect(on_spin_editing_finished)
        max_spin.editingFinished.connect(on_spin_editing_finished)
        auto_selected_btn.clicked.connect(on_auto_selected)
        auto_all_btn.clicked.connect(on_auto_all)

        def on_hist_press(event):
            if event.inaxes != hist_ax or event.xdata is None or selected_index[0] is None:
                return
            x = event.xdata
            result = get_tile_raw_for_hist(selected_index[0])
            if result is None or result[0].size == 0:
                return
            hmin, hmax = hist_range[0]
            rng = max(hmax - hmin, 1e-9)
            thresh = max(rng * 0.02, 1e-6)
            vmin, vmax = current_vmin[0] or hmin, current_vmax[0] or hmax
            if abs(x - vmin) < thresh and abs(x - vmin) <= abs(x - vmax):
                dragging_line[0] = "min"
            elif abs(x - vmax) < thresh:
                dragging_line[0] = "max"
            else:
                mid = (vmin + vmax) / 2
                dragging_line[0] = "min" if x < mid else "max"

        def on_hist_motion(event):
            if event.inaxes != hist_ax or event.xdata is None or dragging_line[0] is None or is_updating_lut[0] or selected_index[0] is None:
                return
            x = event.xdata
            hmin, hmax = hist_range[0]
            is_updating_lut[0] = True
            try:
                vmin, vmax = current_vmin[0] or hmin, current_vmax[0] or hmax
                if dragging_line[0] == "min":
                    new_min = max(hmin, min(x, (current_vmax[0] or hmax) - 1e-9))
                    current_vmin[0] = new_min
                else:
                    new_max = min(hmax, max(x, (current_vmin[0] or hmin) + 1e-9))
                    current_vmax[0] = new_max
                vmin, vmax = current_vmin[0], current_vmax[0]
                sel = selected_index[0]
                if sel is not None and sel < len(dlg._layout.get("tile_labels", [])):
                    dlg._per_tile_lut[self._montage_tile_key(dlg._layout["tile_labels"][sel])] = (vmin, vmax)
                min_spin.blockSignals(True)
                min_spin.setValue(vmin)
                min_spin.setRange(hmin, hmax)
                min_spin.blockSignals(False)
                max_spin.blockSignals(True)
                max_spin.setValue(vmax)
                max_spin.setRange(hmin, hmax)
                max_spin.blockSignals(False)
                refresh_preview(reset_view=False)
                if hist_vmin_line[0] is not None:
                    hist_vmin_line[0].set_xdata([vmin, vmin])
                if hist_vmax_line[0] is not None:
                    hist_vmax_line[0].set_xdata([vmax, vmax])
                if hist_vspan[0] is not None:
                    hist_vspan[0].remove()
                hist_vspan[0] = hist_ax.axvspan(vmin, vmax, alpha=0.15, color="yellow")
                hist_canvas.draw()
            finally:
                is_updating_lut[0] = False

        def on_hist_release(event):
            dragging_line[0] = None

        image_canvas.clicked_image.connect(on_canvas_click)
        image_canvas.drag_started.connect(on_focus_drag_started)
        image_canvas.drag_updated.connect(on_focus_drag_updated)
        image_canvas.drag_finished.connect(on_focus_drag_finished)
        hist_canvas.mpl_connect("button_press_event", on_hist_press)
        hist_canvas.mpl_connect("motion_notify_event", on_hist_motion)
        hist_canvas.mpl_connect("button_release_event", on_hist_release)

        update_histogram_for_tile(None)

        def on_update() -> None:
            # Rebuild with current spacing/border/paths but keep existing per-tile LUT and color overrides
            old_focus_roi = getattr(dlg, "_focus_roi", None)
            old_w = int(dlg._layout.get("out_w", 0) or 0)
            old_h = int(dlg._layout.get("out_h", 0) or 0)
            old_labels = list(dlg._layout.get("tile_labels", []) or [])
            old_meta = list(dlg._layout.get("tile_meta", []) or [])
            for i, label in enumerate(old_labels):
                tile_key = self._montage_tile_key(label)
                if tile_key not in dlg._per_tile_lut:
                    continue
                value = dlg._per_tile_lut[tile_key]
                dlg._per_tile_lut.setdefault(i, value)
                channel_norm = (old_meta[i] if i < len(old_meta) else {}).get("channel_norm")
                if channel_norm:
                    dlg._per_tile_lut.setdefault(channel_norm, value)
            result = self._build_montage_array(
                return_layout=True,
                per_tile_lut=dlg._per_tile_lut,
                per_tile_color_override=getattr(self, "montage_tile_color_overrides", None) or {},
                per_tile_percentile_override=getattr(self, "montage_tile_percentile_overrides", None) or {},
            )
            if result is None:
                return
            out, new_layout = result
            dlg._layout = new_layout
            dlg._current_displayed = out
            new_labels = list(new_layout.get("tile_labels", []) or [])
            new_meta = list(new_layout.get("tile_meta", []) or [])
            for i, label in enumerate(new_labels):
                tile_key = self._montage_tile_key(label)
                if tile_key in dlg._per_tile_lut:
                    continue
                channel_norm = (new_meta[i] if i < len(new_meta) else {}).get("channel_norm")
                if channel_norm and channel_norm in dlg._per_tile_lut:
                    dlg._per_tile_lut[tile_key] = dlg._per_tile_lut[channel_norm]
                elif i in dlg._per_tile_lut:
                    dlg._per_tile_lut[tile_key] = dlg._per_tile_lut[i]
            new_w = int(new_layout.get("out_w", 0) or 0)
            new_h = int(new_layout.get("out_h", 0) or 0)
            if old_focus_roi is not None and old_w > 0 and old_h > 0 and new_w > 0 and new_h > 0:
                x0, y0, x1, y1 = old_focus_roi
                if old_w != new_w or old_h != new_h:
                    sx = new_w / old_w
                    sy = new_h / old_h
                    x0, x1 = int(round(x0 * sx)), int(round(x1 * sx))
                    y0, y1 = int(round(y0 * sy)), int(round(y1 * sy))
                dlg._focus_roi = self._normalize_montage_roi(x0, y0, x1, y1, new_w, new_h)
            dlg._focus_drag_start = None
            selected_index[0] = None
            current_vmin[0], current_vmax[0] = None, None
            selected_label.setText("Click a tile to adjust LUT. Enable Focus inset, then right-drag a box to add a zoom callout. Middle-drag to pan, wheel to zoom.")
            update_histogram_for_tile(None)
            sync_focus_overlay()
            refresh_preview(reset_view=True)

        def on_edit_colors() -> None:
            self._edit_montage_tile_colors(dlg._layout, refresh_callback=on_update)

        def on_save() -> None:
            current_displayed = compose_current_displayed()
            if current_displayed is None:
                if self._montage_export_is_merged_only():
                    QMessageBox.warning(dlg, "Montage", "No merged tile available to export.")
                else:
                    QMessageBox.warning(dlg, "Montage", "Nothing to save. Update preview first.")
                return
            dlg._current_displayed = current_displayed
            paths = self._get_montage_image_paths()
            default_dir = paths[0].parent if paths else Path(".")
            merged_only = self._montage_export_is_merged_only()
            fmt_text = dlg._export_format_combo.currentText()
            ext = self._montage_export_extension(fmt_text)
            path, _ = QFileDialog.getSaveFileName(
                dlg,
                "Save montage",
                str(default_dir / f"{self._default_montage_output_stem(paths, merged_only=merged_only)}{ext}"),
                "PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg);;All (*)",
            )
            if not path:
                return
            try:
                out_path = Path(path)
                if not out_path.suffix:
                    out_path = out_path.with_suffix(ext)
                dpi = self._montage_export_dpi(dlg)
                self._save_montage_array_to_path(current_displayed, out_path, dpi=dpi)
                self.parent_window.log_output(f"Montage saved: {out_path}\n")
                show_task_complete_dialog(
                    self,
                    title="Montage saved",
                    message="Montage image saved.",
                    output_dir=out_path.parent,
                )
                refresh_preview_same_view()
            except Exception as e:
                QMessageBox.warning(dlg, "Montage", f"Save failed: {e}")

        def on_apply_settings_to_files() -> None:
            selected_files = self._select_montage_batch_files(dlg)
            if selected_files is None:
                return
            if not selected_files:
                QMessageBox.information(dlg, "Montage", "No files selected.")
                return
            fmt_text = dlg._export_format_combo.currentText()
            ext = self._montage_export_extension(fmt_text)
            dpi = self._montage_export_dpi(dlg)
            export_mode = self._montage_export_mode()
            merged_only = self._montage_export_is_merged_only()
            summary, warnings = self._summarize_montage_batch_export(
                selected_files,
                export_mode=export_mode,
                export_format=fmt_text,
                export_dpi=dpi,
            )
            confirm = QMessageBox.question(
                dlg,
                "Confirm batch export",
                summary,
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Ok,
            )
            if confirm != QMessageBox.StandardButton.Ok:
                return

            current_labels = list(dlg._layout.get("tile_labels", []) or [])
            current_meta = list(dlg._layout.get("tile_meta", []) or [])
            batch_lut: Dict[Any, Tuple[float, float]] = {}
            for i, label in enumerate(current_labels):
                tile_key = self._montage_tile_key(label)
                if tile_key in dlg._per_tile_lut:
                    batch_lut[i] = dlg._per_tile_lut[tile_key]
                    channel_norm = (current_meta[i] if i < len(current_meta) else {}).get("channel_norm")
                    if channel_norm:
                        batch_lut[channel_norm] = dlg._per_tile_lut[tile_key]
            for key, value in (dlg._per_tile_lut or {}).items():
                if isinstance(key, int):
                    batch_lut[key] = value

            batch_colors: Dict[Any, Tuple[float, float, float]] = {}
            for i, label in enumerate(current_labels):
                tile_key = self._montage_tile_key(label)
                meta = current_meta[i] if i < len(current_meta) else {}
                value = self._montage_lookup_color_override_for_meta(tile_key, i, meta, None)
                if value is not None:
                    batch_colors[i] = value
                    channel_norm = meta.get("channel_norm")
                    if channel_norm:
                        batch_colors[channel_norm] = value

            batch_percentiles: Dict[Any, Tuple[float, float]] = {}
            for i, label in enumerate(current_labels):
                tile_key = self._montage_tile_key(label)
                meta = current_meta[i] if i < len(current_meta) else {}
                value = self._montage_lookup_percentile_override_for_meta(tile_key, i, meta)
                if value is not None:
                    batch_percentiles[i] = value
                    channel_norm = meta.get("channel_norm")
                    if channel_norm:
                        batch_percentiles[channel_norm] = value

            original_path_text = self.montage_path.text() if self._qt_widget_alive(getattr(self, "montage_path", None)) else ""
            saved_paths: List[Path] = []
            failures: List[str] = []
            try:
                from PIL import Image

                for src in selected_files:
                    if self._qt_widget_alive(getattr(self, "montage_path", None)):
                        self.montage_path.blockSignals(True)
                        self.montage_path.setText(str(src))
                        self.montage_path.blockSignals(False)
                    result = self._compose_montage_for_export(
                        per_tile_lut=batch_lut,
                        per_tile_color_override=batch_colors,
                        per_tile_percentile_override=batch_percentiles,
                        focus_roi=getattr(dlg, "_focus_roi", None),
                        merged_only=merged_only,
                        source_layout=dlg._layout,
                    )
                    if result is None:
                        failures.append(f"{src.name}: could not build montage")
                        continue
                    displayed, _batch_layout = result
                    suffix_name = "_merged" if merged_only else "_montage"
                    out_path = src.with_name(f"{src.stem}{suffix_name}{ext}")
                    self._save_montage_array_to_path(displayed, out_path, dpi=dpi)
                    saved_paths.append(out_path)
            except Exception as e:
                failures.append(str(e))
            finally:
                if self._qt_widget_alive(getattr(self, "montage_path", None)):
                    self.montage_path.blockSignals(True)
                    self.montage_path.setText(original_path_text)
                    self.montage_path.blockSignals(False)
                refresh_preview_same_view()

            if saved_paths:
                self.parent_window.log_output(f"Batch montage export: {len(saved_paths)} file(s) saved.\n")
                message = f"Saved {len(saved_paths)} montage file(s)."
                if failures:
                    message += f"\n\n{len(failures)} file(s) failed:\n" + "\n".join(failures[:6])
                show_task_complete_dialog(
                    self,
                    title="Montages saved",
                    message=message,
                    output_dir=saved_paths[0].parent,
                )
            elif failures:
                QMessageBox.warning(dlg, "Montage", "No montages were saved.\n\n" + "\n".join(failures[:8]))

        refresh_preview(reset_view=True)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        preset_row.addWidget(QLabel("Preset:"))
        dlg._montage_preset_combo = QComboBox()
        dlg._montage_preset_combo.setEditable(False)
        dlg._montage_preset_combo.addItem("(none)")
        for name in list_montage_preset_names():
            dlg._montage_preset_combo.addItem(name)
        self._compact_preview_control(dlg._montage_preset_combo, 130)
        preset_row.addWidget(dlg._montage_preset_combo)
        save_preset_btn = QPushButton("Save")
        style_compact_action_button(save_preset_btn, horizontal_padding=10)
        load_preset_btn = QPushButton("Load")
        style_compact_action_button(load_preset_btn, horizontal_padding=10)
        delete_preset_btn = QPushButton("Del")
        style_compact_action_button(delete_preset_btn, horizontal_padding=10)
        preset_row.addWidget(save_preset_btn)
        preset_row.addWidget(load_preset_btn)
        preset_row.addWidget(delete_preset_btn)
        preset_row.addSpacing(12)

        figure_row = QHBoxLayout()
        figure_row.setSpacing(6)
        figure_label = QLabel("Figure:")
        figure_label.setToolTip(
            "Quick style for export and display settings only.\n"
            "Does not change tile order, LUTs, or channel colors."
        )
        figure_row.addWidget(figure_label)
        figure_tooltips = self._montage_figure_preset_tooltips()
        for preset_name in ("Nature", "Presentation", "Supplement"):
            btn = QPushButton(preset_name)
            style_compact_action_button(btn, horizontal_padding=12)
            btn.setToolTip(figure_tooltips.get(preset_name, ""))
            btn.clicked.connect(lambda _checked=False, n=preset_name: (self._apply_montage_figure_preset(n, dlg), refresh_preview_same_view()))
            figure_row.addWidget(btn)
        figure_row.addSpacing(12)

        export_row = preset_row
        export_row.addWidget(QLabel("Export:"))
        self.montage_export_mode = QComboBox()
        self.montage_export_mode.addItems(["Full montage", "Merged only"])
        self.montage_export_mode.setToolTip("Save the full montage grid or only merged channel tiles.")
        self._compact_preview_control(self.montage_export_mode, 115)
        export_row.addWidget(self.montage_export_mode)
        export_row.addWidget(QLabel("Bg:"))
        dlg._export_background_combo = QComboBox()
        dlg._export_background_combo.addItems(["White", "Black", "Gray", "Transparent"])
        current_bg = self._montage_border_color_name()
        bg_idx = dlg._export_background_combo.findText(current_bg)
        if bg_idx >= 0:
            dlg._export_background_combo.setCurrentIndex(bg_idx)
        dlg._export_background_combo.setToolTip("Montage background/border fill. Transparent is preserved in PNG/TIFF exports.")
        self._compact_preview_control(dlg._export_background_combo, 105)
        export_row.addWidget(dlg._export_background_combo)
        export_row.addWidget(QLabel("Fmt:"))
        dlg._export_format_combo = QComboBox()
        dlg._export_format_combo.addItems(["PNG", "TIFF", "JPEG"])
        dlg._export_format_combo.setToolTip("Default export format. You can still type a different extension in the save dialog.")
        self._compact_preview_control(dlg._export_format_combo, 80)
        export_row.addWidget(dlg._export_format_combo)
        export_row.addWidget(QLabel("DPI:"))
        dlg._export_dpi_combo = QComboBox()
        dlg._export_dpi_combo.addItems(["300 DPI", "600 DPI", "900 DPI", "1200 DPI", "2400 DPI"])
        dlg._export_dpi_combo.setToolTip("Resolution (DPI) for PNG/TIFF/JPEG export. Applied when you save from this preview.")
        self._compact_preview_control(dlg._export_dpi_combo, 100)
        export_row.addWidget(dlg._export_dpi_combo)
        export_row.addStretch(1)
        main_layout.addLayout(preset_row)

        def refresh_preset_combo(selected: Optional[str] = None) -> None:
            current = selected or dlg._montage_preset_combo.currentText()
            dlg._montage_preset_combo.blockSignals(True)
            dlg._montage_preset_combo.clear()
            dlg._montage_preset_combo.addItem("(none)")
            names = list_montage_preset_names()
            for name in names:
                dlg._montage_preset_combo.addItem(name)
            idx = dlg._montage_preset_combo.findText(current)
            dlg._montage_preset_combo.setCurrentIndex(idx if idx >= 0 else 0)
            dlg._montage_preset_combo.blockSignals(False)

        def on_save_preset() -> None:
            name, ok = QInputDialog.getText(dlg, "Save montage preset", "Preset name:")
            if not ok or not str(name).strip():
                return
            save_montage_preset(str(name).strip(), self._capture_montage_preset_state(dlg))
            refresh_preset_combo(str(name).strip())

        def on_load_preset() -> None:
            name = dlg._montage_preset_combo.currentText()
            if not name or name == "(none)":
                return
            payload = load_montage_preset(name)
            if not payload:
                QMessageBox.warning(dlg, "Montage preset", f"Preset “{name}” was not found.")
                return
            self._apply_montage_preset_state(payload, dlg)
            bg_idx_local = dlg._export_background_combo.findText(self._montage_border_color_name())
            if bg_idx_local >= 0:
                dlg._export_background_combo.setCurrentIndex(bg_idx_local)
            on_update()

        def on_delete_preset() -> None:
            name = dlg._montage_preset_combo.currentText()
            if not name or name == "(none)":
                return
            if delete_montage_preset(name):
                refresh_preset_combo("(none)")

        def on_export_background_changed(text: str) -> None:
            self._sync_montage_border_color(text)
            combo = getattr(self, "montage_border_color", None)
            if self._qt_widget_alive(combo):
                idx = combo.findText(text)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            refresh_preview_same_view()

        save_preset_btn.clicked.connect(on_save_preset)
        load_preset_btn.clicked.connect(on_load_preset)
        delete_preset_btn.clicked.connect(on_delete_preset)
        dlg._export_background_combo.currentTextChanged.connect(on_export_background_changed)

        btn_row = figure_row
        update_btn = QPushButton("Update")
        style_compact_action_button(update_btn, horizontal_padding=20)
        update_btn.setToolTip("Rebuild montage from the preview layout settings and current paths. Resets per-tile LUT.")
        update_btn.setAutoDefault(False)
        update_btn.setDefault(False)
        update_btn.clicked.connect(on_update)
        edit_colors_btn = QPushButton("Edit tiles")
        style_compact_action_button(edit_colors_btn, horizontal_padding=20)
        edit_colors_btn.setToolTip(
            "Include/exclude tiles, per-tile z projection, color, and percentile normalization."
        )
        edit_colors_btn.setAutoDefault(False)
        edit_colors_btn.setDefault(False)
        edit_colors_btn.clicked.connect(on_edit_colors)
        apply_settings_btn = QPushButton("Batch...")
        style_compact_action_button(apply_settings_btn, horizontal_padding=20)
        apply_settings_btn.setToolTip("Batch export selected ND2/TIFF files with the current montage preview settings.")
        apply_settings_btn.setAutoDefault(False)
        apply_settings_btn.setDefault(False)
        apply_settings_btn.clicked.connect(on_apply_settings_to_files)
        save_btn = QPushButton("Save")
        style_compact_action_button(save_btn, horizontal_padding=20)
        save_btn.setAutoDefault(False)
        save_btn.setDefault(False)
        save_btn.clicked.connect(on_save)
        close_btn = QPushButton("Close")
        style_compact_action_button(close_btn, horizontal_padding=20)
        close_btn.setAutoDefault(False)
        close_btn.setDefault(False)
        if embedded:
            close_btn.clicked.connect(self._close_embedded_post_preview)
        else:
            close_btn.clicked.connect(dlg.close)
        btn_row.addWidget(update_btn)
        btn_row.addWidget(edit_colors_btn)
        btn_row.addStretch()
        btn_row.addWidget(apply_settings_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(close_btn)
        main_layout.addLayout(btn_row)

        close_shortcut = QShortcut(QKeySequence.StandardKey.Close, dlg)
        close_shortcut.activated.connect(self._close_embedded_post_preview if embedded else dlg.close)
        if not embedded:
            minimize_shortcut = QShortcut(QKeySequence("Ctrl+M"), dlg)
            minimize_shortcut.activated.connect(dlg.showMinimized)
        dlg.setLayout(main_layout)
        self._active_montage_preview = dlg
        if embedded and getattr(self, "_module_preview_mount", None) is not None:
            self._embedded_post_preview = dlg
            mount_embedded_preview(
                self._module_preview_mount,
                dlg,
                preview_surface=self._preview_surface,
            )
            self.refresh_post_candidate_list()
            if self._preview_surface is not None:
                self._preview_surface.set_preview_chrome_visible(False)
        else:
            dlg.show()

    def _show_image_dialog(self, img: np.ndarray, title: str) -> None:
        """Show a non-modal preview dialog with ImageCanvas (zoom, pan), Reset view, Cmd+W, and Ctrl+M."""
        dummy_mask = np.zeros(img.shape[:2], dtype=np.int32)
        img_scaled, _ = scale_for_preview(img, dummy_mask)
        pixmap = to_qpixmap(img_scaled)
        w, h = img_scaled.shape[1], img_scaled.shape[0]

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setModal(False)
        dlg.setMinimumSize(900, 700)
        layout = QVBoxLayout(dlg)

        image_canvas = ImageCanvas()
        image_canvas.setToolTip("Zoom: mouse wheel. Pan: left-click and drag (or middle-click and drag).")
        image_canvas.set_pixmap(pixmap, (w, h), reset_view=True)
        layout.addWidget(image_canvas, 1)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("Reset view")
        reset_btn.setToolTip("Reset zoom and pan to fit image.")
        def on_reset_view():
            image_canvas.set_pixmap(pixmap, (w, h), reset_view=True)
        reset_btn.clicked.connect(on_reset_view)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.close)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        close_shortcut = QShortcut(QKeySequence.StandardKey.Close, dlg)
        close_shortcut.activated.connect(dlg.close)
        minimize_shortcut = QShortcut(QKeySequence("Ctrl+M"), dlg)
        minimize_shortcut.activated.connect(dlg.showMinimized)

        dlg.setLayout(layout)
        dlg.show()

    def _get_video_yaxis_limits(self) -> tuple[Optional[float], Optional[float]]:
        y_min = self.yaxis_min.value() if self.yaxis_min.value() > 0 else None
        y_max = self.yaxis_max.value() if self.yaxis_max.value() > 0 else None
        return y_min, y_max

    def _schedule_video_preview_refresh(self, *_args: Any) -> None:
        dlg = getattr(self, "_embedded_post_preview", None)
        if dlg is None or dlg.objectName() != "EmbeddedVideoMontagePreview":
            return
        timer = self._video_preview_refresh_timer
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(200)
            timer.timeout.connect(self._refresh_video_montage_preview)
            self._video_preview_refresh_timer = timer
        timer.start()

    def _invalidate_video_preview_cache(self) -> None:
        self._video_preview_cache = None

    def _load_video_preview_cache(self) -> Optional[dict[str, Any]]:
        if self._video_preview_cache is not None:
            return self._video_preview_cache
        video_paths = self._parse_growth_curve_video_paths(self.video_path.text())
        excel_raw = self.excel_path.text().strip()
        excel_path = Path(excel_raw) if excel_raw else None
        if not video_paths:
            return None
        try:
            video_frames = load_growth_curve_video_frames_rgb(video_paths)
            video_frames = apply_video_montage_overlays(
                video_frames,
                mask_path=Path(self.video_mask_path.text().strip()) if self.video_mask_path.text().strip() else None,
                foci_path=Path(self.video_foci_path.text().strip()) if self.video_foci_path.text().strip() else None,
                show_mask_contours=self.video_mask_contours.isChecked(),
                show_foci=self.video_foci_overlay.isChecked(),
            )
            frames = relative_areas = time_minutes = None
            graph_error = ""
            if excel_path is not None and excel_path.exists():
                try:
                    frames, relative_areas, time_minutes = load_growth_curve_summary_data(
                        excel_path,
                        self.time_interval.value(),
                    )
                except Exception as exc:
                    graph_error = f"Growth graph skipped: {exc}"
        except Exception as exc:
            raise ValueError(str(exc)) from exc
        num_frames = min(len(video_frames), len(frames)) if frames is not None else len(video_frames)
        if num_frames <= 0:
            raise ValueError("No frames available to preview.")
        cache = {
            "video_frames": video_frames[:num_frames],
            "frames": frames[:num_frames] if frames is not None else None,
            "relative_areas": relative_areas[:num_frames] if relative_areas is not None else None,
            "time_minutes": time_minutes[:num_frames] if time_minutes is not None else None,
            "num_frames": num_frames,
            "has_graph": frames is not None,
            "graph_error": graph_error,
        }
        self._video_preview_cache = cache
        return cache

    def _refresh_video_montage_preview(self) -> None:
        dlg = getattr(self, "_embedded_post_preview", None)
        if dlg is None:
            return
        preview_label = getattr(dlg, "_video_preview_label", None)
        frame_slider = getattr(dlg, "_video_frame_slider", None)
        status_label = getattr(dlg, "_video_preview_status", None)
        if preview_label is None:
            return
        try:
            cache = self._load_video_preview_cache()
            if cache is None:
                preview_label.clear()
                if status_label is not None:
                    status_label.setText("Set video input on the left. Excel is optional for the growth graph.")
                return
            num_frames = int(cache["num_frames"])
            if frame_slider is not None:
                frame_slider.blockSignals(True)
                frame_slider.setMaximum(max(0, num_frames - 1))
                frame_idx = int(frame_slider.value())
                frame_slider.blockSignals(False)
            else:
                frame_idx = 0
            frame_idx = max(0, min(frame_idx, num_frames - 1))
            y_min, y_max = self._get_video_yaxis_limits()
            combined = compose_growth_curve_combined_frame(
                frame_idx,
                cache["video_frames"],
                cache["frames"],
                cache["relative_areas"],
                cache["time_minutes"],
                yaxis_min=y_min,
                yaxis_max=y_max,
            )
            pix = to_qpixmap(combined)
            if pix is not None and not pix.isNull():
                target_w = max(320, int(preview_label.width() or 1200))
                target_h = max(240, int(preview_label.height() or 480))
                scaled = pix.scaled(
                    target_w,
                    target_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                preview_label.setPixmap(scaled)
            if status_label is not None:
                mode = "growth graph + video" if cache.get("has_graph") else "video only (no growth graph)"
                suffix = f" {cache.get('graph_error')}" if cache.get("graph_error") else ""
                status_label.setText(
                    f"Frame {frame_idx + 1} / {num_frames} — {mode}.{suffix}"
                )
            self._update_button_state()
        except Exception as exc:
            preview_label.clear()
            if status_label is not None:
                status_label.setText(str(exc))
            if preview_label.parent() is not None:
                preview_label.setToolTip(str(exc))

    def _preview_video_montage(self) -> None:
        if not self.video_path.text():
            QMessageBox.warning(
                self,
                "Video montage",
                "Please select a video input first. Excel is optional for the growth graph.",
            )
            return
        self._set_post_preview_mode("video")
        embedded = getattr(self, "_module_preview_mount", None) is not None
        self._show_video_montage_preview(embedded=embedded)

    def _show_video_montage_preview(self, *, embedded: bool = True) -> None:
        self._close_embedded_post_preview()
        self._invalidate_video_preview_cache()
        if embedded and getattr(self, "_module_preview_mount", None) is not None:
            dlg = QWidget(self)
            dlg.setObjectName("EmbeddedVideoMontagePreview")
            dlg.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        else:
            dlg = QDialog(self)
            dlg.setObjectName("VideoMontagePreviewDialog")
            dlg.setWindowTitle("Video montage preview")
            dlg.setModal(False)
            dlg.setMinimumSize(900, 700)

        main_layout = QVBoxLayout(dlg)
        hint = QLabel(
            "Growth curve (left) and microscopy frame (right). Scrub frames, tune graph/video settings, "
            "then create the montage below."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaa; font-size: 11px;")
        main_layout.addWidget(hint)

        preview_label = QLabel()
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_label.setMinimumHeight(280)
        preview_label.setStyleSheet("background-color: #1a1a1a; border-radius: 6px;")
        preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        dlg._video_preview_label = preview_label
        main_layout.addWidget(preview_label, 1)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Frame:"))
        frame_slider = QSlider(Qt.Orientation.Horizontal)
        frame_slider.setMinimum(0)
        frame_slider.setMaximum(0)
        frame_slider.setValue(0)
        frame_slider.valueChanged.connect(self._schedule_video_preview_refresh)
        dlg._video_frame_slider = frame_slider
        slider_row.addWidget(frame_slider, 1)
        main_layout.addLayout(slider_row)

        status_label = QLabel("")
        status_label.setWordWrap(True)
        status_label.setStyleSheet("color: #9a9a9a; font-size: 10pt;")
        dlg._video_preview_status = status_label
        main_layout.addWidget(status_label)

        settings_panel = self._build_graph_and_video_settings()
        settings_section = CollapsibleSection(
            "Settings & export",
            settings_panel,
            parent=dlg,
            expanded=True,
            compact=True,
        )
        main_layout.addWidget(settings_section)
        self.process_btn.setParent(dlg)
        main_layout.addWidget(self.process_btn)

        dlg.setLayout(main_layout)
        if embedded and getattr(self, "_module_preview_mount", None) is not None:
            self._embedded_post_preview = dlg
            mount_embedded_preview(
                self._module_preview_mount,
                dlg,
                preview_surface=self._preview_surface,
            )
            self.refresh_post_candidate_list()
        else:
            dlg.show()

        QTimer.singleShot(0, self._refresh_video_montage_preview)

    def _update_button_state(self):
        """Update preview/create button state based on input fields."""
        has_video = bool(self.video_path.text())
        if hasattr(self, "preview_video_btn"):
            self.preview_video_btn.setEnabled(has_video)
        has_format = (
            self.format_tif.isChecked()
            or self.format_avi.isChecked()
            or self.format_gif.isChecked()
            or self.format_mp4.isChecked()
        )
        if hasattr(self, "process_btn"):
            self.process_btn.setEnabled(has_video and has_format)
    
    def _on_select_all_changed(self, state):
        """Handle 'Select All' checkbox change"""
        checked = state == 2  # Qt.Checked = 2
        self.format_tif.setChecked(checked)
        self.format_avi.setChecked(checked)
        self.format_gif.setChecked(checked)
        self.format_mp4.setChecked(checked)
    
    def _on_format_changed(self):
        """Handle individual format checkbox change"""
        all_checked = (self.format_tif.isChecked() and self.format_avi.isChecked() and 
                      self.format_gif.isChecked() and self.format_mp4.isChecked())
        self.format_all.blockSignals(True)
        self.format_all.setChecked(all_checked)
        self.format_all.blockSignals(False)
        self._update_button_state()
    
    def select_video_file(self):
        """Select TIFF file for video input"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select TIFF File",
            get_default_browse_dir(),
            "TIFF Files (*.tif *.tiff *.TIF *.TIFF);;All Files (*)"
        )
        if file_path:
            self.video_path.setText(file_path)
    
    def select_video_folder(self):
        """Select folder containing TIFF sequence"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with TIFF Sequence",
            get_default_browse_dir()
        )
        if folder_path:
            self.video_path.setText(folder_path)
            self._auto_fill_video_inputs()

    def select_video_multi_channel(self):
        """Select several single-channel TIFF stacks (one per channel); merged with LUTs in the worker."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select one TIFF per channel (same frame count and size)",
            get_default_browse_dir(),
            "TIFF Files (*.tif *.tiff *.TIF *.TIFF);;All Files (*)",
        )
        if paths:
            self.video_path.setText("|".join(paths))

    def _video_autofill_root(self) -> Optional[Path]:
        """Infer dataset root from the current video/mask/foci/excel path fields."""
        candidates: list[Path] = []
        for raw in (
            self.video_path.text(),
            self.excel_path.text(),
            self.video_mask_path.text(),
            self.video_foci_path.text(),
        ):
            text = (raw or "").strip()
            if not text:
                continue
            first = self._parse_growth_curve_video_paths(text)[0] if ("|" in text or "\n" in text) else Path(text)
            candidates.append(first)

        if not candidates:
            picked = QFileDialog.getExistingDirectory(
                self,
                "Select dataset folder for video montage auto-fill",
                get_default_browse_dir(),
            )
            if not picked:
                return None
            candidates.append(Path(picked))

        start = candidates[0]
        root = start if start.is_dir() else start.parent
        for cur in (root, *root.parents):
            try:
                direct_tifs = [
                    p for p in cur.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in (".tif", ".tiff")
                    and "mask" not in p.name.lower()
                    and "visualization" not in p.name.lower()
                ]
                if direct_tifs:
                    return cur
            except Exception:
                continue
        return root

    @staticmethod
    def _prefer_first(paths: list[Path], predicates: list[Callable[[Path], bool]]) -> Optional[Path]:
        for pred in predicates:
            for path in paths:
                try:
                    if pred(path):
                        return path
                except Exception:
                    continue
        return paths[0] if paths else None

    def _auto_fill_video_inputs(self) -> None:
        """Auto-assign phase/FI channels, analysis workbook, masks, and foci files from a dataset root."""
        root = self._video_autofill_root()
        if root is None or not root.exists():
            return

        direct_tifs = sorted(
            [
                p for p in root.iterdir()
                if p.is_file()
                and p.suffix.lower() in (".tif", ".tiff")
                and "mask" not in p.name.lower()
                and "visualization" not in p.name.lower()
                and not p.name.startswith(".")
            ],
            key=lambda p: p.name.lower(),
        )
        phase = self._prefer_first(
            direct_tifs,
            [
                lambda p: "phase" in p.name.lower(),
                lambda p: "_fi_" not in p.name.lower() and "foci" not in p.name.lower(),
            ],
        )
        fi_channels = [p for p in direct_tifs if p != phase and ("_fi_" in p.name.lower() or "foci" in p.name.lower())]
        video_inputs = [p for p in [phase, *fi_channels] if p is not None]
        if video_inputs:
            self.video_path.setText("|".join(str(p) for p in video_inputs))

        analysis_files = sorted((root / "_analysis").glob("*.xlsx"), key=lambda p: p.name.lower())
        analysis = self._prefer_first(
            [
                p for p in analysis_files
                if "visualization" not in p.name.lower()
                and not p.name.lower().startswith("@summary")
            ],
            [
                lambda p: p.name.lower().endswith("_mask_filtered_analysis.xlsx"),
                lambda p: p.name.lower().endswith("_masks_analysis.xlsx"),
                lambda p: "_analysis" in p.name.lower() and "_gfp" not in p.name.lower() and "_dsred" not in p.name.lower(),
            ],
        )
        if analysis is not None:
            self.excel_path.setText(str(analysis))

        mask_candidates: list[Path] = []
        for seg_dir in sorted(root.glob("_segmentation*"), key=lambda p: p.name.lower()):
            mask_candidates.extend(sorted((seg_dir / "filtered_masks").glob("*_mask_filtered.tif*"), key=lambda p: p.name.lower()))
            mask_candidates.extend(sorted(seg_dir.glob("*_masks.tif*"), key=lambda p: p.name.lower()))
        mask = self._prefer_first(
            mask_candidates,
            [
                lambda p: "filtered" in p.name.lower(),
                lambda p: "masks" in p.name.lower(),
            ],
        )
        if mask is not None:
            self.video_mask_path.setText(str(mask))
            self.video_mask_contours.setChecked(True)

        foci_candidates: list[Path] = []
        for seg_dir in sorted(root.glob("_segmentation*"), key=lambda p: p.name.lower()):
            foci_candidates.extend(sorted((seg_dir / "foci_filtered").glob("*.csv"), key=lambda p: p.name.lower()))
            foci_candidates.extend(sorted((seg_dir / "foci_detection").glob("*.csv"), key=lambda p: p.name.lower()))
        foci = self._prefer_first(
            foci_candidates,
            [
                lambda p: "filtered" in p.name.lower(),
                lambda p: "foci" in p.name.lower(),
            ],
        )
        if foci is not None:
            self.video_foci_path.setText(str(foci))
            self.video_foci_overlay.setChecked(True)

        if not any((video_inputs, analysis, mask, foci)):
            QMessageBox.information(
                self,
                "Video montage auto-fill",
                f"No matching video montage inputs were found in:\n{root}",
            )
        else:
            self._invalidate_video_preview_cache()
            self._schedule_video_preview_refresh()

    @staticmethod
    def _parse_growth_curve_video_paths(text: str) -> list[Path]:
        raw = (text or "").strip()
        if not raw:
            return []
        if "\n" in raw:
            return [Path(p.strip()) for p in raw.splitlines() if p.strip()]
        if "|" in raw:
            return [Path(p.strip()) for p in raw.split("|") if p.strip()]
        return [Path(raw)]
    
    def handle_dropped_video_path(self, dropped_path):
        """Handle dropped file or folder path for growth curve video tab"""
        path_obj = Path(dropped_path)
        # Update video path
        self.video_path.setText(dropped_path)

    def handle_dropped_excel_path(self, dropped_path):
        """Handle dropped path for Excel analysis file (file or folder with Excel files)."""
        path_obj = Path(dropped_path)
        if not path_obj.exists():
            return
        if path_obj.is_file():
            self.excel_path.setText(dropped_path)
        elif path_obj.is_dir():
            xlsx_files = sorted(
                list(path_obj.glob("*.xlsx")) + list(path_obj.glob("*.xls")),
                key=lambda p: p.name.lower(),
            )
            if not xlsx_files:
                return
            preferred = [
                p for p in xlsx_files
                if p.suffix.lower() in (".xlsx", ".xls")
                and (
                    p.name.lower().endswith("_analysis.xlsx")
                    or p.name.lower().endswith("_masks_analysis.xlsx")
                )
                and not p.name.lower().endswith("_foci_analysis.xlsx")
            ]
            self.excel_path.setText(str(preferred[0] if preferred else xlsx_files[0]))

    def select_excel_file(self):
        """Select Excel file from Analyze Masks tab"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel Analysis File",
            get_default_browse_dir(),
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if file_path:
            self.excel_path.setText(file_path)

    def _select_video_mask_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select mask TIFF for video contours",
            get_default_browse_dir(),
            "TIFF Files (*.tif *.tiff *.TIF *.TIFF);;All Files (*)",
        )
        if path:
            self.video_mask_path.setText(path)
            self.video_mask_contours.setChecked(True)

    def _select_video_foci_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select foci CSV/XLSX for video markers",
            get_default_browse_dir(),
            "Foci files (*.csv *.xlsx *.xls);;All Files (*)",
        )
        if path:
            self.video_foci_path.setText(path)
            self.video_foci_overlay.setChecked(True)
    
    def _is_folder_mode(self):
        """Check if video input is a single folder path (not multi-file)."""
        paths = self._parse_growth_curve_video_paths(self.video_path.text())
        if len(paths) != 1:
            return False
        p = paths[0]
        return p.exists() and p.is_dir()
    
    def _get_available_models(self):
        """
        Return list of available Omnipose models only (no Cellpose-only models).
        """
        return [
            'bact_phase_omni',  # Recommended for bacteria
            'bact_fluor_omni',
            'bact_phase_affinity'
        ]
    
    def create_video(self):
        """Create growth curve video with side-by-side graph and video"""
        if not self.video_path.text():
            QMessageBox.warning(self, "Missing Input", "Please select a video input.")
            return
        
        video_paths = self._parse_growth_curve_video_paths(self.video_path.text())
        excel_raw = self.excel_path.text().strip()
        excel_path = Path(excel_raw) if excel_raw else None
        
        if not video_paths:
            QMessageBox.warning(self, "Missing Input", "Please set a TIFF file, folder, or channel paths.")
            return
        if len(video_paths) > 1:
            for vp in video_paths:
                if not vp.is_file():
                    QMessageBox.warning(
                        self,
                        "Invalid path",
                        f"Multi-channel mode expects each path to be a file:\n{vp}",
                    )
                    return
                if not vp.exists():
                    QMessageBox.warning(self, "File Not Found", f"Video input not found:\n{vp}")
                    return
            base_dir = video_paths[0].parent
            input_stem = f"{video_paths[0].stem}_multi_{len(video_paths)}ch"
        else:
            video_path = video_paths[0]
            if not video_path.exists():
                QMessageBox.warning(self, "File Not Found", f"Video input not found:\n{video_path}")
                return
            if video_path.is_file():
                base_dir = video_path.parent
                input_stem = video_path.stem
            else:
                base_dir = video_path
                input_stem = video_path.name
        
        if excel_path is not None and not excel_path.exists():
            QMessageBox.warning(self, "File Not Found", f"Excel file not found:\n{excel_path}")
            return
        
        # Go up one level if in segmentation or Tif_ folder (same logic as other tabs)
        if base_dir.name.startswith('Tif_') or '_segmentation' in base_dir.name:
            base_dir = base_dir.parent
        
        # Create output directory at base level
        output_dir = base_dir / f"{input_stem}_growth_curve_video"
        output_dir.mkdir(exist_ok=True)
        
        # Show progress
        self.process_btn.setEnabled(False)
        self.process_btn.setText("Creating video montage...")
        self.parent_window.progress_bar.setVisible(True)
        self.parent_window.progress_bar.setRange(0, 100)
        self.parent_window.progress_bar.setValue(0)
        
        # Get selected formats
        formats = []
        if self.format_tif.isChecked():
            formats.append('tif')
        if self.format_avi.isChecked():
            formats.append('avi')
        if self.format_gif.isChecked():
            formats.append('gif')
        if self.format_mp4.isChecked():
            formats.append('mp4')
        
        # For completion popup "Open output folder"
        self._last_video_output_dir = output_dir
        # Run in worker thread
        self.worker = GrowthCurveVideoWorker(
            video_paths,
            excel_path,
            output_dir,
            self.time_interval.value(),
            *self._get_video_yaxis_limits(),
            self.fps.value(),
            formats,
            parent=self,
            mask_path=Path(self.video_mask_path.text().strip()) if self.video_mask_path.text().strip() else None,
            foci_path=Path(self.video_foci_path.text().strip()) if self.video_foci_path.text().strip() else None,
            show_mask_contours=self.video_mask_contours.isChecked(),
            show_foci=self.video_foci_overlay.isChecked(),
        )
        self.worker.finished.connect(self.on_video_finished)
        self.worker.progress.connect(self.on_video_progress)
        self.worker.start()
    
    def on_video_progress(self, value, message):
        """Handle progress updates"""
        self.parent_window.progress_bar.setValue(value)
        if message:
            self.parent_window.log_output(message)
    
    def on_video_finished(self, success, message):
        """Handle video creation completion"""
        if hasattr(self, "process_btn") and self._qt_widget_alive(self.process_btn):
            self.process_btn.setText("Create video montage")
        self._update_button_state()
        self.parent_window.progress_bar.setVisible(False)

        if success:
            self.parent_window.log_output(f"\n✓ {message}\n")
            out_dir = getattr(self, "_last_video_output_dir", None)
            show_task_complete_dialog(
                self,
                title="Video montage complete",
                message=message if message else "Video montage output created successfully.",
                output_dir=out_dir,
            )
        else:
            QMessageBox.warning(self, "Error", message)
    
    def get_session_params(self) -> dict[str, Any]:
        """Return current parameter values for session log (parameters only, no paths)."""
        d = get_from_widgets([
            ("time_interval", self.time_interval),
            ("yaxis_min", self.yaxis_min),
            ("yaxis_max", self.yaxis_max),
            ("fps", self.fps),
            ("format_tif", self.format_tif),
            ("format_avi", self.format_avi),
            ("format_gif", self.format_gif),
            ("format_mp4", self.format_mp4),
            ("video_mask_contours", self.video_mask_contours),
            ("video_foci_overlay", self.video_foci_overlay),
        ])
        for key, widget_name in (
            ("video_mask_path", "video_mask_path"),
            ("video_foci_path", "video_foci_path"),
        ):
            w = getattr(self, widget_name, None)
            if self._qt_widget_alive(w):
                d[key] = w.text().strip()
        d["montage_overlay_enable"] = self._safe_checked(getattr(self, "montage_overlay_enable", None), default=False)
        mask_edit = getattr(self, "montage_overlay_mask_path", None)
        if self._qt_widget_alive(mask_edit):
            d["montage_overlay_mask_path"] = mask_edit.text().strip()
        d["montage_overlay_frame"] = self._safe_spin_value(getattr(self, "montage_overlay_frame", None), 0)
        z_proj = getattr(self, "montage_z_projection", None)
        if self._qt_widget_alive(z_proj):
            d["montage_z_projection"] = z_proj.currentText()
        d["montage_border_color"] = self._montage_border_color_name()
        if getattr(self, "montage_tile_percentile_overrides", None):
            d["montage_tile_percentile_overrides"] = {
                str(k): [float(v[0]), float(v[1])]
                for k, v in self.montage_tile_percentile_overrides.items()
            }
        if getattr(self, "montage_excluded_tile_keys", None):
            d["montage_excluded_tile_keys"] = sorted(self.montage_excluded_tile_keys)
        if getattr(self, "montage_tile_order", None):
            d["montage_tile_order"] = list(self.montage_tile_order)
        if getattr(self, "montage_tile_z_modes", None):
            d["montage_tile_z_modes"] = dict(self.montage_tile_z_modes)
        if self._qt_widget_alive(getattr(self, "montage_sb_enable", None)):
            d["montage_sb_enable"] = self.montage_sb_enable.isChecked()
        if self._qt_widget_alive(getattr(self, "montage_focus_enable", None)):
            d["montage_focus_enable"] = self.montage_focus_enable.isChecked()
        for key, widget_name in (
            ("montage_focus_zoom", "montage_focus_zoom"),
            ("montage_focus_position", "montage_focus_position"),
            ("montage_focus_color", "montage_focus_color"),
        ):
            w = getattr(self, widget_name, None)
            if self._qt_widget_alive(w):
                d[key] = w.currentText()
        if self._qt_widget_alive(getattr(self, "montage_focus_line_width", None)):
            d["montage_focus_line_width"] = self._safe_spin_value(self.montage_focus_line_width, 5)
        label_settings = self._montage_label_settings()
        d["montage_show_labels"] = label_settings["show"]
        d["montage_label_position"] = label_settings["position"]
        d["montage_label_color"] = label_settings["color_name"]
        d["montage_label_font_size"] = label_settings["font_size"]
        d["montage_export_mode"] = self._montage_export_mode()
        for key, widget_name, default in (
            ("montage_rows", "montage_rows", 1),
            ("montage_cols", "montage_cols", 0),
            ("montage_spacing", "montage_spacing", 4),
            ("montage_border", "montage_border", 0),
            ("montage_sb_pixel_size", "montage_sb_pixel_size", 0.108),
            ("montage_sb_length_um", "montage_sb_length_um", 10.0),
            ("montage_sb_thickness", "montage_sb_thickness", 5),
            ("montage_sb_font_size", "montage_sb_font_size", 20),
        ):
            w = getattr(self, widget_name, None)
            if self._qt_widget_alive(w):
                try:
                    d[key] = float(w.value())
                except Exception:
                    d[key] = default
        order = getattr(self, "montage_order", None)
        if self._qt_widget_alive(order):
            d["montage_order"] = order.currentText()
        for key, widget_name in (
            ("montage_sb_position", "montage_sb_position"),
            ("montage_sb_color", "montage_sb_color"),
        ):
            w = getattr(self, widget_name, None)
            if self._qt_widget_alive(w):
                d[key] = w.currentText()
        sb_show = getattr(self, "montage_sb_show_label", None)
        if self._qt_widget_alive(sb_show):
            d["montage_sb_show_label"] = sb_show.isChecked()
        return d

    def set_session_params(self, data: dict[str, Any]) -> None:
        """Apply saved parameter values (parameters only)."""
        apply_to_widgets(data, [
            ("time_interval", self.time_interval),
            ("yaxis_min", self.yaxis_min),
            ("yaxis_max", self.yaxis_max),
            ("fps", self.fps),
            ("format_tif", self.format_tif),
            ("format_avi", self.format_avi),
            ("format_gif", self.format_gif),
            ("format_mp4", self.format_mp4),
            ("video_mask_contours", self.video_mask_contours),
            ("video_foci_overlay", self.video_foci_overlay),
        ])
        for key, widget_name in (
            ("video_mask_path", "video_mask_path"),
            ("video_foci_path", "video_foci_path"),
        ):
            if key in data:
                w = getattr(self, widget_name, None)
                if self._qt_widget_alive(w):
                    w.setText(str(data[key] or ""))
        if "montage_overlay_enable" in data:
            w = getattr(self, "montage_overlay_enable", None)
            if self._qt_widget_alive(w):
                w.setChecked(bool(data["montage_overlay_enable"]))
        if "montage_overlay_mask_path" in data:
            w = getattr(self, "montage_overlay_mask_path", None)
            if self._qt_widget_alive(w):
                w.setText(str(data["montage_overlay_mask_path"] or ""))
        if "montage_overlay_frame" in data:
            w = getattr(self, "montage_overlay_frame", None)
            if self._qt_widget_alive(w):
                try:
                    w.setValue(int(data["montage_overlay_frame"]))
                except (TypeError, ValueError):
                    pass
        if "montage_z_projection" in data:
            text = str(data["montage_z_projection"] or "")
            self._sync_montage_z_projection_mode(text)
            w = getattr(self, "montage_z_projection", None)
            if self._qt_widget_alive(w):
                if w.findText(text) < 0:
                    text = "Max projection" if "Max" in text else "First Z"
                idx = w.findText(text)
                if idx >= 0:
                    w.setCurrentIndex(idx)
        if "montage_border_color" in data:
            text = str(data["montage_border_color"] or "Gray")
            self._sync_montage_border_color(text)
            w = getattr(self, "montage_border_color", None)
            if self._qt_widget_alive(w):
                idx = w.findText(text)
                if idx >= 0:
                    w.setCurrentIndex(idx)
        self.montage_tile_percentile_overrides = {}
        mtpo = data.get("montage_tile_percentile_overrides")
        if isinstance(mtpo, dict):
            for k, v in mtpo.items():
                try:
                    if not isinstance(v, (list, tuple)) or len(v) != 2:
                        continue
                    p_low = float(v[0])
                    p_high = float(v[1])
                    p_low = max(0.0, min(99.9, p_low))
                    p_high = max(p_low + 0.1, min(100.0, p_high))
                    self.montage_tile_percentile_overrides[str(k)] = (p_low, p_high)
                except Exception:
                    continue
        excluded = data.get("montage_excluded_tile_keys")
        if isinstance(excluded, (list, tuple, set)):
            self.montage_excluded_tile_keys = {str(x) for x in excluded}
        tile_order = data.get("montage_tile_order")
        if isinstance(tile_order, (list, tuple)):
            self.montage_tile_order = [str(x) for x in tile_order]
        z_modes = data.get("montage_tile_z_modes")
        if isinstance(z_modes, dict):
            self.montage_tile_z_modes = {str(k): str(v) for k, v in z_modes.items()}
        if "montage_focus_enable" in data and self._qt_widget_alive(getattr(self, "montage_focus_enable", None)):
            self.montage_focus_enable.setChecked(bool(data["montage_focus_enable"]))
        for key, widget_name in (
            ("montage_focus_zoom", "montage_focus_zoom"),
            ("montage_focus_position", "montage_focus_position"),
            ("montage_focus_color", "montage_focus_color"),
        ):
            if key in data:
                w = getattr(self, widget_name, None)
                if self._qt_widget_alive(w):
                    idx = w.findText(str(data[key] or ""))
                    if idx >= 0:
                        w.setCurrentIndex(idx)
        if "montage_focus_line_width" in data:
            w = getattr(self, "montage_focus_line_width", None)
            if self._qt_widget_alive(w):
                try:
                    w.setValue(int(data["montage_focus_line_width"]))
                except (TypeError, ValueError):
                    pass
        if "montage_show_labels" in data and self._qt_widget_alive(getattr(self, "montage_show_labels", None)):
            self.montage_show_labels.setChecked(bool(data["montage_show_labels"]))
        for key, widget_name in (
            ("montage_label_position", "montage_label_position"),
            ("montage_label_color", "montage_label_color"),
            ("montage_export_mode", "montage_export_mode"),
        ):
            if key in data:
                w = getattr(self, widget_name, None)
                if self._qt_widget_alive(w):
                    idx = w.findText(str(data[key] or ""))
                    if idx >= 0:
                        w.setCurrentIndex(idx)
        if "montage_label_font_size" in data:
            w = getattr(self, "montage_label_font_size", None)
            if self._qt_widget_alive(w):
                try:
                    w.setValue(int(data["montage_label_font_size"]))
                except (TypeError, ValueError):
                    pass
        int_widget_map = {
            "montage_rows": "montage_rows",
            "montage_cols": "montage_cols",
            "montage_spacing": "montage_spacing",
            "montage_border": "montage_border",
            "montage_sb_thickness": "montage_sb_thickness",
            "montage_sb_font_size": "montage_sb_font_size",
        }
        for key, widget_name in int_widget_map.items():
            if key in data:
                w = getattr(self, widget_name, None)
                if self._qt_widget_alive(w):
                    try:
                        w.setValue(int(data[key]))
                    except (TypeError, ValueError):
                        pass
        float_widget_map = {
            "montage_sb_pixel_size": "montage_sb_pixel_size",
            "montage_sb_length_um": "montage_sb_length_um",
        }
        for key, widget_name in float_widget_map.items():
            if key in data:
                w = getattr(self, widget_name, None)
                if self._qt_widget_alive(w):
                    try:
                        w.setValue(float(data[key]))
                    except (TypeError, ValueError):
                        pass
        if "montage_order" in data:
            w = getattr(self, "montage_order", None)
            if self._qt_widget_alive(w):
                idx = w.findText(str(data["montage_order"] or ""))
                if idx >= 0:
                    w.setCurrentIndex(idx)
        for key, widget_name in (
            ("montage_sb_position", "montage_sb_position"),
            ("montage_sb_color", "montage_sb_color"),
        ):
            if key in data:
                w = getattr(self, widget_name, None)
                if self._qt_widget_alive(w):
                    idx = w.findText(str(data[key] or ""))
                    if idx >= 0:
                        w.setCurrentIndex(idx)
        if "montage_sb_show_label" in data and self._qt_widget_alive(getattr(self, "montage_sb_show_label", None)):
            self.montage_sb_show_label.setChecked(bool(data["montage_sb_show_label"]))
        if data.get("montage_sb_enable") and self._qt_widget_alive(getattr(self, "montage_sb_enable", None)):
            self.montage_sb_enable.setChecked(True)
        elif "montage_sb_enable" in data and self._qt_widget_alive(getattr(self, "montage_sb_enable", None)):
            self.montage_sb_enable.setChecked(bool(data["montage_sb_enable"]))
