"""Interactive lineage inspector: movie, tree graph, events, and measurement plots."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from bactomate.gui.image_canvas import ImageCanvas
from bactomate.gui.image_utils import to_qpixmap
from bactomate.gui.layout_helpers import style_compact_action_button, style_module_primary_button
from bactomate.modules.lineage_integration import (
    REVIEW_STATUS_ACCEPTED,
    REVIEW_STATUS_CANDIDATE,
    REVIEW_STATUS_FLAGGED,
    REVIEW_STATUS_REJECTED,
    apply_division_review_status,
    apply_event_review_status,
    area_table_for_lineage_growth,
    biology_label_map,
    calibrate_area_table,
    collapse_lineage_events_for_display,
    compute_lineage_analysis_summary,
    format_biology_division_split,
    refresh_lineage_events_table,
    rebuild_curated_lineage_views,
)
from bactomate.modules.lineage_viz import (
    plot_lineage_cell_cycle_area,
    plot_lineage_growth,
    plot_lineage_metric_timeseries,
    plot_lineage_tree,
)


class _LineageCanvasPickFilter(QObject):
    """Treat a short left click on the movie as cell pick (pan still uses drag)."""

    def __init__(self, inspector: "LineageInspectorDialog") -> None:
        super().__init__(inspector)
        self._inspector = inspector
        self._press_pos = None
        self._pressed = False

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is not self._inspector._canvas:
            return False
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position()
            self._pressed = True
        elif (
            event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
            and self._pressed
            and self._press_pos is not None
        ):
            if (event.position() - self._press_pos).manhattanLength() <= 5:
                image_pos = self._inspector._canvas._widget_to_image(event.position())
                if image_pos is not None:
                    self._inspector._on_canvas_cell_clicked(image_pos[0], image_pos[1])
            self._pressed = False
            self._press_pos = None
        return False


class SortableTableItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numbers numerically and text lexicographically."""

    def __init__(self, value: Any) -> None:
        text = "" if value is None else str(value)
        super().__init__(text)
        self._sort_value = self._coerce_sort_value(value)

    @staticmethod
    def _coerce_sort_value(value: Any) -> tuple[int, Any]:
        try:
            if value is None:
                return (2, "")
            if isinstance(value, float) and np.isnan(value):
                return (2, "")
            return (0, float(value))
        except (TypeError, ValueError):
            text = "" if value is None else str(value).lower()
            return (1, text)

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, SortableTableItem):
            return self._sort_value < other._sort_value
        return super().__lt__(other)


class LineageInspectorDialog(QDialog):
    """Inspect lineage preview or saved tracking outputs."""

    @staticmethod
    def _coalesce_table(tables: dict[str, Any], *keys: str, fallback: Any = None) -> Any:
        """Return the first non-None table value (safe for empty DataFrames)."""
        for key in keys:
            val = tables.get(key)
            if val is not None:
                return val
        return fallback

    @staticmethod
    def _append_table_row(df: Any, row: dict[str, Any]) -> Any:
        """Append one row without concatenating empty placeholder DataFrames."""
        import pandas as pd

        row_df = pd.DataFrame([row])
        if df is None or not hasattr(df, "columns") or len(df) == 0:
            return row_df
        for col in row_df.columns:
            if col not in df.columns:
                df = df.copy()
                df[col] = np.nan
        for col in df.columns:
            if col not in row_df.columns:
                row_df[col] = np.nan
        return pd.concat([df, row_df[df.columns]], ignore_index=True)

    def _load_manual_curation(self) -> None:
        """Load persisted curation bundle (manual assignments, ignores, division reviews)."""
        if self._curation_path is None:
            return
        from bactomate.modules.lineage_integration import load_lineage_curation

        data = load_lineage_curation(self._curation_path)
        if not data:
            return
        import pandas as pd

        edits = data.get("manual_lineage_edits")
        if isinstance(edits, list):
            self._tables["manual_lineage_edits"] = pd.DataFrame(edits)
        ignored = data.get("manual_ignored_untracked")
        if isinstance(ignored, list):
            pairs: set[tuple[int, int]] = set()
            for item in ignored:
                try:
                    pairs.add((int(item[0]), int(item[1])))
                except (TypeError, ValueError, IndexError):
                    continue
            self._tables["manual_ignored_untracked"] = pairs
        division_reviews = data.get("division_reviews")
        if isinstance(division_reviews, list) and division_reviews:
            self._tables["_pending_division_reviews"] = division_reviews

    def _ensure_lineage_calibration(self) -> None:
        """Fill missing lineage calibration from source paths so old previews do not stay in px."""
        cal = dict(self._tables.get("lineage_calibration") or {})
        ps = float(cal.get("pixel_size_um", 0.0) or 0.0)
        tp = float(cal.get("time_per_frame_min", 0.0) or 0.0)
        if ps > 0 and tp > 0:
            return
        try:
            from bactomate.modules.tiff_metadata import read_metadata_from_path, resolve_calibration_for_mask

            for path, is_mask in (
                (self._original_path, False),
                (self._mask_path, True),
            ):
                if path is None or not Path(path).is_file():
                    continue
                if is_mask:
                    p2, t2, _src, _kind = resolve_calibration_for_mask(Path(path))
                else:
                    meta = read_metadata_from_path(Path(path))
                    p2 = float(meta.get("pixel_size_um") or 0.0)
                    t2 = float(meta.get("time_per_frame_min") or 0.0)
                if ps <= 0 and p2 > 0:
                    ps = float(p2)
                if tp <= 0 and t2 > 0:
                    tp = float(t2)
                if ps > 0 and tp > 0:
                    break
        except Exception:
            pass
        if ps > 0 or tp > 0:
            self._tables["lineage_calibration"] = {
                "pixel_size_um": float(ps),
                "time_per_frame_min": float(tp),
            }
            if self._source_lineage_tables is not None:
                self._source_lineage_tables["lineage_calibration"] = self._tables["lineage_calibration"]

    def _sync_manual_edit_counter(self) -> None:
        edits = self._tables.get("manual_lineage_edits")
        if edits is None or not hasattr(edits, "columns") or len(edits) == 0:
            return
        if "manual_track_id" not in edits.columns:
            return
        ids: list[int] = []
        for v in edits["manual_track_id"].dropna().tolist():
            try:
                ids.append(abs(int(v)))
            except (TypeError, ValueError):
                continue
        if ids:
            self._manual_edit_counter = max(self._manual_edit_counter, max(ids))

    def _ensure_loaded_manual_edits_applied(self) -> None:
        """Replay loaded manual assignments into tables used by tree/growth/overlay."""
        import pandas as pd

        edits = self._tables.get("manual_lineage_edits")
        if edits is None or not hasattr(edits, "columns") or len(edits) == 0:
            return
        required = {"manual_track_id", "parent_track_id", "lineage_id", "frame", "untracked_label", "area_px", "y", "x"}
        if not required.issubset(edits.columns):
            return
        for row in edits.itertuples(index=False):
            manual_tid = int(getattr(row, "manual_track_id"))
            parent_tid = int(getattr(row, "parent_track_id"))
            lineage_id = int(getattr(row, "lineage_id"))
            frame = int(getattr(row, "frame"))
            label = int(getattr(row, "untracked_label"))
            area = float(getattr(row, "area_px"))
            y = float(getattr(row, "y"))
            x = float(getattr(row, "x"))

            edges = self._tables.get("lineage_edges")
            if edges is None or not hasattr(edges, "columns"):
                edges = pd.DataFrame(columns=["parent_track_id", "daughter_track_id", "division_frame", "parent_last_frame", "link_status"])
            edge_exists = (
                len(edges) > 0
                and "daughter_track_id" in edges.columns
                and (edges["daughter_track_id"].astype(int) == manual_tid).any()
            )
            if not edge_exists:
                edge_row = {
                    "parent_track_id": parent_tid,
                    "daughter_track_id": manual_tid,
                    "division_frame": frame,
                    "parent_last_frame": max(0, frame - 1),
                    "link_status": "manual",
                }
                self._tables["lineage_edges"] = self._append_table_row(edges, edge_row)

            pts = self._tables.get("lineage_points")
            if pts is None or not hasattr(pts, "columns"):
                pts = pd.DataFrame(columns=["track_id", "frame", "y", "x", "Cell_ID", "lineage_id", "area", "area_source"])
            point_exists = (
                len(pts) > 0
                and {"track_id", "frame"}.issubset(pts.columns)
                and ((pts["track_id"].astype(int) == manual_tid) & (pts["frame"].astype(int) == frame)).any()
            )
            if not point_exists:
                point_row = {
                    "track_id": manual_tid,
                    "frame": frame,
                    "y": y,
                    "x": x,
                    "Cell_ID": f"manual:{label}",
                    "lineage_id": lineage_id,
                    "area": area,
                    "area_source": "manual_mask_px",
                }
                self._tables["lineage_points"] = self._append_table_row(pts, point_row)

            meta = self._tables.get("lineage_tracks_meta")
            if meta is None or not hasattr(meta, "columns"):
                meta = pd.DataFrame(columns=["track_id", "lineage_id", "founder_track_id", "parent_track_id", "daughter_track_ids", "n_daughters", "generation", "branch_depth", "Cell_ID"])
            meta_exists = (
                len(meta) > 0
                and "track_id" in meta.columns
                and (meta["track_id"].astype(int) == manual_tid).any()
            )
            if not meta_exists:
                meta_row = {
                    "track_id": manual_tid,
                    "lineage_id": lineage_id,
                    "founder_track_id": lineage_id,
                    "parent_track_id": parent_tid,
                    "daughter_track_ids": "[]",
                    "n_daughters": 0,
                    "generation": self._parent_generation(parent_tid) + 1,
                    "branch_depth": 0,
                    "Cell_ID": f"manual:{label}",
                }
                self._tables["lineage_tracks_meta"] = self._append_table_row(meta, meta_row)

            if self._tracks_df is not None and hasattr(self._tracks_df, "columns"):
                track_exists = (
                    len(self._tracks_df) > 0
                    and "track_id" in self._tracks_df.columns
                    and (self._tracks_df["track_id"].astype(int) == manual_tid).any()
                )
                if not track_exists:
                    track_row = {"track_id": manual_tid, "frame": frame, "y": y, "x": x, "lineage_id": lineage_id}
                    self._tracks_df = self._append_table_row(self._tracks_df, track_row)

    def _sync_external_lineage_tables(self) -> None:
        """Keep preview_state lineage tables alive after closing/reopening the inspector."""
        if self._source_lineage_tables is None:
            return
        for key in (
            "manual_lineage_edits",
            "manual_ignored_untracked",
            "lineage_calibration",
            "lineage_edges",
            "lineage_points",
            "lineage_tracks_meta",
            "lineage_events",
            "lineage_tracks_meta_curated",
            "lineage_edges_curated",
            "lineage_growth_timeseries_curated",
            "lineage_tree_nodes_curated",
            "lineage_tree_edges_curated",
            "biology_track_labels",
            "founder_by_track_curated",
        ):
            if key in self._tables:
                self._source_lineage_tables[key] = self._tables[key]

    def _save_manual_curation(self) -> None:
        """Persist curation bundle and sync in-memory preview/output tables."""
        self._finalize_curation()

    def _finalize_curation(self) -> None:
        """Write curation sidecar and push tables back to the preview/output holder."""
        self._sync_external_lineage_tables()
        if self._curation_path is not None:
            try:
                from bactomate.modules.lineage_integration import save_lineage_curation

                save_lineage_curation(self._curation_path, self._tables, mask_path=self._mask_path)
            except Exception:
                pass
        if self._source_lineage_tables is not None:
            self._source_lineage_tables["_tracks_df_curated"] = self._tracks_df
            edges = self._tables.get("lineage_edges")
            if edges is not None:
                self._source_lineage_tables["_edges_df_curated"] = edges

    def accept(self) -> None:
        self._finalize_curation()
        super().accept()

    def reject(self) -> None:
        self._finalize_curation()
        super().reject()

    def closeEvent(self, event) -> None:
        self._finalize_curation()
        super().closeEvent(event)

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        frames_rgb: np.ndarray,
        tracks_df: Any,
        edges_df: Any = None,
        lineage_tables: Optional[dict[str, Any]] = None,
        stack_images: Any = None,
        masks_input: Any = None,
        masks_tracked: Any = None,
        founder_by_track: Optional[dict[int, int]] = None,
        curation_path: Optional[Path] = None,
        mask_path: Optional[Path] = None,
        original_path: Optional[Path] = None,
        title: str = "Lineage review",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(1180, 760)
        self.resize(1420, 860)
        self._frames = np.asarray(frames_rgb, dtype=np.uint8)
        self._tracks_df = tracks_df
        self._edges_df = edges_df
        self._stack_images = stack_images
        self._masks_input = masks_input
        self._masks_tracked = masks_tracked
        self._founder_by_track = founder_by_track or {}
        self._source_lineage_tables = lineage_tables if isinstance(lineage_tables, dict) else None
        self._curation_path = Path(curation_path) if curation_path is not None else None
        self._mask_path = Path(mask_path) if mask_path is not None else None
        self._original_path = Path(original_path) if original_path is not None else None
        self._tables = dict(lineage_tables or {})
        self._ensure_lineage_calibration()
        self._load_manual_curation()
        self._selected_track_id: Optional[int] = None
        self._selected_event_key: Optional[tuple[int, int]] = None  # (parent_track_id, division_frame)
        self._selected_untracked_key: Optional[tuple[int, int]] = None
        self._manual_edit_counter = 0
        self._sync_manual_edit_counter()
        self._ensure_loaded_manual_edits_applied()
        self._current_frame = 0
        self._pinned_cell_hit: Optional[dict[str, Any]] = None
        self._overlay_layers = {
            "masks": True,
            "selected_label": True,
            "cell_labels": False,
            "ids": False,
            "centers": False,
            "untracked": True,
        }
        self._curated_strict = False
        self._build_ui()
        self._install_close_shortcuts()
        self._refresh_curated_views()
        self._populate_track_table()
        self._populate_events_table()
        self._populate_untracked_table()
        self._populate_qc_table()
        if self._frames.shape[0] > 0:
            self._show_frame(0)

    def _install_close_shortcuts(self) -> None:
        close_shortcut = QShortcut(QKeySequence.StandardKey.Close, self)
        close_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        close_shortcut.activated.connect(self.accept)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(max(0, int(self._frames.shape[0]) - 1))
        self._frame_slider.valueChanged.connect(self._on_frame_changed)
        self._frame_label = QLabel("Frame 1")
        top.addWidget(QLabel("Frame"))
        top.addWidget(self._frame_slider, 1)
        top.addWidget(self._frame_label)
        root.addLayout(top)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left.setMinimumWidth(430)
        left_lay = QVBoxLayout(left)
        overlay_row = QHBoxLayout()
        overlay_row.addWidget(QLabel("Overlay:"))
        for key, label, tip in (
            ("masks", "Masks", "Show founder/track-colored cell mask overlay."),
            ("selected_label", "Selected label", "Compact name on the selected/pinned cell only."),
            ("cell_labels", "All labels", "Show biology names on every cell (can overlap when crowded)."),
            ("ids", "Track IDs", "Show raw Trackastra trajectory IDs (Advanced/debug only)."),
            ("centers", "Centers", "Show small centroid markers. These are detections, not lineage identity."),
            ("untracked", "Untracked", "Show orange segmented objects not linked by Trackastra."),
        ):
            cb = QCheckBox(label)
            cb.setChecked(bool(self._overlay_layers.get(key, False)))
            cb.setToolTip(tip)
            cb.stateChanged.connect(lambda _state, k=key, w=cb: self._set_overlay_layer(k, w.isChecked()))
            overlay_row.addWidget(cb)
        overlay_row.addStretch(1)
        left_lay.addLayout(overlay_row)
        self._viewer_help_label = QLabel(
            "Hover any cell for a quick profile. Click to pin selection. Curate fixes divisions and orange cells."
        )
        self._viewer_help_label.setWordWrap(True)
        self._viewer_help_label.setStyleSheet("color: #9fb0c4; padding: 4px 0;")
        left_lay.addWidget(self._viewer_help_label)
        self._canvas = ImageCanvas()
        self._canvas.setToolTip("Zoom: mouse wheel. Pan: click-drag. Hover a cell for its profile; click to pin.")
        left_lay.addWidget(self._canvas, 1)
        self._cell_profile_label = QLabel("Hover a cell for a quick tooltip. Click to pin a cell profile here.")
        self._cell_profile_label.setWordWrap(True)
        self._cell_profile_label.setTextFormat(Qt.TextFormat.RichText)
        self._cell_profile_label.setMinimumHeight(72)
        self._cell_profile_label.setMaximumHeight(130)
        self._cell_profile_label.setStyleSheet(
            "color: #e8eef6; padding: 8px 10px; background: #1a2330; border: 1px solid #2f3f52; border-radius: 8px;"
        )
        left_lay.addWidget(self._cell_profile_label)
        self._install_canvas_interaction()
        split.addWidget(left)

        right = QWidget()
        right.setMinimumWidth(620)
        right_lay = QVBoxLayout(right)
        tabs = QTabWidget()
        tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #2f3f52;
                border-radius: 8px;
                background: #151b22;
                top: -1px;
            }
            QTabBar::tab {
                background: #202733;
                color: #d6dee8;
                border: 1px solid #2f3f52;
                border-bottom: none;
                padding: 8px 18px;
                min-width: 86px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: #2d8cff;
                color: white;
                border-color: #4aa3ff;
                font-weight: 600;
            }
            QTabBar::tab:hover:!selected {
                background: #293445;
            }
            """
        )

        curate_tab = QWidget()
        curate_lay = QVBoxLayout(curate_tab)
        curate_lay.addWidget(
            QLabel(
                "A division means: one mother cell split into one or more daughters. "
                "Each row is one division event (all daughters at the same split frame are grouped). "
                "Click a row → movie jumps to the split frame.<br><br>"
                "<b>Your review</b> (Accept / Reject / Unsure) is your QC verdict. "
                "<b>Link source</b> stays “candidate” for Trackastra-detected links — that column does not change when you accept.<br><br>"
                "By default, <b>candidate</b> links are already used in Analysis and export. "
                "Reject removes a wrong link. Accept records confirmation. "
                "Enable <i>Accepted divisions only</i> on the Analysis tab to require acceptance before a link is used."
            )
        )
        curate_lay.addWidget(QLabel("Division links (read as: mother cell → daughter cell @ frame)"))
        self._review_status_label = QLabel(
            "Example: “Cell 1 → Cell 1.2 @ frame 12” means the founder family split and one daughter appeared at frame 12. "
            "If one mother has two daughters at the same split, the row groups them: Cell 1 → Cell 1.1 + Cell 1.2."
        )
        self._review_status_label.setWordWrap(True)
        self._review_status_label.setStyleSheet("color: #8fbfff; font-weight: 600;")
        curate_lay.addWidget(self._review_status_label)
        self._events_table = QTableWidget()
        self._events_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._events_table.setSortingEnabled(True)
        self._events_table.horizontalHeader().setSortIndicatorShown(True)
        self._events_table.cellClicked.connect(self._on_event_row_clicked)
        self._events_table.currentCellChanged.connect(self._on_event_current_cell_changed)
        curate_lay.addWidget(self._events_table, 2)
        review_row = QHBoxLayout()
        self._accept_all_divisions_btn = QPushButton("Accept all visible")
        style_compact_action_button(self._accept_all_divisions_btn, horizontal_padding=20)
        self._accept_all_divisions_btn.setToolTip(
            "Mark all currently listed division links as accepted. Use this when the overlay/movie looks correct."
        )
        self._accept_all_divisions_btn.clicked.connect(self._accept_all_visible_divisions)
        review_row.addWidget(self._accept_all_divisions_btn)
        for label, status in (
            ("Accept", REVIEW_STATUS_ACCEPTED),
            ("Reject", REVIEW_STATUS_REJECTED),
            ("Unsure", REVIEW_STATUS_FLAGGED),
        ):
            btn = QPushButton(label)
            style_compact_action_button(btn, horizontal_padding=24)
            btn.clicked.connect(lambda _checked=False, s=status: self._set_selected_event_review(s))
            review_row.addWidget(btn)
        curate_lay.addLayout(review_row)
        curate_lay.addWidget(QLabel("Untracked cells"))
        self._untracked_hint = QLabel(
            "Orange = segmented in your filter mask but NOT linked to a lineage by Trackastra. "
            "Select a row to jump to that frame (white outline). Assign it to a nearby parent track, or ignore it."
        )
        self._untracked_hint.setWordWrap(True)
        curate_lay.addWidget(self._untracked_hint)
        self._untracked_table = QTableWidget()
        self._untracked_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._untracked_table.setSortingEnabled(True)
        self._untracked_table.horizontalHeader().setSortIndicatorShown(True)
        self._untracked_table.cellClicked.connect(self._on_untracked_row_clicked)
        curate_lay.addWidget(self._untracked_table, 2)
        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel("Parent cell"))
        self._untracked_parent_combo = QComboBox()
        self._untracked_parent_combo.setToolTip(
            "Choose the mother cell this orange object came from (biology label, e.g. Cell 1.1)."
        )
        self._untracked_parent_combo.setEnabled(False)
        manual_row.addWidget(self._untracked_parent_combo, 1)
        self._assign_untracked_btn = QPushButton("Assign orange cell to parent")
        self._ignore_untracked_btn = QPushButton("Ignore selected")
        for btn in (self._assign_untracked_btn, self._ignore_untracked_btn):
            style_compact_action_button(btn, horizontal_padding=18)
            manual_row.addWidget(btn)
        self._assign_untracked_btn.clicked.connect(self._assign_selected_untracked)
        self._ignore_untracked_btn.clicked.connect(self._ignore_selected_untracked)
        curate_lay.addLayout(manual_row)
        self._untracked_status_label = QLabel("Select an orange untracked object before assigning or ignoring it.")
        self._untracked_status_label.setWordWrap(True)
        self._untracked_status_label.setStyleSheet("color: #8fbfff; font-weight: 600;")
        curate_lay.addWidget(self._untracked_status_label)
        tabs.addTab(curate_tab, "Curate")

        analysis_scroll = QScrollArea()
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        analysis_content = QWidget()
        analysis_lay = QVBoxLayout(analysis_content)
        analysis_lay.addWidget(
            QLabel(
                "After Curate, use Analysis to sanity-check the whole family — not individual track IDs.\n"
                "• Tree plot = who divided from whom (biology names on nodes).\n"
                "• Cell-cycle trace = each branch’s calibrated cell area through time, with division connectors.\n"
                "• Family growth = summed area of the whole family over time.\n"
                "Scroll this panel if the lower plot is clipped."
            )
        )
        self._analysis_summary_label = QLabel("Select a track to see lineage summary.")
        self._analysis_summary_label.setWordWrap(True)
        self._analysis_summary_label.setTextFormat(Qt.TextFormat.RichText)
        self._analysis_summary_label.setMaximumHeight(150)
        self._analysis_summary_label.setStyleSheet(
            "color: #e8eef6; padding: 10px; background: #1a2330; border: 1px solid #2f3f52; border-radius: 8px;"
        )
        analysis_lay.addWidget(self._analysis_summary_label)
        curation_row = QHBoxLayout()
        self._curated_strict_cb = QCheckBox("Accepted divisions only (ignore unreviewed candidates)")
        self._curated_strict_cb.setToolTip(
            "Off (default): use all Trackastra candidate links except those you Reject.\n"
            "On: use only Accepted (+ manual) links in tree and growth plots."
        )
        self._curated_strict_cb.stateChanged.connect(self._on_curated_strict_changed)
        curation_row.addWidget(self._curated_strict_cb)
        curation_row.addStretch(1)
        analysis_lay.addLayout(curation_row)
        self._plot_mode_combo = QComboBox()
        self._rebuild_plot_mode_choices()
        self._plot_mode_combo.currentIndexChanged.connect(self._refresh_plots)
        analysis_lay.addWidget(self._plot_mode_combo)
        self._metric_combo = QComboBox()
        self._metric_combo.addItems(self._measurement_column_choices())
        self._metric_combo.currentIndexChanged.connect(self._refresh_plots)
        self._metric_combo.setVisible(False)
        self._plot_mode_combo.currentIndexChanged.connect(self._sync_metric_combo_visibility)
        analysis_lay.addWidget(self._metric_combo)
        self._plot_notice = QLabel("")
        self._plot_notice.setWordWrap(True)
        self._plot_notice.setStyleSheet("color: #f0b429; font-weight: 600;")
        self._plot_notice.setVisible(False)
        analysis_lay.addWidget(self._plot_notice)
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._tree_plot_help = QLabel(
                "<b>Family tree</b> — each dot is one cell branch. Vertical position ≈ generation; "
                "lines show mother→daughter links after curation."
            )
            self._tree_plot_help.setWordWrap(True)
            self._tree_plot_help.setStyleSheet("color: #9fb0c4; padding: 2px 0;")
            analysis_lay.addWidget(self._tree_plot_help)
            self._tree_fig = Figure(figsize=(5.2, 2.4), tight_layout=True)
            self._tree_canvas = FigureCanvasQTAgg(self._tree_fig)
            self._tree_canvas.setMinimumHeight(200)
            self._tree_ax = self._tree_fig.add_subplot(111)
            analysis_lay.addWidget(self._tree_canvas)
            self._growth_plot_help = QLabel(
                "<b>Cell-cycle trace</b> — calibrated area of each cell branch over time. "
                "At divisions the mother branch connects to daughter branches, giving the expected sawtooth/cycle pattern. "
                "Use the dropdown above for per-cell metrics (Advanced/debug). "
                "Use <i>Lineage total area</i> for the summed family-size check."
            )
            self._growth_plot_help.setWordWrap(True)
            self._growth_plot_help.setStyleSheet("color: #9fb0c4; padding: 6px 0 2px 0;")
            analysis_lay.addWidget(self._growth_plot_help)
            self._fig = Figure(figsize=(7.0, 3.0), tight_layout=True)
            self._plot_canvas = FigureCanvasQTAgg(self._fig)
            self._plot_canvas.setMinimumHeight(260)
            self._ax = self._fig.add_subplot(111)
            analysis_lay.addWidget(self._plot_canvas)
        except Exception:
            self._tree_canvas = QLabel("(Plots unavailable — install matplotlib)")
            self._tree_ax = None
            self._plot_canvas = self._tree_canvas
            self._ax = None
            analysis_lay.addWidget(self._tree_canvas, 1)
        export_row = QHBoxLayout()
        export_row.addStretch(1)
        self._export_analysis_plot_btn = QPushButton("Export current plot PNG...")
        style_compact_action_button(self._export_analysis_plot_btn, horizontal_padding=18)
        self._export_analysis_plot_btn.setToolTip("Save the currently displayed Analysis plot as a PNG.")
        self._export_analysis_plot_btn.clicked.connect(self._export_current_analysis_plot)
        export_row.addWidget(self._export_analysis_plot_btn)
        analysis_lay.addLayout(export_row)
        analysis_lay.addStretch(1)
        analysis_scroll.setWidget(analysis_content)
        tabs.addTab(analysis_scroll, "Analysis")

        advanced_scroll = QScrollArea()
        advanced_scroll.setWidgetResizable(True)
        advanced_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        advanced_content = QWidget()
        advanced_lay = QVBoxLayout(advanced_content)
        advanced_lay.addWidget(
            QLabel(
                "Advanced = export/debug only. You do not need this tab for normal review.\n"
                "• Tracks: internal Trackastra IDs + tree position (Cell name is the biology label).\n"
                "• Division events (raw): one row per daughter link, with areas and QC fields.\n"
                "• QC flags: suspicious links (far daughters, terminal-frame splits, track gaps)."
            )
        )
        advanced_lay.addWidget(QLabel("Tracks — double-click a row to select that branch in the movie"))
        self._track_table = QTableWidget()
        self._track_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._track_table.setSortingEnabled(True)
        self._track_table.horizontalHeader().setSortIndicatorShown(True)
        self._track_table.cellDoubleClicked.connect(self._on_track_row_clicked)
        advanced_lay.addWidget(self._track_table, 2)
        advanced_lay.addWidget(QLabel("Division events (raw)"))
        self._advanced_events_table = QTableWidget()
        self._advanced_events_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._advanced_events_table.setSortingEnabled(True)
        self._advanced_events_table.horizontalHeader().setSortIndicatorShown(True)
        advanced_lay.addWidget(self._advanced_events_table, 2)
        advanced_lay.addWidget(QLabel("QC flags"))
        self._qc_table = QTableWidget()
        self._qc_table.setSortingEnabled(True)
        self._qc_table.horizontalHeader().setSortIndicatorShown(True)
        advanced_lay.addWidget(self._qc_table, 2)
        advanced_lay.addStretch(1)
        advanced_scroll.setWidget(advanced_content)
        tabs.addTab(advanced_scroll, "Advanced")

        tabs.setCurrentWidget(analysis_scroll)
        right_lay.addWidget(tabs, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 6)
        split.setSizes([560, 860])
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        close_btn = QPushButton("Close")
        style_module_primary_button(close_btn)
        close_btn.setMinimumHeight(34)
        close_btn.clicked.connect(self.accept)
        root.addWidget(close_btn)

    def _measurement_column_choices(self) -> list[str]:
        from bactomate.modules.lineage_viz import discover_lineage_metric_columns

        pts = self._tables.get("lineage_points")
        if pts is None or not hasattr(pts, "columns"):
            return ["track presence"]
        discovered = discover_lineage_metric_columns(pts)
        choices = [f"Area ({self._lineage_area_unit_label()})"]
        for group in ("fi", "foci", "coloc", "other"):
            choices.extend(discovered.get(group, []))
        if len(choices) == 1:
            choices.append("y position")
        return choices[:20]

    def _rebuild_plot_mode_choices(self) -> None:
        from bactomate.modules.lineage_viz import discover_lineage_metric_columns

        unit = self._lineage_area_unit_label()
        modes = [
            f"Cell-cycle area trace ({unit})",
            f"Lineage total area ({unit})",
            f"Log lineage area ({unit})",
        ]
        pts = self._tables.get("lineage_points")
        if pts is not None and hasattr(pts, "columns"):
            discovered = discover_lineage_metric_columns(pts)
            for c in discovered.get("fi", []):
                modes.append(f"Lineage FI — {c}")
            for c in discovered.get("foci", []):
                modes.append(f"Lineage foci — {c}")
            for c in discovered.get("coloc", []):
                modes.append(f"Lineage coloc — {c}")
        modes.append("Per-track metric")
        if hasattr(self, "_plot_mode_combo"):
            current = self._plot_mode_combo.currentText()
            self._plot_mode_combo.blockSignals(True)
            self._plot_mode_combo.clear()
            self._plot_mode_combo.addItems(modes)
            idx = self._plot_mode_combo.findText(current)
            if idx < 0:
                idx = 0
            self._plot_mode_combo.setCurrentIndex(idx)
            self._plot_mode_combo.blockSignals(False)

    def _sync_metric_combo_visibility(self) -> None:
        if not hasattr(self, "_metric_combo") or not hasattr(self, "_plot_mode_combo"):
            return
        mode = self._plot_mode_combo.currentText()
        show = mode == "Per-track metric" or mode.startswith("Lineage FI") or mode.startswith("Lineage foci") or mode.startswith("Lineage coloc")
        self._metric_combo.setVisible(show)

    def _on_curated_strict_changed(self) -> None:
        self._curated_strict = bool(self._curated_strict_cb.isChecked()) if hasattr(self, "_curated_strict_cb") else False
        self._refresh_curated_views()

    def _unresolved_untracked_count(self) -> int:
        return sum(1 for rec in self._untracked_records() if rec.get("action") == "unresolved")

    def _biology_labels(self) -> dict[int, str]:
        labels_df = self._coalesce_table(self._tables, "biology_track_labels")
        return biology_label_map(labels_df)

    def _biology_label(self, track_id: Optional[int], *, fallback_track_id: bool = False) -> str:
        if track_id is None:
            return "—"
        tid = int(track_id)
        lbl = self._biology_labels().get(tid)
        if lbl:
            return lbl
        return f"track {tid}" if fallback_track_id else str(tid)

    def _sync_lineage_events(self) -> None:
        self._tables["lineage_events"] = refresh_lineage_events_table(
            self._tables,
            self._tracks_df,
            preserve_review=True,
        )

    def _refresh_curated_views(self) -> None:
        edges = self._tables.get("lineage_edges")
        if edges is None:
            edges = self._edges_df
        if edges is not None and "lineage_edges" not in self._tables:
            self._tables["lineage_edges"] = edges
        self._sync_lineage_events()
        pending = self._tables.pop("_pending_division_reviews", None)
        if pending:
            events = self._tables.get("lineage_events")
            if events is not None and hasattr(events, "columns") and len(events) > 0:
                for rev in pending:
                    if not isinstance(rev, dict):
                        continue
                    p = rev.get("parent_track_id")
                    d = rev.get("daughter_track_id")
                    status = str(rev.get("review_status", REVIEW_STATUS_CANDIDATE)).lower()
                    if p is None or d is None:
                        continue
                    events = apply_event_review_status(
                        events,
                        parent_track_id=int(p),
                        daughter_track_id=int(d),
                        review_status=status,
                    )
                self._tables["lineage_events"] = events
        curated = rebuild_curated_lineage_views(
            self._tables,
            self._tracks_df,
            strict=self._curated_strict,
        )
        self._tables.update(curated)
        self._update_analysis_summary()

    def _founder_lineage_ids(self) -> list[int]:
        lids: set[int] = set()
        pts = self._tables.get("lineage_points")
        if pts is not None and hasattr(pts, "columns") and "lineage_id" in pts.columns:
            for v in pts["lineage_id"].dropna().unique():
                try:
                    lids.add(int(v))
                except (TypeError, ValueError):
                    continue
        summary = self._tables.get("lineage_summary")
        if summary is not None and hasattr(summary, "columns") and "lineage_id" in summary.columns:
            for v in summary["lineage_id"].dropna().unique():
                try:
                    lids.add(int(v))
                except (TypeError, ValueError):
                    continue
        return sorted(lids)

    def _lineage_display_label(self, lineage_id: Optional[int]) -> str:
        if lineage_id is None:
            return "—"
        founders = self._founder_lineage_ids()
        if len(founders) == 1:
            return "Lineage 1"
        if int(lineage_id) in founders:
            return f"Lineage {founders.index(int(lineage_id)) + 1}"
        return f"Lineage ?"

    def _cell_id_for_track(self, track_id: Optional[int]) -> Optional[str]:
        if track_id is None:
            return None
        pts = self._tables.get("lineage_points")
        if pts is None or not hasattr(pts, "columns") or "Cell_ID" not in pts.columns:
            return None
        sub = pts[pts["track_id"].astype(int) == int(track_id)].head(1)
        if len(sub) == 0:
            return None
        val = sub.iloc[0]["Cell_ID"]
        return None if val is None else str(val)

    def _format_analysis_summary_text(self, summary: dict[str, Any]) -> str:
        lid = summary.get("lineage_id")
        if lid is None:
            return (
                "Select any cell in the movie, or click a division row in Curate, to analyze that family."
            )
        display = self._lineage_display_label(lid)
        founder_lbl = self._biology_label(int(lid), fallback_track_id=True)
        lines = [f"<b>{display}</b> — founder <b>{founder_lbl}</b>"]
        lines.append(
            f"Internal founder track ID: {lid} "
            "(Trackastra handle — see Advanced tab for raw IDs)."
        )
        cell_id = self._cell_id_for_track(int(lid))
        if cell_id is not None and not str(cell_id).startswith("manual:"):
            lines.append(f"Analyze Cell_ID at founder: {cell_id}")
        start_f = summary.get("start_frame")
        end_f = summary.get("end_frame")
        n_desc = summary.get("n_descendants_final")
        n_div = int(summary.get("n_divisions_accepted", 0) or 0)
        if start_f is not None and end_f is not None:
            lines.append(f"Movie span: frame {int(start_f) + 1} → {int(end_f) + 1}")
        if n_desc is not None:
            lines.append(f"Cells in family at end: {n_desc}")
        lines.append(
            f"Divisions curated: {n_div} accepted, "
            f"{summary.get('n_divisions_rejected', 0)} rejected, "
            f"{summary.get('untracked_unresolved', 0)} orange cells still unresolved"
        )
        unit = summary.get("area_unit") or "px"
        if summary.get("area_start") is not None and summary.get("area_end") is not None:
            fold = summary.get("area_fold_change")
            fold_txt = f"{fold:.2f}×" if fold is not None else "—"
            lines.append(
                f"Total family area: {summary['area_start']:.1f} → {summary['area_end']:.1f} {unit} ({fold_txt})"
            )
            if unit == "px":
                cal = self._tables.get("lineage_calibration") or {}
                if float(cal.get("pixel_size_um", 0.0) or 0.0) <= 0:
                    lines.append(
                        "Area is in pixels because no µm calibration / Analyze area column was linked."
                    )
        if summary.get("growth_rate") is not None:
            gu = summary.get("growth_unit") or ""
            lines.append(f"Growth rate: {summary['growth_rate']:.4g} {gu}".strip())
        if summary.get("doubling_time") is not None:
            lines.append(f"Doubling time: {summary['doubling_time']:.4g}")
        if n_desc is not None and start_f is not None and end_f is not None:
            lines.append(
                f"<i>Quick read: 1 founder → {n_desc} cells over "
                f"{int(end_f) - int(start_f) + 1} frames; {n_div} accepted division event(s). "
                "See tree (who divided) and growth plot (family size over time) below.</i>"
            )
        return "<br>".join(lines)

    def _update_analysis_summary(self) -> None:
        if not hasattr(self, "_analysis_summary_label"):
            return
        summary = compute_lineage_analysis_summary(
            self._tables,
            lineage_id=self._selected_lineage_id(),
            growth_ts=self._tables.get("lineage_growth_timeseries_curated"),
            untracked_unresolved=self._unresolved_untracked_count(),
        )
        self._analysis_summary_label.setText(self._format_analysis_summary_text(summary))

    def _auto_select_default_lineage(self) -> None:
        founders = self._founder_lineage_ids()
        if len(founders) != 1 or self._selected_track_id is not None:
            return
        lid = int(founders[0])
        pts = self._tables.get("lineage_points")
        if pts is not None and hasattr(pts, "columns") and "lineage_id" in pts.columns:
            sub = pts[pts["lineage_id"].astype(int) == lid]
            if len(sub) and "track_id" in sub.columns:
                try:
                    self._selected_track_id = int(sub.iloc[0]["track_id"])
                    return
                except (TypeError, ValueError):
                    pass
        self._selected_track_id = lid

    def _populate_qc_table(self) -> None:
        if not hasattr(self, "_qc_table"):
            return
        qc = self._tables.get("lineage_qc")
        if qc is None or not hasattr(qc, "columns") or len(qc) == 0:
            self._qc_table.setRowCount(0)
            return
        cols = list(qc.columns)[:12]
        self._qc_table.setSortingEnabled(False)
        self._qc_table.setColumnCount(len(cols))
        self._qc_table.setHorizontalHeaderLabels([self._friendly_table_header(c) for c in cols])
        self._qc_table.setRowCount(len(qc))
        for ri, row in enumerate(qc.itertuples(index=False)):
            for ci, col in enumerate(cols):
                self._qc_table.setItem(ri, ci, SortableTableItem(getattr(row, col, "")))
        self._qc_table.setSortingEnabled(True)

    def _lineage_area_unit_label(self) -> str:
        from bactomate.modules.lineage_integration import resolve_lineage_area_unit

        growth_ts = self._tables.get("lineage_growth_timeseries_curated")
        if growth_ts is None:
            growth_ts = self._tables.get("lineage_growth_timeseries")
        return resolve_lineage_area_unit(self._tables, growth_ts=growth_ts)

    def _untracked_records(self) -> list[dict[str, Any]]:
        if self._masks_input is None or self._masks_tracked is None:
            return []
        inp = np.asarray(self._masks_input)
        tr = np.asarray(self._masks_tracked)
        if inp.ndim != 3 or tr.ndim != 3 or inp.shape[:3] != tr.shape[:3]:
            return []
        ignored = set(self._tables.get("manual_ignored_untracked", set()) or set())
        edits = self._tables.get("manual_lineage_edits")
        assigned = set()
        if edits is not None and hasattr(edits, "columns") and len(edits) > 0:
            for row in edits.itertuples(index=False):
                assigned.add((int(getattr(row, "frame")), int(getattr(row, "untracked_label"))))
        records: list[dict[str, Any]] = []
        tracks = self._tracks_df
        for frame in range(int(inp.shape[0])):
            input_frame = inp[frame]
            tracked_frame = tr[frame]
            for raw_label in np.unique(input_frame):
                raw_label = int(raw_label)
                if raw_label <= 0:
                    continue
                key = (int(frame), int(raw_label))
                region = input_frame == raw_label
                if np.any(region & (tracked_frame > 0)):
                    continue
                ys, xs = np.nonzero(region)
                if len(ys) == 0:
                    continue
                cy = float(np.mean(ys))
                cx = float(np.mean(xs))
                nearest_tid = None
                nearest_dist = np.nan
                if tracks is not None and hasattr(tracks, "columns") and "frame" in tracks.columns and "track_id" in tracks.columns and "y" in tracks.columns and "x" in tracks.columns:
                    sub = tracks[tracks["frame"].astype(int) == int(frame)]
                    if len(sub) == 0:
                        prev = tracks[tracks["frame"].astype(int) < int(frame)]
                        if len(prev) > 0:
                            last_frame = int(prev["frame"].astype(int).max())
                            sub = prev[prev["frame"].astype(int) == last_frame]
                    best = None
                    for row in sub.itertuples(index=False):
                        tid = int(getattr(row, "track_id"))
                        if tid < 0:
                            continue
                        dy = float(getattr(row, "y")) - cy
                        dx = float(getattr(row, "x")) - cx
                        dist = float((dy * dy + dx * dx) ** 0.5)
                        if best is None or dist < best[0]:
                            best = (dist, tid)
                    if best is not None:
                        nearest_dist, nearest_tid = best
                if key in assigned:
                    action = "assigned"
                elif key in ignored:
                    action = "ignored"
                else:
                    action = "unresolved"
                records.append(
                    {
                        "frame": int(frame),
                        "label": int(raw_label),
                        "area_px": int(region.sum()),
                        "y": cy,
                        "x": cx,
                        "nearest_track_id": nearest_tid,
                        "distance_px": nearest_dist,
                        "action": action,
                    }
                )
        return records

    @staticmethod
    def _friendly_table_header(column: str) -> str:
        labels = {
            "biology_label": "Cell name",
            "track_id": "Tracker ID",
            "parent_track_id": "Parent track ID",
            "daughter_track_id": "Daughter track ID",
            "division_frame": "split frame (internal)",
            "division_frame_1based": "split frame",
            "lineage_id": "Founder track ID",
            "founder_track_id": "Founder track ID",
            "area_asymmetry": "daughter/mother area ratio",
            "division_distance_px": "mother–daughter distance (px)",
            "division_distance_um": "mother–daughter distance (µm)",
            "review_status": "your review",
            "link_status": "link source",
            "split_link": "mother → daughter",
            "generation": "generation",
            "branch_depth": "subtree depth",
            "n_daughters": "# daughters",
            "frame": "frame (internal)",
            "frame_1based": "frame",
            "label": "mask label",
            "assigned_parent_track_id": "assigned parent cell",
            "nearest_track_id": "nearest track",
            "distance_px": "distance (px)",
            "flag_type": "QC flag",
            "severity": "severity",
            "related_track_id": "related track",
            "message": "explanation",
            "has_post_frame": "next frame exists?",
            "qc_warning": "QC code",
            "mother_area_before": "mother area before split",
            "daughter_area_after": "daughter area after split",
            "mother_y": "mother Y",
            "mother_x": "mother X",
            "daughter_y": "daughter Y",
            "daughter_x": "daughter X",
            "Cell_ID": "Analyze Cell_ID",
        }
        return labels.get(str(column), str(column))

    def _manual_edit_for_key(self, frame: int, label: int) -> Optional[dict[str, Any]]:
        edits = self._tables.get("manual_lineage_edits")
        if edits is None or not hasattr(edits, "columns") or len(edits) == 0:
            return None
        if not {"frame", "untracked_label"}.issubset(edits.columns):
            return None
        sub = edits[
            (edits["frame"].astype(int) == int(frame))
            & (edits["untracked_label"].astype(int) == int(label))
        ]
        if len(sub) == 0:
            return None
        return dict(sub.iloc[0])

    def _founder_color_for_lineage(self, lineage_id: int) -> tuple[int, int, int]:
        colors = self._track_color_map()
        for tid, color in colors.items():
            if self._lineage_id_for_track(int(tid)) == int(lineage_id):
                return color
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("tab20")
        c = cmap((int(lineage_id) % 20) / 20.0)
        return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))

    def _populate_untracked_table(self) -> None:
        records = self._untracked_records()
        cols = ["frame_1based", "label", "assigned_parent_track_id", "action"]
        table = self._untracked_table
        table.setSortingEnabled(False)
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels([self._friendly_table_header(c) for c in cols])
        table.setRowCount(len(records))
        for ri, rec in enumerate(records):
            key = (int(rec["frame"]), int(rec["label"]))
            for ci, col in enumerate(cols):
                if col == "frame_1based":
                    val = int(rec["frame"]) + 1
                elif col == "assigned_parent_track_id":
                    manual = self._manual_edit_for_key(int(rec["frame"]), int(rec["label"]))
                    if manual is None:
                        val = ""
                    else:
                        val = self._biology_label(
                            manual.get("parent_track_id"),
                            fallback_track_id=True,
                        )
                else:
                    val = rec.get(col, "")
                if isinstance(val, float):
                    val = f"{val:.2f}" if np.isfinite(val) else ""
                item = SortableTableItem(val)
                item.setData(Qt.ItemDataRole.UserRole, key)
                table.setItem(ri, ci, item)
        table.setSortingEnabled(True)
        try:
            table.resizeColumnsToContents()
        except Exception:
            pass

    def _populate_track_table(self) -> None:
        meta = self._tables.get("lineage_tracks_meta")
        if meta is not None and hasattr(meta, "columns") and len(meta) > 0:
            df = meta
        elif self._tracks_df is not None and hasattr(self._tracks_df, "columns"):
            import pandas as pd

            df = (
                self._tracks_df.groupby("track_id", as_index=False)
                .agg({"frame": "count"})
                .rename(columns={"frame": "n_points"})
            )
        else:
            return
        label_map = self._biology_labels()
        preferred = [
            "biology_label",
            "track_id",
            "lineage_id",
            "parent_track_id",
            "generation",
            "n_daughters",
            "branch_depth",
            "Cell_ID",
        ]
        show_cols = ["biology_label"] + [c for c in preferred if c != "biology_label" and c in df.columns]
        self._track_table.setSortingEnabled(False)
        self._track_table.setColumnCount(len(show_cols))
        self._track_table.setHorizontalHeaderLabels([self._friendly_table_header(c) for c in show_cols])
        self._track_table.setRowCount(len(df))
        for ri, row in enumerate(df.itertuples(index=False)):
            track_id = None
            if hasattr(row, "track_id"):
                try:
                    track_id = int(getattr(row, "track_id"))
                except (TypeError, ValueError):
                    track_id = None
            for ci, col in enumerate(show_cols):
                if col == "biology_label":
                    val = self._biology_label(track_id, fallback_track_id=True) if track_id is not None else ""
                else:
                    val = getattr(row, col, "")
                item = SortableTableItem(val)
                if track_id is not None:
                    item.setData(Qt.ItemDataRole.UserRole, track_id)
                self._track_table.setItem(ri, ci, item)
        self._track_table.setSortingEnabled(True)
        try:
            self._track_table.resizeColumnsToContents()
        except Exception:
            pass

    def _division_split_text(self, row: Any) -> str:
        try:
            p = int(getattr(row, "parent_track_id"))
            d = int(getattr(row, "daughter_track_id"))
            div_f = int(getattr(row, "division_frame"))
        except (TypeError, ValueError, AttributeError):
            return ""
        events = self._tables.get("lineage_events")
        return format_biology_division_split(p, d, div_f, events, self._biology_labels())

    def _division_daughters_at_split(self, events: Any, parent_id: int, division_frame: int) -> list[int]:
        if events is None or not hasattr(events, "columns"):
            return []
        sub = events[
            (events["parent_track_id"].astype(int) == int(parent_id))
            & (events["division_frame"].astype(int) == int(division_frame))
        ]
        if len(sub) == 0 or "daughter_track_id" not in sub.columns:
            return []
        return sorted(int(x) for x in sub["daughter_track_id"].dropna().unique().tolist())

    def _division_review_hint(self, events: Any, parent_id: int, daughter_id: int) -> str:
        if events is None:
            return ""
        sub = events[
            (events["parent_track_id"].astype(int) == int(parent_id))
            & (events["daughter_track_id"].astype(int) == int(daughter_id))
        ]
        label_map = self._biology_labels()
        parent_lbl = label_map.get(int(parent_id), f"track {parent_id}")
        daughter_lbl = label_map.get(int(daughter_id), f"track {daughter_id}")
        if len(sub) == 0:
            return f"{parent_lbl} → {daughter_lbl}."
        row = sub.iloc[0]
        div_f = int(row.get("division_frame", 0))
        split_txt = format_biology_division_split(parent_id, daughter_id, div_f, events, label_map)
        gen = row.get("generation")
        gen_txt = f", generation {int(gen)}" if gen is not None and str(gen) != "nan" else ""
        asym = row.get("area_asymmetry")
        asym_txt = ""
        if asym is not None and str(asym) != "nan":
            try:
                asym_txt = f" Daughter/mother area ratio ≈ {float(asym):.2f}."
            except (TypeError, ValueError):
                pass
        status = str(row.get("review_status", "candidate"))
        link_status = str(row.get("link_status", "candidate")).lower()
        manual_note = ""
        if link_status == "manual" or int(daughter_id) < 0:
            manual_note = (
                f" Manual assignment: {parent_lbl} is the mother; the tracked branch that keeps "
                f"tracker ID {parent_id} is the continuing daughter; {daughter_lbl} is the new daughter branch."
            )
        else:
            daughters = self._division_daughters_at_split(events, parent_id, div_f)
            if len(daughters) > 1:
                manual_note = (
                    " Trackastra convention: one daughter often keeps the mother’s tracker ID "
                    f"({parent_id}) as the continuing branch; the other gets a new ID."
                )
        terminal_note = ""
        if "has_post_frame" in sub.columns and not bool(row.get("has_post_frame", True)):
            terminal_note = (
                " This split is at the final movie frame, so there is no next frame to confirm "
                "the post-division state. It should remain Unsure unless you extend the time series."
            )
        return (
            f"{split_txt}{gen_txt}. "
            f"Status: {status}.{asym_txt}{manual_note}{terminal_note} "
            "Check the movie: mother should exist just before the split; daughters appear at/after this frame. "
            "Accept if biologically correct, Reject if wrong link."
        )

    def _selected_event_row(self) -> Optional[Any]:
        if self._selected_event_key is None:
            return None
        events = self._tables.get("lineage_events")
        if events is None or not hasattr(events, "columns") or len(events) == 0:
            return None
        p, div_f = self._selected_event_key
        sub = events[
            (events["parent_track_id"].astype(int) == int(p))
            & (events["division_frame"].astype(int) == int(div_f))
        ]
        if len(sub) == 0:
            return None
        return sub.iloc[0]

    def _division_key_user_role(self, parent_id: int, division_frame: int) -> tuple[int, int]:
        return (int(parent_id), int(division_frame))

    def _track_ids_for_division_key(self, parent_id: int, division_frame: int) -> list[int]:
        events = self._tables.get("lineage_events")
        ids = [int(parent_id)]
        ids.extend(self._division_daughters_at_split(events, parent_id, division_frame))
        return list(dict.fromkeys(ids))

    def _fill_events_table(self, table: QTableWidget, events: Any, *, attach_keys: bool) -> None:
        if events is None or not hasattr(events, "columns") or len(events) == 0:
            table.setRowCount(0)
            return
        display_events = collapse_lineage_events_for_display(events) if attach_keys else events
        if display_events is None or not hasattr(display_events, "columns") or len(display_events) == 0:
            table.setRowCount(0)
            return
        if attach_keys:
            show_cols = ["split_link"] + [
                c
                for c in (
                    "division_frame_1based",
                    "generation",
                    "review_status",
                    "link_status",
                )
                if c in display_events.columns
            ]
        else:
            preferred = [
                "split_link",
                "division_frame_1based",
                "review_status",
                "has_post_frame",
                "parent_track_id",
                "daughter_track_id",
                "mother_area_before",
                "daughter_area_after",
                "area_asymmetry",
                "division_distance_px",
                "link_status",
            ]
            show_cols = [c for c in preferred if c in display_events.columns]
            if "split_link" not in show_cols:
                show_cols = ["split_link"] + show_cols
        table.setSortingEnabled(False)
        table.setColumnCount(len(show_cols))
        table.setHorizontalHeaderLabels([self._friendly_table_header(c) for c in show_cols])
        table.setRowCount(len(display_events))
        for ri, row in enumerate(display_events.itertuples(index=False)):
            event_key = None
            try:
                event_key = self._division_key_user_role(
                    int(getattr(row, "parent_track_id")),
                    int(getattr(row, "division_frame")),
                )
            except (TypeError, ValueError):
                event_key = None
            for ci, col in enumerate(show_cols):
                if col == "split_link":
                    val = self._division_split_text(row)
                else:
                    val = getattr(row, col, "")
                item = SortableTableItem(val)
                if attach_keys and event_key is not None:
                    item.setData(Qt.ItemDataRole.UserRole, event_key)
                table.setItem(ri, ci, item)
        table.setSortingEnabled(True)
        try:
            table.resizeColumnsToContents()
        except Exception:
            pass

    def _populate_events_table(self) -> None:
        events = self._tables.get("lineage_events")
        self._fill_events_table(self._events_table, events, attach_keys=True)
        if hasattr(self, "_advanced_events_table"):
            self._fill_events_table(self._advanced_events_table, events, attach_keys=False)

    def _set_overlay_layer(self, key: str, checked: bool) -> None:
        self._overlay_layers[str(key)] = bool(checked)
        self._show_frame(self._current_frame)

    def _install_canvas_interaction(self) -> None:
        self._canvas_pick_filter = _LineageCanvasPickFilter(self)
        self._canvas.installEventFilter(self._canvas_pick_filter)

    def _compact_biology_label(self, track_id: Optional[int]) -> str:
        lbl = self._biology_label(track_id, fallback_track_id=True)
        if lbl.startswith("Cell "):
            return lbl[5:]
        return lbl

    def _area_unit_label(self) -> str:
        cal = self._tables.get("lineage_calibration") or {}
        if float(cal.get("pixel_size_um", 0.0) or 0.0) > 0:
            return "µm²"
        return "px"

    def _track_meta_row(self, track_id: int) -> dict[str, Any]:
        meta = self._tables.get("lineage_tracks_meta")
        if meta is None or not hasattr(meta, "columns") or len(meta) == 0:
            return {}
        if "track_id" not in meta.columns:
            return {}
        sub = meta[meta["track_id"].astype(int) == int(track_id)]
        if len(sub) == 0:
            return {}
        return dict(sub.iloc[0])

    def _daughter_track_ids(self, track_id: int) -> list[int]:
        import json

        meta = self._track_meta_row(track_id)
        raw = meta.get("daughter_track_ids")
        if raw is not None and str(raw) not in {"", "nan", "[]"}:
            try:
                parsed = json.loads(str(raw))
                if isinstance(parsed, list):
                    return sorted(int(x) for x in parsed)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        edges = self._tables.get("lineage_edges")
        if edges is None or not hasattr(edges, "columns") or len(edges) == 0:
            return []
        sub = edges[edges["parent_track_id"].astype(int) == int(track_id)]
        if len(sub) == 0 or "daughter_track_id" not in sub.columns:
            return []
        return sorted(int(x) for x in sub["daughter_track_id"].dropna().unique().tolist())

    def _area_for_track_frame(self, track_id: int, frame_idx: int) -> Optional[float]:
        area_df = self._tables.get("track_area")
        if area_df is not None and hasattr(area_df, "columns") and len(area_df) > 0:
            sub = area_df[
                (area_df["track_id"].astype(int) == int(track_id))
                & (area_df["frame"].astype(int) == int(frame_idx))
            ]
            if len(sub) and "area" in sub.columns:
                val = sub.iloc[0]["area"]
                try:
                    if val is not None and np.isfinite(float(val)):
                        return float(val)
                except (TypeError, ValueError):
                    pass
        pts = self._tables.get("lineage_points")
        if pts is not None and hasattr(pts, "columns") and len(pts) > 0:
            sub = pts[
                (pts["track_id"].astype(int) == int(track_id))
                & (pts["frame"].astype(int) == int(frame_idx))
            ]
            if len(sub):
                for col in ("Area (µm²)", "area"):
                    if col in sub.columns:
                        val = sub.iloc[0][col]
                        try:
                            if val is not None and np.isfinite(float(val)):
                                return float(val)
                        except (TypeError, ValueError):
                            continue
        return None

    def _tracked_region_at_frame(self, frame_idx: int) -> np.ndarray:
        from bactomate.gui.tabs.tracking import TrackingTab

        h = w = 0
        masks_tr = np.asarray(self._masks_tracked) if self._masks_tracked is not None else None
        tracked_region = None
        if masks_tr is None or masks_tr.ndim != 3 or frame_idx >= masks_tr.shape[0]:
            return np.zeros((1, 1), dtype=bool)
        h, w = int(masks_tr.shape[1]), int(masks_tr.shape[2])
        tracked_region = np.zeros((h, w), dtype=bool)
        mask_frame = masks_tr[frame_idx]
        tracks = self._tracks_df
        if tracks is None or not hasattr(tracks, "columns"):
            return tracked_region
        current = tracks[tracks["frame"].astype(int) == int(frame_idx)]
        for row in current.itertuples(index=False):
            cy = int(getattr(row, "y"))
            cx = int(getattr(row, "x"))
            region = TrackingTab._mask_region_at_centroid(mask_frame, cy, cx)
            if region is None:
                region = mask_frame == int(getattr(row, "track_id"))
            if region is not None and np.any(region):
                tracked_region |= region
        return tracked_region

    def _tracked_cell_hit(self, track_id: int, frame_idx: int, *, y: float, x: float, area_px: int) -> dict[str, Any]:
        return {
            "kind": "tracked",
            "track_id": int(track_id),
            "frame": int(frame_idx),
            "y": float(y),
            "x": float(x),
            "area_px": int(area_px),
        }

    def _untracked_cell_hit(self, frame_idx: int, raw_label: int, *, y: float, x: float, area_px: int) -> dict[str, Any]:
        hit: dict[str, Any] = {
            "kind": "untracked",
            "frame": int(frame_idx),
            "label": int(raw_label),
            "y": float(y),
            "x": float(x),
            "area_px": int(area_px),
        }
        manual = self._manual_edit_for_key(int(frame_idx), int(raw_label))
        if manual is not None:
            hit["kind"] = "manual_assigned"
            hit["track_id"] = int(manual.get("manual_track_id", 0))
            hit["parent_track_id"] = manual.get("parent_track_id")
        else:
            for rec in self._untracked_records():
                if int(rec["frame"]) == int(frame_idx) and int(rec["label"]) == int(raw_label):
                    hit["nearest_track_id"] = rec.get("nearest_track_id")
                    hit["action"] = rec.get("action")
                    break
        return hit

    def _pick_cell_at(self, frame_idx: int, ix: int, iy: int) -> Optional[dict[str, Any]]:
        from bactomate.gui.tabs.tracking import TrackingTab

        masks_tr = np.asarray(self._masks_tracked) if self._masks_tracked is not None else None
        masks_in = np.asarray(self._masks_input) if self._masks_input is not None else None
        tracks = self._tracks_df
        if masks_tr is not None and masks_tr.ndim == 3 and frame_idx < masks_tr.shape[0]:
            h, w = int(masks_tr.shape[1]), int(masks_tr.shape[2])
            iy_c = int(np.clip(iy, 0, h - 1))
            ix_c = int(np.clip(ix, 0, w - 1))
            mask_frame = masks_tr[frame_idx]
            if tracks is not None and hasattr(tracks, "columns") and {"track_id", "frame", "y", "x"}.issubset(tracks.columns):
                current = tracks[tracks["frame"].astype(int) == int(frame_idx)]
                for row in current.itertuples(index=False):
                    tid = int(getattr(row, "track_id"))
                    cy = int(getattr(row, "y"))
                    cx = int(getattr(row, "x"))
                    region = TrackingTab._mask_region_at_centroid(mask_frame, cy, cx)
                    if region is None:
                        region = mask_frame == tid
                    if region is not None and region[iy_c, ix_c]:
                        return self._tracked_cell_hit(
                            tid,
                            frame_idx,
                            y=cy,
                            x=cx,
                            area_px=int(region.sum()),
                        )
        if masks_in is not None and masks_in.ndim == 3 and frame_idx < masks_in.shape[0]:
            h, w = int(masks_in.shape[1]), int(masks_in.shape[2])
            iy_c = int(np.clip(iy, 0, h - 1))
            ix_c = int(np.clip(ix, 0, w - 1))
            raw_label = int(masks_in[frame_idx, iy_c, ix_c])
            if raw_label > 0:
                region = masks_in[frame_idx] == raw_label
                tracked_region = self._tracked_region_at_frame(frame_idx)
                if tracked_region.shape == region.shape and not np.any(region & tracked_region):
                    ys, xs = np.nonzero(region)
                    if len(ys):
                        return self._untracked_cell_hit(
                            frame_idx,
                            raw_label,
                            y=float(np.mean(ys)),
                            x=float(np.mean(xs)),
                            area_px=int(region.sum()),
                        )
        return None

    def _hover_radius_for_area(self, area_px: int) -> float:
        if area_px <= 0:
            return 16.0
        return float(np.clip((area_px / np.pi) ** 0.5 * 0.75, 12.0, 34.0))

    def _format_cell_profile_lines(self, hit: dict[str, Any]) -> list[str]:
        kind = str(hit.get("kind", ""))
        frame_idx = int(hit.get("frame", self._current_frame))
        if kind in {"tracked", "manual_assigned"}:
            tid = int(hit["track_id"])
            lines = [self._biology_label(tid), f"Frame {frame_idx + 1}"]
            lid = self._lineage_id_for_track(tid)
            if lid is not None:
                lines.append(self._lineage_display_label(lid))
            meta = self._track_meta_row(tid)
            gen = meta.get("generation")
            if gen is not None and str(gen) != "nan":
                lines.append(f"Generation {int(gen)}")
            parent = meta.get("parent_track_id")
            if parent is not None and str(parent) != "nan":
                try:
                    lines.append(f"Parent: {self._biology_label(int(parent), fallback_track_id=True)}")
                except (TypeError, ValueError):
                    pass
            daughters = self._daughter_track_ids(tid)
            if daughters:
                d_lbls = ", ".join(self._biology_label(d, fallback_track_id=True) for d in daughters)
                lines.append(f"Daughters: {d_lbls}")
            area = self._area_for_track_frame(tid, frame_idx)
            if area is not None:
                lines.append(f"Area: {area:.1f} {self._area_unit_label()}")
            elif hit.get("area_px"):
                lines.append(f"Area: {int(hit['area_px'])} px")
            if kind == "manual_assigned":
                lines.append("Status: manually assigned daughter")
            lines.append(f"Tracker ID: {tid}")
            return lines
        if kind == "untracked":
            lines = [
                "Untracked segmented object",
                f"Frame {frame_idx + 1}, mask label {int(hit['label'])}",
            ]
            if hit.get("area_px"):
                lines.append(f"Area: {int(hit['area_px'])} px")
            action = hit.get("action")
            if action:
                lines.append(f"Status: {action}")
            nearest = hit.get("nearest_track_id")
            if nearest is not None:
                lines.append(
                    f"Nearest parent: {self._biology_label(int(nearest), fallback_track_id=True)}"
                )
            lines.append("Orange = in mask but not linked by Trackastra")
            return lines
        return []

    def _format_cell_profile_html(self, hit: dict[str, Any]) -> str:
        lines = self._format_cell_profile_lines(hit)
        if not lines:
            return "Hover a cell for a quick tooltip. Click to pin a cell profile here."
        title = lines[0]
        body = "<br>".join(lines[1:])
        return f"<b>{title}</b><br>{body}"

    def _format_cell_profile_tooltip(self, hit: dict[str, Any]) -> str:
        return "\n".join(self._format_cell_profile_lines(hit))

    def _current_selection_hit(self) -> Optional[dict[str, Any]]:
        if self._selected_untracked_key is not None:
            frame, label = self._selected_untracked_key
            for rec in self._untracked_records():
                if int(rec["frame"]) == int(frame) and int(rec["label"]) == int(label):
                    manual = self._manual_edit_for_key(int(frame), int(label))
                    if manual is not None:
                        return self._untracked_cell_hit(
                            int(frame),
                            int(label),
                            y=float(manual.get("y", rec.get("y", 0))),
                            x=float(manual.get("x", rec.get("x", 0))),
                            area_px=int(rec.get("area_px", 0)),
                        )
                    return self._untracked_cell_hit(
                        int(frame),
                        int(label),
                        y=float(rec.get("y", 0)),
                        x=float(rec.get("x", 0)),
                        area_px=int(rec.get("area_px", 0)),
                    )
        if self._selected_track_id is not None:
            tid = int(self._selected_track_id)
            tracks = self._tracks_df
            if tracks is not None and hasattr(tracks, "columns"):
                sub = tracks[
                    (tracks["track_id"].astype(int) == tid)
                    & (tracks["frame"].astype(int) == int(self._current_frame))
                ]
                if len(sub):
                    row = sub.iloc[0]
                    return self._tracked_cell_hit(
                        tid,
                        self._current_frame,
                        y=float(row["y"]),
                        x=float(row["x"]),
                        area_px=0,
                    )
            return self._tracked_cell_hit(tid, self._current_frame, y=0.0, x=0.0, area_px=0)
        return None

    def _update_cell_profile_panel(self) -> None:
        if not hasattr(self, "_cell_profile_label"):
            return
        hit = self._pinned_cell_hit or self._current_selection_hit()
        if hit is None:
            self._cell_profile_label.setText(
                "Hover a cell for a quick tooltip. Click to pin a cell profile here."
            )
            return
        self._cell_profile_label.setText(self._format_cell_profile_html(hit))

    def _build_cell_hover_data(self, frame_idx: int) -> list[tuple[float, float, str, float]]:
        from bactomate.gui.tabs.tracking import TrackingTab

        hover: list[tuple[float, float, str, float]] = []
        masks_tr = np.asarray(self._masks_tracked) if self._masks_tracked is not None else None
        masks_in = np.asarray(self._masks_input) if self._masks_input is not None else None
        tracks = self._tracks_df
        if (
            masks_tr is not None
            and masks_tr.ndim == 3
            and frame_idx < masks_tr.shape[0]
            and tracks is not None
            and hasattr(tracks, "columns")
            and {"track_id", "frame", "y", "x"}.issubset(tracks.columns)
        ):
            mask_frame = masks_tr[frame_idx]
            current = tracks[tracks["frame"].astype(int) == int(frame_idx)]
            for row in current.itertuples(index=False):
                tid = int(getattr(row, "track_id"))
                cy = float(getattr(row, "y"))
                cx = float(getattr(row, "x"))
                region = TrackingTab._mask_region_at_centroid(mask_frame, int(cy), int(cx))
                area_px = int(region.sum()) if region is not None and np.any(region) else 0
                hit = self._tracked_cell_hit(tid, frame_idx, y=cy, x=cx, area_px=area_px)
                hover.append(
                    (
                        cx,
                        cy,
                        self._format_cell_profile_tooltip(hit),
                        self._hover_radius_for_area(area_px),
                    )
                )
        if masks_in is not None and masks_in.ndim == 3 and frame_idx < masks_in.shape[0]:
            tracked_region = self._tracked_region_at_frame(frame_idx)
            input_frame = masks_in[frame_idx]
            for raw_label in np.unique(input_frame):
                raw_label = int(raw_label)
                if raw_label <= 0:
                    continue
                region = input_frame == raw_label
                if tracked_region.shape == region.shape and np.any(region & tracked_region):
                    continue
                ys, xs = np.nonzero(region)
                if len(ys) == 0:
                    continue
                cy = float(np.mean(ys))
                cx = float(np.mean(xs))
                hit = self._untracked_cell_hit(
                    frame_idx,
                    raw_label,
                    y=cy,
                    x=cx,
                    area_px=int(region.sum()),
                )
                hover.append(
                    (
                        cx,
                        cy,
                        self._format_cell_profile_tooltip(hit),
                        self._hover_radius_for_area(int(region.sum())),
                    )
                )
        return hover

    def _apply_cell_hit_selection(self, hit: dict[str, Any]) -> None:
        self._pinned_cell_hit = hit
        kind = str(hit.get("kind", ""))
        if kind in {"tracked", "manual_assigned"}:
            self._selected_track_id = int(hit["track_id"])
            self._selected_untracked_key = None
            self._selected_event_key = None
        elif kind == "untracked":
            self._selected_untracked_key = (int(hit["frame"]), int(hit["label"]))
            self._selected_track_id = None
            self._selected_event_key = None
        self._update_cell_profile_panel()
        self._update_analysis_summary()
        self._refresh_plots()
        self._refresh_tree()

    def _on_canvas_cell_clicked(self, ix: int, iy: int) -> None:
        hit = self._pick_cell_at(self._current_frame, ix, iy)
        if hit is None:
            return
        self._apply_cell_hit_selection(hit)
        self._show_frame(self._current_frame)

    def _selected_label_track_ids(self) -> list[int]:
        ids: list[int] = []
        if self._selected_event_key is not None:
            p, div_f = self._selected_event_key
            ids.extend(self._track_ids_for_division_key(p, div_f))
        elif self._selected_track_id is not None:
            ids.append(int(self._selected_track_id))
        elif self._pinned_cell_hit is not None and self._pinned_cell_hit.get("track_id") is not None:
            ids.append(int(self._pinned_cell_hit["track_id"]))
        return list(dict.fromkeys(ids))

    def _track_color_map(self) -> dict[int, tuple[int, int, int]]:
        import matplotlib.pyplot as plt

        if self._tracks_df is None or not hasattr(self._tracks_df, "columns") or "track_id" not in self._tracks_df.columns:
            return {}
        tids = [int(t) for t in self._tracks_df["track_id"].dropna().unique()]
        founder_by = {int(k): int(v) for k, v in dict(self._founder_by_track or {}).items()}
        if not founder_by:
            pts = self._tables.get("lineage_points")
            if pts is not None and hasattr(pts, "columns") and "track_id" in pts.columns and "lineage_id" in pts.columns:
                for row in pts[["track_id", "lineage_id"]].dropna().drop_duplicates().itertuples(index=False):
                    founder_by[int(row.track_id)] = int(row.lineage_id)
        cmap = plt.get_cmap("tab20")
        founders = sorted(set(founder_by.get(t, t) for t in tids))
        founder_colors = {}
        for i, fid in enumerate(founders):
            c = cmap((i % 20) / 20.0)
            founder_colors[int(fid)] = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
        return {t: founder_colors.get(founder_by.get(t, t), (80, 160, 220)) for t in tids}

    def _compose_overlay_frame(self, frame_idx: int) -> np.ndarray:
        from bactomate.gui.tabs.tracking import TrackingTab

        if self._stack_images is None or getattr(self._stack_images, "ndim", 0) != 3 or frame_idx >= int(self._stack_images.shape[0]):
            return np.asarray(self._frames[frame_idx], dtype=np.uint8).copy()
        img = np.asarray(self._stack_images[frame_idx], dtype=np.float32)
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            base = np.zeros(img.shape, dtype=np.uint8)
        else:
            lo, hi = float(finite.min()), float(finite.max())
            if hi <= lo:
                hi = lo + 1.0
            base = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        frame = np.stack([base, base, base], axis=-1)
        h, w = base.shape[:2]
        layers = self._overlay_layers
        masks_tr = np.asarray(self._masks_tracked) if self._masks_tracked is not None else None
        masks_in = np.asarray(self._masks_input) if self._masks_input is not None else None
        use_masks = masks_tr is not None and masks_tr.ndim == 3 and frame_idx < masks_tr.shape[0] and masks_tr.shape[1:] == (h, w)
        colors = self._track_color_map()
        tracked_region = np.zeros((h, w), dtype=bool)
        tracks = self._tracks_df
        if tracks is not None and hasattr(tracks, "columns") and {"track_id", "frame", "y", "x"}.issubset(tracks.columns):
            current = tracks[tracks["frame"].astype(int) == int(frame_idx)]
        else:
            current = None
        if current is not None and use_masks:
            mask_frame = masks_tr[frame_idx]
            for row in current.itertuples(index=False):
                tid = int(getattr(row, "track_id"))
                region = TrackingTab._mask_region_at_centroid(mask_frame, int(getattr(row, "y")), int(getattr(row, "x")))
                if region is None:
                    region = mask_frame == tid
                if region is None or not np.any(region):
                    continue
                tracked_region |= region
                if layers.get("masks", True):
                    color = colors.get(tid, (80, 160, 220))
                    frame[region, 0] = np.clip(frame[region, 0] * 0.5 + color[0] * 0.5, 0, 255).astype(np.uint8)
                    frame[region, 1] = np.clip(frame[region, 1] * 0.5 + color[1] * 0.5, 0, 255).astype(np.uint8)
                    frame[region, 2] = np.clip(frame[region, 2] * 0.5 + color[2] * 0.5, 0, 255).astype(np.uint8)
        selected_event = self._selected_event_row()
        if selected_event is not None:
            try:
                div_frame = int(selected_event.get("division_frame", -1))
                mother_y = float(selected_event.get("mother_y"))
                mother_x = float(selected_event.get("mother_x"))
                daughter_y = float(selected_event.get("daughter_y"))
                daughter_x = float(selected_event.get("daughter_x"))
            except (TypeError, ValueError):
                div_frame = -1
                mother_y = mother_x = daughter_y = daughter_x = np.nan
            if int(frame_idx) in {div_frame, max(0, div_frame - 1)} and np.all(
                np.isfinite([mother_y, mother_x, daughter_y, daughter_x])
            ):
                pts = np.array([[mother_y, mother_x], [daughter_y, daughter_x]], dtype=int)
                pts[:, 0] = np.clip(pts[:, 0], 0, h - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, w - 1)
                TrackingTab._draw_polyline_on_frame(frame, pts, color=(255, 230, 40), thickness=3)
                for yy, xx, color in (
                    (int(pts[0, 0]), int(pts[0, 1]), (255, 230, 40)),
                    (int(pts[1, 0]), int(pts[1, 1]), (255, 120, 40)),
                ):
                    s = 4
                    frame[max(0, yy - s):min(h, yy + s + 1), max(0, xx - s):min(w, xx + s + 1), :] = color
        manual_points: list[dict[str, Any]] = []
        if layers.get("untracked", True) and masks_in is not None and use_masks and masks_in.ndim == 3 and frame_idx < masks_in.shape[0] and masks_in.shape[1:] == (h, w):
            input_frame = masks_in[frame_idx]
            for raw_label in np.unique(input_frame):
                raw_label = int(raw_label)
                if raw_label <= 0:
                    continue
                region = input_frame == raw_label
                if np.any(region & tracked_region):
                    continue
                manual = self._manual_edit_for_key(frame_idx, raw_label)
                if manual is not None:
                    lid = int(manual.get("lineage_id", manual.get("parent_track_id", 0)))
                    color = self._founder_color_for_lineage(lid)
                    if layers.get("masks", True):
                        frame[region, 0] = np.clip(frame[region, 0] * 0.5 + color[0] * 0.5, 0, 255).astype(np.uint8)
                        frame[region, 1] = np.clip(frame[region, 1] * 0.5 + color[1] * 0.5, 0, 255).astype(np.uint8)
                        frame[region, 2] = np.clip(frame[region, 2] * 0.5 + color[2] * 0.5, 0, 255).astype(np.uint8)
                    manual_points.append(
                        {
                            "track_id": int(manual.get("manual_track_id", 0)),
                            "y": float(manual.get("y", 0)),
                            "x": float(manual.get("x", 0)),
                        }
                    )
                    continue
                frame[region, 0] = np.clip(frame[region, 0] * 0.25 + 255 * 0.75, 0, 255).astype(np.uint8)
                frame[region, 1] = np.clip(frame[region, 1] * 0.25 + 150 * 0.75, 0, 255).astype(np.uint8)
                frame[region, 2] = np.clip(frame[region, 2] * 0.25, 0, 255).astype(np.uint8)
        labels = []
        selected_ids = set(self._selected_label_track_ids())
        if current is not None:
            for row in current.itertuples(index=False):
                tid = int(getattr(row, "track_id"))
                y, x = int(getattr(row, "y")), int(getattr(row, "x"))
                if not (0 <= y < h and 0 <= x < w):
                    continue
                if layers.get("centers", False):
                    s = 2
                    frame[max(0, y - s):min(h, y + s + 1), max(0, x - s):min(w, x + s + 1), :] = (0, 255, 120)
                if layers.get("cell_labels", False):
                    labels.append((self._compact_biology_label(tid), x + 5, y - 5))
                elif layers.get("selected_label", True) and tid in selected_ids:
                    labels.append((self._compact_biology_label(tid), x + 5, y - 5))
                elif layers.get("ids", False):
                    labels.append((str(tid), x + 5, y - 5))
        if manual_points:
            for mp in manual_points:
                tid = int(mp["track_id"])
                if layers.get("cell_labels", False):
                    text = self._compact_biology_label(tid)
                elif layers.get("selected_label", True) and tid in selected_ids:
                    text = self._compact_biology_label(tid)
                elif layers.get("ids", False):
                    text = f"M{abs(tid)}"
                else:
                    continue
                labels.append((text, int(mp["x"]) + 5, int(mp["y"]) - 5))
        if labels:
            try:
                from PIL import Image
                from bactomate.modules.label_render import draw_hq_labels

                frame = np.asarray(draw_hq_labels(Image.fromarray(frame), labels, font_size=12, scale=2, stroke_width=2))
            except Exception:
                pass
        return frame

    def _on_track_row_clicked(self, row: int, _col: int) -> None:
        item = self._track_table.item(row, 0)
        if item is not None:
            tid = item.data(Qt.ItemDataRole.UserRole)
            if tid is None:
                tid = item.text()
            try:
                self._selected_track_id = int(tid)
                self._selected_untracked_key = None
                self._selected_event_key = None
                self._pinned_cell_hit = None
            except (TypeError, ValueError):
                pass
        self._update_cell_profile_panel()
        self._update_analysis_summary()
        self._refresh_plots()
        self._refresh_tree()
        self._show_frame(self._current_frame)

    def _on_event_current_cell_changed(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        if row >= 0:
            self._on_event_row_clicked(row, max(0, _col))

    def _on_event_row_clicked(self, row: int, _col: int) -> None:
        events = self._tables.get("lineage_events")
        if events is None:
            return
        item = self._events_table.item(row, 0)
        event_key = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if event_key is None:
            return
        p, div_f = int(event_key[0]), int(event_key[1])
        self._selected_event_key = (p, div_f)
        daughters = self._division_daughters_at_split(events, p, div_f)
        self._selected_track_id = daughters[0] if daughters else None
        self._selected_untracked_key = None
        self._pinned_cell_hit = None
        if hasattr(self, "_review_status_label"):
            rep_d = daughters[0] if daughters else 0
            self._review_status_label.setText(self._division_review_hint(events, p, rep_d))
        self._frame_slider.blockSignals(True)
        try:
            self._frame_slider.setValue(max(0, min(div_f, self._frame_slider.maximum())))
        finally:
            self._frame_slider.blockSignals(False)
        self._show_frame(div_f)
        self._update_cell_profile_panel()
        self._update_analysis_summary()
        self._refresh_plots()
        self._refresh_tree()

    def _accept_all_visible_divisions(self) -> None:
        events = self._tables.get("lineage_events")
        if events is None or not hasattr(events, "columns") or len(events) == 0:
            if hasattr(self, "_review_status_label"):
                self._review_status_label.setText("No division links to accept.")
            return
        updated = events.copy()
        terminal_mask = None
        if "has_post_frame" in updated.columns:
            terminal_mask = ~updated["has_post_frame"].astype(bool)
        if "review_status" not in updated.columns:
            updated["review_status"] = REVIEW_STATUS_ACCEPTED
        else:
            updated["review_status"] = REVIEW_STATUS_ACCEPTED
        skipped = 0
        if terminal_mask is not None and terminal_mask.any():
            skipped = int(terminal_mask.sum())
            updated.loc[terminal_mask, "review_status"] = REVIEW_STATUS_FLAGGED
        self._tables["lineage_events"] = updated
        self._populate_events_table()
        self._refresh_curated_views()
        self._sync_external_lineage_tables()
        self._refresh_plots()
        self._refresh_tree()
        if hasattr(self, "_review_status_label"):
            collapsed = collapse_lineage_events_for_display(updated)
            if collapsed is not None and hasattr(collapsed, "columns") and "review_status" in collapsed.columns:
                n_acc = int((collapsed["review_status"].astype(str) == REVIEW_STATUS_ACCEPTED).sum())
                n_flagged = int((collapsed["review_status"].astype(str) == REVIEW_STATUS_FLAGGED).sum())
            else:
                n_acc = len(updated) - skipped
                n_flagged = skipped
            msg = f"Marked {n_acc} division event(s) as accepted."
            if n_flagged:
                msg += (
                    f" Kept {n_flagged} final-frame split(s) as Unsure (no frame after the split to verify)."
                )
            if self._curated_strict:
                msg += " Strict mode is on — Analysis now uses only accepted links."
            else:
                msg += (
                    " Candidate links were already included in Analysis; "
                    "Reject is what removes bad links. Turn on “Accepted divisions only” on the Analysis tab "
                    "if you want acceptance to be required."
                )
            self._review_status_label.setText(msg)

    def _set_selected_event_review(self, status: str) -> None:
        if self._selected_event_key is None:
            if hasattr(self, "_review_status_label"):
                self._review_status_label.setText("Select a division row first, then choose Accept, Reject, or Unsure.")
            return
        p, div_f = self._selected_event_key
        events = self._tables.get("lineage_events")
        if events is None:
            return
        terminal_accept_blocked = False
        sub = events[
            (events["parent_track_id"].astype(int) == int(p))
            & (events["division_frame"].astype(int) == int(div_f))
        ]
        if (
            status == REVIEW_STATUS_ACCEPTED
            and len(sub)
            and "has_post_frame" in sub.columns
            and not bool(sub["has_post_frame"].astype(bool).all())
        ):
            status = REVIEW_STATUS_FLAGGED
            terminal_accept_blocked = True
        updated = apply_division_review_status(
            events, parent_track_id=p, division_frame=div_f, review_status=status
        )
        self._tables["lineage_events"] = updated
        self._populate_events_table()
        self._sync_external_lineage_tables()
        if hasattr(self, "_review_status_label"):
            label_map = self._biology_labels()
            rep_d = int(sub.iloc[0]["daughter_track_id"]) if len(sub) else 0
            split_txt = format_biology_division_split(p, rep_d, div_f, events, label_map)
            if terminal_accept_blocked:
                self._review_status_label.setText(
                    f"Kept {split_txt} as Unsure: it occurs in the final frame, so there is no next frame "
                    "to confirm a real division."
                )
            else:
                if status == REVIEW_STATUS_ACCEPTED:
                    extra = (
                        "Counts in Analysis (strict mode requires acceptance)."
                        if self._curated_strict
                        else "Recorded — candidate links were already in Analysis unless you enable strict mode."
                    )
                elif status == REVIEW_STATUS_REJECTED:
                    extra = "Excluded from Analysis and export curation."
                else:
                    extra = "Still shown but flagged for follow-up."
                self._review_status_label.setText(f"Marked {split_txt} as {status}. {extra}")
        self._refresh_curated_views()
        self._refresh_plots()
        self._refresh_tree()
        self._save_manual_curation()

    def _on_untracked_row_clicked(self, row: int, _col: int) -> None:
        item = self._untracked_table.item(row, 0)
        key = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if key is None:
            return
        self._selected_untracked_key = (int(key[0]), int(key[1]))
        self._selected_track_id = None
        self._selected_event_key = None
        self._pinned_cell_hit = None
        if hasattr(self, "_untracked_status_label"):
            self._untracked_status_label.setText(
                f"Orange cell selected — frame {int(key[0]) + 1} (table index {int(key[0])}), "
                f"segmentation label {int(key[1])}. White outline marks the untracked object."
            )
        self._update_cell_profile_panel()
        frame = int(key[0])
        self._populate_untracked_parent_combo(self._selected_untracked_record())
        self._frame_slider.blockSignals(True)
        try:
            self._frame_slider.setValue(max(0, min(frame, self._frame_slider.maximum())))
        finally:
            self._frame_slider.blockSignals(False)
        self._show_frame(frame)

    def _lineage_id_for_track(self, track_id: int) -> Optional[int]:
        meta = self._tables.get("lineage_tracks_meta")
        if meta is not None and hasattr(meta, "columns") and "track_id" in meta.columns:
            sub = meta[meta["track_id"].astype(int) == int(track_id)]
            if len(sub) and "lineage_id" in sub.columns:
                try:
                    return int(sub.iloc[0]["lineage_id"])
                except Exception:
                    pass
        pts = self._tables.get("lineage_points")
        if pts is not None and hasattr(pts, "columns") and "track_id" in pts.columns and "lineage_id" in pts.columns:
            sub = pts[pts["track_id"].astype(int) == int(track_id)].head(1)
            if len(sub):
                try:
                    return int(sub.iloc[0]["lineage_id"])
                except Exception:
                    pass
        return int(track_id)

    def _parent_generation(self, track_id: int) -> int:
        meta = self._tables.get("lineage_tracks_meta")
        if meta is not None and hasattr(meta, "columns") and "track_id" in meta.columns and "generation" in meta.columns:
            sub = meta[meta["track_id"].astype(int) == int(track_id)]
            if len(sub):
                try:
                    return int(sub.iloc[0]["generation"])
                except Exception:
                    pass
        return 0

    def _selected_untracked_record(self) -> Optional[dict[str, Any]]:
        if self._selected_untracked_key is None:
            return None
        frame, label = self._selected_untracked_key
        for rec in self._untracked_records():
            if int(rec["frame"]) == int(frame) and int(rec["label"]) == int(label):
                return rec
        return None

    def _populate_untracked_parent_combo(self, rec: Optional[dict[str, Any]]) -> None:
        if not hasattr(self, "_untracked_parent_combo"):
            return
        combo = self._untracked_parent_combo
        combo.blockSignals(True)
        combo.clear()
        if rec is None or self._tracks_df is None or not hasattr(self._tracks_df, "columns"):
            combo.addItem("Select orange cell first", None)
            combo.setEnabled(False)
            combo.blockSignals(False)
            return
        tracks = self._tracks_df
        needed = {"track_id", "frame", "y", "x"}
        if not needed.issubset(tracks.columns):
            combo.addItem("No parent tracks available", None)
            combo.setEnabled(False)
            combo.blockSignals(False)
            return
        frame = int(rec["frame"])
        y = float(rec["y"])
        x = float(rec["x"])
        candidates = tracks[tracks["frame"].astype(int).between(max(0, frame - 1), frame)]
        rows: list[tuple[float, int, int]] = []
        for row in candidates.itertuples(index=False):
            tid = int(getattr(row, "track_id"))
            dy = float(getattr(row, "y")) - y
            dx = float(getattr(row, "x")) - x
            dist = float((dy * dy + dx * dx) ** 0.5)
            row_frame = int(getattr(row, "frame"))
            rows.append((dist, tid, row_frame))
        seen: set[int] = set()
        nearest_tid = rec.get("nearest_track_id")
        selected_idx = 0
        for dist, tid, row_frame in sorted(rows, key=lambda item: (item[0], item[1])):
            if tid in seen:
                continue
            seen.add(tid)
            cell_lbl = self._biology_label(tid, fallback_track_id=True)
            label = f"{cell_lbl}  ({dist:.1f} px away, frame {row_frame + 1})"
            if nearest_tid is not None and int(tid) == int(nearest_tid):
                label += "  nearest"
            combo.addItem(label, tid)
            if nearest_tid is not None and int(tid) == int(nearest_tid):
                selected_idx = combo.count() - 1
        if combo.count() == 0:
            combo.addItem("No nearby parent tracks", None)
            combo.setEnabled(False)
        else:
            combo.setCurrentIndex(selected_idx)
            combo.setEnabled(True)
        combo.blockSignals(False)

    def _selected_untracked_parent_track_id(self) -> Optional[int]:
        if hasattr(self, "_untracked_parent_combo") and self._untracked_parent_combo.isEnabled():
            val = self._untracked_parent_combo.currentData()
            try:
                if val is not None:
                    return int(val)
            except (TypeError, ValueError):
                pass
        return self._selected_track_id

    def _assign_selected_untracked(self) -> None:
        rec = self._selected_untracked_record()
        if rec is None:
            if hasattr(self, "_untracked_status_label"):
                self._untracked_status_label.setText("Select an untracked object first.")
            return
        if rec.get("action") == "assigned":
            manual = self._manual_edit_for_key(int(rec["frame"]), int(rec["label"]))
            if hasattr(self, "_untracked_status_label"):
                if manual is not None:
                    parent_lbl = self._biology_label(manual.get("parent_track_id"), fallback_track_id=True)
                    child_lbl = self._biology_label(manual.get("manual_track_id"), fallback_track_id=True)
                    self._untracked_status_label.setText(
                        f"Already assigned as {child_lbl}, daughter of {parent_lbl}. "
                        "It should appear in lineage color, not orange. Open Analysis to verify the tree."
                    )
                else:
                    self._untracked_status_label.setText("This untracked object is already assigned.")
            return
        parent_tid = self._selected_untracked_parent_track_id()
        if parent_tid is None and rec.get("nearest_track_id") is not None:
            parent_tid = int(rec["nearest_track_id"])
        if parent_tid is None:
            if hasattr(self, "_untracked_status_label"):
                self._untracked_status_label.setText("Select a parent track first, or use an object with a nearest track.")
            return
        lineage_id = self._lineage_id_for_track(int(parent_tid))
        generation = self._parent_generation(int(parent_tid)) + 1
        self._manual_edit_counter += 1
        manual_tid = -int(self._manual_edit_counter)
        frame = int(rec["frame"])
        label = int(rec["label"])
        area = float(rec["area_px"])
        y = float(rec["y"])
        x = float(rec["x"])

        import pandas as pd

        parent_y: Optional[float] = None
        parent_x: Optional[float] = None
        if self._tracks_df is not None and hasattr(self._tracks_df, "columns"):
            parent_pts = self._tracks_df[
                (self._tracks_df["track_id"].astype(int) == int(parent_tid))
                & (self._tracks_df["frame"].astype(int) == frame)
            ]
            if len(parent_pts) > 0:
                parent_y = float(parent_pts.iloc[0]["y"])
                parent_x = float(parent_pts.iloc[0]["x"])
        edit_row = {
            "frame": frame,
            "untracked_label": label,
            "manual_track_id": manual_tid,
            "parent_track_id": int(parent_tid),
            "parent_y": parent_y,
            "parent_x": parent_x,
            "lineage_id": int(lineage_id),
            "area_px": area,
            "y": y,
            "x": x,
            "review_status": "manual",
        }
        edits = self._tables.get("manual_lineage_edits")
        if edits is None or not hasattr(edits, "columns"):
            edits = pd.DataFrame(columns=list(edit_row.keys()))
        edits = edits[
            ~(
                (edits.get("frame", pd.Series(dtype=int)).astype(int) == frame)
                & (edits.get("untracked_label", pd.Series(dtype=int)).astype(int) == label)
            )
        ] if len(edits) else edits
        self._tables["manual_lineage_edits"] = self._append_table_row(edits, edit_row)

        meta = self._tables.get("lineage_tracks_meta")
        meta_row = {
            "track_id": manual_tid,
            "lineage_id": int(lineage_id),
            "founder_track_id": int(lineage_id),
            "parent_track_id": int(parent_tid),
            "daughter_track_ids": "[]",
            "n_daughters": 0,
            "generation": int(generation),
            "branch_depth": 0,
            "Cell_ID": f"manual:{label}",
        }
        if meta is not None and hasattr(meta, "columns"):
            self._tables["lineage_tracks_meta"] = self._append_table_row(meta, meta_row)

        edges = self._tables.get("lineage_edges")
        edge_row = {"parent_track_id": int(parent_tid), "daughter_track_id": manual_tid, "division_frame": frame, "parent_last_frame": max(0, frame - 1), "link_status": "manual"}
        if edges is not None and hasattr(edges, "columns"):
            self._tables["lineage_edges"] = self._append_table_row(edges, edge_row)

        pts = self._tables.get("lineage_points")
        point_row = {"track_id": manual_tid, "frame": frame, "y": y, "x": x, "Cell_ID": f"manual:{label}", "lineage_id": int(lineage_id), "area": area, "area_source": "manual_mask_px"}
        if pts is not None and hasattr(pts, "columns"):
            self._tables["lineage_points"] = self._append_table_row(pts, point_row)

        track_row = {"track_id": manual_tid, "frame": frame, "y": y, "x": x, "lineage_id": int(lineage_id)}
        if self._tracks_df is not None and hasattr(self._tracks_df, "columns"):
            self._tracks_df = self._append_table_row(self._tracks_df, track_row)
        else:
            self._tracks_df = pd.DataFrame([track_row])

        self._selected_track_id = manual_tid
        self._tables.pop("lineage_tree_nodes", None)
        self._tables.pop("lineage_tree_edges", None)
        self._refresh_curated_views()
        self._populate_track_table()
        self._populate_events_table()
        self._populate_untracked_table()
        self._populate_untracked_parent_combo(None)
        self._save_manual_curation()
        self._refresh_tree()
        self._refresh_plots()
        self._show_frame(frame)
        if hasattr(self, "_untracked_status_label"):
            parent_lbl = self._biology_label(int(parent_tid), fallback_track_id=True)
            child_lbl = self._biology_label(manual_tid, fallback_track_id=True)
            self._untracked_status_label.setText(
                f"Assigned orange mask at frame {frame + 1} as {child_lbl}, new daughter of {parent_lbl}. "
                f"The tracked branch that keeps tracker ID {parent_tid} is the other daughter (continuing branch). "
                "The cell should turn lineage-colored. Open Analysis to verify tree/growth."
            )

    def _ignore_selected_untracked(self) -> None:
        if self._selected_untracked_key is None:
            if hasattr(self, "_untracked_status_label"):
                self._untracked_status_label.setText("Select an untracked object first.")
            return
        ignored = set(self._tables.get("manual_ignored_untracked", set()) or set())
        ignored.add((int(self._selected_untracked_key[0]), int(self._selected_untracked_key[1])))
        self._tables["manual_ignored_untracked"] = ignored
        self._populate_untracked_table()
        self._populate_untracked_parent_combo(None)
        self._refresh_curated_views()
        self._save_manual_curation()
        if hasattr(self, "_untracked_status_label"):
            self._untracked_status_label.setText("Ignored selected untracked object.")

    def _on_frame_changed(self, idx: int) -> None:
        self._show_frame(int(idx))

    def _highlight_selected_untracked(self, frame_rgb: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw a white outline on the selected orange untracked segmentation."""
        key = self._selected_untracked_key
        if key is None or int(key[0]) != int(frame_idx) or self._masks_input is None:
            return frame_rgb
        masks_in = np.asarray(self._masks_input)
        if masks_in.ndim != 3 or frame_idx >= masks_in.shape[0]:
            return frame_rgb
        from scipy.ndimage import binary_dilation

        region = masks_in[frame_idx] == int(key[1])
        if not np.any(region):
            return frame_rgb
        outline = binary_dilation(region, iterations=2) & ~region
        out = np.asarray(frame_rgb, dtype=np.uint8).copy()
        out[outline] = (255, 255, 255)
        return out

    def _highlight_frame(self, frame_rgb: np.ndarray, frame_idx: int) -> np.ndarray:
        from bactomate.gui.tabs.tracking import TrackingTab

        out = np.asarray(frame_rgb, dtype=np.uint8).copy()
        untracked_on_frame = (
            self._selected_untracked_key is not None
            and int(self._selected_untracked_key[0]) == int(frame_idx)
        )
        highlight_ids: list[int] = []
        if self._selected_event_key is not None:
            p, div_f = self._selected_event_key
            highlight_ids.extend(self._track_ids_for_division_key(p, div_f))
        elif self._selected_track_id is not None:
            highlight_ids.append(int(self._selected_track_id))
        if not untracked_on_frame and highlight_ids and self._tracks_df is not None:
            for tid in dict.fromkeys(highlight_ids):
                out = TrackingTab._apply_selected_track_halo(
                    out,
                    frame_idx,
                    int(tid),
                    self._masks_tracked,
                    self._tracks_df,
                )
        if untracked_on_frame:
            out = self._highlight_selected_untracked(out, frame_idx)
        return out

    def _show_frame(self, idx: int) -> None:
        idx = max(0, min(idx, int(self._frames.shape[0]) - 1))
        self._current_frame = idx
        frame = self._compose_overlay_frame(idx)
        frame = self._highlight_frame(frame, idx)
        h, w = frame.shape[:2]
        pix = to_qpixmap(frame)
        self._canvas.set_pixmap(pix, (w, h), reset_view=False)
        self._canvas.set_foci_hover_data(self._build_cell_hover_data(idx))
        self._update_cell_profile_panel()
        self._frame_label.setText(f"Frame {idx + 1} / {self._frames.shape[0]}")

    def _selected_lineage_id(self) -> Optional[int]:
        import pandas as pd

        pts = self._tables.get("lineage_points")
        if self._selected_track_id is None or pts is None or not isinstance(pts, pd.DataFrame):
            return None
        if "lineage_id" not in pts.columns:
            return self._selected_track_id
        row0 = pts[pts["track_id"] == int(self._selected_track_id)].head(1)
        if len(row0) == 0:
            return None
        lid = row0.iloc[0].get("lineage_id")
        return None if pd.isna(lid) else int(lid)

    def _refresh_tree(self) -> None:
        if self._tree_ax is None:
            return
        nodes = self._tables.get("lineage_tree_nodes_curated")
        edges = self._tables.get("lineage_tree_edges_curated")
        if nodes is None:
            from bactomate.modules.lineage_viz import build_lineage_tree_layout

            edges_src = self._coalesce_table(
                self._tables, "lineage_edges_curated", "lineage_edges", fallback=self._edges_df
            )
            meta = self._coalesce_table(self._tables, "lineage_tracks_meta_curated", "lineage_tracks_meta")
            lid = self._selected_lineage_id()
            lineage_ids = [int(lid)] if lid is not None else None
            nodes, edges = build_lineage_tree_layout(edges_src, meta, lineage_ids=lineage_ids)
        else:
            lid = self._selected_lineage_id()
            if lid is not None and hasattr(nodes, "columns") and "lineage_id" in nodes.columns:
                filtered = nodes[nodes["lineage_id"].astype(int) == int(lid)]
                if len(filtered) == 0:
                    from bactomate.modules.lineage_viz import build_lineage_tree_layout

                    edges_src = self._coalesce_table(
                        self._tables, "lineage_edges_curated", "lineage_edges", fallback=self._edges_df
                    )
                    meta = self._coalesce_table(self._tables, "lineage_tracks_meta_curated", "lineage_tracks_meta")
                    nodes, edges = build_lineage_tree_layout(edges_src, meta, lineage_ids=[int(lid)])
                else:
                    nodes = filtered
                    if edges is not None and hasattr(edges, "columns") and len(nodes) > 0:
                        tids = set(int(t) for t in nodes["track_id"].tolist())
                        edges = edges[
                            edges["parent_track_id"].isin(tids) & edges["daughter_track_id"].isin(tids)
                        ]
        lid = self._selected_lineage_id()
        label = self._lineage_display_label(lid)
        title = f"{label} tree" if self._curated_strict else f"{label} tree (rejected hidden)"
        plot_lineage_tree(
            self._tree_ax,
            nodes,
            edges,
            highlight_track_id=self._selected_track_id,
            track_labels=self._biology_labels(),
            title=title,
        )
        self._tree_canvas.draw()

    def _refresh_plots(self) -> None:
        if self._ax is None:
            return
        import pandas as pd

        mode = self._plot_mode_combo.currentText() if hasattr(self, "_plot_mode_combo") else "Per-track metric"
        events = self._tables.get("lineage_events")
        lid = self._selected_lineage_id()
        include_candidates = not self._curated_strict
        if mode.startswith("Cell-cycle area"):
            area_df = area_table_for_lineage_growth(self._tables)
            cal = self._tables.get("lineage_calibration") or {}
            area_df = calibrate_area_table(area_df, float(cal.get("pixel_size_um", 0.0) or 0.0))
            founder = self._tables.get("founder_by_track_curated") or self._tables.get("founder_by_track") or {}
            if area_df is not None and hasattr(area_df, "columns") and "track_id" in area_df.columns:
                area_df = area_df.copy()
                area_df["lineage_id"] = area_df["track_id"].map(
                    lambda t: founder.get(int(t), int(t)) if t is not None else None
                )
            edges = self._coalesce_table(self._tables, "lineage_edges_curated", "lineage_edges", fallback=self._edges_df)
            plot_lineage_cell_cycle_area(
                self._ax,
                area_df,
                edges,
                lineage_id=lid,
                label_map=self._biology_labels(),
                events_df=events,
                include_candidate_divisions=include_candidates,
                title=f"{self._lineage_display_label(lid)} cell-cycle area",
            )
            if hasattr(self, "_plot_notice"):
                self._plot_notice.setVisible(False)
            self._plot_canvas.draw()
            return
        if mode.startswith("Lineage total") or mode.startswith("Log lineage"):
            growth_ts = self._coalesce_table(
                self._tables, "lineage_growth_timeseries_curated", "lineage_growth_timeseries"
            )
            plot_lineage_growth(
                self._ax,
                growth_ts,
                lineage_id=lid,
                log_scale=mode.startswith("Log"),
                title=f"{self._lineage_display_label(lid)} growth",
                events_df=events,
                show_division_markers=True,
                include_candidate_divisions=include_candidates,
            )
            if hasattr(self, "_plot_notice"):
                self._plot_notice.setVisible(False)
            self._plot_canvas.draw()
            return
        if mode.startswith("Lineage FI") or mode.startswith("Lineage foci") or mode.startswith("Lineage coloc"):
            metric_col = mode.split(" — ", 1)[-1] if " — " in mode else self._metric_combo.currentText()
            pts = self._tables.get("lineage_points")
            aggregate = "mean" if "intensity" in metric_col.lower() or "fi" in metric_col.lower() else "sum"
            plot_lineage_metric_timeseries(
                self._ax,
                pts,
                metric_column=metric_col,
                lineage_id=lid,
                aggregate=aggregate,
                log_scale="intensity" in metric_col.lower(),
                title=mode.split(" — ")[0] if " — " in mode else "Lineage metric",
                events_df=events,
                show_division_markers=True,
                include_candidate_divisions=include_candidates,
            )
            if hasattr(self, "_plot_notice"):
                self._plot_notice.setVisible(False)
            self._plot_canvas.draw()
            return

        self._ax.clear()
        pts = self._tables.get("lineage_points")
        if pts is None or not isinstance(pts, pd.DataFrame) or len(pts) == 0:
            self._ax.set_title("No lineage points")
            self._plot_canvas.draw()
            return
        col_choice = self._metric_combo.currentText()
        sub = pts.copy()
        if self._selected_track_id is not None and "track_id" in sub.columns:
            tid = int(self._selected_track_id)
            founder = None
            if "lineage_id" in sub.columns:
                row0 = sub[sub["track_id"] == tid].head(1)
                if len(row0) > 0:
                    founder = row0.iloc[0].get("lineage_id")
            if founder is not None and not pd.isna(founder):
                sub = sub[sub["lineage_id"] == founder]
            else:
                sub = sub[sub["track_id"] == tid]
        ycol = None
        if col_choice.startswith("Area ("):
            for c in ("Area (µm²)", "area", "Area"):
                if c in sub.columns:
                    ycol = c
                    break
        elif col_choice in sub.columns:
            ycol = col_choice
        elif col_choice.startswith("y") and "y" in sub.columns:
            ycol = "y"
        elif "y" in sub.columns:
            ycol = "y"
        if ycol is None or "frame" not in sub.columns:
            self._ax.set_title("No plottable column")
            if hasattr(self, "_plot_notice"):
                self._plot_notice.setVisible(False)
            self._plot_canvas.draw()
            return
        position_only = ycol == "y"
        for tid, g in sub.groupby("track_id"):
            g = g.sort_values("frame")
            self._ax.plot(g["frame"].astype(float) + 1.0, g[ycol], marker="o", markersize=3, label=str(tid))
        self._ax.set_xlabel("Frame")
        if position_only:
            ylabel = "Y position (px)"
        elif str(col_choice).startswith("Area ("):
            ylabel = str(col_choice)
        else:
            ylabel = str(ycol)
        self._ax.set_ylabel(ylabel)
        self._ax.set_title(
            f"Track position — lineage {self._selected_track_id or 'all'}"
            if position_only
            else f"Lineage {self._selected_track_id or 'all'}"
        )
        if hasattr(self, "_plot_notice"):
            if position_only:
                self._plot_notice.setText(
                    "Position preview only — not growth data. Join an Analyze workbook for area/FI plots."
                )
                self._plot_notice.setVisible(True)
            else:
                self._plot_notice.setVisible(False)
        if sub["track_id"].nunique() <= 8:
            self._ax.legend(fontsize=7, loc="best")
        self._plot_canvas.draw()

    def _export_current_analysis_plot(self) -> None:
        if not hasattr(self, "_fig") or self._fig is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export current Analysis plot",
            "lineage_analysis_plot.png",
            "PNG image (*.png);;PDF file (*.pdf)",
        )
        if not path:
            return
        try:
            self._fig.savefig(path, dpi=180)
            if hasattr(self, "_plot_notice"):
                self._plot_notice.setText(f"Saved plot to {path}")
                self._plot_notice.setVisible(True)
        except Exception as exc:
            if hasattr(self, "_plot_notice"):
                self._plot_notice.setText(f"Could not save plot: {exc}")
                self._plot_notice.setVisible(True)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._sync_metric_combo_visibility()
        self._auto_select_default_lineage()
        self._refresh_curated_views()
        self._refresh_tree()
        self._refresh_plots()


def open_lineage_inspector_from_preview(parent: QWidget, result: Any) -> None:
    """Open inspector from TrackingPreviewResult."""
    if result is None or not getattr(result, "is_lineage_preview", False):
        return
    frames = getattr(result, "frames_rgb", None)
    if frames is None:
        return
    from bactomate.modules.lineage_integration import resolve_lineage_curation_path

    mask_path = getattr(result, "mask_path", None)
    curation_path = resolve_lineage_curation_path(mask_path) if mask_path is not None else None
    dlg = LineageInspectorDialog(
        parent,
        frames_rgb=frames,
        tracks_df=getattr(result, "tracks_df", None),
        edges_df=getattr(result, "edges_df", None),
        lineage_tables=getattr(result, "lineage_tables", None),
        stack_images=getattr(result, "stack_images", None),
        masks_tracked=getattr(result, "masks_tracked", None),
        masks_input=getattr(result, "masks_input", None),
        founder_by_track=getattr(result, "founder_by_track", None),
        curation_path=curation_path,
        mask_path=mask_path,
        original_path=getattr(result, "original_path", None),
        title="Lineage review",
    )
    dlg.exec()
    _sync_preview_result_from_inspector(result, dlg)


def open_lineage_inspector_from_output_dir(parent: QWidget, output_dir: Path) -> None:
    """Load lineage tables from a completed tracking output folder."""
    from bactomate.modules.lineage_integration import load_lineage_tables_from_output_dir

    output_dir = Path(output_dir)
    tables = load_lineage_tables_from_output_dir(output_dir)
    tracks_df = pts
    vis = sorted(output_dir.glob("*_lineage.tif"))
    if vis:
        from bactomate.gui.utils import safe_imread

        frames = np.asarray(safe_imread(str(vis[0])))
        if frames.ndim == 3 and frames.shape[-1] != 3:
            frames = np.stack([frames, frames, frames], axis=-1)
    else:
        frames = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    masks_path = sorted(output_dir.glob("*_masks_tracked.tif"))
    masks_tracked = None
    if masks_path:
        from bactomate.gui.utils import safe_imread

        masks_tracked = np.asarray(safe_imread(str(masks_path[0])))
    dlg = LineageInspectorDialog(
        parent,
        frames_rgb=frames,
        tracks_df=tracks_df,
        edges_df=tables.get("lineage_edges"),
        lineage_tables=tables,
        masks_tracked=masks_tracked,
        curation_path=output_dir / "tables" / "lineage_curation.json",
        title="Lineage review",
    )
    dlg.exec()


def _sync_preview_result_from_inspector(result: Any, dlg: LineageInspectorDialog) -> None:
    """Push curated tables back into an in-memory preview result."""
    if result is None:
        return
    tables = getattr(result, "lineage_tables", None)
    if isinstance(tables, dict):
        curated_tracks = tables.get("_tracks_df_curated")
        curated_edges = tables.get("_edges_df_curated")
        if curated_tracks is not None:
            result.tracks_df = curated_tracks
        if curated_edges is not None:
            result.edges_df = curated_edges
        tables.pop("_tracks_df_curated", None)
        tables.pop("_edges_df_curated", None)
