"""Interactive lineage inspector: movie, tree graph, events, and measurement plots."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
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
    REVIEW_STATUS_FLAGGED,
    REVIEW_STATUS_REJECTED,
    apply_event_review_status,
)
from bactomate.modules.lineage_viz import plot_lineage_growth, plot_lineage_tree


class LineageInspectorDialog(QDialog):
    """Inspect lineage preview or saved tracking outputs."""

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        frames_rgb: np.ndarray,
        tracks_df: Any,
        edges_df: Any = None,
        lineage_tables: Optional[dict[str, Any]] = None,
        masks_tracked: Any = None,
        title: str = "Lineage review",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(1100, 720)
        self._frames = np.asarray(frames_rgb, dtype=np.uint8)
        self._tracks_df = tracks_df
        self._edges_df = edges_df
        self._masks_tracked = masks_tracked
        self._tables = dict(lineage_tables or {})
        self._selected_track_id: Optional[int] = None
        self._selected_event_key: Optional[tuple[int, int]] = None
        self._current_frame = 0
        self._build_ui()
        self._install_close_shortcuts()
        self._populate_track_table()
        self._populate_events_table()
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
        left_lay = QVBoxLayout(left)
        self._canvas = ImageCanvas()
        left_lay.addWidget(self._canvas, 1)
        split.addWidget(left)

        right = QWidget()
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

        events_tab = QWidget()
        events_lay = QVBoxLayout(events_tab)
        events_lay.addWidget(
            QLabel(
                "Start here: review division candidates. Select a row to jump to that frame, "
                "then Accept, Reject, or Flag each event before trusting the lineage tree."
            )
        )
        self._events_table = QTableWidget()
        self._events_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._events_table.cellClicked.connect(self._on_event_row_clicked)
        events_lay.addWidget(self._events_table, 1)
        review_row = QHBoxLayout()
        for label, status in (
            ("Accept", REVIEW_STATUS_ACCEPTED),
            ("Reject", REVIEW_STATUS_REJECTED),
            ("Flag", REVIEW_STATUS_FLAGGED),
        ):
            btn = QPushButton(label)
            style_compact_action_button(btn, horizontal_padding=24)
            btn.clicked.connect(lambda _checked=False, s=status: self._set_selected_event_review(s))
            review_row.addWidget(btn)
        events_lay.addLayout(review_row)
        tabs.addTab(events_tab, "Divisions")

        tracks_tab = QWidget()
        tracks_lay = QVBoxLayout(tracks_tab)
        tracks_lay.addWidget(QLabel("Tracks (double-click to select a lineage branch)"))
        self._track_table = QTableWidget()
        self._track_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._track_table.cellDoubleClicked.connect(self._on_track_row_clicked)
        tracks_lay.addWidget(self._track_table, 1)
        tabs.addTab(tracks_tab, "Tracks")

        tree_tab = QWidget()
        tree_lay = QVBoxLayout(tree_tab)
        tree_lay.addWidget(QLabel("Lineage tree (founder-colored)"))
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._tree_fig = Figure(figsize=(3.6, 2.8), tight_layout=True)
            self._tree_canvas = FigureCanvasQTAgg(self._tree_fig)
            self._tree_ax = self._tree_fig.add_subplot(111)
            tree_lay.addWidget(self._tree_canvas, 1)
        except Exception:
            self._tree_canvas = QLabel("(Tree plot unavailable — install matplotlib)")
            tree_lay.addWidget(self._tree_canvas)
            self._tree_ax = None
        tabs.addTab(tree_tab, "Tree")

        plots_tab = QWidget()
        plots_lay = QVBoxLayout(plots_tab)
        plots_lay.addWidget(QLabel("Measurements over time"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems(self._measurement_column_choices())
        self._metric_combo.currentIndexChanged.connect(self._refresh_plots)
        plots_lay.addWidget(self._metric_combo)
        self._plot_mode_combo = QComboBox()
        self._plot_mode_combo.addItems(["Per-track metric", "Lineage total area", "Log lineage area"])
        growth_ts = self._tables.get("lineage_growth_timeseries")
        if growth_ts is not None and hasattr(growth_ts, "__len__") and len(growth_ts) > 0:
            idx = self._plot_mode_combo.findText("Lineage total area")
            if idx >= 0:
                self._plot_mode_combo.setCurrentIndex(idx)
        self._plot_mode_combo.currentIndexChanged.connect(self._refresh_plots)
        plots_lay.addWidget(self._plot_mode_combo)
        self._plot_notice = QLabel("")
        self._plot_notice.setWordWrap(True)
        self._plot_notice.setStyleSheet("color: #f0b429; font-weight: 600;")
        self._plot_notice.setVisible(False)
        plots_lay.addWidget(self._plot_notice)
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(4, 2.4), tight_layout=True)
            self._plot_canvas = FigureCanvasQTAgg(self._fig)
            self._ax = self._fig.add_subplot(111)
            plots_lay.addWidget(self._plot_canvas, 1)
        except Exception:
            self._plot_canvas = QLabel("(Plot unavailable — install matplotlib)")
            plots_lay.addWidget(self._plot_canvas)
            self._ax = None
        tabs.addTab(plots_tab, "Plots")

        tabs.setCurrentWidget(events_tab)
        right_lay.addWidget(tabs, 1)
        qc = self._tables.get("lineage_qc")
        if qc is not None and hasattr(qc, "__len__") and len(qc) > 0:
            right_lay.addWidget(QLabel(f"QC flags: {len(qc)} (see lineage report)"))
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        close_btn = QPushButton("Close")
        style_module_primary_button(close_btn)
        close_btn.setMinimumHeight(34)
        close_btn.clicked.connect(self.accept)
        root.addWidget(close_btn)

    def _measurement_column_choices(self) -> list[str]:
        pts = self._tables.get("lineage_points")
        if pts is None or not hasattr(pts, "columns"):
            return ["track presence"]
        choices = ["Area (µm²)"]
        for c in pts.columns:
            cl = str(c).lower()
            if "fi" in cl or "foci" in cl or "coloc" in cl:
                choices.append(str(c))
        if len(choices) == 1:
            choices.append("y position")
        return choices[:16]

    def _populate_track_table(self) -> None:
        meta = self._tables.get("lineage_tracks_meta")
        if meta is not None and hasattr(meta, "columns") and len(meta) > 0:
            df = meta
            cols = [c for c in df.columns if c != "daughter_track_ids"][:10]
        elif self._tracks_df is not None and hasattr(self._tracks_df, "columns"):
            import pandas as pd

            df = (
                self._tracks_df.groupby("track_id", as_index=False)
                .agg({"frame": "count"})
                .rename(columns={"frame": "n_points"})
            )
            cols = list(df.columns)
        else:
            return
        self._track_table.setColumnCount(len(cols))
        self._track_table.setHorizontalHeaderLabels([str(c) for c in cols])
        self._track_table.setRowCount(len(df))
        for ri, row in enumerate(df.itertuples(index=False)):
            for ci, col in enumerate(cols):
                val = getattr(row, col, "")
                self._track_table.setItem(ri, ci, QTableWidgetItem(str(val)))

    def _populate_events_table(self) -> None:
        events = self._tables.get("lineage_events")
        if events is None or not hasattr(events, "columns") or len(events) == 0:
            self._events_table.setRowCount(0)
            return
        show_cols = [
            c
            for c in (
                "parent_track_id",
                "daughter_track_id",
                "division_frame",
                "division_frame_1based",
                "area_asymmetry",
                "division_distance_px",
                "review_status",
                "link_status",
            )
            if c in events.columns
        ]
        if not show_cols:
            show_cols = list(events.columns)[:8]
        self._events_table.setColumnCount(len(show_cols))
        self._events_table.setHorizontalHeaderLabels([str(c) for c in show_cols])
        self._events_table.setRowCount(len(events))
        for ri, row in enumerate(events.itertuples(index=False)):
            for ci, col in enumerate(show_cols):
                val = getattr(row, col, "")
                self._events_table.setItem(ri, ci, QTableWidgetItem(str(val)))

    def _on_track_row_clicked(self, row: int, _col: int) -> None:
        meta = self._tables.get("lineage_tracks_meta")
        if meta is not None and "track_id" in meta.columns and row < len(meta):
            self._selected_track_id = int(meta.iloc[row]["track_id"])
        else:
            item = self._track_table.item(row, 0)
            if item is not None:
                try:
                    self._selected_track_id = int(item.text())
                except ValueError:
                    pass
        self._refresh_plots()
        self._refresh_tree()
        self._show_frame(self._current_frame)

    def _on_event_row_clicked(self, row: int, _col: int) -> None:
        events = self._tables.get("lineage_events")
        if events is None or row >= len(events):
            return
        er = events.iloc[row]
        p = int(er["parent_track_id"])
        d = int(er["daughter_track_id"])
        self._selected_event_key = (p, d)
        self._selected_track_id = d
        div_f = int(er.get("division_frame", 0))
        self._frame_slider.blockSignals(True)
        try:
            self._frame_slider.setValue(max(0, min(div_f, self._frame_slider.maximum())))
        finally:
            self._frame_slider.blockSignals(False)
        self._show_frame(div_f)
        self._refresh_plots()
        self._refresh_tree()

    def _set_selected_event_review(self, status: str) -> None:
        if self._selected_event_key is None:
            return
        p, d = self._selected_event_key
        events = self._tables.get("lineage_events")
        if events is None:
            return
        updated = apply_event_review_status(events, parent_track_id=p, daughter_track_id=d, review_status=status)
        self._tables["lineage_events"] = updated
        self._populate_events_table()

    def _on_frame_changed(self, idx: int) -> None:
        self._show_frame(int(idx))

    def _highlight_frame(self, frame_rgb: np.ndarray, frame_idx: int) -> np.ndarray:
        from bactomate.gui.tabs.tracking import TrackingTab

        out = np.asarray(frame_rgb, dtype=np.uint8).copy()
        if self._selected_track_id is not None and self._tracks_df is not None:
            out = TrackingTab._apply_selected_track_halo(
                out,
                frame_idx,
                int(self._selected_track_id),
                self._masks_tracked,
                self._tracks_df,
            )
        return out

    def _show_frame(self, idx: int) -> None:
        idx = max(0, min(idx, int(self._frames.shape[0]) - 1))
        self._current_frame = idx
        frame = self._highlight_frame(self._frames[idx], idx)
        h, w = frame.shape[:2]
        pix = to_qpixmap(frame)
        self._canvas.set_pixmap(pix, (w, h), reset_view=False)
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
        nodes = self._tables.get("lineage_tree_nodes")
        edges = self._tables.get("lineage_tree_edges")
        if nodes is None:
            from bactomate.modules.lineage_viz import build_lineage_tree_layout

            nodes, edges = build_lineage_tree_layout(
                self._tables.get("lineage_edges") or self._edges_df,
                self._tables.get("lineage_tracks_meta"),
            )
        plot_lineage_tree(
            self._tree_ax,
            nodes,
            edges,
            highlight_track_id=self._selected_track_id,
            title="Lineage tree",
        )
        self._tree_canvas.draw()

    def _refresh_plots(self) -> None:
        if self._ax is None:
            return
        import pandas as pd

        mode = self._plot_mode_combo.currentText() if hasattr(self, "_plot_mode_combo") else "Per-track metric"
        if mode.startswith("Lineage total") or mode.startswith("Log lineage"):
            growth_ts = self._tables.get("lineage_growth_timeseries")
            plot_lineage_growth(
                self._ax,
                growth_ts,
                lineage_id=self._selected_lineage_id(),
                log_scale=mode.startswith("Log"),
                title="Lineage growth",
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
        if col_choice == "Area (µm²)":
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
            self._ax.plot(g["frame"], g[ycol], marker="o", markersize=3, label=str(tid))
        self._ax.set_xlabel("Frame (0-based)")
        self._ax.set_ylabel("Y position (px)" if position_only else ycol)
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

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._refresh_tree()
        self._refresh_plots()


def open_lineage_inspector_from_preview(parent: QWidget, result: Any) -> None:
    """Open inspector from TrackingPreviewResult."""
    if result is None or not getattr(result, "is_lineage_preview", False):
        return
    frames = getattr(result, "frames_rgb", None)
    if frames is None:
        return
    dlg = LineageInspectorDialog(
        parent,
        frames_rgb=frames,
        tracks_df=getattr(result, "tracks_df", None),
        edges_df=getattr(result, "edges_df", None),
        lineage_tables=getattr(result, "lineage_tables", None),
        masks_tracked=getattr(result, "masks_tracked", None),
        title=f"Lineage review — {Path(result.mask_path).name}",
    )
    dlg.exec()


def open_lineage_inspector_from_output_dir(parent: QWidget, output_dir: Path) -> None:
    """Load lineage tables from a completed tracking output folder."""
    import pandas as pd

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    if not tables_dir.is_dir():
        raise FileNotFoundError(f"No tables/ in {output_dir}")
    tables: dict[str, Any] = {}
    for key in (
        "lineage_points",
        "lineage_edges",
        "lineage_tracks_meta",
        "lineage_events",
        "lineage_qc",
        "lineage_summary",
        "lineage_growth_timeseries",
        "lineage_tree_nodes",
        "lineage_tree_edges",
    ):
        for ext in (".parquet", ".csv.gz"):
            p = tables_dir / f"{key}{ext}"
            if p.is_file():
                if ext == ".parquet":
                    tables[key] = pd.read_parquet(p)
                else:
                    tables[key] = pd.read_csv(p)
                break
    pts = tables.get("lineage_points")
    if pts is None:
        raise FileNotFoundError("lineage_points table not found")
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
        title=f"Lineage review — {output_dir.name}",
    )
    dlg.exec()
