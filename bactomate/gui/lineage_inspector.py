"""Interactive lineage inspector: synchronized image, track table, and growth plot."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from bactomate.gui.image_canvas import ImageCanvas
from bactomate.gui.image_utils import to_qpixmap


class LineageInspectorDialog(QDialog):
    """Inspect lineage preview or saved tracking outputs in memory."""

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        frames_rgb: np.ndarray,
        tracks_df: Any,
        edges_df: Any = None,
        lineage_tables: Optional[dict[str, Any]] = None,
        title: str = "Lineage inspector",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(960, 640)
        self._frames = np.asarray(frames_rgb, dtype=np.uint8)
        self._tracks_df = tracks_df
        self._edges_df = edges_df
        self._tables = lineage_tables or {}
        self._selected_track_id: Optional[int] = None
        self._build_ui()
        self._populate_track_table()
        if self._frames.shape[0] > 0:
            self._show_frame(0)

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
        right_lay.addWidget(QLabel("Tracks (double-click to select)"))
        self._track_table = QTableWidget()
        self._track_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._track_table.cellDoubleClicked.connect(self._on_track_row_clicked)
        right_lay.addWidget(self._track_table, 2)

        right_lay.addWidget(QLabel("Area over time (selected track / lineage)"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems(self._measurement_column_choices())
        self._metric_combo.currentIndexChanged.connect(self._refresh_plot)
        right_lay.addWidget(self._metric_combo)
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(4, 2.2), tight_layout=True)
            self._plot_canvas = FigureCanvasQTAgg(self._fig)
            self._ax = self._fig.add_subplot(111)
            right_lay.addWidget(self._plot_canvas, 1)
        except Exception:
            self._plot_canvas = QLabel("(Plot unavailable — install matplotlib)")
            right_lay.addWidget(self._plot_canvas)
            self._ax = None

        qc = self._tables.get("lineage_qc")
        if qc is not None and hasattr(qc, "__len__") and len(qc) > 0:
            right_lay.addWidget(QLabel(f"QC flags: {len(qc)} (see lineage report)"))
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        close_btn = QPushButton("Close")
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
        return choices[:12]

    def _populate_track_table(self) -> None:
        meta = self._tables.get("lineage_tracks_meta")
        if meta is not None and hasattr(meta, "columns") and len(meta) > 0:
            df = meta
            cols = list(df.columns)
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
        self._refresh_plot()

    def _on_frame_changed(self, idx: int) -> None:
        self._show_frame(int(idx))

    def _show_frame(self, idx: int) -> None:
        idx = max(0, min(idx, int(self._frames.shape[0]) - 1))
        frame = self._frames[idx]
        h, w = frame.shape[:2]
        pix = to_qpixmap(frame)
        self._canvas.set_pixmap(pix, (w, h), reset_view=False)
        self._frame_label.setText(f"Frame {idx + 1} / {self._frames.shape[0]}")

    def _refresh_plot(self) -> None:
        if self._ax is None:
            return
        import pandas as pd

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
        elif "y" in sub.columns:
            ycol = "y"
        if ycol is None or "frame" not in sub.columns:
            self._ax.set_title("No plottable column")
            self._plot_canvas.draw()
            return
        for tid, g in sub.groupby("track_id"):
            g = g.sort_values("frame")
            self._ax.plot(g["frame"], g[ycol], marker="o", markersize=3, label=str(tid))
        self._ax.set_xlabel("Frame (0-based)")
        self._ax.set_ylabel(ycol)
        self._ax.set_title(f"Lineage {self._selected_track_id or 'all'}")
        if sub["track_id"].nunique() <= 8:
            self._ax.legend(fontsize=7, loc="best")
        self._plot_canvas.draw()


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
        title=f"Lineage inspector — {Path(result.mask_path).name}",
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
        "lineage_qc",
        "lineage_summary",
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
    dlg = LineageInspectorDialog(
        parent,
        frames_rgb=frames,
        tracks_df=tracks_df,
        edges_df=tables.get("lineage_edges"),
        lineage_tables=tables,
        title=f"Lineage inspector — {output_dir.name}",
    )
    dlg.exec()
