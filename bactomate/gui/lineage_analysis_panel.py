"""Lineage analysis: path on the tracking left pane; interactive workspace on the right preview mount."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from bactomate.gui.layout_helpers import (
    module_path_action_buttons_layout,
    module_path_action_outer_layout,
    module_path_drop_row,
    new_module_path_action_panel,
    style_compact_action_button,
    style_module_primary_button,
)
from bactomate.gui.widgets import create_info_icon, show_message_dialog


class LineageAnalysisWorkspace(QWidget):
    """Right-hand workspace: lineage/metric pickers, plots, and export."""

    def __init__(self, controller: "LineageAnalysisPanel", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._status_label = QLabel("Load lineage tracking output using the path on the left.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #9fb0c4;")
        root.addWidget(self._status_label)

        controls = QHBoxLayout()
        controls.setSpacing(12)

        lineage_grp = QGroupBox("Lineages")
        lineage_lay = QVBoxLayout(lineage_grp)
        self._lineage_list = QListWidget()
        self._lineage_list.setMaximumHeight(140)
        self._lineage_list.itemChanged.connect(self._controller._on_selection_changed)
        lineage_lay.addWidget(self._lineage_list)
        lineage_btns = QHBoxLayout()
        sel_all = QPushButton("All")
        sel_none = QPushButton("None")
        style_compact_action_button(sel_all)
        style_compact_action_button(sel_none)
        sel_all.clicked.connect(lambda: self._controller._set_all_lineages(True))
        sel_none.clicked.connect(lambda: self._controller._set_all_lineages(False))
        lineage_btns.addWidget(sel_all)
        lineage_btns.addWidget(sel_none)
        lineage_btns.addStretch(1)
        lineage_lay.addLayout(lineage_btns)
        controls.addWidget(lineage_grp, 1)

        metric_grp = QGroupBox("Metrics")
        metric_lay = QVBoxLayout(metric_grp)
        self._metric_list = QListWidget()
        self._metric_list.setMaximumHeight(140)
        self._metric_list.itemChanged.connect(self._controller._on_selection_changed)
        metric_lay.addWidget(self._metric_list)
        metric_btns = QHBoxLayout()
        m_all = QPushButton("All")
        m_none = QPushButton("None")
        style_compact_action_button(m_all)
        style_compact_action_button(m_none)
        m_all.clicked.connect(lambda: self._controller._set_all_metrics(True))
        m_none.clicked.connect(lambda: self._controller._set_all_metrics(False))
        metric_btns.addWidget(m_all)
        metric_btns.addWidget(m_none)
        metric_btns.addStretch(1)
        metric_lay.addLayout(metric_btns)
        controls.addWidget(metric_grp, 2)

        side_col = QVBoxLayout()
        self._overlay_cb = QCheckBox("Overlay comparison")
        self._overlay_cb.setChecked(True)
        self._overlay_cb.setToolTip("Plot selected lineages on the same axes (Overlay tab).")
        self._overlay_cb.stateChanged.connect(self._controller._refresh_plots)
        side_col.addWidget(self._overlay_cb)
        export_btn = QPushButton("Export figures")
        style_module_primary_button(export_btn)
        export_btn.clicked.connect(self._controller._export_figures)
        side_col.addWidget(export_btn)
        side_col.addStretch(1)
        controls.addLayout(side_col)
        root.addLayout(controls)

        self._tabs = QTabWidget()
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            tree_tab = QWidget()
            tree_lay = QVBoxLayout(tree_tab)
            tree_hdr = QHBoxLayout()
            tree_hdr.addWidget(QLabel("Family tree for:"))
            self._tree_lineage_combo = QComboBox()
            self._tree_lineage_combo.currentIndexChanged.connect(self._controller._refresh_tree_plot)
            tree_hdr.addWidget(self._tree_lineage_combo, 1)
            tree_lay.addLayout(tree_hdr)
            self._tree_fig = Figure(figsize=(8.0, 3.8), tight_layout=True)
            self._tree_canvas = FigureCanvasQTAgg(self._tree_fig)
            self._tree_ax = self._tree_fig.add_subplot(111)
            self._tree_canvas.setMinimumHeight(280)
            tree_lay.addWidget(self._tree_canvas, 1)
            self._tabs.addTab(tree_tab, "Family tree")

            trace_tab = QWidget()
            trace_lay = QVBoxLayout(trace_tab)
            trace_lay.addWidget(QLabel("Per-lineage trace for one checked lineage and the first selected metric."))
            self._trace_fig = Figure(figsize=(8.0, 3.2), tight_layout=True)
            self._trace_canvas = FigureCanvasQTAgg(self._trace_fig)
            self._trace_ax = self._trace_fig.add_subplot(111)
            self._trace_canvas.setMinimumHeight(260)
            trace_lay.addWidget(self._trace_canvas, 1)
            self._tabs.addTab(trace_tab, "Per-lineage traces")

            overlay_tab = QWidget()
            overlay_lay = QVBoxLayout(overlay_tab)
            overlay_lay.addWidget(QLabel("Compare checked lineages on the same axes."))
            self._overlay_fig = Figure(figsize=(8.0, 3.2), tight_layout=True)
            self._overlay_canvas = FigureCanvasQTAgg(self._overlay_fig)
            self._overlay_ax = self._overlay_fig.add_subplot(111)
            self._overlay_canvas.setMinimumHeight(260)
            overlay_lay.addWidget(self._overlay_canvas, 1)
            self._tabs.addTab(overlay_tab, "Overlay comparison")
        except Exception:
            self._tabs.addTab(QLabel("Matplotlib not available for interactive plots."), "Plots")
            self._tree_ax = None
            self._trace_ax = None
            self._overlay_ax = None
            self._tree_canvas = None
            self._trace_canvas = None
            self._overlay_canvas = None

        root.addWidget(self._tabs, 1)


class LineageAnalysisPanel:
    """Coordinates path entry (left) and analysis workspace (right preview mount)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._parent = parent
        self._tables: dict[str, Any] = {}
        self._source_parent: Optional[Path] = None
        self._stem = ""
        self._figures_dir: Optional[Path] = None
        self._mount: Optional[QWidget] = None
        self._preview_surface: Any = None
        self._workspace = LineageAnalysisWorkspace(self)
        self._path_bar = self._build_path_bar()
        self._workspace_visible = False

    def path_bar(self) -> QWidget:
        return self._path_bar

    def _main_window(self) -> QWidget:
        if self._parent is not None:
            w = self._parent.window()
            if w is not None:
                return w
        return self._parent or self._path_bar

    def _build_path_bar(self) -> QWidget:
        host = self._parent if self._parent is not None else QWidget()
        path_panel = new_module_path_action_panel(host)
        path_lay = module_path_action_outer_layout(path_panel)
        file_font = QFont()
        file_font.setPointSize(12)
        self._source_edit = QLineEdit()
        self._source_edit.setPlaceholderText("Tif_files/, …/_tracking/{stem}_tracking/, or *_lineage_report.xlsx")
        self._source_edit.setMinimumWidth(0)
        self._source_edit.setFont(file_font)
        self._source_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._source_edit.editingFinished.connect(self._load_from_edit)
        folder_btn = QPushButton("Browse Folder...")
        folder_btn.clicked.connect(self._browse_folder)
        file_btn = QPushButton("Browse File...")
        file_btn.clicked.connect(self._browse_file)
        path_lay.addWidget(
            module_path_drop_row(
                path_panel,
                self._handle_dropped_path,
                line_edit=self._source_edit,
                buttons=(file_btn, folder_btn),
                info_icon=create_info_icon(
                    "Lineage tracking output folder, parent folder (e.g. Tif_files with _tracking/), "
                    "or *_lineage_report.xlsx.\n\n"
                    "BactoMate searches for _tracking/{stem}_tracking/ automatically. "
                    "Filled after Run lineage tracking; plots open on the right preview pane.",
                    host,
                ),
                path_label="Output:",
            )
        )
        actions_layout = module_path_action_buttons_layout()
        load_btn = QPushButton("Load lineage analysis")
        style_module_primary_button(load_btn)
        load_btn.setToolTip("Load tables from the path above and open plots on the right.")
        load_btn.clicked.connect(lambda: self._load_from_edit(force=True))
        actions_layout.addWidget(load_btn)
        path_lay.addLayout(actions_layout)
        return path_panel

    def _handle_dropped_path(self, dropped_path: str) -> None:
        if dropped_path:
            self.set_source_path(dropped_path, load=True, show_workspace=True)

    def _browse_folder(self) -> None:
        start = self._source_edit.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self._main_window(),
            "Select lineage tracking output folder",
            start,
        )
        if folder:
            self.set_source_path(folder, load=True, show_workspace=True)

    def _browse_file(self) -> None:
        start = self._source_edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self._main_window(),
            "Select lineage report workbook",
            start,
            "Lineage report (*.xlsx);;All files (*)",
        )
        if path:
            self.set_source_path(path, load=True, show_workspace=True)

    def attach_preview_mount(self, mount: Optional[QWidget], preview_surface: Any = None) -> None:
        self._mount = mount
        self._preview_surface = preview_surface
        if self._tables and not self._workspace_visible:
            self.show_workspace()

    def show_workspace(self) -> None:
        if self._mount is None:
            return
        from bactomate.gui.module_shell import mount_embedded_preview

        mount_embedded_preview(
            self._mount,
            self._workspace,
            preview_surface=self._preview_surface,
        )
        if self._preview_surface is not None and hasattr(self._preview_surface, "set_preview_chrome_visible"):
            try:
                self._preview_surface.set_preview_chrome_visible(False)
            except Exception:
                pass
        self._workspace_visible = True

    def set_source_path(self, path: str | Path, *, load: bool = True, show_workspace: bool = True) -> None:
        self._source_edit.setText(str(path))
        if load:
            self._load_source(Path(path), show_workspace=show_workspace)

    def _browse_source(self) -> None:
        """Legacy entry point — opens folder picker first, then file."""
        self._browse_folder()

    def _load_from_edit(self, *, force: bool = False) -> None:
        raw = self._source_edit.text().strip()
        if not raw:
            if force:
                show_message_dialog(
                    self._main_window(),
                    "No path",
                    "Enter or browse to a lineage tracking output folder or report workbook.",
                    kind="info",
                )
            return
        p = Path(raw)
        if p.exists():
            self._load_source(p, show_workspace=True)
        elif force:
            show_message_dialog(
                self._main_window(),
                "Path not found",
                f"Could not find:\n{raw}",
                kind="warning",
            )

    def _load_source(self, source: Path, *, show_workspace: bool = True) -> None:
        from bactomate.modules.lineage_integration import load_lineage_tables_from_source
        from bactomate.modules.lineage_viz import discover_lineage_metric_columns, lineage_ids_from_tables

        try:
            tables, parent, stem, fig_dir = load_lineage_tables_from_source(source)
        except Exception as exc:
            show_message_dialog(self._main_window(), "Load failed", f"Could not load lineage data:\n{exc}", kind="warning")
            return
        self._tables = tables
        self._source_parent = parent
        self._stem = stem
        self._figures_dir = fig_dir
        self._source_edit.setText(str(parent))

        ws = self._workspace
        lids = lineage_ids_from_tables(tables)
        summary = tables.get("lineage_summary")
        n_div = 0
        if summary is not None and hasattr(summary, "columns") and "total_divisions" in summary.columns:
            try:
                n_div = int(summary["total_divisions"].fillna(0).sum())
            except Exception:
                n_div = 0
        ws._status_label.setText(
            f"<b>{stem}</b> — {len(lids)} lineage(s), {n_div} division(s). "
            "Use the controls above the plots to pick lineages and metrics."
        )

        ws._lineage_list.blockSignals(True)
        ws._lineage_list.clear()
        for lid in lids:
            label = f"Lineage {lid}"
            if summary is not None and hasattr(summary, "columns"):
                row = summary[summary["lineage_id"] == lid]
                if len(row) > 0:
                    tr = row.iloc[0].get("n_tracks")
                    dv = row.iloc[0].get("total_divisions")
                    if tr is not None:
                        label += f"  ({tr} tracks"
                        if dv is not None:
                            label += f", {dv} divisions"
                        label += ")"
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, int(lid))
            ws._lineage_list.addItem(item)
        ws._lineage_list.blockSignals(False)

        ws._tree_lineage_combo.blockSignals(True)
        ws._tree_lineage_combo.clear()
        for lid in lids:
            label = f"Lineage {lid}"
            if summary is not None and hasattr(summary, "columns"):
                row = summary[summary["lineage_id"] == lid]
                if len(row) > 0:
                    tr = row.iloc[0].get("n_tracks")
                    if tr is not None:
                        label += f" ({tr} tracks)"
            ws._tree_lineage_combo.addItem(label, int(lid))
        ws._tree_lineage_combo.blockSignals(False)

        discovered = discover_lineage_metric_columns(tables.get("lineage_points"))
        metric_choices: list[str] = []
        for group in ("area", "fi", "foci", "coloc"):
            metric_choices.extend(discovered.get(group, []))
        ws._metric_list.blockSignals(True)
        ws._metric_list.clear()
        for col in metric_choices:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            ws._metric_list.addItem(item)
        ws._metric_list.blockSignals(False)

        if show_workspace:
            self.show_workspace()
        parent = self._parent
        if parent is not None and hasattr(parent, "_reveal_lineage_analysis"):
            try:
                parent._reveal_lineage_analysis()
            except Exception:
                pass
        self._refresh_plots()

    def _selected_lineage_ids(self) -> list[int]:
        out: list[int] = []
        lst = self._workspace._lineage_list
        for i in range(lst.count()):
            item = lst.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                lid = item.data(Qt.ItemDataRole.UserRole)
                if lid is not None:
                    out.append(int(lid))
        return sorted(out)

    def _selected_metrics(self) -> list[str]:
        out: list[str] = []
        lst = self._workspace._metric_list
        for i in range(lst.count()):
            item = lst.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(str(item.text()))
        return out

    def _set_all_lineages(self, on: bool) -> None:
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        lst = self._workspace._lineage_list
        lst.blockSignals(True)
        for i in range(lst.count()):
            lst.item(i).setCheckState(state)
        lst.blockSignals(False)
        self._refresh_plots()

    def _set_all_metrics(self, on: bool) -> None:
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        lst = self._workspace._metric_list
        lst.blockSignals(True)
        for i in range(lst.count()):
            lst.item(i).setCheckState(state)
        lst.blockSignals(False)
        self._refresh_plots()

    def _on_selection_changed(self, *_args) -> None:
        self._refresh_plots()

    def _tree_lineage_id(self) -> Optional[int]:
        data = self._workspace._tree_lineage_combo.currentData()
        return int(data) if data is not None else None

    def _refresh_tree_plot(self, *_args) -> None:
        ws = self._workspace
        if not self._tables or ws._tree_ax is None:
            return
        from bactomate.modules.lineage_viz import (
            _biology_label_map_from_tables,
            build_lineage_tree_layout,
            plot_lineage_tree,
        )

        lid = self._tree_lineage_id()
        if lid is None:
            return
        meta = self._tables.get("lineage_tracks_meta")
        edges = self._tables.get("lineage_edges")
        label_map = _biology_label_map_from_tables(self._tables)
        nodes, tree_edges = build_lineage_tree_layout(edges, meta, lineage_ids=[int(lid)])
        plot_lineage_tree(
            ws._tree_ax,
            nodes,
            tree_edges,
            track_labels=label_map,
            title=f"Lineage {lid} — family tree",
        )
        ws._tree_canvas.draw()

    def _refresh_plots(self, *_args) -> None:
        ws = self._workspace
        if not self._tables or ws._trace_ax is None:
            return
        self._refresh_tree_plot()
        from bactomate.modules.lineage_viz import (
            _biology_label_map_from_tables,
            _track_area_from_tables,
            default_metric_aggregate,
            plot_lineage_cell_cycle_area,
            plot_lineage_growth,
            plot_lineage_growth_overlay,
            plot_lineage_metric_overlay,
            plot_lineage_metric_timeseries,
        )

        lids = self._selected_lineage_ids()
        metrics = self._selected_metrics()
        meta = self._tables.get("lineage_tracks_meta")
        edges = self._tables.get("lineage_edges")
        events = self._tables.get("lineage_events")
        growth_ts = self._tables.get("lineage_growth_timeseries")
        pts = self._tables.get("lineage_points")
        label_map = _biology_label_map_from_tables(self._tables)
        area_df = _track_area_from_tables(self._tables)

        ws._trace_ax.clear()
        if len(lids) == 1 and metrics and pts is not None and metrics[0] in pts.columns:
            plot_lineage_metric_timeseries(
                ws._trace_ax,
                pts,
                metric_column=metrics[0],
                lineage_id=int(lids[0]),
                aggregate=default_metric_aggregate(metrics[0]),
                log_scale="intensity" in metrics[0].lower(),
                title=f"{metrics[0]} — lineage {lids[0]}",
                events_df=events,
            )
        elif len(lids) == 1 and growth_ts is not None:
            plot_lineage_growth(ws._trace_ax, growth_ts, lineage_id=int(lids[0]), title=f"Growth — lineage {lids[0]}", events_df=events)
        elif len(lids) == 1 and area_df is not None:
            plot_lineage_cell_cycle_area(
                ws._trace_ax,
                area_df,
                edges,
                lineage_id=int(lids[0]),
                label_map=label_map,
                events_df=events,
                title=f"Cell-cycle area — lineage {lids[0]}",
            )
        elif len(lids) > 1:
            ws._trace_ax.text(
                0.5,
                0.5,
                "Select one lineage for per-lineage trace,\nor use the Overlay tab.",
                ha="center",
                va="center",
                transform=ws._trace_ax.transAxes,
            )
            ws._trace_ax.set_title("Per-lineage traces")
        else:
            ws._trace_ax.text(0.5, 0.5, "Select at least one lineage.", ha="center", va="center", transform=ws._trace_ax.transAxes)
        ws._trace_canvas.draw()

        ws._overlay_ax.clear()
        if ws._overlay_cb.isChecked() and len(lids) >= 1:
            if metrics and pts is not None and metrics[0] in pts.columns:
                plot_lineage_metric_overlay(
                    ws._overlay_ax,
                    pts,
                    metrics[0],
                    lids,
                    aggregate=default_metric_aggregate(metrics[0]),
                    log_scale="intensity" in metrics[0].lower(),
                    title=f"Overlay — {metrics[0]}",
                )
            elif growth_ts is not None:
                plot_lineage_growth_overlay(ws._overlay_ax, growth_ts, lids, title="Growth overlay", events_df=events)
            else:
                ws._overlay_ax.text(0.5, 0.5, "No growth or metric data for overlay.", ha="center", va="center", transform=ws._overlay_ax.transAxes)
        else:
            ws._overlay_ax.text(
                0.5,
                0.5,
                "Enable overlay comparison to plot multiple lineages.",
                ha="center",
                va="center",
                transform=ws._overlay_ax.transAxes,
            )
        ws._overlay_canvas.draw()

    def _export_figures(self) -> None:
        if not self._tables or self._figures_dir is None or not self._stem:
            show_message_dialog(self._main_window(), "Export", "Load lineage data first.", kind="info")
            return
        from bactomate.modules.lineage_viz import export_lineage_analysis_figures, export_lineage_figures

        lids = self._selected_lineage_ids()
        metrics = self._selected_metrics()
        fig_root = Path(self._figures_dir)
        fig_root.mkdir(parents=True, exist_ok=True)
        try:
            written = export_lineage_figures(fig_root, self._tables, stem=self._stem)
            written.extend(
                export_lineage_analysis_figures(
                    fig_root,
                    self._tables,
                    stem=self._stem,
                    lineage_ids=lids,
                    metric_columns=metrics,
                )
            )
        except Exception as exc:
            show_message_dialog(self._main_window(), "Export failed", str(exc), kind="error")
            return
        show_message_dialog(
            self._main_window(),
            "Export complete",
            f"Wrote {len(written)} figure(s) to:\n{fig_root}\n\nDetailed plots: {fig_root / 'analysis'}",
            kind="info",
        )
