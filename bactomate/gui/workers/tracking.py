"""
Trackastra tracking worker (preview + full export, lineage tables).
"""
import logging
import os
import sys
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from bactomate.gui.log_config import get_logger, log_exception
from bactomate.gui.workers.tracking_io import (
    assign_cell_ids_to_tracks_df,
    blend_path_pixel_on_frame,
    build_track_points_by_id,
    compact_tracking_label,
    draw_track_centroid_marker,
    draw_trajectory_paths_on_frame,
    overlay_label_for_track,
    prepare_preview_display_stacks,
    resolve_track_at_centroid,
    scale_track_points_by_id,
    scale_tracks_df_for_display,
    should_draw_track_id_labels,
    tracking_export_upscale_factor,
    tracking_export_viz_style,
    upscale_mask_stack,
    upscale_rgb_stack,
)
from bactomate.modules.label_render import draw_hq_labels
from bactomate.modules.tiff_stack_io import read_tiff_stack

_TRACKASTRA_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_TRACKASTRA_MODEL_CACHE_LOCK = threading.Lock()

class _TrackingLogHandler(logging.Handler):
    """Forwards log records to a TrackingWorker's progress signal so the GUI log shows trackastra status."""

    def __init__(self, worker, progress_pct: int):
        super().__init__(level=logging.INFO)
        self._worker = worker
        self._progress_pct = progress_pct

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage().strip()
            if msg:
                self._worker.progress.emit(self._progress_pct, msg)
        except Exception:
            pass


def _cached_trackastra_model(Trackastra: Any, model_name: str, device: str, debug_output: list[str]) -> Any:
    """Reuse Trackastra models across preview/run attempts in the current GUI process."""
    key = (str(model_name), str(device))
    with _TRACKASTRA_MODEL_CACHE_LOCK:
        model = _TRACKASTRA_MODEL_CACHE.get(key)
    if model is not None:
        debug_output.append(f"✓ Reusing cached Trackastra model ({model_name}, {device})")
        return model

    model = Trackastra.from_pretrained(model_name, device=device)
    with _TRACKASTRA_MODEL_CACHE_LOCK:
        _TRACKASTRA_MODEL_CACHE[key] = model
    return model
@dataclass
class TrackingPreviewResult:
    """In-memory tracking preview (no files written)."""

    mask_path: Path
    frames_rgb: np.ndarray
    tracks_df: Any
    original_path: Optional[Path] = None
    n_tracks: int = 0
    # (T, Y, X) intensity stack used for preview rendering (display resolution).
    stack_images: Any = None
    masks_input: Any = None
    masks_tracked: Any = None
    edges_df: Any = None
    founder_by_track: Optional[dict[int, int]] = None
    lineage_tables: Optional[dict[str, Any]] = None
    is_lineage_preview: bool = False
    # Map full-resolution track coords → display-stack coords (1.0 when not scaled).
    display_scale_y: float = 1.0
    display_scale_x: float = 1.0


class TrackingWorker(QThread):
    """Worker thread for running tracking"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)  # progress percentage, status message
    preview_ready = pyqtSignal(object)  # TrackingPreviewResult when write_outputs is False

    @staticmethod
    def _imagej_compatible_label_stack(mask_stack: Any) -> tuple[np.ndarray, bool]:
        """Return a TIFF label stack compatible with ImageJ when label values allow it."""
        arr = np.asarray(mask_stack)
        if arr.size == 0:
            return arr.astype(np.uint16, copy=False), True
        max_label = int(np.nanmax(arr))
        min_label = int(np.nanmin(arr))
        if min_label < 0:
            arr = np.clip(arr, 0, None)
            min_label = 0
        if max_label <= np.iinfo(np.uint16).max:
            return arr.astype(np.uint16, copy=False), True
        return arr.astype(np.uint32, copy=False), False

    @staticmethod
    def _track_max_jump_px(
        filter_params: Optional[dict],
        *,
        original_shape: tuple[int, ...] | None = None,
        tracking_shape: tuple[int, ...] | None = None,
    ) -> float:
        """Convert configured max-link jump to tracking-space pixels."""
        params = dict(filter_params or {})
        jump = float(params.get("track_max_jump", 0.0) or 0.0)
        if jump <= 0:
            return 0.0
        pixel_size = float(params.get("pixel_size", 0.0) or 0.0)
        if pixel_size <= 0 or not bool(params.get("use_physical_units", True)):
            return jump
        scale = 1.0
        if original_shape is not None and tracking_shape is not None and len(original_shape) >= 3 and len(tracking_shape) >= 3:
            try:
                orig_y, orig_x = float(original_shape[-2]), float(original_shape[-1])
                track_y, track_x = float(tracking_shape[-2]), float(tracking_shape[-1])
                if orig_y > 0 and orig_x > 0 and track_y > 0 and track_x > 0:
                    scale_y = track_y / orig_y
                    scale_x = track_x / orig_x
                    scale = max(1e-9, 0.5 * (scale_y + scale_x))
            except Exception:
                scale = 1.0
        # After spatial downsampling, each tracking-space pixel spans more microns.
        tracking_pixel_size = pixel_size / scale
        return float(jump) / float(tracking_pixel_size) if tracking_pixel_size > 0 else 0.0

    @staticmethod
    def _prune_long_jump_edges(
        track_graph: Any,
        masks_original: Any,
        *,
        filter_params: Optional[dict] = None,
        original_shape: tuple[int, ...] | None = None,
        debug_output: Optional[list[str]] = None,
    ) -> tuple[Any, Any, int]:
        """Remove graph edges whose step distance exceeds the configured hard cap."""
        params = dict(filter_params or {})
        max_jump_px = TrackingWorker._track_max_jump_px(
            params,
            original_shape=original_shape,
            tracking_shape=getattr(masks_original, "shape", None),
        )
        if max_jump_px <= 0 or track_graph is None or masks_original is None:
            return track_graph, masks_original, 0

        removed: list[tuple[Any, Any, int, int, float]] = []
        pruned = track_graph.copy()
        for (src, dst) in list(pruned.edges()):
            try:
                src_node = pruned.nodes[src]
                dst_node = pruned.nodes[dst]
                src_coords = np.asarray(src_node.get("coords"), dtype=float).ravel()
                dst_coords = np.asarray(dst_node.get("coords"), dtype=float).ravel()
                if src_coords.size < 2 or dst_coords.size < 2:
                    continue
                dist_px = float(np.hypot(dst_coords[-2] - src_coords[-2], dst_coords[-1] - src_coords[-1]))
                if dist_px <= max_jump_px:
                    continue
                src_t = int(src_node.get("time", src_node.get("frame", -1)))
                dst_t = int(dst_node.get("time", dst_node.get("frame", -1)))
                removed.append((src, dst, src_t, dst_t, dist_px))
            except Exception:
                continue

        if not removed:
            return track_graph, masks_original, 0

        pruned.remove_edges_from([(src, dst) for (src, dst, _t1, _t2, _dist) in removed])
        try:
            from trackastra.tracking import graph_to_ctc

            _ctc_df, masks_pruned = graph_to_ctc(pruned, masks_original, check=False)
        except Exception as exc:
            if debug_output is not None:
                debug_output.append(f"⚠ Max link jump gate could not rebuild tracked masks: {exc}")
            return track_graph, masks_original, 0

        if debug_output is not None:
            examples = ", ".join(
                f"t{t1}->{t2}:{dist:.1f}px" for (_s, _d, t1, t2, dist) in removed[:5]
            )
            debug_output.append(
                f"✓ Max link jump gate: removed {len(removed)} edge(s) above {max_jump_px:.1f} px"
                + (f" ({examples})" if examples else "")
            )
        return pruned, masks_pruned, len(removed)

    @staticmethod
    def _apply_jump_prune(
        track_graph: Any,
        masks_tracked: Any,
        masks_input: Any,
        *,
        filter_params: Optional[dict] = None,
        original_shape: tuple[int, ...] | None = None,
        debug_output: Optional[list[str]] = None,
    ) -> tuple[Any, Any, int]:
        """Prune long-jump edges; only replace ``masks_tracked`` when edges were removed.

        ``_prune_long_jump_edges`` rebuilds masks from ``masks_input`` (raw segmentation)
        via ``graph_to_ctc``. On no-op paths it returns that input stack — callers must
        not overwrite Trackastra tracked masks unless ``n_removed > 0``.
        """
        pruned_graph, masks_pruned, n_removed = TrackingWorker._prune_long_jump_edges(
            track_graph,
            masks_input,
            filter_params=filter_params,
            original_shape=original_shape,
            debug_output=debug_output,
        )
        if int(n_removed) > 0:
            return pruned_graph, masks_pruned, int(n_removed)
        return pruned_graph, masks_tracked, 0

    def __init__(
        self,
        mask_file,
        original_file,
        excel_file,
        model_name,
        mode,
        device,
        parent=None,
        output_dir=None,
        track_action: str = "movement",
        frame_limit: Optional[int] = None,
        *,
        write_outputs: bool = True,
        run_metadata: Optional[dict] = None,
        filter_params: Optional[dict] = None,
        write_visualization: bool = True,
        preview_lineage: bool = False,
        lineage_curation_path: Optional[Path] = None,
    ):
        super().__init__(parent)
        self.setObjectName(type(self).__name__)
        self.mask_file = mask_file
        self.original_file = original_file
        self.excel_file = excel_file if write_outputs else None
        self.model_name = model_name
        self.mode = mode
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        self.track_action = track_action  # "preview" | "movement" | "lineage"
        self.preview_lineage = bool(preview_lineage)
        self.frame_limit = frame_limit
        self.write_outputs = bool(write_outputs)
        self.run_metadata = run_metadata or {}
        self.filter_params = filter_params or {}
        self.write_visualization = bool(write_visualization)
        self.lineage_curation_path = Path(lineage_curation_path) if lineage_curation_path else None
        self._lineage_curation_review_map: dict[tuple[int, int], str] = {}
        profile = os.getenv("BACTOMATE_TRACKING_OUTPUT_PROFILE", "minimal")
        profile = str(profile).strip().lower()
        if profile not in {"minimal", "legacy", "both"}:
            profile = "minimal"
        self.output_profile = profile

    def run(self) -> None:
        # Initialize debug output list
        debug_output = []
        debug_output.append("=== Tracking Debug Information ===")
        
        try:
            # NumPy 2.0+ removed np.bool; geff (trackastra dependency) still uses it in typing
            import numpy as np
            if not hasattr(np, "bool"):
                np.bool = np.bool_
            from trackastra.model import Trackastra
            from trackastra.tracking import graph_to_edge_table, graph_to_ctc, graph_to_napari_tracks
            import tifffile
            import pandas as pd
            from pathlib import Path
            import networkx as nx

            def _read_tiff_stack(path: str | Path, label: str) -> tuple[np.ndarray, int | None, bool]:
                return read_tiff_stack(
                    path,
                    label=label,
                    frame_limit=int(self.frame_limit or 0),
                    preview_mode=not self.write_outputs,
                    log=debug_output.append,
                )
            
            debug_output.append("✓ All imports successful")
            self.progress.emit(5, "Loading masks...")
            
            # Load masks
            debug_output.append(f"\nLoading masks from: {self.mask_file}")
            masks, mask_total_frames, mask_was_limited_read = _read_tiff_stack(self.mask_file, "Masks")
            debug_output.append(f"✓ Masks loaded - Shape: {masks.shape}")
            self.progress.emit(10, "Masks loaded")
            
            if len(masks.shape) != 3:
                error_msg = f"Expected 3D mask array (T, Y, X), got shape: {masks.shape}\n\n" + "\n".join(debug_output)
                self.finished.emit(False, error_msg)
                return
            
            # Load original images if provided (required for trackastra)
            if self.original_file:
                self.progress.emit(15, "Loading original images...")
                debug_output.append(f"\nLoading original images from: {self.original_file}")
                imgs, image_total_frames, image_was_limited_read = _read_tiff_stack(
                    self.original_file, "Images"
                )
                debug_output.append(f"✓ Images loaded - Shape: {imgs.shape}")
                self.progress.emit(20, "Images loaded")
                
                if len(imgs.shape) != 3:
                    error_msg = f"Expected 3D image array (T, Y, X), got shape: {imgs.shape}\n\n" + "\n".join(debug_output)
                    self.finished.emit(False, error_msg)
                    return
                # Ensure images and masks have same number of frames
                if imgs.shape[0] != masks.shape[0]:
                    error_msg = f"Image frames ({imgs.shape[0]}) don't match mask frames ({masks.shape[0]})\n\n" + "\n".join(debug_output)
                    self.finished.emit(False, error_msg)
                    return
            else:
                # If no original images provided, create dummy images (trackastra needs them)
                # Use masks as images (will work but not ideal)
                self.progress.emit(15, "Creating images from masks...")
                debug_output.append("\nNo original images provided - using masks as images")
                imgs = masks.astype(np.float32)
                debug_output.append("✓ Created dummy images from masks")
                self.progress.emit(20, "Images ready")
                image_total_frames = mask_total_frames
                image_was_limited_read = mask_was_limited_read

            full_T = int(mask_total_frames or masks.shape[0])
            if self.frame_limit is not None and int(self.frame_limit) > 0:
                n_use = min(int(self.frame_limit), full_T)
                if masks.shape[0] > n_use:
                    masks = masks[:n_use]
                if imgs.shape[0] > n_use:
                    imgs = imgs[:n_use]
                if n_use < full_T:
                    debug_output.append(
                        f"Frame limit ({self.track_action}): using first {n_use} of {full_T} frames"
                    )
                elif mask_was_limited_read or image_was_limited_read:
                    debug_output.append(
                        f"Frame limit ({self.track_action}): using all {n_use} available frame(s)"
                    )

            pre_downsample_mask_shape = tuple(masks.shape)
            if not self.write_outputs:
                debug_output.append(
                    f"Preview mode: tracking uses full resolution (shape {masks.shape}); only the displayed preview is downsampled"
                )
            
            # Load Excel data if provided
            cell_data = None
            if self.excel_file:
                self.progress.emit(25, "Loading Excel analysis data...")
                try:
                    excel_path = Path(self.excel_file)
                    debug_output.append(f"\nLoading Excel file: {excel_path}")
                    # Try to read the per-cell data sheet
                    excel_data = pd.read_excel(excel_path, sheet_name=None)
                    debug_output.append(f"✓ Excel file loaded with sheets: {list(excel_data.keys())}")
                    
                    # Look for sheet with per-cell data (usually named "Per-Cell Data" or similar)
                    for sheet_name in excel_data.keys():
                        if 'cell' in sheet_name.lower() or 'per' in sheet_name.lower():
                            cell_data = excel_data[sheet_name]
                            debug_output.append(f"✓ Using sheet: {sheet_name} (per-cell data)")
                            break
                    if cell_data is None:
                        # Use first sheet if no per-cell sheet found
                        cell_data = list(excel_data.values())[0]
                        debug_output.append(f"✓ Using first sheet: {list(excel_data.keys())[0]}")
                    
                    debug_output.append(f"  - Cell data shape: {cell_data.shape}")
                    debug_output.append(f"  - Cell data columns: {list(cell_data.columns)}")
                    if len(cell_data) > 0:
                        debug_output.append(f"  - Sample row: {cell_data.iloc[0].to_dict()}")
                    self.progress.emit(30, "Excel data loaded")
                except Exception as e:
                    import traceback
                    error_msg = f"Error reading Excel file: {str(e)}\n\n{traceback.format_exc()}"
                    self.finished.emit(False, error_msg)
                    return
            
            # Load pretrained model with detailed debugging
            self.progress.emit(35, f"Loading Trackastra model ({self.model_name})...")
            debug_output.append(f"\nLoading Trackastra model: {self.model_name}")
            debug_output.append(f"Device: {self.device}")
            
            try:
                import torch
                if self.device == 'cuda':
                    debug_output.append(f"CUDA available: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        debug_output.append(f"GPU: {torch.cuda.get_device_name(0)}")
                        debug_output.append(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                elif self.device == 'mps':
                    debug_output.append(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        debug_output.append("GPU: Apple Silicon (MPS)")
                else:
                    debug_output.append("Using CPU")
            except:
                pass
            
            try:
                debug_output.append("Calling Trackastra.from_pretrained()...")
                model = _cached_trackastra_model(
                    Trackastra,
                    self.model_name,
                    self.device,
                    debug_output,
                )
                debug_output.append("✓ Model loaded successfully")
                self.progress.emit(50, "Model loaded, starting tracking...")
            except Exception as e:
                import traceback
                from bactomate.gui.trackastra_utils import format_trackastra_model_error

                error_details = traceback.format_exc()
                debug_output.append(f"✗ Error loading model: {str(e)}")
                debug_output.append(f"\nFull traceback:\n{error_details}")

                short = format_trackastra_model_error(e)
                if short != str(e):
                    error_msg = f"Error loading Trackastra model:\n\n{short}\n\n"
                    error_msg += "Debug information:\n" + "\n".join(debug_output[-8:])
                else:
                    error_msg = f"Error loading Trackastra model: {str(e)}\n\n"
                    error_msg += "Debug information:\n" + "\n".join(debug_output)
                    error_msg += "\n\nPossible issues:"
                    error_msg += "\n- Model name incorrect"
                    error_msg += "\n- Trackastra not properly installed"
                    error_msg += "\n- CUDA/GPU issues (if using cuda device)"
                    error_msg += "\n- Network issues (downloading model)"

                self.finished.emit(False, error_msg)
                return
            
            # Run tracking with detailed debugging
            link_mode = self.mode
            if (
                self.track_action == "lineage" or getattr(self, "preview_lineage", False)
            ) and link_mode == "greedy_nodiv":
                link_mode = "greedy"
                debug_output.append(
                    "Lineage run: greedy_nodiv cannot represent divisions — using greedy linking for model.track()."
                )
            debug_output.append(f"\nStarting tracking with mode: {link_mode}")
            debug_output.append(f"Input shapes - Images: {imgs.shape}, Masks: {masks.shape}")
            
            try:
                import torch
                if self.device == 'cuda' and torch.cuda.is_available():
                    debug_output.append(f"GPU memory before tracking: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                    debug_output.append(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                elif self.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    debug_output.append("MPS device ready for tracking")
            except:
                pass
            
            # Initialize trackid_to_cellid mapping (will be set if Excel file is provided)
            trackid_to_cellid = None
            
            # Forward trackastra logging (Extracting features, Building windows, etc.) to GUI status log
            trackastra_logger = logging.getLogger("trackastra")
            log_handler = _TrackingLogHandler(self, 55)
            trackastra_logger.addHandler(log_handler)
            try:
                self.progress.emit(
                    55,
                    "Running tracking algorithm (feature extraction & linking — this may take several minutes, please wait)..."
                )
                debug_output.append("Calling model.track()...")
                # Run model.track() in a background thread so we can emit "still running" updates
                result_holder = [None, None, None]  # track_graph, masks_tracked, exception
                def run_track():
                    try:
                        g, m = model.track(imgs, masks, mode=link_mode)
                        result_holder[0], result_holder[1] = g, m
                    except Exception as e:
                        result_holder[2] = e
                tracking_thread = threading.Thread(target=run_track, daemon=False)
                tracking_thread.start()
                elapsed_sec = 0
                update_interval_sec = 15
                while tracking_thread.is_alive():
                    time.sleep(update_interval_sec)
                    elapsed_sec += update_interval_sec
                    if elapsed_sec < 60:
                        time_str = f"{elapsed_sec} s"
                    else:
                        time_str = f"{elapsed_sec // 60} min"
                    self.progress.emit(
                        55,
                        f"Still running… ({time_str}) — feature extraction and linking can take several minutes"
                    )
                tracking_thread.join()
                if result_holder[2] is not None:
                    raise result_holder[2]
                track_graph, masks_tracked = result_holder[0], result_holder[1]
                track_graph, masks_tracked, n_pruned_jump_edges = self._apply_jump_prune(
                    track_graph,
                    masks_tracked,
                    masks,
                    filter_params=self.filter_params,
                    original_shape=pre_downsample_mask_shape,
                    debug_output=debug_output,
                )
                if n_pruned_jump_edges > 0:
                    debug_output.append(
                        f"  - Rebuilt tracked masks after splitting {n_pruned_jump_edges} long-jump link(s)"
                    )
                debug_output.append("✓ Tracking completed successfully")
                self.progress.emit(70, "Tracking completed, processing results...")
                
                # Map track IDs back to analysis CellIDs so one CellID stays with the same cell
                if self.excel_file and Path(self.excel_file).exists():
                    self.progress.emit(72, "Mapping track IDs to analysis CellIDs (from Excel)...")
                    debug_output.append("\nMapping track IDs to analysis CellIDs (from Excel)...")
                    try:
                        masks_tracked, trackid_to_cellid = self._map_track_ids_to_analysis_cellids(
                            masks, masks_tracked, self.excel_file
                        )
                        debug_output.append(f"✓ Mapped {len(trackid_to_cellid)} track IDs to analysis CellIDs")
                    except Exception as e:
                        import traceback
                        debug_output.append(f"⚠ Could not map track IDs to analysis CellIDs: {str(e)}")
                        debug_output.append(f"  {traceback.format_exc()}")
                        trackid_to_cellid = None
                else:
                    # No Excel: preserve CellIDs from input masks when they already have them (e.g. from Analyze Masks)
                    total_unique = len(np.unique(masks))
                    per_frame_uniques = [len(np.unique(masks[i])) for i in range(masks.shape[0])]
                    avg_per_frame = sum(per_frame_uniques) / len(per_frame_uniques) if per_frame_uniques else 0
                    # Persistent IDs reuse the same labels across frames; per-frame labels do not
                    input_looks_persistent = avg_per_frame > 0 and total_unique <= 2.0 * avg_per_frame
                    if input_looks_persistent:
                        self.progress.emit(72, "Preserving CellIDs from input masks...")
                        try:
                            masks_tracked, trackid_to_cellid = self._map_track_ids_to_cellids_from_input_masks(
                                masks, masks_tracked
                            )
                            if trackid_to_cellid:
                                debug_output.append(f"✓ Preserved {len(trackid_to_cellid)} CellIDs from input masks")
                            else:
                                trackid_to_cellid = None
                        except Exception as e:
                            debug_output.append(f"  (Could not preserve CellIDs from input: {e})")
                            trackid_to_cellid = None
                    else:
                        trackid_to_cellid = None
                debug_output.append(f"  - Track graph type: {type(track_graph)}")
                debug_output.append(f"  - Track graph nodes: {track_graph.number_of_nodes() if hasattr(track_graph, 'number_of_nodes') else 'N/A'}")
                debug_output.append(f"  - Track graph edges: {track_graph.number_of_edges() if hasattr(track_graph, 'number_of_edges') else 'N/A'}")
                debug_output.append(f"  - Tracked masks shape: {masks_tracked.shape}")
                debug_output.append(f"  - Tracked masks dtype: {masks_tracked.dtype}")
                debug_output.append(f"  - Original masks max: {masks.max()}")
                debug_output.append(f"  - Tracked masks max: {masks_tracked.max()}")
                debug_output.append(f"  - Unique labels in first frame (original): {len(np.unique(masks[0]))}")
                debug_output.append(f"  - Unique labels in first frame (tracked): {len(np.unique(masks_tracked[0]))}")
                debug_output.append(f"  - Unique labels in last frame (original): {len(np.unique(masks[-1]))}")
                debug_output.append(f"  - Unique labels in last frame (tracked): {len(np.unique(masks_tracked[-1]))}")
                # Check if track IDs are consistent across frames
                if masks_tracked.shape[0] > 1:
                    unique_tracks = set()
                    for i in range(masks_tracked.shape[0]):
                        unique_tracks.update(np.unique(masks_tracked[i]))
                    debug_output.append(f"  - Total unique track IDs across all frames: {len(unique_tracks)}")
                    # Check if masks changed (should have consistent track IDs)
                    if np.array_equal(masks, masks_tracked):
                        debug_output.append("  ⚠ WARNING: Tracked masks are identical to input masks!")
                        debug_output.append("    This suggests track IDs were not properly assigned.")
                    else:
                        debug_output.append("  ✓ Tracked masks differ from input (track IDs assigned)")
                
                try:
                    import torch
                    if self.device == 'cuda' and torch.cuda.is_available():
                        debug_output.append(f"GPU memory after tracking: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                        torch.cuda.empty_cache()
                        debug_output.append("✓ GPU cache cleared")
                    elif self.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        debug_output.append("✓ MPS cache cleared")
                except:
                    pass
                    
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                debug_output.append(f"✗ Error during tracking: {str(e)}")
                debug_output.append(f"\nFull traceback:\n{error_details}")
                
                try:
                    import torch
                    if self.device == 'cuda' and torch.cuda.is_available():
                        debug_output.append(f"\nGPU memory at error: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                        debug_output.append(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                    elif self.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        debug_output.append("\nMPS device error occurred")
                except:
                    pass
                
                error_msg = f"Error during tracking: {str(e)}\n\n"
                error_msg += "Debug information:\n" + "\n".join(debug_output)
                error_msg += "\n\nPossible issues:"
                error_msg += "\n- GPU out of memory (try CPU mode)"
                error_msg += "\n- CUDA runtime error"
                error_msg += "\n- Input data format issues"
                error_msg += "\n- Model compatibility issues"
                
                self.finished.emit(False, error_msg)
                return
            finally:
                trackastra_logger.removeHandler(log_handler)

            # Extract track information from the graph (preview and full runs)
            self.progress.emit(75, "Extracting track information...")
            debug_output.append("\nExtracting track information from graph...")

            # Method 1: Use graph_to_napari_tracks to get proper track data
            napari_graph = None
            try:
                napari_tracks, napari_graph, _ = graph_to_napari_tracks(track_graph)
                debug_output.append(f"✓ Extracted {len(napari_tracks)} track points from graph")
                debug_output.append(f"  - Napari tracks shape: {napari_tracks.shape if hasattr(napari_tracks, 'shape') else 'N/A'}")
                
                # Convert napari tracks to DataFrame
                # napari_tracks is numpy array with columns: [track_id, frame, y, x, ...]
                if len(napari_tracks) > 0 and hasattr(napari_tracks, 'shape'):
                    # napari_tracks is typically (N, 4+) array: [track_id, frame, y, x, ...]
                    num_cols = napari_tracks.shape[1] if len(napari_tracks.shape) > 1 else 4
                    col_names = ['track_id', 'frame', 'y', 'x']
                    if num_cols > 4:
                        col_names.extend([f'property_{i}' for i in range(num_cols - 4)])
                    
                    tracks_df = pd.DataFrame(napari_tracks, columns=col_names[:num_cols])
                    # Ensure track_id and frame are integers
                    if 'track_id' in tracks_df.columns:
                        tracks_df['track_id'] = tracks_df['track_id'].astype(int)
                    if 'frame' in tracks_df.columns:
                        tracks_df['frame'] = tracks_df['frame'].astype(int)
                    debug_output.append(f"✓ Created tracks DataFrame with {len(tracks_df)} rows")
                    debug_output.append(f"  - Columns: {list(tracks_df.columns)}")
                else:
                    debug_output.append("⚠ Warning: No tracks extracted from graph or invalid format")
                    tracks_df = pd.DataFrame(columns=['track_id', 'frame', 'y', 'x'])
            except Exception as e:
                debug_output.append(f"⚠ graph_to_napari_tracks failed: {str(e)}")
                # Fallback: Extract tracks manually from graph
                debug_output.append("Trying manual track extraction from graph...")
                tracks_list = []
                
                # Extract all nodes and their properties
                for node_id, node_data in track_graph.nodes(data=True):
                    # Node data typically has 'time' or 'frame' and position info
                    frame = node_data.get('time', node_data.get('frame', node_data.get('t', None)))
                    if frame is not None:
                        track_id = node_data.get('track_id', node_id)
                        y = node_data.get('y', node_data.get('centroid_y', 0))
                        x = node_data.get('x', node_data.get('centroid_x', 0))
                        tracks_list.append({
                            'track_id': track_id,
                            'frame': int(frame),
                            'y': y,
                            'x': x,
                            'node_id': node_id
                        })
                
                if tracks_list:
                    tracks_df = pd.DataFrame(tracks_list)
                    debug_output.append(f"✓ Manually extracted {len(tracks_df)} track points")
                else:
                    # Last resort: use edge table
                    debug_output.append("Trying graph_to_edge_table as fallback...")
                    try:
                        edge_df = graph_to_edge_table(track_graph)
                        debug_output.append(f"✓ Edge table has {len(edge_df)} edges")
                        # Try to extract track info from edges
                        if len(edge_df) > 0:
                            tracks_df = edge_df
                        else:
                            tracks_df = pd.DataFrame(columns=['track_id', 'frame', 'y', 'x'])
                    except Exception as e2:
                        debug_output.append(f"✗ All extraction methods failed: {str(e2)}")
                        tracks_df = pd.DataFrame(columns=['track_id', 'frame', 'y', 'x'])
            
            if len(tracks_df) > 0:
                tracks_df = assign_cell_ids_to_tracks_df(
                    tracks_df,
                    masks,
                    excel_file=self.excel_file,
                    trackid_to_cellid=trackid_to_cellid,
                )
                if "Cell_ID" in tracks_df.columns:
                    debug_output.append(
                        f"✓ Assigned analysis Cell_IDs per detection ({tracks_df['Cell_ID'].nunique()} unique)"
                    )
            
            # Add debug info about tracks
            debug_output.append(f"\nTrack DataFrame info:")
            debug_output.append(f"  - Shape: {tracks_df.shape}")
            debug_output.append(f"  - Columns: {list(tracks_df.columns)}")
            if len(tracks_df) > 0:
                debug_output.append(f"  - Unique track IDs: {tracks_df['track_id'].nunique() if 'track_id' in tracks_df.columns else 'N/A'}")
                if 'Cell_ID' in tracks_df.columns:
                    debug_output.append(f"  - Unique Cell IDs: {tracks_df['Cell_ID'].nunique()}")
                debug_output.append(f"  - Frame range: {tracks_df['frame'].min() if 'frame' in tracks_df.columns else 'N/A'} to {tracks_df['frame'].max() if 'frame' in tracks_df.columns else 'N/A'}")

            if not self.write_outputs:
                self._emit_preview_result(
                    debug_output,
                    imgs,
                    masks,
                    masks_tracked,
                    tracks_df,
                    napari_graph=napari_graph,
                )
                return

            # Save tracking results (full runs only — preview never reaches here)
            mask_path = Path(self.mask_file)
            if self.output_dir is not None:
                output_dir = self.output_dir
            else:
                from bactomate.gui.workers.tracking_io import tracking_output_dir

                output_dir = tracking_output_dir(mask_path, track_action=str(self.track_action or "movement"))
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(self.mask_file).stem
            write_legacy_outputs = self.output_profile in {"legacy", "both"}
            write_minimal_outputs = self.output_profile in {"minimal", "both"}
            debug_output.append(
                f"Output profile: {self.output_profile} (minimal={write_minimal_outputs}, legacy={write_legacy_outputs})"
            )

            # Read calibration from original or mask TIFF so we can preserve it in written TIFFs
            from bactomate.modules.tiff_metadata import build_metadata_dict, resolve_calibration_for_mask
            _cal_path = Path(self.mask_file) if self.mask_file else None
            if _cal_path is not None:
                self._cal_ps, self._cal_tp, _, _ = resolve_calibration_for_mask(_cal_path)
            else:
                self._cal_ps, self._cal_tp = 0.0, 0.0

            precuration_edges_df = None
            if self.track_action == "lineage":
                applied = self._maybe_apply_saved_lineage_curation(
                    tracks_df=tracks_df,
                    masks_tracked=masks_tracked,
                    masks_input=masks,
                    napari_graph=napari_graph,
                    debug_output=debug_output,
                )
                if applied is not None:
                    tracks_df, masks_tracked, precuration_edges_df = applied

            # Optional proximity stitch (gap closing) before writing tracks / masks.
            stitch_id_map: dict[int, int] = {}
            n_stitch = 0
            try:
                from bactomate.gui.track_stitch import (
                    apply_stitch_from_params,
                    remap_napari_lineage_graph,
                )

                tracks_df, masks_tracked, n_stitch, stitch_pairs, stitch_id_map = apply_stitch_from_params(
                    tracks_df,
                    masks_tracked,
                    self.filter_params,
                )
                if n_stitch > 0:
                    debug_output.append(
                        f"✓ Proximity stitch: merged {n_stitch} track fragment pair(s) "
                        f"(e.g. {stitch_pairs[:5]})"
                    )
                    if stitch_id_map and napari_graph is not None:
                        napari_graph = remap_napari_lineage_graph(napari_graph, stitch_id_map)
                    if stitch_id_map and precuration_edges_df is not None:
                        from bactomate.gui.track_stitch import remap_lineage_edges_df

                        precuration_edges_df = remap_lineage_edges_df(
                            precuration_edges_df, stitch_id_map
                        )
            except Exception as e:
                debug_output.append(f"⚠ Proximity stitch skipped: {e}")
                stitch_id_map = {}
                n_stitch = 0

            # Save tracks as CSV: track_id, frame, y, x (and Cell_ID if analysis Excel was used).
            # y,x come from Trackastra's track graph (centroid-like node positions; napari format or node centroid_y/x).
            self.progress.emit(80, "Saving tracks...")
            tracks_file = output_dir / f"{stem}_tracks.csv"
            tracks_df.to_csv(tracks_file, index=False)
            debug_output.append(f"✓ Saved tracks to: {tracks_file}")
            
            # Save tracked masks
            self.progress.emit(85, "Saving tracked masks...")
            masks_tracked_file = output_dir / f"{stem}_masks_tracked.tif"
            debug_output.append(f"\nSaving tracked masks...")
            debug_output.append(f"  - Shape: {masks_tracked.shape}")
            debug_output.append(f"  - Original dtype: {np.asarray(masks_tracked).dtype}")
            debug_output.append(f"  - Unique labels per frame: {[len(np.unique(masks_tracked[i])) for i in range(min(5, masks_tracked.shape[0]))]}")
            T = masks_tracked.shape[0]
            _meta = build_metadata_dict(
                pixel_size_um=self._cal_ps if self._cal_ps > 0 else None,
                time_per_frame_min=self._cal_tp if self._cal_tp > 0 else None,
                images=T, frames=T,
            )
            masks_to_write, imagej_ok = self._imagej_compatible_label_stack(masks_tracked)
            debug_output.append(f"  - Saved dtype: {masks_to_write.dtype}")
            if imagej_ok:
                tifffile.imwrite(str(masks_tracked_file), masks_to_write, metadata=_meta, imagej=True)
            else:
                debug_output.append(
                    "  - More than 65535 labels detected; saving uint32 TIFF without ImageJ wrapper."
                )
                tifffile.imwrite(str(masks_tracked_file), masks_to_write, metadata=_meta)
            debug_output.append(f"✓ Saved tracked masks to: {masks_tracked_file}")
            
            # Create overlay frames with track IDs in memory only (used as base for trajectories)
            overlay_array = None
            if self.write_visualization and self.original_file:
                self.progress.emit(87, "Creating overlay with track IDs...")
                debug_output.append("\nCreating tracked overlay with track IDs...")
                try:
                    overlay_file = output_dir / f"{stem}_overlay_tracked.tif"

                    def overlay_progress(frame_idx, total, _msg=None):
                        pct = 87 + int((frame_idx + 1) / max(total, 1) * 2)  # 87–89%
                        msg = f"Creating overlay: {frame_idx + 1}/{total}"
                        self.progress.emit(pct, msg)
                    overlay_array, overlay_export = self._create_overlay_with_track_ids(
                        imgs,
                        masks_tracked,
                        tracks_df,
                        output_path=overlay_file,
                        progress_callback=overlay_progress,
                    )
                    debug_output.append(f"✓ Saved tracked overlay to: {overlay_file}")
                except Exception as e:
                    import traceback
                    debug_output.append(f"⚠ Could not create tracked overlay: {str(e)}")
                    debug_output.append(f"  {traceback.format_exc()}")
            
            # Create visualization with trajectories overlaid (use overlay as base when available — much faster)
            if self.write_visualization and self.original_file:
                self.progress.emit(90, "Creating trajectory visualization...")
                debug_output.append("\nCreating trajectory visualization...")
                if overlay_array is not None:
                    debug_output.append("  (Using overlay as base — drawing trajectories only.)")
                else:
                    debug_output.append("  (This may take a while for large datasets - optimizing...)")
                try:
                    vis_file = output_dir / f"{stem}_trajectories.tif"
                    def progress_callback(pct, msg):
                        self.progress.emit(pct, msg)
                    self._create_trajectory_visualization(
                        imgs, masks_tracked, tracks_df, vis_file, progress_callback,
                        overlay_frames=overlay_export if overlay_export is not None else overlay_array,
                    )
                    debug_output.append(f"✓ Saved trajectory visualization to: {vis_file}")
                except Exception as e:
                    import traceback
                    debug_output.append(f"⚠ Could not create trajectory visualization: {str(e)}")
                    debug_output.append(f"  {traceback.format_exc()}")
                    self.progress.emit(96, f"WARNING: Trajectory file was not created: {e}")
            
            edges_df = pd.DataFrame(
                columns=["parent_track_id", "daughter_track_id", "division_frame", "parent_last_frame"]
            )
            edges_path = None
            lineage_summary_legacy = None
            if self.track_action == "lineage":
                self.progress.emit(92, "Building lineage tracking outputs...")
                try:
                    from bactomate.modules.lineage_integration import lineage_edges_from_napari_graph

                    if precuration_edges_df is not None:
                        edges_df = precuration_edges_df
                    else:
                        edges_df = lineage_edges_from_napari_graph(napari_graph, tracks_df)
                    if write_legacy_outputs:
                        edges_path = output_dir / f"{stem}_lineage_edges.csv"
                        edges_df.to_csv(edges_path, index=False)
                        debug_output.append(f"✓ Saved lineage edges: {edges_path.name}")
                        summary_df = self._lineage_per_track_summary(edges_df, tracks_df)
                        lineage_summary_legacy = output_dir / f"{stem}_lineage_summary.xlsx"
                        with pd.ExcelWriter(lineage_summary_legacy, engine="openpyxl") as writer:
                            edges_df.to_excel(writer, sheet_name="Lineage edges", index=False)
                            summary_df.to_excel(writer, sheet_name="Per-track summary", index=False)
                        debug_output.append(f"✓ Saved lineage summary: {lineage_summary_legacy.name}")
                    if self.write_visualization and self.original_file:
                        try:
                            from bactomate.modules.lineage_integration import (
                                build_biology_track_labels,
                                compact_biology_label_map,
                            )

                            founder_by_track = self._lineage_founder_map(edges_df, tracks_df)
                            lineage_meta = self._lineage_per_track_summary(edges_df, tracks_df)
                            bio_labels_df = build_biology_track_labels(
                                edges_df, tracks_df, lineage_meta
                            )
                            biology_label_by_track = compact_biology_label_map(bio_labels_df)
                            vis_lineage = output_dir / f"{stem}_lineage.tif"
                            self._create_lineage_visualization(
                                imgs,
                                masks_tracked,
                                tracks_df,
                                edges_df,
                                founder_by_track,
                                vis_lineage,
                                progress_callback=lambda pct, msg: self.progress.emit(
                                    min(99, max(92, int(pct))), msg
                                ),
                                overlay_frames=None,
                                masks_input=masks,
                                biology_label_by_track=biology_label_by_track,
                            )
                            debug_output.append(
                                f"✓ Saved lineage visualization: {vis_lineage.name} "
                                "(founder colors, paths, biology labels e.g. 7.1.1)"
                            )
                        except Exception as e:
                            import traceback
                            debug_output.append(f"⚠ Lineage visualization failed: {e}")
                            debug_output.append(traceback.format_exc())
                except Exception as e:
                    import traceback
                    debug_output.append(f"⚠ Lineage tracking output failed: {e}")
                    debug_output.append(traceback.format_exc())
            
            # Save in CTC format (Cell Tracking Challenge format)
            try:
                ctc_dir = output_dir / "ctc_format"
                ctc_dir.mkdir(exist_ok=True)
                ctc_graph = track_graph
                if n_stitch > 0:
                    from bactomate.gui.track_stitch import track_graph_from_tracks_df

                    ctc_graph = track_graph_from_tracks_df(tracks_df)
                    debug_output.append(
                        "  - CTC rebuilt from stitched tracks (Trackastra graph was remapped via stitch)"
                    )
                graph_to_ctc(ctc_graph, masks_tracked, outdir=str(ctc_dir))
            except Exception as e:
                # CTC format is optional, don't fail if it doesn't work
                pass
            
            # If Excel data was provided, merge with tracks
            merged_data = None
            merged_file = None
            if cell_data is not None:
                self.progress.emit(95, "Merging with analysis data...")
                # Merge tracking data with cell properties
                merged_data = self._merge_tracking_with_analysis(tracks_df, cell_data, masks_tracked)
                if write_legacy_outputs:
                    merged_file = output_dir / f"{stem}_tracks_with_analysis.xlsx"
                    with pd.ExcelWriter(merged_file, engine='openpyxl') as writer:
                        merged_data.to_excel(writer, sheet_name='Tracks with Analysis', index=False)
                        tracks_df.to_excel(writer, sheet_name='Tracks', index=False)
                        if isinstance(cell_data, pd.DataFrame):
                            cell_data.to_excel(writer, sheet_name='Original Analysis', index=False)
                    debug_output.append(f"✓ Saved merged tracks workbook: {merged_file.name}")

            report_file = None
            tables_dir = None
            written_tables: dict[str, str] = {}
            if write_minimal_outputs:
                try:
                    self.progress.emit(97, "Building simplified tabular outputs...")
                    trajectory_tables = self._build_trajectory_tables(
                        tracks_df=tracks_df,
                        masks_tracked=masks_tracked,
                        merged_data=merged_data,
                    )
                    lineage_tables = None
                    lineage_tables = None
                    if self.track_action == "lineage":
                        lineage_tables = self._build_integrated_lineage_tables(
                            tracks_df=tracks_df,
                            edges_df=edges_df,
                            masks_tracked=masks_tracked,
                            merged_data=merged_data,
                            output_dir=output_dir,
                            stem=stem,
                        )
                    report_file = output_dir / f"{stem}_report.xlsx"
                    self._write_report_workbook(
                        output_path=report_file,
                        trajectory_tables=trajectory_tables,
                        lineage_tables=lineage_tables,
                        run_metadata=self.run_metadata,
                        filter_params=self.filter_params,
                    )
                    debug_output.append(f"✓ Saved report workbook: {report_file.name}")
                    tables_dir, written_tables = self._write_canonical_tables(
                        output_dir=output_dir,
                        trajectory_tables=trajectory_tables,
                        lineage_tables=lineage_tables,
                    )
                    debug_output.append(f"✓ Saved canonical tables in: {tables_dir}")
                except Exception as e:
                    import traceback
                    debug_output.append(f"⚠ Could not write simplified tabular outputs: {e}")
                    debug_output.append(traceback.format_exc())

            track_count = int(tracks_df["track_id"].nunique()) if ("track_id" in tracks_df.columns and len(tracks_df) > 0) else 0
            track_points = int(len(tracks_df))
            frame_count = int(masks_tracked.shape[0]) if hasattr(masks_tracked, "shape") and len(masks_tracked.shape) > 0 else 0
            division_count = int(len(edges_df)) if edges_df is not None and len(edges_df) > 0 else 0
            self.progress.emit(
                99,
                "[RESULT][TRACK] "
                f"mask={Path(self.mask_file).resolve()} "
                f"original={Path(self.original_file).resolve() if self.original_file else 'none'} "
                f"analysis_excel={Path(self.excel_file).resolve() if self.excel_file else 'none'} "
                f"output_dir={Path(output_dir).resolve()}",
            )
            self.progress.emit(
                99,
                "[RESULT][TRACK] "
                f"tracks={track_count} track_points={track_points} frames={frame_count} "
                f"divisions={division_count} action={self.track_action}",
            )
            self.progress.emit(
                99,
                "[RESULT][TRACK] "
                f"tracks_csv={Path(tracks_file).resolve()} "
                f"tracked_masks={Path(masks_tracked_file).resolve()} "
                f"report={Path(report_file).resolve() if report_file is not None else 'none'}",
            )

            self.progress.emit(100, "Tracking completed!")
            success_msg = "Tracking completed!\n\n"
            success_msg += "\n".join(debug_output)
            success_msg += f"\n\nTracks saved to: {tracks_file}\n"
            success_msg += f"Tracked masks saved to: {masks_tracked_file}\n"
            if report_file is not None:
                success_msg += f"Report workbook saved to: {report_file}\n"
            if tables_dir is not None:
                success_msg += f"Canonical tables saved to: {tables_dir}\n"
                if written_tables:
                    success_msg += "Canonical table files:\n"
                    for k in sorted(written_tables.keys()):
                        success_msg += f"  - {k}: {written_tables[k]}\n"
            if edges_path is not None:
                success_msg += f"Legacy lineage edges saved to: {edges_path}\n"
            if lineage_summary_legacy is not None:
                success_msg += f"Legacy lineage summary saved to: {lineage_summary_legacy}\n"
            if merged_file is not None:
                success_msg += f"Merged data saved to: {merged_file}\n"
            self.finished.emit(True, success_msg)
                
        except ImportError as e:
            debug_output.append(f"\n✗ Import error: {str(e)}")
            error_msg = f"trackastra is not installed or import failed: {str(e)}\n\n"
            error_msg += "Debug information:\n" + "\n".join(debug_output)
            err_str = str(e).lower()
            if "cbuffer_sizes" in err_str and "numcodecs" in err_str:
                error_msg += (
                    "\n\nThis is a known numcodecs/zarr compatibility issue. Try one of:\n"
                    "  1. Force-reinstall numcodecs:  pip install --force-reinstall numcodecs==0.15.1\n"
                    "  2. Or use zarr 3 (no cbuffer_sizes):  pip install 'zarr>=3.0'\n"
                )
            else:
                error_msg += "\n\nPlease install trackastra: pip install trackastra"
            self.finished.emit(False, error_msg)
        except Exception as e:
            log_exception(get_logger(), "TrackingWorker.run failed")
            import traceback
            debug_output.append(f"\n✗ Unexpected error: {str(e)}")
            debug_output.append(f"\nFull traceback:\n{traceback.format_exc()}")
            error_msg = f"Tracking error: {str(e)}\n\n"
            error_msg += "Debug information:\n" + "\n".join(debug_output)
            self.finished.emit(False, error_msg)
    
    def _map_track_ids_to_analysis_cellids(self, masks_original, masks_tracked, excel_file):
        """
        Map track IDs back to analysis CellIDs by matching cells between original and tracked masks.
        
        Parameters:
        -----------
        masks_original : array
            Original masks (may have analysis CellIDs if from analyzed masks)
        masks_tracked : array
            Tracked masks with track IDs
        excel_file : str
            Path to Excel file with analysis data containing Cell_ID column
            
        Returns:
        --------
        masks_tracked_remapped : array
            Tracked masks remapped to use analysis CellIDs
        trackid_to_cellid : dict
            Mapping from track_id to analysis CellID
        """
        import pandas as pd
        import numpy as np
        from bactomate.modules.cell_id_linking import centroids_from_mask, hungarian_link_centroids
        
        # Load Excel data to get Cell_IDs
        try:
            excel_data = pd.read_excel(excel_file, sheet_name='Per-Cell Data')
        except:
            # Try other sheet names
            try:
                excel_data = pd.read_excel(excel_file, sheet_name=0)
            except:
                return masks_tracked, {}
        
        # Find Cell_ID column
        cell_id_col = None
        for col in ['Cell_ID', 'cell_id', 'Cell ID', 'CellID']:
            if col in excel_data.columns:
                cell_id_col = col
                break
        
        if cell_id_col is None:
            # No Cell_ID column found, return original tracked masks
            return masks_tracked, {}
        
        # Find frame column
        frame_col = None
        for col in ['Frame', 'frame', 't', 'time', 'Time']:
            if col in excel_data.columns:
                frame_col = col
                break
        
        if frame_col is None:
            return masks_tracked, {}
        
        # Build mapping: (frame, original_label) -> Cell_ID from Excel
        excel_mapping = {}
        if "Cell Label" in excel_data.columns:
            valid = excel_data[[frame_col, "Cell Label", cell_id_col]].dropna()
            for frame_val, orig_label, cell_id in zip(
                valid[frame_col].astype(int) - 1,
                valid["Cell Label"].astype(int),
                valid[cell_id_col].astype(int),
            ):
                excel_mapping[(int(frame_val), int(orig_label))] = int(cell_id)
        
        if not excel_mapping:
            return masks_tracked, {}
        
        # Now map track IDs to analysis CellIDs by matching cells
        # Strategy: Match cells in each frame between original and tracked masks
        trackid_to_cellid = {}
        num_frames = min(masks_original.shape[0], masks_tracked.shape[0])
        max_distance = 10.0  # pixels
        
        for frame_idx in range(num_frames):
            frame_original = masks_original[frame_idx]
            frame_tracked = masks_tracked[frame_idx]
            
            orig_centroids = centroids_from_mask(frame_original, pixel_size=1.0)
            tracked_centroids = centroids_from_mask(frame_tracked, pixel_size=1.0)
            
            orig_label_to_cellid = {
                label: excel_mapping[(frame_idx, label)]
                for label in orig_centroids
                if (frame_idx, label) in excel_mapping
            }
            
            if not orig_label_to_cellid or not tracked_centroids:
                continue
            
            orig_labels = np.array(
                [label for label in orig_centroids if label in orig_label_to_cellid],
                dtype=int,
            )
            tracked_labels = np.array(list(tracked_centroids.keys()), dtype=int)
            orig_points = np.array([orig_centroids[int(label)] for label in orig_labels], dtype=np.float64)
            tracked_points = np.array(list(tracked_centroids.values()), dtype=np.float64)
            
            linked, _ = hungarian_link_centroids(
                orig_points,
                orig_labels,
                tracked_points,
                tracked_labels,
                max_distance,
            )
            for orig_label, track_id in linked:
                trackid_to_cellid[int(track_id)] = orig_label_to_cellid[int(orig_label)]
        
        # Remap tracked masks to use analysis CellIDs
        masks_tracked_remapped = np.zeros_like(masks_tracked)
        for frame_idx in range(num_frames):
            frame_tracked = masks_tracked[frame_idx]
            frame_remapped = masks_tracked_remapped[frame_idx]
            
            for track_id, cell_id in trackid_to_cellid.items():
                frame_remapped[frame_tracked == track_id] = cell_id
        
        return masks_tracked_remapped, trackid_to_cellid
    
    def _map_track_ids_to_cellids_from_input_masks(self, masks_original, masks_tracked):
        """
        Preserve CellIDs from input masks when they already contain persistent IDs
        (e.g. from Analyze Masks). Build track_id -> cell_id by sampling the original
        mask at each track's centroid per frame and using the most common value.
        No Excel required.
        """
        import numpy as np
        from skimage.measure import regionprops
        from collections import Counter
        
        num_frames = min(masks_original.shape[0], masks_tracked.shape[0])
        track_id_to_values = {}  # track_id -> list of (frame_idx, original_label at centroid)
        
        for frame_idx in range(num_frames):
            frame_orig = masks_original[frame_idx]
            frame_tracked = masks_tracked[frame_idx]
            h, w = frame_orig.shape[:2]
            for region in regionprops(frame_tracked):
                if region.label == 0:
                    continue
                track_id = region.label
                r, c = int(round(region.centroid[0])), int(round(region.centroid[1]))
                if 0 <= r < h and 0 <= c < w:
                    orig_val = int(frame_orig[r, c])
                    if track_id not in track_id_to_values:
                        track_id_to_values[track_id] = []
                    track_id_to_values[track_id].append(orig_val)
        
        trackid_to_cellid = {}
        for track_id, values in track_id_to_values.items():
            # Exclude background (0); take most common label as CellID
            non_zero = [v for v in values if v != 0]
            if not non_zero:
                continue
            mode_val, count = Counter(non_zero).most_common(1)[0]
            # Only assign if the same ID appears in at least half of the samples (persistent)
            if count >= max(1, len(values) // 2):
                trackid_to_cellid[track_id] = mode_val
        
        if not trackid_to_cellid:
            return masks_tracked, {}
        
        # Remap tracked masks to use those CellIDs (same as Excel path)
        masks_tracked_remapped = np.zeros_like(masks_tracked)
        for frame_idx in range(num_frames):
            frame_tracked = masks_tracked[frame_idx]
            frame_remapped = masks_tracked_remapped[frame_idx]
            for track_id, cell_id in trackid_to_cellid.items():
                frame_remapped[frame_tracked == track_id] = cell_id
            # Preserve track_ids that were not mapped (keep original track_id)
            unmapped = (frame_tracked > 0) & (frame_remapped == 0)
            frame_remapped[unmapped] = frame_tracked[unmapped]
        
        return masks_tracked_remapped, trackid_to_cellid
    
    def _merge_tracking_with_analysis(self, tracks_df, cell_data, masks_tracked):
        """Merge tracking results with Excel analysis data"""
        import pandas as pd
        import numpy as np
        
        # Ensure cell_data is a DataFrame
        if not isinstance(cell_data, pd.DataFrame):
            cell_data = pd.DataFrame(cell_data)
        
        # Debug: Check what columns we have
        debug_info = []
        debug_info.append(f"Tracks DataFrame columns: {list(tracks_df.columns)}")
        debug_info.append(f"Cell data DataFrame columns: {list(cell_data.columns)}")
        debug_info.append(f"Tracks shape: {tracks_df.shape}, Cell data shape: {cell_data.shape}")
        
        # Find frame column in cell_data (could be 'frame', 'Frame', 't', 'time', etc.)
        frame_col = None
        for col in ['frame', 'Frame', 't', 'time', 'Time', 'T']:
            if col in cell_data.columns:
                frame_col = col
                break
        
        if frame_col is None:
            debug_info.append("⚠ Warning: Could not find frame column in cell_data")
            # Return tracks_df with a note
            return tracks_df
        
        debug_info.append(f"Using frame column: {frame_col}")
        
        # Find cell_id column in cell_data
        cell_id_col = None
        for col in ['cell_id', 'Cell_ID', 'cell', 'Cell', 'label', 'Label']:
            if col in cell_data.columns:
                cell_id_col = col
                break
        
        # Extract track information from tracked masks
        # In tracked masks, each label is a track_id that persists across frames
        # We need to map (frame, track_id) to original cell properties from Excel
        
        merged_list = []
        
        # For each track point, find matching cell data
        matches_found = 0
        matches_by_position = 0
        matches_by_first = 0
        
        # Track which frames we've already warned about
        frames_with_no_cells = set()
        
        # Detect if Excel uses 1-based indexing by checking if frame 0 has no data but frame 1 does
        frame_0_count = len(cell_data[cell_data[frame_col].astype(int) == 0]) if frame_col else 0
        frame_1_count = len(cell_data[cell_data[frame_col].astype(int) == 1]) if frame_col else 0
        excel_is_1_indexed = (frame_0_count == 0 and frame_1_count > 0)
        if excel_is_1_indexed:
            debug_info.append(f"Detected 1-based frame indexing in Excel (Frame 0 has {frame_0_count} cells, Frame 1 has {frame_1_count} cells)")

        cell_data_by_frame: dict[int, pd.DataFrame] = {}
        if frame_col:
            for frame_key, group in cell_data.groupby(cell_data[frame_col].astype(int)):
                cell_data_by_frame[int(frame_key)] = group.copy()

        def _frame_cells_for_track(frame: int) -> pd.DataFrame:
            excel_frame = frame + 1 if excel_is_1_indexed else frame
            if excel_frame in cell_data_by_frame:
                return cell_data_by_frame[excel_frame]
            alt_frame = frame + 1 if not excel_is_1_indexed else frame
            return cell_data_by_frame.get(alt_frame, pd.DataFrame())
        
        for track_row in tracks_df.to_dict("records"):
            frame = None
            for col in ['frame', 't', 'time', 'Frame', 'T', 'Time']:
                if col in track_row and pd.notna(track_row[col]):
                    frame = int(track_row[col])
                    break
            
            # Get track_id
            track_id = None
            for col in ['track_id', 'track', 'Track', 'track_ID']:
                if col in track_row and pd.notna(track_row[col]):
                    track_id = int(track_row[col])
                    break
            
            if frame is None:
                debug_info.append("  Skipping row: frame is None")
                continue
            if track_id is None:
                debug_info.append("  Skipping row: track_id is None")
                continue
            
            # Get track position from tracks_df
            track_y = None
            track_x = None
            for col in ['y', 'Y', 'centroid_y', 'centroid-0']:
                if col in track_row and pd.notna(track_row[col]):
                    track_y = float(track_row[col])
                    break
            for col in ['x', 'X', 'centroid_x', 'centroid-1']:
                if col in track_row and pd.notna(track_row[col]):
                    track_x = float(track_row[col])
                    break
            
            # Start with track information
            merged_row = dict(track_row)
            
            # Find cells in this frame from cell_data
            frame_cells = _frame_cells_for_track(frame)
            excel_frame = frame + 1 if excel_is_1_indexed else frame
            
            if len(frame_cells) > 0:
                matched = False
                # Try to match by position if we have coordinates
                if track_y is not None and track_x is not None:
                    # Find centroid columns in cell_data
                    y_col = None
                    x_col = None
                    for col in ['centroid_y', 'Centroid_Y', 'y', 'Y', 'centroid-0', 'centroid_y-0']:
                        if col in frame_cells.columns:
                            y_col = col
                            break
                    for col in ['centroid_x', 'Centroid_X', 'x', 'X', 'centroid-1', 'centroid_x-1']:
                        if col in frame_cells.columns:
                            x_col = col
                            break
                    
                    if y_col and x_col:
                        # Calculate distances and find closest match
                        try:
                            distances = np.sqrt(
                                (frame_cells[y_col].astype(float) - track_y)**2 + 
                                (frame_cells[x_col].astype(float) - track_x)**2
                            )
                            closest_idx = distances.idxmin()
                            closest_dist = distances.loc[closest_idx]
                            
                            # If close enough (within 20 pixels), use this match
                            if closest_dist < 20:
                                matched_cell = frame_cells.loc[closest_idx]
                                # Add all cell properties to merged row
                                for col in matched_cell.index:
                                    if col not in merged_row or pd.isna(merged_row.get(col)):
                                        merged_row[col] = matched_cell[col]
                                matches_by_position += 1
                                matched = True
                        except Exception as e:
                            debug_info.append(f"  Error matching by position: {str(e)}")
                
                if not matched and cell_id_col:
                    # Fall back only to an explicit Cell_ID match. Using the first cell in the
                    # frame fabricates repeated area values and makes lineage growth plots misleading.
                    candidate_ids = [track_id]
                    try:
                        if "Cell_ID" in track_row and pd.notna(track_row["Cell_ID"]):
                            candidate_ids.insert(0, int(track_row["Cell_ID"]))
                    except Exception:
                        pass
                    matched_cell = None
                    for candidate_id in candidate_ids:
                        try:
                            matching = frame_cells[pd.to_numeric(frame_cells[cell_id_col], errors="coerce") == int(candidate_id)]
                        except Exception:
                            matching = pd.DataFrame()
                        if len(matching) > 0:
                            matched_cell = matching.iloc[0]
                            break
                    if matched_cell is not None:
                        for col in matched_cell.index:
                            if col not in merged_row or pd.isna(merged_row.get(col)):
                                merged_row[col] = matched_cell[col]
                        matches_by_first += 1
                        matched = True
                
                if matched:
                    matches_found += 1
            else:
                # Only log once per frame to reduce noise
                if frame not in frames_with_no_cells:
                    frames_with_no_cells.add(frame)
                    debug_info.append(f"  Frame {frame} (Excel Frame {excel_frame}): No cells found in cell_data")
            
            merged_list.append(merged_row)
        
        debug_info.append(f"\nMerge statistics:")
        debug_info.append(f"  - Total track points: {len(tracks_df)}")
        debug_info.append(f"  - Matches found: {matches_found}")
        debug_info.append(f"  - Matched by position: {matches_by_position}")
        debug_info.append(f"  - Matched by first cell: {matches_by_first}")
        
        if merged_list:
            merged = pd.DataFrame(merged_list)
            debug_info.append(f"✓ Merged DataFrame created with {len(merged)} rows")
        else:
            debug_info.append("⚠ Warning: No merged rows created")
            merged = tracks_df.copy()
        
        # Log debug info
        import sys
        print("\n".join(debug_info), file=sys.stderr)
        
        return merged

    @staticmethod
    def _json_list(values: list[int]) -> str:
        import json

        return json.dumps([int(v) for v in values], separators=(",", ":"))

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            if isinstance(value, float) and np.isnan(value):
                return None
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            f = float(value)
            if np.isnan(f):
                return None
            return f
        except Exception:
            return None

    def _build_track_cell_table(self, tracks_df) -> tuple[dict[int, list[int]], Any]:
        import pandas as pd

        empty_df = pd.DataFrame(columns=["track_id", "cell_id"])
        if tracks_df is None or len(tracks_df) == 0 or "track_id" not in tracks_df.columns:
            return {}, empty_df
        cell_col = "Cell_ID" if "Cell_ID" in tracks_df.columns else "track_id"
        mapping: dict[int, list[int]] = {}
        rows: list[dict[str, int]] = []
        for tid_raw, g in tracks_df.groupby("track_id"):
            tid = self._safe_int(tid_raw)
            if tid is None:
                continue
            vals: list[int] = []
            for v in g[cell_col].tolist():
                iv = self._safe_int(v)
                if iv is None:
                    continue
                vals.append(iv)
            unique_vals = sorted(set(vals)) if vals else [tid]
            mapping[tid] = unique_vals
            for cid in unique_vals:
                rows.append({"track_id": tid, "cell_id": cid})
        return mapping, (pd.DataFrame(rows) if rows else empty_df)

    @staticmethod
    def _guess_area_column(df) -> Optional[str]:
        if df is None or len(df) == 0:
            return None
        preferred = [
            "area_um2",
            "area_µm2",
            "area_um^2",
            "Area (µm²)",
            "Area (um²)",
            "Area (um^2)",
            "area",
            "Area",
            "cell_area",
            "Cell_Area",
            "Area_um2",
            "size",
            "Size",
        ]
        cols = list(df.columns)
        for c in preferred:
            if c in cols:
                return c
        for c in cols:
            s = str(c).lower()
            if "area" in s:
                return c
        return None

    def _compute_track_metrics_table(
        self,
        tracks_df,
        pixel_size_um: float,
        time_per_frame_min: float,
    ):
        from bactomate.gui.tracking_filters import compute_track_metrics

        return compute_track_metrics(tracks_df, pixel_size_um, time_per_frame_min)

    def _build_track_information_table(self, tracks_df, metrics_df):
        import pandas as pd

        cols = [
            "DisplayID",
            "ID_Source",
            "Cell_ID",
            "TrackID",
            "trajectory_length",
            "duration_frames",
            "gap_frames",
            "duration_min",
            "first_frame",
            "last_frame",
            "displacement_px",
            "displacement_um",
            "path_length_px",
            "path_length_um",
            "mean_speed_px_frame",
            "mean_speed_um_min",
            "mean_speed_um_s",
            "straight_line_speed_px_frame",
            "straight_line_speed_um_min",
            "straight_line_speed_um_s",
            "straightness",
        ]
        if metrics_df is None or len(metrics_df) == 0:
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame()
        cell_source = None
        if "Cell_ID_Source" in tracks_df.columns:
            cell_source = tracks_df.groupby("track_id")["Cell_ID_Source"].first()
        if "Cell_ID" in tracks_df.columns:
            first_cell = tracks_df.groupby("track_id")["Cell_ID"].first()
            out["Cell_ID"] = metrics_df["track_id"].map(first_cell).fillna(metrics_df["track_id"]).astype(int)
        else:
            out["Cell_ID"] = metrics_df["track_id"].astype(int)
        out["TrackID"] = metrics_df["track_id"].astype(int)
        if cell_source is not None:
            out["ID_Source"] = (
                metrics_df["track_id"].map(cell_source).fillna("track").astype(str)
            )
        else:
            out["ID_Source"] = "track"
        out["DisplayID"] = [
            compact_tracking_label(
                int(tid),
                int(cid),
                cell_id_source=str(src),
                region_label=int(tid),
            )
            for tid, cid, src in zip(
                out["TrackID"].tolist(),
                out["Cell_ID"].tolist(),
                out["ID_Source"].tolist(),
            )
        ]
        out["trajectory_length"] = metrics_df["trajectory_length"].astype(int)
        out["duration_frames"] = metrics_df["duration_frames"].astype(int)
        out["gap_frames"] = metrics_df["gap_frames"].astype(int)
        out["duration_min"] = np.round(metrics_df["duration_min"], 4)
        out["first_frame"] = metrics_df["first_frame"].astype(int)
        out["last_frame"] = metrics_df["last_frame"].astype(int)
        # Report filters must compare calculation precision, as preview filters do.
        out["displacement_px"] = metrics_df["displacement_px"]
        out["displacement_um"] = metrics_df["displacement_um"]
        out["path_length_px"] = metrics_df["path_length_px"]
        out["path_length_um"] = metrics_df["path_length_um"]
        out["mean_speed_px_frame"] = metrics_df["mean_speed_px_frame"]
        out["mean_speed_um_s"] = metrics_df["mean_speed_um_s"]
        out["mean_speed_um_min"] = np.round(metrics_df["mean_speed_um_s"] * 60.0, 6)
        out["straight_line_speed_px_frame"] = np.round(metrics_df["straight_line_speed_px_frame"], 6)
        out["straight_line_speed_um_s"] = np.round(metrics_df["straight_line_speed_um_s"], 6)
        out["straight_line_speed_um_min"] = np.round(metrics_df["straight_line_speed_um_s"] * 60.0, 6)
        out["straightness"] = metrics_df["straightness"]
        return out[cols]

    def _track_area_table(self, tracks_df, masks_tracked, merged_data=None):
        import pandas as pd

        cols = ["track_id", "frame", "area", "area_source"]
        if tracks_df is None or len(tracks_df) == 0 or "track_id" not in tracks_df.columns or "frame" not in tracks_df.columns:
            return pd.DataFrame(columns=cols)
        point_cols = ["track_id", "frame"]
        for coord_col in ("y", "x"):
            if coord_col in tracks_df.columns:
                point_cols.append(coord_col)
        if "Cell_ID" in tracks_df.columns:
            point_cols.append("Cell_ID")
        points = tracks_df[point_cols].copy()
        points["track_id"] = pd.to_numeric(points["track_id"], errors="coerce")
        points["frame"] = pd.to_numeric(points["frame"], errors="coerce")
        for coord_col in ("y", "x"):
            if coord_col in points.columns:
                points[coord_col] = pd.to_numeric(points[coord_col], errors="coerce")
        if "Cell_ID" in points.columns:
            points["Cell_ID"] = pd.to_numeric(points["Cell_ID"], errors="coerce")
        points = points.dropna(subset=["track_id", "frame"]).astype({"track_id": int, "frame": int}).drop_duplicates()
        if len(points) == 0:
            return pd.DataFrame(columns=cols)

        area_map: dict[tuple[int, int], tuple[float, str]] = {}
        if (
            merged_data is not None
            and len(merged_data) > 0
            and "track_id" in merged_data.columns
            and "frame" in merged_data.columns
        ):
            area_col = self._guess_area_column(merged_data)
            if area_col is not None:
                m = merged_data[["track_id", "frame", area_col]].copy()
                m["track_id"] = pd.to_numeric(m["track_id"], errors="coerce")
                m["frame"] = pd.to_numeric(m["frame"], errors="coerce")
                m[area_col] = pd.to_numeric(m[area_col], errors="coerce")
                m = m.dropna()
                if len(m) > 0:
                    m = (
                        m.groupby(["track_id", "frame"], as_index=False)[area_col]
                        .median()
                        .astype({"track_id": int, "frame": int})
                    )
                    for _, r in m.iterrows():
                        v = float(r[area_col])
                        if v > 0:
                            area_map[(int(r["track_id"]), int(r["frame"]))] = (v, f"analysis:{area_col}")

        frame_counts: dict[int, dict[int, int]] = {}
        for fr in sorted(points["frame"].unique().tolist()):
            if fr < 0 or fr >= int(masks_tracked.shape[0]):
                frame_counts[fr] = {}
                continue
            vals, cnts = np.unique(masks_tracked[fr], return_counts=True)
            counts = {}
            for v, c in zip(vals, cnts):
                iv = int(v)
                if iv > 0:
                    counts[iv] = int(c)
            frame_counts[fr] = counts

        rows = []
        for r in points.itertuples(index=False):
            tid = int(r.track_id)
            fr = int(r.frame)
            area, source = area_map.get((tid, fr), (None, None))
            if area is None:
                label_candidates: list[int] = []
                try:
                    y = getattr(r, "y", None)
                    x = getattr(r, "x", None)
                    if y is not None and x is not None and fr in frame_counts and 0 <= fr < int(masks_tracked.shape[0]):
                        yy = int(round(float(y)))
                        xx = int(round(float(x)))
                        if 0 <= yy < int(masks_tracked.shape[1]) and 0 <= xx < int(masks_tracked.shape[2]):
                            label_at_centroid = int(masks_tracked[fr, yy, xx])
                            if label_at_centroid > 0:
                                label_candidates.append(label_at_centroid)
                except Exception:
                    pass
                if "Cell_ID" in points.columns:
                    try:
                        cell_id = getattr(r, "Cell_ID")
                        if cell_id is not None and not pd.isna(cell_id):
                            label_candidates.append(int(cell_id))
                    except Exception:
                        pass
                label_candidates.append(tid)
                counts = frame_counts.get(fr, {})
                area = 0.0
                for label_id in label_candidates:
                    area = float(counts.get(int(label_id), 0))
                    if area > 0:
                        break
                source = "mask_px"
            rows.append({"track_id": tid, "frame": fr, "area": float(area), "area_source": source})
        return pd.DataFrame(rows, columns=cols)

    def _build_trajectory_tables(self, tracks_df, masks_tracked, merged_data=None) -> dict[str, Any]:
        import pandas as pd

        pixel_size = float(getattr(self, "_cal_ps", 0.0) or 0.0)
        time_per_frame = float(getattr(self, "_cal_tp", 0.0) or 0.0)
        metrics_df = self._compute_track_metrics_table(tracks_df, pixel_size, time_per_frame)
        track_info_df = self._build_track_information_table(tracks_df, metrics_df)
        track_cell_map, track_cells_df = self._build_track_cell_table(tracks_df)
        area_df = self._track_area_table(tracks_df, masks_tracked, merged_data=merged_data)
        summary_cols = [
            "track_id",
            "containing_cell_ids",
            "start_frame",
            "start_coord",
            "start_y",
            "start_x",
            "end_frame",
            "end_coord",
            "end_y",
            "end_x",
            "total_traveled",
            "total_traveled_unit",
            "total_traveled_px",
            "total_traveled_um",
            "displacement_px",
            "displacement_um",
            "duration_frames",
            "gap_frames",
            "duration_min",
            "mean_speed_px_frame",
            "mean_speed_um_s",
            "straightness",
        ]

        summary_rows = []
        tracks_sorted = tracks_df.sort_values(["track_id", "frame"]) if len(tracks_df) > 0 and "frame" in tracks_df.columns else tracks_df
        metrics_lookup = (
            metrics_df.set_index("track_id").to_dict(orient="index")
            if len(metrics_df) > 0 and "track_id" in metrics_df.columns
            else {}
        )
        for tid_raw, g in tracks_sorted.groupby("track_id") if len(tracks_sorted) > 0 and "track_id" in tracks_sorted.columns else []:
            tid = int(tid_raw)
            m = metrics_lookup.get(tid, {})
            start_row = g.iloc[0]
            end_row = g.iloc[-1]
            start_frame = self._safe_int(start_row.get("frame"))
            end_frame = self._safe_int(end_row.get("frame"))
            start_y = self._safe_float(start_row.get("y")) if "y" in g.columns else None
            start_x = self._safe_float(start_row.get("x")) if "x" in g.columns else None
            end_y = self._safe_float(end_row.get("y")) if "y" in g.columns else None
            end_x = self._safe_float(end_row.get("x")) if "x" in g.columns else None
            start_coord = self._json_list([int(round(start_y)), int(round(start_x))]) if start_y is not None and start_x is not None else ""
            end_coord = self._json_list([int(round(end_y)), int(round(end_x))]) if end_y is not None and end_x is not None else ""
            containing_cells = track_cell_map.get(tid, [tid])
            total_traveled_px = float(m.get("path_length_px", 0.0))
            total_traveled_um = float(m.get("path_length_um", total_traveled_px))
            if pixel_size > 0:
                total_traveled = total_traveled_um
                total_traveled_unit = "um"
            else:
                total_traveled = total_traveled_px
                total_traveled_unit = "px"
            summary_rows.append(
                {
                    "track_id": tid,
                    "containing_cell_ids": self._json_list(containing_cells),
                    "start_frame": start_frame,
                    "start_coord": start_coord,
                    "start_y": start_y,
                    "start_x": start_x,
                    "end_frame": end_frame,
                    "end_coord": end_coord,
                    "end_y": end_y,
                    "end_x": end_x,
                    "total_traveled": total_traveled,
                    "total_traveled_unit": total_traveled_unit,
                    "total_traveled_px": total_traveled_px,
                    "total_traveled_um": total_traveled_um,
                    "displacement_px": float(m.get("displacement_px", 0.0)),
                    "displacement_um": float(m.get("displacement_um", 0.0)),
                    "duration_frames": int(m.get("duration_frames", 0) or 0),
                    "gap_frames": int(m.get("gap_frames", 0) or 0),
                    "duration_min": float(m.get("duration_min", 0.0)),
                    "mean_speed_px_frame": float(m.get("mean_speed_px_frame", 0.0)),
                    "mean_speed_um_s": float(m.get("mean_speed_um_s", 0.0)),
                    "straightness": float(m.get("straightness", 0.0)),
                }
            )
        track_summary_df = pd.DataFrame(summary_rows, columns=summary_cols)

        track_points_df = tracks_df.copy()
        if "Cell_ID" in track_points_df.columns:
            track_points_df["cell_id"] = track_points_df["Cell_ID"]
        elif "track_id" in track_points_df.columns:
            track_points_df["cell_id"] = track_points_df["track_id"]
        if "Cell_ID_Source" in track_points_df.columns:
            track_points_df["cell_id_source"] = track_points_df["Cell_ID_Source"].astype(str)
        if "track_id" in track_points_df.columns and "frame" in track_points_df.columns:
            track_points_df = track_points_df.sort_values(["track_id", "frame"])

        return {
            "track_summary": track_summary_df,
            "track_information": track_info_df,
            "track_points": track_points_df,
            "track_cells": track_cells_df,
            "track_area": area_df,
        }

    def _build_integrated_lineage_tables(
        self,
        tracks_df,
        edges_df,
        masks_tracked,
        merged_data=None,
        *,
        output_dir: Path,
        stem: str,
    ) -> dict[str, Any]:
        """Core lineage tables plus analysis joins, QC, events, and lineage report workbook."""
        from bactomate.modules.lineage_integration import (
            LINEAGE_SCHEMA_VERSION,
            canonical_lineage_tables,
            enrich_lineage_tables,
            lineage_run_info_rows,
            load_per_cell_table,
            preflight_lineage_run,
            write_lineage_report_workbook,
        )

        base = self._build_lineage_tables(
            tracks_df=tracks_df,
            edges_df=edges_df,
            masks_tracked=masks_tracked,
            merged_data=merged_data,
        )
        area_df = self._track_area_table(tracks_df, masks_tracked, merged_data=merged_data)
        per_cell_df = None
        excel_path = Path(self.excel_file) if self.excel_file else None
        if excel_path is not None and excel_path.is_file():
            try:
                per_cell_df = load_per_cell_table(excel_path)
            except Exception:
                get_logger().debug("Could not load Per-Cell Data for lineage integration", exc_info=True)
        full = enrich_lineage_tables(
            base,
            tracks_df=tracks_df,
            edges_df=edges_df,
            area_df=area_df,
            per_cell_df=per_cell_df,
            time_per_frame_min=float(getattr(self, "_cal_tp", 0.0) or 0.0),
            pixel_size_um=float(getattr(self, "_cal_ps", 0.0) or 0.0),
        )
        review_map = getattr(self, "_lineage_curation_review_map", None) or {}
        if review_map:
            from bactomate.modules.lineage_integration import apply_curation_review_map_to_events

            events = full.get("lineage_events")
            if events is not None:
                full["lineage_events"] = apply_curation_review_map_to_events(events, review_map)
        try:
            pf = preflight_lineage_run(
                Path(self.mask_file),
                linker_mode=str(self.mode or ""),
                model_name=str(self.model_name or ""),
            )
            run_info = lineage_run_info_rows(
                mask_path=Path(self.mask_file),
                analysis_workbook=excel_path,
                linker_mode=str(self.mode or ""),
                model_name=str(self.model_name or ""),
                preflight=pf,
            )
            report_path = Path(output_dir) / f"{stem}_lineage_report.xlsx"
            write_lineage_report_workbook(report_path, full, run_info_rows=run_info)
            get_logger().info("Wrote lineage report: %s", report_path)
            # macOS can overflow the QThread stack while Matplotlib renders lineage
            # trees from this worker thread. Write the workbook here and let the
            # dedicated GUI export action render PNG figures on demand.
            get_logger().info(
                "Skipped automatic lineage figure export in TrackingWorker; use the lineage export action to write PNG figures."
            )
        except Exception:
            get_logger().debug("Lineage report workbook write failed", exc_info=True)
        return canonical_lineage_tables(full)

    def _build_lineage_tables(
        self,
        tracks_df,
        edges_df,
        masks_tracked,
        merged_data=None,
    ) -> dict[str, Any]:
        import pandas as pd

        if tracks_df is None:
            tracks_df = pd.DataFrame(columns=["track_id", "frame", "y", "x"])
        if edges_df is None:
            edges_df = pd.DataFrame(
                columns=["parent_track_id", "daughter_track_id", "division_frame", "parent_last_frame"]
            )

        founder_by_track = self._lineage_founder_map(edges_df, tracks_df)
        if not founder_by_track and "track_id" in tracks_df.columns:
            founder_by_track = {int(t): int(t) for t in tracks_df["track_id"].dropna().unique().tolist()}

        track_cell_map, _ = self._build_track_cell_table(tracks_df)
        lineage_tracks_rows = [
            {"lineage_id": int(fid), "track_id": int(tid)}
            for tid, fid in sorted(founder_by_track.items(), key=lambda x: (int(x[1]), int(x[0])))
        ]
        lineage_tracks_df = pd.DataFrame(lineage_tracks_rows, columns=["lineage_id", "track_id"])

        lineage_cells_rows = []
        for lid in sorted(set(founder_by_track.values())):
            tracks_in_lineage = [int(tid) for tid, lf in founder_by_track.items() if int(lf) == int(lid)]
            cells: set[int] = set()
            for tid in tracks_in_lineage:
                cells.update(track_cell_map.get(tid, [tid]))
            for cid in sorted(cells):
                lineage_cells_rows.append({"lineage_id": int(lid), "cell_id": int(cid)})
        lineage_cells_df = pd.DataFrame(lineage_cells_rows, columns=["lineage_id", "cell_id"])

        edge_lineage_count: dict[int, int] = defaultdict(int)
        if len(edges_df) > 0 and "parent_track_id" in edges_df.columns and "daughter_track_id" in edges_df.columns:
            for er in edges_df.itertuples(index=False):
                parent = self._safe_int(getattr(er, "parent_track_id", None))
                daughter = self._safe_int(getattr(er, "daughter_track_id", None))
                lid = None
                if parent is not None:
                    lid = founder_by_track.get(parent)
                if lid is None and daughter is not None:
                    lid = founder_by_track.get(daughter)
                if lid is not None:
                    edge_lineage_count[int(lid)] += 1

        lineage_summary_cols = [
            "lineage_id",
            "containing_cells",
            "containing_tracks",
            "start_frame",
            "start_coord",
            "start_y",
            "start_x",
            "end_frame",
            "end_coord",
            "end_y",
            "end_x",
            "total_divisions",
            "n_tracks",
            "n_cells",
        ]
        lineage_summary_rows = []
        if len(tracks_df) > 0 and "track_id" in tracks_df.columns and "frame" in tracks_df.columns:
            for lid in sorted(set(founder_by_track.values())):
                tracks_in_lineage = [int(tid) for tid, lf in founder_by_track.items() if int(lf) == int(lid)]
                sub = tracks_df[tracks_df["track_id"].isin(tracks_in_lineage)].copy()
                if len(sub) == 0:
                    continue
                sub = sub.sort_values(["frame", "track_id"])
                first = sub.iloc[0]
                last = sub.iloc[-1]
                start_frame = self._safe_int(first.get("frame"))
                end_frame = self._safe_int(last.get("frame"))
                start_y = self._safe_float(first.get("y")) if "y" in sub.columns else None
                start_x = self._safe_float(first.get("x")) if "x" in sub.columns else None
                end_y = self._safe_float(last.get("y")) if "y" in sub.columns else None
                end_x = self._safe_float(last.get("x")) if "x" in sub.columns else None
                start_coord = self._json_list([int(round(start_y)), int(round(start_x))]) if start_y is not None and start_x is not None else ""
                end_coord = self._json_list([int(round(end_y)), int(round(end_x))]) if end_y is not None and end_x is not None else ""
                cells = sorted(
                    set(
                        lineage_cells_df.loc[lineage_cells_df["lineage_id"] == int(lid), "cell_id"].astype(int).tolist()
                    )
                )
                lineage_summary_rows.append(
                    {
                        "lineage_id": int(lid),
                        "containing_cells": self._json_list(cells),
                        "containing_tracks": self._json_list(sorted(tracks_in_lineage)),
                        "start_frame": start_frame,
                        "start_coord": start_coord,
                        "start_y": start_y,
                        "start_x": start_x,
                        "end_frame": end_frame,
                        "end_coord": end_coord,
                        "end_y": end_y,
                        "end_x": end_x,
                        "total_divisions": int(edge_lineage_count.get(int(lid), 0)),
                        "n_tracks": int(len(tracks_in_lineage)),
                        "n_cells": int(len(cells)),
                    }
                )
        lineage_summary_df = pd.DataFrame(lineage_summary_rows, columns=lineage_summary_cols)

        area_df = self._track_area_table(tracks_df, masks_tracked, merged_data=merged_data)
        lineage_growth_cols = [
            "lineage_id",
            "growth",
            "doubling_time",
            "growth_unit",
            "metric_used",
            "fit_r2",
            "n_points",
        ]
        lineage_growth_rows = []
        if len(area_df) > 0 and "track_id" in area_df.columns and "frame" in area_df.columns:
            area_df = area_df.copy()
            area_df["lineage_id"] = area_df["track_id"].map(
                lambda t: founder_by_track.get(int(t), int(t)) if self._safe_int(t) is not None else None
            )
            time_per_frame = float(getattr(self, "_cal_tp", 0.0) or 0.0)
            for lid, g in area_df.groupby("lineage_id"):
                if self._safe_int(lid) is None:
                    continue
                gsum = g.groupby("frame", as_index=False)["area"].sum().sort_values("frame")
                positive = gsum[gsum["area"] > 0].copy()
                sources = sorted(set(g["area_source"].astype(str).tolist()))
                source_label = "mask_px"
                analysis_sources = [s for s in sources if s.startswith("analysis:")]
                if analysis_sources and len(sources) > 1:
                    source_label = f"mixed({','.join(analysis_sources + ['mask_px'])})"
                elif analysis_sources:
                    source_label = analysis_sources[0]
                n_points = int(len(positive))
                growth = np.nan
                doubling_time = np.nan
                r2 = np.nan
                if n_points >= 2:
                    x = positive["frame"].to_numpy(dtype=float)
                    if time_per_frame > 0:
                        x = x * time_per_frame
                    y = np.log(positive["area"].to_numpy(dtype=float))
                    try:
                        slope, intercept = np.polyfit(x, y, 1)
                        y_pred = slope * x + intercept
                        ss_res = float(np.sum((y - y_pred) ** 2))
                        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                        growth = float(slope)
                        if growth > 0:
                            doubling_time = float(np.log(2.0) / growth)
                    except Exception:
                        pass
                lineage_growth_rows.append(
                    {
                        "lineage_id": int(lid),
                        "growth": growth,
                        "doubling_time": doubling_time,
                        "growth_unit": "1/min" if time_per_frame > 0 else "1/frame",
                        "metric_used": source_label,
                        "fit_r2": r2,
                        "n_points": n_points,
                    }
                )
        lineage_growth_df = pd.DataFrame(lineage_growth_rows, columns=lineage_growth_cols)

        lineage_points_df = tracks_df.copy()
        if "track_id" in lineage_points_df.columns:
            lineage_points_df["lineage_id"] = lineage_points_df["track_id"].map(
                lambda t: founder_by_track.get(int(t), int(t))
                if self._safe_int(t) is not None
                else None
            )
            if "frame" in lineage_points_df.columns:
                lineage_points_df = lineage_points_df.sort_values(["lineage_id", "track_id", "frame"])

        return {
            "lineage_summary": lineage_summary_df,
            "lineage_growth": lineage_growth_df,
            "lineage_edges": edges_df.copy(),
            "lineage_tracks": lineage_tracks_df,
            "lineage_cells": lineage_cells_df,
            "lineage_points": lineage_points_df,
        }

    def _write_trajectory_workbook(
        self,
        tracks_df,
        masks_tracked,
        output_path: Path,
        merged_data=None,
    ) -> None:
        import pandas as pd

        tables = self._build_trajectory_tables(
            tracks_df=tracks_df,
            masks_tracked=masks_tracked,
            merged_data=merged_data,
        )
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for name in ("track_summary", "track_information", "track_points", "track_cells", "track_area"):
                tables[name].to_excel(writer, sheet_name=name, index=False)

    def _write_lineage_workbook(
        self,
        tracks_df,
        edges_df,
        masks_tracked,
        output_path: Path,
        merged_data=None,
    ) -> None:
        import pandas as pd

        tables = self._build_lineage_tables(
            tracks_df=tracks_df,
            edges_df=edges_df,
            masks_tracked=masks_tracked,
            merged_data=merged_data,
        )
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for name in (
                "lineage_summary",
                "lineage_growth",
                "lineage_edges",
                "lineage_tracks",
                "lineage_cells",
                "lineage_points",
            ):
                tables[name].to_excel(writer, sheet_name=name, index=False)

    @staticmethod
    def _autofit_excel_sheet(writer, sheet_name: str, df, *, max_width: float = 72.0) -> None:
        """Set column widths from header + cell content (openpyxl has no true Excel autofit)."""
        try:
            ws = writer.sheets[sheet_name]
        except KeyError:
            return
        from openpyxl.utils import get_column_letter

        if df is None or not hasattr(df, "columns"):
            return
        for col_idx, column in enumerate(df.columns, start=1):
            letter = get_column_letter(col_idx)
            best = len(str(column))
            try:
                series = df[column]
                if len(series) > 0:
                    sample = series.astype(str).head(2500)
                    best = max(best, int(sample.str.len().max()))
            except Exception:
                pass
            ws.column_dimensions[letter].width = min(max(best + 2, 8), max_width)

    def _build_run_info_sheet(self, run_metadata: dict) -> Any:
        import pandas as pd

        from bactomate.gui.tracking_filters import filter_params_run_info_rows

        rows: list[tuple[str, str]] = []
        for key in (
            "generated_utc",
            "mask_file",
            "original_file",
            "analysis_excel",
            "trackastra_model",
            "linking_mode",
            "device",
            "track_action",
            "frame_limit",
        ):
            val = run_metadata.get(key)
            if val is not None and str(val) != "":
                rows.append((key, str(val)))
        rows.append(("track_speed_time_basis", "elapsed_frame_intervals"))
        rows.append(("pixel_size_um_worker", str(float(getattr(self, "_cal_ps", 0.0) or 0.0))))
        rows.append(("time_per_frame_min_worker", str(float(getattr(self, "_cal_tp", 0.0) or 0.0))))
        fp = run_metadata.get("filter_params") or {}
        if fp:
            rows.append(("", ""))
            rows.append(("filter_settings", ""))
            rows.extend(filter_params_run_info_rows(fp))
        return pd.DataFrame(rows, columns=["parameter", "value"])

    def _tracks_with_pass_column(
        self,
        track_info_df: Any,
        filter_params: dict,
    ) -> tuple[Any, Any]:
        import pandas as pd

        from bactomate.gui.tracking_filters import apply_track_filters

        raw_df = track_info_df.copy()
        if raw_df.empty or "TrackID" not in raw_df.columns:
            raw_df["Pass"] = []
            return raw_df, raw_df.iloc[0:0].copy()
        pixel_size = float(getattr(self, "_cal_ps", 0.0) or 0.0)
        time_per_frame = float(getattr(self, "_cal_tp", 0.0) or 0.0)
        if not filter_params:
            filter_params = {
                "pixel_size": pixel_size,
                "time_per_frame": time_per_frame,
                "use_physical_units": pixel_size > 0,
                "use_speed_physical": pixel_size > 0 and time_per_frame > 0,
            }
        metrics_cols = [
            "track_id",
            "trajectory_length",
            "duration_frames",
            "gap_frames",
            "displacement_px",
            "displacement_um",
            "path_length_px",
            "path_length_um",
            "mean_speed_px_frame",
            "mean_speed_um_s",
            "straightness",
        ]
        metrics_df = raw_df.rename(columns={"TrackID": "track_id"})
        for col in metrics_cols:
            if col not in metrics_df.columns and col in raw_df.columns:
                metrics_df[col] = raw_df[col]
        passing = apply_track_filters(metrics_df, filter_params)
        raw_df["Pass"] = raw_df["TrackID"].map(lambda t: "Yes" if int(t) in passing else "No")
        filtered_df = raw_df[raw_df["Pass"] == "Yes"].copy()
        return raw_df, filtered_df

    def _write_report_workbook(
        self,
        output_path: Path,
        trajectory_tables: dict[str, Any],
        lineage_tables: Optional[dict[str, Any]] = None,
        *,
        run_metadata: Optional[dict] = None,
        filter_params: Optional[dict] = None,
    ) -> None:
        import pandas as pd

        meta = run_metadata or {}
        fp = filter_params if filter_params is not None else meta.get("filter_params") or {}
        run_info_df = self._build_run_info_sheet(meta)
        track_info = trajectory_tables["track_information"]
        raw_df, filtered_df = self._tracks_with_pass_column(track_info, fp)

        # Excel stays lean: Run info + all tracks (raw/Pass) + filtered only.
        # track_summary lives in tables/ (parquet) for pipelines; it duplicated metrics here.
        sheets: list[tuple[str, Any]] = [
            ("Run info", run_info_df),
            ("raw", raw_df),
            ("filtered", filtered_df),
        ]
        if lineage_tables is not None:
            sheets.append(("lineage_summary", lineage_tables["lineage_summary"]))
            sheets.append(("lineage_growth", lineage_tables["lineage_growth"]))

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name, df in sheets:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                self._autofit_excel_sheet(writer, sheet_name, df)

    def _write_canonical_tables(
        self,
        output_dir: Path,
        trajectory_tables: dict[str, Any],
        lineage_tables: Optional[dict[str, Any]] = None,
    ) -> tuple[Path, dict[str, str]]:
        import json

        tables_dir = output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}

        all_tables: list[tuple[str, Any]] = list(trajectory_tables.items())
        if lineage_tables is not None:
            all_tables.extend(list(lineage_tables.items()))

        for key, df in all_tables:
            if df is None:
                continue
            if hasattr(df, "empty") and df.empty:
                # Write empty datasets too so schema is explicit downstream.
                pass
            base = tables_dir / key
            try:
                out_path = base.with_suffix(".parquet")
                df.to_parquet(out_path, index=False)
            except Exception:
                out_path = base.with_suffix(".csv.gz")
                df.to_csv(out_path, index=False, compression="gzip")
            written[key] = out_path.name

        schema_version = 1
        if lineage_tables is not None:
            try:
                from bactomate.modules.lineage_integration import LINEAGE_SCHEMA_VERSION

                schema_version = int(LINEAGE_SCHEMA_VERSION)
            except Exception:
                schema_version = 2
        meta = {
            "schema_version": schema_version,
            "output_profile": self.output_profile,
            "track_speed_time_basis": "elapsed_frame_intervals",
            "pixel_size_um": float(getattr(self, "_cal_ps", 0.0) or 0.0),
            "time_per_frame_min": float(getattr(self, "_cal_tp", 0.0) or 0.0),
            "tables": written,
        }
        meta_path = tables_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=True)
        written["metadata"] = meta_path.name
        return tables_dir, written
    
    def _create_trajectory_visualization(
        self,
        images,
        masks_tracked,
        tracks_df,
        output_path,
        progress_callback=None,
        overlay_frames=None,
        *,
        preview_fast: bool = False,
    ):
        """Create visualization showing original images with tracked cell IDs and trajectories.
        If overlay_frames is provided (T, H, W, 3), use it as base and only draw trajectories — much faster.
        ``preview_fast`` skips expensive text labels and uses coarser trajectory lines."""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib import cm
        from scipy.ndimage import label, binary_dilation
        import tifffile
        
        if progress_callback is None and hasattr(self, '_progress_callback'):
            progress_callback = self._progress_callback
        
        debug_output = []
        debug_output.append("Creating trajectory visualization...")
        use_overlay_base = overlay_frames is not None and overlay_frames.ndim == 4
        
        if use_overlay_base:
            num_frames = overlay_frames.shape[0]
            if images is not None and getattr(images, "ndim", 0) >= 3 and images.shape[0] == num_frames:
                native_height, native_width = images.shape[1], images.shape[2]
            else:
                native_height, native_width = overlay_frames.shape[1], overlay_frames.shape[2]
            debug_output.append("  - Using overlay as base (trajectories only)")
        else:
            num_frames = images.shape[0]
            native_height, native_width = images.shape[1], images.shape[2]
            if progress_callback:
                progress_callback(91, "Normalizing images...")
            img_min = images.min(axis=(1, 2), keepdims=True)
            img_max = images.max(axis=(1, 2), keepdims=True)
            img_range = img_max - img_min
            img_range[img_range == 0] = 1
            images_norm = ((images - img_min) / img_range * 255).astype(np.uint8)

        export_factor = 1
        if output_path is not None and not preview_fast:
            export_factor = tracking_export_upscale_factor(native_height, native_width)
        height = native_height * export_factor
        width = native_width * export_factor
        if export_factor > 1:
            debug_output.append(
                f"  - Rendering export at {width}×{height} ({export_factor}× native)"
            )
        
        # Get unique track IDs and colormap
        if 'track_id' in tracks_df.columns:
            unique_tracks = tracks_df['track_id'].unique()
            num_tracks = len(unique_tracks)
            debug_output.append(f"  - Found {num_tracks} unique tracks")
        else:
            unique_tracks = np.unique(masks_tracked)
            unique_tracks = unique_tracks[unique_tracks > 0]
            num_tracks = len(unique_tracks)
            debug_output.append(f"  - Found {num_tracks} unique track IDs in masks")
        
        cmap = plt.get_cmap('gist_ncar')
        track_colors = {}
        for i, track_id in enumerate(unique_tracks):
            color = cmap(i / max(num_tracks, 1))
            track_colors[int(track_id)] = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        
        visualization = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        track_points_by_id = build_track_points_by_id(tracks_df)
        if export_factor > 1:
            track_points_by_id = scale_track_points_by_id(track_points_by_id, export_factor)
        unique_track_set = {int(t) for t in unique_tracks}
        export_style = tracking_export_viz_style(height, width)
        use_aa_paths = bool(output_path is not None and not preview_fast and export_factor > 1)
        draw_track_labels = (
            not preview_fast
            and should_draw_track_id_labels(num_tracks, height, width)
            and not use_overlay_base
        )
        draw_centroid_markers = not preview_fast and not use_overlay_base

        if use_overlay_base:
            if (
                export_factor > 1
                and overlay_frames.shape[1] == native_height * export_factor
                and overlay_frames.shape[2] == native_width * export_factor
            ):
                base_frames = overlay_frames
            elif export_factor > 1:
                base_frames = upscale_rgb_stack(overlay_frames, export_factor)
            else:
                base_frames = overlay_frames
        elif export_factor > 1:
            gray_rgb = np.stack([images_norm, images_norm, images_norm], axis=-1)
            base_frames = upscale_rgb_stack(gray_rgb, export_factor)
            masks_tracked = upscale_mask_stack(masks_tracked, export_factor)
        else:
            base_frames = None
        
        for frame_idx in range(num_frames):
            try:
                if progress_callback:
                    progress_pct = int(90 + (frame_idx / num_frames) * 10)
                    progress_callback(progress_pct, f"Creating visualization: {frame_idx + 1}/{num_frames} ({progress_pct}%)")
                
                if use_overlay_base or export_factor > 1:
                    vis_frame = np.ascontiguousarray(base_frames[frame_idx].copy())
                else:
                    img = images_norm[frame_idx]
                    vis_frame = np.stack([img, img, img], axis=-1)
                    mask_frame = masks_tracked[frame_idx]
                    for track_id in unique_tracks:
                        track_id_int = int(track_id)
                        if track_id_int in track_colors:
                            color = track_colors[track_id_int]
                            mask_region = (mask_frame == track_id_int)
                            vis_frame[mask_region, 0] = np.clip(vis_frame[mask_region, 0] * 0.45 + color[0] * 0.55, 0, 255)
                            vis_frame[mask_region, 1] = np.clip(vis_frame[mask_region, 1] * 0.45 + color[1] * 0.55, 0, 255)
                            vis_frame[mask_region, 2] = np.clip(vis_frame[mask_region, 2] * 0.45 + color[2] * 0.55, 0, 255)
                
                labels_to_draw = []
                if track_points_by_id:
                    vis_frame = draw_trajectory_paths_on_frame(
                        vis_frame,
                        track_points_by_id=track_points_by_id,
                        track_colors=track_colors,
                        unique_track_set=unique_track_set,
                        frame_idx=frame_idx,
                        height=height,
                        width=width,
                        export_style=export_style,
                        preview_fast=preview_fast,
                        use_antialiased_paths=use_aa_paths,
                    )
                    
                    if draw_centroid_markers:
                        tracks_in_frame = [
                            track_id_int
                            for track_id_int, points in track_points_by_id.items()
                            if track_id_int in unique_track_set and any(f == frame_idx for f, _, _ in points)
                        ]
                        for track_id_int in tracks_in_frame:
                            if track_id_int not in track_colors:
                                continue
                            color = track_colors[track_id_int]
                            current_points = [
                                (y, x)
                                for f, y, x in track_points_by_id[track_id_int]
                                if f == frame_idx
                            ]
                            if current_points:
                                y = int(current_points[0][0])
                                x = int(current_points[0][1])
                                if 0 <= y < height and 0 <= x < width:
                                    draw_track_centroid_marker(vis_frame, y, x, color, export_style)
                                    if draw_track_labels:
                                        cell_id = None
                                        cell_id_source = "track"
                                        if hasattr(tracks_df, "columns") and "Cell_ID" in tracks_df.columns:
                                            sub = tracks_df[
                                                (tracks_df["track_id"].astype(int) == int(track_id_int))
                                                & (tracks_df["frame"].astype(int) == int(frame_idx))
                                            ]
                                            if len(sub):
                                                try:
                                                    import pandas as pd
                                                    val = sub.iloc[0]["Cell_ID"]
                                                    if val is not None and not pd.isna(val):
                                                        cid = int(val)
                                                        cell_id = cid
                                                    if "Cell_ID_Source" in sub.columns:
                                                        src_val = sub.iloc[0]["Cell_ID_Source"]
                                                        if src_val is not None and not pd.isna(src_val):
                                                            cell_id_source = str(src_val)
                                                except Exception:
                                                    pass
                                        label_text = compact_tracking_label(
                                            int(track_id_int),
                                            cell_id,
                                            cell_id_source=cell_id_source,
                                            region_label=int(track_id_int),
                                        )
                                        labels_to_draw.append(
                                            (
                                                label_text,
                                                x + export_style.label_offset_x,
                                                y + export_style.label_offset_y,
                                            )
                                        )

                if labels_to_draw and draw_track_labels:
                    try:
                        from PIL import Image

                        pil_img = Image.fromarray(vis_frame)
                        pil_img = draw_hq_labels(
                            pil_img,
                            labels_to_draw,
                            font_size=export_style.font_size,
                            scale=export_style.label_scale,
                            stroke_width=export_style.stroke_width,
                        )
                        vis_frame = np.array(pil_img)
                    except Exception:
                        pass
                
                visualization[frame_idx] = vis_frame
            except Exception as e:
                debug_output.append(f"  Frame {frame_idx}: trajectory drawing failed: {e}")
                if use_overlay_base or export_factor > 1:
                    vis_frame = np.ascontiguousarray(base_frames[frame_idx].copy())
                else:
                    img = images_norm[frame_idx]
                    vis_frame = np.stack([img, img, img], axis=-1)
                visualization[frame_idx] = vis_frame
        
        if output_path is not None:
            upscale = export_factor
            if progress_callback:
                progress_callback(99, "Saving visualization file...")
            T = visualization.shape[0]
            meta = None
            try:
                from bactomate.modules.tiff_metadata import build_metadata_dict
                ps = getattr(self, '_cal_ps', 0) or 0
                tp = getattr(self, '_cal_tp', 0) or 0
                if upscale > 1 and ps > 0:
                    ps = float(ps) / float(upscale)
                meta = build_metadata_dict(
                    pixel_size_um=ps if ps > 0 else None,
                    time_per_frame_min=tp if tp > 0 else None,
                    axes="TYXC",
                    images=T,
                    frames=T,
                )
            except Exception:
                pass
            try:
                if meta:
                    tifffile.imwrite(str(output_path), visualization, metadata=meta, compression='zlib')
                else:
                    tifffile.imwrite(str(output_path), visualization, compression='zlib')
                debug_output.append(f"✓ Visualization saved with {num_tracks} tracks (compressed)")
            except Exception as e:
                debug_output.append(f"⚠ Compression failed, saving uncompressed: {str(e)}")
                tifffile.imwrite(str(output_path), visualization, metadata=meta if meta else {})
                debug_output.append(f"✓ Visualization saved with {num_tracks} tracks (uncompressed)")
        return visualization

    def _emit_preview_result(
        self,
        debug_output: list[str],
        imgs: np.ndarray,
        masks_input: np.ndarray,
        masks_tracked: np.ndarray,
        tracks_df: Any,
        *,
        napari_graph: Any = None,
    ) -> None:
        """Build trajectory or lineage overlay in memory and emit preview_ready (no disk writes)."""
        import pandas as pd
        from pathlib import Path as _Path

        mask_path = _Path(self.mask_file)
        original_path = _Path(self.original_file) if self.original_file else None
        from bactomate.modules.tiff_metadata import resolve_calibration_for_mask

        self._cal_ps, self._cal_tp, _, _ = resolve_calibration_for_mask(mask_path)
        is_lineage = bool(getattr(self, "preview_lineage", False))
        edges_df = None
        founder_by_track: dict[int, int] = {}
        lineage_tables = None
        if is_lineage and napari_graph is not None:
            from bactomate.modules.lineage_integration import (
                enrich_lineage_tables,
                lineage_edges_from_napari_graph,
                lineage_founder_map,
                load_per_cell_table,
            )

            edges_df = lineage_edges_from_napari_graph(napari_graph, tracks_df)
            founder_by_track = lineage_founder_map(edges_df, tracks_df)
            base = self._build_lineage_tables(
                tracks_df=tracks_df,
                edges_df=edges_df,
                masks_tracked=masks_tracked,
                merged_data=None,
            )
            per_cell_df = None
            if self.excel_file and Path(self.excel_file).is_file():
                try:
                    per_cell_df = load_per_cell_table(Path(self.excel_file))
                except Exception:
                    pass
            lineage_tables = enrich_lineage_tables(
                base,
                tracks_df=tracks_df,
                edges_df=edges_df,
                area_df=self._track_area_table(tracks_df, masks_tracked, merged_data=None),
                per_cell_df=per_cell_df,
                time_per_frame_min=float(getattr(self, "_cal_tp", 0.0) or 0.0),
                pixel_size_um=float(getattr(self, "_cal_ps", 0.0) or 0.0),
            )

        self.progress.emit(82, "Building preview overlay…")
        if is_lineage:
            debug_output.append(
                "\nLineage preview: founder colors and division markers (candidate links)."
            )
        else:
            debug_output.append(
                "\nPreview mode: in-memory trajectories only (no disk writes; labels show Trackastra track ID)."
            )

        # Tracking used full-resolution data; downsample only for the GUI overlay / compose path.
        imgs_disp, masks_in_disp, masks_tr_disp, scale_y, scale_x = prepare_preview_display_stacks(
            imgs,
            masks_input,
            masks_tracked,
        )
        tracks_disp = scale_tracks_df_for_display(tracks_df, scale_y, scale_x)
        if abs(scale_y - 1.0) > 1e-12 or abs(scale_x - 1.0) > 1e-12:
            debug_output.append(
                f"Preview display downsampled for GUI "
                f"({imgs.shape[1]}x{imgs.shape[2]} → {imgs_disp.shape[1]}x{imgs_disp.shape[2]}); "
                "Trackastra still used full resolution."
            )

        frames_rgb = None
        try:
            def vis_progress(pct: int, msg: str) -> None:
                self.progress.emit(min(99, int(pct)), str(msg))

            if is_lineage and founder_by_track:
                from bactomate.modules.lineage_integration import (
                    build_biology_track_labels,
                    compact_biology_label_map,
                )

                lineage_meta = self._lineage_per_track_summary(
                    edges_df if edges_df is not None else pd.DataFrame(),
                    tracks_df,
                )
                bio_labels_df = build_biology_track_labels(
                    edges_df if edges_df is not None else pd.DataFrame(),
                    tracks_df,
                    lineage_meta,
                )
                biology_label_by_track = compact_biology_label_map(bio_labels_df)
                frames_rgb = self._create_lineage_visualization(
                    imgs_disp,
                    masks_tr_disp,
                    tracks_disp,
                    edges_df if edges_df is not None else pd.DataFrame(),
                    founder_by_track,
                    output_path=None,
                    progress_callback=vis_progress,
                    overlay_frames=None,
                    masks_input=masks_in_disp,
                    biology_label_by_track=biology_label_by_track,
                )
            else:
                frames_rgb = self._create_trajectory_visualization(
                    imgs_disp,
                    masks_tr_disp,
                    tracks_disp,
                    output_path=None,
                    progress_callback=vis_progress,
                    overlay_frames=None,
                    preview_fast=True,
                )
        except Exception as e:
            debug_output.append(f"⚠ Preview visualization failed: {e}")
        if frames_rgb is None:
            self.finished.emit(
                False,
                "Preview tracking finished but visualization could not be built.\n\n"
                + "\n".join(debug_output),
            )
            return

        n_tracks = int(tracks_df["track_id"].nunique()) if (
            isinstance(tracks_df, pd.DataFrame)
            and not tracks_df.empty
            and "track_id" in tracks_df.columns
        ) else 0
        payload = TrackingPreviewResult(
            mask_path=mask_path,
            frames_rgb=np.asarray(frames_rgb, dtype=np.uint8),
            # Keep full-resolution coordinates for metrics / filters; display scale is applied in the GUI.
            tracks_df=tracks_df,
            original_path=original_path if original_path and original_path.is_file() else None,
            n_tracks=n_tracks,
            stack_images=np.asarray(imgs_disp),
            masks_input=np.asarray(masks_in_disp),
            masks_tracked=np.asarray(masks_tr_disp),
            edges_df=edges_df,
            founder_by_track=founder_by_track or None,
            lineage_tables=lineage_tables,
            is_lineage_preview=is_lineage,
            display_scale_y=float(scale_y),
            display_scale_x=float(scale_x),
        )
        self.preview_ready.emit(payload)
        self.progress.emit(100, "Preview ready")
        label = "Lineage preview" if is_lineage else "Preview tracking"
        success_msg = (
            f"{label} complete ({n_tracks} tracks, {frames_rgb.shape[0]} frames). "
            "Nothing was written to disk.\n\n"
        )
        success_msg += "\n".join(debug_output[-12:])
        self.finished.emit(True, success_msg)
    
    def _create_overlay_with_track_ids(self, images, masks_tracked, tracks_df, output_path=None, progress_callback=None):
        """Create overlay visualization with track / Cell ID labels matched from tracks_df.
        If progress_callback(frame_idx, total, msg) is given, it is called each frame and tqdm is disabled."""
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        import tifffile
        
        num_frames = images.shape[0]
        height, width = images.shape[1], images.shape[2]
        
        # Normalize images
        img_min = images.min(axis=(1, 2), keepdims=True)
        img_max = images.max(axis=(1, 2), keepdims=True)
        img_range = img_max - img_min
        img_range[img_range == 0] = 1
        images_norm = ((images - img_min) / img_range * 255).astype(np.uint8)
        
        # Get unique track IDs
        unique_tracks = np.unique(masks_tracked)
        unique_tracks = unique_tracks[unique_tracks > 0]  # Remove background
        num_tracks = len(unique_tracks)
        
        # Generate colormap for tracks
        cmap = plt.get_cmap('gist_ncar')
        colors = cmap(np.linspace(0, 1, max(num_tracks + 1, 1000)))[:num_tracks + 1]
        colors[0] = [0, 0, 0, 0]  # Background transparent
        
        # Create color mapping
        track_colors = {}
        for i, track_id in enumerate(unique_tracks):
            track_colors[int(track_id)] = colors[i + 1]  # +1 because 0 is background
        
        # Try to import PIL for text rendering
        try:
            from PIL import Image
            from skimage.measure import regionprops
            PIL_AVAILABLE = True
        except ImportError:
            PIL_AVAILABLE = False
        
        use_callback = callable(progress_callback)
        export_factor = tracking_export_upscale_factor(height, width) if output_path is not None else 1
        export_height = height * export_factor
        export_width = width * export_factor
        native_style = tracking_export_viz_style(height, width)
        export_style = tracking_export_viz_style(export_height, export_width)
        draw_track_labels_native = should_draw_track_id_labels(num_tracks, height, width)
        draw_track_labels_export = should_draw_track_id_labels(num_tracks, export_height, export_width)
        overlays = []
        export_overlays = []
        for frame_idx in tqdm(range(num_frames), desc="Creating tracked overlay", disable=use_callback):
            if use_callback:
                progress_callback(frame_idx, num_frames, "Creating overlay")
            img = images_norm[frame_idx]
            mask = masks_tracked[frame_idx]
            
            # Convert grayscale to RGB
            img_rgb = np.stack([img, img, img], axis=-1)
            
            # Colorize mask using track IDs
            mask_rgb = np.zeros((height, width, 3), dtype=np.float32)
            for track_id, color in track_colors.items():
                mask_rgb[mask == track_id] = color[:3]
            mask_rgb = (mask_rgb * 255).astype(np.uint8)
            
            # Blend
            overlay = (img_rgb * 0.5 + mask_rgb * 0.5).astype(np.uint8)
            
            # Add track ID labels if PIL is available
            if PIL_AVAILABLE and mask.max() > 0:
                try:
                    # Convert to PIL Image
                    pil_img = Image.fromarray(overlay)

                    # Get cell centroids and track IDs
                    regions = regionprops(mask)
                    labels_to_draw = []
                    for region in regions:
                        if region.label == 0:
                            continue
                        
                        centroid_y, centroid_x = region.centroid
                        x, y = int(centroid_x), int(centroid_y)
                        track_id, cell_id = resolve_track_at_centroid(
                            tracks_df, frame_idx, centroid_y, centroid_x
                        )
                        label_text = overlay_label_for_track(track_id, cell_id, int(region.label))
                        text_x = x + native_style.label_offset_x
                        text_y = y + native_style.label_offset_y
                        labels_to_draw.append((label_text, text_x, text_y))

                    if draw_track_labels_native:
                        pil_img = draw_hq_labels(
                            pil_img,
                            labels_to_draw,
                            font_size=native_style.font_size,
                            scale=native_style.label_scale,
                            stroke_width=native_style.stroke_width,
                        )
                    overlay = np.array(pil_img)
                except Exception:
                    # If text rendering fails, continue without labels
                    pass
            
            overlays.append(overlay)

            if export_factor > 1:
                img_export = upscale_rgb_stack(img_rgb[np.newaxis, ...], export_factor)[0]
                mask_export = upscale_mask_stack(mask[np.newaxis, ...], export_factor)[0]
                mask_rgb_export = np.zeros((export_height, export_width, 3), dtype=np.float32)
                for track_id, color in track_colors.items():
                    mask_rgb_export[mask_export == track_id] = color[:3]
                mask_rgb_export = (mask_rgb_export * 255).astype(np.uint8)
                overlay_export = (img_export * 0.5 + mask_rgb_export * 0.5).astype(np.uint8)
                if PIL_AVAILABLE and mask_export.max() > 0:
                    try:
                        pil_export = Image.fromarray(overlay_export)
                        regions_export = regionprops(mask_export)
                        labels_export = []
                        for region in regions_export:
                            if region.label == 0:
                                continue
                            centroid_y, centroid_x = region.centroid
                            x = int(centroid_x)
                            y = int(centroid_y)
                            track_id, cell_id = resolve_track_at_centroid(
                                tracks_df, frame_idx, centroid_y, centroid_x
                            )
                            label_text = overlay_label_for_track(track_id, cell_id, int(region.label))
                            labels_export.append(
                                (
                                    label_text,
                                    x + export_style.label_offset_x,
                                    y + export_style.label_offset_y,
                                )
                            )
                        if draw_track_labels_export:
                            pil_export = draw_hq_labels(
                                pil_export,
                                labels_export,
                                font_size=export_style.font_size,
                                scale=export_style.label_scale,
                                stroke_width=export_style.stroke_width,
                            )
                        overlay_export = np.array(pil_export)
                    except Exception:
                        pass
                export_overlays.append(overlay_export)
        
        overlay_array = np.stack(overlays, axis=0)
        export_overlay_array = np.stack(export_overlays, axis=0) if export_overlays else None
        if output_path is not None:
            upscale = export_factor
            save_array = export_overlay_array if export_overlay_array is not None else overlay_array
            T = save_array.shape[0]
            try:
                from bactomate.modules.tiff_metadata import build_metadata_dict
                ps = getattr(self, '_cal_ps', 0) or 0
                tp = getattr(self, '_cal_tp', 0) or 0
                if upscale > 1 and ps > 0:
                    ps = float(ps) / float(upscale)
                # axes TYXC so metadata matches 4D RGB stack (T, H, W, 3); avoids tifffile "axes do not match shape" warning
                meta = build_metadata_dict(
                    pixel_size_um=ps if ps > 0 else None,
                    time_per_frame_min=tp if tp > 0 else None,
                    axes="TYXC",
                    images=T,
                    frames=T,
                )
                tifffile.imwrite(str(output_path), save_array, metadata=meta, compression='zlib')
            except Exception:
                tifffile.imwrite(str(output_path), save_array, compression='zlib')
        # Return native resolution so trajectory/lineage drawing uses the same y/x space as tracks_df.
        return overlay_array, export_overlay_array

    @staticmethod
    def _lineage_edges_dataframe(napari_graph: Any, tracks_df) -> Any:
        """Napari convention: graph[child_track_id] = [parent_track_id, ...]."""
        import numpy as np
        import pandas as pd

        cols = ["parent_track_id", "daughter_track_id", "division_frame", "parent_last_frame"]
        empty = pd.DataFrame(columns=cols)
        if napari_graph is None or tracks_df is None or len(tracks_df) == 0:
            return empty
        if "track_id" not in tracks_df.columns or "frame" not in tracks_df.columns:
            return empty
        first_frame = tracks_df.groupby("track_id")["frame"].min()
        rows = []
        seen = set()
        try:
            items = list(napari_graph.items())
        except Exception:
            items = []
        for child_raw, parents in items:
            try:
                child = int(child_raw)
            except (TypeError, ValueError):
                continue
            if parents is None:
                continue
            if isinstance(parents, (int, np.integer)):
                plist = [int(parents)]
            else:
                try:
                    plist = [int(p) for p in parents if p is not None]
                except (TypeError, ValueError):
                    continue
            for p in plist:
                key = (p, child)
                if key in seen:
                    continue
                seen.add(key)
                if child not in first_frame.index:
                    continue
                div_f = int(first_frame.loc[child])
                pf = tracks_df.loc[tracks_df["track_id"] == p, "frame"]
                before = pf[pf < div_f]
                parent_last = int(before.max()) if len(before) > 0 else (int(pf.max()) if len(pf) > 0 else "")
                rows.append(
                    {
                        "parent_track_id": p,
                        "daughter_track_id": child,
                        "division_frame": div_f,
                        "parent_last_frame": parent_last,
                    }
                )
        return pd.DataFrame(rows, columns=cols) if rows else empty

    def _maybe_apply_saved_lineage_curation(
        self,
        *,
        tracks_df,
        masks_tracked,
        masks_input,
        napari_graph,
        debug_output,
    ):
        """Apply preview/saved lineage curation before writing outputs."""
        import pandas as pd

        from bactomate.modules.lineage_integration import (
            apply_lineage_curation_to_run,
            lineage_edges_from_napari_graph,
            load_lineage_curation,
            resolve_lineage_curation_path,
        )

        path = self.lineage_curation_path
        if path is None or not Path(path).is_file():
            path = resolve_lineage_curation_path(Path(self.mask_file))
        if not Path(path).is_file():
            return None
        curation = load_lineage_curation(path)
        if not curation:
            return None
        edges_df = pd.DataFrame(
            columns=["parent_track_id", "daughter_track_id", "division_frame", "parent_last_frame"]
        )
        if napari_graph is not None:
            try:
                edges_df = lineage_edges_from_napari_graph(napari_graph, tracks_df)
            except Exception:
                pass
        tracks_df, edges_df, masks_tracked, review_map, stats = apply_lineage_curation_to_run(
            curation,
            tracks_df=tracks_df,
            edges_df=edges_df,
            masks_tracked=masks_tracked,
            masks_input=masks_input,
        )
        self._lineage_curation_review_map = review_map
        debug_output.append(f"✓ Applied saved lineage curation ({Path(path).name})")
        if stats.get("manual_edits_applied"):
            debug_output.append(f"  - Manual orange-cell links: {stats['manual_edits_applied']}")
        if stats.get("division_reviews_applied"):
            debug_output.append(f"  - Division review decisions: {stats['division_reviews_applied']}")
        if stats.get("divisions_dropped"):
            debug_output.append(f"  - Rejected divisions removed: {stats['divisions_dropped']}")
        return tracks_df, masks_tracked, edges_df

    @staticmethod
    def _lineage_founder_map(edges_df, tracks_df) -> dict[int, int]:
        parent_of: dict[int, int] = {}
        if edges_df is not None and len(edges_df) > 0:
            for _, row in edges_df.iterrows():
                parent_of[int(row["daughter_track_id"])] = int(row["parent_track_id"])
        cache: dict[int, int] = {}

        def founder(tid: int) -> int:
            if tid in cache:
                return cache[tid]
            p = parent_of.get(tid)
            if p is None:
                cache[tid] = tid
                return tid
            f = founder(p)
            cache[tid] = f
            return f

        out: dict[int, int] = {}
        if tracks_df is not None and "track_id" in tracks_df.columns:
            for t in tracks_df["track_id"].unique():
                out[int(t)] = founder(int(t))
        return out

    def _lineage_per_track_summary(self, edges_df, tracks_df) -> Any:
        import pandas as pd

        parent_of: dict[int, int] = {}
        children_of: dict[int, list[int]] = defaultdict(list)
        if edges_df is not None and len(edges_df) > 0:
            for _, row in edges_df.iterrows():
                p, d = int(row["parent_track_id"]), int(row["daughter_track_id"])
                parent_of[d] = p
                children_of[p].append(d)

        def subtree_depth(tid: int) -> int:
            ch = children_of.get(tid, [])
            if not ch:
                return 0
            return 1 + max(subtree_depth(c) for c in ch)

        def generation(tid: int) -> int:
            g = 0
            t = tid
            while t in parent_of:
                t = parent_of[t]
                g += 1
            return g

        founder_map = self._lineage_founder_map(edges_df, tracks_df)
        rows = []
        if tracks_df is not None and "track_id" in tracks_df.columns:
            for tid in sorted(tracks_df["track_id"].unique()):
                tid = int(tid)
                cell_id = None
                if "Cell_ID" in tracks_df.columns:
                    sub = tracks_df.loc[tracks_df["track_id"] == tid, "Cell_ID"]
                    if len(sub) > 0:
                        v = sub.iloc[0]
                        cell_id = None if pd.isna(v) else v
                rows.append(
                    {
                        "track_id": tid,
                        "founder_track_id": founder_map.get(tid, tid),
                        "generation": generation(tid),
                        "branch_depth": subtree_depth(tid),
                        "Cell_ID": cell_id,
                    }
                )
        return pd.DataFrame(rows)

    def _create_lineage_visualization(
        self,
        images,
        masks_tracked,
        tracks_df,
        edges_df,
        founder_by_track: dict[int, int],
        output_path: Optional[Path],
        progress_callback=None,
        overlay_frames=None,
        masks_input=None,
        *,
        biology_label_by_track: dict[int, str] | None = None,
    ) -> Optional[np.ndarray]:
        """RGB stack: founder-family colors, trajectories, division markers, biology labels (e.g. 7.1.1)."""
        import numpy as np
        import matplotlib.pyplot as plt
        import tifffile
        from bactomate.gui.workers.tracking_io import (
            blend_path_pixel_on_frame,
            draw_track_centroid_marker,
            overlay_label_for_track,
            should_draw_track_id_labels,
            tracking_export_viz_style,
        )

        use_biology_labels = bool(biology_label_by_track)
        # Full lineage render: never reuse the plain track-ID overlay as base.
        use_overlay_base = (
            not use_biology_labels
            and overlay_frames is not None
            and overlay_frames.ndim == 4
        )
        if use_overlay_base:
            num_frames, height, width = overlay_frames.shape[0], overlay_frames.shape[1], overlay_frames.shape[2]
            images_norm = None
        else:
            num_frames = images.shape[0]
            height, width = images.shape[1], images.shape[2]
            if progress_callback:
                progress_callback(91, "Normalizing images (lineage)...")
            img_min = images.min(axis=(1, 2), keepdims=True)
            img_max = images.max(axis=(1, 2), keepdims=True)
            img_range = img_max - img_min
            img_range[img_range == 0] = 1
            images_norm = ((images - img_min) / img_range * 255).astype(np.uint8)

        unique_founders = sorted(set(founder_by_track.values()))
        cmap = plt.get_cmap("tab20")
        founder_rgb = {}
        for i, fid in enumerate(unique_founders):
            c = cmap((i % 20) / 20.0)
            founder_rgb[int(fid)] = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))

        track_colors = {}
        for tid, f in founder_by_track.items():
            track_colors[int(tid)] = founder_rgb.get(int(f), (180, 180, 180))

        def _region_at_centroid(mask_frame: np.ndarray, y: float, x: float) -> Optional[np.ndarray]:
            yy = int(round(float(y)))
            xx = int(round(float(x)))
            if yy < 0 or yy >= mask_frame.shape[0] or xx < 0 or xx >= mask_frame.shape[1]:
                return None
            label = int(mask_frame[yy, xx])
            if label <= 0:
                return None
            region = mask_frame == label
            return region if np.any(region) else None

        div_marks: list[tuple[int, int, int]] = []
        if edges_df is not None and len(edges_df) > 0 and "y" in tracks_df.columns and "x" in tracks_df.columns:
            for _, er in edges_df.iterrows():
                df = int(er["division_frame"])
                did = int(er["daughter_track_id"])
                sub = tracks_df[(tracks_df["track_id"] == did) & (tracks_df["frame"] == df)]
                if len(sub) == 0:
                    sub = tracks_df[tracks_df["track_id"] == did].sort_values("frame").head(1)
                    if len(sub) == 0:
                        continue
                    df = int(sub["frame"].iloc[0])
                y = int(sub.iloc[0]["y"])
                x = int(sub.iloc[0]["x"])
                div_marks.append((df, y, x))

        if "track_id" not in tracks_df.columns or "frame" not in tracks_df.columns:
            return None
        unique_tracks = tracks_df["track_id"].unique()
        export_style = tracking_export_viz_style(height, width)
        lineage_draw_labels = use_biology_labels or (
            should_draw_track_id_labels(len(unique_tracks), height, width) and not use_overlay_base
        )

        def _lineage_label_text(track_id_int: int, cell_id: int | None) -> str:
            if biology_label_by_track and track_id_int in biology_label_by_track:
                return str(biology_label_by_track[track_id_int])
            return overlay_label_for_track(track_id_int, cell_id, track_id_int)

        visualization = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

        for frame_idx in range(num_frames):
            if progress_callback:
                progress_callback(
                    90 + int((frame_idx + 1) / max(num_frames, 1) * 9),
                    f"Lineage visualization: {frame_idx + 1}/{num_frames}",
                )

            if use_overlay_base:
                vis_frame = np.ascontiguousarray(overlay_frames[frame_idx].copy())
            else:
                img = images_norm[frame_idx]
                vis_frame = np.stack([img, img, img], axis=-1)
                mask_frame = masks_tracked[frame_idx]
                tracked_region_frame = np.zeros((height, width), dtype=bool)
                current_rows = tracks_df[tracks_df["frame"] == frame_idx] if "y" in tracks_df.columns and "x" in tracks_df.columns else []
                for _, tr in current_rows.iterrows() if hasattr(current_rows, "iterrows") else []:
                    tid = int(tr["track_id"])
                    color = track_colors.get(tid)
                    if color is None:
                        continue
                    region = _region_at_centroid(mask_frame, float(tr["y"]), float(tr["x"]))
                    if region is None:
                        region = mask_frame == tid
                    if not np.any(region):
                        continue
                    tracked_region_frame |= region
                    vis_frame[region, 0] = np.clip(vis_frame[region, 0] * 0.5 + color[0] * 0.5, 0, 255).astype(np.uint8)
                    vis_frame[region, 1] = np.clip(vis_frame[region, 1] * 0.5 + color[1] * 0.5, 0, 255).astype(np.uint8)
                    vis_frame[region, 2] = np.clip(vis_frame[region, 2] * 0.5 + color[2] * 0.5, 0, 255).astype(np.uint8)
                if (
                    masks_input is not None
                    and getattr(masks_input, "ndim", 0) == 3
                    and frame_idx < int(masks_input.shape[0])
                    and masks_input.shape[1] == height
                    and masks_input.shape[2] == width
                ):
                    input_frame = masks_input[frame_idx]
                    for raw_label in np.unique(input_frame):
                        raw_label = int(raw_label)
                        if raw_label <= 0:
                            continue
                        raw_region = input_frame == raw_label
                        if np.any(raw_region & tracked_region_frame):
                            continue
                        # Orange means: present in the filtered segmentation, but not linked by Trackastra.
                        vis_frame[raw_region, 0] = np.clip(vis_frame[raw_region, 0] * 0.25 + 255 * 0.75, 0, 255).astype(np.uint8)
                        vis_frame[raw_region, 1] = np.clip(vis_frame[raw_region, 1] * 0.25 + 150 * 0.75, 0, 255).astype(np.uint8)
                        vis_frame[raw_region, 2] = np.clip(vis_frame[raw_region, 2] * 0.25, 0, 255).astype(np.uint8)

            if "y" in tracks_df.columns and "x" in tracks_df.columns:
                tracks_in_frame = tracks_df[
                    (tracks_df["frame"] <= frame_idx) & (tracks_df["track_id"].isin(unique_tracks))
                ]["track_id"].unique()
                labels_to_draw = []
                for track_id_int in tracks_in_frame:
                    track_id_int = int(track_id_int)
                    if track_id_int not in track_colors:
                        continue
                    track_points = tracks_df[
                        (tracks_df["track_id"] == track_id_int) & (tracks_df["frame"] <= frame_idx)
                    ]
                    color = track_colors[track_id_int]
                    if len(track_points) > 1:
                        points = track_points[["y", "x"]].values.astype(int)
                        points[:, 0] = np.clip(points[:, 0], 0, height - 1)
                        points[:, 1] = np.clip(points[:, 1], 0, width - 1)
                        for i in range(len(points) - 1):
                            y1, x1 = points[i]
                            y2, x2 = points[i + 1]
                            dy, dx = abs(y2 - y1), abs(x2 - x1)
                            if dx == 0 and dy == 0:
                                continue
                            max_steps = max(dx, dy)
                            step_size = 1
                            num_steps = max_steps // step_size + 1
                            if dx > dy:
                                x_coords = np.linspace(x1, x2, num_steps, dtype=int)
                                y_coords = np.round(np.linspace(y1, y2, num_steps)).astype(int)
                            else:
                                y_coords = np.linspace(y1, y2, num_steps, dtype=int)
                                x_coords = np.round(np.linspace(x1, x2, num_steps)).astype(int)
                            y_coords = np.clip(y_coords, 0, height - 1)
                            x_coords = np.clip(x_coords, 0, width - 1)
                            for y, x in zip(y_coords, x_coords):
                                if export_style.thick_path:
                                    ys, ye = int(y) - 1, int(y) + 2
                                    xs_, xe = int(x) - 1, int(x) + 2
                                    vis_frame[ys:ye, xs_:xe, 0] = (
                                        vis_frame[ys:ye, xs_:xe, 0] * export_style.path_bg_weight
                                        + color[0] * export_style.path_fg_weight
                                    ).astype(np.uint8)
                                    vis_frame[ys:ye, xs_:xe, 1] = (
                                        vis_frame[ys:ye, xs_:xe, 1] * export_style.path_bg_weight
                                        + color[1] * export_style.path_fg_weight
                                    ).astype(np.uint8)
                                    vis_frame[ys:ye, xs_:xe, 2] = (
                                        vis_frame[ys:ye, xs_:xe, 2] * export_style.path_bg_weight
                                        + color[2] * export_style.path_fg_weight
                                    ).astype(np.uint8)
                                else:
                                    blend_path_pixel_on_frame(
                                        vis_frame,
                                        int(y),
                                        int(x),
                                        color,
                                        bg_weight=export_style.path_bg_weight,
                                        fg_weight=export_style.path_fg_weight,
                                    )
                    current_track = track_points[track_points["frame"] == frame_idx]
                    if lineage_draw_labels and len(current_track) > 0:
                        y = int(current_track.iloc[0]["y"])
                        x = int(current_track.iloc[0]["x"])
                        if 0 <= y < height and 0 <= x < width:
                            draw_track_centroid_marker(vis_frame, y, x, color, export_style)
                            cell_id = None
                            if "Cell_ID" in tracks_df.columns:
                                try:
                                    import pandas as pd
                                    val = current_track.iloc[0]["Cell_ID"]
                                    if val is not None and not pd.isna(val):
                                        cid = int(val)
                                        if cid != int(track_id_int):
                                            cell_id = cid
                                except Exception:
                                    pass
                            label_text = _lineage_label_text(int(track_id_int), cell_id)
                            labels_to_draw.append(
                                (
                                    label_text,
                                    x + export_style.label_offset_x,
                                    y + export_style.label_offset_y,
                                )
                            )

                if labels_to_draw and lineage_draw_labels:
                    try:
                        from PIL import Image

                        pil_img = Image.fromarray(vis_frame)
                        pil_img = draw_hq_labels(
                            pil_img,
                            labels_to_draw,
                            font_size=export_style.font_size,
                            scale=export_style.label_scale,
                            stroke_width=export_style.stroke_width,
                        )
                        vis_frame = np.array(pil_img)
                    except Exception:
                        pass

            for df, y, x in div_marks:
                if df != frame_idx or not (0 <= y < height and 0 <= x < width):
                    continue
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width:
                        vis_frame[yy, xx] = (0, 255, 100)

            visualization[frame_idx] = vis_frame

        if output_path is None:
            return visualization
        upscale = tracking_export_upscale_factor(height, width)
        if upscale > 1:
            visualization = upscale_rgb_stack(visualization, upscale)
        if progress_callback:
            progress_callback(99, "Saving lineage visualization...")
        T = visualization.shape[0]
        meta = None
        try:
            from bactomate.modules.tiff_metadata import build_metadata_dict

            ps = getattr(self, "_cal_ps", 0) or 0
            tp = getattr(self, "_cal_tp", 0) or 0
            if upscale > 1 and ps > 0:
                ps = float(ps) / float(upscale)
            meta = build_metadata_dict(
                pixel_size_um=ps if ps > 0 else None,
                time_per_frame_min=tp if tp > 0 else None,
                axes="TYXC",
                images=T,
                frames=T,
            )
        except Exception:
            pass
        try:
            if meta:
                tifffile.imwrite(str(output_path), visualization, metadata=meta, compression="zlib")
            else:
                tifffile.imwrite(str(output_path), visualization, compression="zlib")
        except Exception:
            tifffile.imwrite(str(output_path), visualization, metadata=meta if meta else {})
        return visualization
