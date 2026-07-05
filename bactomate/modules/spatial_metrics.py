"""
Spatial, fluorescence-distribution, and temporal metrics for Analyze Masks.

Tier 1: cell geometry, FI distribution, foci-to-contour / axis / pole distances
Tier 2: multi-foci spread, membrane-proximal fraction, population Summary helpers
Tier 3: per-Cell_ID temporal deltas when persistent IDs span multiple frames
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, binary_erosion

# Default membrane-proximity threshold for foci (µm).
DEFAULT_MEMBRANE_PROXIMITY_UM = 0.3


def cell_contour_mask(cell_mask: np.ndarray, contour_thickness: int = 1) -> np.ndarray:
    """Return boolean contour pixels for a single-cell binary mask."""
    cell_mask = np.asarray(cell_mask, dtype=bool)
    if not np.any(cell_mask):
        return np.zeros_like(cell_mask, dtype=bool)
    eroded = binary_erosion(cell_mask)
    contour = cell_mask & (~eroded)
    thickness = max(1, int(contour_thickness))
    if thickness > 1:
        for _ in range(thickness - 1):
            contour = binary_dilation(contour) & cell_mask
    return contour


def enrich_cell_geometry(prop: dict, region, pixel_size: float) -> None:
    """Add centroid, orientation, perimeter, and circularity to a property dict."""
    cy, cx = region.centroid
    perimeter_px = float(getattr(region, "perimeter", 0.0) or 0.0)
    area_px = float(getattr(region, "area", 0.0) or 0.0)
    prop["centroid_y"] = float(cy)
    prop["centroid_x"] = float(cx)
    prop["orientation_deg"] = float(np.degrees(region.orientation))
    prop["perimeter_um"] = perimeter_px * pixel_size
    if perimeter_px > 0:
        prop["circularity"] = float(4.0 * np.pi * area_px / (perimeter_px**2))
    else:
        prop["circularity"] = np.nan


def _cell_axis_pca(cell_mask: np.ndarray) -> dict[str, Any] | None:
    """Major-axis geometry via PCA on cell pixels (row, col)."""
    coords = np.argwhere(np.asarray(cell_mask, dtype=bool))
    if coords.shape[0] < 3:
        return None
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = np.cov(centered.T)
    if not np.all(np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        return None
    axis = axis / norm
    projections = centered @ axis
    proj_min = float(projections.min())
    proj_max = float(projections.max())
    pole1 = centroid + axis * proj_min
    pole2 = centroid + axis * proj_max
    return {
        "centroid_y": float(centroid[0]),
        "centroid_x": float(centroid[1]),
        "axis": axis,
        "proj_min": proj_min,
        "proj_max": proj_max,
        "pole1_y": float(pole1[0]),
        "pole1_x": float(pole1[1]),
        "pole2_y": float(pole2[0]),
        "pole2_x": float(pole2[1]),
    }


def _distance_px_to_contour(y: float, x: float, contour_mask: np.ndarray) -> float:
    coords = np.argwhere(np.asarray(contour_mask, dtype=bool))
    if coords.size == 0:
        return float("nan")
    dy = coords[:, 0].astype(np.float64) - float(y)
    dx = coords[:, 1].astype(np.float64) - float(x)
    return float(np.sqrt(dy * dy + dx * dx).min())


def focus_spatial_metrics(
    y: float,
    x: float,
    cell_mask: np.ndarray,
    pixel_size: float,
    *,
    contour_thickness: int = 1,
    membrane_proximity_um: float = DEFAULT_MEMBRANE_PROXIMITY_UM,
) -> dict[str, float | bool]:
    """Per-focus spatial metrics relative to cell contour and major axis."""
    cell_mask = np.asarray(cell_mask, dtype=bool)
    contour = cell_contour_mask(cell_mask, contour_thickness=contour_thickness)
    dist_px = _distance_px_to_contour(y, x, contour)
    dist_um = dist_px * pixel_size if np.isfinite(dist_px) else float("nan")

    axis_info = _cell_axis_pca(cell_mask)
    out: dict[str, float | bool] = {
        "distance_to_contour_um": dist_um,
        "membrane_proximal": bool(np.isfinite(dist_um) and dist_um <= membrane_proximity_um),
    }
    if axis_info is None:
        out.update(
            {
                "longitudinal_position": float("nan"),
                "radial_offset_um": float("nan"),
                "distance_to_nearest_pole_um": float("nan"),
            }
        )
        return out

    rel = np.array([float(y), float(x)], dtype=np.float64) - np.array(
        [axis_info["centroid_y"], axis_info["centroid_x"]], dtype=np.float64
    )
    axis = axis_info["axis"]
    along = float(np.dot(rel, axis))
    proj_span = axis_info["proj_max"] - axis_info["proj_min"]
    if proj_span > 0:
        longitudinal = float((along - axis_info["proj_min"]) / proj_span)
        longitudinal = float(np.clip(longitudinal, 0.0, 1.0))
    else:
        longitudinal = 0.5
    perp = rel - along * axis
    radial_um = float(np.linalg.norm(perp) * pixel_size)
    d_pole1 = float(np.hypot(y - axis_info["pole1_y"], x - axis_info["pole1_x"]) * pixel_size)
    d_pole2 = float(np.hypot(y - axis_info["pole2_y"], x - axis_info["pole2_x"]) * pixel_size)
    out.update(
        {
            "longitudinal_position": longitudinal,
            "radial_offset_um": radial_um,
            "distance_to_nearest_pole_um": float(min(d_pole1, d_pole2)),
        }
    )
    return out


def aggregate_foci_spatial(
    focus_metrics: list[dict[str, float | bool]],
) -> dict[str, float]:
    """Collapse per-focus spatial metrics to per-cell aggregates."""
    if not focus_metrics:
        return {
            "foci_min_dist_contour_um": np.nan,
            "foci_mean_dist_contour_um": np.nan,
            "foci_mean_longitudinal_position": np.nan,
            "foci_longitudinal_spread": np.nan,
            "foci_mean_radial_offset_um": np.nan,
            "foci_min_dist_nearest_pole_um": np.nan,
            "foci_nearest_neighbor_um": np.nan,
            "foci_membrane_proximal_fraction": np.nan,
        }

    dists = [float(m["distance_to_contour_um"]) for m in focus_metrics if np.isfinite(m.get("distance_to_contour_um", np.nan))]
    longs = [float(m["longitudinal_position"]) for m in focus_metrics if np.isfinite(m.get("longitudinal_position", np.nan))]
    radials = [float(m["radial_offset_um"]) for m in focus_metrics if np.isfinite(m.get("radial_offset_um", np.nan))]
    poles = [
        float(m["distance_to_nearest_pole_um"])
        for m in focus_metrics
        if np.isfinite(m.get("distance_to_nearest_pole_um", np.nan))
    ]
    prox = [bool(m.get("membrane_proximal", False)) for m in focus_metrics]

    nn_um = float("nan")
    if len(focus_metrics) >= 2:
        coords = np.array(
            [[float(m.get("y", np.nan)), float(m.get("x", np.nan))] for m in focus_metrics],
            dtype=np.float64,
        )
        if np.all(np.isfinite(coords)):
            nn_px = float("inf")
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    nn_px = min(nn_px, float(np.linalg.norm(coords[i] - coords[j])))
            if np.isfinite(nn_px) and nn_px < float("inf"):
                # pixel_size stored on metrics dict when available
                ps = float(focus_metrics[0].get("_pixel_size", 1.0))
                nn_um = nn_px * ps

    return {
        "foci_min_dist_contour_um": float(np.min(dists)) if dists else np.nan,
        "foci_mean_dist_contour_um": float(np.mean(dists)) if dists else np.nan,
        "foci_mean_longitudinal_position": float(np.mean(longs)) if longs else np.nan,
        "foci_longitudinal_spread": float(np.std(longs)) if len(longs) > 1 else (0.0 if longs else np.nan),
        "foci_mean_radial_offset_um": float(np.mean(radials)) if radials else np.nan,
        "foci_min_dist_nearest_pole_um": float(np.min(poles)) if poles else np.nan,
        "foci_nearest_neighbor_um": nn_um,
        "foci_membrane_proximal_fraction": float(np.mean(prox)) if prox else np.nan,
    }


def compute_fi_distribution_metrics(
    mask: np.ndarray,
    fi_image: np.ndarray,
    cell_label: int,
    *,
    contour_thickness: int = 1,
    do_contour: bool = True,
) -> dict[str, float]:
    """Extended FI stats: interior mean, std, CV, brightest-decile mean."""
    cell_mask = np.asarray(mask == cell_label, dtype=bool)
    if not np.any(cell_mask):
        return {
            "interior_fi": np.nan,
            "fi_std": np.nan,
            "fi_cv": np.nan,
            "fi_top10_mean": np.nan,
        }

    fi_plane = np.asarray(fi_image, dtype=np.float64)
    if fi_plane.ndim > 2:
        fi_plane = fi_plane[:, :, 0]

    cell_pixels = fi_plane[cell_mask]
    mean_fi = float(np.mean(cell_pixels)) if cell_pixels.size else np.nan
    fi_std = float(np.std(cell_pixels)) if cell_pixels.size else np.nan
    fi_cv = float(fi_std / mean_fi) if cell_pixels.size and np.isfinite(mean_fi) and mean_fi > 0 else np.nan

    if do_contour:
        contour = cell_contour_mask(cell_mask, contour_thickness=contour_thickness)
        interior = cell_mask & (~contour)
        interior_pixels = fi_plane[interior] if np.any(interior) else cell_pixels
    else:
        interior_pixels = cell_pixels
    interior_fi = float(np.mean(interior_pixels)) if interior_pixels.size else np.nan

    fi_top10_mean = np.nan
    if cell_pixels.size:
        k = max(1, int(np.ceil(cell_pixels.size * 0.10)))
        fi_top10_mean = float(np.mean(np.partition(cell_pixels, -k)[-k:]))

    return {
        "interior_fi": interior_fi,
        "fi_std": fi_std,
        "fi_cv": fi_cv,
        "fi_top10_mean": fi_top10_mean,
        "mean_fi": mean_fi,
    }


def apply_foci_spatial_to_frame(
    cells: list[dict],
    foci_rows: list[dict],
    mask: np.ndarray,
    pixel_size: float,
    *,
    contour_thickness: int = 1,
    membrane_proximity_um: float = DEFAULT_MEMBRANE_PROXIMITY_UM,
) -> None:
    """Annotate foci rows and parent cells with spatial metrics (in-place)."""
    by_label: dict[int, list[dict]] = defaultdict(list)
    for row in foci_rows:
        if not row.get("in_cell") or row.get("cell_id") is None:
            continue
        label = int(row["cell_id"])
        cell_mask = np.asarray(mask == label, dtype=bool)
        if not np.any(cell_mask):
            continue
        y, x = float(row.get("y", 0)), float(row.get("x", 0))
        spatial = focus_spatial_metrics(
            y,
            x,
            cell_mask,
            pixel_size,
            contour_thickness=contour_thickness,
            membrane_proximity_um=membrane_proximity_um,
        )
        row.update(spatial)
        row["cell_orientation_deg"] = _cell_axis_orientation_deg(cell_mask)
        tagged = dict(spatial)
        tagged["y"] = y
        tagged["x"] = x
        tagged["_pixel_size"] = pixel_size
        by_label[label].append(tagged)

    for cell in cells:
        label = int(cell["label"])
        metrics = by_label.get(label, [])
        cell.update(aggregate_foci_spatial(metrics))


def _cell_axis_orientation_deg(cell_mask: np.ndarray) -> float:
    labeled = np.asarray(cell_mask, dtype=np.uint8)
    regions = regionprops(labeled)
    if not regions:
        return float("nan")
    return float(np.degrees(regions[0].orientation))


def apply_temporal_metrics(all_frame_data: list[dict]) -> None:
    """
    Add temporal deltas and first-focus frame per persistent Cell_ID (in-place).

    Requires ``cell_id`` on each cell dict (assigned before this runs).
    """
    history: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for frame_data in all_frame_data:
        frame = int(frame_data["frame"])
        for cell in frame_data.get("cells") or []:
            cid = cell.get("cell_id")
            if cid is None:
                continue
            history[int(cid)].append((frame, cell))

    numeric_keys = (
        "area_um2",
        "mean_fi",
        "foci_count",
        "foci_min_dist_contour_um",
        "foci_mean_dist_contour_um",
        "contour_fi",
        "interior_fi",
    )
    delta_map = {
        "area_um2": "delta_area_um2",
        "mean_fi": "delta_mean_fi",
        "foci_count": "delta_foci_count",
        "foci_min_dist_contour_um": "delta_foci_min_dist_contour_um",
        "foci_mean_dist_contour_um": "delta_foci_mean_dist_contour_um",
        "contour_fi": "delta_contour_fi",
        "interior_fi": "delta_interior_fi",
    }

    for _cid, entries in history.items():
        entries.sort(key=lambda item: item[0])
        first_focus_frame = None
        for frame, cell in entries:
            if int(cell.get("foci_count") or 0) > 0:
                first_focus_frame = frame
                break
        for idx, (frame, cell) in enumerate(entries):
            if first_focus_frame is not None:
                cell["time_of_first_focus"] = first_focus_frame
            else:
                cell["time_of_first_focus"] = np.nan
            if idx == 0:
                for dk in delta_map.values():
                    cell[dk] = np.nan
                cell["delta_area_um2_per_frame"] = np.nan
                continue
            prev_frame, prev = entries[idx - 1]
            dt = frame - prev_frame
            for src, dst in delta_map.items():
                cur_v = cell.get(src)
                prev_v = prev.get(src)
                if cur_v is None and prev_v is None:
                    cell[dst] = np.nan
                    continue
                try:
                    cur_f = float(cur_v) if cur_v is not None else np.nan
                    prev_f = float(prev_v) if prev_v is not None else np.nan
                except (TypeError, ValueError):
                    cell[dst] = np.nan
                    continue
                if np.isfinite(cur_f) and np.isfinite(prev_f):
                    cell[dst] = cur_f - prev_f
                else:
                    cell[dst] = np.nan
            if dt > 0 and np.isfinite(cell.get("delta_area_um2", np.nan)):
                cell["delta_area_um2_per_frame"] = float(cell["delta_area_um2"]) / dt
            else:
                cell["delta_area_um2_per_frame"] = np.nan


def frame_summary_spatial_metrics(cells: list[dict]) -> dict[str, float]:
    """Population-level spatial/foci stats for one Summary row."""
    if not cells:
        return {
            "Median Foci per Cell": np.nan,
            "Fraction Cells with Foci": np.nan,
            "Mean Min Foci-Contour Distance (µm)": np.nan,
            "Mean Membrane-Proximal Foci Fraction": np.nan,
        }
    foci_counts = [int(c.get("foci_count") or 0) for c in cells]
    with_foci = sum(1 for n in foci_counts if n > 0)
    min_dists = [
        float(c["foci_min_dist_contour_um"])
        for c in cells
        if np.isfinite(c.get("foci_min_dist_contour_um", np.nan))
    ]
    prox_fracs = [
        float(c["foci_membrane_proximal_fraction"])
        for c in cells
        if np.isfinite(c.get("foci_membrane_proximal_fraction", np.nan))
    ]
    return {
        "Median Foci per Cell": float(np.median(foci_counts)),
        "Fraction Cells with Foci": with_foci / len(cells),
        "Mean Min Foci-Contour Distance (µm)": float(np.mean(min_dists)) if min_dists else np.nan,
        "Mean Membrane-Proximal Foci Fraction": float(np.mean(prox_fracs)) if prox_fracs else np.nan,
    }
