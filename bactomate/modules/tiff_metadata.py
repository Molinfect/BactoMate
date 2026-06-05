"""
Central TIFF (and ND2) calibration and acquisition metadata: read, format, and build for write.
Single source of truth for: pixel size, time per frame, channel names/wavelengths, objective,
exposure, optics/capturing. Used by file conversion (display + write), tracking, workers.
All time-per-frame values are in minutes when returned or passed to build_metadata_dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


def _rational_tag_float(value: Any) -> Optional[float]:
    """Coerce TIFF tag values (float, int, or numerator/denominator rational) to float."""
    if value is None:
        return None
    try:
        if hasattr(value, "numerator") and getattr(value, "denominator", 0):
            return float(value.numerator) / float(value.denominator)
        if isinstance(value, (tuple, list)) and len(value) >= 1:
            num = float(value[0])
            den = float(value[1]) if len(value) > 1 else 1.0
            return num / den if den else num
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def infer_tiff_stack_dims(
    *,
    num_pages: int = 0,
    imagej_metadata: Optional[dict[str, Any]] = None,
    series_axes: str = "",
    series_shape: Optional[tuple[int, ...]] = None,
    page_ndim: int = 2,
    page_shape: Optional[tuple[int, ...]] = None,
) -> tuple[int, int, int]:
    """Infer (num_time, num_channels, num_z) for a TIFF.

    ImageJ ``images`` is the total number of 2D planes (T×C×Z), not always the time count.
    When tifffile exposes series axes (e.g. CYX), those take precedence over ambiguous ImageJ tags.
    """
    meta = imagej_metadata if isinstance(imagej_metadata, dict) else {}
    channels = max(1, int(meta.get("channels") or 1))
    slices = max(1, int(meta.get("slices") or 1))

    axes = (series_axes or "").upper()
    shape = tuple(int(x) for x in series_shape) if series_shape else ()
    if axes and shape and len(axes) == len(shape):
        t = int(shape[axes.index("T")]) if "T" in axes else 1
        c = int(shape[axes.index("C")]) if "C" in axes else channels
        z = int(shape[axes.index("Z")]) if "Z" in axes else slices
        return max(1, t), max(1, c), max(1, z)

    ps = tuple(int(x) for x in page_shape) if page_shape else ()
    if page_ndim == 3 and len(ps) >= 3:
        if ps[0] < 64 and channels <= 1:
            return max(1, ps[0]), 1, 1
        if ps[0] < 64:
            return 1, max(1, ps[0]), 1
    if page_ndim == 4 and len(ps) >= 4:
        if ps[1] < 64:
            return max(1, ps[0]), max(1, ps[1]), 1
        if ps[0] < 64:
            return 1, max(1, ps[1]), max(1, ps[0])

    if meta.get("frames") is not None:
        t = max(1, int(meta["frames"]))
    else:
        denom = channels * slices
        if num_pages > 0 and denom > 0:
            if meta.get("images") is not None:
                im = int(meta["images"])
                if im > 0 and im % denom == 0:
                    t = max(1, im // denom)
                elif num_pages % denom == 0:
                    t = max(1, num_pages // denom)
                else:
                    t = max(1, num_pages)
            elif num_pages % denom == 0:
                t = max(1, num_pages // denom)
            else:
                t = max(1, num_pages)
        else:
            t = 1
    return max(1, t), max(1, channels), max(1, slices)


def _pixel_size_um_from_tiff_page(page0: Any, imagej_metadata: Optional[dict[str, Any]] = None) -> float:
    """XY pixel size in µm/px. Does not use ImageJ Z ``spacing`` (that is step size, not XY)."""
    pixel_size = 0.0
    meta = imagej_metadata if isinstance(imagej_metadata, dict) else {}
    if meta.get("pixelSize") is not None:
        try:
            pixel_size = float(meta["pixelSize"])
        except (TypeError, ValueError):
            pass
    tags = getattr(page0, "tags", None) or {}
    desc_raw = None
    for key in ("ImageDescription", 270):
        ttag = tags.get(key) if tags and hasattr(tags, "get") else None
        if ttag is not None:
            desc_raw = getattr(ttag, "value", ttag)
            break
    if desc_raw is None and hasattr(page0, "ImageDescription"):
        desc_raw = page0.ImageDescription
    if desc_raw is not None and pixel_size <= 0:
        desc = desc_raw.decode("utf-8", errors="ignore") if isinstance(desc_raw, bytes) else str(desc_raw)
        for part in desc.replace("\n", " ").replace("\r", " ").split():
            if part.startswith("pixelSize="):
                try:
                    pixel_size = float(part.split("=", 1)[1].strip())
                    break
                except (IndexError, ValueError):
                    pass
    if pixel_size > 0:
        return pixel_size

    xres_val: Optional[float] = None
    unit_val: Any = None
    if hasattr(page0, "resolution") and page0.resolution:
        res = page0.resolution
        if isinstance(res, (tuple, list)) and len(res) >= 1:
            xres_val = _rational_tag_float(res[0])
        if hasattr(page0, "resolutionunit"):
            unit_val = page0.resolutionunit
    if xres_val is None and tags is not None:
        for key in (282, "XResolution"):
            t = tags.get(key) if hasattr(tags, "get") else None
            if t is not None:
                xres_val = _rational_tag_float(getattr(t, "value", t))
                break
        for key in (296, "ResolutionUnit"):
            t = tags.get(key) if hasattr(tags, "get") else None
            if t is not None:
                unit_val = getattr(t, "value", t)
                break
    if xres_val is not None and xres_val > 0:
        try:
            unit_int = int(unit_val) if unit_val is not None else 1
        except (TypeError, ValueError):
            unit_int = 1
        if unit_int == 3:
            return 1e4 / xres_val
        if unit_int == 2:
            return 25400.0 / xres_val
        # ImageJ/Fiji often leave ResolutionUnit=NONE with XResolution=(1,1); do not treat as 1 px/µm.
        if unit_int in (1, None) and xres_val <= 2.0:
            return 0.0
        if 2.0 < xres_val <= 500.0:
            return 1.0 / xres_val
    return 0.0


def normalize_tiff_array_to_tcyx(
    arr: np.ndarray,
    *,
    num_time: int,
    num_channels: int,
    num_z: int = 1,
    series_axes: str = "",
) -> np.ndarray:
    """Reshape a TIFF array to (T, C, Y, X) for preview and channel split."""
    a = np.asarray(arr)
    axes = (series_axes or "").upper()
    t, c, z = max(1, int(num_time)), max(1, int(num_channels)), max(1, int(num_z))

    if axes and len(axes) == a.ndim:
        axis_map = {ax: i for i, ax in enumerate(axes)}
        perm = []
        for ax in "TCZYX":
            if ax in axis_map:
                perm.append(axis_map[ax])
        if len(perm) == a.ndim:
            a = np.transpose(a, perm)
        if a.ndim == 5:
            if z == 1 and a.shape[2] == 1:
                a = a[:, :, 0, :, :]
            return a
        if a.ndim == 4:
            return a
        if a.ndim == 3 and "C" in axes and "T" not in axes:
            return a[np.newaxis, ...]

    if a.ndim == 2:
        return a[np.newaxis, np.newaxis, :, :]
    if a.ndim == 3:
        n = a.shape[0]
        if n == t * c * z:
            if z > 1:
                return a.reshape(t, c, z, a.shape[1], a.shape[2])
            return a.reshape(t, c, a.shape[1], a.shape[2])
        if c > 1 and n == c:
            return a[np.newaxis, :, :, :]
        if t > 1 and c == 1:
            return a[:, np.newaxis, :, :]
        return a[:, np.newaxis, :, :]
    if a.ndim == 4:
        if c > 1 and a.shape[1] == c:
            return a
        if a.shape[0] == t and a.shape[1] == c:
            return a
        if c == 1 and a.shape[1] > 1:
            return a[:, 0:1, :, :]
    if a.ndim == 5 and a.shape[0] == t and a.shape[1] == c:
        return a[:, :, 0, :, :] if z == 1 else a[:, :, 0, :, :]
    if a.ndim >= 2:
        yx = a.reshape(-1, a.shape[-2], a.shape[-1])
        n = yx.shape[0]
        if n == t * c:
            return yx.reshape(t, c, yx.shape[1], yx.shape[2])
        if n == c and t == 1:
            return yx[np.newaxis, :, :, :]
    return a[np.newaxis, np.newaxis, :, :] if a.ndim == 2 else a


def _looks_like_channel_name(s: str) -> bool:
    if not s or len(s) > 60:
        return False
    s_lower = s.lower()
    if "c:1/" in s_lower or "t:1/" in s_lower or " c:" in s_lower or " t:" in s_lower:
        return False
    if "ldm " in s_lower and "series" in s_lower:
        return False
    if s.strip().startswith("c:") or "/" in s and any(c.isdigit() for c in s.split("/")[0]):
        return False
    return True


def _lut_display_color_name(lut_rgb: np.ndarray) -> str:
    """Guess a short display name from an ImageJ LUT (3 x 256, RGB ramps)."""
    lut = np.asarray(lut_rgb)
    if lut.ndim != 2 or lut.shape[0] < 3:
        return ""
    r = float(np.mean(lut[0]))
    g = float(np.mean(lut[1]))
    b = float(np.mean(lut[2]))
    if abs(r - g) < 18 and abs(g - b) < 18 and abs(r - b) < 18:
        return "Gray"
    if b >= r and b >= g and b > 35:
        return "Blue"
    if g >= r and g >= b and g > 35:
        return "Green"
    if r >= g and r >= b and r > 35:
        return "Red"
    if b > r and b > g:
        return "Cyan"
    if r > g and g > b:
        return "Yellow"
    return "Gray"


def _channel_names_from_imagej_info(meta: dict[str, Any], channel_count: int) -> list[str]:
    """Parse ImageJ ``info`` / ImageDescription text for ``Channel:`` or ``labelN=`` lines."""
    chunks: list[str] = []
    info = meta.get("Info") or meta.get("info")
    if info is not None:
        if isinstance(info, bytes):
            chunks.append(info.decode("utf-8", errors="ignore"))
        else:
            chunks.append(str(info))
    names: list[str] = []
    for chunk in chunks:
        for line in chunk.replace("\r", "\n").split("\n"):
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("channel:") or low.startswith("channel "):
                val = line.split(":", 1)[-1].strip() if ":" in line else line.split(None, 1)[-1].strip()
                if val and _looks_like_channel_name(val):
                    names.append(val)
            elif "=" in line:
                key, val = line.split("=", 1)
                key = key.strip().lower()
                val = val.strip()
                if key in ("label", "labels") and val:
                    names.extend(s.strip() for s in val.split(",") if s.strip())
                elif key.startswith("channel") and val and _looks_like_channel_name(val):
                    names.append(val)
    out = [n for n in names if _looks_like_channel_name(n)]
    return out[:channel_count]


def _channel_names_from_imagej_luts(meta: dict[str, Any], channel_count: int) -> list[str]:
    luts = meta.get("LUTs")
    if luts is None:
        return []
    arr = np.asarray(luts)
    names: list[str] = []
    try:
        if arr.ndim == 3 and arr.shape[0] >= channel_count:
            for i in range(channel_count):
                nm = _lut_display_color_name(arr[i])
                names.append(nm or f"Channel_{i + 1}")
            return names
        if arr.ndim == 3 and arr.shape[1] == channel_count:
            for i in range(channel_count):
                nm = _lut_display_color_name(arr[:, i, :])
                names.append(nm or f"Channel_{i + 1}")
            return names
    except (TypeError, ValueError, IndexError):
        pass
    return []


def extract_tiff_channel_names(
    *,
    imagej_metadata: Optional[dict[str, Any]] = None,
    ome_xml: Optional[str] = None,
    image_description: Optional[str] = None,
    channel_count: int = 1,
) -> tuple[list[str], str]:
    """Return (channel_names, source) where source describes how names were obtained.

    source is one of: labels, ome, info, luts, default
    """
    cc = max(1, int(channel_count))
    meta = imagej_metadata if isinstance(imagej_metadata, dict) else {}

    labels = meta.get("Labels") or meta.get("labels")
    if labels is not None:
        if isinstance(labels, (list, tuple)):
            names = [str(x).strip() for x in labels if x][:cc]
        else:
            names = [s.strip() for s in str(labels).split(",") if s.strip()][:cc]
        if len(names) == cc and all(_looks_like_channel_name(n) for n in names):
            return (names, "labels")

    if ome_xml:
        try:
            import re
            xml = ome_xml if isinstance(ome_xml, str) else ome_xml.decode("utf-8", errors="ignore")
            names = []
            for m in re.finditer(r'<Channel[^>]*Name="([^"]*)"', xml):
                raw = (m.group(1) or "").strip()
                names.append(raw if _looks_like_channel_name(raw) else f"Channel_{len(names) + 1}")
                if len(names) >= cc:
                    break
            if len(names) == cc and all(_looks_like_channel_name(n) for n in names):
                return (names, "ome")
        except (TypeError, AttributeError):
            pass

    info_names = _channel_names_from_imagej_info(meta, cc)
    if len(info_names) == cc:
        return (info_names, "info")

    if image_description:
        desc_meta = {"Info": image_description}
        desc_names = _channel_names_from_imagej_info(desc_meta, cc)
        if len(desc_names) == cc:
            return (desc_names, "info")

    lut_names = _channel_names_from_imagej_luts(meta, cc)
    if len(lut_names) == cc and any(n not in (f"Channel_{i + 1}" for i in range(cc)) for n in lut_names):
        return (lut_names, "luts")

    return ([f"Channel_{i + 1}" for i in range(cc)], "default")


def get_tiff_file_info(path: Union[str, Path]) -> Optional[dict[str, Any]]:
    """Read TIFF dimensions and calibration in BactoMate file-conversion shape.

    Returns dict with file_type, num_time, num_channels, num_z, channel_names, pixel_size,
    series_axes, or None if unreadable.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        import tifffile
    except ImportError:
        return None
    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.pages:
                return None
            page0 = tif.pages[0]
            shape = tuple(int(x) for x in page0.shape)
            ndim = len(shape)
            num_pages = len(tif.pages)
            ij_meta = getattr(tif, "imagej_metadata", None) or {}
            series_axes = ""
            series_shape: Optional[tuple[int, ...]] = None
            try:
                if getattr(tif, "series", None) and len(tif.series) > 0:
                    s0 = tif.series[0]
                    series_axes = str(getattr(s0, "axes", None) or "")
                    sh = getattr(s0, "shape", None)
                    if sh is not None:
                        series_shape = tuple(int(x) for x in sh)
            except (IndexError, TypeError, AttributeError):
                pass

            num_time, num_channels, num_z = infer_tiff_stack_dims(
                num_pages=num_pages,
                imagej_metadata=ij_meta if isinstance(ij_meta, dict) else {},
                series_axes=series_axes,
                series_shape=series_shape,
                page_ndim=ndim,
                page_shape=shape,
            )
            pixel_size = _pixel_size_um_from_tiff_page(page0, ij_meta if isinstance(ij_meta, dict) else {})
            desc_raw = None
            try:
                ttag = page0.tags.get(270) if hasattr(page0, "tags") else None
                if ttag is not None:
                    desc_raw = getattr(ttag, "value", ttag)
            except Exception:
                pass
            if isinstance(desc_raw, bytes):
                desc_raw = desc_raw.decode("utf-8", errors="ignore")
            ome_xml = None
            if hasattr(tif, "ome_metadata") and tif.ome_metadata:
                ome_xml = (
                    tif.ome_metadata
                    if isinstance(tif.ome_metadata, str)
                    else tif.ome_metadata.decode("utf-8", errors="ignore")
                )
            channel_names, ch_src = extract_tiff_channel_names(
                imagej_metadata=ij_meta if isinstance(ij_meta, dict) else {},
                ome_xml=ome_xml,
                image_description=str(desc_raw) if desc_raw else None,
                channel_count=num_channels,
            )
            return {
                "file_type": "tiff",
                "num_time": num_time,
                "num_channels": num_channels,
                "num_z": num_z,
                "channel_names": channel_names,
                "channel_names_source": ch_src,
                "pixel_size": pixel_size if pixel_size > 0 else None,
                "series_axes": series_axes,
                "series_shape": series_shape,
            }
    except Exception:
        return None


def read_metadata_from_path(image_path: Optional[Path]) -> dict[str, Any]:
    """Read full acquisition metadata from ND2 or TIFF. Returns a dict with keys:
    pixel_size_um, time_per_frame_min, frame_count, channel_count, channel_names,
    channel_wavelengths (list of dicts: name, excitation_nm, emission_nm),
    objective_name, objective_magnification, zoom_magnification, exposure_ms,
    optics, capturing, description_filter_snippet. Missing values are None or [].
    Use this for display and for building write metadata; use read_calibration_from_path for (pixel_size, time) only.
    """
    out: dict[str, Any] = {
        "pixel_size_um": None,
        "time_per_frame_min": None,
        "frame_count": None,
        "channel_count": None,
        "z_count": None,
        "channel_names": [],
        "channel_wavelengths": [],
        "objective_name": None,
        "objective_magnification": None,
        "zoom_magnification": None,
        "exposure_ms": None,
        "optics": None,
        "capturing": None,
        "description_filter_snippet": None,
        "channel_exposure_ms": [],
        "channel_filter_snippet": [],
        "bits_per_component": None,
        "dia_percent": None,  # DIA/phase condenser % when found in description (e.g. 50)
        "nd2_axis_order": None,  # list of axis names matching ndarray shape (from nd2.ND2File.sizes)
    }
    if not image_path or not image_path.is_file():
        return out
    suffix = image_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        ps, tp = read_calibration_from_path(image_path)
        out["pixel_size_um"] = ps if ps > 0 else None
        out["time_per_frame_min"] = tp if tp > 0 else None
        _read_tiff_metadata_into(image_path, out)
        return out
    if suffix == ".nd2":
        _read_nd2_metadata_into(image_path, out)
        return out
    if suffix in (".lif", ".czi"):
        _read_lif_czi_metadata_into(image_path, out)
    return out


def _read_lif_czi_metadata_into(path: Path, out: dict[str, Any]) -> None:
    """Fill frame_count, channel_count, z_count, channel_names, pixel_size from LIF/CZI via BioIO when available."""
    try:
        from bactomate.modules.bioio_reader import get_info as bioio_get_info
        info = bioio_get_info(path)
        if not info:
            return
        out["frame_count"] = info.get("num_time") or 1
        out["channel_count"] = info.get("num_channels") or 1
        out["z_count"] = info.get("num_z") or 1
        if info.get("channel_names"):
            out["channel_names"] = list(info["channel_names"])
        if info.get("pixel_size") is not None:
            try:
                out["pixel_size_um"] = float(info["pixel_size"])
            except (TypeError, ValueError):
                pass
    except Exception:
        pass


def _read_tiff_metadata_into(path: Path, out: dict[str, Any]) -> None:
    """Fill frame_count, channel_count, bits_per_component, and placeholder channel_wavelengths from TIFF."""
    try:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            if not tif.pages:
                return
            page0 = tif.pages[0]
            meta = getattr(tif, "imagej_metadata", None) or {}
            if not isinstance(meta, dict):
                meta = {}
            series_axes = ""
            series_shape: Optional[tuple[int, ...]] = None
            try:
                if getattr(tif, "series", None) and len(tif.series) > 0:
                    s0 = tif.series[0]
                    series_axes = str(getattr(s0, "axes", None) or "")
                    sh = getattr(s0, "shape", None)
                    if sh is not None:
                        series_shape = tuple(int(x) for x in sh)
            except (IndexError, TypeError, AttributeError):
                pass
            num_time, num_channels, num_z = infer_tiff_stack_dims(
                num_pages=len(tif.pages),
                imagej_metadata=meta,
                series_axes=series_axes,
                series_shape=series_shape,
                page_ndim=len(page0.shape),
                page_shape=tuple(int(x) for x in page0.shape),
            )
            out["frame_count"] = num_time
            out["channel_count"] = num_channels
            out["z_count"] = num_z
            ps = _pixel_size_um_from_tiff_page(page0, meta)
            if ps > 0 and (out.get("pixel_size_um") is None or float(out.get("pixel_size_um") or 0) <= 0):
                out["pixel_size_um"] = ps
            # Bits per pixel from dtype
            dtype = getattr(page0, "dtype", None)
            if dtype is not None:
                kind = getattr(dtype, "kind", "")
                if "u" in str(kind) or "i" in str(dtype):
                    try:
                        out["bits_per_component"] = int(dtype.itemsize * 8)
                    except (TypeError, ValueError):
                        pass
            cc = out.get("channel_count") or 1
            desc_raw = None
            try:
                ttag = page0.tags.get(270) if hasattr(page0, "tags") else None
                if ttag is not None:
                    desc_raw = getattr(ttag, "value", ttag)
            except Exception:
                pass
            if isinstance(desc_raw, bytes):
                desc_raw = desc_raw.decode("utf-8", errors="ignore")
            ome_xml = None
            if hasattr(tif, "ome_metadata") and tif.ome_metadata:
                ome_xml = (
                    tif.ome_metadata
                    if isinstance(tif.ome_metadata, str)
                    else tif.ome_metadata.decode("utf-8", errors="ignore")
                )
            channel_names, ch_src = extract_tiff_channel_names(
                imagej_metadata=meta,
                ome_xml=ome_xml,
                image_description=str(desc_raw) if desc_raw else None,
                channel_count=int(cc),
            )
            out["channel_names"] = channel_names
            out["channel_names_source"] = ch_src
            out["channel_wavelengths"] = [{"name": n} for n in channel_names]
    except Exception:
        pass


def _nd2_reconcile_dims_from_shape(r: Any, out: dict[str, Any]) -> None:
    """Set frame/channel/z counts from ND2 array shape + axis order (authoritative vs contents.frameCount)."""
    try:
        keys = tuple(r.sizes.keys())
        shp = r.shape
        if len(keys) != len(shp):
            return
        out["nd2_axis_order"] = list(keys)
        for i, k in enumerate(keys):
            if k == "T":
                out["frame_count"] = int(shp[i])
            elif k == "C":
                out["channel_count"] = int(shp[i])
            elif k == "Z":
                out["z_count"] = int(shp[i])
        # nd2 omits axes with size 1 from sizes — so T=1 acquisitions have no "T" key.
        # Axis order is then e.g. Z,C,Y,X. contents.frameCount is sequence length (planes/loops), not timepoints.
        if "T" not in keys:
            out["frame_count"] = 1
    except Exception:
        pass


def _read_nd2_metadata_into(path: Path, out: dict[str, Any]) -> None:
    """Fill out dict with full ND2 metadata (dimensions, pixel size, time, channels, wavelengths, objective, exposure, text_info)."""
    try:
        import nd2
        with nd2.ND2File(path) as r:
            sizes = getattr(r, "sizes", None)
            if sizes:
                out["frame_count"] = sizes.get("T", 1)
                out["channel_count"] = sizes.get("C", 1)
                out["z_count"] = sizes.get("Z", 1)
            meta = getattr(r, "metadata", None)
            if meta is None:
                ps, tp = read_calibration_from_path(path)
                out["pixel_size_um"] = ps if ps > 0 else None
                out["time_per_frame_min"] = tp if tp > 0 else None
                _nd2_reconcile_dims_from_shape(r, out)
                return
            # Contents (may disagree with actual nd2 shape / axis order)
            contents = getattr(meta, "contents", None)
            if contents:
                out["frame_count"] = getattr(contents, "frameCount", None)
                out["channel_count"] = getattr(contents, "channelCount", None)
            # Pixel size
            if hasattr(meta, "channels") and meta.channels:
                ch0 = meta.channels[0]
                vol = getattr(getattr(ch0, "volume", None), "axesCalibration", None)
                if vol and len(vol) >= 2:
                    out["pixel_size_um"] = float(vol[0])
                # Channel names and wavelengths; mark phase contrast so we don't show bogus Ex/Em
                names = []
                wavelengths = []
                for ch in meta.channels:
                    cmeta = getattr(ch, "channel", None)
                    name = getattr(cmeta, "name", None) if cmeta else None
                    names.append(name or f"Channel_{len(names)+1}")
                    wl = {}
                    if cmeta:
                        wl["name"] = getattr(cmeta, "name", None) or names[-1]
                        mic = getattr(ch, "microscope", None)
                        flags = getattr(mic, "modalityFlags", None) if mic else None
                        is_phase = (
                            (flags and "phaseContrast" in (flags if isinstance(flags, (list, tuple)) else []))
                            or (name and "phase" in str(name).lower())
                        )
                        if is_phase:
                            wl["phase_contrast"] = True
                        else:
                            ex = getattr(cmeta, "excitationLambdaNm", None)
                            em = getattr(cmeta, "emissionLambdaNm", None)
                            if ex is not None:
                                wl["excitation_nm"] = float(ex)
                            if em is not None:
                                wl["emission_nm"] = float(em)
                    wavelengths.append(wl)
                out["channel_names"] = names
                out["channel_wavelengths"] = wavelengths
                # Actual bits per pixel in file (Volume.bitsPerComponentInMemory); avoids misleading "12-bit" from capturing
                vol = getattr(ch0, "volume", None)
                if vol is not None:
                    bpc = getattr(vol, "bitsPerComponentInMemory", None)
                    if bpc is not None:
                        out["bits_per_component"] = int(bpc)
                # Microscope (first channel)
                mic = getattr(ch0, "microscope", None)
                if mic:
                    out["objective_name"] = getattr(mic, "objectiveName", None)
                    mag = getattr(mic, "objectiveMagnification", None)
                    out["objective_magnification"] = float(mag) if mag is not None else None
                    zoom = getattr(mic, "zoomMagnification", None)
                    out["zoom_magnification"] = float(zoom) if zoom is not None else None
            # Time per frame: experiment loops then frame_metadata fallback
            tp = None
            if hasattr(r, "experiment") and r.experiment and getattr(r.experiment, "loops", None):
                for loop in r.experiment.loops:
                    params = getattr(loop, "parameters", None)
                    if not params:
                        continue
                    period_ms = getattr(params, "periodMs", None)
                    if period_ms is not None and float(period_ms) > 0:
                        tp = float(period_ms) / 60000.0
                        break
                    periods = getattr(params, "periods", None)
                    if periods and len(periods) > 0:
                        p0 = periods[0]
                        period_ms = getattr(p0, "periodMs", None)
                        if period_ms is not None and float(period_ms) > 0:
                            tp = float(period_ms) / 60000.0
                            break
                    period_sec = getattr(params, "period", None)
                    if period_sec is not None and float(period_sec) > 0:
                        tp = float(period_sec) / 60.0
                        break
            if (tp is None or tp <= 0) and hasattr(r, "frame_metadata"):
                tp = _nd2_frame_interval_from_frame_metadata(r)
            out["time_per_frame_min"] = tp if tp and tp > 0 else None
            # Exposure from events()
            if hasattr(r, "events"):
                try:
                    ev = r.events()
                    if ev and len(ev) > 0 and isinstance(ev[0], dict) and "Exposure Time [ms]" in ev[0]:
                        out["exposure_ms"] = ev[0]["Exposure Time [ms]"]
                except Exception:
                    pass
            # text_info: optics, capturing, and parse description for per-channel exposure/filter
            if hasattr(r, "text_info"):
                try:
                    ti = r.text_info
                    if isinstance(ti, dict):
                        out["optics"] = ti.get("optics")
                        out["capturing"] = ti.get("capturing")
                        desc = ti.get("description")
                        if isinstance(desc, str):
                            if "Filter" in desc:
                                idx = desc.find("Filter")
                                out["description_filter_snippet"] = desc[idx : idx + 80].split("\n")[0].strip()
                            # Per-channel: parse "Plane #N:" blocks for Exposure and Filter
                            per_ch = _parse_nd2_description_per_channel(desc)
                            if per_ch:
                                out["channel_exposure_ms"] = [c.get("exposure_ms") for c in per_ch]
                                out["channel_filter_snippet"] = [c.get("filter_snippet") for c in per_ch]
                            # DIA % (phase condenser intensity) when present in description
                            dia_pct = _parse_nd2_dia_percent(desc)
                            if dia_pct is not None:
                                out["dia_percent"] = dia_pct
                except Exception:
                    pass
            # Prefer r.shape axis sizes over contents.frameCount (fixes ND2 where metadata counts total planes vs T)
            _nd2_reconcile_dims_from_shape(r, out)
    except Exception:
        pass


def _nd2_rel_time_ms(frame_meta: Any) -> Optional[float]:
    """Extract relativeTimeMs from one frame_metadata result (object or dict)."""
    if frame_meta is None:
        return None
    if isinstance(frame_meta, dict):
        chs = frame_meta.get("channels")
        if chs and len(chs) > 0:
            t = chs[0].get("time") if isinstance(chs[0], dict) else getattr(chs[0], "time", None)
            if isinstance(t, dict):
                return t.get("relativeTimeMs")
            if t is not None:
                return getattr(t, "relativeTimeMs", None)
        return None
    ch0 = getattr(frame_meta, "channels", None)
    if not ch0 or len(ch0) == 0:
        return None
    t = getattr(ch0[0], "time", None)
    return getattr(t, "relativeTimeMs", None) if t is not None else None


def _nd2_frame_interval_from_frame_metadata(nd2_reader: Any) -> Optional[float]:
    """Compute time per frame (min) from frame_metadata timestamps. Supports object and dict return types.
    Tries frame 0 vs 1 first; then frame 0 vs last if needed."""
    try:
        sizes = getattr(nd2_reader, "sizes", None)
        n = sizes.get("T", 1) if sizes else 1
        if n < 2:
            return None
        r0 = _nd2_rel_time_ms(nd2_reader.frame_metadata(0))
        r1 = _nd2_rel_time_ms(nd2_reader.frame_metadata(1))
        if r0 is not None and r1 is not None:
            delta_ms = float(r1) - float(r0)
            if delta_ms > 0:
                return delta_ms / 60000.0
        # Fallback: first and last frame
        r_last = _nd2_rel_time_ms(nd2_reader.frame_metadata(int(n) - 1))
        if r0 is not None and r_last is not None and n > 1:
            delta_ms = float(r_last) - float(r0)
            if delta_ms > 0:
                return (delta_ms / (int(n) - 1)) / 60000.0
    except Exception:
        pass
    return None


def _parse_nd2_description_per_channel(description: str) -> list[dict[str, Any]]:
    """Parse ND2 text_info['description'] for per-plane (per-channel) exposure and filter.
    Returns list of dicts: [{"exposure_ms": float|None, "filter_snippet": str|None}, ...] by channel index."""
    import re
    result: list[dict[str, Any]] = []
    if not description or not isinstance(description, str):
        return result
    # Split by "Plane #1:", "Plane #2:", etc. (split includes capturing group values in list)
    parts = re.split(r"\n\s*Plane\s*#\s*(\d+)\s*:", description, flags=re.IGNORECASE)
    # parts = [preamble, "1", block1, "2", block2, ...]
    i = 1
    while i + 1 < len(parts):
        block = parts[i + 1]
        exp_ms = None
        filter_snippet = None
        m = re.search(r"Exposure\s*:\s*(\d+(?:\.\d+)?)\s*ms", block, re.IGNORECASE)
        if m:
            try:
                exp_ms = float(m.group(1))
            except (ValueError, IndexError):
                pass
        m = re.search(r"Filter(?:Changer)?\s*[^(]*\([^)]+\)[^\n]*", block)
        if m:
            filter_snippet = m.group(0).strip()[:100]
        result.append({"exposure_ms": exp_ms, "filter_snippet": filter_snippet})
        i += 2
    return result


def _parse_nd2_dia_percent(description: str) -> Optional[int]:
    """Parse ND2 description for DIA/phase condenser percentage (e.g. DIA: 50%, 50% DIA, Phase 25%)."""
    import re
    if not description or not isinstance(description, str):
        return None
    # Patterns: "DIA: 50%", "DIA 50%", "50% DIA", "DIA 50", "Phase: 25%", "Condenser: 50%"
    patterns = [
        r"DIA\s*[:\s]*(\d+)\s*%?",
        r"(\d+)\s*%\s*DIA",
        r"Phase\s*[:\s]*(\d+)\s*%",
        r"Condenser\s*[:\s]*(\d+)\s*%",
        r"(\d+)\s*%\s*Phase",
    ]
    for pat in patterns:
        m = re.search(pat, description, re.IGNORECASE)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    return v
            except (ValueError, IndexError):
                pass
    return None


def build_acquisition_info_string(
    meta: dict[str, Any],
    channel_name: Optional[str] = None,
    source_name: Optional[str] = None,
) -> str:
    """Build a short acquisition info string for embedding in output TIFF (ImageDescription/Info).
    Includes objective, pixel size, time per frame, and optionally channel name and source file."""
    lines: list[str] = []
    if source_name:
        lines.append(f"Source: {source_name}")
    obj_name = meta.get("objective_name")
    obj_mag = meta.get("objective_magnification")
    if obj_name or obj_mag is not None:
        parts = [str(obj_name)] if obj_name else []
        if obj_mag is not None:
            parts.append(f"{float(obj_mag):.1f}x")
        lines.append("Objective: " + " ".join(parts))
    ps = meta.get("pixel_size_um")
    if ps is not None and float(ps) > 0:
        lines.append(f"Pixel size: {float(ps):.4f} µm")
    tp = meta.get("time_per_frame_min")
    if tp is not None and float(tp) > 0:
        lines.append(f"Time per frame: {float(tp) * 60:.4f} s")
    if channel_name:
        lines.append(f"Channel: {channel_name}")
    return "\n".join(lines) if lines else ""


def format_metadata_for_display(meta: dict[str, Any]) -> str:
    """Format read_metadata_from_path result as a string for the conversion tab metadata panel.
    Shows per-channel exposure and filter when available; otherwise global exposure/filter with clear labels."""
    lines = ["Metadata:"]
    ps = meta.get("pixel_size_um")
    if ps is not None and float(ps) > 0:
        lines.append(f"  Pixel Size: {float(ps):.4f} µm")
    else:
        lines.append("  Pixel Size: Not found in metadata")
    tp = meta.get("time_per_frame_min")
    if tp is not None and float(tp) > 0:
        lines.append(f"  Time per frame: {float(tp):.6f} min ({float(tp) * 60:.4f} s)")
    else:
        lines.append("  Time per frame: Not found in metadata")
    fc = meta.get("frame_count")
    if fc is not None:
        lines.append(f"  Frame Count: {fc}")
    cc = meta.get("channel_count")
    if cc is not None:
        lines.append(f"  Channel Count: {cc}")
    ch_src = meta.get("channel_names_source")
    ch_names = meta.get("channel_names") or []
    if ch_names:
        lines.append(f"  Channel names: {', '.join(str(n) for n in ch_names)}")
        if ch_src:
            lines.append(
                f"  Channel names source: {ch_src}"
                + (" (edit names in Channel options before convert)" if ch_src in ("luts", "default") else "")
            )
    zc = meta.get("z_count")
    if zc is not None and int(zc) > 1:
        lines.append(f"  Z slices: {zc}")
    ao = meta.get("nd2_axis_order")
    if ao:
        lines.append(f"  ND2 axis order: {', '.join(ao)}")
        if "T" not in ao:
            lines.append(
                "  Note: T axis omitted (single timepoint). Frame count above is timepoints, not NIS sequence length."
            )

    # Per-channel block: name, wavelengths, exposure (per-channel or —), filter (per-channel or —)
    channel_exposure = meta.get("channel_exposure_ms") or []
    channel_filter = meta.get("channel_filter_snippet") or []
    wavelengths = meta.get("channel_wavelengths") or []
    lines.append("  --- Channels ---")
    hw_filter = meta.get("description_filter_snippet") or ""
    dia_percent = meta.get("dia_percent")
    for i, wl in enumerate(wavelengths):
        name = wl.get("name") or f"Channel {i+1}"
        is_phase = wl.get("phase_contrast") is True
        filt_str = (
            (channel_filter[i] or "").strip()
            if i < len(channel_filter) and channel_filter[i]
            else ""
        )
        wl_parts = []
        if is_phase:
            # DIA when explicit in filter text, or assume DIA when channel name indicates phase (e.g. Phase60X)
            has_dia = (
                "DIA" in hw_filter
                or (filt_str and "DIA" in filt_str)
                or ("phase" in (name or "").lower())
            )
            if has_dia and dia_percent is not None:
                label = f"Phase contrast (DIA, {dia_percent}%)"
            elif has_dia:
                label = "Phase contrast (DIA)"
            elif dia_percent is not None:
                label = f"Phase contrast ({dia_percent}%)"
            else:
                label = "Phase contrast"
            wl_parts.append(label)
        else:
            if wl.get("excitation_nm") is not None:
                wl_parts.append(f"Ex {float(wl['excitation_nm']):.0f} nm")
            if wl.get("emission_nm") is not None:
                wl_parts.append(f"Em {float(wl['emission_nm']):.0f} nm")
        if i < len(channel_exposure) and channel_exposure[i] is not None:
            exp_str = f"{channel_exposure[i]} ms"
        elif channel_exposure:
            exp_str = "—"
        elif meta.get("exposure_ms") is not None:
            exp_str = "(see acquisition exposure below)"
        else:
            exp_str = "—"
        lines.append(f"  Channel {i+1}: {name}")
        if wl_parts:
            lines.append(f"    Wavelengths: {', '.join(wl_parts)}")
        lines.append(f"    Exposure: {exp_str}")
        if filt_str:
            lines.append(f"    Filter: {filt_str}")

    # Global acquisition / hardware (only when not already shown per-channel)
    lines.append("  --- Acquisition / hardware ---")
    obj_name = meta.get("objective_name")
    obj_mag = meta.get("objective_magnification")
    zoom = meta.get("zoom_magnification")
    if obj_name or obj_mag is not None:
        parts = []
        if obj_name:
            parts.append(str(obj_name))
        if obj_mag is not None:
            parts.append(f"{float(obj_mag):.1f}x")
        if zoom is not None and float(zoom) != 1.0:
            parts.append(f"zoom {float(zoom):.1f}x")
        lines.append(f"  Objective: {', '.join(parts)}")
    ex_ms = meta.get("exposure_ms")
    if ex_ms is not None:
        lines.append(f"  Exposure (acquisition): {ex_ms} ms")
    bpc = meta.get("bits_per_component")
    if bpc is not None:
        lines.append(f"  Bits per pixel: {bpc}")
    if meta.get("optics"):
        lines.append(f"  Optics: {meta['optics']}")
    cap = meta.get("capturing")
    if cap:
        for ln in (cap.split("\n")[:5] if isinstance(cap, str) else [str(cap)[:80]]):
            ln = ln.strip()
            if ln and "Bit Depth:" not in ln:
                lines.append(f"  Capturing: {ln}")
    if meta.get("description_filter_snippet"):
        lines.append(f"  Filter (hardware): {meta['description_filter_snippet']}")
    return "\n".join(lines)


def read_calibration_from_path(image_path: Optional[Path]) -> tuple[float, float]:
    """Read pixel size (µm/px) and time per frame (min) from a single file.
    Returns (pixel_size_um, time_per_frame_min). Use 0.0 for unknown.
    Supports .tif/.tiff (ImageJ metadata), .nd2, and .lif/.czi (via BioIO when available)."""
    pixel_size = 0.0
    time_per_frame = 0.0
    if not image_path or not image_path.is_file():
        return (pixel_size, time_per_frame)
    suffix = image_path.suffix.lower()
    try:
        if suffix in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(image_path) as tif:
                if tif.pages:
                    page0 = tif.pages[0]
                    ij = getattr(tif, "imagej_metadata", None) or {}
                    pixel_size = _pixel_size_um_from_tiff_page(
                        page0, ij if isinstance(ij, dict) else {}
                    )
                if hasattr(tif, "imagej_metadata") and tif.imagej_metadata:
                    m = tif.imagej_metadata
                    if "finterval" in m:
                        val = float(m["finterval"])
                        time_per_frame = (val / 60.0) if val > 0 else 0.0
                    elif "interval" in m:
                        val = float(m["interval"])
                        time_per_frame = (val / 60.0) if val > 0 else 0.0
                # OME-TIFF: PhysicalSizeX/Y in µm
                if pixel_size <= 0 and hasattr(tif, "ome_metadata") and tif.ome_metadata:
                    try:
                        import re
                        xml = tif.ome_metadata if isinstance(tif.ome_metadata, str) else (tif.ome_metadata.decode("utf-8", errors="ignore") if isinstance(tif.ome_metadata, bytes) else "")
                        if xml:
                            m = re.search(r'PhysicalSizeX="([^"]+)"', xml)
                            if m:
                                pixel_size = float(m.group(1))
                            if pixel_size <= 0:
                                m = re.search(r'PhysicalSizeY="([^"]+)"', xml)
                                if m:
                                    pixel_size = float(m.group(1))
                    except (TypeError, ValueError, AttributeError):
                        pass
        elif suffix in (".lif", ".czi"):
            try:
                from bactomate.modules.bioio_reader import get_info as bioio_get_info
                info = bioio_get_info(image_path)
                if info and info.get("pixel_size") is not None:
                    try:
                        pixel_size = float(info["pixel_size"])
                        if pixel_size > 0:
                            return (pixel_size, time_per_frame)
                    except (TypeError, ValueError):
                        pass
            except Exception:
                pass
            return (pixel_size, time_per_frame)
        elif suffix == ".nd2":
            try:
                import nd2
                with nd2.ND2File(image_path) as nd2_reader:
                    if hasattr(nd2_reader, "metadata") and nd2_reader.metadata:
                        meta = nd2_reader.metadata
                        if hasattr(meta, "channels") and meta.channels:
                            first_channel = meta.channels[0]
                            if hasattr(first_channel, "volume") and first_channel.volume:
                                vol = first_channel.volume
                                if hasattr(vol, "axesCalibration") and vol.axesCalibration and len(vol.axesCalibration) >= 2:
                                    pixel_size = float(vol.axesCalibration[0])
                    # Frame interval: nd2 uses periodMs (ms) in TimeLoopParams, or NETimeLoopParams.periods[].periodMs
                    if hasattr(nd2_reader, "experiment") and nd2_reader.experiment and hasattr(nd2_reader.experiment, "loops"):
                        for loop in nd2_reader.experiment.loops:
                            params = getattr(loop, "parameters", None)
                            if params is None:
                                continue
                            period_ms = getattr(params, "periodMs", None)
                            if period_ms is not None and float(period_ms) > 0:
                                time_per_frame = float(period_ms) / 60000.0  # ms -> minutes
                                break
                            periods = getattr(params, "periods", None)
                            if periods and len(periods) > 0:
                                p0 = periods[0]
                                period_ms = getattr(p0, "periodMs", None)
                                if period_ms is not None and float(period_ms) > 0:
                                    time_per_frame = float(period_ms) / 60000.0
                                    break
                            period_sec = getattr(params, "period", None)
                            if period_sec is not None and float(period_sec) > 0:
                                time_per_frame = float(period_sec) / 60.0
                                break
                    # Fallback: compute from frame_metadata() timestamps (relativeTimeMs)
                    if time_per_frame <= 0 and hasattr(nd2_reader, "frame_metadata"):
                        tp_from_fm = _nd2_frame_interval_from_frame_metadata(nd2_reader)
                        if tp_from_fm and tp_from_fm > 0:
                            time_per_frame = tp_from_fm
            except Exception:
                pass
    except Exception:
        pass
    return (pixel_size, time_per_frame)


def _looks_like_mask_path(image_path: Path) -> bool:
    stem = image_path.stem.lower()
    return "_masks" in stem or stem.endswith("_mask") or "_mask_filtered" in stem


def resolve_calibration_for_mask(
    image_path: Optional[Path],
) -> tuple[float, float, Optional[Path], str]:
    """Resolve calibration for a mask or image path.

    Returns (pixel_size_um, time_per_frame_min, source_path, source_kind).
    source_kind is one of: ``original``, ``mask``, ``image``, ``none``.

    For mask TIFFs, the original (phase) stack is preferred over mask metadata so
    segmentation outputs cannot override the acquisition pixel size.
    """
    if not image_path or not image_path.is_file():
        return (0.0, 0.0, None, "none")
    if _looks_like_mask_path(image_path):
        original = resolve_original_for_mask(image_path)
        if original and original.is_file():
            ps, tp = read_calibration_from_path(original)
            if ps > 0 or tp > 0:
                return (ps, tp, original, "original")
        ps, tp = read_calibration_from_path(image_path)
        if ps > 0 or tp > 0:
            return (ps, tp, image_path, "mask")
        return (0.0, 0.0, original, "none")
    ps, tp = read_calibration_from_path(image_path)
    if ps > 0 or tp > 0:
        return (ps, tp, image_path, "image")
    return (0.0, 0.0, None, "none")


def read_calibration_from_path_or_original(image_path: Optional[Path]) -> tuple[float, float]:
    """Read pixel size and time per frame; for masks, prefer the resolved original phase TIFF."""
    ps, tp, _src, _kind = resolve_calibration_for_mask(image_path)
    return (ps, tp)


def build_metadata_dict(
    pixel_size_um: Optional[float] = None,
    time_per_frame_min: Optional[float] = None,
    axes: str = "TYX",
    images: Optional[int] = None,
    frames: Optional[int] = None,
    acquisition_metadata: Optional[dict[str, Any]] = None,
) -> dict:
    """Build ImageJ-style metadata dict for tifffile.imwrite(..., metadata=..., imagej=True).
    finterval is written in seconds for ImageJ compatibility.
    If acquisition_metadata is provided (from read_metadata_from_path), pixel_size_um and time_per_frame_min
    are taken from it when not passed explicitly."""
    if acquisition_metadata:
        if pixel_size_um is None:
            v = acquisition_metadata.get("pixel_size_um")
            pixel_size_um = float(v) if v is not None and float(v) > 0 else None
        if time_per_frame_min is None:
            v = acquisition_metadata.get("time_per_frame_min")
            time_per_frame_min = float(v) if v is not None and float(v) > 0 else None
    metadata = {
        "axes": axes,
        "ImageJ": "1.53c",
        "unit": "um",
    }
    if images is not None:
        metadata["images"] = images
    if frames is not None:
        metadata["frames"] = frames
        metadata["slices"] = 1
        metadata["loop"] = False
        metadata["mode"] = "grayscale"
    if pixel_size_um is not None and pixel_size_um > 0:
        metadata["pixelSize"] = pixel_size_um
    if time_per_frame_min is not None and time_per_frame_min > 0:
        # ImageJ expects finterval in seconds
        metadata["finterval"] = time_per_frame_min * 60.0
    return metadata


def build_metadata_dict_for_mask_from_source(
    source_image_path: Optional[Path],
    num_frames: int,
    axes: str = "TYX",
) -> dict:
    """Build ImageJ-style metadata for a mask TIFF from the source image path.
    Use when writing *_masks.tif so pixel size and time-per-frame are preserved.
    If source_image_path is None or missing, returns minimal metadata with frames/images only."""
    if not source_image_path or not source_image_path.is_file():
        return build_metadata_dict(frames=num_frames, images=num_frames, axes=axes)
    acquisition = read_metadata_from_path(source_image_path)
    return build_metadata_dict(
        acquisition_metadata=acquisition,
        frames=num_frames,
        images=num_frames,
        axes=axes,
    )


def _candidate_original_stems(mask_stem: str) -> list[str]:
    stem = (
        mask_stem.replace("_masks", "")
        .replace("_mask_filtered", "")
        .replace("_segmentation_omni", "")
        .replace("_segmentation_unet", "")
    )
    stems = [stem]
    if "_" in stem:
        stems.append(stem.rsplit("_", 1)[0])
    out: list[str] = []
    for s in stems:
        if s and s not in out:
            out.append(s)
    return out


def _base_dir_for_mask(mask_path: Path) -> Path:
    """Sample-level directory (sibling of ``_segmentation_*`` / ``Tif_*``). Mirrors analysis_paths."""
    base = mask_path.parent
    if base.name == "filtered_masks" and "_segmentation" in base.parent.name:
        base = base.parent
    if base.name.startswith("Tif_") or "_segmentation" in base.name or base.name == "_analysis":
        base = base.parent
    return base


def _search_dirs_for_original(mask_path: Path) -> list[Path]:
    """Directories to search for the phase TIFF (mask folder → sample base → parents)."""
    dirs: list[Path] = [mask_path.parent]
    base = _base_dir_for_mask(mask_path)
    dirs.extend([base, base.parent, base.parent.parent])
    # Walk up through segmentation folders (e.g. Tif_files/_segmentation_* → swimming/).
    p = mask_path.parent
    for _ in range(4):
        if p.parent != p:
            p = p.parent
            dirs.append(p)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for d in dirs:
        try:
            key = d.resolve()
        except OSError:
            key = d
        if not d.is_dir() or key in seen:
            continue
        seen.add(key)
        ordered.append(d)
    return ordered


def _find_original_in_dir(search_dir: Path, stems: list[str]) -> Optional[Path]:
    for stem in stems:
        for f in search_dir.iterdir():
            if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                continue
            if "_fi" in f.stem.lower() or "_masks" in f.stem.lower():
                continue
            if f.stem == stem:
                return f
    for sub in search_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        for stem in stems:
            for f in sub.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                    continue
                if "_fi" in f.stem.lower() or "_masks" in f.stem.lower():
                    continue
                if f.stem == stem:
                    return f
    return None


def resolve_original_for_mask(mask_path: Optional[Path]) -> Optional[Path]:
    """Find the original (phase) TIFF for a mask file by stem and sample layout."""
    if not mask_path or not mask_path.is_file():
        return None
    stems = _candidate_original_stems(mask_path.stem)
    for search_dir in _search_dirs_for_original(mask_path):
        found = _find_original_in_dir(search_dir, stems)
        if found is not None:
            return found
    return None


def _resolve_original_for_mask(mask_path: Optional[Path]) -> Optional[Path]:
    """Backward-compatible alias for :func:`resolve_original_for_mask`."""
    return resolve_original_for_mask(mask_path)


def _resolve_mask_from_trajectories_path(traj_path: Path) -> Optional[Path]:
    """From a *_trajectories.tif path, find the corresponding mask TIFF."""
    if not traj_path.is_file():
        return None
    out_dir = traj_path.parent
    stem = traj_path.stem.replace("_trajectories", "")
    mask_stem = stem
    search_bases = [out_dir.parent.parent, out_dir.parent]
    for base in search_bases:
        if not base.is_dir():
            continue
        for f in base.iterdir():
            if f.is_file() and f.suffix.lower() in (".tif", ".tiff") and f.stem == mask_stem:
                return f
        seg = base / "_segmentation"
        if seg.is_dir():
            for f in seg.iterdir():
                if f.is_file() and f.suffix.lower() in (".tif", ".tiff") and f.stem == mask_stem:
                    return f
        for sub in base.iterdir():
            if not sub.is_dir():
                continue
            for f in sub.iterdir():
                if f.is_file() and f.suffix.lower() in (".tif", ".tiff") and f.stem == mask_stem:
                    return f
        for pattern in (f"*{mask_stem}.tif", f"*{mask_stem}.tiff"):
            for f in base.glob(pattern):
                if f.is_file():
                    return f
            if seg.is_dir():
                for f in seg.glob(pattern):
                    if f.is_file():
                        return f
    return None


def resolve_calibration_from_trajectories_path(traj_path: Optional[Path]) -> tuple[float, float]:
    """Given a *_trajectories.tif path, resolve the pipeline (traj → mask → original)
    and return (pixel_size_um, time_per_frame_min) from the first file that has metadata.
    Order: original first (source of truth), then mask, then trajectory TIF."""
    if not traj_path or not traj_path.is_file():
        return (0.0, 0.0)
    mask_path = _resolve_mask_from_trajectories_path(traj_path)
    if mask_path is not None:
        ps, tp, _src, _kind = resolve_calibration_for_mask(mask_path)
        if ps > 0 or tp > 0:
            return (ps, tp)
    original_path = resolve_original_for_mask(mask_path) if mask_path else None
    for path in (original_path, mask_path, traj_path):
        if path and path.is_file():
            ps, tp = read_calibration_from_path(path)
            if ps > 0 or tp > 0:
                return (ps, tp)
    return (0.0, 0.0)
