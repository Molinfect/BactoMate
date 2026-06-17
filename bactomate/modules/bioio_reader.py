"""
Unified microscopy image reading via BioIO (https://github.com/bioio-devs/bioio).

Provides a single API for ND2, LIF, CZI, and TIFF: get_info() and load_data().
If BioIO or the required plugin is not available, callers should fall back to
format-specific readers (aicsimageio, czifile, readlif, nd2, tifffile).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

# Lazy import of bioio in functions so missing optional deps don't break import

_BIOIO_AVAILABLE: Optional[bool] = None


def is_available() -> bool:
    """Return True if bioio and at least one reader plugin can be imported."""
    global _BIOIO_AVAILABLE
    if _BIOIO_AVAILABLE is not None:
        return _BIOIO_AVAILABLE
    try:
        from bioio import BioImage  # noqa: F401
        _BIOIO_AVAILABLE = True
    except ImportError:
        _BIOIO_AVAILABLE = False
    return _BIOIO_AVAILABLE


def _suffix_to_file_type(suffix: str) -> Optional[str]:
    """Map file suffix to BactoMate file_type (nd2, lif, czi, tiff)."""
    s = suffix.lower()
    if s == ".nd2":
        return "nd2"
    if s in (".lif",):
        return "lif"
    if s in (".czi",):
        return "czi"
    if s in (".tif", ".tiff", ".ome.tiff", ".ome.tif"):
        return "tiff"
    return None


def _tiff_has_ome_metadata(path: Path) -> bool:
    """True when TIFF ImageDescription or ome_metadata contains OME XML."""
    try:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            ome = getattr(tif, "ome_metadata", None)
            if ome:
                text = ome if isinstance(ome, str) else ome.decode("utf-8", errors="ignore")
                if "<OME" in text:
                    return True
            if not tif.pages:
                return False
            desc = tif.pages[0].description or ""
            if not isinstance(desc, str):
                desc = str(desc)
            return "<OME" in desc
    except Exception:
        return False


def _open_bioimage(path: Path):
    """Open BioImage with TIFF reader order that avoids OME parse warnings on plain TIFF."""
    from bioio import BioImage

    file_type = _suffix_to_file_type(path.suffix)
    if file_type != "tiff":
        return BioImage(path)

    has_ome = _tiff_has_ome_metadata(path)
    readers: list[type] = []
    if has_ome:
        try:
            from bioio_ome_tiff import Reader as OmeTiffReader

            readers.append(OmeTiffReader)
        except ImportError:
            pass
    else:
        try:
            from bioio_tifffile import Reader as TiffReader

            readers.append(TiffReader)
        except ImportError:
            return None
        try:
            from bioio_ome_tiff import Reader as OmeTiffReader

            readers.append(OmeTiffReader)
        except ImportError:
            pass

    if readers:
        return BioImage(path, reader=readers)
    return None if not has_ome else BioImage(path)


def get_info(
    path: Path | str,
    scene_index: int = 0,
) -> Optional[dict[str, Any]]:
    """
    Read file metadata using BioIO. Returns a dict compatible with BactoMate's
    file_conversion tab: file_type, num_time, num_channels, num_z, channel_names,
    pixel_size, scenes (list of scene ids), or None if BioIO cannot read the file.
    """
    if not is_available():
        return None
    path = Path(path)
    if not path.is_file():
        return None
    file_type = _suffix_to_file_type(path.suffix)
    if file_type is None:
        return None
    try:
        import numpy as np

        img = _open_bioimage(path)
        if img is None:
            return None
        try:
            scenes = list(img.scenes) if hasattr(img, "scenes") and img.scenes else ["0"]
            if scene_index != 0 and scenes and scene_index < len(scenes):
                img.set_scene(scenes[scene_index])
            elif scene_index != 0 and hasattr(img, "set_scene"):
                img.set_scene(scene_index)

            dims = img.dims
            order = dims.order if hasattr(dims, "order") else str(dims)
            shape = img.shape
            if not isinstance(shape, (tuple, list)):
                shape = tuple(shape) if hasattr(shape, "__iter__") else (shape,)

            num_time = int(shape[order.index("T")]) if "T" in order else 1
            num_channels = int(shape[order.index("C")]) if "C" in order else 1
            num_z = int(shape[order.index("Z")]) if "Z" in order else 1

            channel_names = []
            if hasattr(img, "channel_names") and img.channel_names:
                channel_names = list(img.channel_names)
            if len(channel_names) != num_channels:
                channel_names = [f"Channel_{i+1}" for i in range(num_channels)]

            pixel_size = None
            if hasattr(img, "physical_pixel_sizes") and img.physical_pixel_sizes is not None:
                pps = img.physical_pixel_sizes
                if hasattr(pps, "X") and pps.X is not None:
                    try:
                        pixel_size = float(pps.X)
                    except (TypeError, ValueError):
                        pass
                if pixel_size is None and hasattr(pps, "Y") and pps.Y is not None:
                    try:
                        pixel_size = float(pps.Y)
                    except (TypeError, ValueError):
                        pass

            return {
                "file_type": file_type,
                "num_time": num_time,
                "num_channels": num_channels,
                "num_z": num_z,
                "channel_names": channel_names,
                "pixel_size": pixel_size,
                "scenes": scenes,
                "dims_order": order,
                "shape": shape,
            }
        finally:
            if hasattr(img, "close"):
                try:
                    img.close()
                except Exception:
                    pass
    except Exception:
        return None


def load_data(
    path: Path | str,
    scene_index: int = 0,
    max_frames: Optional[int] = None,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    use_first_z_only: bool = True,
    max_z_projection: bool = False,
) -> Optional[Any]:
    """
    Load image data as numpy array (T, C, Y, X) using BioIO.
    - max_frames: when set (and frame_start/frame_end not set), take first max_frames time points.
    - frame_start, frame_end: optional 0-based range [frame_start, frame_end). If frame_start is set,
      only that range is returned (frame_end defaults to frame_start+1 for single-frame extraction).
    - use_first_z_only: if True and Z>1, take only Z=0 (unless max_z_projection).
    - max_z_projection: if True and Z>1, take max over Z.
    Returns numpy array or None on failure.
    """
    if not is_available():
        return None
    path = Path(path)
    if not path.is_file():
        return None
    if _suffix_to_file_type(path.suffix) is None:
        return None
    try:
        import numpy as np

        img = _open_bioimage(path)
        if img is None:
            return None
        try:
            scenes = list(img.scenes) if hasattr(img, "scenes") and img.scenes else ["0"]
            if scene_index != 0 and scenes and scene_index < len(scenes):
                img.set_scene(scenes[scene_index])
            elif scene_index != 0 and hasattr(img, "set_scene"):
                img.set_scene(scene_index)

            # .data is 5D TCZYX (numpy) per BioIO spec
            data = np.asarray(img.data)
            if data.size == 0:
                return None

            order = img.dims.order if hasattr(img.dims, "order") else str(img.dims)
            t_idx = order.index("T") if "T" in order else 0
            z_idx = order.index("Z") if "Z" in order else None

            # Limit time frames: explicit range [frame_start, frame_end) or first max_frames
            num_t = data.shape[t_idx]
            if frame_start is not None:
                end = frame_end if frame_end is not None else frame_start + 1
                start = max(0, min(frame_start, num_t))
                end = min(end, num_t)
                if start < end:
                    slices = [slice(None)] * data.ndim
                    slices[t_idx] = slice(start, end)
                    data = data[tuple(slices)]
                else:
                    return None
            elif max_frames is not None and num_t > max_frames:
                slices = [slice(None)] * data.ndim
                slices[t_idx] = slice(0, max_frames)
                data = data[tuple(slices)]

            # Collapse Z: first slice or max projection
            if z_idx is not None and data.shape[z_idx] > 1:
                if max_z_projection:
                    data = np.max(data, axis=z_idx).astype(data.dtype)
                elif use_first_z_only:
                    slices = [slice(None)] * data.ndim
                    slices[z_idx] = 0
                    data = data[tuple(slices)]

            # Result is (T, C, Y, X) or (T, Y, X); ensure 4D
            if data.ndim == 3:
                data = data[:, np.newaxis, :, :]
            elif data.ndim == 2:
                data = data[np.newaxis, np.newaxis, :, :]
            return np.ascontiguousarray(data)
        finally:
            if hasattr(img, "close"):
                try:
                    img.close()
                except Exception:
                    pass
    except Exception:
        return None


def get_available_plugins() -> list[str]:
    """Return list of format names that BioIO can read (based on installed plugins)."""
    if not is_available():
        return []
    # BioIO doesn't expose a simple "list plugins" API; we report based on import
    result = []
    try:
        from bioio import BioImage
        # Try to instantiate a reader for each type; that would require files.
        # Instead, check for plugin packages (optional imports)
        for pkg, name in [
            ("bioio_ome_tiff", "OME-TIFF"),
            ("bioio_tifffile", "TIFF"),
            ("bioio_czi", "CZI"),
            ("bioio_lif", "LIF"),
            ("bioio_nd2", "ND2"),
        ]:
            try:
                __import__(pkg)
                result.append(name)
            except ImportError:
                pass
    except Exception:
        pass
    return result
