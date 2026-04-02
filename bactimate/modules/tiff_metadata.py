"""
Central TIFF (and ND2) calibration and acquisition metadata: read, format, and build for write.
Single source of truth for: pixel size, time per frame, channel names/wavelengths, objective,
exposure, optics/capturing. Used by file conversion (display + write), tracking, workers.
All time-per-frame values are in minutes when returned or passed to build_metadata_dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


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
        from bactimate.modules.bioio_reader import get_info as bioio_get_info
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
            # ImageJ / dimensions
            meta = getattr(tif, "imagej_metadata", None) or {}
            frames = meta.get("frames") or meta.get("images") or meta.get("slices")
            if frames is not None:
                out["frame_count"] = int(frames)
            channels = meta.get("channels")
            if channels is not None:
                out["channel_count"] = int(channels)
            # Shape from series (e.g. (T, Y, X) or (T, C, Z, Y, X)). Skip series access when we already
            # have frame_count from ImageJ metadata to avoid tifffile "axes do not match shape" on old files.
            try:
                if out["frame_count"] is None:
                    series = tif.series[0] if getattr(tif, "series", None) and len(tif.series) > 0 else None
                    if series is not None:
                        shape = getattr(series, "shape", None)
                        axes = getattr(series, "axes", None) or ""
                        if shape and axes:
                            if "T" in axes:
                                idx = axes.find("T")
                                if idx is not None and idx < len(shape):
                                    out["frame_count"] = int(shape[idx])
                            if out["channel_count"] is None and "C" in axes:
                                idx = axes.find("C")
                                if idx is not None and idx < len(shape):
                                    out["channel_count"] = int(shape[idx])
                        if out["frame_count"] is None and shape and len(shape) >= 1:
                            out["frame_count"] = int(shape[0])
            except (IndexError, TypeError, AttributeError):
                pass
            if out["frame_count"] is None and len(tif.pages) > 1:
                out["frame_count"] = len(tif.pages)
            if out["channel_count"] is None:
                out["channel_count"] = 1
            # Bits per pixel from dtype
            dtype = getattr(page0, "dtype", None)
            if dtype is not None:
                kind = getattr(dtype, "kind", "")
                if "u" in str(kind) or "i" in str(dtype):
                    try:
                        out["bits_per_component"] = int(dtype.itemsize * 8)
                    except (TypeError, ValueError):
                        pass
            # Channel names: ImageJ "Labels" or OME Channel Name (skip Leica-style position strings like "c:1/3 t:1/50 - LDM 001/Series001")
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

            cc = out.get("channel_count") or 1
            channel_names: list[str] = []
            labels = meta.get("Labels") if isinstance(meta, dict) else None
            if labels is not None:
                if isinstance(labels, (list, tuple)):
                    channel_names = [str(x).strip() for x in labels if x][:cc]
                elif isinstance(labels, str):
                    channel_names = [s.strip() for s in labels.split(",") if s.strip()][:cc]
            if not channel_names and hasattr(tif, "ome_metadata") and tif.ome_metadata:
                try:
                    import re
                    xml = tif.ome_metadata if isinstance(tif.ome_metadata, str) else (tif.ome_metadata.decode("utf-8", errors="ignore") if isinstance(tif.ome_metadata, bytes) else "")
                    if xml:
                        for m in re.finditer(r'<Channel[^>]*Name="([^"]*)"', xml):
                            raw = (m.group(1) or "").strip()
                            channel_names.append(raw if _looks_like_channel_name(raw) else f"Channel_{len(channel_names)+1}")
                            if len(channel_names) >= cc:
                                break
                except (TypeError, AttributeError):
                    pass
            # Use extracted names only if they all look like real channel names (not position strings like "c:1/3 t:1/50 - LDM 001/Series001")
            if channel_names and len(channel_names) == cc and all(_looks_like_channel_name(n) for n in channel_names):
                out["channel_names"] = channel_names
                out["channel_wavelengths"] = [{"name": n} for n in channel_names]
            else:
                # Use simple Channel_1, Channel_2 so UI doesn't show confusing Leica/OME position strings
                out["channel_names"] = [f"Channel_{i+1}" for i in range(cc)]
                if not out.get("channel_wavelengths") and cc >= 1:
                    out["channel_wavelengths"] = [
                        {"name": "Grayscale" if cc == 1 else f"Channel {i+1}"} for i in range(cc)
                    ]
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
                    tags = getattr(page0, "tags", None) or {}
                    # ImageDescription: tag 270 or key "ImageDescription"; some backends use int keys
                    desc_raw = None
                    for key in ("ImageDescription", 270):
                        ttag = tags.get(key) if tags and hasattr(tags, "get") else None
                        if ttag is not None:
                            desc_raw = getattr(ttag, "value", ttag)
                            break
                    if desc_raw is None and hasattr(page0, "ImageDescription"):
                        desc_raw = page0.ImageDescription
                    if desc_raw is not None:
                        desc = desc_raw.decode("utf-8", errors="ignore") if isinstance(desc_raw, bytes) else str(desc_raw)
                        if desc:
                            for part in desc.replace("\n", " ").replace("\r", " ").split():
                                if part.startswith("pixelSize="):
                                    try:
                                        pixel_size = float(part.split("=")[1].strip())
                                        break
                                    except (IndexError, ValueError):
                                        pass
                                if pixel_size <= 0 and part.startswith("spacing="):
                                    try:
                                        pixel_size = float(part.split("=")[1].strip())
                                        break
                                    except (IndexError, ValueError):
                                        pass
                if hasattr(tif, "imagej_metadata") and tif.imagej_metadata:
                    m = tif.imagej_metadata
                    if "pixelSize" in m:
                        pixel_size = float(m["pixelSize"])
                    if "finterval" in m:
                        val = float(m["finterval"])
                        time_per_frame = (val / 60.0) if val > 0 else 0.0
                    elif "interval" in m:
                        val = float(m["interval"])
                        time_per_frame = (val / 60.0) if val > 0 else 0.0
                # Fallback: pixel size from TIFF Resolution (pixels per cm)
                if pixel_size <= 0 and tif.pages:
                    try:
                        page0 = tif.pages[0]
                        xres_val = None
                        unit_val = None
                        # Try attribute first (e.g. page.resolution, page.resolutionunit)
                        if hasattr(page0, "resolution") and page0.resolution:
                            res = page0.resolution
                            if isinstance(res, (tuple, list)) and len(res) >= 1:
                                xres_val = float(res[0])
                            if hasattr(page0, "resolutionunit"):
                                unit_val = page0.resolutionunit
                        if xres_val is None:
                            tags = getattr(page0, "tags", None)
                            if tags is not None:
                                def _tag_values():
                                    if hasattr(tags, "values"):
                                        return list(tags.values())
                                    if hasattr(tags, "items"):
                                        return [v for _k, v in tags.items()]
                                    return list(tags)
                                tag_list = _tag_values()
                                # Try .get() first (tifffile may key by int or name)
                                for key in (282, "XResolution", "Resolution"):
                                    t = tags.get(key) if hasattr(tags, "get") else None
                                    if t is not None:
                                        xres_val = getattr(t, "value", t)
                                        break
                                if xres_val is None:
                                    for tag in tag_list:
                                        code = getattr(tag, "code", getattr(tag, "name", None))
                                        if code == 282 or str(code) == "XResolution":
                                            xres_val = getattr(tag, "value", None)
                                            break
                                if xres_val is not None and not isinstance(xres_val, (int, float)):
                                    if hasattr(xres_val, "numerator") and getattr(xres_val, "denominator", 0):
                                        xres_val = float(xres_val.numerator) / float(xres_val.denominator)
                                    else:
                                        xres_val = float(xres_val)
                                elif xres_val is not None:
                                    xres_val = float(xres_val)
                                for key in (296, "ResolutionUnit"):
                                    t = tags.get(key) if hasattr(tags, "get") else None
                                    if t is not None:
                                        unit_val = getattr(t, "value", t)
                                        break
                                if unit_val is None:
                                    for tag in tag_list:
                                        code = getattr(tag, "code", getattr(tag, "name", None))
                                        if code == 296 or str(code) == "ResolutionUnit":
                                            unit_val = getattr(tag, "value", None)
                                            break
                        if xres_val is not None and xres_val > 0 and unit_val == 3:  # RESUNIT_CENTIMETER
                            pixel_size = 1e4 / xres_val
                    except (TypeError, ValueError, ZeroDivisionError, AttributeError):
                        pass
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
                from bactimate.modules.bioio_reader import get_info as bioio_get_info
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


def read_calibration_from_path_or_original(image_path: Optional[Path]) -> tuple[float, float]:
    """Read pixel size and time per frame from image_path; if none found and path looks like a mask,
    try the resolved original (phase) TIFF so mask-based workflows still get calibration."""
    if not image_path or not image_path.is_file():
        return (0.0, 0.0)
    ps, tp = read_calibration_from_path(image_path)
    if ps > 0 or tp > 0:
        return (ps, tp)
    original = _resolve_original_for_mask(image_path)
    if original and original.is_file():
        return read_calibration_from_path(original)
    return (0.0, 0.0)


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


def _resolve_original_for_mask(mask_path: Optional[Path]) -> Optional[Path]:
    """Find the original (phase) TIFF for a mask file by stem."""
    if not mask_path or not mask_path.is_file():
        return None
    parent = mask_path.parent
    stem = mask_path.stem.replace("_masks", "").replace("_segmentation_omni", "").replace("_segmentation_unet", "")
    search_dirs: list[Path] = [parent, parent.parent]
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for f in search_dir.iterdir():
            if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                continue
            if "_fi" in f.stem.lower():
                continue
            if f.stem == stem:
                return f
        base = stem.rsplit("_", 1)[0] if "_" in stem else stem
        for f in search_dir.iterdir():
            if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                continue
            if "_fi" in f.stem.lower():
                continue
            if f.stem == base:
                return f
        for sub in search_dir.iterdir():
            if not sub.is_dir():
                continue
            for f in sub.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".tif", ".tiff"):
                    continue
                if "_fi" in f.stem.lower():
                    continue
                if f.stem == stem or f.stem == base:
                    return f
    return None


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
    original_path = _resolve_original_for_mask(mask_path) if mask_path else None
    for path in (original_path, mask_path, traj_path):
        if path and path.is_file():
            ps, tp = read_calibration_from_path(path)
            if ps > 0 or tp > 0:
                return (ps, tp)
    return (0.0, 0.0)
