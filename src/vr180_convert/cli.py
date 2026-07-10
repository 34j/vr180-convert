"""
CLI for vr180-convert.

Transforms fisheye images to equirectangular VR180 format.
"""

from __future__ import annotations

import re
from hashlib import sha256
from logging import DEBUG, INFO, basicConfig, getLogger
from pathlib import Path
from typing import Literal

import array_api_compat
import cv2 as cv
import numpy as np
from cyclopts import App
from rich.logging import RichHandler

from vr180_convert.cli_gui import get_position_gui
from vr180_convert.divide import Concater
from vr180_convert.merge import Merger
from vr180_convert.remapper.base import RemapperBase
from vr180_convert.remapper.equidistant import EquirectangularEncoder, equidistant_to_3d
from vr180_convert.remapper.euclidean import Euclidean3DRotator
from vr180_convert.remapper.fisheye import FisheyeDecoder
from vr180_convert.remapper.normalize import NormalizeRemapper
from vr180_convert.remapper.radius import AutoDenormalizeRemapper, _get_radius_smart
from vr180_convert.remapper.rotation_match import (
    PerEyeRotator,
    RotationMatchRemapper,
    rotation_match_robust,
)
from vr180_convert.remapper.transformer import RemapperTransformer
from vr180_convert.search import find_time_matched_image

LOG = getLogger(__name__)
DEFAULT_EXTENSION = "png"

FisheyeMapping = Literal["rectilinear", "stereographic", "equidistant", "equisolid", "orthographic"]


def _init_logging(verbose: bool) -> None:
    basicConfig(
        level=DEBUG if verbose else INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def _parse_size(s: str) -> tuple[int, int]:
    a, b = s.split("x")
    return int(a), int(b)


def _resolve_radius(opt: str, left_path: Path, right_path: Path) -> float | str:
    if opt in ("auto", "max"):
        images = [cv.imread(str(left_path)), cv.imread(str(right_path))]
        return _get_radius_smart(opt, np.stack(images))
    return float(opt)


def _pipeline(
    radius: float | str,
    size_out: tuple[int, int],
    automatch: str = "",
    mapping_type: str = "equidistant",
) -> list[RemapperBase]:
    pipes: list[RemapperBase] = [
        AutoDenormalizeRemapper(strategy=radius),
        FisheyeDecoder(mapping_type),
    ]
    if automatch and automatch.startswith("fm"):
        pipes.append(RotationMatchRemapper())
    pipes += [EquirectangularEncoder(), NormalizeRemapper()]
    return pipes


def _find_first_encoder_index(remappers: list[RemapperBase]) -> int:
    """Find the index of the first encoder in the remapper list."""
    for i, r in enumerate(remappers):
        name = r.__class__.__name__
        if name == "EquirectangularEncoder" or name.endswith("Encoder"):
            return i
    raise ValueError("No encoder found in the pipeline")


def _transform_points_through_decoder(
    points_l: np.ndarray,
    points_r: np.ndarray,
    radius: float,
    center: tuple[float, float],
    decoder_remappers: list[RemapperBase],
) -> tuple[np.ndarray, np.ndarray]:
    """Transform 2D points through the decoder chain to get 3D vectors."""
    xp = array_api_compat.array_namespace(np.asarray(points_l))
    x = xp.stack([xp.asarray(points_l[:, 0]), xp.asarray(points_r[:, 0])], axis=0)
    y = xp.stack([xp.asarray(points_l[:, 1]), xp.asarray(points_r[:, 1])], axis=0)
    for remapper in reversed(decoder_remappers):
        x, y = remapper.inverse_remap(x, y)
    v = equidistant_to_3d(x, y)
    vl = np.asarray(v[0, ...])
    vr = np.asarray(v[1, ...])
    return vl, vr


def _compute_per_eye_rotation(
    points_l: list[tuple[float, float]],
    points_r: list[tuple[float, float]],
    radius: float,
    center: tuple[float, float],
    decoder_remappers: list[RemapperBase],
) -> tuple[Euclidean3DRotator, Euclidean3DRotator]:
    """Compute half-angle rotation from corresponding points."""
    vl, vr = _transform_points_through_decoder(
        np.array(points_l), np.array(points_r), radius, center, decoder_remappers
    )
    q, _ = rotation_match_robust(points_to_be_rotated=vr, points=vl)
    one = np.quaternion(1, 0, 0, 0)
    half_q = (one + q) / np.abs(one + q)
    childl = Euclidean3DRotator(rotation=np.conj(half_q))
    childr = Euclidean3DRotator(rotation=half_q)
    return childl, childr


def _output_path(left: Path, right: Path, out: Path, unique: bool, tag: str = "") -> Path:
    name = f"{left.stem}-{right.stem}"
    if unique:
        name = f"{name}-{sha256(tag.encode()).hexdigest()[:8]}"
    filename = f"{name}.{DEFAULT_EXTENSION}"
    if out == Path(""):
        return left.parent / filename
    if out.is_dir():
        return out / filename
    return out


app = App(help_format="markdown")


@app.default
def lr(
    left_path: Path,
    right_path: Path,
    out_path: Path = Path(""),
    *,
    verbose: bool = False,
    size: str = "4096x4096",
    radius: str = "auto",
    swap: bool = False,
    automatch: str = "",
    mapping_type: str = "equidistant",
    transformer: str = "",
    merge: bool = False,
    savematch: bool = False,
    time_search: Path | None = None,
    time_calib: float = 0.0,
    unique: bool = False,
) -> None:
    """
    Remap a pair of fisheye images to a side-by-side equirectangular VR180 image.

    Projects fisheye coordinates onto the unit sphere via the chosen projection model,
    optionally aligns left/right views via rotation matching, then encodes to equirectangular.

    Parameters
    ----------
    left_path
        Left fisheye image path.
    right_path
        Right fisheye image path.
    out_path
        Output image path. If a directory, the output filename is auto-generated.
    verbose
        Enable verbose logging.
    size
        Output size as WxH, e.g. ``4096x4096``.
    radius
        Fisheye radius: ``"auto"``, ``"max"``, or a pixel value.
    swap
        Swap left and right images.
    automatch
        Rotation matching mode. ``"fm"`` (or ``"fm<scale>"``) for AKAZE feature matching,
        ``"gui"`` (or ``"gui<N>"``) for GUI-based point selection,
        ``"x1,y1;x2,y2;..."`` for manual point pairs.
    mapping_type
        Fisheye projection model: ``"rectilinear"``, ``"stereographic"``,
        ``"equidistant"``, ``"equisolid"``, or ``"orthographic"``.
    transformer
        Custom transformer Python expression (eval'd). Overrides the default pipeline.
        Example: ``EquirectangularEncoder()
        * Euclidean3DRotator(from_rotation_vector([0, 1.57, 0]))
        * FisheyeDecoder("equidistant")``
    merge
        Export as an anaglyph image (red-cyan) for calibration checking.
    savematch
        Save the feature match visualization (only with ``automatch=fm``).
    time_search
        Search a directory for the time-matched partner of the given image.
    time_calib
        Right-camera time offset in seconds (``right_time -= time_calib``).
    unique
        Append a unique hash to the output filename.

    """
    _init_logging(verbose)
    if swap:
        left_path, right_path = right_path, left_path
        time_calib = -time_calib

    # Handle same-path (SBS image split)
    if left_path == right_path:
        LOG.info("Same path provided, splitting image into left/right halves")
        sbs_img = cv.imread(str(left_path))
        mid = sbs_img.shape[1] // 2
        left_np = sbs_img[:, :mid, :]
        right_np = sbs_img[:, mid:, :]
        size_out = _parse_size(size)
        radius_val = _resolve_radius(radius, left_path, right_path)
        # Build pipeline for same-path case
        remappers: list[RemapperBase]
        if transformer:
            remappers = _eval_transformer(transformer)
        else:
            pipes: list[RemapperBase] = [
                AutoDenormalizeRemapper(strategy=radius_val),
                FisheyeDecoder(mapping_type),
            ]
            if automatch and automatch.startswith("fm"):
                pipes.append(RotationMatchRemapper())
            elif automatch and automatch not in ("", "fm"):
                # GUI or manual points: compute rotation and insert PerEyeRotator
                _handle_manual_automatch(automatch, pipes, left_np, right_np, radius_val)
            pipes += [EquirectangularEncoder(), NormalizeRemapper()]
            remappers = pipes
        transformer_obj = RemapperTransformer(remappers=remappers, size_output=size_out) * Concater()
        xp = array_api_compat.array_namespace(np.asarray(left_np))
        img = xp.stack([xp.asarray(left_np), xp.asarray(right_np)], axis=0)
        result = transformer_obj.transform(img)
        out = _output_path(left_path, right_path, out_path, unique, tag=str(remappers))
        cv.imwrite(str(out), np.asarray(result))
        LOG.info("Saved -> %s", out)
        return

    if time_search is not None:
        right_path = find_time_matched_image(left_path, time_search, search_path_earlier_diff=time_calib)
    elif left_path.is_dir() or right_path.is_dir():
        if left_path.is_dir() and not right_path.is_dir():
            left_path = find_time_matched_image(right_path, left_path, search_path_earlier_diff=time_calib)
        elif right_path.is_dir() and not left_path.is_dir():
            right_path = find_time_matched_image(left_path, right_path, search_path_earlier_diff=time_calib)
        else:
            raise ValueError("Both paths cannot be directories")

    LOG.info("L: %s, R: %s", left_path, right_path)
    size_out = _parse_size(size)

    img_l_np = cv.imread(str(left_path))
    img_r_np = cv.imread(str(right_path))
    radius_val = _resolve_radius(radius, left_path, right_path)

    if transformer:
        remappers = _eval_transformer(transformer)
    else:
        remappers = _pipeline(radius_val, size_out, automatch=automatch, mapping_type=mapping_type)
        if automatch and not automatch.startswith("fm"):
            # GUI or manual points: need to handle after building pipeline
            # Remove the placeholder PerEyeRotator (if any) and compute from points
            pipes: list[RemapperBase] = [
                AutoDenormalizeRemapper(strategy=radius_val),
                FisheyeDecoder(mapping_type),
            ]
            _handle_manual_automatch(automatch, pipes, img_l_np, img_r_np, radius_val)
            pipes += [EquirectangularEncoder(), NormalizeRemapper()]
            remappers = pipes

    transformer_obj = RemapperTransformer(remappers=remappers, size_output=size_out) * (
        Merger() if merge else Concater()
    )

    xp = array_api_compat.array_namespace(img_l_np, img_r_np)
    img = xp.stack([xp.asarray(img_l_np), xp.asarray(img_r_np)], axis=0)
    result = transformer_obj.transform(img)

    out = _output_path(left_path, right_path, out_path, unique, tag=str(remappers))
    cv.imwrite(str(out), np.asarray(result))
    LOG.info("Saved -> %s", out)

    # Save match visualization if requested
    if savematch and automatch.startswith("fm"):
        _save_match_image(remappers, left_path, right_path, out)


@app.command
def s(
    in_paths: list[Path],
    out_path: Path = Path(""),
    *,
    verbose: bool = False,
    size: str = "4096x4096",
    radius: str = "auto",
    mapping_type: str = "equidistant",
    transformer: str = "",
) -> None:
    """
    Remap single fisheye images to equirectangular format.

    Parameters
    ----------
    in_paths
        One or more input image paths.
    out_path
        Output path or directory. If a directory, each input keeps its name.
    verbose
        Enable verbose logging.
    size
        Output size as WxH, e.g. ``4096x4096``.
    radius
        Fisheye radius: ``"auto"``, ``"max"``, or a pixel value.
    mapping_type
        Fisheye projection model: ``"rectilinear"``, ``"stereographic"``,
        ``"equidistant"``, ``"equisolid"``, or ``"orthographic"``.
    transformer
        Custom transformer Python expression (eval'd). Overrides the default pipeline.

    """
    _init_logging(verbose)
    size_out = _parse_size(size)

    if out_path == Path(""):
        out_paths = [p.with_suffix(f".out.{DEFAULT_EXTENSION}") for p in in_paths]
    elif out_path.is_dir():
        out_paths = [out_path / p.name for p in in_paths]
    else:
        if len(in_paths) > 1:
            raise ValueError("Multiple inputs require a directory output path")
        out_paths = [out_path]

    for in_path, out in zip(in_paths, out_paths, strict=True):
        img_np = cv.imread(str(in_path))
        xp = array_api_compat.array_namespace(img_np)
        rv = _get_radius_smart(
            radius if radius in ("auto", "max") else float(radius),
            [img_np],
        )
        if transformer:
            remappers = _eval_transformer(transformer)
        else:
            remappers: list[RemapperBase] = [
                AutoDenormalizeRemapper(strategy=rv),
                FisheyeDecoder(mapping_type),
                EquirectangularEncoder(),
                NormalizeRemapper(),
            ]
        transformer_obj = RemapperTransformer(remappers=remappers, size_output=size_out)
        result = transformer_obj.transform(xp.asarray(img_np))
        cv.imwrite(str(out), np.asarray(result))
        LOG.info("Saved %s -> %s", in_path, out)


@app.command
def swap(
    in_paths: list[Path],
    *,
    verbose: bool = False,
    overwrite: bool = True,
) -> None:
    """
    Swap the left and right halves of a side-by-side VR180 image.

    Parameters
    ----------
    in_paths
        One or more SBS image paths to swap.
    verbose
        Enable verbose logging.
    overwrite
        Overwrite the original file (instead of creating a ``.swap`` copy).

    """
    _init_logging(verbose)
    for in_path in in_paths:
        img = cv.imread(str(in_path))
        mid = img.shape[1] // 2
        swapped = np.hstack([img[:, mid:], img[:, :mid]])
        out = in_path if overwrite else in_path.with_suffix(f".swap{in_path.suffix}")
        cv.imwrite(str(out), swapped)
        LOG.info("Swapped %s -> %s", in_path, out)


def _eval_transformer(expr: str) -> list[RemapperBase]:
    """Evaluate a transformer expression and return a list of remappers."""
    import numpy as np  # noqa: F401 - needed for eval
    from numpy.quaternion import from_rotation_vector  # noqa: F401 - needed for eval

    from vr180_convert.remapper.equidistant import EquirectangularEncoder  # noqa: F401 - needed for eval
    from vr180_convert.remapper.euclidean import Euclidean3DRotator  # noqa: F401 - needed for eval
    from vr180_convert.remapper.fisheye import FisheyeDecoder  # noqa: F401 - needed for eval
    from vr180_convert.remapper.polar_roll import PolarRollRemapper  # noqa: F401 - needed for eval

    result = eval(expr)  # noqa: S307
    if isinstance(result, RemapperBase):
        return [result]
    if isinstance(result, (list, tuple)):
        return list(result)
    raise ValueError(f"Invalid transformer expression: {expr!r}")


def _handle_manual_automatch(
    automatch: str,
    pipes: list[RemapperBase],
    img_l: np.ndarray,
    img_r: np.ndarray,
    radius_val: float,
) -> None:
    """Handle GUI or manual point automatch, inserting PerEyeRotator into pipes."""
    if automatch.startswith("gui"):
        n_match = re.match(r"gui(\d+)", automatch)
        n_points = int(n_match.group(1)) if n_match else 2
        points = get_position_gui([img_l, img_r] * n_points)
        LOG.info("GUI points: %s", ";".join(",".join(map(str, p)) for p in points))
        points_l = points[0::2]
        points_r = points[1::2]
    elif automatch.startswith("fm"):
        # Handled by RotationMatchRemapper already
        return
    else:
        # Manual points: "x1,y1;x2,y2;..."
        chunks = automatch.split(";")
        coords = [(int(c.split(",")[0]), int(c.split(",")[1])) for c in chunks]
        points_l = coords[0::2]
        points_r = coords[1::2]

    center = (img_l.shape[1] // 2, img_l.shape[0] // 2)
    childl, childr = _compute_per_eye_rotation(points_l, points_r, radius_val, center, pipes)
    pipes.append(PerEyeRotator(childl=childl, childr=childr))


def _save_match_image(
    remappers: list[RemapperBase],
    left_path: Path,
    right_path: Path,
    out_path: Path,
) -> None:
    """Save match visualization image for fm automatch (inliers only)."""
    for r in remappers:
        if isinstance(r, RotationMatchRemapper) and r.match is not None:
            img_l = cv.imread(str(left_path))
            img_r = cv.imread(str(right_path))
            match = r.match
            bad_idx = r.bad_idx

            kp1_list = list(match.kp1) if hasattr(match.kp1, "__iter__") else []
            kp2_list = list(match.kp2) if hasattr(match.kp2, "__iter__") else []
            matches_list = list(match.matches) if hasattr(match.matches, "__iter__") else []

            if matches_list and bad_idx is not None:
                # Filter out outlier matches using bad_idx
                good_matches = [m for m, bad in zip(matches_list, bad_idx, strict=False) if not bad]
                import random

                sample_matches = random.sample(good_matches, min(100, len(good_matches)))
                img_match = cv.drawMatches(img_l, kp1_list, img_r, kp2_list, sample_matches, None)
                match_path = out_path.with_suffix(f".match{out_path.suffix}")
                cv.imwrite(str(match_path), img_match)
                LOG.info(
                    "Match visualization saved -> %s (%d inliers, %d outliers)",
                    match_path,
                    len(good_matches),
                    bad_idx.sum(),
                )
            elif matches_list:
                import random

                sample_matches = random.sample(matches_list, min(100, len(matches_list)))
                img_match = cv.drawMatches(img_l, kp1_list, img_r, kp2_list, sample_matches, None)
                match_path = out_path.with_suffix(f".match{out_path.suffix}")
                cv.imwrite(str(match_path), img_match)
                LOG.info("Match visualization saved -> %s", match_path)
            break
