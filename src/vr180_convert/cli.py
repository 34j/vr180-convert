"""
CLI for vr180-convert.

Transforms fisheye images to equirectangular VR180 format.
"""

from __future__ import annotations

from hashlib import sha256
from logging import DEBUG, INFO, basicConfig, getLogger
from pathlib import Path
from typing import Literal

import array_api_compat
import cv2 as cv
import numpy as np
from cyclopts import App
from rich.logging import RichHandler

from vr180_convert.divide import Concater
from vr180_convert.remapper.base import RemapperBase
from vr180_convert.remapper.equidistant import EquirectangularEncoder
from vr180_convert.remapper.fisheye import FisheyeDecoder
from vr180_convert.remapper.normalize import NormalizeRemapper
from vr180_convert.remapper.radius import AutoDenormalizeRemapper, _get_radius_smart
from vr180_convert.remapper.rotation_match import RotationMatchRemapper
from vr180_convert.remapper.transformer import RemapperTransformer
from vr180_convert.search import find_time_matched_image

LOG = getLogger(__name__)
DEFAULT_EXTENSION = "png"

FisheyeMapping = Literal["rectilinear", "stereographic", "equidistant", "equisolid", "orthographic"]
"""Supported fisheye projection types."""


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
    automatch: bool = False,
    mapping_type: str = "equidistant",
) -> list[RemapperBase]:
    pipes: list[RemapperBase] = [
        AutoDenormalizeRemapper(strategy=radius),
        FisheyeDecoder(mapping_type),
    ]
    if automatch:
        pipes.append(RotationMatchRemapper())
    pipes += [EquirectangularEncoder(), NormalizeRemapper()]
    return pipes


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
    automatch: bool = False,
    mapping_type: str = "equidistant",
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
        Enable automatic rotation matching between left/right views.
    mapping_type
        Fisheye projection model: ``"rectilinear"``, ``"stereographic"``,
        ``"equidistant"``, ``"equisolid"``, or ``"orthographic"``.
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
    radius_val = _resolve_radius(radius, left_path, right_path)
    remappers = _pipeline(radius_val, size_out, automatch=automatch, mapping_type=mapping_type)
    transformer = RemapperTransformer(remappers=remappers, size_output=size_out) * Concater()

    img_l = cv.imread(str(left_path))
    img_r = cv.imread(str(right_path))
    xp = array_api_compat.array_namespace(img_l, img_r)
    img = xp.stack([xp.asarray(img_l), xp.asarray(img_r)], axis=0)
    result = transformer.transform(img)

    out = _output_path(left_path, right_path, out_path, unique, tag=str(remappers))
    cv.imwrite(str(out), np.asarray(result))
    LOG.info("Saved → %s", out)


@app.command
def s(
    in_paths: list[Path],
    out_path: Path = Path(""),
    *,
    verbose: bool = False,
    size: str = "4096x4096",
    radius: str = "auto",
    mapping_type: str = "equidistant",
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
        remappers: list[RemapperBase] = [
            AutoDenormalizeRemapper(strategy=rv),
            FisheyeDecoder(mapping_type),
            EquirectangularEncoder(),
            NormalizeRemapper(),
        ]
        transformer = RemapperTransformer(remappers=remappers, size_output=size_out)
        result = transformer.transform(xp.asarray(img_np))
        cv.imwrite(str(out), np.asarray(result))
        LOG.info("Saved %s → %s", in_path, out)


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
        LOG.info("Swapped %s → %s", in_path, out)
