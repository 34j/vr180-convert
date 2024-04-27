from datetime import datetime, timezone
from enum import auto
from logging import DEBUG, INFO, basicConfig, getLogger
from pathlib import Path

import cv2 as cv
import typer
from quaternion import *  # noqa
from rich.logging import RichHandler
from strenum import StrEnum
from typing_extensions import Annotated

from vr180_convert.transformer import *  # noqa
from vr180_convert.transformer import EquirectangularEncoder, FisheyeDecoder

from .remapper import apply, apply_lr

LOG = getLogger(__name__)
DEFAULT_EXTENSION = "png"

app = typer.Typer()


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    level = INFO
    if verbose:
        level = DEBUG
    basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


class _InterpolationFlags(StrEnum):
    """Interpolation flags enum for typer."""

    INTER_NEAREST = auto()
    INTER_LINEAR = auto()
    INTER_CUBIC = auto()
    INTER_AREA = auto()
    INTER_LANCZOS4 = auto()
    INTER_MAX = auto()
    WARP_FILL_OUTLIERS = auto()
    WARP_INVERSE_MAP = auto()


class _BorderTypes(StrEnum):
    """Border types enum for typer."""

    BORDER_CONSTANT = auto()
    BORDER_REPLICATE = auto()
    BORDER_REFLECT = auto()
    BORDER_WRAP = auto()
    BORDER_REFLECT_101 = auto()
    BORDER_TRANSPARENT = auto()
    BORDER_ISOLATED = auto()


@app.command()
def lr(
    left_path: Annotated[Path, typer.Argument(help="Left image path")],
    right_path: Annotated[Path, typer.Argument(help="Right image path")],
    transformer: Annotated[
        str, typer.Option(help="Transformer Python code (to be `eval()`ed)")
    ] = "",
    out_path: Annotated[
        Path,
        typer.Option(
            help="Output image path, defaults to left_path.with_suffix('.out.jpg')"
        ),
    ] = Path(""),
    size: Annotated[
        str, typer.Option(help="Output image size, defaults to 2048x2048")
    ] = "2048x2048",
    interpolation: Annotated[
        _InterpolationFlags,
        typer.Option(help="Interpolation method, defaults to lanczos4"),
    ] = _InterpolationFlags.INTER_LANCZOS4,  # type: ignore
    boarder_mode: Annotated[
        _BorderTypes, typer.Option(help="Border mode, defaults to constant")
    ] = _BorderTypes.BORDER_CONSTANT,  # type: ignore
    boarder_value: int = 0,
    radius: Annotated[
        str, typer.Option(help="Radius of the fisheye image, defaults to 'auto'")
    ] = "auto",
    merge: Annotated[bool, typer.Option(help="Merge left and right images")] = False,
) -> None:
    """Remap a pair of fisheye images to a pair of SBS equirectangular images."""
    # evaluate transformer
    if transformer == "":
        transformer_ = EquirectangularEncoder() * FisheyeDecoder("equidistant")
    else:
        transformer_ = eval(transformer)  # noqa

    # find closest time-matched images
    if left_path.is_dir() and not right_path.is_dir():
        # find closest time-matched right image
        # sort with time
        right_time = right_path.stat().st_mtime
        left_path_candidates = sorted(
            left_path.rglob("*"),
            key=lambda p: abs(p.stat().st_mtime - right_time),
            reverse=False,
        )
        left_path_candidates = [
            p
            for p in left_path_candidates
            if (p != right_path) and (p.suffix == right_path.suffix)
        ]
        if len(left_path_candidates) == 0:
            raise ValueError("No time-matched left image found")
        left_path = left_path_candidates[0]
    elif not left_path.is_dir() and right_path.is_dir():
        # find closest time-matched left image
        # sort with time
        left_time = left_path.stat().st_mtime
        right_path_candidates = sorted(
            right_path.rglob("*"),
            key=lambda p: abs(p.stat().st_mtime - left_time),
            reverse=False,
        )
        right_path_candidates = [
            p
            for p in right_path_candidates
            if (p != left_path) and (p.suffix == left_path.suffix)
        ]
        if len(right_path_candidates) == 0:
            raise ValueError("No time-matched right image found")
        right_path = right_path_candidates[0]
    elif left_path.is_dir() and right_path.is_dir():
        raise ValueError("Both left and right paths must not be directories")
    LOG.info(
        f"L: {left_path}"
        f"@{datetime.fromtimestamp(left_path.stat().st_mtime, timezone.utc)}, "
        f"R: {right_path}"
        f"@{datetime.fromtimestamp(right_path.stat().st_mtime, timezone.utc)}"
    )

    # apply transformer
    apply_lr(
        transformer=transformer_,
        left_path=left_path,
        right_path=right_path,
        out_path=(
            Path(left_path).parent
            / f"{Path(left_path).stem}-{Path(right_path).stem}.{DEFAULT_EXTENSION}"
            if out_path == Path("")
            else out_path
        ),
        radius=float(radius) if radius not in ["auto", "max"] else radius,  # type: ignore
        size_output=tuple(map(int, size.split("x"))),  # type: ignore
        interpolation=getattr(cv, interpolation.upper()),
        boarder_mode=getattr(cv, boarder_mode.upper()),
        boarder_value=boarder_value,
        merge=merge,
    )


@app.command()
def s(
    in_paths: Annotated[list[Path], typer.Argument(help="Image paths")],
    transformer: Annotated[
        str, typer.Option(help="Transformer Python code (to be `eval()`ed)")
    ] = "",
    out_path: Annotated[
        Path,
        typer.Option(
            help="Output image path, defaults to left_path.with_suffix('.out.jpg')"
        ),
    ] = Path(""),
    size: Annotated[
        str, typer.Option(help="Output image size, defaults to 2048x2048")
    ] = "2048x2048",
    interpolation: Annotated[
        _InterpolationFlags,
        typer.Option(help="Interpolation method, defaults to lanczos4"),
    ] = _InterpolationFlags.INTER_LANCZOS4,  # type: ignore
    boarder_mode: Annotated[
        _BorderTypes, typer.Option(help="Border mode, defaults to constant")
    ] = _BorderTypes.BORDER_CONSTANT,  # type: ignore
    boarder_value: int = 0,
    radius: Annotated[
        str, typer.Option(help="Radius of the fisheye image, defaults to 'auto'")
    ] = "auto",
) -> None:
    """Remap fisheye images to SBS equirectangular images."""
    if transformer == "":
        transformer_ = EquirectangularEncoder() * FisheyeDecoder("equidistant")
    else:
        transformer_ = eval(transformer)  # noqa

    if out_path == Path(""):
        out_paths = [p.with_suffix(f".out.{DEFAULT_EXTENSION}") for p in in_paths]
    elif out_path.is_dir():
        out_paths = [out_path / p.name for p in in_paths]
    else:
        if len(in_paths) > 1:
            raise ValueError(
                "Output path must be a directory when multiple input paths are provided"
            )
        out_paths = [out_path for p in in_paths]

    apply(
        transformer=transformer_,
        in_paths=in_paths,
        out_paths=out_paths,
        radius=float(radius) if radius not in ["auto", "max"] else radius,  # type: ignore
        size_output=tuple(map(int, size.split("x"))),  # type: ignore
        interpolation=getattr(cv, interpolation.upper()),
        boarder_mode=getattr(cv, boarder_mode.upper()),
        boarder_value=boarder_value,
    )
