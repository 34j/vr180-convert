from enum import auto
from pathlib import Path

import cv2 as cv
import typer
from quaternion import *  # noqa
from strenum import StrEnum
from typing_extensions import Annotated

from vr180_convert.transformer import *  # noqa
from vr180_convert.transformer import EquirectangularFormatEncoder, FisheyeFormatDecoder

from .remapper import apply, apply_lr

DEFAULT_EXTENSION = ".png"

app = typer.Typer()


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
) -> None:
    """Remap a pair of fisheye images to a pair of SBS equirectangular images."""
    if transformer == "":
        transformer_ = EquirectangularFormatEncoder() * FisheyeFormatDecoder(
            "equidistant"
        )
    else:
        transformer_ = eval(transformer)  # noqa
    apply_lr(
        transformer=transformer_,
        left_path=left_path,
        right_path=right_path,
        out_path=(
            Path(left_path).with_suffix(f".out.{DEFAULT_EXTENSION}")
            if out_path == Path("")
            else out_path
        ),
        radius=float(radius) if radius not in ["auto", "max"] else radius,  # type: ignore
        size_output=tuple(map(int, size.split("x"))),  # type: ignore
        interpolation=getattr(cv, interpolation.upper()),
        boarder_mode=getattr(cv, boarder_mode.upper()),
        boarder_value=boarder_value,
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
        transformer_ = EquirectangularFormatEncoder() * FisheyeFormatDecoder(
            "equidistant"
        )
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
