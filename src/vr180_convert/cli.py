from datetime import datetime, timezone
from enum import auto
from hashlib import sha256
from logging import DEBUG, INFO, basicConfig, getLogger
from pathlib import Path
from typing import Any

import cv2 as cv
import typer
from quaternion import *  # noqa
from rich.logging import RichHandler
from strenum import StrEnum
from typing_extensions import Annotated

from vr180_convert.transformer import *  # noqa
from vr180_convert.transformer import (
    EquirectangularEncoder,
    Euclidean3DRotator,
    FisheyeDecoder,
)

from .remapper import apply, apply_lr, match_lr

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


def _get_position_gui(image_path: Path) -> tuple[int, int]:
    """Get the position of the GUI window."""
    window_name = "Select position"
    cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow(window_name, cv.imread(image_path.as_posix()))

    x_, y_ = 0, 0

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal x_, y_
        if event == cv.EVENT_LBUTTONDOWN:
            x_, y_ = x, y
            cv.destroyWindow(window_name)

    cv.setMouseCallback(window_name, on_mouse)
    cv.waitKey(0)
    cv.destroyAllWindows()
    LOG.info(f"{x_}, {y_}")
    return x_, y_


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
        str, typer.Option(help="Output image size, defaults to 4096x4096")
    ] = "4096x4096",
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
    autosearch_timestamp_calib_r_earlier_l: Annotated[
        float,
        typer.Option(
            "--autosearch-timestamp-calib-r-earlier-l",
            "-ac",
            help="Autosearch timestamp calibration "
            "(right timestamp -= autosearch_timestamp_calib_r_earlier_l) (in seconds)",
        ),
    ] = 0.0,
    swap: Annotated[bool, typer.Option(help="Swap left and right images")] = False,
    name_unique: Annotated[bool, typer.Option(help="Make output name unique")] = False,
    automatch: Annotated[
        str, typer.Option(help="Automatch left and right images")
    ] = "",
) -> None:
    """Remap a pair of fisheye images to a pair of SBS equirectangular images."""
    if swap:
        left_path, right_path = right_path, left_path
        autosearch_timestamp_calib_r_earlier_l = -autosearch_timestamp_calib_r_earlier_l

    # find closest time-matched images
    if left_path.is_dir() and not right_path.is_dir():
        # find closest time-matched right image
        # sort with time
        right_time = right_path.stat().st_mtime
        left_path_candidates = sorted(
            left_path.rglob("*"),
            key=lambda p: abs(
                p.stat().st_mtime - right_time + autosearch_timestamp_calib_r_earlier_l
            ),
            reverse=False,
        )
        left_path_candidates = [
            p
            for p in left_path_candidates
            if (p != right_path) and (p.suffix == right_path.suffix)
        ]
        if len(left_path_candidates) == 0:
            raise ValueError("No time-matched left image found")
        if (
            len(left_path_candidates) > 1
            and left_path_candidates[0].stat().st_mtime
            == left_path_candidates[1].stat().st_mtime
        ):
            raise ValueError(
                f"Multiple time-matched left images found: {left_path_candidates}"
            )
        left_path = left_path_candidates[0]
    elif not left_path.is_dir() and right_path.is_dir():
        # find closest time-matched left image
        # sort with time
        left_time = left_path.stat().st_mtime
        right_path_candidates = sorted(
            right_path.rglob("*"),
            key=lambda p: abs(
                p.stat().st_mtime - left_time - autosearch_timestamp_calib_r_earlier_l
            ),
            reverse=False,
        )
        right_path_candidates = [
            p
            for p in right_path_candidates
            if (p != left_path) and (p.suffix == left_path.suffix)
        ]
        if len(right_path_candidates) == 0:
            raise ValueError("No time-matched right image found")
        if (
            len(right_path_candidates) > 1
            and right_path_candidates[0].stat().st_mtime
            == right_path_candidates[1].stat().st_mtime
        ):
            raise ValueError(
                f"Multiple time-matched right images found: {right_path_candidates}"
            )
        right_path = right_path_candidates[0]
    elif left_path.is_dir() and right_path.is_dir():
        raise ValueError("Both left and right paths must not be directories")

    LOG.info(
        f"L: {left_path}"
        f"@{datetime.fromtimestamp(left_path.stat().st_mtime, timezone.utc)}, "
        f"R: {right_path}"
        f"@{datetime.fromtimestamp(right_path.stat().st_mtime, timezone.utc)}"
    )

    # evaluate automatch
    transformer_: Any
    if automatch != "":
        if automatch == "auto":
            automatch_ = [
                _get_position_gui(left_path),
                _get_position_gui(right_path),
                _get_position_gui(left_path),
                _get_position_gui(right_path),
            ]
        else:
            automatch_ = [tuple(chunk.split(",")) for chunk in automatch.split(";")]  # type: ignore
        q = match_lr(
            FisheyeDecoder("equidistant"),
            # odd
            automatch_[1::2],
            # even
            automatch_[::2],
            radius=float(radius) if radius not in ["auto", "max"] else radius,  # type: ignore
            in_paths=[left_path, right_path],
        )
        LOG.info(f"Automatched quaternion: {q}")
        transformer_ = (
            EquirectangularEncoder()
            * Euclidean3DRotator(q)
            * FisheyeDecoder("equidistant"),
            EquirectangularEncoder() * FisheyeDecoder("equidistant"),
        )
    else:
        # evaluate transformer
        if transformer == "":
            transformer_ = EquirectangularEncoder() * FisheyeDecoder("equidistant")
        else:
            transformer_ = eval(transformer)  # noqa

    if swap:
        if isinstance(transformer_, tuple):
            transformer_ = transformer_[1], transformer_[0]

    # apply transformer
    name_unique_content = (
        (
            "-"
            + sha256(
                "".join(
                    [
                        transformer,
                        size,
                        interpolation,
                        boarder_mode,
                        str(boarder_value),
                        radius,
                        str(merge),
                        str(autosearch_timestamp_calib_r_earlier_l),
                        str(swap),
                    ]
                ).encode("utf-8")
            ).hexdigest()[:8]
        )
        if name_unique
        else ""
    )
    filename_default = (
        f"{Path(left_path).stem}-"
        + f"{Path(right_path).stem}{name_unique_content}.{DEFAULT_EXTENSION}"
    )
    apply_lr(
        transformer=transformer_,
        left_path=left_path,
        right_path=right_path,
        out_path=(
            Path(left_path).parent / filename_default
            if out_path == Path("")
            else out_path / filename_default if out_path.is_dir() else out_path
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
        str, typer.Option(help="Output image size, defaults to 4096x4096")
    ] = "4096x4096",
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
