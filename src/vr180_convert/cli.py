from datetime import datetime, timezone
from enum import auto
from hashlib import sha256
from logging import DEBUG, INFO, basicConfig, getLogger
from pathlib import Path
from typing import Any, Sequence

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
    MultiTransformer,
    TransformerBase,
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


def _get_position_gui(image_paths: Sequence[Path]) -> list[tuple[int, int]]:
    """Get the position of the GUI window."""
    window_name = "Select position"
    cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    i = 0
    res = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal res, i
        if event == cv.EVENT_LBUTTONDOWN:
            res.append((x, y))
            LOG.info(f"Position {i}: ({x}, {y})")

    cv.setMouseCallback(window_name, on_mouse)
    cv.imshow(window_name, cv.imread(image_paths[i].as_posix()))
    while True:
        cv.waitKey(10)
        if len(res) == i + 1:
            i += 1
            if i == len(image_paths):
                break
            cv.imshow(window_name, cv.imread(image_paths[i].as_posix()))
    cv.destroyAllWindows()
    return res


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
    border_mode: Annotated[
        _BorderTypes, typer.Option(help="Border mode, defaults to constant")
    ] = _BorderTypes.BORDER_CONSTANT,  # type: ignore
    border_value: int = 0,
    radius: Annotated[
        str, typer.Option(help="Radius of the fisheye image, defaults to 'auto'")
    ] = "auto",
    merge: Annotated[bool, typer.Option(help="Export as an anaglyph")] = False,
    autosearch_timestamp_calib_r_earlier_l: Annotated[
        float,
        typer.Option(
            "--autosearch-timestamp-calib-r-earlier-l",
            "-ac",
            help="Autosearch timestamp calibration "
            "(right timestamp -= autosearch_timestamp_calib_r_earlier_l) (in seconds)",
        ),
    ] = 0.0,
    swap: Annotated[
        bool,
        typer.Option(help="Swap left and right images as well as transformer, etc."),
    ] = False,
    name_unique: Annotated[bool, typer.Option(help="Make output name unique")] = False,
    automatch: Annotated[
        str,
        typer.Option(
            help='Calibrate rotation. e.g. "0,0;0,0;1,1;1,1". If "gui", use GUI'
        ),
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
    transformer_: TransformerBase | tuple[TransformerBase, TransformerBase]
    # evaluate transformer
    if transformer == "":
        transformer_ = EquirectangularEncoder() * FisheyeDecoder("equidistant")
    else:
        transformer_ = eval(transformer)  # noqa
    if automatch != "":
        if not isinstance(transformer_, MultiTransformer):
            raise ValueError("Automatch requires MultiTransformer")

        transformer_is_encoder = [
            x.__class__.__name__.endswith("Encoder") for x in transformer_.transformers
        ]
        transformer_first_encoder_index = transformer_is_encoder.index(True)
        transformer_until_encoder = MultiTransformer(
            transformer_.transformers[: transformer_first_encoder_index + 1]
        )
        transformer_after_encoder = MultiTransformer(
            transformer_.transformers[transformer_first_encoder_index + 1 :]
        )
        LOG.debug(f"{transformer_until_encoder=}, {transformer_after_encoder=}")

        if automatch == "gui":
            automatch_ = _get_position_gui(
                [left_path, right_path, left_path, right_path]
            )
            LOG.info(
                f"Automatched position: {';'.join([','.join(map(str, p)) for p in automatch_])}"
            )
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
            transformer_until_encoder
            * Euclidean3DRotator(q)
            * transformer_after_encoder,
            transformer_,
        )
        LOG.info(f"Automatched transformer: {transformer_}")

    # if swap:
    #     if isinstance(transformer_, tuple) and automatch == "":
    #         transformer_ = transformer_[1], transformer_[0]

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
                        border_mode,
                        str(border_value),
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
        boarder_mode=getattr(cv, border_mode.upper()),
        boarder_value=border_value,
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
