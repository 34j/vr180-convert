import re
from datetime import datetime, timezone
from enum import auto
from hashlib import sha256
from logging import DEBUG, INFO, basicConfig, getLogger
from pathlib import Path
from typing import Any, Sequence

import cv2 as cv
import numpy as np
import typer
from numpy.typing import NDArray
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

from .remapper import (
    apply,
    apply_lr,
    match_lr,
    match_points,
    rotation_match,
    rotation_match_robust,
)

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


def _get_position_gui(
    image_paths: Sequence[Path | NDArray[Any]],
) -> list[tuple[int, int]]:
    """Get the position of the GUI window."""
    images = [
        cv.imread(image_path.as_posix()) if isinstance(image_path, Path) else image_path
        for image_path in image_paths
    ]
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
    cv.imshow(window_name, images[i])
    while True:
        cv.waitKey(10)
        if len(res) == i + 1:
            i += 1
            if i == len(image_paths):
                break
            cv.imshow(window_name, images[i])
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
    merge: Annotated[
        bool, typer.Option("-m", "--merge", "--anaglyph", help="Export as an anaglyph")
    ] = False,
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

    # evaluate automatch
    if automatch != "":
        # split transformer into two parts (before and after the first encoder)
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

        # match points using the original images
        img_l, img_r = cv.imread(left_path.as_posix()), cv.imread(right_path.as_posix())
        if automatch.startswith("fm"):
            scale_match = re.match(r"fm([\d\.]+)", automatch)
            scale_default = 1
            scale = (
                float(scale_match.group(1) or scale_default)
                if scale_match
                else scale_default
            )
            points_l, points_r, kp1, kp2, matches, img_l, img_r = match_points(
                img_l, img_r, scale=scale
            )
            points_l, points_r = (
                points_r,
                points_l,
            )  # TODO: don't understand why this is needed
        else:
            if automatch.startswith("gui"):
                n_points_match = re.match(r"gui(\d+)", automatch)
                n_points_default = 2
                n_points = (
                    int(n_points_match.group(1) or n_points_default)
                    if n_points_match
                    else n_points_default
                )
                automatch_ = _get_position_gui([img_l, img_r] * n_points)
                LOG.info(
                    f"Automatched position: {';'.join([','.join(map(str, p)) for p in automatch_])}"
                )
            else:
                automatch_ = [
                    (int(chunk.split(",")[0]), int(chunk.split(",")[1]))
                    for chunk in automatch.split(";")
                ]
            points_l, points_r = automatch_[1::2], automatch_[::2]  # odd, even

        # transform matched points
        vl, vr = match_lr(
            transformer_after_encoder,
            points_l,
            points_r,
            radius=float(radius) if radius not in ["auto", "max"] else radius,  # type: ignore
            in_paths=[left_path, right_path],
        )

        # now vl, vr is normalized, we can do 3d rotation match
        if automatch.startswith("fm"):
            from random import sample

            q, bad_idx = rotation_match_robust(vl, vr)
            img_match = cv.drawMatches(
                img_l, kp1, img_r, kp2, sample(list(matches[~bad_idx]), 100), None
            )
        else:
            q = rotation_match(vl, vr)
        LOG.info(f"Automatched quaternion: {q}")

        # insert the rotation transformer
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
    out_path = (
        Path(left_path).parent / filename_default
        if out_path == Path("")
        else out_path / filename_default if out_path.is_dir() else out_path
    )
    if automatch.startswith("fm"):
        cv.imwrite(
            out_path.with_suffix(f".match{out_path.suffix}").as_posix(), img_match
        )
    apply_lr(
        transformer=transformer_,
        left_path=left_path,
        right_path=right_path,
        out_path=out_path,
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


@app.command()
def xmp(
    in_paths: Annotated[list[Path], typer.Argument(help="Image paths")],
    wslpath: Annotated[
        bool,
        typer.Option(
            "-wsl",
            "--wslpath",
            help="Convert Windows path to WSL path, be careful as it uses subprocess",
        ),
    ] = False,
) -> None:
    """Add XMP metadata to the image."""
    import base64
    import subprocess as sp
    from tempfile import NamedTemporaryFile

    try:
        from libxmp import XMPFiles, XMPMeta
    except Exception as e:
        import os
        import sys

        if os.name != "nt":
            raise e

        LOG.info("Trying to install this package in WSL...")
        command = (
            "wsl -- sudo apt install -y exempi pipx python3.11 "
            "&& pipx run --python=python3.11 --spec=vr180-convert[xmp] "
            f'vr180-convert {" ".join(sys.argv[1:])} -wsl'
        )
        LOG.info(f"Running command: {command}")
        sp.run(command, check=True)  # noqa

        return

    for in_path in in_paths:
        if wslpath:
            in_path = Path(
                sp.run(["wslpath", "-u", "-a", in_path], capture_output=True)  # noqa
                .stdout.decode()
                .strip()
            )

        left_path = in_path.with_suffix(f".xmp{in_path.suffix}")

        # read combined image
        image = cv.imread(in_path.as_posix())

        # extract left image
        left_image = image[:, : image.shape[1] // 2]
        right_image = image[:, image.shape[1] // 2 :]
        width = image.shape[1]
        height = image.shape[0]
        with NamedTemporaryFile(suffix=left_path.suffix) as right_file:
            cv.imwrite(left_path.as_posix(), left_image)
            cv.imwrite(right_file.name, right_image)

            # use left file as a base
            xmpfile = XMPFiles(file_path=left_path.as_posix(), open_forupdate=True)
            lxmp = XMPMeta()

            LOG.debug(f"{in_path=}, {lxmp=}")

            # Google's namespace
            XMP_GIMAGE = "http://ns.google.com/photos/1.0/image/"
            XMP_GPANO = "http://ns.google.com/photos/1.0/panorama/"
            XMP_NOTE = "http://ns.adobe.com/xmp/note/"
            XMPMeta.register_namespace(XMP_GIMAGE, "GImage")
            XMPMeta.register_namespace(XMP_GPANO, "GPano")
            XMPMeta.register_namespace(XMP_NOTE, "xmpNote")

            # Set GPano properties
            lxmp.set_property(XMP_GPANO, "UsePanoramaViewer", "True")
            lxmp.set_property(XMP_GPANO, "ProjectionType", "equirectangular")
            lxmp.set_property_int(XMP_GPANO, "CroppedAreaImageWidthPixels", width / 2)
            lxmp.set_property_int(XMP_GPANO, "CroppedAreaImageHeightPixels", height)
            lxmp.set_property_int(XMP_GPANO, "CroppedAreaLeftPixels", width / 4)
            lxmp.set_property_int(XMP_GPANO, "CroppedAreaTopPixels", 0)
            lxmp.set_property_int(XMP_GPANO, "FullPanoWidthPixels", width)
            lxmp.set_property_int(XMP_GPANO, "FullPanoHeightPixels", height)
            lxmp.set_property_int(XMP_GPANO, "PosePitchDegrees", 0)
            lxmp.set_property_int(XMP_GPANO, "PoseRollDegrees", 0)
            lxmp.set_property_int(XMP_GPANO, "InitialViewHeadingDegrees", 180)

            # Set GImage properties
            lxmp.set_property(XMP_GIMAGE, "Mime", "image/jpeg")
            lxmp.set_property(
                XMP_GIMAGE, "Data", base64.b64encode(right_file.read()).decode()
            )

            # Set xmpNote properties and write right image
            lxmp.set_property(
                XMP_NOTE, "HasExtendedXMP", "06A56CB0A1A7FAFDA459CA3FAA14B474"
            )

            if not xmpfile.can_put_xmp(lxmp):
                raise ValueError(f"Cannot put XMP to {in_path}")

            xmpfile.put_xmp(lxmp)
            xmpfile.close_file()


@app.command()
def swap(
    in_paths: Annotated[list[Path], typer.Argument(help="Image paths")],
) -> None:
    """Swap left and right images."""
    for in_path in in_paths:
        out_path = in_path.with_suffix(f".swap{in_path.suffix}")
        image = cv.imread(in_path.as_posix())
        left, right = image[:, : image.shape[1] // 2], image[:, image.shape[1] // 2 :]
        image_swapped = np.hstack([right, left])
        cv.imwrite(out_path.as_posix(), image_swapped)
