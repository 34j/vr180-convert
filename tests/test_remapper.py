from pathlib import Path
from typing import Literal

import cv2 as cv
import ivy
import numpy as np
import pytest
from ivy import Array

from vr180_convert.remapper.equidistant import (
    EquirectangularEncoder,
)
from vr180_convert.remapper.fisheye import FisheyeDecoder, FisheyeEncoder
from vr180_convert.remapper.normalize import NormalizeRemapper
from vr180_convert.remapper.radius import AutoDenormalizeRemapper
from vr180_convert.remapper.transformer import RemapperTransformer
from vr180_convert.testing import generate_test_image

_TEST_DIR = Path("tests/.cache")
_TEST_IMAGE_PATH = _TEST_DIR / "test.jpg"


@pytest.fixture(scope="session", autouse=True)
def image() -> Array:
    _TEST_DIR.mkdir(exist_ok=True)
    return generate_test_image(256, _TEST_IMAGE_PATH)


@pytest.mark.parametrize(
    "format",
    [
        "rectilinear",
        "stereographic",
        "equidistant",
        "equisolid",
        "orthographic",
        "equirectangular",
    ],
)
def test_apply(
    format: Literal[
        "rectilinear",
        "stereographic",
        "equidistant",
        "equisolid",
        "orthographic",
        "equirectangular",
    ],
    image: Array,
) -> None:
    encoder = (
        FisheyeEncoder(format)
        if format != "equirectangular"
        else EquirectangularEncoder()
    )
    t = RemapperTransformer(
        remappers=[
            NormalizeRemapper(),
            encoder,
            FisheyeDecoder("equidistant"),
            AutoDenormalizeRemapper(strategy="max"),
        ],
        size_output=(256, 256),
    )
    image = t.transform(image)
    # save image
    cv.imwrite(
        (_TEST_DIR / f"test.apply.{format}.jpg").as_posix(),
        ivy.to_numpy(image).astype(np.float32),
    )


# @pytest.mark.parametrize(
#     "transformer",
#     [
#         Euclidean3DRotator(from_euler_angles(0.0, np.pi / 4, 0.0)),
#         PolynomialScaler([0, 1, -0.1]),
#     ],
# )
# def test_transformer(transformer: RemapperBase) -> None:
#     apply(
#         FisheyeEncoder("equidistant") * transformer * FisheyeDecoder("equidistant"),
#         in_paths=_TEST_IMAGE_PATH,
#         out_paths=_TEST_DIR / f"test.transformer.{transformer.__class__.__name__}.jpg",
#         radius="max",
#         size_output=(256, 256),
#     )


# @pytest.mark.parametrize(
#     "transformer",
#     [
#         Euclidean3DRotator(from_euler_angles(0.0, np.pi / 4, 0.0)),
#         PolynomialScaler(),
#     ],
# )
# def test_lr(transformer: RemapperBase) -> None:
#     apply_lr(
#         EquirectangularEncoder() * transformer * FisheyeDecoder("equidistant"),
#         left_path=_TEST_IMAGE_PATH,
#         right_path=_TEST_IMAGE_PATH,
#         out_path=_TEST_DIR / f"test.lr.{transformer.__class__.__name__}.jpg",
#         radius="max",
#         size_output=(256, 256),
#     )


# def test_equidistant_3d():
#     x = np.random.rand(101, 100)
#     y = np.random.rand(101, 100)
#     assert_allclose(equidistant_from_3d(equidistant_to_3d(x, y)), (x, y))
