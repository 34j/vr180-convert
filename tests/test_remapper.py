from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose
from quaternion import from_euler_angles

from vr180_convert import (
    EquirectangularFormatEncoder,
    Euclidean3DRotator,
    FisheyeFormatDecoder,
    FisheyeFormatEncoder,
    apply,
    apply_lr,
)
from vr180_convert.testing import generate_test_image
from vr180_convert.transformer import equidistant_from_3d, equidistant_to_3d

_TEST_DIR = Path("tests/.cache")
_TEST_IMAGE_PATH = _TEST_DIR / "test.jpg"


@pytest.fixture(scope="session", autouse=True)
def generate_image():
    _TEST_DIR.mkdir(exist_ok=True)
    generate_test_image(2048, _TEST_IMAGE_PATH)


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
    ]
) -> None:
    encoder = (
        FisheyeFormatEncoder(format)
        if format != "equirectangular"
        else EquirectangularFormatEncoder()
    )
    apply(
        encoder * FisheyeFormatDecoder("equidistant"),
        in_paths=_TEST_IMAGE_PATH,
        out_paths=_TEST_DIR / f"test.format.{format}.jpg",
        radius="max",
    )


def test_rotate() -> None:
    apply(
        FisheyeFormatEncoder("equidistant")
        * Euclidean3DRotator(from_euler_angles(0.0, np.pi / 4, 0.0))
        * FisheyeFormatDecoder("equidistant"),
        in_paths=_TEST_IMAGE_PATH,
        out_paths=_TEST_IMAGE_PATH / "test.rotate.jpg",
        radius="max",
    )


def test_lr() -> None:
    apply_lr(
        FisheyeFormatEncoder("equidistant")
        * Euclidean3DRotator(from_euler_angles(0.0, np.pi / 4, 0.0))
        * FisheyeFormatDecoder("equidistant"),
        left_path=_TEST_IMAGE_PATH,
        right_path=_TEST_IMAGE_PATH,
        out_path=_TEST_DIR / "test.lr.jpg",
        radius="max",
    )


def test_equidistant_3d():
    x = np.random.rand(101, 100)
    y = np.random.rand(101, 100)
    assert_allclose(equidistant_from_3d(equidistant_to_3d(x, y)), (x, y))
