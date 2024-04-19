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

generate_test_image(2048, "test.jpg")


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
        in_paths="test.jpg",
        out_paths=f"test.{format}.jpg",
        radius="max",
    )


def test_rotate() -> None:
    apply(
        FisheyeFormatEncoder("equidistant")
        * Euclidean3DRotator(from_euler_angles(0.0, np.pi / 4, 0.0))
        * FisheyeFormatDecoder("equidistant"),
        in_paths="test.jpg",
        out_paths="test.rotate.jpg",
        radius="max",
    )


def test_lr() -> None:
    apply_lr(
        FisheyeFormatEncoder("equidistant")
        * Euclidean3DRotator(from_euler_angles(0.0, np.pi / 4, 0.0))
        * FisheyeFormatDecoder("equidistant"),
        left_path="test.jpg",
        right_path="test.jpg",
        out_path="test.out.jpg",
        radius="max",
    )


def test_equidistant_3d():
    x = np.random.rand(101, 100)
    y = np.random.rand(101, 100)
    assert_allclose(equidistant_from_3d(equidistant_to_3d(x, y)), (x, y))
