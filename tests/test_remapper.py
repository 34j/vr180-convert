from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose
from quaternion import from_euler_angles

from vr180_convert import (
    EquirectangularEncoder,
    Euclidean3DRotator,
    FisheyeDecoder,
    FisheyeEncoder,
    apply,
    apply_lr,
)
from vr180_convert.testing import generate_test_image
from vr180_convert.transformer import (
    PolynomialScaler,
    TransformerBase,
    equidistant_from_3d,
    equidistant_to_3d,
)

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
        FisheyeEncoder(format)
        if format != "equirectangular"
        else EquirectangularEncoder()
    )
    apply(
        encoder * FisheyeDecoder("equidistant"),
        in_paths=_TEST_IMAGE_PATH,
        out_paths=_TEST_DIR / f"test.format.{format}.jpg",
        radius="max",
    )


@pytest.mark.parametrize(
    "transformer",
    [
        Euclidean3DRotator(from_euler_angles(0.0, np.pi / 4, 0.0)),
        PolynomialScaler([0, 1, -0.1, 0.05]),
    ],
)
def test_transformer(transformer: TransformerBase) -> None:
    apply(
        FisheyeEncoder("equidistant") * transformer * FisheyeDecoder("equidistant"),
        in_paths=_TEST_IMAGE_PATH,
        out_paths=_TEST_DIR / f"test.transformer.{transformer.__class__.__name__}.jpg",
        radius="max",
    )


def test_lr() -> None:
    apply_lr(
        FisheyeEncoder("equidistant")
        * Euclidean3DRotator(from_euler_angles(0.0, np.pi / 4, 0.0))
        * FisheyeDecoder("equidistant"),
        left_path=_TEST_IMAGE_PATH,
        right_path=_TEST_IMAGE_PATH,
        out_path=_TEST_DIR / "test.lr.jpg",
        radius="max",
    )


def test_equidistant_3d():
    x = np.random.rand(101, 100)
    y = np.random.rand(101, 100)
    assert_allclose(equidistant_from_3d(equidistant_to_3d(x, y)), (x, y))
