from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose
from quaternion import (
    allclose,
    from_euler_angles,
    from_rotation_vector,
    quaternion,
    rotate_vectors,
)

from vr180_convert import (
    EquirectangularEncoder,
    Euclidean3DRotator,
    FisheyeDecoder,
    FisheyeEncoder,
    apply,
    apply_lr,
)
from vr180_convert.remapper import rotation_match
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
    generate_test_image(256, _TEST_IMAGE_PATH)


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
        size_output=(256, 256),
    )


@pytest.mark.parametrize(
    "transformer",
    [
        Euclidean3DRotator(from_euler_angles(0.0, np.pi / 4, 0.0)),
        PolynomialScaler([0, 1, -0.1]),
    ],
)
def test_transformer(transformer: TransformerBase) -> None:
    apply(
        FisheyeEncoder("equidistant") * transformer * FisheyeDecoder("equidistant"),
        in_paths=_TEST_IMAGE_PATH,
        out_paths=_TEST_DIR / f"test.transformer.{transformer.__class__.__name__}.jpg",
        radius="max",
        size_output=(256, 256),
    )


@pytest.mark.parametrize(
    "transformer",
    [
        Euclidean3DRotator(from_euler_angles(0.0, np.pi / 4, 0.0)),
        PolynomialScaler(),
    ],
)
def test_lr(transformer: TransformerBase) -> None:
    apply_lr(
        EquirectangularEncoder() * transformer * FisheyeDecoder("equidistant"),
        left_path=_TEST_IMAGE_PATH,
        right_path=_TEST_IMAGE_PATH,
        out_path=_TEST_DIR / f"test.lr.{transformer.__class__.__name__}.jpg",
        radius="max",
        size_output=(256, 256),
    )


def test_equidistant_3d():
    x = np.random.rand(101, 100)
    y = np.random.rand(101, 100)
    assert_allclose(equidistant_from_3d(equidistant_to_3d(x, y)), (x, y))


@pytest.mark.parametrize(
    "rotation",
    [
        from_rotation_vector([0.1, 0.2, 0.3]),
    ],
)
def test_rotation_match(rotation: quaternion) -> None:
    random_points = np.random.rand(100, 3)
    random_points_rotated = rotate_vectors(rotation, random_points)
    rotation_est = rotation_match(random_points, random_points_rotated)
    assert allclose(rotation, rotation_est, atol=1e-3) or allclose(
        -rotation, rotation_est, atol=1e-3
    )
