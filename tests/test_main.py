import os
from typing import Literal

os.environ["OPENBLAS_NUM_THREADS"] = "-1"
os.environ["MKL_NUM_THREADS"] = "-1"
os.environ["VECLIB_NUM_THREADS"] = "-1"

from vr180_convert.main import EquirectangularFormatEncoder, Euclidean3DRotator, FisheyeFormatDecoder, FisheyeFormatEncoder, apply, apply_lr, equidistant_from_3d, equidistant_to_3d, Euclidean3DTransformer
from numpy.testing import assert_allclose
import numpy as np
from quaternion import from_euler_angles
import pytest


@pytest.mark.parametrize("format", ["rectilinear", "stereographic",
                                    "equidistant", "equisolid", "orthographic", "equirectangular"])
def test_apply(format: Literal["rectilinear", "stereographic", "equidistant", "equisolid", "orthographic", "equirectangular"]):
    encoder = FisheyeFormatEncoder(format) if format != "equirectangular" else EquirectangularFormatEncoder()
    apply(
        "fish2sphere180.jpg",
        f"fish2sphere180.{format}.jpg",
        encoder * FisheyeFormatDecoder("equidistant") ,
        radius="max"
    )
    
def test_rotate():
    apply(
        "fish2sphere180.jpg",
        "fish2sphere180.rotate.jpg",
        FisheyeFormatEncoder("equidistant") * Euclidean3DRotator(from_euler_angles(0.0, np.pi / 4, 0.0)) * FisheyeFormatDecoder("equidistant") ,
        radius="max"
    )

def test_lr():
    apply_lr(
        "0.jpg",
        "10.jpg",
        "test.out.jpg",
        FisheyeFormatEncoder("equidistant") * Euclidean3DRotator(from_euler_angles(0.0, np.pi / 4, 0.0)) * FisheyeFormatDecoder("equidistant") ,
        radius = 2568.5
    )

def test_equidistant_3d():
    x = np.random.rand(101, 100)
    y = np.random.rand(101, 100)
    assert_allclose(equidistant_from_3d(equidistant_to_3d(x, y)), (x, y))