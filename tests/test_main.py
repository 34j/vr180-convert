
import os
os.environ["OPENBLAS_NUM_THREADS"] = "-1"
os.environ["MKL_NUM_THREADS"] = "-1"
os.environ["VECLIB_NUM_THREADS"] = "-1"
from typing import Literal


from vr180_convert import EquirectangularFormatEncoder, Euclidean3DRotator, FisheyeFormatDecoder, FisheyeFormatEncoder, apply, apply_lr
from vr180_convert.transformer import equidistant_from_3d, equidistant_to_3d
from vr180_convert.testing import generate_test_image
from numpy.testing import assert_allclose
import numpy as np
from quaternion import from_euler_angles
import pytest

generate_test_image(2048, "test.jpg")

@pytest.mark.parametrize("format", ["rectilinear", "stereographic",
                                    "equidistant", "equisolid", "orthographic", "equirectangular"])
def test_apply(format: Literal["rectilinear", "stereographic", "equidistant", "equisolid", "orthographic", "equirectangular"]):
    encoder = FisheyeFormatEncoder(format) if format != "equirectangular" else EquirectangularFormatEncoder()
    apply(
         encoder * FisheyeFormatDecoder("equidistant") ,
         in_paths="test.jpg",
        out_paths= f"test.{format}.jpg",
        radius="max"
    )
    
def test_rotate():
    apply(FisheyeFormatEncoder("equidistant") * Euclidean3DRotator(from_euler_angles(0.0, np.pi / 4, 0.0)) * FisheyeFormatDecoder("equidistant") ,
        in_paths="test.jpg",
        out_paths= "test.rotate.jpg",
        
        radius="max"
    )

def test_lr():
    apply_lr(
        FisheyeFormatEncoder("equidistant") * Euclidean3DRotator(from_euler_angles(0.0, np.pi / 4, 0.0)) * FisheyeFormatDecoder("equidistant") ,
       left_path= "test.jpg",
      right_path=  "test.jpg",
      out_path=  "test.out.jpg",
        radius = "max"
    )

def test_equidistant_3d():
    x = np.random.rand(101, 100)
    y = np.random.rand(101, 100)
    assert_allclose(equidistant_from_3d(equidistant_to_3d(x, y)), (x, y))