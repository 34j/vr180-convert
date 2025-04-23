from pathlib import Path
from typing import Any, Literal

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


@pytest.fixture(autouse=True, scope="session", params=["numpy", "torch"])
def setup(request: pytest.FixtureRequest) -> None:
    ivy.set_backend(request.param)
    ivy.set_default_dtype(ivy.float64)


@pytest.fixture(scope="session", autouse=True)
def image(setup: Any) -> Array:
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
def test_fisheye(
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
            AutoDenormalizeRemapper(strategy="max"),
            FisheyeDecoder("equidistant"),
            encoder,
            NormalizeRemapper(),
        ],
        size_output=(256, 256),
    )
    image = t.transform(image)
    # save image
    cv.imwrite(
        (_TEST_DIR / f"test.fisheye.{format}.jpg").as_posix(),
        ivy.to_numpy(image).astype(np.float32),
    )
