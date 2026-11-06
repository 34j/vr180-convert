from pathlib import Path
from typing import Literal

import cv2 as cv
import numpy as np
import pytest
from array_api.latest import Array

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


@pytest.fixture(scope="session")
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
    encoder = FisheyeEncoder(format) if format != "equirectangular" else EquirectangularEncoder()
    t = RemapperTransformer(
        remappers=[
            AutoDenormalizeRemapper(strategy="max"),
            FisheyeDecoder("equidistant"),
            encoder,
            NormalizeRemapper(),
        ],
        size_output=(512, 512),
    )
    image = t.transform(image)
    # save image
    cv.imwrite(
        (_TEST_DIR / f"test.fisheye.{format}.jpg").as_posix(),
        np.asarray(image).astype(np.float32),
    )
