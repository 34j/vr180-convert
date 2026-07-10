from pathlib import Path

import cv2 as cv
import numpy as np
import pytest
from array_api.latest import Array

from vr180_convert.divide import Concater
from vr180_convert.remapper.equidistant import (
    EquirectangularEncoder,
)
from vr180_convert.remapper.fisheye import FisheyeDecoder
from vr180_convert.remapper.normalize import NormalizeRemapper
from vr180_convert.remapper.radius import AutoDenormalizeRemapper
from vr180_convert.remapper.rotation_match import RotationMatchRemapper
from vr180_convert.remapper.transformer import RemapperTransformer

_TEST_DIR = Path("tests/.cache")


@pytest.fixture(scope="session", params=["numpy", "torch"])
def image(request: pytest.FixtureRequest) -> Array:
    if request.param == "numpy":
        import array_api_compat.numpy as xp
    elif request.param == "torch":
        import array_api_compat.torch as xp
    img = xp.stack(
        (
            xp.asarray(cv.imread("tests/assets/001L.JPG")),
            xp.asarray(cv.imread("tests/assets/001R.JPG")),
        ),
        axis=0,
    )
    return xp.astype(img, xp.float64)


def test_real(
    image: Array,
) -> None:
    t = (
        RemapperTransformer(
            remappers=[
                AutoDenormalizeRemapper(strategy="auto"),
                FisheyeDecoder("equidistant"),
                RotationMatchRemapper(),
                EquirectangularEncoder(),
                NormalizeRemapper(),
            ],
            size_output=(1024, 1024),
        )
        * Concater()
    )
    image = t.transform(image)
    # save image
    cv.imwrite(
        (_TEST_DIR / "test.real.jpg").as_posix(),
        np.asarray(image).astype(np.float32),
    )
