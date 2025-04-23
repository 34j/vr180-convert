from pathlib import Path
from typing import Any

import cv2 as cv
import ivy
import numpy as np
import pytest
from ivy import Array

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


@pytest.fixture(autouse=True, scope="session", params=["numpy", "torch"])
def setup(request: pytest.FixtureRequest) -> None:
    ivy.set_backend(request.param)
    ivy.set_default_dtype(ivy.float64)


@pytest.fixture(scope="session", autouse=True)
def image(setup: Any) -> Array:
    return ivy.stack(
        (
            ivy.asarray(cv.imread("tests/assets/001L.JPG")),
            ivy.asarray(cv.imread("tests/assets/001R.JPG")),
        ),
        axis=0,
    )


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
        ivy.to_numpy(image).astype(np.float32),
    )
