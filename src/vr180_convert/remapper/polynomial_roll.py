from collections.abc import Sequence
from typing import Any

import attrs
import numpy as np
from ivy import Array

from vr180_convert.remapper.polar_roll import PolarRollRemapper


@attrs.define()
class PolynomialScaler(PolarRollRemapper):
    """Scale the polar coordinates using polynomial."""

    coefs_reverse: Sequence[float] = [0, 1]
    """The coefficients of the polynomial in reverse order.
    [0, 1] means y = 0 + 1 * x."""

    def transform_polar(
        self, theta: Array, roll: Array, **kwargs: Any
    ) -> tuple[Array, Array]:
        return np.polyval(np.flip(self.coefs_reverse), theta), roll

    def inverse_transform_polar(
        self, theta: Array, roll: Array, **kwargs: Any
    ) -> tuple[Array, Array]:
        raise NotImplementedError(
            "PolynomialScaler does not support inverse transform."
        )
