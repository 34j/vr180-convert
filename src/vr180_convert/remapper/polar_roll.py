from abc import abstractmethod
from typing import Any

import attrs
import numpy as np
from ivy import Array

from vr180_convert.remapper.base import RemapperBase


@attrs.define()
class PolarRollRemapper(RemapperBase):
    """Transform using polar coordinates."""

    @abstractmethod
    def transform_polar(
        self, theta: Array, roll: Array, **kwargs: Any
    ) -> tuple[Array, Array]:
        """
        Transform using polar coordinates.

        Parameters
        ----------
        theta : Array
            The distance or angle from the center (front-facing direction)
        roll : Array
            The angle around the center (front-facing direction)
        **kwargs : Any
            Any additional keyword arguments.

        Returns
        -------
        tuple[Array, Array]
            theta and roll after transformation.

        """
        pass

    @abstractmethod
    def inverse_transform_polar(
        self, theta: Array, roll: Array, **kwargs: Any
    ) -> tuple[Array, Array]:
        """
        Inverse transform using polar coordinates.

        Parameters
        ----------
        theta : Array
            The distance or angle from the center (front-facing direction)
        roll : Array
            The angle around the center (front-facing direction)
        **kwargs : Any
            Any additional keyword arguments.

        Returns
        -------
        tuple[Array, Array]
            theta and roll after transformation.

        """
        pass

    def remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        theta = np.sqrt(x**2 + y**2)
        roll = np.arctan2(y, x)
        theta, roll = self.transform_polar(theta, roll, **kwargs)
        x = theta * np.cos(roll)
        y = theta * np.sin(roll)
        return x, y

    def inverse_remap(
        self, x: Array, y: Array, /, **kwargs: Any
    ) -> tuple[Array, Array]:
        theta = np.sqrt(x**2 + y**2)
        roll = np.arctan2(y, x)
        theta, roll = self.inverse_transform_polar(theta, roll, **kwargs)
        x = theta * np.cos(roll)
        y = theta * np.sin(roll)
        return x, y
