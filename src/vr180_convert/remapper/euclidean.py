from abc import abstractmethod
from typing import Any

import attrs
from ivy import Array
from quaternion import quaternion, rotate_vectors

from vr180_convert.remapper.base import RemapperBase
from vr180_convert.remapper.equidistant import equidistant_from_3d, equidistant_to_3d


@attrs.define()
class Euclidean3DRemapper(RemapperBase):
    """Transform as 3D unit vector."""

    @abstractmethod
    def transform_v(self, v: Array) -> Array:
        """
        Transform 3D unit vector.

        Parameters
        ----------
        v : Array
            The 3D unit vector.
            z axis is forward, x axis is right, y axis is up.

        Returns
        -------
        Array
            The transformed 3D unit vector.
            z axis is forward, x axis is right, y axis is up.

        """
        pass

    @abstractmethod
    def inverse_transform_v(self, v: Array) -> Array:
        """
        Inverse transform 3D unit vector.

        Parameters
        ----------
        v : Array
            The 3D unit vector.
            z axis is forward, x axis is right, y axis is up.

        Returns
        -------
        Array
            The inverse transformed 3D unit vector.
            z axis is forward, x axis is right, y axis is up.

        """
        pass

    def remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        v = equidistant_to_3d(x, y)
        v = self.transform_v(v)
        x, y = equidistant_from_3d(v)
        return x, y

    def inverse_remap(
        self, x: Array, y: Array, /, **kwargs: Any
    ) -> tuple[Array, Array]:
        v = equidistant_to_3d(x, y)
        v = self.transform_v(v)
        x, y = equidistant_from_3d(v)
        return x, y


@attrs.define()
class Euclidean3DRotator(Euclidean3DRemapper):
    """Rotate as 3D unit vector."""

    rotation: quaternion
    """The rotation quaternion."""

    def transform_v(self, v: Array) -> Array:
        return rotate_vectors(self.rotation, v)

    def inverse_transform_v(self, v: Array) -> Array:
        return rotate_vectors(self.rotation.inverse(), v)
