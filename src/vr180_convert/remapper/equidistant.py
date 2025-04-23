from typing import Any

import attrs
import ivy
from ivy import Array

from .base import RemapperBase
from .inverse import InverseRemapper


def equidistant_to_3d(x: Array, y: Array) -> Array:
    """
    Convert 2D coordinates to 3D unit vector.

    z axis is forward, x axis is right, y axis is up.

    Parameters
    ----------
    x : Array
        The x coordinate in equidistant fisheye format.
    y : Array
        The y coordinate in equidistant fisheye format.

    Returns
    -------
    Array
        The 3D unit vector.

    """
    phi = ivy.atan2(x, y)
    theta = ivy.sqrt(x**2 + y**2)
    v = ivy.stack(
        [ivy.sin(theta) * ivy.sin(phi), ivy.sin(theta) * ivy.cos(phi), ivy.cos(theta)],
        axis=-1,
    )
    return v


def equidistant_from_3d(v: Array) -> tuple[Array, Array]:
    """
    Convert 3D unit vector to 2D coordinates.

    Parameters
    ----------
    v : Array
        The 3D unit vector.

    Returns
    -------
    tuple[Array, Array]
        The x and y coordinates in equidistant fisheye format.

    """
    theta = ivy.acos(v[..., 2])
    phi = ivy.atan2(v[..., 0], v[..., 1])
    x = theta * ivy.sin(phi)
    y = theta * ivy.cos(phi)
    return x, y


@attrs.define()
class EquirectangularEncoder(RemapperBase):
    """Encodes equirectangular image."""

    is_latitude_y: bool = True
    """Whether latitude is encoded in y axis."""

    def remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        # latitude: 日本語で緯度, phi
        # longitude: 日本語で経度, theta
        if self.is_latitude_y:
            theta_lat = y * (ivy.pi / 2)
            phi_lon = x * (ivy.pi / 2)
            v = ivy.stack(
                [
                    ivy.cos(theta_lat) * ivy.sin(phi_lon),
                    ivy.sin(theta_lat),
                    ivy.cos(theta_lat) * ivy.cos(phi_lon),
                ],
                axis=-1,
            )
        else:
            theta_lat = x * (ivy.pi / 2)
            phi_lon = y * (ivy.pi / 2)
            v = ivy.stack(
                [
                    ivy.sin(theta_lat),
                    ivy.cos(theta_lat) * ivy.sin(phi_lon),
                    ivy.cos(theta_lat) * ivy.cos(phi_lon),
                ],
                axis=-1,
            )

        return equidistant_from_3d(v)

    def inverse_remap(
        self, x: Array, y: Array, /, **kwargs: Any
    ) -> tuple[Array, Array]:
        v = equidistant_to_3d(x, y)
        if self.is_latitude_y:
            theta_lat = ivy.asin(v[..., 1])
            phi_lon = ivy.atan2(v[..., 0], v[..., 2])
            x = phi_lon / (ivy.pi / 2)
            y = theta_lat / (ivy.pi / 2)
        else:
            theta_lat = ivy.asin(v[..., 0])
            phi_lon = ivy.atan2(v[..., 1], v[..., 2])
            x = theta_lat / (ivy.pi / 2)
            y = phi_lon / (ivy.pi / 2)
        return x, y


def EquirectangularDecoder(
    is_latitude_y: bool = True,
) -> InverseRemapper[EquirectangularEncoder]:
    """
    Decodes equirectangular image.

    Parameters
    ----------
    is_latitude_y : bool, optional
        Whether latitude is encoded in y axis, by default True

    Returns
    -------
    InverseRemapper[EquirectangularEncoder]
        The equirectangular decoder.

    """
    return InverseRemapper(EquirectangularEncoder(is_latitude_y))
