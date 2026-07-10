from typing import Any

import array_api_compat
import attrs
from array_api.latest import Array

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
    xp = array_api_compat.array_namespace(x, y)
    phi = xp.atan2(x, y)
    theta = xp.sqrt(x**2 + y**2)
    v = xp.stack(
        [xp.sin(theta) * xp.sin(phi), xp.sin(theta) * xp.cos(phi), xp.cos(theta)],
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
    xp = array_api_compat.array_namespace(v)
    theta = xp.acos(v[..., 2])
    phi = xp.atan2(v[..., 0], v[..., 1])
    x = theta * xp.sin(phi)
    y = theta * xp.cos(phi)
    return x, y


@attrs.define()
class EquirectangularEncoder(RemapperBase):
    """Encodes equirectangular image."""

    is_latitude_y: bool = True
    """Whether latitude is encoded in y axis."""

    def remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        xp = array_api_compat.array_namespace(x, y)
        # latitude: 日本語で緯度, phi
        # longitude: 日本語で経度, theta
        if self.is_latitude_y:
            theta_lat = y * (xp.pi / 2)
            phi_lon = x * (xp.pi / 2)
            v = xp.stack(
                [
                    xp.cos(theta_lat) * xp.sin(phi_lon),
                    xp.sin(theta_lat),
                    xp.cos(theta_lat) * xp.cos(phi_lon),
                ],
                axis=-1,
            )
        else:
            theta_lat = x * (xp.pi / 2)
            phi_lon = y * (xp.pi / 2)
            v = xp.stack(
                [
                    xp.sin(theta_lat),
                    xp.cos(theta_lat) * xp.sin(phi_lon),
                    xp.cos(theta_lat) * xp.cos(phi_lon),
                ],
                axis=-1,
            )

        return equidistant_from_3d(v)

    def inverse_remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        xp = array_api_compat.array_namespace(x, y)
        v = equidistant_to_3d(x, y)
        if self.is_latitude_y:
            theta_lat = xp.asin(v[..., 1])
            phi_lon = xp.atan2(v[..., 0], v[..., 2])
            x = phi_lon / (xp.pi / 2)
            y = theta_lat / (xp.pi / 2)
        else:
            theta_lat = xp.asin(v[..., 0])
            phi_lon = xp.atan2(v[..., 1], v[..., 2])
            x = theta_lat / (xp.pi / 2)
            y = phi_lon / (xp.pi / 2)
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
