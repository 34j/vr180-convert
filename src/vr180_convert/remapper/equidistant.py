from typing import Any

import attrs
import numpy as np
from ivy import Array

from vr180_convert.remapper.base import RemapperBase
from vr180_convert.remapper.inverse import InverseRemapper


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
    phi = np.arctan2(x, y)
    theta = np.sqrt(x**2 + y**2)
    v = np.stack(
        [np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi), np.cos(theta)],
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
    theta = np.arccos(v[..., 2])
    phi = np.arctan2(v[..., 0], v[..., 1])
    x = theta * np.sin(phi)
    y = theta * np.cos(phi)
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
            theta_lat = y * (np.pi / 2)
            phi_lon = x * (np.pi / 2)
            v = np.stack(
                [
                    np.cos(theta_lat) * np.sin(phi_lon),
                    np.sin(theta_lat),
                    np.cos(theta_lat) * np.cos(phi_lon),
                ],
                axis=-1,
            )
        else:
            theta_lat = x * (np.pi / 2)
            phi_lon = y * (np.pi / 2)
            v = np.stack(
                [
                    np.sin(theta_lat),
                    np.cos(theta_lat) * np.sin(phi_lon),
                    np.cos(theta_lat) * np.cos(phi_lon),
                ],
                axis=-1,
            )

        return equidistant_from_3d(v)

    def inverse_remap(
        self, x: Array, y: Array, /, **kwargs: Any
    ) -> tuple[Array, Array]:
        v = equidistant_to_3d(x, y)
        if self.is_latitude_y:
            theta_lat = np.arcsin(v[..., 1])
            phi_lon = np.arctan2(v[..., 0], v[..., 2])
            x = phi_lon / (np.pi / 2)
            y = theta_lat / (np.pi / 2)
        else:
            theta_lat = np.arcsin(v[..., 0])
            phi_lon = np.arctan2(v[..., 1], v[..., 2])
            x = theta_lat / (np.pi / 2)
            y = phi_lon / (np.pi / 2)
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
