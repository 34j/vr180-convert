from abc import ABCMeta, abstractmethod
from typing import Any, Literal

import attrs
import numpy as np
from numpy.typing import NDArray
from quaternion import quaternion, rotate_vectors
from sklearn.base import BaseEstimator, TransformerMixin


class TransformerBase(
    BaseEstimator,
    TransformerMixin,
    metaclass=ABCMeta,
):
    """Base class for transformers."""

    # def fit(self, image: NDArray, **kwargs: Any) -> None:
    #     pass

    @abstractmethod
    def transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        """
        Transform the input coordinates.

        Parameters
        ----------
        x : NDArray
            x (left-right) coordinates.
        y : NDArray
            y (up-down) coordinates.
        **kwargs : Any
            Any additional keyword arguments.

        Returns
        -------
        tuple[NDArray, NDArray]
            x and y coordinates after transformation.

        """
        pass

    # @abstractmethod
    # def inverse_transform(self, x: NDArray, y: NDArray, **kwargs: Any)
    # -> tuple[NDArray, NDArray]:
    #     pass

    def __mul__(self, other: "TransformerBase") -> "MultiTransformer":
        """Multiply two transformers together."""
        if isinstance(self, MultiTransformer) and isinstance(other, MultiTransformer):
            return MultiTransformer(
                transformers=[*self.transformers, *other.transformers]
            )
        if isinstance(self, MultiTransformer):
            return MultiTransformer(transformers=[*self.transformers, other])
        if isinstance(other, MultiTransformer):
            return MultiTransformer(transformers=[self, *other.transformers])
        return MultiTransformer(transformers=[self, other])


@attrs.define()
class MultiTransformer(TransformerBase):
    """A transformer that applies multiple transformers in sequence."""

    transformers: list[TransformerBase]

    def transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        print(f"{y[:, y.shape[0] // 2].max()=}, {x[x.shape[1] // 2, :].max()=}")

        for transformer in self.transformers:
            x, y = transformer.transform(x, y, **kwargs)
            print(
                f"{transformer=}, {y[:, y.shape[0] // 2].max()=}, "
                f"{x[x.shape[1] // 2, :].max()=}"
            )
        return x, y

    # def inverse_transform(self, x: NDArray, y: NDArray,
    # **kwargs: Any) -> tuple[NDArray, NDArray]:
    #     for transformer in reversed(self.transformers):
    #         x, y = transformer.inverse_transform(x, y, **kwargs)
    #     return x, y


def get_radius(input: NDArray, *, threshold: int = 10) -> float:
    """
    Estimate the radius of the circle in the image.

    Parameters
    ----------
    input : NDArray
        The input image.
    threshold : int, optional
        The threshold to determine if a pixel is black, by default 10

    Returns
    -------
    float
        The estimated radius.

    """
    height = input.shape[0]
    center_row = input[height // 2, :, :]
    center_row_is_black = np.mean(center_row, axis=-1) < threshold
    center_row_is_black_deriv = np.diff(center_row_is_black.astype(int))

    # first and last 1 in the derivative
    center_row_black_start = np.where(center_row_is_black_deriv == 1)[0][0]
    center_row_black_end = np.where(center_row_is_black_deriv == -1)[0][-1]
    radius = (center_row_black_end - center_row_black_start) / 2
    return radius


@attrs.define()
class NormalizeTransformer(TransformerBase):
    center: tuple[float, float] | None = None
    scale: tuple[float, float] | None = None

    def transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        center = self.center or (x.shape[1] / 2, x.shape[0] / 2)
        scale = self.scale or (x.shape[1], x.shape[0])
        x = (x - center[0]) / scale[0] * 2
        y = (y - center[1]) / scale[1] * 2
        return x, y

    # def inverse_transform(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    #     center = self.center or (x.shape[1] / 2, x.shape[0] / 2)
    #     scale = self.scale or (x.shape[1] / 2, x.shape[0] / 2)
    #     x = x * scale[0] + center[0]
    #     y = y * scale[1] + center[1]
    #     return x, y


# @attrs.define()
# class AutoDetectRadiusNormalizeTransformer(NormalizeTransformer):
#     def fit(self, image: NDArray, **kwargs: Any) -> None:
#         radius = get_radius(image)
#         self.center = (image.shape[1] // 2, image.shape[0] // 2)
#         self.scale = (radius, radius)


@attrs.define()
class DenormalizeTransformer(TransformerBase):
    scale: tuple[float, float]
    center: tuple[float, float]

    def transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        scale = self.scale
        center = self.center
        x = x * scale[0] + center[0]
        y = y * scale[1] + center[1]
        return x, y

    # def inverse_transform(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    #     scale = self.scale
    #     center = self.center
    #     x = (x - center[0]) / scale[0]
    #     y = (y - center[1]) / scale[1]
    #     return x, y


@attrs.define()
class PolarRollTransformer(TransformerBase):
    @abstractmethod
    def transform_polar(
        self, theta: NDArray, roll: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        pass

    def transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        theta = np.sqrt(x**2 + y**2)
        roll = np.arctan2(y, x)
        theta, roll = self.transform_polar(theta, roll, **kwargs)
        x = theta * np.cos(roll)
        y = theta * np.sin(roll)
        return x, y


@attrs.define()
class FisheyeFormatEncoder(PolarRollTransformer):
    mapping_type: Literal[
        "rectilinear", "stereographic", "equidistant", "equisolid", "orthographic"
    ]

    def transform_polar(
        self, theta: NDArray, roll: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        """[-1, 1] -> [-pi/2, pi/2]."""
        if self.mapping_type == "rectilinear":
            return np.arctan(theta), roll
        elif self.mapping_type == "stereographic":
            return 2 * np.arctan(theta), roll
        elif self.mapping_type == "equidistant":
            return theta * (np.pi / 2), roll
        elif self.mapping_type == "equisolid":
            return 2 * np.arcsin(theta / np.sqrt(2)), roll
        elif self.mapping_type == "orthographic":
            return np.arcsin(theta), roll
        else:
            raise ValueError(f"Unknown mapping type: {self.mapping_type}")


@attrs.define()
class FisheyeFormatDecoder(PolarRollTransformer):
    mapping_type: Literal[
        "rectilinear", "stereographic", "equidistant", "equisolid", "orthographic"
    ]

    def transform_polar(
        self, theta: NDArray, roll: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        """[-pi/2, pi/2] -> [-1, 1]."""
        if self.mapping_type == "rectilinear":
            return np.tan(theta), roll
        elif self.mapping_type == "stereographic":
            return 2 * np.tan(theta / 2), roll
        elif self.mapping_type == "equidistant":
            return theta / (np.pi / 2), roll
        elif self.mapping_type == "equisolid":
            return np.sqrt(2) * np.sin(theta / 2), roll
        elif self.mapping_type == "orthographic":
            return np.sin(theta), roll
        else:
            raise ValueError(f"Unknown mapping type: {self.mapping_type}")


@attrs.define()
class ZoomTransformer(TransformerBase):
    scale: float

    def transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        x = x * self.scale
        y = y * self.scale
        return x, y


def equidistant_to_3d(x: NDArray, y: NDArray) -> NDArray:
    """
    Convert 2D coordinates to 3D unit vector.

    z axis is forward, x axis is right, y axis is up.

    Parameters
    ----------
    x : NDArray
        The x coordinate in equidistant fisheye format.
    y : NDArray
        The y coordinate in equidistant fisheye format.

    Returns
    -------
    NDArray
        The 3D unit vector.

    """
    phi = np.arctan2(x, y)
    theta = np.sqrt(x**2 + y**2)
    v = np.stack(
        [np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi), np.cos(theta)],
        axis=-1,
    )
    return v


def equidistant_from_3d(v: NDArray) -> tuple[NDArray, NDArray]:
    """
    Convert 3D unit vector to 2D coordinates.

    Parameters
    ----------
    v : NDArray
        The 3D unit vector.

    Returns
    -------
    tuple[NDArray, NDArray]
        The x and y coordinates in equidistant fisheye format.

    """
    theta = np.arccos(v[..., 2])
    phi = np.arctan2(v[..., 0], v[..., 1])
    x = theta * np.sin(phi)
    y = theta * np.cos(phi)
    return x, y


@attrs.define()
class EquirectangularFormatEncoder(TransformerBase):
    is_latitude_y: bool = True

    def transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
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
            phi_lon = y * np.cos(theta_lat) * (np.pi / 2)
            v = np.stack(
                [
                    np.sin(theta_lat),
                    np.cos(theta_lat) * np.sin(phi_lon),
                    np.cos(theta_lat) * np.cos(phi_lon),
                ],
                axis=-1,
            )

        return equidistant_from_3d(v)


@attrs.define()
class Euclidean3DTransformer(TransformerBase):
    @abstractmethod
    def transform_v(self, v: NDArray) -> NDArray:
        pass

    def transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        v = equidistant_to_3d(x, y)
        print(v.shape)
        v = self.transform_v(v)
        x, y = equidistant_from_3d(v)
        return x, y


@attrs.define()
class Euclidean3DRotator(Euclidean3DTransformer):
    rotation: quaternion

    def transform_v(self, v: NDArray) -> NDArray:
        return rotate_vectors(self.rotation, v)


# @attrs.define()
# class EquirectangularFormatEncoder2(TransformerBase):
#     is_latitude_y: bool = False

#     def transform(
#         self, x: NDArray, y: NDArray, **kwargs: Any
#     ) -> tuple[NDArray, NDArray]:
#         if self.is_latitude_y:
#             y = y * (np.pi / 2)
#             x = x / np.cos(y) * (np.pi / 2)
#         else:
#             x = x * (np.pi / 2)
#             y = y * (np.pi / 2) * np.cos(x)
#         return x, y
# def inverse_transform(
#     self, x: NDArray, y: NDArray, **kwargs: Any
# ) -> tuple[NDArray, NDArray]:
#     v = equidistant_to_3d(x, y)
#     longitude = np.arctan2(v[1], v[0])
#     latitude = np.arcsin(v[2])
#     x = longitude * np.cos(latitude) / (np.pi / 2)
#     y = latitude / (np.pi / 2)
#     return x, y
