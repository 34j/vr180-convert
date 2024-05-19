from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, Generic, Literal, Sequence, TypeVar

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

    @abstractmethod
    def inverse_transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        """
        Inverse transform the input coordinates.

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

    def __mul__(self, other: TransformerBase) -> MultiTransformer:
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


T = TypeVar("T", bound=TransformerBase)


@attrs.define()
class MultiTransformer(TransformerBase):
    """A transformer that applies multiple transformers in sequence."""

    transformers: list[TransformerBase]

    def transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        for transformer in self.transformers:
            x, y = transformer.transform(x, y, **kwargs)
        return x, y

    def inverse_transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        for transformer in reversed(self.transformers):
            x, y = transformer.inverse_transform(x, y, **kwargs)
        return x, y


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
    height, width = input.shape[:2]
    if width > height:
        center_row = input[height // 2, :, :]
    else:
        center_row = input[:, width // 2, :]
    del height, width

    # determine if a pixel is black
    center_row_is_black = np.mean(center_row, axis=-1) < threshold
    center_row_is_black_deriv = np.diff(center_row_is_black.astype(int))

    # first and last 1 in the derivative
    center_row_black_start = np.where(center_row_is_black_deriv == 1)[0][0]
    center_row_black_end = np.where(center_row_is_black_deriv == -1)[0][-1]
    radius = (center_row_black_end - center_row_black_start) / 2
    return radius


@attrs.define()
class NormalizeTransformer(TransformerBase):
    """Normalize the coordinates to [-1, 1]."""

    center: tuple[float, float] | None = None
    """The center of the image. If None, the center is the center of the image."""
    scale: tuple[float, float] | Literal["min", "max"] | None = None
    """The scale of the image. If "min" or None, the scale is the minimum of the width and height.
    If "max", the scale is the maximum of the width and height."""

    def transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        center = self.center or (x.shape[1] / 2, x.shape[0] / 2)
        scale = (
            min(x.shape[1], x.shape[0])
            if self.scale in ["min", None]
            else max(x.shape[1], x.shape[0]) if self.scale == "max" else self.scale
        )
        x = (x - center[0]) / scale * 2
        y = (y - center[1]) / scale * 2
        return x, y

    def inverse_transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        center = self.center or (x.shape[1] / 2, x.shape[0] / 2)
        scale = (
            min(x.shape[1], x.shape[0])
            if self.scale in ["min", None]
            else max(x.shape[1], x.shape[0]) if self.scale == "max" else self.scale
        )
        x = x * scale[0] + center[0]
        y = y * scale[1] + center[1]
        return x, y


# @attrs.define()
# class AutoDetectRadiusNormalizeTransformer(NormalizeTransformer):
#     def fit(self, image: NDArray, **kwargs: Any) -> None:
#         radius = get_radius(image)
#         self.center = (image.shape[1] // 2, image.shape[0] // 2)
#         self.scale = (radius, radius)


@attrs.define()
class DenormalizeTransformer(TransformerBase):
    """Denormalize the coordinates from [-1, 1] to the original image size."""

    scale: tuple[float, float]
    """The scale of the image. Recommended to be the half of the width and height of the result image."""
    center: tuple[float, float]
    """The center of the image. Recommended to be the center of the result image."""

    def transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        scale = self.scale
        center = self.center
        x = x * scale[0] + center[0]
        y = y * scale[1] + center[1]
        return x, y

    def inverse_transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        scale = self.scale
        center = self.center
        x = (x - center[0]) / scale[0]
        y = (y - center[1]) / scale[1]
        return x, y


@attrs.define()
class PolarRollTransformer(TransformerBase):
    """Transform using polar coordinates."""

    @abstractmethod
    def transform_polar(
        self, theta: NDArray, roll: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        """
        Transform using polar coordinates.

        Parameters
        ----------
        theta : NDArray
            The distance or angle from the center (front-facing direction)
        roll : NDArray
            The angle around the center (front-facing direction)
        **kwargs : Any
            Any additional keyword arguments.

        Returns
        -------
        tuple[NDArray, NDArray]
            theta and roll after transformation.

        """
        pass

    @abstractmethod
    def inverse_transform_polar(
        self, theta: NDArray, roll: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        """
        Inverse transform using polar coordinates.

        Parameters
        ----------
        theta : NDArray
            The distance or angle from the center (front-facing direction)
        roll : NDArray
            The angle around the center (front-facing direction)
        **kwargs : Any
            Any additional keyword arguments.

        Returns
        -------
        tuple[NDArray, NDArray]
            theta and roll after transformation.

        """
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

    def inverse_transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        theta = np.sqrt(x**2 + y**2)
        roll = np.arctan2(y, x)
        theta, roll = self.inverse_transform_polar(theta, roll, **kwargs)
        x = theta * np.cos(roll)
        y = theta * np.sin(roll)
        return x, y


@attrs.define()
class RectilinearDecoder(PolarRollTransformer):
    """Encodes rectilinear image."""

    # https://en.wikipedia.org/wiki/Image_sensor_format
    focal_length: float
    """The focal length of the lens in mm."""
    sensor_width: (
        Literal["35mm", "APS-H", "APS-C", "APS-C-Canon", "Foveon", "MFT"] | str | float
    ) = "35mm"
    """The sensor width of the camera in mm if float,
    or in inches if str, or a standard sensor width if str."""

    @property
    def sensor_width_mm(self) -> float:
        """Sensor width in mm."""
        if self.sensor_width in ["35mm", "APS-C", "1/2.3"]:
            warnings.warn(
                "Sensor size may vary by about 0.2 mm depending on the camera model. "
                "To get very accurate results, consider setting the sensor width in mm manually.",
                UserWarning,
                stacklevel=2,
            )
        # https://en.wikipedia.org/wiki/Image_sensor_format#Table_of_sensor_formats_and_sizes
        known_widths = {
            "35mm": 36.0,  # ~ 35.8mm
            "APS-H": 27.90,
            "APS-C": 23.6,  # ~ 23.7mm
            "APS-C-Canon": 22.30,
            "MFT": 17.30,
            "1": 13.20,
            "1/1.12": 11.43,
            "1/1.2": 10.67,
            "1/1.33": 9.6,
            "1/1.6": 8.08,
            "1/1.7": 7.60,
            "1/1.8": 7.18,
            "1/2": 6.40,
            "1/2.3": 6.17,
        }
        if isinstance(self.sensor_width, str):
            return known_widths[self.sensor_width]
        return self.sensor_width

    @property
    def factor(self) -> float:
        """Zoom factor applied after tan."""
        return 2 * self.focal_length / self.sensor_width_mm

    def transform_polar(
        self, theta: NDArray, roll: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        return np.tan(theta) * self.factor, roll

    def inverse_transform_polar(
        self, theta: NDArray, roll: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        # fov = 2 arctan sensor_width / (2 * focal_length)
        return np.arctan(theta / self.factor), roll


@attrs.define()
class FisheyeEncoder(PolarRollTransformer):
    """Encodes fisheye image."""

    mapping_type: Literal[
        "rectilinear", "stereographic", "equidistant", "equisolid", "orthographic"
    ]
    """The mapping type of the fisheye image."""

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
            raise ValueError(
                f"Unknown mapping type: {self.mapping_type}, "
                "should be one of 'rectilinear', 'stereographic', 'equidistant', 'equisolid', 'orthographic'."
            )

    def inverse_transform_polar(
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
            raise ValueError(
                f"Unknown mapping type: {self.mapping_type}, "
                "should be one of 'rectilinear', 'stereographic', 'equidistant', 'equisolid', 'orthographic'."
            )


@attrs.define()
class InverseTransformer(TransformerBase, Generic[T]):
    """Transformer that calls inverse_transform() in transform() and vice versa."""

    transformer: T
    """The transformer to be inverted."""

    def transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        return self.transformer.inverse_transform(x, y, **kwargs)

    def inverse_transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        return self.transformer.transform(x, y, **kwargs)


def FisheyeDecoder(
    mapping_type: Literal[
        "rectilinear", "stereographic", "equidistant", "equisolid", "orthographic"
    ]
) -> InverseTransformer[FisheyeEncoder]:
    """
    Decodes fisheye image.

    Parameters
    ----------
    mapping_type : Literal['rectilinear', 'stereographic', 'equidistant', 'equisolid', 'orthographic']
        The mapping type of the fisheye image.

    Returns
    -------
    InverseTransformer
        The fisheye decoder.

    """
    return InverseTransformer(FisheyeEncoder(mapping_type))


@attrs.define()
class PolynomialScaler(PolarRollTransformer):
    """Scale the polar coordinates using polynomial."""

    coefs_reverse: Sequence[float] = [0, 1]
    """The coefficients of the polynomial in reverse order.
    [0, 1] means y = 0 + 1 * x."""

    def transform_polar(
        self, theta: NDArray, roll: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        return np.polyval(np.flip(self.coefs_reverse), theta), roll

    def inverse_transform_polar(
        self, theta: NDArray, roll: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        raise NotImplementedError(
            "PolynomialScaler does not support inverse transform."
        )


@attrs.define()
class ZoomTransformer(TransformerBase):
    """Zoom the image."""

    scale: float
    """The zoom scale."""

    def transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        x = x / self.scale
        y = y / self.scale
        return x, y

    def inverse_transform(
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
class EquirectangularEncoder(TransformerBase):
    """Encodes equirectangular image."""

    is_latitude_y: bool = True
    """Whether latitude is encoded in y axis."""

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

    def inverse_transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
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
) -> InverseTransformer[EquirectangularEncoder]:
    """
    Decodes equirectangular image.

    Parameters
    ----------
    is_latitude_y : bool, optional
        Whether latitude is encoded in y axis, by default True

    Returns
    -------
    InverseTransformer[EquirectangularEncoder]
        The equirectangular decoder.

    """
    return InverseTransformer(EquirectangularEncoder(is_latitude_y))


@attrs.define()
class Euclidean3DTransformer(TransformerBase):
    """Transform as 3D unit vector."""

    @abstractmethod
    def transform_v(self, v: NDArray) -> NDArray:
        """
        Transform 3D unit vector.

        Parameters
        ----------
        v : NDArray
            The 3D unit vector.
            z axis is forward, x axis is right, y axis is up.

        Returns
        -------
        NDArray
            The transformed 3D unit vector.
            z axis is forward, x axis is right, y axis is up.

        """
        pass

    @abstractmethod
    def inverse_transform_v(self, v: NDArray) -> NDArray:
        """
        Inverse transform 3D unit vector.

        Parameters
        ----------
        v : NDArray
            The 3D unit vector.
            z axis is forward, x axis is right, y axis is up.

        Returns
        -------
        NDArray
            The inverse transformed 3D unit vector.
            z axis is forward, x axis is right, y axis is up.

        """
        pass

    def transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        v = equidistant_to_3d(x, y)
        v = self.transform_v(v)
        x, y = equidistant_from_3d(v)
        return x, y

    def inverse_transform(
        self, x: NDArray, y: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        v = equidistant_to_3d(x, y)
        v = self.transform_v(v)
        x, y = equidistant_from_3d(v)
        return x, y


@attrs.define()
class Euclidean3DRotator(Euclidean3DTransformer):
    """Rotate as 3D unit vector."""

    rotation: quaternion
    """The rotation quaternion."""

    def transform_v(self, v: NDArray) -> NDArray:
        return rotate_vectors(self.rotation, v)

    def inverse_transform_v(self, v: NDArray) -> NDArray:
        return rotate_vectors(self.rotation.inverse(), v)
