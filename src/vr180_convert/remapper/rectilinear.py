from __future__ import annotations

import warnings
from typing import Any, Literal

import array_api_compat
import attrs
from array_api.latest import Array

from vr180_convert.remapper.inverse import InverseRemapper
from vr180_convert.remapper.polar_roll import PolarRollRemapper


@attrs.define()
class RectilinearEncoder(PolarRollRemapper):
    """
    Encodes a rectilinear (perspective) image into equidistant coordinates.

    Maps normalized image coordinates to angular coordinates using the
    rectilinear (perspective) projection model, taking into account the
    focal length and sensor size.
    """

    focal_length: float
    """The focal length of the lens in mm."""
    sensor_width: Literal["35mm", "APS-H", "APS-C", "APS-C-Canon", "Foveon", "MFT"] | str | float = "35mm"
    """The sensor width of the camera in mm if float,
    or a standard sensor format name if str."""

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
        known_widths = {
            "35mm": 36.0,
            "APS-H": 27.90,
            "APS-C": 23.6,
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
        """Scaling factor derived from focal length and sensor width."""
        return 2 * self.focal_length / self.sensor_width_mm

    def transform_polar(self, theta: Array, roll: Array, **kwargs: Any) -> tuple[Array, Array]:
        """
        Map normalized radius to angular radius (image -> angle).

        For a rectilinear projection: angle = atan(normalized_radius / factor).
        """
        xp = array_api_compat.array_namespace(theta, roll)
        return xp.atan(theta / self.factor), roll

    def inverse_transform_polar(self, theta: Array, roll: Array, **kwargs: Any) -> tuple[Array, Array]:
        """
        Map angular radius to normalized radius (angle -> image).

        For a rectilinear projection: normalized_radius = tan(angle) * factor.
        """
        xp = array_api_compat.array_namespace(theta, roll)
        return xp.tan(theta) * self.factor, roll


def RectilinearDecoder(
    focal_length: float,
    sensor_width: Literal["35mm", "APS-H", "APS-C", "APS-C-Canon", "Foveon", "MFT"] | str | float = "35mm",
) -> InverseRemapper[RectilinearEncoder]:
    """
    Decode equidistant coordinates back to a rectilinear (perspective) image.

    Parameters
    ----------
    focal_length : float
        The focal length of the lens in mm.
    sensor_width : str or float
        The sensor width in mm, or a standard format name.

    Returns
    -------
    InverseRemapper[RectilinearEncoder]
        The rectilinear decoder.

    """
    return InverseRemapper(RectilinearEncoder(focal_length=focal_length, sensor_width=sensor_width))
