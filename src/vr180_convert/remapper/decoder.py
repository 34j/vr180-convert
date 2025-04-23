import warnings
from typing import Any, Literal

import attrs
import ivy
from ivy import Array

from vr180_convert.remapper.polar_roll import PolarRollRemapper


@attrs.define()
class RectilinearDecoder(PolarRollRemapper):
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
        self, theta: Array, roll: Array, **kwargs: Any
    ) -> tuple[Array, Array]:
        return ivy.tan(theta) * self.factor, roll

    def inverse_transform_polar(
        self, theta: Array, roll: Array, **kwargs: Any
    ) -> tuple[Array, Array]:
        # fov = 2 arctan sensor_width / (2 * focal_length)
        return ivy.atan(theta / self.factor), roll
