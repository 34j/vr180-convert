from typing import Any, Literal

import array_api_compat
import attrs
from array_api.latest import Array

from .inverse import InverseRemapper
from .polar_roll import PolarRollRemapper


@attrs.define()
class FisheyeEncoder(PolarRollRemapper):
    """Encodes fisheye image."""

    mapping_type: Literal["rectilinear", "stereographic", "equidistant", "equisolid", "orthographic"]
    """The mapping type of the fisheye image."""

    def transform_polar(self, theta: Array, roll: Array, **kwargs: Any) -> tuple[Array, Array]:
        """[-1, 1] -> [-pi/2, pi/2]."""
        xp = array_api_compat.array_namespace(theta, roll)
        if self.mapping_type == "rectilinear":
            return xp.atan(theta), roll
        elif self.mapping_type == "stereographic":
            return 2 * xp.atan(theta), roll
        elif self.mapping_type == "equidistant":
            return theta * (xp.pi / 2), roll
        elif self.mapping_type == "equisolid":
            return 2 * xp.asin(theta / xp.sqrt(2)), roll
        elif self.mapping_type == "orthographic":
            return xp.asin(theta), roll
        else:
            raise ValueError(
                f"Unknown mapping type: {self.mapping_type}, "
                "should be one of 'rectilinear', 'stereographic', 'equidistant', 'equisolid', 'orthographic'."
            )

    def inverse_transform_polar(self, theta: Array, roll: Array, **kwargs: Any) -> tuple[Array, Array]:
        """[-pi/2, pi/2] -> [-1, 1]."""
        xp = array_api_compat.array_namespace(theta, roll)
        if self.mapping_type == "rectilinear":
            return xp.tan(theta), roll
        elif self.mapping_type == "stereographic":
            return 2 * xp.tan(theta / 2), roll
        elif self.mapping_type == "equidistant":
            return theta / (xp.pi / 2), roll
        elif self.mapping_type == "equisolid":
            return xp.sqrt(2) * xp.sin(theta / 2), roll
        elif self.mapping_type == "orthographic":
            return xp.sin(theta), roll
        else:
            raise ValueError(
                f"Unknown mapping type: {self.mapping_type}, "
                "should be one of 'rectilinear', 'stereographic', 'equidistant', 'equisolid', 'orthographic'."
            )


def FisheyeDecoder(
    mapping_type: Literal["rectilinear", "stereographic", "equidistant", "equisolid", "orthographic"],
) -> InverseRemapper[FisheyeEncoder]:
    """
    Decodes fisheye image.

    Parameters
    ----------
    mapping_type : Literal['rectilinear', 'stereographic', 'equidistant', 'equisolid', 'orthographic']
        The mapping type of the fisheye image.

    Returns
    -------
    InverseRemapper
        The fisheye decoder.

    """
    return InverseRemapper(FisheyeEncoder(mapping_type))
