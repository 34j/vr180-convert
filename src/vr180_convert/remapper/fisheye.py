from typing import Any, Literal

import attrs
import ivy
from ivy import Array

from .inverse import InverseRemapper
from .polar_roll import PolarRollRemapper


@attrs.define()
class FisheyeEncoder(PolarRollRemapper):
    """Encodes fisheye image."""

    mapping_type: Literal[
        "rectilinear", "stereographic", "equidistant", "equisolid", "orthographic"
    ]
    """The mapping type of the fisheye image."""

    def transform_polar(
        self, theta: Array, roll: Array, **kwargs: Any
    ) -> tuple[Array, Array]:
        """[-1, 1] -> [-pi/2, pi/2]."""
        if self.mapping_type == "rectilinear":
            return ivy.atan(theta), roll
        elif self.mapping_type == "stereographic":
            return 2 * ivy.atan(theta), roll
        elif self.mapping_type == "equidistant":
            return theta * (ivy.pi / 2), roll
        elif self.mapping_type == "equisolid":
            return 2 * ivy.asin(theta / ivy.sqrt(2)), roll
        elif self.mapping_type == "orthographic":
            return ivy.asin(theta), roll
        else:
            raise ValueError(
                f"Unknown mapping type: {self.mapping_type}, "
                "should be one of 'rectilinear', 'stereographic', 'equidistant', 'equisolid', 'orthographic'."
            )

    def inverse_transform_polar(
        self, theta: Array, roll: Array, **kwargs: Any
    ) -> tuple[Array, Array]:
        """[-pi/2, pi/2] -> [-1, 1]."""
        if self.mapping_type == "rectilinear":
            return ivy.tan(theta), roll
        elif self.mapping_type == "stereographic":
            return 2 * ivy.tan(theta / 2), roll
        elif self.mapping_type == "equidistant":
            return theta / (ivy.pi / 2), roll
        elif self.mapping_type == "equisolid":
            return ivy.sqrt(2) * ivy.sin(theta / 2), roll
        elif self.mapping_type == "orthographic":
            return ivy.sin(theta), roll
        else:
            raise ValueError(
                f"Unknown mapping type: {self.mapping_type}, "
                "should be one of 'rectilinear', 'stereographic', 'equidistant', 'equisolid', 'orthographic'."
            )


def FisheyeDecoder(
    mapping_type: Literal[
        "rectilinear", "stereographic", "equidistant", "equisolid", "orthographic"
    ]
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
