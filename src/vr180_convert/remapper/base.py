from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

import attrs
from ivy import Array


@attrs.define(kw_only=True)
class RemapperBase(metaclass=ABCMeta):
    requires_image: bool = False

    @abstractmethod
    def remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        """
        Transform the input coordinates.

        Parameters
        ----------
        x : Array
            x (left-right) coordinates.
        y : Array
            y (up-down) coordinates.
        **kwargs : Any
            Any additional keyword arguments.

        Returns
        -------
        tuple[Array, Array]
            x and y coordinates after transformation.

        """

    @abstractmethod
    def inverse_remap(
        self, x: Array, y: Array, /, **kwargs: Any
    ) -> tuple[Array, Array]:
        """
        Inverse transform the input coordinates.

        Parameters
        ----------
        x : Array
            x (left-right) coordinates.
        y : Array
            y (up-down) coordinates.
        **kwargs : Any
            Any additional keyword arguments.

        Returns
        -------
        tuple[Array, Array]
            x and y coordinates after transformation.

        """

    def __mul__(self, other: RemapperBase, /) -> MultiRemapper:
        """Multiply two transformers together."""
        if isinstance(self, MultiRemapper) and isinstance(other, MultiRemapper):
            return MultiRemapper(transformers=[*self.transformers, *other.transformers])
        elif isinstance(self, MultiRemapper):
            return MultiRemapper(transformers=[*self.transformers, other])
        elif isinstance(other, MultiRemapper):
            return MultiRemapper(transformers=[self, *other.transformers])
        return MultiRemapper(transformers=[self, other])


@attrs.define(kw_only=True)
class MultiRemapper(RemapperBase):
    """A transformer that applies multiple transformers in sequence."""

    transformers: list[RemapperBase]

    def remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        for transformer in self.transformers:
            x, y = transformer.remap(x, y, **kwargs)
        return x, y

    def inverse_remap(
        self, x: Array, y: Array, /, **kwargs: Any
    ) -> tuple[Array, Array]:
        for transformer in reversed(self.transformers):
            x, y = transformer.inverse_remap(x, y, **kwargs)
        return x, y
