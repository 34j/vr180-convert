from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import attrs

if TYPE_CHECKING:
    from array_api.latest import Array


class TransformerBase(ABC):
    @abstractmethod
    def transform(self, image: Array, /, **kwargs: Any) -> Array:
        """
        Transform the input data.

        Parameters
        ----------
        image : Array
            Image of shape (..., lr, height, width, channels).

        Returns
        -------
        Array
            Transformed image of shape (..., lr, height, width, channels).

        """

    @abstractmethod
    def inverse_transform(self, image: Array, /, **kwargs: Any) -> Array:
        """
        Inverse transform the input data.

        Parameters
        ----------
        image : Array
            Image of shape (..., lr, height, width, channels).

        Returns
        -------
        Array
            Transformed image of shape (..., lr, height, width, channels).

        """

    def __mul__(self, other: TransformerBase) -> TransformerBase:
        if isinstance(self, MultiTransformer) and isinstance(other, MultiTransformer):
            return MultiTransformer(transformers=[*self.transformers, *other.transformers])
        if isinstance(self, MultiTransformer):
            return MultiTransformer(transformers=[*self.transformers, other])
        if isinstance(other, MultiTransformer):
            return MultiTransformer(transformers=[self, *other.transformers])
        return MultiTransformer(transformers=[self, other])


@attrs.define()
class MultiTransformer(TransformerBase):
    transformers: Sequence[TransformerBase]

    def transform(self, image: Array, /, **kwargs: Any) -> Array:
        for transformer in self.transformers:
            image = transformer.transform(image, **kwargs)
        return image

    def inverse_transform(self, image: Array, /, **kwargs: Any) -> Array:
        for transformer in reversed(self.transformers):
            image = transformer.inverse_transform(image, **kwargs)
        return image
