from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any

import attrs
from ivy import Array


class TransformerBase(metaclass=ABCMeta):
    @abstractmethod
    def transform(self, x: Array, /, **kwargs: Any) -> Array:
        """
        Transform the input data.

        Parameters
        ----------
        x : Array
            Image of shape (..., lr, width, height, channels).

        Returns
        -------
        Array
            Transformed image of shape (..., lr, width, height, channels).

        """
        pass

    @abstractmethod
    def inverse_transform(self, x: Array, /, **kwargs: Any) -> Array:
        """
        Inverse transform the input data.

        Parameters
        ----------
        x : Array
            Image of shape (..., lr, width, height, channels).

        Returns
        -------
        Array
            Transformed image of shape (..., lr, width, height, channels).

        """
        pass

    def __mul__(self, other: TransformerBase) -> TransformerBase:
        if isinstance(self, MultiTransformer) and isinstance(other, MultiTransformer):
            return MultiTransformer(
                transformers=[*self.transformers, *other.transformers]
            )
        elif isinstance(self, MultiTransformer):
            return MultiTransformer(transformers=[*self.transformers, other])
        elif isinstance(other, MultiTransformer):
            return MultiTransformer(transformers=[self, *other.transformers])
        return MultiTransformer(transformers=[self, other])


@attrs.define()
class MultiTransformer(TransformerBase):
    transformers: Sequence[TransformerBase]

    def transform(self, x: Array, /, **kwargs: Any) -> Array:
        for transformer in self.transformers:
            x = transformer.transform(x, **kwargs)
        return x

    def inverse_transform(self, x: Array, /, **kwargs: Any) -> Array:
        for transformer in reversed(self.transformers):
            x = transformer.inverse_transform(x, **kwargs)
        return x
