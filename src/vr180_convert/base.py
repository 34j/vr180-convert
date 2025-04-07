from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Sequence

import attrs
from ivy import Array


class TransformerBase(metaclass=ABCMeta):
    @abstractmethod
    def transform(self, x: Array, /, **kwargs: Any) -> Array:
        pass

    @abstractmethod
    def inverse_transform(self, x: Array, /, **kwargs: Any) -> Array:
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
