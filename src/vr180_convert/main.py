from pathlib import Path
from typing import Callable, Literal
import cv2 as cv
import numpy as np
from numpy.typing import NDArray
import attrs

from abc import ABCMeta, abstractmethod
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin
from quaternion import rotate_vectors
import cv2 as cv

class TransformerBase(BaseEstimator, TransformerMixin, metaclass=ABCMeta, ):
    def fit(self, image: NDArray, **kwargs: Any) -> None:
        pass
    
    @abstractmethod
    def transform(self, x: NDArray, y: NDArray, **kwargs: Any) -> tuple[NDArray, NDArray]:
        pass
    
    # @abstractmethod
    # def inverse_transform(self, x: NDArray, y: NDArray, **kwargs: Any) -> tuple[NDArray, NDArray]:
    #     pass
    
    def __mul__(self, other: 'TransformerBase') -> 'MultiTransformer':
        return MultiTransformer(transformers=[self, other])

@attrs.define()
class MultiTransformer(TransformerBase):
    transformers: list[TransformerBase]
    
    def transform(self, x: NDArray, y: NDArray, **kwargs: Any) -> tuple[NDArray, NDArray]:
        for transformer in self.transformers:
            x, y = transformer.transform(x, y, **kwargs)
        return x, y
    
    def inverse_transform(self, x: NDArray, y: NDArray, **kwargs: Any) -> tuple[NDArray, NDArray]:
        for transformer in reversed(self.transformers):
            x, y = transformer.inverse_transform(x, y, **kwargs)
        return x, y
    
def get_radius(input: NDArray) -> float:
    height = input.shape[0]
    center_row = input[height//2, :, :]
    center_row_is_black = (np.mean(center_row, axis=-1) < 10)
    center_row_is_black_deriv = np.diff(center_row_is_black.astype(int))
    
    # first and last 1 in the derivative
    center_row_black_start = np.where(center_row_is_black_deriv == 1)[0][0]
    center_row_black_end = np.where(center_row_is_black_deriv == -1)[0][-1]
    radius = (center_row_black_end - center_row_black_start) / 2
    return radius
    
@attrs.define()
class NormalizeTransformer(TransformerBase):
    center: tuple[float, float] | None = None
    scale: tuple[float, float]| None = None
    
    def transform(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
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
    
@attrs.define()
class AutoDetectRadiusNormalizeTransformer(NormalizeTransformer):
    def fit(self, image: NDArray, **kwargs: Any) -> None:
        radius = get_radius(image)
        self.center = (image.shape[1] // 2, image.shape[0] // 2)
        self.scale = (radius, radius)
    
@attrs.define()
class DenormalizeTransformer(TransformerBase):
    scale: tuple[float, float]
    center: tuple[float, float]
    
    def transform(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
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
    def transform_polar(self, theta: NDArray, roll: NDArray) -> tuple[NDArray, NDArray]:
        pass
    
    def transform(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        theta = np.sqrt(x**2 + y**2)
        roll = np.arctan2(y, x)
        theta, roll = self.transform_polar(theta, roll)
        x = theta * np.cos(roll)
        y = theta * np.sin(roll)
        return x, y
    
@attrs.define()
class FisheyeFormatEncoder(PolarRollTransformer):
    mapping_type: Literal["rectilinear", "stereographic", "equidistant", "equisolid", "orthographic"]
    
    def transform_polar(self, theta: NDArray, roll: NDArray) -> tuple[NDArray, NDArray]:
        """[-1, 1] -> [-pi/2, pi/2]"""
        if self.mapping_type == "rectilinear":
            return np.arctan(theta), roll
        elif self.mapping_type == "stereographic":
            return 2 * np.arctan(theta), roll
        elif self.mapping_type == "equidistant":
            return theta * (np.pi / 2), roll
        elif self.mapping_type == "equisolid":
            return 2 * np.arcsin(theta), roll
        elif self.mapping_type == "orthographic":
            return np.arcsin(theta), roll
        else:
            raise ValueError(f"Unknown mapping type: {self.mapping_type}")
        
@attrs.define()
class FisheyeFormatDecoder(PolarRollTransformer):
    mapping_type: Literal["rectilinear", "stereographic", "equidistant", "equisolid", "orthographic"]
    
    def transform_polar(self, theta: NDArray, roll: NDArray) -> tuple[NDArray, NDArray]:
        """[-pi/2, pi/2] -> [-1, 1]"""
        if self.mapping_type == "rectilinear":
            return np.tan(theta) , roll
        elif self.mapping_type == "stereographic":
            return 2 * np.tan(theta / 2), roll
        elif self.mapping_type == "equidistant":
            return theta / (np.pi / 2), roll
        elif self.mapping_type == "equisolid":
            return 2 * np.sin(theta / 2) , roll
        elif self.mapping_type == "orthographic":
            return np.sin(theta), roll
        else:
            raise ValueError(f"Unknown mapping type: {self.mapping_type}")
        
@attrs.define()
class ZoomTransformer(TransformerBase):
    scale: float
    
    def transform(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        x = x * self.scale
        y = y * self.scale
        return x, y
    
@attrs.define()
class Euclidean3DTransformer(TransformerBase):
    @abstractmethod
    def transform_v(self, v: NDArray) -> NDArray:
        pass
    
    def transform(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        phi = np.arctan2(y, x)
        theta = np.sqrt(x**2 + y**2)
        v = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        v = self.transform_v(v)
        theta = np.arccos(v[2])
        phi = np.arctan2(v[1], v[0])
        x = theta * np.cos(phi)
        y = theta * np.sin(phi)
        return x, y

def apply(from_path: Path | str, to_path: Path | str, transformer: TransformerBase, size: tuple[int, int] = (2048, 2048), interpolation: int = cv.INTER_LANCZOS4
          , boarder_mode: int = cv.BORDER_CONSTANT, boarder_value: int | tuple[int, int, int] = (0, 33, 0)
          ) -> None:
    img = cv.imread(Path(from_path).as_posix())
    height, width = img.shape[:2]
    radius = get_radius(img)
    xmap, ymap = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    xmap, ymap = (NormalizeTransformer() * transformer * DenormalizeTransformer(scale=(radius, radius), center=(width/2, height/2))).transform(xmap, ymap)
    xmap, ymap = xmap.astype(np.float32), ymap.astype(np.float32)
    warped = cv.remap(img, xmap, ymap, interpolation=interpolation, borderMode=boarder_mode, borderValue=boarder_value)
    cv.imwrite(Path(to_path).as_posix(), warped)