from __future__ import annotations

from collections.abc import Callable
from typing import Any

import attrs
import ivy
import numpy as np
from ivy import Array
from quaternion import as_float_array, as_quat_array, quaternion, rotate_vectors

from vr180_convert.remapper.base import RemapperBase, UnfitError

from .euclidean import Euclidean3DRotator, equidistant_to_3d
from .feature_match import MatchResult, feature_match_points


def rotation_match(
    points_to_be_rotated: Array,
    points: Array,
) -> quaternion:
    """
    Match the rotation of two sets of 3d points.

    Parameters
    ----------
    points_to_be_rotated : _type_
        Array of shape (..., 3)
    points : Array
        Array of shape (..., 3)

    Returns
    -------
    quaternion
        quaternion that minimizes the distance between the rotated
        `points_to_be_rotated` and `points`.

    References
    ----------
    https://lisyarus.github.io/blog/posts/3d-shape-matching-with-quaternions.html

    """
    # 3d point matching
    # https://lisyarus.github.io/blog/posts/3d-shape-matching-with-quaternions.html
    # E := ||Ra_k - b_k||^2, Ra := qaq^{-1}
    # E = ||qa_kq^{-1} - b_k||^2 = ||qa_k - b_kq||^2

    # extend to 4d
    a = np.concatenate(
        [points_to_be_rotated, np.zeros_like(points_to_be_rotated[..., :1])], axis=1
    )
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b = np.concatenate([points, np.zeros_like(points[..., :1])], axis=1)
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    right_mult_matrix = np.asarray(
        [[aw, -az, ay, -ax], [az, aw, -ax, -ay], [-ay, ax, aw, -az], [ax, ay, az, aw]]
    )
    left_mult_matrix = np.asarray(
        [[bw, bz, -by, -bx], [-bz, bw, bx, -by], [by, -bx, bw, -bz], [bx, by, bz, bw]]
    )
    S = right_mult_matrix - left_mult_matrix
    B = np.einsum("...jik,...jlk->...il", S, S)
    eigenvalues, eigenvectors = np.linalg.eig(B)
    q = eigenvectors[..., :, np.argmin(eigenvalues)]
    return as_quat_array([q[..., 3], q[..., 0], q[..., 1], q[..., 2]])


def rotation_match_robust(
    points_to_be_rotated: Array,
    points: Array,
    n_iter: int = 15,
    quantile: float = 0.9,
) -> quaternion:
    """
    Match the rotation of two sets of 3d points.

    Repeats calling rotation_match() and removing the outliers.

    Parameters
    ----------
    points_to_be_rotated : _type_
        Array of shape (..., 3)
    points : Array
        Array of shape (..., 3)

    Returns
    -------
    quaternion
        quaternion that minimizes the distance between the rotated
        `points_to_be_rotated` and `points`.

    References
    ----------
    https://lisyarus.github.io/blog/posts/3d-shape-matching-with-quaternions.html

    """
    bad_idx = np.full(len(points), False)
    for i in range(n_iter):
        q = rotation_match(points_to_be_rotated=points_to_be_rotated, points=points)
        if i == n_iter - 1:
            break
        error = np.linalg.norm(
            rotate_vectors(q, points_to_be_rotated) - points, axis=-1
        )
        threshold = np.quantile(error, quantile)
        bad_idx_current = error > threshold
        bad_idx[~bad_idx] = bad_idx_current
        points_to_be_rotated = points_to_be_rotated[~bad_idx_current]
        points = points[~bad_idx_current]
    return q, bad_idx


@attrs.define(kw_only=True)
class RotationMatchRemapper(RemapperBase):
    """The previous remapper should output equidistant points."""

    requires_image: bool = attrs.field(default=True, init=False)
    rotation_match_kwargs: dict[str, Any] | None = None
    childl: Euclidean3DRotator | None = None
    childr: Euclidean3DRotator | None = None
    match: MatchResult | None = None

    def fit(
        self, image: Array, inv: Callable[[Array, Array], tuple[Array, Array]]
    ) -> None:
        shape = image.shape
        # [B, 2, H, W, C]
        image = ivy.reshape(image, (-1, *shape[-4:]))
        qs = []
        for images_lr in image:
            match = feature_match_points(
                images_lr[0, ...], images_lr[1, ...], scale=0.25
            )
            self.match = match
            points_x = ivy.stack([match.points1[:, 0], match.points2[:, 0]], axis=0)
            points_y = ivy.stack([match.points1[:, 1], match.points2[:, 1]], axis=0)
            points_translated_x, points_translated_y = inv(points_x, points_y)
            points_v = equidistant_to_3d(points_translated_x, points_translated_y)
            q, _ = rotation_match_robust(
                points_to_be_rotated=points_v[0, ...],
                points=points_v[1, ...],
                **(self.rotation_match_kwargs or {}),
            )
            qs.append(as_float_array(q))
        qs = as_quat_array(np.stack(qs).reshape(shape[:-4] + (4,)))
        phi = np.arccos(qs.w)  # type: ignore
        half_qs = np.sin(phi / 2) / np.sin(phi) * qs + 0.5
        self.childl = Euclidean3DRotator(
            rotation=np.conj(half_qs),
        )
        self.childr = Euclidean3DRotator(
            rotation=half_qs,
        )

    def remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        if self.childl is None or self.childr is None:
            raise UnfitError(self)
        x = ivy.broadcast_to(x, (*x.shape[:-3], 2, *x.shape[-2:]))
        y = ivy.broadcast_to(y, (*y.shape[:-3], 2, *y.shape[-2:]))
        xl, yl = x[..., 0, :, :], y[..., 0, :, :]
        xr, yr = x[..., 1, :, :], y[..., 1, :, :]
        xr, yr = self.childl.remap(xr, yr, **kwargs)
        xl, yl = self.childr.remap(xl, yl, **kwargs)
        return ivy.stack([xl, xr], axis=-3), ivy.stack([yl, yr], axis=-3)

    def inverse_remap(
        self, x: Array, y: Array, /, **kwargs: Any
    ) -> tuple[Array, Array]:
        if self.childl is None or self.childr is None:
            raise ValueError("Remapper has not been called yet.")
        x = ivy.broadcast_to(x, (*x.shape[:-3], 2, *x.shape[-2:]))
        y = ivy.broadcast_to(y, (*y.shape[:-3], 2, *y.shape[-2:]))
        xl, yl = x[..., 0, :, :], y[..., 0, :, :]
        xr, yr = x[..., 1, :, :], y[..., 1, :, :]
        xr, yr = self.childl.inverse_remap(xr, yr, **kwargs)
        xl, yl = self.childr.inverse_remap(xl, yl, **kwargs)
        return ivy.stack([xl, xr], axis=-3), ivy.stack([yl, yr], axis=-3)
