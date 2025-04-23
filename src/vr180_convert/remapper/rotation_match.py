from __future__ import annotations

from typing import Any

import ivy
import numpy as np
from ivy import Array
from quaternion import as_quat_array, quaternion, rotate_vectors

from vr180_convert.remapper.base import RemapperBase

from .euclidean import Euclidean3DRotator
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


class RotationMatchRemapper(RemapperBase):
    requires_image: bool = True
    rotation_match_kwargs: dict[str, Any] | None = None
    child: Euclidean3DRotator | None = None
    match: MatchResult | None = None

    def remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        images = kwargs.pop("images")
        if images is None:
            raise ValueError("images must be provided.")
        shape = images.shape
        images = ivy.reshape((-1, *shape[-3:]), images)
        qs = []
        for images_lr in images:
            match = feature_match_points(images_lr[0, :, :], images_lr[1, :, :])
            self.match = match
            q = rotation_match_robust(
                points_to_be_rotated=match.points1,
                points=match.points2,
                **(self.rotation_match_kwargs or {}),
            )
            qs.append(q)
        qs = np.asarray(qs).reshape(shape[:-3] + (4,))
        self.child = Euclidean3DRotator(
            rotation=qs,
        )
        return self.child.remap(x, y, **kwargs)

    def inverse_remap(
        self, x: Array, y: Array, /, **kwargs: Any
    ) -> tuple[Array, Array]:
        if self.child is None:
            raise ValueError("Remapper has not been called yet.")
        return self.child.inverse_remap(x, y, **kwargs)
