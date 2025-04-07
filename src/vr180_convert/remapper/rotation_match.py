from __future__ import annotations

from typing import Sequence

import cv2 as cv
import numpy as np
from ivy import Array
from quaternion import as_quat_array, quaternion, rotate_vectors


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

    right_mult_matrix = np.array(
        [[aw, -az, ay, -ax], [az, aw, -ax, -ay], [-ay, ax, aw, -az], [ax, ay, az, aw]]
    )
    left_mult_matrix = np.array(
        [[bw, bz, -by, -bx], [-bz, bw, bx, -by], [by, -bx, bw, -bz], [bx, by, bz, bw]]
    )
    S = right_mult_matrix - left_mult_matrix
    B = np.einsum("jik,jlk->il", S, S)
    eigenvalues, eigenvectors = np.linalg.eig(B)
    q = eigenvectors[:, np.argmin(eigenvalues)]
    np.sqrt(eigenvalues.min()) / len(points)
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


def match_points(image1: Array, image2: Array, *, scale: float = 1) -> tuple[
    Sequence[tuple[float, float]],
    Sequence[tuple[float, float]],
    Array,
    Array,
    Array,
    Array,
    Array,
]:
    """
    Match the points in two images.

    Parameters
    ----------
    image1 : Array
        Image 1.
    image2 : Array
        Image 2.

    Returns
    -------
    tuple[Sequence[tuple[float, float]], Sequence[tuple[float, float]]]
        The points in image1 and image2.

    """
    akaze = cv.AKAZE_create()
    if scale != 1:
        image1 = cv.resize(
            image1, (int(image1.shape[1] * scale), int(image1.shape[0] * scale))
        )
        image2 = cv.resize(
            image2, (int(image2.shape[1] * scale), int(image2.shape[0] * scale))
        )
    kp1, des1 = akaze.detectAndCompute(image1, None)
    kp2, des2 = akaze.detectAndCompute(image2, None)
    bf = cv.BFMatcher()
    matches = bf.match(des1, des2)
    points1, points2 = [], []
    for m in matches:
        points1.append(kp1[m.queryIdx].pt)
        points2.append(kp2[m.trainIdx].pt)
    points1_ = np.array(points1)
    points2_ = np.array(points2)
    if scale != 1:
        points1_ /= scale
        points2_ /= scale
    return (
        points1_,
        points2_,
        np.array(kp1),
        np.array(kp2),
        np.array(matches),
        image1,
        image2,
    )
