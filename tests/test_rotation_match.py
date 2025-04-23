import ivy
import pytest
from numpy import allclose
from quaternion import from_rotation_vector, quaternion, rotate_vectors

from vr180_convert.remapper.rotation_match import rotation_match


@pytest.mark.parametrize(
    "rotation",
    [
        from_rotation_vector([0.1, 0.2, 0.3]),
    ],
)
def test_rotation_match(rotation: quaternion) -> None:
    random_points = ivy.random.random_normal(100, 3)
    random_points_rotated = rotate_vectors(rotation, random_points)
    rotation_est = rotation_match(random_points, random_points_rotated)
    assert allclose(rotation, rotation_est, atol=1e-3) or allclose(
        -rotation, rotation_est, atol=1e-3
    )
