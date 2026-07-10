from collections.abc import Callable
from typing import Any

import pytest
from array_api._2024_12 import Array, ArrayNamespaceFull
from ie_circle import CircleShape, KressShape, Shape


@pytest.fixture(scope="session", params=["numpy"])
def xp(request: pytest.FixtureRequest) -> ArrayNamespaceFull:
    backend = request.param
    if backend == "numpy":
        from array_api_compat import numpy as xp

        rng = xp.random.default_rng()

        def random_uniform(low=0, high=1, shape=None, device=None, dtype=None):
            return rng.random(shape, dtype=dtype) * (high - low) + low

        def integers(low, high=None, shape=None, device=None, dtype=None):
            return rng.integers(low, high, size=shape, dtype=dtype)

        xp.random.random_uniform = random_uniform
        xp.random.integers = integers
    elif backend == "torch":
        from array_api_compat import torch as xp

        def random_uniform(low=0, high=1, shape=None, device=None, dtype=None):
            return xp.rand(shape, device=device, dtype=dtype) * (high - low) + low

        def integers(low, high=None, shape=None, device=None, dtype=None):
            return xp.randint(low, high, size=shape, device=device, dtype=dtype)

        xp.random.random_uniform = random_uniform
        xp.random.integers = integers
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return xp


@pytest.fixture(scope="session", params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest, xp: ArrayNamespaceFull) -> Any:
    device = request.param
    try:
        _ = xp.asarray(1, device=device)
    except Exception:
        pytest.skip(f"{device=} is not available")
    return device


@pytest.fixture(scope="session", params=["float64"])
def dtype(request: pytest.FixtureRequest, xp: ArrayNamespaceFull) -> str:
    return getattr(xp, request.param)


@pytest.fixture(params=[CircleShape(1.0), KressShape()])
def shape(request: pytest.FixtureRequest) -> Shape:
    return request.param


@pytest.fixture(params=[CircleShape(1.0), KressShape()])
def shape_h(request: pytest.FixtureRequest) -> Shape:
    return request.param


@pytest.fixture
def shape_central_difference(
    shape: Shape, shape_h: Shape
) -> Callable[[float], tuple[Shape, Shape]]:
    class _PerturbedShape:
        """Shape perturbed by adding/subtracting another shape."""

        def __init__(self, base: Shape, pert: Shape, eps: float) -> None:
            self._base = base
            self._pert = pert
            self._eps = eps

        def x(self, t: Array, /) -> Array:
            return self._base.x(t) + self._eps * self._pert.x(t)

        def dx(self, t: Array, /) -> Array:
            return self._base.dx(t) + self._eps * self._pert.dx(t)

        def ddx(self, t: Array, /) -> Array:
            return self._base.ddx(t) + self._eps * self._pert.ddx(t)

    def _make(eps: float) -> tuple[Shape, Shape]:
        return _PerturbedShape(shape, shape_h, eps), _PerturbedShape(shape, shape_h, -eps)

    return _make
