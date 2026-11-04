import numpy as np
from math import sqrt, ulp

from typing import Callable


type Vector = np.ndarray[tuple[int], np.dtype[np.floating]]


def two_norm(vec: Vector) -> float:
    """Compute the 2-norm of `vec`, formally equivalent to the Frobenius norm."""
    return sqrt(vec.dot(vec))

def rms_norm(vec: Vector) -> float:
    return sqrt(vec.dot(vec) / len(vec))


SQRT_MACHINE_EPS = sqrt(ulp(1.0))

def linearize(function: Callable[[Vector], Vector], origin: Vector, f_origin: Vector | None = None):
    """Approximate the directional derivative operator of `function` at `origin`."""
    if f_origin is None:
        f_origin = function(origin)

    def linearization(direction: Vector) -> Vector:
        """Approximate the directional derivative at `origin` in direction of `direction`."""
        direction_norm = two_norm(direction)
        if direction_norm == 0.0:
            return np.zeros_like(f_origin)
        eps = SQRT_MACHINE_EPS / direction_norm
        return (function(origin + eps * direction) - f_origin) / eps

    return linearization
