from __future__ import annotations

import math
from typing import Iterable


Vector = tuple[float, float, float, float, float]


ZERO: Vector = (0.0, 0.0, 0.0, 0.0, 0.0)


def add(left: Vector, right: Vector) -> Vector:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def sub(left: Vector, right: Vector) -> Vector:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def scale(vector: Vector, amount: float) -> Vector:
    return tuple(value * amount for value in vector)  # type: ignore[return-value]


def divide(vector: Vector, amount: float) -> Vector:
    if amount == 0:
        return ZERO
    return scale(vector, 1.0 / amount)


def norm(vector: Vector) -> float:
    return math.sqrt(sum(value * value for value in vector))


def sum_vectors(vectors: Iterable[Vector]) -> Vector:
    total = ZERO
    for vector in vectors:
        total = add(total, vector)
    return total
