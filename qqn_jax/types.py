"""Typed interfaces for QQN-JAX.

All array types are annotated using ``chex.Array`` / ``jaxtyping`` so that
shapes and dtypes are documented and (optionally) runtime-checkable.
"""

from typing import Any, Callable, Tuple

import chex
from jaxtyping import Array, Float, Scalar

# A flat parameter / gradient vector.
Params = Float[Array, " n"]
Grad = Float[Array, " n"]
Direction = Float[Array, " n"]

# Scalar function value.
Value = Float[Array, ""]

# Objective function: params -> scalar.
ObjectiveFn = Callable[..., Scalar]

# Value-and-grad function: params -> (value, grad).
ValueAndGradFn = Callable[..., Tuple[Value, Grad]]

__all__ = [
    "Params",
    "Grad",
    "Direction",
    "Value",
    "ObjectiveFn",
    "ValueAndGradFn",
    "Any",
    "chex",
]
