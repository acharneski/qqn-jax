"""Parameter-space regularizers for QQN training.

These pure-JAX helpers compute scalar penalties over a parameter pytree
(or a flat parameter vector) that can be *added to a loss function*. They
complement the *projective* regions in :mod:`qqn_jax.regions`:

  * A **region** changes the geometry of the feasible set (it projects
    the step). It does not alter the objective value.
  * A **regularizer** changes the *objective* itself (it adds a penalty),
    biasing the optimizer toward sparse / small / quantization-friendly
    weights.

The two are complementary and compose: e.g. pair an :class:`OrthantRegion`
(geometric sparsity) with :func:`l1_penalty` (objective sparsity), or a
:class:`QuantizationRegion` (cell-confinement) with
:func:`quantization_delta_penalty` (rounding-error attraction).

All functions operate on either a pytree of arrays or a single flat array,
and are ``jit`` / ``vmap`` / ``grad`` compatible.
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp

__all__ = [
    "l1_penalty",
    "l2_penalty",
    "quantization_delta_penalty",
    "elastic_net_penalty",
    "select_weights",
    "round_to_grid",
]


def _leaves(params: Any):
    """Return the numeric leaves of a pytree (or [array] for a bare array)."""
    return jax.tree_util.tree_leaves(params)


def round_to_grid(
    x: jnp.ndarray,
    bits: Optional[int] = None,
    step: Optional[float] = None,
    lo: float = -1.0,
    hi: float = 1.0,
) -> jnp.ndarray:
    """Round ``x`` onto the uniform grid over ``[lo, hi]``.
    Single source of truth for grid rounding shared by
    :func:`quantization_delta_penalty` (the penalty) and the example's
    post-rounding evaluation, so the two can never silently diverge.
    """
    if step is None and bits is None:
        raise ValueError("round_to_grid requires either `bits` or `step`.")
    dt = x.dtype
    lo_v = jnp.asarray(lo, dtype=dt)
    hi_v = jnp.asarray(hi, dtype=dt)
    if step is not None:
        delta = jnp.asarray(step, dtype=dt)
    else:
        assert bits is not None
        delta = jnp.asarray((hi - lo) / ((2 ** int(bits)) - 1), dtype=dt)
    x_clipped = jnp.clip(x, lo_v, hi_v)
    k = jnp.round((x_clipped - lo_v) / delta)
    k = jnp.clip(k, 0.0, jnp.floor((hi_v - lo_v) / delta))
    return lo_v + k * delta


def select_weights(params: Any, key: str = "w"):
    """Extract weight matrices from a list-of-dict MLP parameter pytree.

    If ``params`` is a list of dicts (the MLP layout used by the examples),
    return the list of ``layer[key]`` arrays. Otherwise treat ``params`` as a
    generic pytree and return all of its leaves (so regularizers still work on
    flat parameter vectors).
    """
    if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
        return [layer[key] for layer in params if key in layer]
    return _leaves(params)


def l1_penalty(
    params: Any, scale: float = 1e-4, weights_only: bool = False
) -> jnp.ndarray:
    """L1 norm penalty ``scale · Σ|θ|`` — promotes sparsity.

    Args:
        params: parameter pytree or flat array.
        scale: penalty coefficient.
        weights_only: if ``True`` and ``params`` is an MLP list-of-dicts,
            only the ``"w"`` matrices are penalized (biases are spared).
    """
    leaves = select_weights(params) if weights_only else _leaves(params)
    total = jnp.asarray(sum(jnp.sum(jnp.abs(p)) for p in leaves))
    return scale * total


def l2_penalty(
    params: Any, scale: float = 1e-4, weights_only: bool = False
) -> jnp.ndarray:
    """Squared-L2 (ridge) penalty ``scale · Σθ²`` — promotes small weights."""
    leaves = select_weights(params) if weights_only else _leaves(params)
    total = jnp.asarray(sum(jnp.sum(p**2) for p in leaves))
    return scale * total


def elastic_net_penalty(
    params: Any,
    l1: float = 1e-4,
    l2: float = 1e-4,
    weights_only: bool = False,
) -> jnp.ndarray:
    """Elastic-net penalty ``l1·Σ|θ| + l2·Σθ²``."""
    return l1_penalty(params, l1, weights_only) + l2_penalty(params, l2, weights_only)


def quantization_delta_penalty(
    params: Any,
    scale: float = 1e-4,
    bits: Optional[int] = None,
    step: Optional[float] = None,
    lo: float = -1.0,
    hi: float = 1.0,
    weights_only: bool = False,
) -> jnp.ndarray:
    """L1 norm of the rounding delta ``scale · Σ|θ − round_grid(θ)|``.

    This is the objective-space counterpart to :func:`QuantizationRegion
    <qqn_jax.regions.QuantizationRegion>`. Quantization over ``[lo, hi]`` with
    ``bits`` levels (or an explicit ``step``) defines a uniform grid spaced by
    ``Δ = (hi − lo) / (2**bits − 1)``. The *rounding delta* is the (signed)
    distance from each weight to its nearest grid point; this penalty is the L1
    norm of that delta — a sawtooth whose minima (zero error) sit exactly on the
    grid points and whose maxima (``Δ/2``) sit on the midpoints between them.

    Minimizing it draws weights toward representable grid values, making the
    network *precision-optimized* (quantization-aware) without hard-snapping.

    Args:
        params: parameter pytree or flat array.
        scale: penalty coefficient.
        bits: number of quantization bits (provide either ``bits`` or ``step``).
        step: explicit grid spacing ``Δ`` (overrides ``bits`` when given).
        lo, hi: quantization range; values are clipped before the delta is taken.
        weights_only: if ``True``, only penalize ``"w"`` matrices of an MLP.
    """
    if step is None and bits is None:
        raise ValueError("quantization_delta_penalty requires either `bits` or `step`.")

    leaves = select_weights(params) if weights_only else _leaves(params)

    def _delta(dtype):
        if step is not None:
            return jnp.asarray(step, dtype=dtype)
        assert bits is not None
        levels = (2 ** int(bits)) - 1
        return jnp.asarray((hi - lo) / levels, dtype=dtype)

    def leaf_penalty(p):
        dt = p.dtype
        delta = _delta(dt)
        lo_v = jnp.asarray(lo, dtype=dt)
        hi_v = jnp.asarray(hi, dtype=dt)
        x = jnp.clip(p, lo_v, hi_v)
        # Nearest grid point g_k = lo + round((x-lo)/delta)*delta.
        k = jnp.round((x - lo_v) / delta)
        k_max = jnp.floor((hi_v - lo_v) / delta)
        k = jnp.clip(k, 0.0, k_max)
        grid = lo_v + k * delta
        return jnp.sum(jnp.abs(x - grid))

    total = jnp.asarray(sum(leaf_penalty(p) for p in leaves))
    return scale * total
