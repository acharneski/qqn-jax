"""Oracle abstraction for QQN.

The *oracle* supplies the ``t = 1`` endpoint of the quadratic path

    d(t) = t(1 - t)(-∇f) + t²(-H∇f)

i.e. the curvature-aware (or otherwise accelerated) direction ``-H∇f``.
The default oracle is L-BFGS, which reproduces the original behavior
byte-for-byte.

Every oracle is a pure, functional JAX object so it composes with
``jit``, ``vmap``, ``pmap`` and ``grad``. Oracles operate on flat
parameter / gradient vectors (consistent with the rest of ``qqn-jax``).
"""

from typing import Any, Callable, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp

from qqn_jax.lbfgs import (
    LBFGSState,
    init_lbfgs_state,
    lbfgs_direction,
    update_lbfgs_history,
)
from qqn_jax.utils import tree_negative


class Oracle(NamedTuple):
    """Pure, swappable oracle interface.

    Attributes:
        init: ``params -> oracle_state`` (use ``()`` when stateless).
        direction: ``(params, grad, state) -> (direction, new_state)``.
            ``direction`` is the ``t = 1`` endpoint ``-H∇f``.
        update: ``(state, info) -> state`` (no-op for stateless oracles).
    """

    init: Callable[[Any], Any]
    direction: Callable[[Any, Any, Any], Tuple[Any, Any]]
    update: Callable[[Any, Any], Any]


class OracleInfo(NamedTuple):
    """Information passed to ``Oracle.update`` after a step is accepted.

    Attributes:
        params: iterate ``x`` before the step.
        new_params: accepted iterate ``x_new``.
        grad: gradient ``∇f(x)`` before the step.
        new_grad: gradient ``∇f(x_new)`` after the step.
        t: chosen interpolation parameter.
        step_size: accepted step size ``α``.
    """

    params: Any = None
    new_params: Any = None
    grad: Any = None
    new_grad: Any = None
    t: Any = None
    step_size: Any = None


# --- L-BFGS Oracle (default) ------------------------------------------


def LBFGSOracle(history_size: int = 10) -> Oracle:
    """Limited-memory BFGS quasi-Newton oracle.

    Wraps the existing ``qqn_jax.lbfgs`` two-loop recursion so the default
    behavior is byte-for-byte equivalent to the original optimizer.
    """

    def init(params):
        # ``grad`` is unknown at init; use zeros for ``prev_grad`` so the
        # very first curvature pair is computed once a real gradient lands.
        grad = jax.tree_util.tree_map(jnp.zeros_like, params)
        return init_lbfgs_state(params, grad, history_size)

    def direction(params, grad, state):
        d = lbfgs_direction(state, grad)
        return d, state

    def update(state, info):
        return update_lbfgs_history(state, info.new_params, info.new_grad, history_size)

    return Oracle(init=init, direction=direction, update=update)


# --- Momentum Oracle --------------------------------------------------


class MomentumState(NamedTuple):
    velocity: jnp.ndarray


def MomentumOracle(beta: float = 0.9) -> Oracle:
    """First-order accelerated (heavy-ball) oracle.

    The ``t = 1`` endpoint is the negative of an exponentially-weighted
    history of past gradients::

        v_new = β · v + (1 − β) · ∇f
        direction = -v_new
    """

    def init(params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return MomentumState(velocity=zeros)

    def direction(params, grad, state):
        v_new = jax.tree_util.tree_map(
            lambda v, g: beta * v + (1.0 - beta) * g, state.velocity, grad
        )
        d = tree_negative(v_new)
        return d, MomentumState(velocity=v_new)

    def update(state, info):
        # Velocity already committed in ``direction``; nothing to do.
        return state

    return Oracle(init=init, direction=direction, update=update)


# --- Shampoo Oracle ---------------------------------------------------


class ShampooState(NamedTuple):
    L: jnp.ndarray
    R: jnp.ndarray
    step: jnp.ndarray


def _matrix_inverse_pth_root(mat, p, epsilon):
    """Compute ``mat^{-1/p}`` for a symmetric PSD matrix via eigh."""
    n = mat.shape[0]
    mat = mat + epsilon * jnp.eye(n, dtype=mat.dtype)
    w, v = jnp.linalg.eigh(mat)
    w = jnp.maximum(w, epsilon)
    inv_root = w ** (-1.0 / p)
    return (v * inv_root) @ v.T


def ShampooOracle(
    block_size: int = 128,
    update_freq: int = 20,
    epsilon: float = 1e-6,
) -> Oracle:
    """Structure-aware preconditioned oracle (Shampoo).

    Operates on the flat parameter vector by reshaping it into a single
    matrix block. For the flat-vector setting used throughout
    ``qqn-jax`` the gradient ``g`` (shape ``(n,)``) is treated as a column
    and preconditioned via accumulated second-moment statistics.

    The inverse roots are recomputed on a fixed static cadence
    (``update_freq``) so the per-step cost stays amortized and the whole
    computation remains ``jit``-friendly.
    """

    def init(params):
        n = params.shape[0]
        return ShampooState(
            L=jnp.zeros((n, n), dtype=params.dtype),
            R=jnp.zeros((1, 1), dtype=params.dtype),
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def direction(params, grad, state):
        g = grad.reshape(-1, 1)  # (n, 1)
        L_new = state.L + g @ g.T
        R_new = state.R + g.T @ g

        do_refresh = (state.step % update_freq) == 0

        def refresh(_):
            Lr = _matrix_inverse_pth_root(L_new, 4.0, epsilon)
            Rr = _matrix_inverse_pth_root(R_new, 4.0, epsilon)
            precond = (Lr @ g) @ Rr  # (n, 1)
            return precond.reshape(-1)

        def keep(_):
            # Fall back to scaled gradient when not refreshing roots.
            return grad

        precond = jax.lax.cond(do_refresh, refresh, keep, operand=None)
        d = -precond
        new_state = ShampooState(L=L_new, R=R_new, step=state.step + 1)
        return d, new_state

    def update(state, info):
        return state

    return Oracle(init=init, direction=direction, update=update)


# --- Combinator: Fallback ---------------------------------------------


def Fallback(oracles: Sequence[Oracle]) -> Oracle:
    """Use the first oracle's direction when valid, else fall back.

    Validity is detected as a finite, non-zero direction (e.g. an L-BFGS
    oracle with an empty history returns ``-H∇f = -∇f`` which is valid;
    a degenerate ``NaN``/``inf`` direction triggers the fallback). All
    selection uses ``jnp.where`` / ``lax.select`` — no Python conditionals.
    """
    oracles = tuple(oracles)

    def init(params):
        return tuple(o.init(params) for o in oracles)

    def direction(params, grad, state):
        new_states = []
        chosen = None
        chosen_valid = None
        for o, s in zip(oracles, state):
            d, ns = o.direction(params, grad, s)
            new_states.append(ns)
            valid = jnp.all(jnp.isfinite(d)) & (
                jnp.vdot(d, d) > jnp.asarray(0.0, dtype=d.dtype)
            )
            if chosen is None:
                chosen = d
                chosen_valid = valid
            else:
                take_prev = chosen_valid
                chosen = jnp.where(take_prev, chosen, d)
                chosen_valid = chosen_valid | valid
        return chosen, tuple(new_states)

    def update(state, info):
        return tuple(o.update(s, info) for o, s in zip(oracles, state))

    return Oracle(init=init, direction=direction, update=update)


# --- Resolution -------------------------------------------------------


def resolve_oracle(oracle, history_size: int = 10) -> Oracle:
    """Map a string shortcut or ``Oracle`` instance to a concrete oracle."""
    if oracle is None or oracle == "lbfgs":
        return LBFGSOracle(history_size=history_size)
    if isinstance(oracle, str):
        if oracle == "momentum":
            return MomentumOracle()
        if oracle == "shampoo":
            return ShampooOracle()
        raise ValueError(
            f"Unknown oracle: {oracle!r}. "
            "Available: 'lbfgs', 'momentum', 'shampoo' or an Oracle instance."
        )
    if isinstance(oracle, Oracle):
        return oracle
    raise TypeError(f"oracle must be a string, Oracle, or None; got {type(oracle)!r}.")


__all__ = [
    "Oracle",
    "OracleInfo",
    "LBFGSOracle",
    "MomentumOracle",
    "ShampooOracle",
    "Fallback",
    "resolve_oracle",
]
