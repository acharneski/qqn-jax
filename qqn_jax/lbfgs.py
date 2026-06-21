"""L-BFGS oracle wrapper.

This module delegates the limited-memory BFGS two-loop recursion to
JAXopt's proven ``inv_hessian_product`` implementation, exposing it as a
single-step *oracle* that produces the quasi-Newton direction ``-H∇f``.

We keep our own thin, fixed-size circular-buffer state so the whole thing
stays JIT/vmap compatible and so the oracle remains swappable.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jaxopt._src.lbfgs import inv_hessian_product


class LBFGSState(NamedTuple):
    """State for the L-BFGS oracle.

    Attributes:
        s_history: buffer of parameter differences, shape (history_size, n).
        y_history: buffer of gradient differences, shape (history_size, n).
        rho_history: buffer of 1 / (yᵀs), shape (history_size,).
        count: number of valid entries currently stored.
        gamma: scaling factor for the initial Hessian H0 = gamma * I.
        prev_params: previous parameters (for computing s).
        prev_grad: previous gradient (for computing y).
    """

    s_history: jnp.ndarray
    y_history: jnp.ndarray
    rho_history: jnp.ndarray
    count: jnp.ndarray
    gamma: jnp.ndarray
    prev_params: jnp.ndarray
    prev_grad: jnp.ndarray


def init_lbfgs_state(params, grad, history_size: int) -> LBFGSState:
    """Initialize an empty L-BFGS state for the given parameter shape."""
    n = params.shape[0]
    return LBFGSState(
        s_history=jnp.zeros((history_size, n), dtype=params.dtype),
        y_history=jnp.zeros((history_size, n), dtype=params.dtype),
        rho_history=jnp.zeros((history_size,), dtype=params.dtype),
        count=jnp.asarray(0, dtype=jnp.int32),
        gamma=jnp.asarray(1.0, dtype=params.dtype),
        prev_params=params,
        prev_grad=grad,
    )


def update_lbfgs_history(
    state: LBFGSState, params, grad, history_size: int
) -> LBFGSState:
    """Push a new (s, y) pair into the circular history buffer.

    JAXopt's ``inv_hessian_product`` expects the *oldest* entry first and
    treats unfilled (zero) slots as no-ops, so we append at the end via a
    roll-by-one toward index 0... actually we keep most-recent-first and
    flip when calling the oracle (see ``lbfgs_direction``).

    The update is only applied if the curvature condition ``yᵀs > eps`` is
    satisfied; otherwise the history is left unchanged (a standard L-BFGS
    safeguard for non-convex problems).
    """
    s = params - state.prev_params
    y = grad - state.prev_grad
    ys = jnp.vdot(y, s)
    yy = jnp.vdot(y, y)

    eps = jnp.asarray(1e-10, dtype=params.dtype)
    valid = ys > eps

    # Roll buffers to make room at index 0 (most recent first).
    new_s = jnp.where(
        valid,
        jnp.roll(state.s_history, shift=1, axis=0).at[0].set(s),
        state.s_history,
    )
    new_y = jnp.where(
        valid,
        jnp.roll(state.y_history, shift=1, axis=0).at[0].set(y),
        state.y_history,
    )
    rho = jnp.where(valid, 1.0 / ys, 0.0)
    new_rho = jnp.where(
        valid,
        jnp.roll(state.rho_history, shift=1, axis=0).at[0].set(rho),
        state.rho_history,
    )
    new_count = jnp.where(
        valid,
        jnp.minimum(state.count + 1, history_size),
        state.count,
    )
    new_gamma = jnp.where(valid, ys / yy, state.gamma)

    return LBFGSState(
        s_history=new_s,
        y_history=new_y,
        rho_history=new_rho,
        count=new_count,
        gamma=new_gamma,
        prev_params=params,
        prev_grad=grad,
    )


def lbfgs_direction(state: LBFGSState, grad) -> jnp.ndarray:
    """Compute the L-BFGS direction ``-H∇f`` via JAXopt's two-loop recursion.

    JAXopt's ``inv_hessian_product`` returns ``H∇f`` (the product of the
    implicit inverse Hessian with the gradient), so the descent direction
    is its negation.

    JAXopt orders history oldest-first; we store most-recent-first, so we
    flip the buffers before the call. Unfilled slots are zero (s=y=rho=0),
    which contribute nothing to the recursion, so masking is automatic.
    """
    pytree_product = inv_hessian_product(
        pytree=grad,
        s_history=jnp.flip(state.s_history, axis=0),
        y_history=jnp.flip(state.y_history, axis=0),
        rho_history=jnp.flip(state.rho_history, axis=0),
        gamma=state.gamma,
    )
    return -pytree_product