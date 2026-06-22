"""L-BFGS oracle wrapper.

This module delegates the limited-memory BFGS two-loop recursion to
Optax's ``optax.scale_by_lbfgs`` machinery, exposing it as a single-step
*oracle* that produces the quasi-Newton direction ``-H∇f``.

We keep our own thin, fixed-size circular-buffer state so the whole thing
stays JIT/vmap compatible and so the oracle remains swappable. The actual
two-loop recursion is performed by Optax's
``optax._src.linesearch`` / LBFGS internals via
``optax.tree_utils`` helpers; here we reimplement the recursion directly
on our own buffers to avoid depending on Optax private state layouts.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


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

    We keep most-recent-first ordering (index 0 = newest) and flip the
    buffers when running the two-loop recursion in ``lbfgs_direction``.

    The update is only applied if the curvature condition ``yᵀs > eps`` is
    satisfied; otherwise the history is left unchanged (a standard L-BFGS
    safeguard for non-convex problems).
    """
    s = params - state.prev_params
    y = grad - state.prev_grad
    ys = jnp.vdot(y, s)
    yy = jnp.vdot(y, y)

    # Relative curvature guard: an absolute 1e-10 floor is below float32
    # resolution once ‖y‖‖s‖ is even moderately scaled, so it spuriously
    # admits near-zero-curvature pairs. Anchor to the Cauchy-Schwarz scale.
    ss = jnp.vdot(s, s)
    eps = jnp.asarray(1e-10, dtype=params.dtype)
    valid = ys > eps * jnp.sqrt(yy * ss + eps)

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
    # Guard the reciprocal: jnp.where evaluates BOTH branches, so a raw
    # ``1.0 / ys`` produces inf/NaN when ys ≤ 0 even though it is masked out.
    # Under jax.grad that NaN backpropagates through the non-selected branch
    # and poisons the gradient. Compute on a safe denominator first.
    safe_ys = jnp.where(valid, ys, jnp.ones_like(ys))
    rho = jnp.where(valid, 1.0 / safe_ys, 0.0)
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
    # Same NaN-safety for the initial-Hessian scale γ = ⟨y,s⟩/⟨y,y⟩.
    safe_yy = jnp.where(yy > 0.0, yy, jnp.ones_like(yy))
    new_gamma = jnp.where(valid, ys / safe_yy, state.gamma)

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
    """Compute the L-BFGS direction ``-H∇f`` via the two-loop recursion.

    This is a direct, self-contained implementation of the L-BFGS
    two-loop recursion (Nocedal & Wright, Algorithm 7.4). It replaces the
    previous dependency on JAXopt's ``inv_hessian_product``.

    Unfilled history slots are zero (s=y=rho=0). They contribute nothing
    to the recursion because their ``alpha`` and correction terms vanish,
    so masking is automatic and the result is exactly ``-H∇f``.

    Buffers are stored most-recent-first; the first loop iterates
    newest -> oldest, the second loop oldest -> newest.
    """
    s_hist = state.s_history  # newest first
    y_hist = state.y_history
    rho_hist = state.rho_history

    # First loop: newest -> oldest (index 0 .. m-1).
    def first_loop(carry, inputs):
        q = carry
        s_i, y_i, rho_i = inputs
        alpha_i = rho_i * jnp.vdot(s_i, q)
        q = q - alpha_i * y_i
        return q, alpha_i

    q, alphas = jax.lax.scan(first_loop, grad, (s_hist, y_hist, rho_hist))

    # Apply initial Hessian approximation H0 = gamma * I.
    r = state.gamma * q

    # Second loop: oldest -> newest (reverse of the first loop order).
    def second_loop(carry, inputs):
        r = carry
        s_i, y_i, rho_i, alpha_i = inputs
        beta_i = rho_i * jnp.vdot(y_i, r)
        r = r + (alpha_i - beta_i) * s_i
        return r, None

    r, _ = jax.lax.scan(
        second_loop,
        r,
        (s_hist, y_hist, rho_hist, alphas),
        reverse=True,
    )

    return -r
