"""Line search strategies for QQN.

The line search is a *first-class component* of QQN. It operates over the
quadratic path direction ``d`` (already constructed) and selects a step
size ``α`` satisfying sufficient decrease (Armijo) and, optionally, the
curvature (strong Wolfe) condition.

We delegate the strong-Wolfe search to Optax's proven, JIT/vmap-compatible
``optax.scale_by_zoom_linesearch`` and provide a self-contained
backtracking (Armijo) search. Both are adapted to the QQN interface so
the strategies remain swappable.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from qqn_jax.utils import tree_add_scaled, tree_vdot


class LineSearchResult(NamedTuple):
    """Result of a line search.

    Attributes:
        step_size: chosen step size ``α``.
        new_value: function value at ``params + α·d``.
        new_grad: gradient at ``params + α·d``.
        new_params: the updated parameters.
        done: whether the search satisfied its conditions.
    """

    step_size: jnp.ndarray
    new_value: jnp.ndarray
    new_grad: jnp.ndarray
    new_params: jnp.ndarray
    done: jnp.ndarray


def backtracking_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    init_step: float = 1.0,
    c1: float = 1e-4,
    shrink: float = 0.5,
    max_iter: int = 30,
) -> LineSearchResult:
    """Backtracking line search (Armijo), self-contained for Optax.

    Repeatedly shrinks the step size by ``shrink`` until the Armijo
    sufficient-decrease condition ``f(x + α d) ≤ f(x) + c1 α gᵀd`` holds
    or ``max_iter`` is reached. Implemented with ``lax.while_loop`` to stay
    JIT/vmap compatible.
    """
    dg = tree_vdot(grad, direction)  # directional derivative gᵀd

    def cond(carry):
        alpha, i, val, _g = carry
        armijo = val <= value + c1 * alpha * dg
        return jnp.logical_and(jnp.logical_not(armijo), i < max_iter)

    def body(carry):
        alpha, i, _val, _g = carry
        alpha = alpha * shrink
        new_params = tree_add_scaled(params, alpha, direction)
        new_val, new_g = value_and_grad_fn(new_params, *args)
        return alpha, i + 1, new_val, new_g

    # Evaluate at the initial step first.
    init_params = tree_add_scaled(params, init_step, direction)
    init_val, init_g = value_and_grad_fn(init_params, *args)

    alpha, _i, final_val, final_g = jax.lax.while_loop(
        cond, body, (init_step, jnp.asarray(0), init_val, init_g)
    )
    new_params = tree_add_scaled(params, alpha, direction)
    armijo = final_val <= value + c1 * alpha * dg
    return LineSearchResult(
        step_size=alpha,
        new_value=final_val,
        new_grad=final_g,
        new_params=new_params,
        done=armijo,
    )


def strong_wolfe_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    init_step: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 30,
) -> LineSearchResult:
    """Strong Wolfe line search via Optax ``scale_by_zoom_linesearch``.

    Enforces Armijo sufficient decrease and the strong curvature
    condition, which keeps the L-BFGS curvature updates well-conditioned.

    Optax's zoom line search is a ``GradientTransformationExtraArgs`` whose
    ``update`` step rescales the provided *updates* (here, ``direction``)
    by the discovered step size. We wrap a value-only objective for it and
    recompute value/grad at the accepted point.
    """

    def fun_only(p, *fa, **fkw):
        v, _ = value_and_grad_fn(p, *args)
        return v

    ls = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_iter,
        curv_rtol=c2,  # strong Wolfe curvature constant
        slope_rtol=c1,  # sufficient decrease (Armijo) constant
        tol=c1,  # sufficient decrease tolerance
        initial_guess_strategy="one",
    )
    ls_state = ls.init(params)

    # The zoom line search expects ``updates`` to be the search direction
    # and uses value_fn / grad to find the step. It returns rescaled
    # updates equal to ``alpha * direction``.
    scaled_updates, _new_state = ls.update(
        updates=direction,
        state=ls_state,
        params=params,
        value=value,
        grad=grad,
        value_fn=fun_only,
    )

    new_params = optax.apply_updates(params, scaled_updates)
    new_value, new_grad = value_and_grad_fn(new_params, *args)

    # Recover the step size from the scaling of the direction.
    d_norm_sq = tree_vdot(direction, direction)
    step_size = jnp.where(
        d_norm_sq > 0.0,
        tree_vdot(scaled_updates, direction) / d_norm_sq,
        jnp.asarray(0.0, dtype=new_value.dtype),
    )

    return LineSearchResult(
        step_size=step_size,
        new_value=new_value,
        new_grad=new_grad,
        new_params=new_params,
        done=new_value < value,
    )
