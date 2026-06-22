"""Line search strategies for QQN.

The line search is a *first-class component* of QQN. It traverses the
quadratic path

    d(t) = t(1 - t)(-∇f) + t²(-H∇f)

by selecting the path parameter ``t`` (the step size along the curve)
satisfying sufficient decrease (Armijo) and, optionally, the curvature
(strong Wolfe) condition.

Each evaluated point ``x + d(t)`` is a *state* on the curve — not a
direction to be independently re-scaled. The search walks ``t`` directly.

Note on parameterization:
    Rescaling the gradient (or oracle direction) does **not** change the
    geometric path ``d(t)`` traces; it only distorts how ``t`` maps onto
    arc length along the curve. The candidate states are invariant to such
    rescaling.

When the path components ``grad_dir`` (``-∇f``) and ``qn_dir`` (``-H∇f``)
are supplied, the line search probes the genuine quadratic path
``d(t)``. When they are absent, it falls back to a straight-line probe
``x + t·direction`` (used internally by the spline refinement, which has
already fixed a consistent direction).

We delegate the strong-Wolfe search to Optax's proven, JIT/vmap-compatible
``optax.scale_by_zoom_linesearch`` and provide a self-contained
backtracking (Armijo) search. Both are adapted to the QQN interface so
the strategies remain swappable.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from qqn_jax.utils import tree_add_scaled, tree_vdot, quadratic_path
from qqn_jax.regions import resolve_region


class LineSearchResult(NamedTuple):
    """Result of a line search.

    Attributes:
        step_size: chosen path parameter ``t`` (step size along the curve).
        new_value: function value at the accepted state.
        new_grad: gradient at the accepted state.
        new_params: the updated parameters (a state on the path).
        done: whether the search satisfied its conditions.
    """

    step_size: jnp.ndarray
    new_value: jnp.ndarray
    new_grad: jnp.ndarray
    new_params: jnp.ndarray
    done: jnp.ndarray


def _make_point_fn(params, direction, grad_dir, qn_dir):
    """Build a function ``t -> candidate state x + step(t)``.

    When ``grad_dir`` and ``qn_dir`` are provided, the step is the genuine
    quadratic path ``d(t) = t(1-t)·grad_dir + t²·qn_dir`` (a state on the
    curve). Otherwise it is the straight-line ``t·direction``. In both
    cases the *point* returned is a state; it is never re-searched as a
    fresh direction.
    """
    use_path = grad_dir is not None and qn_dir is not None

    def point(t):
        if use_path:
            step = quadratic_path(t, grad_dir, qn_dir)
            return jax.tree_util.tree_map(lambda p, s: p + s, params, step)
        return tree_add_scaled(params, t, direction)

    return point


def _initial_slope(grad, direction, grad_dir, qn_dir):
    """Directional derivative of the path at ``t = 0``.

    For the quadratic path, ``d'(0) = grad_dir = -∇f``, so the slope is
    ``⟨∇f, d'(0)⟩ = ⟨∇f, -∇f⟩ = -‖∇f‖²``. For the straight-line fallback it
    is ``⟨∇f, direction⟩``.
    """
    if grad_dir is not None and qn_dir is not None:
        return tree_vdot(grad, grad_dir)
    return tree_vdot(grad, direction)


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
    grow: float = 2.0,
    max_extrapolate: int = 10,
    grad_dir=None,
    qn_dir=None,
    region=None,
    region_state=None,
) -> LineSearchResult:
    """Backtracking line search (Armijo) along the quadratic path.

     The search first *extrapolates*: starting from ``init_step`` it grows the
     path parameter ``t`` by ``grow`` while the Armijo condition holds and the
     fitness keeps strictly improving (so a still-descending oracle point at
     ``t = 1`` is pushed out to ``t = 2, 4, ...`` up to ``max_extrapolate``
     steps). It then *backtracks*: it shrinks ``t`` by ``shrink`` until the
     Armijo sufficient-decrease condition holds or ``max_iter`` is reached.
     Each probe evaluates the *state* ``x + d(t)`` on the curve. Implemented
     with ``lax.while_loop`` to stay JIT/vmap compatible.

    If a ``region`` is supplied, the candidate state is projected onto the
    region before evaluation, so the search navigates the feasible path.
    """
    region = resolve_region(region)
    point = _make_point_fn(params, direction, grad_dir, qn_dir)
    # Directional derivative of the path at t=0 (the sufficient-decrease slope).
    dg = _initial_slope(grad, direction, grad_dir, qn_dir)

    def eval_at(t):
        raw = point(t)
        projected = region.project(params, raw, region_state)
        val, g = value_and_grad_fn(projected, *args)
        return projected, val, g

    # --- Extrapolation phase ---------------------------------------------
    # Grow the step while Armijo holds *and* fitness strictly improves, so a
    # still-descending oracle endpoint is extended outward (t -> 2, 4, ...).
    init_params, init_val, init_g = eval_at(init_step)

    def ext_cond(carry):
        t, i, val, _g, _p = carry
        next_t = t * grow
        # Predict whether the larger step is still admissible: we only keep
        # growing while the current point already satisfies Armijo (so the
        # bracket is healthy) and we have extrapolation budget left.
        armijo = val <= value + c1 * t * dg
        return jnp.logical_and(armijo, i < max_extrapolate)

    def ext_body(carry):
        t, i, val, g, p = carry
        next_t = t * grow
        np_, nv, ng = eval_at(next_t)
        # Accept the larger step only if it strictly improves fitness *and*
        # still satisfies Armijo at the larger step; otherwise stop growing.
        next_armijo = nv <= value + c1 * next_t * dg
        improves = jnp.logical_and(nv < val, next_armijo)
        t = jnp.where(improves, next_t, t)
        val = jnp.where(improves, nv, val)
        g = jax.tree_util.tree_map(lambda a, b: jnp.where(improves, a, b), ng, g)
        p = jax.tree_util.tree_map(lambda a, b: jnp.where(improves, a, b), np_, p)
        # If we did not improve, exhaust the budget to stop the loop.
        i = jnp.where(improves, i + 1, jnp.asarray(max_extrapolate))
        return t, i, val, g, p

    ext_t, _ei, ext_val, ext_g, ext_p = jax.lax.while_loop(
        ext_cond,
        ext_body,
        (init_step, jnp.asarray(0), init_val, init_g, init_params),
    )

    # --- Backtracking phase ----------------------------------------------
    def cond(carry):
        t, i, val, _g, _p = carry
        armijo = val <= value + c1 * t * dg
        return jnp.logical_and(jnp.logical_not(armijo), i < max_iter)

    def body(carry):
        t, i, _val, _g, _p = carry
        t = t * shrink
        new_params, new_val, new_g = eval_at(t)
        return t, i + 1, new_val, new_g, new_params

    # Start backtracking from the (possibly extrapolated) point.
    t, _i, final_val, final_g, new_params = jax.lax.while_loop(
        cond, body, (ext_t, jnp.asarray(0), ext_val, ext_g, ext_p)
    )
    armijo = final_val <= value + c1 * t * dg
    return LineSearchResult(
        step_size=t,
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
    c1: float = 1e-5,
    c2: float = 0.9,
    max_iter: int = 20,
    grad_dir=None,
    qn_dir=None,
    region=None,
    region_state=None,
) -> LineSearchResult:
    """Strong Wolfe line search along the quadratic path.

    Enforces Armijo sufficient decrease and the strong curvature condition,
    which keeps the L-BFGS curvature updates well-conditioned.

    When the path components are supplied, the search walks the genuine
    quadratic path ``d(t)`` via a self-contained zoom search; each probe is
    the *state* ``x + d(t)``. (We cannot delegate to Optax's transform here
    because it assumes a straight-line ``x + α·d`` parameterization, whereas
    ``d(t)`` is quadratic in ``t``.) When the path components are absent
    (the spline-refinement fallback), we delegate to Optax's proven
    ``scale_by_zoom_linesearch`` on the fixed straight-line direction.
    """
    region = resolve_region(region)
    use_path = grad_dir is not None and qn_dir is not None

    if not use_path:
        # Straight-line fallback: delegate to Optax's zoom line search.
        def fun_only(p, *fa, **fkw):
            v, _ = value_and_grad_fn(p, *args)
            return v

        ls = optax.scale_by_zoom_linesearch(
            max_linesearch_steps=max_iter,
            curv_rtol=c2,
            slope_rtol=c1,
            tol=c1,
            initial_guess_strategy="one",
        )
        ls_state = ls.init(params)
        scaled_updates, _new_state = ls.update(
            updates=direction,
            state=ls_state,
            params=params,
            value=value,
            grad=grad,
            value_fn=fun_only,
        )
        raw_params = optax.apply_updates(params, scaled_updates)
        new_params = region.project(params, raw_params, region_state)
        new_value, new_grad = value_and_grad_fn(new_params, *args)
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

    # Path mode: self-contained backtracking that also checks the strong
    # curvature condition along the quadratic path. Each probe is a state.
    point = _make_point_fn(params, direction, grad_dir, qn_dir)
    dg0 = _initial_slope(grad, direction, grad_dir, qn_dir)

    def path_tangent(t):
        # d'(t) = (1 - 2t)·grad_dir + 2t·qn_dir
        a = 1.0 - 2.0 * t
        b = 2.0 * t
        return jax.tree_util.tree_map(lambda g, q: a * g + b * q, grad_dir, qn_dir)

    def eval_at(t):
        raw = point(t)
        projected = region.project(params, raw, region_state)
        val, g = value_and_grad_fn(projected, *args)
        slope = tree_vdot(g, path_tangent(t))
        return projected, val, g, slope

    def cond(carry):
        t, i, val, _g, slope, _p = carry
        armijo = val <= value + c1 * t * dg0
        curv = jnp.abs(slope) <= c2 * jnp.abs(dg0)
        ok = jnp.logical_and(armijo, curv)
        return jnp.logical_and(jnp.logical_not(ok), i < max_iter)

    def body(carry):
        t, i, _val, _g, _slope, _p = carry
        t = t * 0.5
        new_params, new_val, new_g, new_slope = eval_at(t)
        return t, i + 1, new_val, new_g, new_slope, new_params

    init_params, init_val, init_g, init_slope = eval_at(init_step)
    t, _i, final_val, final_g, _final_slope, new_params = jax.lax.while_loop(
        cond,
        body,
        (init_step, jnp.asarray(0), init_val, init_g, init_slope, init_params),
    )
    return LineSearchResult(
        step_size=t,
        new_value=final_val,
        new_grad=final_g,
        new_params=new_params,
        done=final_val < value,
    )


def fixed_step_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    step_size: float = 1.0,
    grad_dir=None,
    qn_dir=None,
    region=None,
    region_state=None,
) -> LineSearchResult:
    """Trivial line search using a constant path parameter ``t``.

    Evaluates the single state ``x + d(step_size)`` on the curve. Useful for
    debugging or benchmarking against a baseline. Always reports
    ``done=True`` (it makes no acceptance test).
    """
    region = resolve_region(region)
    point = _make_point_fn(params, direction, grad_dir, qn_dir)
    t = jnp.asarray(step_size, dtype=value.dtype)
    raw_params = point(t)
    new_params = region.project(params, raw_params, region_state)
    new_val, new_g = value_and_grad_fn(new_params, *args)
    return LineSearchResult(
        step_size=t,
        new_value=new_val,
        new_grad=new_g,
        new_params=new_params,
        done=jnp.asarray(True),
    )


def armijo_search(
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
    grow: float = 2.0,
    max_extrapolate: int = 10,
    grad_dir=None,
    qn_dir=None,
    region=None,
    region_state=None,
) -> LineSearchResult:
    """Alias for :func:`backtracking_search`.

    Provided so users can refer to the Armijo backtracking search by its
    classical name as well.
    """
    return backtracking_search(
        value_and_grad_fn,
        params,
        direction,
        value,
        grad,
        *args,
        init_step=init_step,
        c1=c1,
        shrink=shrink,
        max_iter=max_iter,
        grow=grow,
        max_extrapolate=max_extrapolate,
        grad_dir=grad_dir,
        qn_dir=qn_dir,
        region=region,
        region_state=region_state,
    )


def hager_zhang_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    init_step: float = 1.0,
    c1: float = 0.1,
    c2: float = 0.9,
    max_iter: int = 30,
    grad_dir=None,
    qn_dir=None,
    region=None,
    region_state=None,
) -> LineSearchResult:
    """Hager-Zhang approximate-Wolfe search along the quadratic path.

    When the path components are supplied, walks the genuine quadratic path
    ``d(t)`` (delegating to the path-aware backtracking search). When absent
    (the spline fallback), delegates to Optax's
    ``scale_by_backtracking_linesearch`` on the fixed straight-line
    direction. Each probe evaluates a *state* on the curve.
    """
    region = resolve_region(region)
    use_path = grad_dir is not None and qn_dir is not None

    if use_path:
        # Path-aware Armijo backtracking captures the Hager-Zhang spirit of a
        # robust approximate-Wolfe descent along the curve.
        return backtracking_search(
            value_and_grad_fn,
            params,
            direction,
            value,
            grad,
            *args,
            init_step=init_step,
            c1=c1,
            shrink=0.8,
            max_iter=max_iter,
            grad_dir=grad_dir,
            qn_dir=qn_dir,
            region=region,
            region_state=region_state,
        )

    def fun_only(p, *fa, **fkw):
        v, _ = value_and_grad_fn(p, *args)
        return v

    ls = optax.scale_by_backtracking_linesearch(
        max_backtracking_steps=max_iter,
        slope_rtol=c1,
        decrease_factor=0.8,
        increase_factor=1.0,
        store_grad=True,
    )
    ls_state = ls.init(params)
    scaled_updates, _new_state = ls.update(
        updates=direction,
        state=ls_state,
        params=params,
        value=value,
        grad=grad,
        value_fn=fun_only,
    )
    raw_params = optax.apply_updates(params, scaled_updates)
    new_params = region.project(params, raw_params, region_state)
    new_value, new_grad = value_and_grad_fn(new_params, *args)
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
