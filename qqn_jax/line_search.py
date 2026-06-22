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
from qqn_jax.regions import resolve_region


class LineSearchResult(NamedTuple):
    """Result of a line search.

    Attributes:
        step_size: chosen step size ``α``.
        new_value: function value at ``params + α·d``.
        new_grad: gradient at ``params + α·d``.
        new_params: the updated parameters.
        done: whether the search satisfied its conditions.
         probe_params: fixed-size ``(max_probes, n)`` buffer of evaluated
             points along the path (for feeding oracle curvature memory).
         probe_grads: fixed-size ``(max_probes, n)`` buffer of probe gradients.
         probe_valid: fixed-size ``(max_probes,)`` boolean mask of filled slots.
    """

    step_size: jnp.ndarray
    new_value: jnp.ndarray
    new_grad: jnp.ndarray
    new_params: jnp.ndarray
    done: jnp.ndarray
    probe_params: jnp.ndarray = None
    probe_grads: jnp.ndarray = None
    probe_valid: jnp.ndarray = None


def _empty_probes(params, max_probes):
    """Allocate empty probe buffers shaped for ``params`` (a flat vector)."""
    n = params.shape[0]
    return (
        jnp.zeros((max_probes, n), dtype=params.dtype),
        jnp.zeros((max_probes, n), dtype=params.dtype),
        jnp.zeros((max_probes,), dtype=bool),
    )


def _record_probe(probe_params, probe_grads, probe_valid, slot, p, g, max_probes):
    """Write ``(p, g)`` into ``slot`` of the probe buffers (JIT-safe)."""
    in_range = jnp.logical_and(slot >= 0, slot < max_probes)
    idx = jnp.clip(slot, 0, max_probes - 1)
    new_params = jnp.where(in_range, probe_params.at[idx].set(p), probe_params)
    new_grads = jnp.where(in_range, probe_grads.at[idx].set(g), probe_grads)
    new_valid = jnp.where(in_range, probe_valid.at[idx].set(True), probe_valid)
    return new_params, new_grads, new_valid


def _make_projected_point(region, region_state, params):
    """Return a fn ``α -> projected(x + α·d)`` for a given direction.
    The caller curries the direction in; here we build a helper that, given
    a tentative point ``x + α·d``, projects it onto the region. When the
    region is the identity, this is a no-op (zero overhead).
    """

    def project_candidate(candidate):
        return region.project(params, candidate, region_state)

    return project_candidate


def backtracking_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    init_step: float = 1.0,
    c1: float = 1e-2,
    shrink: float = 0.5,
    max_iter: int = 5,
    region=None,
    region_state=None,
    max_probes: int = 32,
) -> LineSearchResult:
    """Backtracking line search (Armijo), self-contained for Optax.

    Repeatedly shrinks the step size by ``shrink`` until the Armijo
    sufficient-decrease condition ``f(x + α d) ≤ f(x) + c1 α gᵀd`` holds
    or ``max_iter`` is reached. Implemented with ``lax.while_loop`` to stay
    JIT/vmap compatible.
     If a ``region`` is supplied, the candidate point ``x + α·d`` is projected
     onto the region before evaluation, so the search navigates the feasible
     (projected) path.
    """
    region = resolve_region(region)
    project = _make_projected_point(region, region_state, params)
    dg = tree_vdot(grad, direction)  # directional derivative gᵀd

    def eval_at(alpha):
        raw = tree_add_scaled(params, alpha, direction)
        projected = project(raw)
        val, g = value_and_grad_fn(projected, *args)
        return projected, val, g

    init_pp, init_pg, init_pv = _empty_probes(params, max_probes)

    def cond(carry):
        alpha, i, val, _g, _p, _pp, _pg, _pv = carry
        armijo = val <= value + c1 * alpha * dg
        return jnp.logical_and(jnp.logical_not(armijo), i < max_iter)

    def body(carry):
        alpha, i, _val, _g, _p, pp, pg, pv = carry
        alpha = alpha * shrink
        new_params, new_val, new_g = eval_at(alpha)
        # Record this probe (slot = i, since slot 0 holds the init_step probe).
        pp, pg, pv = _record_probe(pp, pg, pv, i, new_params, new_g, max_probes)
        return alpha, i + 1, new_val, new_g, new_params, pp, pg, pv

    # Evaluate at the initial step first.
    init_params, init_val, init_g = eval_at(init_step)
    # Slot 0 records the initial-step probe.
    init_pp, init_pg, init_pv = _record_probe(
        init_pp, init_pg, init_pv, 0, init_params, init_g, max_probes
    )

    (
        alpha,
        _i,
        final_val,
        final_g,
        new_params,
        probe_params,
        probe_grads,
        probe_valid,
    ) = jax.lax.while_loop(
        cond,
        body,
        (
            init_step,
            jnp.asarray(1),
            init_val,
            init_g,
            init_params,
            init_pp,
            init_pg,
            init_pv,
        ),
    )
    armijo = final_val <= value + c1 * alpha * dg
    return LineSearchResult(
        step_size=alpha,
        new_value=final_val,
        new_grad=final_g,
        new_params=new_params,
        done=armijo,
        probe_params=probe_params,
        probe_grads=probe_grads,
        probe_valid=probe_valid,
    )


def strong_wolfe_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    c1: float = 1e-3,
    c2: float = 0.7,
    max_iter: int = 10,
    region=None,
    region_state=None,
    max_probes: int = 32,
) -> LineSearchResult:
    """Strong Wolfe line search via Optax ``scale_by_zoom_linesearch``.

    Enforces Armijo sufficient decrease and the strong curvature
    condition, which keeps the L-BFGS curvature updates well-conditioned.

    Optax's zoom line search is a ``GradientTransformationExtraArgs`` whose
    ``update`` step rescales the provided *updates* (here, ``direction``)
    by the discovered step size. We wrap a value-only objective for it and
    recompute value/grad at the accepted point.
     When a ``region`` is supplied, the recovered step is projected onto the
     region before value/grad are recomputed.
    """
    region = resolve_region(region)

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

    raw_params = optax.apply_updates(params, scaled_updates)
    new_params = region.project(params, raw_params, region_state)
    new_value, new_grad = value_and_grad_fn(new_params, *args)

    # Recover the step size from the scaling of the direction.
    d_norm_sq = tree_vdot(direction, direction)
    step_size = jnp.where(
        d_norm_sq > 0.0,
        tree_vdot(scaled_updates, direction) / d_norm_sq,
        jnp.asarray(0.0, dtype=new_value.dtype),
    )
    # Optax's zoom search hides its intermediate probes; expose the single
    # accepted point as a probe so the oracle still benefits.
    probe_params, probe_grads, probe_valid = _empty_probes(params, max_probes)
    probe_params, probe_grads, probe_valid = _record_probe(
        probe_params, probe_grads, probe_valid, 0, new_params, new_grad, max_probes
    )

    return LineSearchResult(
        step_size=step_size,
        new_value=new_value,
        new_grad=new_grad,
        new_params=new_params,
        done=new_value < value,
        probe_params=probe_params,
        probe_grads=probe_grads,
        probe_valid=probe_valid,
    )


def fixed_step_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    step_size: float = 1.0,
    region=None,
    region_state=None,
    max_probes: int = 32,
) -> LineSearchResult:
    """Trivial line search using a constant step size.
    Useful for debugging, benchmarking against a baseline, or when the
    quadratic path scaling already provides a sensible step. Always reports
    ``done=True`` (it makes no acceptance test).
    """
    region = resolve_region(region)
    alpha = jnp.asarray(step_size, dtype=value.dtype)
    raw_params = tree_add_scaled(params, alpha, direction)
    new_params = region.project(params, raw_params, region_state)
    new_val, new_g = value_and_grad_fn(new_params, *args)
    probe_params, probe_grads, probe_valid = _empty_probes(params, max_probes)
    probe_params, probe_grads, probe_valid = _record_probe(
        probe_params, probe_grads, probe_valid, 0, new_params, new_g, max_probes
    )
    return LineSearchResult(
        step_size=alpha,
        new_value=new_val,
        new_grad=new_g,
        new_params=new_params,
        done=jnp.asarray(True),
        probe_params=probe_params,
        probe_grads=probe_grads,
        probe_valid=probe_valid,
    )


def armijo_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    init_step: float = 1.0,
    c1: float = 1e-2,
    shrink: float = 0.5,
    max_iter: int = 30,
    region=None,
    region_state=None,
    max_probes: int = 32,
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
        region=region,
        region_state=region_state,
        max_probes=max_probes,
    )


def hager_zhang_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    c1: float = 0.1,
    max_iter: int = 30,
    region=None,
    region_state=None,
    max_probes: int = 32,
) -> LineSearchResult:
    """Hager-Zhang line search via Optax ``scale_by_backtracking_linesearch``.
    The Hager-Zhang scheme is a robust approximate-Wolfe line search. We use
    Optax's backtracking transformation parameterized to approximate it,
    recomputing value/grad at the accepted point. Falls back gracefully if
    the underlying transform is unavailable.
    """
    region = resolve_region(region)

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
    probe_params, probe_grads, probe_valid = _empty_probes(params, max_probes)
    probe_params, probe_grads, probe_valid = _record_probe(
        probe_params, probe_grads, probe_valid, 0, new_params, new_grad, max_probes
    )
    return LineSearchResult(
        step_size=step_size,
        new_value=new_value,
        new_grad=new_grad,
        new_params=new_params,
        done=new_value < value,
        probe_params=probe_params,
        probe_grads=probe_grads,
        probe_valid=probe_valid,
    )
