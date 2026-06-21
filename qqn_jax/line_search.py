"""Line search strategies for QQN.

The line search is a *first-class component* of QQN. It operates over the
quadratic path direction ``d`` (already constructed) and selects a step
size ``α`` satisfying sufficient decrease (Armijo) and, optionally, the
curvature (strong Wolfe) condition.

Rather than reimplement these well-studied algorithms, we delegate to
JAXopt's proven, JIT/vmap-compatible ``ZoomLineSearch`` (strong Wolfe) and
``BacktrackingLineSearch`` (Armijo), adapting them to the QQN interface.
This keeps the strategies swappable and the proven code authoritative.
"""

from typing import Callable, NamedTuple

import jax.numpy as jnp
from jaxopt import BacktrackingLineSearch, ZoomLineSearch

from qqn_jax.utils import tree_add_scaled


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


def _run_jaxopt_ls(ls, value_and_grad_fn, params, direction,
                   value, grad, init_step, *args):
    """Shared adapter: run a JAXopt line search and repackage the result.

    JAXopt line searches take ``value_and_grad=True`` callables and a
    ``descent_direction``; they return ``(stepsize, LineSearchState)`` with
    the value/grad already recomputed at the accepted point.
    """
    stepsize, ls_state = ls.run(
        init_stepsize=init_step,
        params=params,
        value=value,
        grad=grad,
        descent_direction=direction,
        *args,
    )
    new_params = tree_add_scaled(params, stepsize, direction)
    return LineSearchResult(
        step_size=stepsize,
        new_value=ls_state.value,
        new_grad=ls_state.grad,
        new_params=new_params,
        done=ls_state.done,
    )


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
    """Backtracking line search (Armijo) via JAXopt ``BacktrackingLineSearch``."""
    ls = BacktrackingLineSearch(
        fun=value_and_grad_fn,
        value_and_grad=True,
        maxiter=max_iter,
        decrease_factor=shrink,
        c1=c1,
        condition="armijo",
    )
    return _run_jaxopt_ls(
        ls, value_and_grad_fn, params, direction, value, grad,
        init_step, *args,
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
    """Strong Wolfe line search via JAXopt ``ZoomLineSearch`` (bracket+zoom).

    Enforces Armijo sufficient decrease and the strong curvature condition,
    which keeps the L-BFGS curvature updates well-conditioned.
    """
    ls = ZoomLineSearch(
        fun=value_and_grad_fn,
        value_and_grad=True,
        maxiter=max_iter,
        c1=c1,
        c2=c2,
    )
    return _run_jaxopt_ls(
        ls, value_and_grad_fn, params, direction, value, grad,
        init_step, *args,
    )