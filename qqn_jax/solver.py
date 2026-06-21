"""QQN (Quasi-Quadratic-Newton) solver.

QQN constructs the quadratic interpolation path

    d(t) = t(1-t)(-∇f) + t²(-H∇f)

blending the steepest-descent direction (``-∇f``) with the L-BFGS direction
(``-H∇f``), and performs a line search over this path.

The solver follows the JAXopt-style ``init_state`` / ``update`` / ``run``
interface and keeps all state in JIT-compatible NamedTuples.
"""

from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional

import jax
import jax.numpy as jnp

from qqn_jax.line_search import (
    armijo_search,
    backtracking_search,
    fixed_step_search,
    hager_zhang_search,
    strong_wolfe_search,
)
from qqn_jax.spline_search import spline_wrap
from qqn_jax.oracles import OracleInfo, resolve_oracle
from qqn_jax.regions import RegionInfo, resolve_region
from qqn_jax.utils import (
    make_value_and_grad,
    quadratic_path,
    tree_l2_norm,
    tree_negative,
)


# Registry mapping line-search names to their implementations.
_LINE_SEARCHES = {
    "strong_wolfe": strong_wolfe_search,
    "backtracking": backtracking_search,
    "armijo": armijo_search,
    "hager_zhang": hager_zhang_search,
    "fixed": fixed_step_search,
}


class QQNState(NamedTuple):
    """Immutable state container for QQN.

    Attributes:
        iter: iteration counter.
        value: current objective value.
        grad: current gradient.
         oracle_state: state of the oracle (e.g. L-BFGS history).
        step_size: last accepted step size ``α``.
        error: gradient norm (convergence metric).
        done: whether convergence has been reached.
        aux: optional auxiliary output of the objective.
         region_state: optional state for the projective region.
    """

    iter: jnp.ndarray
    value: jnp.ndarray
    grad: jnp.ndarray
    oracle_state: Any
    step_size: jnp.ndarray
    error: jnp.ndarray
    done: jnp.ndarray
    aux: Any = None
    region_state: Any = ()


class QQN:
    """Quasi-Quadratic-Newton optimizer.

    Args:
        fun: objective function ``f(params, *args) -> scalar`` (or
            ``(scalar, aux)`` if ``has_aux=True``).
        maxiter: maximum number of iterations.
        tol: convergence tolerance on the gradient L2 norm.
        history_size: L-BFGS memory size ``m``.
         line_search: name of the line-search strategy. One of
             ``"strong_wolfe"`` (default), ``"backtracking"``, ``"armijo"``,
             ``"hager_zhang"`` or ``"fixed"``.
         line_search_options: optional dict of keyword arguments forwarded to
             the chosen line-search function (e.g. ``c1``, ``c2``, ``max_iter``,
             ``init_step``, ``shrink``, ``step_size``). These override the
             line-search defaults.
         spline: when ``True``, enable the cubic Hermite spline refinement. This
             is orthogonal to ``line_search``: every probe along the (consistent)
             path is reused as a control point and the spline's stationary points
             guide the search. It composes with any chosen line search.
        has_aux: whether ``fun`` returns auxiliary data.
        t_grid: candidate interpolation parameters ``t`` to evaluate. The
            best (lowest line-searched value) is chosen each iteration.
    """

    def __init__(
        self,
        fun: Callable,
        maxiter: int = 100,
        tol: float = 1e-5,
        history_size: int = 10,
        line_search: str = "armijo",
        line_search_options: Optional[Dict[str, Any]] = None,
        spline: bool = False,
        has_aux: bool = False,
        t_grid: Optional[jnp.ndarray] = None,
        region=None,
        oracle="lbfgs",
    ):
        self.fun = fun
        self.maxiter = maxiter
        self.tol = tol
        self.history_size = history_size
        self.line_search = line_search
        self.line_search_options = dict(line_search_options or {})
        self.spline = spline
        self.has_aux = has_aux
        self._value_and_grad = make_value_and_grad(fun, has_aux=has_aux)
        self.region = resolve_region(region)
        self.oracle = resolve_oracle(oracle, history_size=history_size)

        if t_grid is None:
            # A small set of blends from gradient (small t) to L-BFGS (t=1).
            t_grid = jnp.array([0.25, 0.5, 0.75, 1.0])
        self.t_grid = jnp.asarray(t_grid)

        if line_search not in _LINE_SEARCHES:
            raise ValueError(
                f"Unknown line_search: {line_search!r}. "
                f"Available: {sorted(_LINE_SEARCHES)}."
            )
        # The spline refinement is orthogonal to the chosen line search: rather
        # than replacing it, it *wraps* it. The spline is an expanded definition
        # of the curve — it reuses every probe (with its gradient) as a control
        # point of a cubic Hermite spline along the consistent path, then tries
        # to improve on the inner search's accepted point. It composes with any
        # line search.
        base_ls = _LINE_SEARCHES[line_search]
        opts = self.line_search_options
        if opts:
            base_ls = partial(base_ls, **opts)
        self._ls = spline_wrap(base_ls) if self.spline else base_ls

    # --- Internal helpers -------------------------------------------------

    def _eval(self, params, *args):
        """Evaluate value and grad, splitting off aux if present."""
        if self.has_aux:
            (value, aux), grad = self._value_and_grad(params, *args)
        else:
            value, grad = self._value_and_grad(params, *args)
            aux = None
        return value, grad, aux

    def _plain_value_and_grad(self, params, *args):
        """Value-and-grad returning only ``(value, grad)`` for line search."""
        if self.has_aux:
            (value, _aux), grad = self._value_and_grad(params, *args)
        else:
            value, grad = self._value_and_grad(params, *args)
        return value, grad

    # --- JAXopt-style interface ------------------------------------------

    def init_state(self, params, *args) -> QQNState:
        """Initialize solver state at ``params``."""
        value, grad, aux = self._eval(params, *args)
        oracle_state = self.oracle.init(params)
        error = tree_l2_norm(grad)
        region_state = self.region.init(params)
        return QQNState(
            iter=jnp.asarray(0, jnp.int32),
            value=value,
            grad=grad,
            oracle_state=oracle_state,
            step_size=jnp.asarray(1.0),
            error=error,
            done=error <= self.tol,
            aux=aux,
            region_state=region_state,
        )

    def update(self, params, state: QQNState, *args):
        """Perform a single QQN iteration.

        Returns ``(new_params, new_state)``.
        """
        grad = state.grad

        # 1. Oracle: L-BFGS direction (-H∇f).
        qn_dir, _ = self.oracle.direction(params, grad, state.oracle_state)

        # 2. Gradient: steepest descent direction (-∇f).
        grad_dir = tree_negative(grad)

        # 3 & 4. Quadratic path + line search over each candidate t,
        #         selecting the blend that yields the lowest value.
        def search_one_t(t):
            d = quadratic_path(t, grad_dir, qn_dir)
            res = self._ls(
                self._plain_value_and_grad,
                params,
                d,
                state.value,
                grad,
                *args,
                region=self.region,
                region_state=state.region_state,
            )
            return res

        results = jax.vmap(search_one_t)(self.t_grid)

        # Pick the candidate with the smallest resulting value.
        best_idx = jnp.argmin(results.new_value)
        new_params = jax.tree_util.tree_map(lambda a: a[best_idx], results.new_params)
        new_value = results.new_value[best_idx]
        new_grad = jax.tree_util.tree_map(lambda a: a[best_idx], results.new_grad)
        step_size = results.step_size[best_idx]
        best_t = self.t_grid[best_idx]

        # Recompute aux at the accepted point if needed.
        if self.has_aux:
            (_, aux), _ = self._value_and_grad(new_params, *args)
        else:
            aux = None

        # Update the oracle state (e.g. L-BFGS curvature pair, momentum).
        oracle_info = OracleInfo(
            params=params,
            new_params=new_params,
            grad=grad,
            new_grad=new_grad,
            t=best_t,
            step_size=step_size,
        )
        new_oracle_state = self.oracle.update(state.oracle_state, oracle_info)
        # Update region state (e.g. adaptive trust-region radius).
        actual_reduction = state.value - new_value
        info = RegionInfo(
            params=params,
            new_params=new_params,
            pred_reduction=actual_reduction,
            actual_reduction=actual_reduction,
            t=best_t,
            step_size=step_size,
        )
        new_region_state = self.region.update(state.region_state, info)

        error = tree_l2_norm(new_grad)
        new_state = QQNState(
            iter=state.iter + 1,
            value=new_value,
            grad=new_grad,
            oracle_state=new_oracle_state,
            step_size=step_size,
            error=error,
            done=error <= self.tol,
            aux=aux,
            region_state=new_region_state,
        )
        return new_params, new_state

    def run(self, init_params, *args):
        """Run QQN to convergence (or ``maxiter``).

        Uses ``lax.while_loop`` so the whole optimization is JIT/vmap
        compatible.
        """
        state = self.init_state(init_params, *args)

        def cond(carry):
            params, state = carry
            not_converged = jnp.logical_not(state.done)
            not_maxiter = state.iter < self.maxiter
            return jnp.logical_and(not_converged, not_maxiter)

        def body(carry):
            params, state = carry
            new_params, new_state = self.update(params, state, *args)
            return new_params, new_state

        final_params, final_state = jax.lax.while_loop(cond, body, (init_params, state))
        return final_params, final_state
