"""QQN (Quasi-Quadratic-Newton) solver.

QQN constructs the quadratic interpolation path

    d(t) = t(1-t)(-∇f) + t²(-H∇f)

blending the steepest-descent direction (``-∇f``) with the L-BFGS direction
(``-H∇f``), and performs a line search over this path.

The solver follows the JAXopt-style ``init_state`` / ``update`` / ``run``
interface and keeps all state in JIT-compatible NamedTuples.
"""

from functools import partial
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from qqn_jax.lbfgs import (
    LBFGSState,
    init_lbfgs_state,
    lbfgs_direction,
    update_lbfgs_history,
)
from qqn_jax.line_search import (
    backtracking_search,
    strong_wolfe_search,
)
from qqn_jax.utils import (
    make_value_and_grad,
    quadratic_path,
    tree_l2_norm,
    tree_negative,
)


class QQNState(NamedTuple):
    """Immutable state container for QQN.

    Attributes:
        iter: iteration counter.
        value: current objective value.
        grad: current gradient.
        lbfgs_state: state of the L-BFGS oracle.
        step_size: last accepted step size ``α``.
        error: gradient norm (convergence metric).
        done: whether convergence has been reached.
        aux: optional auxiliary output of the objective.
    """

    iter: jnp.ndarray
    value: jnp.ndarray
    grad: jnp.ndarray
    lbfgs_state: LBFGSState
    step_size: jnp.ndarray
    error: jnp.ndarray
    done: jnp.ndarray
    aux: Any = None


class QQN:
    """Quasi-Quadratic-Newton optimizer.

    Args:
        fun: objective function ``f(params, *args) -> scalar`` (or
            ``(scalar, aux)`` if ``has_aux=True``).
        maxiter: maximum number of iterations.
        tol: convergence tolerance on the gradient L2 norm.
        history_size: L-BFGS memory size ``m``.
        line_search: ``"strong_wolfe"`` (default) or ``"backtracking"``.
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
        line_search: str = "strong_wolfe",
        has_aux: bool = False,
        t_grid: Optional[jnp.ndarray] = None,
    ):
        self.fun = fun
        self.maxiter = maxiter
        self.tol = tol
        self.history_size = history_size
        self.line_search = line_search
        self.has_aux = has_aux
        self._value_and_grad = make_value_and_grad(fun, has_aux=has_aux)

        if t_grid is None:
            # A small set of blends from gradient (small t) to L-BFGS (t=1).
            t_grid = jnp.array([0.25, 0.5, 0.75, 1.0])
        self.t_grid = jnp.asarray(t_grid)

        if line_search == "strong_wolfe":
            self._ls = strong_wolfe_search
        elif line_search == "backtracking":
            self._ls = backtracking_search
        else:
            raise ValueError(
                f"Unknown line_search: {line_search!r}. "
                "Use 'strong_wolfe' or 'backtracking'."
            )

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
        lbfgs_state = init_lbfgs_state(params, grad, self.history_size)
        error = tree_l2_norm(grad)
        return QQNState(
            iter=jnp.asarray(0, jnp.int32),
            value=value,
            grad=grad,
            lbfgs_state=lbfgs_state,
            step_size=jnp.asarray(1.0),
            error=error,
            done=error <= self.tol,
            aux=aux,
        )

    def update(self, params, state: QQNState, *args):
        """Perform a single QQN iteration.

        Returns ``(new_params, new_state)``.
        """
        grad = state.grad

        # 1. Oracle: L-BFGS direction (-H∇f).
        qn_dir = lbfgs_direction(state.lbfgs_state, grad)

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
            )
            return res

        results = jax.vmap(search_one_t)(self.t_grid)

        # Pick the candidate with the smallest resulting value.
        best_idx = jnp.argmin(results.new_value)
        new_params = jax.tree_util.tree_map(lambda a: a[best_idx], results.new_params)
        new_value = results.new_value[best_idx]
        new_grad = jax.tree_util.tree_map(lambda a: a[best_idx], results.new_grad)
        step_size = results.step_size[best_idx]

        # Recompute aux at the accepted point if needed.
        if self.has_aux:
            (_, aux), _ = self._value_and_grad(new_params, *args)
        else:
            aux = None

        # Update the L-BFGS oracle with the new (s, y) curvature pair.
        new_lbfgs_state = update_lbfgs_history(
            state.lbfgs_state, new_params, new_grad, self.history_size
        )

        error = tree_l2_norm(new_grad)
        new_state = QQNState(
            iter=state.iter + 1,
            value=new_value,
            grad=new_grad,
            lbfgs_state=new_lbfgs_state,
            step_size=step_size,
            error=error,
            done=error <= self.tol,
            aux=aux,
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
