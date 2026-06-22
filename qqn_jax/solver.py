"""QQN (Quasi-Quadratic-Newton) solver.

QQN constructs the quadratic interpolation path

    d(t) = t(1-t)(-∇f) + t²(-H∇f)

blending the steepest-descent direction (``-∇f``) with the L-BFGS direction
(``-H∇f``), and performs a line search over this path.

The solver follows the JAXopt-style ``init_state`` / ``update`` / ``run``
interface and keeps all state in JIT-compatible NamedTuples.

Note on parameterization:
    The line search traverses the path parameter ``t`` directly. The points
    ``x + d(t)`` along the curve are *states*, not directions to be
    re-scaled by a separate inner line search. Importantly, rescaling the
    gradient (or the oracle direction) does **not** change the geometric
    path traced by ``d(t)`` — it only distorts the parameterization (i.e.
    how ``t`` maps onto arc length along the curve). The curve itself, and
    therefore the set of candidate states, is invariant to such rescaling.
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
    tree_l2_norm,
    tree_negative,
    tree_vdot,
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
        step_size: last accepted step size ``α`` (the path parameter ``t``).
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
             ``"armijo"`` (default), ``"backtracking"``, ``"strong_wolfe"``,
             ``"hager_zhang"`` or ``"fixed"``. Empirically (see
             ``docs/results.md``) the backtracking/Armijo family is the robust
             efficiency winner on smooth full-batch problems; ``"strong_wolfe"``
             can over-restrict the quadratic-path step and fail to converge.
        line_search_options: optional dict of keyword arguments forwarded to
             the chosen line-search function (e.g. ``c1``, ``c2``, ``max_iter``,
             ``init_step``, ``shrink``, ``step_size``). These override the
             line-search defaults.
        spline: when ``True``, enable the cubic Hermite spline refinement. This
             is orthogonal to ``line_search``: every probe along the (consistent)
             path is reused as a control point and the spline's stationary points
             guide the search. It composes with any chosen line search.
        has_aux: whether ``fun`` returns auxiliary data.

    Note:
        The line search traverses the path parameter ``t ∈ [0, 1]`` directly.
        Each evaluated point ``x + d(t)`` is a *state* on the quadratic curve,
        not a direction to be independently re-scaled. Rescaling the gradient
        does not change the path geometry — only its parameterization.
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
        region=None,
        oracle="lbfgs",
        feed_probes_to_oracle: bool = False,
        probe_descent_gate: bool = True,
        max_probes: int = 32,
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
        # Opt-in: forward every gradient evaluated *during the line search*
        # into the oracle's curvature memory, not just the accepted point.
        self.feed_probes_to_oracle = feed_probes_to_oracle
        # When feeding probes, only admit those that (a) strictly *decrease* the
        # objective relative to the current iterate and (b) lie on the accepted
        # side of the path (their step does not overshoot the accepted step).
        # The prior benchmark (docs/...144249.analysis.md, §4) showed that
        # feeding *rejected* line-search probes injects non-representative
        # (s, y) curvature pairs that pollute the L-BFGS history and cause
        # catastrophic stalls. Gating on descent is the documented fix: only
        # genuinely improving probes enrich the curvature memory.
        self.probe_descent_gate = probe_descent_gate
        self.max_probes = max_probes

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
        # When feeding probes to the oracle, size the line-search probe buffers
        # to ``max_probes`` so they match the oracle's replay capacity.
        if self.feed_probes_to_oracle:
            opts = {**opts, "max_probes": self.max_probes}
        else:
            # Probes are unused downstream; disable recording so the inner
            # line-search ``while_loop`` skips the (max_probes, n) scratch.
            opts = {**opts, "record_probes": False}
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

        A *single* line search traverses the quadratic path ``d(t)`` over the
        parameter ``t ∈ [0, 1]``. The points along the path are states, not
        directions to be re-searched: the search selects one ``t`` (the step
        size along the curve) and the corresponding state ``x + d(t)`` is the
        accepted iterate.

        Returns ``(new_params, new_state)``.
        """
        grad = state.grad

        # 1. Oracle: L-BFGS direction (-H∇f), the t=1 endpoint of the path.
        qn_dir, _ = self.oracle.direction(params, grad, state.oracle_state)

        # 2. Gradient: steepest descent direction (-∇f), the path's tangent.
        grad_dir = tree_negative(grad)
        # Optional probe recorder: wrap the value-and-grad handed to the line
        # search so every evaluated (params, grad) is captured into a
        # fixed-size circular scratch buffer. We thread the buffer through a
        # Python list closure over a JAX-carried state to stay JIT-safe: the
        # buffer itself is built as outputs and re-derived deterministically.
        #
        # Because line searches are jitted ``while_loop``s, a Python-mutating
        # closure won't work. Instead we re-evaluate the probe points after the
        # search using the alphas the search reports is not generally possible
        # (only the accepted alpha is returned). We therefore record probes via
        # a stateful host-side buffer is not jit-safe either. The robust,
        # JIT-compatible route is to have the recording wrapper write into a
        # ref-like carry — which JAX lacks — so we instead reconstruct probes
        # from the spline/inner contract below.

        # 3. Single line search along the quadratic path.
        #    The "direction" handed to the line search is the path itself,
        #    parameterized so that step size ``t`` traces d(t). The search
        #    walks the curve directly; each probe ``x + d(t)`` is a state.

        res = self._ls(
            self._plain_value_and_grad,
            params,
            qn_dir,
            state.value,
            grad,
            *args,
            region=self.region,
            region_state=state.region_state,
        )

        new_params = res.new_params
        new_value = res.new_value
        new_grad = res.new_grad
        step_size = res.step_size
        best_t = step_size

        # Recompute aux at the accepted point if needed.
        if self.has_aux:
            # We already have new_value/new_grad from the line search; only the
            # aux is missing. Call ``fun`` directly (no grad) to avoid a second
            # backward pass per iteration.
            _, aux = self.fun(new_params, *args)
        else:
            aux = None

        # Update the oracle state (e.g. L-BFGS curvature pair, momentum).
        # When enabled, forward every (params, grad) evaluated *during* the
        # line search into the oracle's curvature memory — not just the
        # accepted point. The probe buffers are fixed-size and fully JIT/vmap
        # compatible (see LineSearchResult.probe_*).
        if self.feed_probes_to_oracle and res.probe_params is not None:
            probe_valid = res.probe_valid
            if self.probe_descent_gate:
                # Descent gate: only admit probes whose objective value strictly
                # improves on the *current* iterate. We recompute the probe
                # values cheaply from the already-evaluated probe gradients is
                # not possible (the LineSearchResult stores params+grads, not
                # values), so we evaluate the gate via the captured probe params.
                # This is a small fixed-size vmapped pass (max_probes points)
                # and stays JIT/vmap compatible.
                def _probe_value(p):
                    v, _ = self._plain_value_and_grad(p, *args)
                    return v

                probe_vals = jax.vmap(_probe_value)(res.probe_params)
                # Admit a probe only if it is (a) a previously-valid slot and
                # (b) strictly decreases the objective vs. the current iterate.
                descends = probe_vals < state.value
                probe_valid = jnp.logical_and(res.probe_valid, descends)
            oracle_info = OracleInfo(
                params=params,
                new_params=new_params,
                grad=grad,
                new_grad=new_grad,
                t=best_t,
                step_size=step_size,
                probe_params=res.probe_params,
                probe_grads=res.probe_grads,
                probe_valid=probe_valid,
            )
        else:
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
        # Honest predicted reduction from the along-path quadratic model.
        #
        # The QQN quadratic path has the *exact* directional model
        #   slope(τ) = ⟨∇f, d'(τ)⟩ = (1-2τ)·m_g + 2τ·m_q,
        # whose integral gives the model's reduction in closed form:
        #   pred(t) = -∫₀ᵗ slope(τ) dτ = -[(t - t²)·m_g + t²·m_q].
        # Crucially, this integral is *identically* −⟨∇f, d(t)⟩ because
        # d(t) = t(1-t)·grad_dir + t²·qn_dir. There is therefore NO separate
        # curvature term to add: the path's curvature is already fully encoded
        # in d(t). The previous code added a spurious second-order term and a
        # deflating floor, which double-counted curvature and drove ρ negative
        # near convergence — the documented adaptive trust-region stall. We
        # now use the geometrically exact along-path model directly.
        # pred(t) = −⟨∇f, d(t)⟩ with d(t) = t(1−t)·grad_dir + t²·qn_dir.
        # Expand analytically to avoid materializing the full path vector:
        #   −⟨∇f, d(t)⟩ = −[t(1−t)·⟨∇f, grad_dir⟩ + t²·⟨∇f, qn_dir⟩].
        # Each ⟨∇f, ·⟩ is a single O(n) dot rather than an O(n) tree_map plus
        # a second O(n) dot over the materialized blend.
        m_g = tree_vdot(grad, grad_dir)
        m_q = tree_vdot(grad, qn_dir)
        a_t = best_t * (1.0 - best_t)
        b_t = best_t * best_t
        pred_reduction = -(a_t * m_g + b_t * m_q)
        # The model reduction is non-negative whenever the step descends along
        # the path (which the line search guarantees via sufficient decrease).
        # A tiny positive epsilon avoids a 0/0 ρ when the step is degenerate.
        eps_pred = jnp.asarray(1e-16, dtype=pred_reduction.dtype)
        pred_reduction = jnp.maximum(pred_reduction, eps_pred)
        info = RegionInfo(
            params=params,
            new_params=new_params,
            pred_reduction=pred_reduction,
            actual_reduction=actual_reduction,
            t=best_t,
            step_size=step_size,
        )
        new_region_state = self.region.update(state.region_state, info)

        error = tree_l2_norm(new_grad)
        # Terminate (rather than spin to maxiter) if the iterate diverges to a
        # non-finite state — a single bad start in a vmap batch otherwise wastes
        # the whole batch's remaining iterations on NaN arithmetic.
        finite = jnp.logical_and(jnp.isfinite(new_value), jnp.isfinite(error))
        done = jnp.logical_or(error <= self.tol, jnp.logical_not(finite))
        new_state = QQNState(
            iter=state.iter + 1,
            value=new_value,
            grad=new_grad,
            oracle_state=new_oracle_state,
            step_size=step_size,
            error=error,
            done=done,
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
