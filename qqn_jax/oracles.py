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
    init_lbfgs_state,
    lbfgs_direction,
    update_lbfgs_history,
    update_lbfgs_history_batch,
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
         probe_params: optional ``(k, n)`` buffer of line-search probe points.
         probe_grads: optional ``(k, n)`` buffer of probe gradients.
         probe_valid: optional ``(k,)`` boolean mask of filled probe slots.
    """

    params: Any = None
    new_params: Any = None
    grad: Any = None
    new_grad: Any = None
    t: Any = None
    step_size: Any = None
    probe_params: Any = None
    probe_grads: Any = None
    probe_valid: Any = None
    probe_alphas: Any = None


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
        # When line-search probes are supplied, replay them oldest-first so
        # every gradient evaluated along the path contributes curvature, then
        # finish with the accepted point as the newest pair. Otherwise fall
        # back to the single-pair update (byte-for-byte legacy behavior).
        # The replay path additionally needs probe_alphas (to sort the probes
        # into monotone-α order) and probe_valid (to mask empty slots). Some
        # line searches (e.g. the spline-wrapped variant) record probe params
        # and grads but neither alphas nor a valid mask — in that case we
        # cannot meaningfully replay, so fall back to the single-pair update.
        if (
            info.probe_params is None
            or info.probe_alphas is None
            or info.probe_valid is None
        ):
            return update_lbfgs_history(
                state, info.new_params, info.new_grad, history_size
            )
        # Replay probes in INCREASING-α order. The probes are collected by the
        # line search in slot order (slot 0 = init_step, the *largest* α, then
        # shrinking) — feeding them in that arbitrary order makes the secant
        # differences s_k = p_k - p_{k-1} point back-and-forth along the search
        # ray, producing sign-flipping yᵀs that the curvature guard rejects as
        # noise. Sorting by α yields monotone-spaced probes whose differences
        # are consistently oriented, which is the only ordering under which the
        # replayed pairs carry meaningful (1-D) curvature.
        #
        # NOTE: all probes are collinear (they lie on the single ray
        # x + α·d), so even sorted they only enrich curvature *along d*. This
        # cannot substitute for genuine cross-iteration curvature; it is a
        # mild secant refinement of the t=1 endpoint at best.
        order = jnp.argsort(jnp.where(info.probe_valid, info.probe_alphas, jnp.inf))
        probe_params = info.probe_params[order]
        probe_grads = info.probe_grads[order]
        probe_valid = info.probe_valid[order]
        # Append the accepted point as the final (newest) probe.
        params_seq = jnp.concatenate([probe_params, info.new_params[None, :]], axis=0)
        grad_seq = jnp.concatenate([probe_grads, info.new_grad[None, :]], axis=0)
        valid_seq = jnp.concatenate([probe_valid, jnp.asarray([True])], axis=0)
        return update_lbfgs_history_batch(
            state, params_seq, grad_seq, valid_seq, history_size
        )

    return Oracle(init=init, direction=direction, update=update)


# --- Momentum Oracle --------------------------------------------------
class MomentumState(NamedTuple):
    # ``velocity`` is the decaying-weight average of the *actual* per-iteration
    # parameter deltas Δx = x_new − x (the realized steps), NOT of gradients.
    velocity: jnp.ndarray


def MomentumOracle(beta: float = 0.9) -> Oracle:
    """First-order accelerated (heavy-ball) oracle.

    The ``t = 1`` endpoint blends the current steepest-descent move with a
    decaying-weight average of the *actual per-iteration deltas* Δx = x_new − x
    that the solver has already realized::

        (committed in update, after each accepted step)
        v_new      = β · v + (1 − β) · Δx        Δx = x_new − x

        (returned by direction at the current iterate)
        direction  = -∇f + β · v

    Here ``v`` is the running momentum of realized steps, so the oracle nudges
    the t=1 endpoint along the direction the optimizer has actually been
    travelling — true heavy-ball momentum — rather than along an average of raw
    gradients. On the very first step ``v = 0`` and the endpoint reduces to
    plain steepest descent, preserving the ``d'(0)`` anchor.
    """

    def init(params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return MomentumState(velocity=zeros)

    def direction(params, grad, state):
        # The endpoint is the steepest-descent move plus the accumulated
        # momentum of past *realized deltas*. Do NOT mutate state here:
        # ``direction`` is called for its endpoint only, and the solver
        # discards the returned oracle state — the persisted state comes solely
        # from ``update`` (which commits the new delta after a step lands).
        d = jax.tree_util.tree_map(lambda g, v: -g + beta * v, grad, state.velocity)
        return d, state

    def update(state, info):
        # Accumulate the *actual per-iteration delta* Δx = x_new − x into the
        # decaying-weight average. This is the only state the solver persists
        # across iterations, so the realized-step momentum must accumulate here.
        delta = jax.tree_util.tree_map(
            lambda xn, x: xn - x, info.new_params, info.params
        )
        v_new = jax.tree_util.tree_map(
            lambda v, dx: beta * v + (1.0 - beta) * dx, state.velocity, delta
        )
        return MomentumState(velocity=v_new)

    return Oracle(init=init, direction=direction, update=update)


# --- Shampoo Oracle ---------------------------------------------------
# --- Secant (Barzilai-Borwein) Oracle --------------------------------
class SecantState(NamedTuple):
    prev_params: jnp.ndarray
    prev_grad: jnp.ndarray
    alpha: jnp.ndarray  # current inverse-curvature step scale
    step_count: jnp.ndarray


def SecantOracle(alpha0: float = 1.0, alpha_max: float = 1e3) -> Oracle:
    """Barzilai-Borwein curvature oracle (matrix-free, O(n) memory).
    The ``t = 1`` endpoint is the gradient scaled by an inverse-curvature
    estimate inferred from the *realized* secant of the previous step::
        s = x      - x_prev
        y = ∇f     - ∇f_prev
        α = ⟨s, s⟩ / ⟨s, y⟩        (BB1 step; the Rayleigh quotient's inverse)
        direction = -α · ∇f
    This reuses the curvature signal that the path *already measured* —
    no Hessian, no history buffers. It is a featherweight companion for a
    ``Fallback`` and a probe of how much curvature lives in a single step.
    The very first step (no secant yet) falls back to ``-alpha0 · ∇f``,
    i.e. plain scaled steepest descent, preserving the ``d'(0)`` anchor.
    """
    eps = 1e-12

    def init(params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return SecantState(
            prev_params=params,
            prev_grad=zeros,
            alpha=jnp.asarray(alpha0, dtype=params.dtype),
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def direction(params, grad, state):
        d = tree_negative(jax.tree_util.tree_map(lambda g: state.alpha * g, grad))
        return d, state

    def update(state, info):
        s = jax.tree_util.tree_map(lambda a, b: a - b, info.new_params, info.params)
        y = jax.tree_util.tree_map(lambda a, b: a - b, info.new_grad, info.grad)
        ss = sum(jnp.vdot(si, si) for si in jax.tree_util.tree_leaves(s))
        sy = sum(
            jnp.vdot(si, yi)
            for si, yi in zip(
                jax.tree_util.tree_leaves(s), jax.tree_util.tree_leaves(y)
            )
        )
        # BB1 step; guard against non-positive curvature by retaining prior α.
        curvature_ok = sy > eps
        bb = ss / jnp.where(curvature_ok, sy, 1.0)
        new_alpha = jnp.where(curvature_ok, jnp.clip(bb, eps, alpha_max), state.alpha)
        return SecantState(
            prev_params=info.new_params,
            prev_grad=info.new_grad,
            alpha=new_alpha.astype(state.alpha.dtype),
            step_count=state.step_count + 1,
        )

    return Oracle(init=init, direction=direction, update=update)


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

        do_refresh = (state.step % update_freq) == 0

        # The (n,n) outer product ``g gᵀ`` is O(n²) every step; only the L
        # accumulator is rank-meaningful here (R is 1×1). Accumulate R cheaply
        # always, but only pay for the dense L update + eigh on a refresh.
        # ``grad @ grad`` is a single O(n) dot; keep R as a (1,1) matrix so the
        # shape is stable for ``_matrix_inverse_pth_root`` on refresh.
        R_new = state.R + jnp.vdot(grad, grad).reshape(1, 1)

        def refresh(_):
            L_new = state.L + g @ g.T
            Lr = _matrix_inverse_pth_root(L_new, 4.0, epsilon)
            Rr = _matrix_inverse_pth_root(R_new, 4.0, epsilon)
            precond = (Lr @ g) @ Rr  # (n, 1)
            return precond.reshape(-1), L_new

        def keep(_):
            # Fall back to scaled gradient when not refreshing roots.
            return grad, state.L

        precond, L_new = jax.lax.cond(do_refresh, refresh, keep, operand=None)
        d = -precond
        new_state = ShampooState(L=L_new, R=R_new, step=state.step + 1)
        return d, new_state

    def update(state, info):
        return state

    return Oracle(init=init, direction=direction, update=update)


# --- Anderson Acceleration Oracle ------------------------------------
class AndersonState(NamedTuple):
    """Residual/iterate windows for Anderson (Type-II) acceleration.

    Attributes:
        g_history: window of recent gradients (residuals), (m, n).
        x_history: window of recent iterates,             (m, n).
         step_count:     number of valid columns currently stored.
    """

    g_history: jnp.ndarray
    x_history: jnp.ndarray
    step_count: jnp.ndarray


def AndersonOracle(window: int = 5, reg: float = 1e-8, beta: float = 1.0) -> Oracle:
    """Anderson-accelerated (Type-II) oracle — the variational ideal that
    L-BFGS approximates.

    The ``t = 1`` endpoint is formed by solving a tiny constrained
    least-squares problem over recent gradient *differences*::

        min_θ ‖ ∇f − ΔG θ ‖²  (+ reg·‖θ‖²)
         direction = −β·( ∇f − ΔG θ )  −  ΔX θ

    where ΔG, ΔX are first-differences of the stored gradient/iterate
    windows. With ``window=1`` this reduces to a secant step; with a deep
    window it captures multi-step curvature the single-secant cannot. No
    Hessian is ever formed; the only solve is an ``(m × m)`` system.
     The *coupling constant* ``β`` (the mixing parameter of the classical
     Anderson scheme) rescales the accelerated residual toward the
     gradient's natural magnitude. ``β = 1`` recovers the pure Type-II
     update; ``β > 1`` lets the deep-residual descent stretch, converting
     this oracle's leading trajectory-AUC into a leading *iteration* count.
    """

    def init(params):
        n = params.shape[0]
        zeros = jnp.zeros((window, n), dtype=params.dtype)
        return AndersonState(
            g_history=zeros,
            x_history=zeros,
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def direction(params, grad, state):
        # Build first-difference matrices from the window (newest-first).
        # ΔG[:, k] = g_k − g_{k+1}, ΔX[:, k] = x_k − x_{k+1}. Unfilled
        # slots are zero and contribute nothing to the normal equations.
        g_hist = state.g_history
        x_hist = state.x_history
        # Differences anchored on the present iterate, computed without the
        # extra (window+1, n) concat allocations: the first column is
        # (current - newest_stored), the rest are stored[k] - stored[k+1].
        # dG[:, 0] = grad - g_hist[0]; dG[:, k>=1] = g_hist[k-1] - g_hist[k].
        dG_first = (grad - g_hist[0])[:, None]
        dX_first = (params - x_hist[0])[:, None]
        dG_rest = (g_hist[:-1] - g_hist[1:]).T  # (n, window-1)
        dX_rest = (x_hist[:-1] - x_hist[1:]).T
        dG = jnp.concatenate([dG_first, dG_rest], axis=1)  # (n, window)
        dX = jnp.concatenate([dX_first, dX_rest], axis=1)

        # Solve (dGᵀ dG + reg·I) θ = dGᵀ ∇f  — an (m × m) system.
        m = dG.shape[1]
        gram = dG.T @ dG
        # Scale-aware Tikhonov: anchor the regularizer to the Gram trace so
        # conditioning is invariant to the magnitude of the residual window.
        trace = jnp.trace(gram)
        scale = jnp.where(trace > 0.0, trace / m, 1.0)
        eye_m = jnp.eye(m, dtype=grad.dtype)
        A = gram + reg * scale * eye_m
        b = dG.T @ grad
        # Mask columns with no stored history so empty windows are inert.
        active = jnp.arange(m) < state.step_count
        b = jnp.where(active, b, 0.0)
        # Mask inactive rows/cols to the identity and add an absolute diagonal
        # ridge in one fused step: a degenerate window can otherwise leave A
        # near-singular, making solve() emit NaN that backprops through the
        # downstream safeguard. The ridge guarantees SPD-ness.
        active_mask = active[:, None] & active[None, :]
        A = jnp.where(active_mask, A, eye_m) + (
            jnp.asarray(1e-12, dtype=grad.dtype) * eye_m
        )
        theta = jnp.linalg.solve(A, b)
        theta = jnp.where(active, theta, 0.0)

        # Accelerated residual and the corresponding iterate correction.
        residual = grad - dG @ theta
        d = -beta * residual - dX @ theta
        # Safeguard: fall back to steepest descent if the solve degenerates.
        ok = jnp.all(jnp.isfinite(d)) & (state.step_count > 0)
        d = jnp.where(ok, d, -grad)
        return d, state

    def update(state, info):
        # Roll the windows, inserting the freshly-accepted (x, g).
        new_x = jnp.roll(state.x_history, shift=1, axis=0).at[0].set(info.new_params)
        new_g = jnp.roll(state.g_history, shift=1, axis=0).at[0].set(info.new_grad)
        new_count = jnp.minimum(state.step_count + 1, window)
        return AndersonState(g_history=new_g, x_history=new_x, step_count=new_count)

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
            # Validity is *descent*, not mere non-zeroness. A finite, non-zero
            # quasi-Newton direction that points uphill (⟨∇f, d⟩ ≥ 0) is worse
            # than useless — it betrays a degenerate curvature estimate. The
            # fallback must trigger on misalignment, not just on collapse.
            gd = jnp.vdot(grad, d)
            finite = jnp.all(jnp.isfinite(d))
            nonzero = jnp.vdot(d, d) > jnp.asarray(0.0, dtype=d.dtype)
            descent = gd < jnp.asarray(0.0, dtype=d.dtype)
            valid = finite & nonzero & descent
            if chosen is None:
                chosen = d
                chosen_valid = valid
            else:
                # Order-critical: snapshot chosen_valid BEFORE the OR-update so
                # we keep the *first* valid direction. Reordering these three
                # lines silently breaks the fallback priority.
                assert chosen is not None
                assert chosen_valid is not None
                take_prev = chosen_valid
                chosen = jnp.where(take_prev, chosen, d)
                chosen_valid = chosen_valid | valid
        # Terminal safety net: if *every* oracle produced an invalid (uphill /
        # non-finite / zero) direction, fall back to steepest descent so the
        # path's t=1 endpoint can never be a non-descent or NaN direction.
        neg_grad = tree_negative(grad)
        assert chosen is not None
        assert chosen_valid is not None
        chosen = jnp.where(chosen_valid, chosen, neg_grad)
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
        if oracle == "secant":
            return SecantOracle()
        if oracle == "anderson":
            return AndersonOracle()
        if oracle == "anderson+secant":
            # The variational ideal, safeguarded by a featherweight secant —
            # a strictly-dominant pairing when the residual solve degenerates.
            return Fallback([AndersonOracle(window=5), SecantOracle()])
        if oracle == "lbfgs+secant":
            # Your data's "best zero-storage safety net": deep curvature while
            # healthy, finite curvature the instant the history collapses.
            return Fallback([LBFGSOracle(history_size=history_size), SecantOracle()])
        raise ValueError(
            f"Unknown oracle: {oracle!r}. "
            "Available: 'lbfgs', 'momentum', 'shampoo', 'secant', 'anderson', "
            "'anderson+secant', 'lbfgs+secant' or an Oracle instance."
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
    "SecantOracle",
    "AndersonOracle",
    "Fallback",
    "resolve_oracle",
]
