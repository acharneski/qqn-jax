"""Spline (cubic Hermite) line search for QQN.

Each evaluation along the quadratic path ``d(t)`` yields both a fitness
value ``f(d(t))`` and a directional derivative ``m = ⟨∇f, d'(t)⟩``. The
standard backtracking search discards the gradient information; this search
instead treats every measurement as a reusable *control point* and builds a
piecewise cubic Hermite spline model of the objective along the path.

Candidate steps are proposed by locating stationary points of the cubic
segments (closed-form roots of the quadratic derivative). Tangents are
oriented via the upstream/downstream symmetry rule so spurious inflections
do not mislead the search.

See ``spline_search.md`` for the full specification.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from qqn_jax.utils import tree_add_scaled, tree_vdot, quadratic_path_derivative
from qqn_jax.regions import resolve_region
from qqn_jax.line_search import LineSearchResult


def _orient_tangents(h, f0, m0, f1, m1):
    """Apply the upstream/downstream symmetry correction to tangents.

    For a segment spanning ``(t0, f0)`` and ``(t1, f1)`` with width ``h``,
    the secant slope is ``Δ = (f1 - f0) / h``. Any endpoint tangent whose
    sign opposes ``Δ`` (and ``Δ ≠ 0``) is reflected so it aligns with the
    channel's natural flow. When ``Δ = 0`` the raw tangents are kept.
    """
    delta = (f1 - f0) / h
    flat = delta == 0.0

    def reflect(m):
        opposed = jnp.logical_and(jnp.sign(m) != jnp.sign(delta), delta != 0.0)
        m_corr = jnp.where(opposed, -m, m)
        # When the secant is flat, keep the raw tangent untouched.
        return jnp.where(flat, m, m_corr)

    return reflect(m0), reflect(m1)


def _segment_value(s, h, f0, m0, f1, m1):
    """Cubic Hermite interpolated fitness at normalized parameter ``s``."""
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * f0 + h10 * h * m0 + h01 * f1 + h11 * h * m1


def _segment_stationary_candidates(t0, t1, f0, m0, f1, m1):
    """Return up to two candidate ``(t, predicted_value)`` stationary points.

    Differentiating the cubic Hermite segment w.r.t. ``s`` gives a quadratic
    ``A s² + B s + C = 0``. We solve it in closed form, mask roots outside
    ``[0, 1]`` (or non-real ones), and map valid roots back to ``t``.

    Returns arrays ``(t_cands, val_cands, valid)`` each of length 2.
    """
    h = t1 - t0
    m0o, m1o = _orient_tangents(h, f0, m0, f1, m1)

    # f'(s) coefficients (see spline_search.md):
    #   f'(s) = (6s² - 6s)·f0 + (3s² - 4s + 1)·h·m0
    #         + (-6s² + 6s)·f1 + (3s² - 2s)·h·m1
    hm0 = h * m0o
    hm1 = h * m1o
    A = 6.0 * f0 + 3.0 * hm0 - 6.0 * f1 + 3.0 * hm1
    B = -6.0 * f0 - 4.0 * hm0 + 6.0 * f1 - 2.0 * hm1
    C = hm0

    eps = jnp.asarray(1e-12, dtype=f0.dtype)
    disc = B * B - 4.0 * A * C
    disc_ok = disc >= 0.0
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))

    # Quadratic branch (A != 0).
    denom = jnp.where(jnp.abs(A) > eps, 2.0 * A, 1.0)
    root1 = (-B + sqrt_disc) / denom
    root2 = (-B - sqrt_disc) / denom

    # Linear fallback when A ~ 0: B s + C = 0.
    lin_root = jnp.where(
        jnp.abs(B) > eps, -C / jnp.where(jnp.abs(B) > eps, B, 1.0), -1.0
    )
    is_quad = jnp.abs(A) > eps

    s1 = jnp.where(is_quad, root1, lin_root)
    s2 = jnp.where(is_quad, root2, -1.0)  # second root invalid in linear case

    def finalize(s, extra_valid):
        in_range = jnp.logical_and(s >= 0.0, s <= 1.0)
        valid = jnp.logical_and(in_range, extra_valid)
        s_clip = jnp.clip(s, 0.0, 1.0)
        t = t0 + s_clip * h
        val = _segment_value(s_clip, h, f0, m0o, f1, m1o)
        # Invalid candidates get +inf so argmin never selects them.
        val = jnp.where(valid, val, jnp.asarray(jnp.inf, dtype=f0.dtype))
        return t, val, valid

    t_c1, v_c1, ok1 = finalize(s1, jnp.logical_and(disc_ok, is_quad) | (~is_quad))
    t_c2, v_c2, ok2 = finalize(s2, jnp.logical_and(disc_ok, is_quad))

    t_cands = jnp.stack([t_c1, t_c2])
    val_cands = jnp.stack([v_c1, v_c2])
    valid = jnp.stack([ok1, ok2])
    return t_cands, val_cands, valid


def spline_search(
    value_and_grad_fn: Callable,
    params,
    direction,
    value,
    grad,
    *args,
    grad_dir=None,
    qn_dir=None,
    t=None,
    init_step: float = 1.0,
    c1: float = 1e-4,
    max_iter: int = 10,
    region=None,
    region_state=None,
) -> LineSearchResult:
    """Cubic Hermite spline line search over the path direction ``direction``.

    The search accumulates ``(α, f, m)`` control points where ``α`` is the
    step size, ``f`` the measured fitness, and ``m`` the measured directional
    derivative ``⟨∇f(x + α·d), d⟩`` along the *fixed* direction ``d``. A cubic
    Hermite spline is fit to the active bracket and its stationary points
    guide the next probe.

    This implementation parameterizes by the step size ``α`` along the
    provided ``direction`` (the spec's path parameter ``t`` maps to the QQN
    ``t``-grid handled by the solver; here ``d`` is already constructed).

    Args:
        grad_dir, qn_dir, t: optional QQN path metadata (unused directly but
            accepted for interface compatibility with the solver).
    """
    region = resolve_region(region)

    def project(candidate):
        return region.project(params, candidate, region_state)

    dd = tree_vdot(direction, direction)

    def eval_at(alpha):
        raw = tree_add_scaled(params, alpha, direction)
        projected = project(raw)
        val, g = value_and_grad_fn(projected, *args)
        # Directional derivative along the fixed direction d.
        slope = tree_vdot(g, direction)
        return projected, val, g, slope

    dtype = value.dtype

    # Control point 0: the current point (alpha = 0).
    a0 = jnp.asarray(0.0, dtype=dtype)
    f0 = value
    # Slope at alpha=0 is gᵀd (already available).
    m0 = tree_vdot(grad, direction)

    # Control point 1: the initial probe (alpha = init_step).
    a1 = jnp.asarray(init_step, dtype=dtype)
    p1, f1, g1, m1 = eval_at(a1)

    # Carry holds the active bracket: two control points (a0,f0,m0),(a1,f1,m1)
    # plus the best-so-far accepted point.
    def armijo_ok(alpha, fval):
        return fval <= value + c1 * alpha * m0

    # Initialize best with whichever of the two endpoints is feasible+lowest.
    best_alpha = a1
    best_params = p1
    best_val = f1
    best_grad = g1

    InitCarry = (
        a0,
        f0,
        m0,  # left control point
        a1,
        f1,
        m1,  # right control point
        best_alpha,
        best_val,  # best so far
        best_params,
        best_grad,
        jnp.asarray(0, jnp.int32),
    )

    def cond(carry):
        (la, lf, lm, ra, rf, rm, ba, bv, bp, bg, i) = carry
        # Stop once Armijo is satisfied at the best point or iters exhausted.
        satisfied = armijo_ok(ba, bv)
        return jnp.logical_and(jnp.logical_not(satisfied), i < max_iter)

    def body(carry):
        (la, lf, lm, ra, rf, rm, ba, bv, bp, bg, i) = carry

        # Propose stationary points of the cubic over the bracket [la, ra].
        t_cands, v_cands, valid = _segment_stationary_candidates(la, ra, lf, lm, rf, rm)

        # Fallback proposal: midpoint of the bracket (always valid).
        mid = 0.5 * (la + ra)
        mid_val = jnp.asarray(jnp.inf, dtype=dtype)  # unknown until evaluated

        # Choose the candidate with the lowest predicted value; if neither
        # stationary point is valid, use the midpoint.
        any_valid = jnp.any(valid)
        best_c_idx = jnp.argmin(v_cands)
        cand_alpha = jnp.where(any_valid, t_cands[best_c_idx], mid)

        # Keep the proposal strictly inside the bracket to ensure progress.
        lo = jnp.minimum(la, ra)
        hi = jnp.maximum(la, ra)
        span = hi - lo
        margin = 1e-3 * jnp.maximum(span, 1e-12)
        cand_alpha = jnp.clip(cand_alpha, lo + margin, hi - margin)

        # Evaluate the proposed step.
        cp, cf, cg, cm = eval_at(cand_alpha)

        # Update best-so-far if this point improves on it.
        improves = cf < bv
        n_ba = jnp.where(improves, cand_alpha, ba)
        n_bv = jnp.where(improves, cf, bv)
        n_bp = jax.tree_util.tree_map(
            lambda new, old: jnp.where(improves, new, old), cp, bp
        )
        n_bg = jax.tree_util.tree_map(
            lambda new, old: jnp.where(improves, new, old), cg, bg
        )

        # Tighten the bracket: replace the endpoint with the higher fitness.
        # This keeps the cubic anchored around the lower region.
        replace_right = lf <= rf
        n_la = jnp.where(replace_right, la, cand_alpha)
        n_lf = jnp.where(replace_right, lf, cf)
        n_lm = jnp.where(replace_right, lm, cm)
        n_ra = jnp.where(replace_right, cand_alpha, ra)
        n_rf = jnp.where(replace_right, cf, rf)
        n_rm = jnp.where(replace_right, cm, rm)

        return (
            n_la,
            n_lf,
            n_lm,
            n_ra,
            n_rf,
            n_rm,
            n_ba,
            n_bv,
            n_bp,
            n_bg,
            i + 1,
        )

    final = jax.lax.while_loop(cond, body, InitCarry)
    (_, _, _, _, _, _, fa, fv, fp, fg, _) = final

    done = armijo_ok(fa, fv)
    return LineSearchResult(
        step_size=fa,
        new_value=fv,
        new_grad=fg,
        new_params=fp,
        done=done,
    )


__all__ = ["spline_search"]
