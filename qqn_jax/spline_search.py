"""Spline (cubic Hermite) augmentation for QQN line searches.

Each evaluation along the quadratic path ``d(t)`` yields both a fitness
value ``f(d(t))`` and a directional derivative ``m = ⟨∇f, d'(t)⟩``. The
spline does *not* replace the line search; it is an **expanded definition
of the curve** that reuses every measured point as a reusable *control
point* of a piecewise cubic Hermite spline model of the objective along
the (consistent) path.

``spline_wrap(inner_search)`` returns a line-search-compatible callable
that first runs ``inner_search`` (any registered strategy), then attempts
to *improve* on its accepted point by probing the stationary points of the
cubic Hermite spline fit through the control points gathered so far. Because
the path direction is consistent across all measured points, every probe —
regardless of the underlying line search — is a valid control point.

Candidate steps are proposed by locating stationary points of the cubic
segments (closed-form roots of the quadratic derivative). Tangents are
oriented via the upstream/downstream symmetry rule so spurious inflections
do not mislead the search.

See ``spline_search.md`` for the full specification.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from qqn_jax.utils import (
    tree_add_scaled,
    tree_vdot,
    quadratic_path,
    quadratic_path_derivative,
)
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


def spline_wrap(inner_search: Callable) -> Callable:
    """Augment ``inner_search`` with a cubic Hermite spline refinement.

    Returns a line-search-compatible callable with the same signature as the
    wrapped ``inner_search``. The spline is an *expanded definition of the
    curve*, not a competing line search: it reuses the consistent path's
    measured points as control points and probes the stationary points of
    the resulting cubic Hermite spline to try to improve on the inner
    search's accepted step.

    The wrapped search:

    1. Runs ``inner_search`` to obtain a baseline accepted point.
    2. Forms control points from ``α = 0`` (current point, slope ``gᵀd``)
       and ``α = α_inner`` (the inner search's accepted point, with its
       measured slope).
    3. Probes the spline's stationary points, projecting through the region
       and keeping the lowest-value feasible point found.
    4. Returns the better of the inner result and the spline probes.

    Because every probe lies on the *same* fixed direction ``d`` (the path
    stays consistent w.r.t. all measured points), this composes correctly
    with any underlying line search.
    """

    def wrapped(
        value_and_grad_fn: Callable,
        params,
        direction,
        value,
        grad,
        *args,
        spline_max_iter: int = 6,
        spline_extrapolate: float = 2.0,
        grad_dir=None,
        qn_dir=None,
        region=None,
        region_state=None,
        **inner_kwargs,
    ) -> LineSearchResult:
        region = resolve_region(region)
        use_path = grad_dir is not None and qn_dir is not None

        def project(candidate):
            return region.project(params, candidate, region_state)

        def eval_at(alpha):
            if use_path:
                step = quadratic_path(alpha, grad_dir, qn_dir)
                raw = jax.tree_util.tree_map(lambda p, s: p + s, params, step)
                tangent = quadratic_path_derivative(alpha, grad_dir, qn_dir)
            else:
                raw = tree_add_scaled(params, alpha, direction)
                tangent = direction
            projected = project(raw)
            val, g = value_and_grad_fn(projected, *args)
            slope = tree_vdot(g, tangent)
            return projected, val, g, slope

        dtype = value.dtype

        # 1. Run the wrapped inner line search to get a baseline.
        inner = inner_search(
            value_and_grad_fn,
            params,
            direction,
            value,
            grad,
            *args,
            grad_dir=grad_dir,
            qn_dir=qn_dir,
            region=region,
            region_state=region_state,
            **inner_kwargs,
        )

        # 2. Control points: alpha=0 (current point) and the inner result.
        a0 = jnp.asarray(0.0, dtype=dtype)
        f0 = value
        # Slope at alpha=0 along the path tangent d'(0). For the quadratic path
        # d'(0) = grad_dir = -∇f, so the slope is ⟨∇f, -∇f⟩ = -‖∇f‖².
        if use_path:
            m0 = tree_vdot(grad, grad_dir)
        else:
            m0 = tree_vdot(grad, direction)

        a1 = inner.step_size
        f1 = inner.new_value
        if use_path:
            m1 = tree_vdot(
                inner.new_grad, quadratic_path_derivative(a1, grad_dir, qn_dir)
            )
        else:
            m1 = tree_vdot(inner.new_grad, direction)

        # --- Superlinear probe (the symmetry your doc promised) ----------
        # If the downstream tangent still descends (m1 < 0), the minimum is
        # *beyond* the inner step. Extend the bracket once and let the cubic
        # locate the stationary point in closed form. Region-projected and
        # gated on strict improvement, so it is monotone-safe and free when
        # the inner step already overshot (m1 >= 0 leaves the bracket intact).
        still_descending = m1 < 0.0
        # Adaptive superlinear extension. A constant cap throttles the very
        # stretch we intend. The minimizer of a descending quadratic lies at
        # α* = a1 · m0 / (m0 - m1) (linear-slope model along the curve). When
        # the downstream slope m1 is still close to the initial slope m0, the
        # path is barely bending and the minimum is far ahead; when m1 → 0 it
        # is nearly upon us. We project toward that estimated stationary point,
        # clamped to a generous multiple of the inner step for safety.
        eps_ext = jnp.asarray(1e-12, dtype=dtype)
        denom = jnp.where(jnp.abs(m0 - m1) > eps_ext, m0 - m1, eps_ext)
        alpha_star = a1 * (m0 / denom)
        # The minimizer of a descending quadratic can lie far beyond a fixed
        # multiple of a1 when m1 ≪ m0 (the deep superlinear regime). A linear
        # leash throttles arrival at precisely the moment of acceleration. We
        # relax the cap by the log-ratio of the slopes, letting genuinely
        # superlinear steps stretch while remaining bounded and monotone-gated.
        slope_ratio = jnp.abs(m0) / (jnp.abs(m1) + eps_ext)
        relaxed_cap = spline_extrapolate * (1.0 + jnp.log1p(slope_ratio))
        a_ext = jnp.where(
            still_descending,
            jnp.clip(
                alpha_star,
                jnp.maximum(a1, 1e-3),
                relaxed_cap * jnp.maximum(a1, 1e-3),
            ),
            a1,
        )
        ep, ef, eg, em = eval_at(a_ext)
        # Adopt the extended point as the right anchor when it both extends
        # the bracket and does not worsen fitness; otherwise keep the inner.
        take_ext = jnp.logical_and(still_descending, ef <= f1)
        a1 = jnp.where(take_ext, a_ext, a1)
        f1 = jnp.where(take_ext, ef, f1)
        m1 = jnp.where(take_ext, em, m1)
        ext_params = jax.tree_util.tree_map(
            lambda new, old: jnp.where(take_ext, new, old), ep, inner.new_params
        )
        ext_grad = jax.tree_util.tree_map(
            lambda new, old: jnp.where(take_ext, new, old), eg, inner.new_grad
        )

        # Best-so-far starts at the inner search's accepted point.
        InitCarry = (
            a0,
            f0,
            m0,
            a1,
            f1,
            m1,
            jnp.where(take_ext, a1, inner.step_size),
            jnp.where(take_ext, f1, inner.new_value),
            ext_params,
            ext_grad,
            jnp.asarray(0, jnp.int32),
        )

        def cond(carry):
            (_, _, _, _, _, _, _, _, _, _, i) = carry
            return i < spline_max_iter

        def body(carry):
            (la, lf, lm, ra, rf, rm, ba, bv, bp, bg, i) = carry

            # Stationary points of the cubic over the bracket [la, ra].
            t_cands, v_cands, valid = _segment_stationary_candidates(
                la, ra, lf, lm, rf, rm
            )

            # Midpoint fallback when no stationary point is valid.
            mid = 0.5 * (la + ra)
            any_valid = jnp.any(valid)
            best_c_idx = jnp.argmin(v_cands)
            cand_alpha = jnp.where(any_valid, t_cands[best_c_idx], mid)

            # Keep the proposal strictly inside the bracket for progress.
            lo = jnp.minimum(la, ra)
            hi = jnp.maximum(la, ra)
            span = hi - lo
            margin = 1e-3 * jnp.maximum(span, 1e-12)
            cand_alpha = jnp.clip(cand_alpha, lo + margin, hi - margin)

            # Evaluate the proposed step on the consistent path.
            cp, cf, cg, cm = eval_at(cand_alpha)

            # Keep the spline probe only if it genuinely improves.
            improves = cf < bv
            n_ba = jnp.where(improves, cand_alpha, ba)
            n_bv = jnp.where(improves, cf, bv)
            n_bp = jax.tree_util.tree_map(
                lambda new, old: jnp.where(improves, new, old), cp, bp
            )
            n_bg = jax.tree_util.tree_map(
                lambda new, old: jnp.where(improves, new, old), cg, bg
            )

            # Tighten the bracket so it still ENCLOSES the minimum.
            #
            # The candidate cf lies strictly inside [la, ra]. The minimum of a
            # well-bracketed unimodal segment is on the side of the candidate
            # whose endpoint has the *lower* fitness: if f(left) < f(right) the
            # minimum is in [left, candidate], else in [candidate, right]. This
            # is the standard golden-section/Brent retention rule and—unlike
            # the previous "keep the better endpoint + candidate" heuristic—
            # provably preserves a straddling bracket (it never drops the side
            # containing the true minimizer).
            keep_left = lf < rf
            n_la = jnp.where(keep_left, la, cand_alpha)
            n_lf = jnp.where(keep_left, lf, cf)
            n_lm = jnp.where(keep_left, lm, cm)
            n_ra = jnp.where(keep_left, cand_alpha, ra)
            n_rf = jnp.where(keep_left, cf, rf)
            n_rm = jnp.where(keep_left, cm, rm)

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

        # The spline only ever *improves* on the inner result, so the
        # acceptance status is at least as good as the inner search's.
        done = jnp.logical_or(inner.done, fv < inner.new_value)
        return LineSearchResult(
            step_size=fa,
            new_value=fv,
            new_grad=fg,
            new_params=fp,
            done=done,
        )

    return wrapped


__all__ = ["spline_wrap"]
