"""Projective regions for QQN.

A *projective region* remaps a proposed parameter update onto a feasible
(or otherwise preferred) set before it is applied. Because QQN searches a
single continuous quadratic path ``d(t)``, regions integrate cleanly: the
line search navigates the *projected* path

    d_R(t) = project_R(x, x + d(t)) - x

All regions are pure, functional JAX so they compose with ``jit``,
``vmap``, ``pmap`` and ``grad``. When the region is the identity
(``IdentityRegion`` / ``region=None``), behavior is byte-for-byte
equivalent to the un-regioned optimizer.
"""

from typing import Any, Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp

from qqn_jax.utils import tree_l2_norm


class Region(NamedTuple):
    """Pure, composable projection interface.

    Attributes:
        init: ``params -> region_state`` (use ``()`` when stateless).
        project: ``(params, candidate, state) -> projected_candidate``.
        update: ``(state, info) -> state`` (no-op for stateless regions).
    """

    init: Callable[[Any], Any]
    project: Callable[[Any, Any, Any], Any]
    update: Callable[[Any, Any], Any]


class RegionInfo(NamedTuple):
    """Information passed to ``Region.update`` after a step.

    Attributes:
        params: iterate ``x`` before the step.
        new_params: accepted iterate ``x + α·d_R(t)``.
        pred_reduction: predicted reduction from the along-path model.
        actual_reduction: actual reduction ``f(x) - f(x_new)``.
        t: chosen interpolation parameter.
        step_size: accepted step size ``α``.
    """

    params: Any = None
    new_params: Any = None
    pred_reduction: Any = None
    actual_reduction: Any = None
    t: Any = None
    step_size: Any = None


# --- Tree helpers -----------------------------------------------------


def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_sub(a, b):
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


# --- Identity (default, zero-overhead) --------------------------------


def _identity_init(params):
    return ()


def _identity_project(params, candidate, state):
    return candidate


def _identity_update(state, info):
    return state


def IdentityRegion() -> Region:
    """The trivial region: projection is the identity (no constraints)."""
    return Region(
        init=_identity_init,
        project=_identity_project,
        update=_identity_update,
    )


# --- Box / Min-Max Region ---------------------------------------------


def BoxRegion(lo=None, hi=None) -> Region:
    """Enforce elementwise bounds ``lo ≤ x_new ≤ hi``.

    ``lo``/``hi`` may be scalars, pytrees broadcastable to the parameter
    structure, or ``None`` (mapped to ∓inf).
    """
    lo_val = -jnp.inf if lo is None else lo
    hi_val = jnp.inf if hi is None else hi

    def project(params, candidate, state):
        return jax.tree_util.tree_map(lambda c: jnp.clip(c, lo_val, hi_val), candidate)

    return Region(
        init=_identity_init,
        project=project,
        update=_identity_update,
    )


# --- Orthant Region (OWL-QN style sparsity) ---------------------------


def OrthantRegion(l1: float = 0.0) -> Region:
    """Constrain each step to remain within the orthant of the current
    point's signs, zeroing coordinates that would cross zero.

    When ``l1 > 0`` the pseudo-gradient ``∇f + l1·sign(x)`` chooses the
    orthant for zero coordinates (OWL-QN convention). The pseudo-gradient
    is approximated using ``candidate - params`` as a step proxy, which is
    the direction the line search proposes.
    """

    def project(params, candidate, state):
        def proj_leaf(x, c):
            step = c - x
            # Chosen orthant sign ξ. For nonzero x, the orthant is x's own
            # sign. For genuinely-zero coordinates, OWL-QN selects the orthant
            # from the *pseudo-gradient* −(∇f + l1·sign(x)); with x=0 and the
            # step as our descent proxy, the l1 term biases toward the axis,
            # so a coordinate only leaves zero when |step| exceeds the l1 pull.
            l1c = jnp.asarray(l1, dtype=c.dtype)
            # Effective forcing magnitude after the l1 soft-threshold.
            forced = jnp.abs(step) > l1c
            xi = jnp.where(
                x != 0.0,
                jnp.sign(x),
                jnp.where(forced, jnp.sign(step), 0.0),
            )
            keep = jnp.sign(c) == xi
            return jnp.where(keep, c, 0.0)

        return jax.tree_util.tree_map(proj_leaf, params, candidate)

    return Region(
        init=_identity_init,
        project=project,
        update=_identity_update,
    )


# --- Trust-Region Sphere ----------------------------------------------


class TrustRegionState(NamedTuple):
    radius: jnp.ndarray


def TrustRegion(
    radius: float = 1.0,
    radius_max: float = 1e3,
    adaptive: bool = True,
    shrink: float = 0.5,
    expand: float = 2.0,
    eta_lo: float = 0.1,
    eta_hi: float = 0.75,
) -> Region:
    """Enforce ``‖x_new − x‖₂ ≤ Δ`` by radially clipping the step.

    With ``adaptive=True`` the radius grows/shrinks according to the ratio
     ``ρ = ared / pred`` of actual to predicted reduction.
     Esoteric note (from the Andromeda gradient-clusters): on a *curved*
     path the chord-length the radius constrains and the arc-length the
     predicted-reduction model integrates are different coordinates. We
     therefore (a) only shrink on a genuinely poor ``ρ`` (``< eta_lo``),
     (b) shrink *gently* (``shrink``, default 0.5 not 0.25), and (c) hold
     the radius in the wide acceptable band ``[eta_lo, eta_hi]`` so the
     adaptive feedback does not over-react to the chord/arc mismatch that
     stalls the naive ``ρ < 0.25`` rule.
    """
    eps = 1e-12

    def init(params):
        return TrustRegionState(radius=jnp.asarray(radius, dtype=jnp.float32))

    def project(params, candidate, state):
        step = _tree_sub(candidate, params)
        n = tree_l2_norm(step)
        scale = jnp.minimum(1.0, state.radius / (n + eps))
        return jax.tree_util.tree_map(lambda x, s: x + scale * s, params, step)

    def update(state, info):
        if not adaptive:
            return state
        pred = info.pred_reduction
        ared = info.actual_reduction
        rho = ared / (pred + eps)
        step = _tree_sub(info.new_params, info.params)
        n = tree_l2_norm(step)
        at_boundary = n >= state.radius - 1e-6
        # Only contract on a genuinely poor agreement, and contract gently.
        # The wide central band [eta_lo, eta_hi] is the *stable attractor*:
        # the radius is held constant there, immune to the chord/arc-length
        # disagreement that drives the naive rule to collapse.
        new_radius = jnp.where(
            rho < eta_lo,
            shrink * state.radius,
            jnp.where(
                jnp.logical_and(rho > eta_hi, at_boundary),
                jnp.minimum(expand * state.radius, radius_max),
                state.radius,
            ),
        )
        # A radius can never usefully fall below the machine-floor of the
        # step it is meant to bound; clamp it away from collapse.
        new_radius = jnp.maximum(new_radius, eps)
        return TrustRegionState(radius=new_radius)

    return Region(init=init, project=project, update=update)


# --- Combinator: Sequential -------------------------------------------
# --- No-Decrease Region (multi-objective guard) -----------------------
def NoDecreaseRegion(secondary_grad_fn: Callable) -> Region:
    """Project each step onto the half-space ``{s : ⟨∇g, s⟩ ≤ 0}``.
    Given a secondary objective ``g`` whose gradient is supplied by
    ``secondary_grad_fn(params) -> ∇g``, this region removes only the
    component of the proposed step that would *increase* ``g`` — preserving
    fitness on a protected objective while optimizing the primary one. This
    is the geometry of continual learning and constrained fine-tuning: the
    step is free to move in any direction that does not climb ``g``.
    The projection is the orthogonal removal of the offending component::
        step = candidate - x
        c    = ⟨∇g, step⟩
        s_proj = step - relu(c) / (‖∇g‖² + eps) · ∇g
    Only the *positive* (g-increasing) component is removed; descent on ``g``
    is permitted to pass through untouched.
    """
    eps = 1e-12

    def project(params, candidate, state):
        g = secondary_grad_fn(params)
        step = _tree_sub(candidate, params)
        c = sum(
            jnp.vdot(gi, si)
            for gi, si in zip(
                jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(step)
            )
        )
        gg = sum(jnp.vdot(gi, gi) for gi in jax.tree_util.tree_leaves(g))
        # Remove only the g-increasing component (relu(c) gates the sign).
        coeff = jnp.maximum(c, 0.0) / (gg + eps)
        s_proj = jax.tree_util.tree_map(lambda si, gi: si - coeff * gi, step, g)
        return _tree_add(params, s_proj)

    return Region(
        init=_identity_init,
        project=project,
        update=_identity_update,
    )


def Sequential(regions: Sequence[Region]) -> Region:
    """Compose regions by applying their projections in order.

    ``project = R_k ∘ ... ∘ R_1``. State is a tuple of child states and
    ``update`` fans out to each child.
    """
    regions = tuple(regions)

    def init(params):
        return tuple(r.init(params) for r in regions)

    def project(params, candidate, state):
        c = candidate
        for r, s in zip(regions, state):
            c = r.project(params, c, s)
        return c

    def update(state, info):
        return tuple(r.update(s, info) for r, s in zip(regions, state))

    return Region(init=init, project=project, update=update)


def resolve_region(region: Optional[Region]) -> Region:
    """Return ``region`` or the identity region when ``None``."""
    return IdentityRegion() if region is None else region


__all__ = [
    "Region",
    "RegionInfo",
    "RegionState",
    "IdentityRegion",
    "BoxRegion",
    "OrthantRegion",
    "TrustRegion",
    "TrustRegionState",
    "NoDecreaseRegion",
    "Sequential",
    "resolve_region",
]

# Backwards-compat alias used in docstrings/specs.
RegionState = Any
