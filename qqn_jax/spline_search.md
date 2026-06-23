# Spline Search — Cubic Hermite Refinement for QQN Line Searches

## Overview

`spline_search.py` implements a **cubic Hermite spline augmentation** that
wraps any QQN-compatible line search and attempts to improve on its accepted
step by reusing every measured point along the consistent quadratic path as a
control point of a piecewise cubic Hermite spline model of the objective.

The spline is **not** a competing line search. It is an *expanded definition
of the curve*: because every probe in a QQN iteration lies on the same fixed
direction `d` (the quadratic blend path), all measured `(α, f(α), f'(α))`
triples are valid control points for a single consistent spline model. The
refinement probes the stationary points of that model to find a better step
than the inner search's accepted point.

---

## The Quadratic Blend Path

QQN constructs a parametric path in parameter space:

```
d(t) = t(1 − t)·(−∇f) + t²·(−H∇f)
```

where `−∇f` is the steepest-descent direction and `−H∇f` is the
quasi-Newton direction (e.g. from L-BFGS). The line search traverses the
scalar parameter `t ∈ [0, 1]`, evaluating `x + d(t)` at candidate values.

Because the path direction is **consistent** across all measured points —
every probe uses the same `d` — the sequence of `(t, f(t), f'(t))` triples
forms a coherent dataset for a 1-D spline model of the objective along the
path. This is the key property that makes spline refinement correct: it does
not assume a global quadratic model, only that the measured points lie on the
same curve.

---

## Algorithm

### 1. Run the Inner Line Search

`spline_wrap(inner_search)` first runs `inner_search` (any registered
strategy: `"armijo"`, `"backtracking"`, `"strong_wolfe"`, etc.) to obtain a
baseline accepted point `(α₁, f₁, ∇f₁)`.

### 2. Establish a Three-Point Bracket

Three points are always measured before the spline loop begins:

| Point   | Alpha   | Value   | Slope   | Source                      |
|---------|---------|---------|---------|-----------------------------|
| `p0`    | `0`     | `f₀`    | `m₀`    | Current iterate (free)      |
| `p1`    | `α₁`    | `f₁`    | `m₁`    | Inner search accepted point |
| `p_ext` | `α_ext` | `f_ext` | `m_ext` | Expansion probe (one eval)  |

The expansion point `α_ext` is chosen adaptively:

- If the path is **still descending** at the inner point (`m₁ < 0`), the
  minimum lies further along the path, so `α_ext = 2·α₁` (an expansion).
- Otherwise the minimum is already bracketed by `[0, α₁]`, so
  `α_ext = 0.5·α₁` (a contraction, providing a tighter bracket).
- If the inner search returned a zero-length step, `α_ext = 1.0` (the
  full path endpoint) is used as a safe fallback.

### 3. Select the Initial Bracket

The two-segment bracket `[la, ra]` is seeded from the three measured points
by choosing the segment most likely to contain the minimum:

- If the inner point is still descending **and** the extension overshot
  (`f_ext < f₁`), use `[α₁, α_ext]`.
- Otherwise use `[0, α₁]`.

The best-so-far point `(ba, bv, bp, bg)` is initialised to the minimum of
all three measured values.

### 4. Spline Refinement Loop (`lax.while_loop`)

Each iteration of the loop:

1. **Locate stationary points** of the cubic Hermite segment over `[la, ra]`
   by solving the quadratic derivative equation in closed form
   (see [Cubic Hermite Derivative](#cubic-hermite-derivative) below).
2. **Select the best candidate** (lowest predicted value among valid roots).
   Fall back to the bracket midpoint `0.5·(la + ra)` if no valid root exists.
3. **Clip** the candidate to `[la + ε, ra − ε]` (margin `ε = 1e-3·span`) to
   prevent degenerate bracket collapse.
4. **Evaluate** `f(α_cand)` and `∇f(α_cand)` via `value_and_grad_fn`,
   projecting through the active `Region`.
5. **Update best-so-far** if the new value improves on `bv`.
6. **Narrow the bracket** using the slope sign at the candidate:
    - `m_cand < 0` → minimum is to the right → new bracket `[α_cand, ra]`.
    - `m_cand ≥ 0` → minimum is to the left → new bracket `[la, α_cand]`.

The loop runs for at most `spline_max_iter` iterations (default `6`).

### 5. Return the Better Result

The spline result is returned if `fv < f_inner`; otherwise the inner result
is returned unchanged. The `done` flag is set if either the inner search
converged or the spline improved on it.

---

## Cubic Hermite Derivative

Given a segment `[t₀, t₁]` with endpoint values `(f₀, f₁)` and endpoint
slopes `(m₀, m₁)` (true directional derivatives `⟨∇f, d'(t)⟩`), the cubic
Hermite interpolant in normalised coordinates `s = (t − t₀) / h`,
`h = t₁ − t₀`, is:

```
p(s) = h₀₀(s)·f₀ + h₁₀(s)·h·m₀ + h₀₁(s)·f₁ + h₁₁(s)·h·m₁
```

where the basis polynomials are:

```
h₀₀(s) =  2s³ − 3s² + 1
h₁₀(s) =   s³ − 2s² + s
h₀₁(s) = −2s³ + 3s²
h₁₁(s) =   s³ −  s²
```

Differentiating with respect to `s`:

```
p'(s) = (6s² − 6s)·f₀  + (3s² − 4s + 1)·h·m₀
      + (−6s² + 6s)·f₁ + (3s² − 2s)·h·m₁
```

Setting `p'(s) = 0` gives the quadratic `A·s² + B·s + C = 0` with:

```
A =  6·f₀ + 3·h·m₀ − 6·f₁ + 3·h·m₁
B = −6·f₀ − 4·h·m₀ + 6·f₁ − 2·h·m₁
C =  h·m₀
```

Roots are found in closed form via the quadratic formula. When `|A| < ε`
(near-linear derivative), the linear fallback `s = −C / B` is used. Roots
outside `[0, 1]` or with negative discriminant are masked as invalid.

---

## Tangent Orientation (`_orient_tangents`)

The endpoint slopes `m₀` and `m₁` passed to `_segment_stationary_candidates`
are **true measured directional derivatives** `⟨∇f, d'(t)⟩`. These are used
directly without re-orientation in the refinement loop, because re-orienting
real curvature information would corrupt the spline model.

The `_orient_tangents` helper is provided for use with **synthetic tangents**
(e.g. finite-difference approximations or secant estimates) where the sign
convention may be ambiguous. It reflects any tangent with a negative dot
product against the path's forward direction so that all oriented tangents
agree with the natural flow of the curve:

```python
def reflect(m):
    return jnp.where(m < 0.0, -m, m)
```

> **Note:** `_orient_tangents` is *not* called inside `spline_wrap` on
> measured slopes. It is exposed as a public helper for callers that construct
> synthetic control points.

---

## Evaluation Counting

`spline_wrap` maintains an honest count of every `value_and_grad_fn` call:

| Source                     | Count              |
|----------------------------|--------------------|
| Inner line search          | `inner.num_evals`  |
| Expansion probe (`α_ext`)  | `+1`               |
| Each spline body iteration | `+1` per iteration |

The total is returned in `LineSearchResult.num_evals` so that benchmarks
comparing *work done* (not just iteration counts) remain accurate.

---

## Region / Projection Support

Every candidate point is projected through the active `Region` before
evaluation:

```python
def project(candidate):
    return region.project(params, candidate, region_state)
```

This ensures that spline probes respect box constraints, trust-region radii,
orthant restrictions, and any other projective region registered with the
solver. The `region` and `region_state` arguments are forwarded from the
outer `QQN.update` call through `spline_wrap`'s keyword arguments.

---

## Probe Forwarding

The spline wrapper forwards the inner search's `probe_params`, `probe_grads`,
`probe_valid`, `probe_values`, and `probe_alphas` fields unchanged in the
returned `LineSearchResult`. This ensures that:

- The oracle's curvature memory (`feed_probes_to_oracle=True`) receives the
  inner search's probes as usual.
- The descent gate (`probe_descent_gate=True`) can filter probes by value
  without recomputing them.
- The spline's own body-iteration probes are **not** separately threaded
  through the fixed-size probe buffer; the inner probes already cover the
  path, and the accepted point is appended by the oracle update.

---

## Public API

### `spline_wrap(inner_search) -> Callable`

Wraps any line-search callable and returns a new callable with the same
signature, augmented with cubic Hermite spline refinement.

**Signature of the returned callable:**

```python
wrapped(
    value_and_grad_fn,  # Callable: params -> (value, grad)
    params,  # Current parameter pytree
    direction,  # Search direction pytree (d(t) blend)
    value,  # f(params)
    grad,  # ∇f(params)
    *args,  # Extra arguments forwarded to value_and_grad_fn
    spline_max_iter=6,  # Maximum spline refinement iterations
    region=None,  # Region instance or None
    region_state=None,  # Region state or None
    **inner_kwargs,  # Forwarded to inner_search
) -> LineSearchResult
```

### `spline_search`

A ready-to-use instance: `spline_wrap(backtracking_search)`. This is the
callable referenced by `line_search="spline"` in the `QQN` constructor.

```python
from qqn_jax import spline_search
from qqn_jax.spline_search import spline_wrap
```

---

## Internal Helpers

### `_segment_value(s, h, f0, m0, f1, m1) -> scalar`

Evaluates the cubic Hermite interpolant at normalised parameter `s ∈ [0, 1]`.
Satisfies `_segment_value(0, ...) == f0` and `_segment_value(1, ...) == f1`
exactly.

### `_segment_stationary_candidates(t0, t1, f0, m0, f1, m1)`

Returns `(t_cands, val_cands, valid)` — arrays of length 2 containing the
up-to-two stationary points of the cubic over `[t0, t1]`, their predicted
values, and a boolean validity mask. Invalid candidates receive `+inf` values
so `jnp.argmin` never selects them.

### `_orient_tangents(h, f0, m0, f1, m1) -> (m0_oriented, m1_oriented)`

Reflects tangents with negative dot products against the path's forward
direction. Intended for synthetic tangents only; do not apply to measured
directional derivatives.

---

## Usage Examples

### As a line search name

```python
from qqn_jax import QQN

solver = QQN(fun, line_search="spline")
x_opt, state = solver.run(x0)
```

### Equivalent explicit form

```python
QQN(fun, line_search="backtracking", spline=True)
```

### Wrapping a custom inner search

```python
from qqn_jax.spline_search import spline_wrap
from qqn_jax.line_search import armijo_search

spline_armijo = spline_wrap(armijo_search)
```

### Direct use of `spline_search`

```python
from qqn_jax import spline_search
from qqn_jax.utils import make_value_and_grad

vg = make_value_and_grad(fun)
value, grad = vg(params)
direction = -grad

result = spline_search(vg, params, direction, value, grad, init_step=1.0)
print(result.step_size, result.new_value)
```

---

## Composition with Other Features

| Feature                     | Compatibility                                      |
|-----------------------------|----------------------------------------------------|
| `jit`                       | ✓ Full — uses `lax.while_loop` internally          |
| `vmap`                      | ✓ Full — no Python-level branching on array values |
| `pmap`                      | ✓ Full                                             |
| `grad` (through the solver) | ✓ Full                                             |
| All `Region` types          | ✓ Projects every probe through `region.project`    |
| `feed_probes_to_oracle`     | ✓ Inner probes forwarded; spline probes excluded   |
| `has_aux`                   | ✓ Transparent — aux is carried by the solver layer |
| Any inner line search       | ✓ `spline_wrap` is agnostic to the inner strategy  |

---

## Design Notes

- **Consistency is the key invariant.** The spline is valid because every
  probe lies on the same fixed direction `d`. If the direction changed between
  probes (as in a multi-step method), the control points would be
  incoherent and the spline model would be meaningless.

- **The inner search is not replaced.** The spline only *improves* on the
  inner result; it never returns a worse point. If the spline finds no
  improvement, the inner result is returned unchanged.

- **Bracket narrowing uses true slopes.** The bisection rule (`m_cand < 0`
  → bracket right half) is exact because `m_cand` is a true measured
  directional derivative, not a finite-difference approximation.

- **NaN safety.** The discriminant is clamped to `≥ 0` before `sqrt`, and
  all divisions are guarded against zero denominators. Invalid candidates
  receive `+inf` values so they are never selected by `argmin`.

- **No pytree stacking.** The best-so-far `params` and `grad` pytrees are
  tracked through the `while_loop` carry without materialising a stack of
  pytrees. The three-way selection at initialisation uses `lax.switch` to
  avoid Python-level branching on traced values.

- **Eval count is honest.** Every `value_and_grad_fn` call — including the
  expansion probe and each spline body iteration — is counted and returned
  in `num_evals`. This ensures that benchmarks comparing *work done* across
  optimizers remain fair.