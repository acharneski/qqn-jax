# Spline Search: An Improvement on the Quadratic Path

[demo](spline_demo.html)

## Motivation

In the QQN line search, every function evaluation along the quadratic path `d(t)`
yields more than a single scalar fitness value — it provides both a **fitness
measurement** and a **gradient measurement** at the evaluated point. The standard
quadratic-path search discards much of this information after each step.

Spline search treats each measurement as a reusable **control point**, allowing the
search to build an increasingly accurate model of the objective along the path.

## Core Idea

Each measurement during the path search contributes two pieces of information:

1. **Fitness** — the function value `f(d(t))` at the measured point.
2. **Gradient** — the directional derivative, which constrains the *tangent* of the
   path model at that point.

We use these to define a spline through the search space:

- The curve is **forced to intersect** the measured point (fitness constraint).
- The curve's **tangent is corrected** to match the measured gradient (slope
  constraint).

This produces a Hermite-style interpolation in which both position and slope are
honored at each control point, rather than a single global quadratic fit.

## Implementation: Cubic Hermite Spline

The interpolation is implemented as a **piecewise cubic Hermite spline** over the
one-dimensional parameter `t`. Each control point `i` stores a tuple:

```
(t_i, f_i, m_i)
```

where:

- `t_i` is the path parameter (position along `d(t)`),
- `f_i = f(d(t_i))` is the measured fitness,
- `m_i = d/dt f(d(t))` is the measured directional derivative (the slope of the
  objective along the path at `t_i`).

Note that `m_i` is the *scalar* projection of the full gradient `∇f` onto the path
tangent `d'(t_i)`:

```
m_i = ⟨ ∇f(d(t_i)), d'(t_i) ⟩
```

### Segment Evaluation

For two adjacent control points `(t_0, f_0, m_0)` and `(t_1, f_1, m_1)`, define the
local normalized parameter and segment width:

```
h = t_1 - t_0
s = (t - t_0) / h        for t ∈ [t_0, t_1], s ∈ [0, 1]
```

The cubic Hermite basis functions are:

```
h00(s) =  2s³ - 3s² + 1
h10(s) =      s³ - 2s² + s
h01(s) = -2s³ + 3s²
h11(s) =      s³ -  s²
```

The interpolated fitness within the segment is:

```
f(s) = h00(s)·f_0 + h10(s)·h·m_0 + h01(s)·f_1 + h11(s)·h·m_1
```

The factor `h` scales the tangents because the slopes `m_i` are expressed in the
original `t` units, while the basis functions operate on the normalized `s`.

### Locating Candidate Steps

To propose the next step, differentiate the segment with respect to `t` and solve
for stationary points (`f'(s) = 0`). Since `f(s)` is cubic, `f'(s)` is quadratic,
yielding at most two roots in closed form:

```
f'(s) = (6s² - 6s)·f_0 + (3s² - 4s + 1)·h·m_0
    + (-6s² + 6s)·f_1 + (3s² - 2s)·h·m_1
```

Any real root with `s ∈ [0, 1]` is mapped back via `t = t_0 + s·h` and becomes a
candidate minimizer. The candidate with the lowest predicted fitness across all
bracketed segments is selected as the next evaluation point.

## Gradient Orientation: Upstream/Downstream Symmetry

### The Terrain Analogy

We are effectively **mapping terrain** along the path. The spline acts as a
watercourse model: it is naturally attracted to the lines of steepest descent — the
paths of natural water flow. However, the *direction* of flow is not intrinsic to
the terrain itself; a valley is the same valley whether you traverse it upstream or
downstream.

We therefore treat **upstream and downstream as one symmetric feature**. What
matters is the *shape* of the channel, not the arbitrary sign convention of the
parameterization.

### The Degeneracy Problem

A naive Hermite construction inserts the measured tangent `m_i` with its raw sign.
If a control point's gradient "goes against" the spline — i.e., its tangent points
in a direction inconsistent with the local trend established by the neighboring
bracketing anchors — the resulting cubic segment can develop a spurious inflection,
overshoot, or even a non-monotone loop. This produces:

- **Phantom minima** that do not correspond to real structure,
- **Oscillating step proposals** that waste evaluations,
- **Numerically ill-conditioned** segments near sign reversals.

### The Correction Rule

Before incorporating a control point's tangent into a segment, compare its
orientation to the **secant** of the segment it participates in. For a segment
spanning `(t_0, f_0)` and `(t_1, f_1)`, define the secant slope:

```
Δ = (f_1 - f_0) / (t_1 - t_0)
```

For each endpoint tangent `m`, if it is oriented *against* the established flow of
the segment, reflect it so that upstream and downstream are mapped consistently:

```
if sign(m) ≠ sign(Δ) and Δ ≠ 0:
  m ← -m        # reverse to align with the channel's natural flow
```

Equivalently, we enforce that the tangent's component along the secant direction is
non-negative. This reflection is the explicit statement that **the channel is
symmetric**: reversing a tangent that points "uphill against the current" recovers
the same terrain feature traversed in the opposite sense, without introducing a
spurious feature into the model.

> **Note**: When `Δ = 0` (a flat secant, e.g., near a bracketed minimum), no
> reflection is applied; the raw tangents define the local curvature directly and
> are essential for locating the minimum precisely.
> **Open soundness caveat.** Reflecting a *measured* directional derivative is a
> heuristic, not a proven-safe operation: it discards the sign information of a
> true gradient measurement and could, in principle, introduce a spurious
> stationary point rather than remove one. No formal monotonicity- or
> descent-preserving guarantee is given here. Safety in practice rests entirely
> on the outer line search: the spline only ever *proposes* candidates, and a
> candidate is accepted **only if it strictly improves fitness** (see
> `spline_wrap`'s `improves = cf < bv` gate). The descent guarantee of QQN is
> therefore inherited from the inner search's sufficient-decrease test, not from
> the reflection rule. Treat the reflection as an unproven refinement pending a
> rigorous justification.

### Why This Preserves Information

Reflection does not discard the gradient measurement — it preserves its *magnitude*
(the steepness of the terrain) while normalizing its *sign* to the path's local
orientation. The full directional derivative still constrains the curvature of the
segment; we have only removed the arbitrary orientation that would otherwise corrupt
the symmetric terrain model.

## Reusing Anchor Points

Future measurements can exploit the curve defined by the existing **bracketing
anchor points**. As the search accumulates control points, the interval between any
two bracketing anchors is described by a locally accurate spline segment, improving
the quality of subsequent step proposals.

Concretely, control points are kept sorted by `t`, and the active bracket is the
pair of adjacent anchors `(t_lo, t_hi)` that straddle the best-known region. Each
new evaluation either tightens this bracket or extends the model, and the cubic
Hermite segment for the active bracket is re-evaluated to propose the next step.

## Search Workflow

In the general case:

1. **Test the oracle's prediction** — evaluate the point proposed by the oracle
   (e.g., the L-BFGS / quasi-Newton step).
2. **Map the interval** — the region from the current point to the oracle point is
   then better characterized using the accumulated fitness/gradient control points,
   with tangents oriented via the upstream/downstream symmetry rule.
3. **Backtrack efficiently** — if the oracle point is unsatisfactory, the spline
   model of the bracketed interval guides backtracking with far fewer additional
   evaluations than a naive line search. Candidate steps are drawn from the cubic
   stationary points described above.

## Benefits

- **Information reuse**: Gradient data, normally discarded, refines the path model.
- **Fewer evaluations**: A richer model means better step proposals and cheaper
  backtracking.
- **Smooth, accurate interpolation**: Hermite spline segments respect both value and
  slope, avoiding the systematic error of a single global quadratic.
- **Robust orientation handling**: The symmetric upstream/downstream treatment
  prevents degenerate segments from misleading gradients, keeping the terrain model
  faithful to the underlying channel structure.