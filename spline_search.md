# Spline Search: An Improvement on the Quadratic Path

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

## Reusing Anchor Points

Future measurements can exploit the curve defined by the existing **bracketing
anchor points**. As the search accumulates control points, the interval between any
two bracketing anchors is described by a locally accurate spline segment, improving
the quality of subsequent step proposals.

## Search Workflow

In the general case:

1. **Test the oracle's prediction** — evaluate the point proposed by the oracle
(e.g., the L-BFGS / quasi-Newton step).
2. **Map the interval** — the region from the current point to the oracle point is
then better characterized using the accumulated fitness/gradient control points.
3. **Backtrack efficiently** — if the oracle point is unsatisfactory, the spline
model of the bracketed interval guides backtracking with far fewer additional
evaluations than a naive line search.

## Benefits

- **Information reuse**: Gradient data, normally discarded, refines the path model.
- **Fewer evaluations**: A richer model means better step proposals and cheaper
backtracking.
- **Smooth, accurate interpolation**: Hermite spline segments respect both value and
slope, avoiding the systematic error of a single global quadratic.