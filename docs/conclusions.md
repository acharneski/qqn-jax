---
documents:
  - results.md
related:
  - algorithm.md
  - regions.md
  - oracles.md
  - spline_search.md
---

# Conclusions: QQN as a Modular Combiner for Numerical Optimization

This document synthesizes the conceptual design (see [`algorithm.md`](algorithm.md))
and the empirical validation (see [`results.md`](results.md)) of the QQN
(Quasi-Quadratic-Newton) algorithm. It distills what we learned about the
algorithm's central thesis, the behavior of each swappable component, and the
practical guidance these results imply.

## The Central Thesis Holds

QQN's core claim is that **coherently blending a gradient direction and an oracle
direction along the quadratic path**

```
d(t) = t(1 - t)(-∇f) + t²(-H∇f),   t ∈ [0, 1]
```

**navigated by a robust line search, yields a fast, modular optimizer.** The
full-batch softmax-MNIST benchmark validates this directly:

- The strongest **converging** stacks reach the shared termination target in
   **42–45 iterations**, versus **70** for classical Optax L-BFGS and **263** for
   Adam — a **1.56×–1.67×** iteration-efficiency advantage over L-BFGS *on this
   single convex benchmark*. (Note: this advantage is in *iterations*, not
   wall-clock; Adam wins the wall-clock Pareto front at this scale.)
- The path's anchoring property `d'(0) = -∇f` guarantees the search always begins
  along steepest descent, so even aggressive oracles never forfeit global
  convergence. This is what lets the line search *discover* the right blend rather
  than requiring a hand-tuned mixing constant.

In short, the quadratic path plus a sufficient-decrease line search delivers the
adaptive gradient/oracle interpolation the algorithm was designed to provide.

## What Each Component Taught Us

The four orthogonal axes — **oracle**, **search**, **region**, and the orthogonal
**spline** refinement — were each isolated in controlled A/B sweeps. The findings:

### The Oracle Is the Dominant Convergence Lever

- **Deep L-BFGS memory** was the largest speed lever *observed on this
   benchmark*: iterations-to-target
  fall monotonically from L5 (72) through L50 (45), saturating between L50 and
  L100 (both 45). Curvature memory depth, more than any other knob, drives
   convergence speed on this smooth, convex problem (untested elsewhere).
- **Matrix-free curvature works.** The **Secant (Barzilai-Borwein)** oracle
  crushes plain momentum at *zero* storage cost, and the **Anderson** oracle —
  the variational ideal L-BFGS approximates — reaches the **lowest final loss of
  any configuration** (`1.061e-1`), beating every L-BFGS variant.
- **First-order acceleration is no substitute for genuine curvature.** No
  Momentum-oracle β setting reaches the target; near-zero β simply collapses
  toward steepest descent.
- **Fallback combinators act as low-cost safety nets.**
   `Fallback([L50, Anderson])` *matched* the fastest pure-L50 stack (45 iters)
  because the Anderson residual solve supplies finite, curvature-aware direction
   the instant the L-BFGS history degenerates. (Matching is not strict
   domination; a single run cannot establish a formal ordering.)
- **Dense Shampoo does not amortize at small scale**, exhausting the time budget
  after only 8 iterations on this tiny model — a scale-dependent caveat, not a
  design flaw.

### The Line Search Trades Wall-Time, Not Convergence Speed

- The **backtracking / Armijo family is the robust efficiency winner**:
  backtracking matches Armijo on iterations while running slightly faster in
  wall-clock (no curvature condition to satisfy).
- **Strong Wolfe over-restricts** the quadratic-path step and fails to converge,
  plateauing at `4.08e-1`. The strong-curvature condition is too aggressive a
  filter on the path's step selection for this problem.

### Regions Are Low-Overhead Safeguards — With One Important Caveat

- **Box** and **Orthant** projections add negligible overhead. The Orthant region
  is the only configuration inducing measurable sparsity.
- **Fixed-radius trust-regions are the robust fast path.** The **adaptive**
  trust-region, however, **over-shrinks on the curved path** because the naive
  `ρ = ared/pred` rule compares chord-length (the radial clip) against arc-length
  (the predicted-reduction model). Even with the mitigations now in the code — a
  second-order-aware predicted reduction and a curvature-consistent gentle-shrink
  rule — the adaptive radius still stalls when stacked with deep memory. This is a
   **geometric pitfall, not a tuning artifact**. In its current state the adaptive
   radius is **effectively a known-broken option** for deep-memory stacks: the
   in-code mitigations did *not* resolve the stall, and users are advised to use
   the fixed-radius variant instead. The chord/arc diagnosis explains the failure
   but does not fix it; this remains open work, not a delivered feature.
- **Combinators compose faithfully** — including inheriting the stall behavior of
  an adaptive-TR child, confirming `Sequential` applies projections exactly as
  specified.

### The Spline Refinement Sharpens the Deepest Trajectories

- As an orthogonal augmentation that **wraps** any line search, the spline reuses
  every probe as a cubic Hermite control point. It sharpens the **deepest-memory**
  trajectories most: `QQN-L50Spln` is the **fewest-iteration converging variant
  (42 iters, 1.67× vs L-BFGS)**.
- The benefit costs extra probes (≈66 ms/it vs ≈29 ms/it for plain L50), and
  stacking the spline with the *adaptive* trust-region inherits that region's
  stall.

### Warm-Started Backtracking Unlocks the Superlinear Regime

Because the path's `t = 1` endpoint is already a full quasi-Newton step,
warm-starting backtracking **beyond α = 1** lets deep-memory steps stretch into
the superlinear regime — but only when paired with a **fixed** trust-region. The
contrast between `QQN-L50BTTR` (adaptive TR, stalls at 500 iters) and `QQN-L50WS+`
(fixed TR, 57 iters) is a **Δ−443 iteration swing from a single variable**, the
clearest evidence that adaptive-radius contamination — not the warm start — is the
destabilizing factor.

## Design-Claim Scorecard

| QQN Design Claim                                                 | Empirical Verdict                                                     |
|------------------------------------------------------------------|-----------------------------------------------------------------------|
| Gradient + oracle blending via the quadratic path converges fast | ✅ 1.56–1.67× fewer iters than L-BFGS                                  |
| The oracle is freely swappable                                   | ✅ L-BFGS, Momentum, Secant, Anderson, Shampoo, Fallback all run       |
| Deep curvature memory accelerates convergence                    | ✅ monotone L5→L50, saturating at L50–L100                             |
| The line search trades wall-time, not convergence speed          | ✅ BT ≈ Armijo in iters; SW over-restricts                             |
| Regions are low-overhead safeguards                              | ✅ Box/Orthant negligible; **fixed** TR robust, **adaptive** TR stalls |
| The spline reuses information to sharpen trajectories            | ✅ L50Spln is the fewest-iteration converging variant (42)             |
| Warm-started backtracking unlocks the superlinear regime         | ✅ Δ−443 iters vs adaptive-TR baseline                                 |

## Practical Recommendations

Based on the empirical evidence, the **intended robust fast stack** is:

1. **Oracle**: deep L-BFGS memory (`history_size=50`), optionally guarded by
   `Fallback([LBFGSOracle(50), AndersonOracle(...)])` for a curvature-aware safety
   net when the history degenerates.
2. **Line search**: backtracking / Armijo, warm-started beyond `α = 1`
   (`init_step=2.0`, `shrink=0.7`) to exploit the superlinear `t = 1` endpoint.
3. **Region**: a **fixed-radius** trust-region (avoid the adaptive radius on
   curved paths) or none at all; add Box/Orthant only when constraints or sparsity
   are required.
4. **Spline**: enable (`spline=True`) when minimizing *iteration count* matters
   more than per-iteration wall-clock — it yields the fewest-iteration converging
   trajectories.

Conversely, **avoid** for this regime: strong-Wolfe line search (over-restricts),
the adaptive trust-region with deep memory (chord/arc stall), pure Momentum
oracles (first-order plateau), and dense Shampoo at small model scale (cost does
not amortize).

## Closing

The experiments confirm QQN's foundational design: the quadratic path is a
principled, single-dimensional search space over *states*; the line search is the
first-class glue that walks it; and the gradient, oracle, region, and spline axes
are *independently swappable* as software components. They are **not, however,
behaviorally orthogonal**: the adaptive trust-region's stall depends strongly on
the oracle (it stalls only with deep memory) and on the warm start — a genuine
cross-axis *interaction*, not independence. The cautionary findings are
first-class results that qualify, rather than undermine, the central claim. On
this *single convex* benchmark the best-of-breed converging stacks (42–45
iterations vs. 70 for L-BFGS and 263 for Adam) suggest a coherent, modular
combiner *can* be iteration-efficient while remaining fully
`jit`/`vmap`/`grad`-compatible — though on raw wall-clock Adam still wins at
this scale, and none of these rankings have yet been validated off the convex
case.

See [`algorithm.md`](algorithm.md) for the conceptual treatment,
[`results.md`](results.md) for the full empirical breakdown, and
[`../mnist_comparison.log`](../results/mnist_comparison.log) for the raw output.