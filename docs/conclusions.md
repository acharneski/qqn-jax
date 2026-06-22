---
documents:
  - results.md
related:
  - algorithm.md
  - regions.md
  - oracles.md
  - spline_search.md
---

# Conclusions

This document synthesizes the findings from the QQN experimental evaluation
(see [`results.md`](results.md)) and assesses how the empirical evidence
supports the algorithmic claims made in [`algorithm.md`](algorithm.md),
[`oracles.md`](oracles.md), [`regions.md`](regions.md), and
[`spline_search.md`](spline_search.md).

## Summary of Findings

The MNIST optimizer comparison validated QQN as a practical, competitive
optimizer on a smooth, deterministic, full-batch problem. Because every
optimizer raced against the **same shared termination bounds** (`f_target =
1.1e-1`, `gtol = 1.0e-4`, `time_budget = 15.0s`), the headline metric is
*iterations-to-target* — the iteration at which the shared loss/gradient
target was first reached. The headline results are:

- **QQN reaches the shared target in fewer iterations than L-BFGS at a
  fraction of the cost.** On the softmax MNIST benchmark, QQN reached the
  shared `f_target = 1.1e-1` in **65 iterations** (final loss `1.096e-01`) —
  fewer than Optax's L-BFGS (70 iterations) — while running roughly **1.4×
  faster** in wall-clock time (1.546s vs. 2.117s), a **1.08×** iteration
  speedup.
- **QQN clearly beats first-order baselines on convergence speed.** It reached
  the target in ~4× fewer iterations than Adam (which needed 263), and SGD
  **never reached** the target within 500 iterations (plateauing at
  `2.266e-01`), confirming the benefit of quasi-Newton acceleration on smooth
  deterministic objectives.
- **The four-axis modular design behaves as specified.** Each swappable
  component (gradient, oracle, search, region) could be substituted
  independently, and the defaults (`oracle="lbfgs"`, `region=None`) reproduced
  the baseline behavior exactly.
- **Deep L-BFGS memory wins the race at a 1.63× iteration speedup over
  L-BFGS.** The target was deliberately tuned to `1.1e-1` (the previous
  `1.0e-1` was unreachable by every method within budget) so the
  iterations-to-target column became informative. Under it, the best-of-breed
  deep-memory oracles (`QQN-L50`, `QQN-L100`) won the race at **43 iterations**
  versus L-BFGS's 70 — a **1.63× iteration speedup**.

## Validation of Core Algorithmic Claims

### The Combiner Model Holds

The central thesis of [`algorithm.md`](algorithm.md) — that QQN is a
**combiner** of orthogonal, independently swappable components — is borne out
by the controlled A/B sweeps. Swapping the oracle, line search, or region in
isolation produced predictable, decomposable effects on iterations-to-target
and wall-time, with no cross-component coupling that would undermine the
modularity claim.

### Global Convergence via the Steepest-Descent Anchor

Across every oracle (including the deliberately aggressive Shampoo and the
weak high-`beta` momentum oracle), the line search always returned a
decreasing step or rejected it. This is consistent with the theoretical
guarantee that the path property `d'(0) = -∇f` anchors global convergence
regardless of oracle quality, leaving the oracle free to be aggressive.

## Component-Level Conclusions

### Oracle Choice Is the Dominant Lever

- **L-BFGS history depth** is the single most important convergence-speed
  lever, with a monotone reduction in iterations-to-target `L5 > L10 > L20 >
  L50` (80 → 65 → 59 → 43), clear diminishing returns past size 50, and a hard
  plateau at 100 (`L50 == L100` at **43 iterations**, a **1.63× iteration
  speedup over L-BFGS**). The converged final loss is essentially flat across
  depths (every variant hits the shared target), so the lever here is *speed
  of convergence*, not final loss. Very deep histories (L100) buy *no* extra
  speed for their additional cost on this problem.
- **The Anderson-acceleration oracle is the AUC champion.** `QQN-And`
  (`AndersonOracle(window=5)`) solves a tiny `(m × m)` constrained
  least-squares problem over recent gradient *differences* — the variational
  ideal that L-BFGS only approximates — forming the `t = 1` endpoint with no
  Hessian ever built. It reaches the shared target in **269 iterations** (far
  more than any L-BFGS depth) but posts the **best trajectory AUC of any
  method in the study (`-0.84`)**, edging out even Adam (`-0.82`). Its deep,
  optimal residual combination keeps the whole trajectory at very low loss
  even though many cheap steps are needed to cross the tight final target.
- **The matrix-free secant (Barzilai-Borwein) oracle is a strong,
  zero-storage curvature signal.** `QQN-Sec` reaches the shared target (in 311
  iterations — far more than any L-BFGS depth, but far fewer than momentum,
  which never converges) while posting an excellent trajectory AUC (`-0.74`,
  among the best of the QQN variants). Despite carrying no Hessian and no
  history buffers, the single-step secant `α = ⟨s,s⟩/⟨s,y⟩` descends fast and
  deep on average, making it an excellent zero-storage fallback that strictly
  dominates a momentum fallback.
- **Momentum** behaved as a first-order accelerator and **never reached the
  target** within 500 iterations; notably, lighter damping (`beta = 0.01`) —
  which collapses toward steepest descent — converged to a lower loss than
  heavier momentum on this smooth problem (the sweep is monotone in `beta`).
- **Shampoo** did not scale to this high-dimensional softmax problem: even with
  a *blocked* preconditioner (`block_size=64`, `update_freq=25`), its dense
  inverse-root refresh exhausted the 15-second wall-clock budget after only
  9 iterations (~1815 ms/it), landing at a much higher loss than even the
  momentum oracle.

### Line Search Trades Time, Not Convergence Speed — Except Strong-Wolfe

Within the **backtracking/Armijo** family, the line search choice had
negligible effect on the iterations-to-target (or converged loss) but a large
effect on wall-time. Backtracking was the clear efficiency winner (`QQN-BT`,
1.294s, target at iteration 65); plain Armijo (`QQN`, 1.546s) and the spline
refinement (2.638s, target at 64) matched its iterations-to-target at higher
cost. This confirms that the more expensive searches do **not** degrade quality
on a well-conditioned objective where curvature information is easy to exploit.

**The notable exception is strong-Wolfe**, which *fails to converge* on this
problem: `QQN-SW` plateaued at `4.077e-01` after exhausting all 500 iterations.
Its tight curvature condition over-restricts the step along the quadratic path
here, so the strong-Wolfe search is not a safe default for this class of
objective.

### The Spline Refinement Composes, As Designed — but No Longer Rescues the Adaptive Region

Consistent with [`spline_search.md`](spline_search.md), the spline behaves as
an orthogonal enhancement that *wraps* (rather than replaces) the inner search,
reusing every probe as a cubic Hermite control point and performing an
adaptive superlinear extension probe when the downstream tangent still
descends. It did not materially change the iterations-to-target for
shallow-memory variants on the smooth objective, but it **sharpened the
deep-memory trajectory**: `QQN-L50Spln` reached the target in **43 iterations**
(tying the spline-less L50 baseline) while achieving the **lowest loss among
the converging spline variants (`1.091e-01`)** and crossing the tight `1.2e-1`
milestone fastest of all (iteration 37).

> **Reversal from earlier runs:** under the current honest predicted-reduction
> model (`pred ≈ -⟨∇f, d(t)⟩` with a curvature correction and a first-order
> floor), the spline's strict-improvement gating is **no longer sufficient to
> rescue the adaptive trust-region**. The spline-wrapped adaptive-TR stacks
> `QQN-SplnTR` (plateau at `2.039e-01`) and `QQN-L50SplnTR` (plateau at
> `1.653e-01`) now **stall**, exhausting all 500 iterations. The monotone
> gating keeps their trajectories low (yielding deceptively good AUCs around
> `-0.76`) but cannot overcome the over-shrinking adaptive radius. **Prefer a
> fixed radius (or no region) when combining the spline with a trust-region.**

The extra per-probe spline fitting costs roughly **2×** wall-time, which does
not pay off for shallow-memory variants on this smooth objective.

### Regions Are Low-Overhead Safeguards — but the Adaptive Trust-Region Is Fragile

> **Cautionary result:** under the current honest predicted-reduction model
> (`pred = -⟨∇f, d(t)⟩`) combined with the default Armijo (`init_step=1.0`)
> line search, the **adaptive trust-region destabilizes convergence** on this
> problem.

- The **box** and **orthant** regions are cheap, well-behaved safeguards. Box
  bounds weights at negligible cost (target at 65); the **orthant** region is
  the only configuration to induce measurable weight sparsity (`0.0027`),
  exactly as its sign-preserving projection predicts (target at 70).
- The **fixed-radius** trust-region converges cleanly (`QQN-TRfix`, 68
  iterations), confirming radial step-clipping itself is sound.
- The **adaptive** trust-region, however, **stalls and never reaches the
  target** across every radius (`TR025` ends at `1.983e+00`, `TR` at
  `6.272e-01`, `TR2` at `6.256e-01`). Switching the radius from fixed to
  adaptive moves the result from *converged at 68 iterations* to *never
  reaches the target*. The honest predicted-reduction model drives the adaptive
  radius to over-shrink and stall the search.
- This fragility propagates through any stack that relies on a bare adaptive
  trust-region: `Sequential([Box, TrustRegion])`, `QQN-L50TR`, `QQN-L100TR`,
  `QQN-L50BTTR`, `QQN-L50Sec`, and the warm-started variants all inherit the
  same stall, converging to the *same* `6.272e-01` plateau — confirming the
  cause is the region, not the oracle or search.

**Recommendation:** on this class of well-conditioned smooth problem, prefer no
region (or a fixed radius / box / orthant) over an adaptive trust-region. The
spline's monotone gating no longer neutralizes the instability under the
current honest `pred` model.

### The t-Grid Is a Cheap Tuning Knob

Sweeping the t-grid granularity (2, 4, and 8 points) had a negligible effect on
iterations-to-target and converged loss, with only a modest effect on wall-time
(a finer grid runs more line searches per iteration). The coarse 2-point grid
was essentially as good as the default 4-point grid on this smooth problem,
confirming the t-grid is a tuning knob here rather than a convergence driver.

### Combinators Work Correctly

The `Fallback` validity test is now **descent-based**, not merely non-zero: a
finite, non-zero quasi-Newton direction that points *uphill* (`⟨∇f, d⟩ ≥ 0`) is
rejected and the fallback triggers — the gate is misalignment, not just
collapse.

`Fallback([L-BFGS, Momentum])` reproduced the L-BFGS baseline exactly (`QQN-Fall`,
target at iteration 65), and `Fallback([L-BFGS(50), Anderson])` reproduced the
deep-L50 baseline exactly (`QQN-L50And`, target at 43), because the L-BFGS
direction was always a valid descent direction and the secondary oracle never
triggered — the intended behavior. Stacked oracle/region combinators ran
correctly: the deeper L20 oracle let `QQN-L20Box` reach the target in 60
iterations, ahead of the shallow box variant (65). However, combinators that
nest a bare adaptive trust-region (e.g. `QQN-Seq`, and `QQN-L50Sec` =
`Fallback([L-BFGS(50), Secant]) + TR`) **inherit the adaptive-trust-region
stall** (`6.272e-01`, never reaching target): the composition is correct, but
the nested adaptive region carries the destabilizing behavior documented above.

## Best-of-Breed Recommendation

The strongest **converging** stacks here are the **bare deep-memory oracles**,
the **deep-memory + spline** combinations, and the **deep-memory + warm-started
backtracking + fixed-radius trust-region** combinations — *not* the deep-memory
+ adaptive-trust-region stacks (which stall). The fewest iterations to target
(**43**) are reached by the bare deep-memory oracles `QQN-L50` and `QQN-L100`,
the Anderson-fallback `QQN-L50And`, and the deep-memory + spline combos
`QQN-L50Spln` / `QQN-L100Spln`, at the lowest wall-time (~1.04–1.17s for the
region-free variants), a strong Pareto point on iterations vs. time
(**1.63× iteration speedup over L-BFGS**). The **lowest loss observed across
the whole study (`1.090e-01`)** is reached by `QQN-Fast` (L100 + warm-started
backtracking + a *fixed* trust-region) at 57 iterations, which also sits on the
Pareto frontier.

For smooth, deterministic, full-batch problems, the robust default is therefore
**deep L-BFGS memory (L50/L100) + (warm-started) backtracking + a fixed
trust-region (or no region)**, optionally adding the **spline refinement** when
the lowest possible loss is worth ~2× wall-time. **Avoid the bare adaptive
trust-region** on this class of problem: stacks that rely on it (`QQN-L50TR`,
`QQN-L100TR`, `QQN-L50BTTR`, `QQN-L50Sec`, the spline-wrapped `QQN-SplnTR` /
`QQN-L50SplnTR`) all stall at `6.272e-01` (or `≤2e-1`) and never reach the
target, because the over-shrinking adaptive radius interacts badly with the
radial trust-region clip.

## Limitations and Caveats

The conclusions above are drawn from a **single, smooth, deterministic,
full-batch convex benchmark** (softmax MNIST). They should be read with the
following caveats:

- **Smoothness flatters cheap searches.** On non-smooth or ill-conditioned
  objectives, the stronger Wolfe/Hager-Zhang searches and the spline
  refinement may pay off where they did not here — and strong-Wolfe, which
  *fails* on this smooth problem, may be necessary elsewhere.
- **The adaptive trust-region instability is problem-specific.** On
  ill-conditioned or non-convex objectives, the adaptive radius driven by
  `ρ = ared/pred` may be exactly the safeguard that prevents divergence; its
  stall here reflects the honest predicted-reduction model interacting with a
  well-conditioned objective, not a universal defect.
- **Iteration-count vs. AUC can disagree.** The Hessian-free Anderson and
  secant oracles take *many more iterations* to cross the tight final target
  than deep L-BFGS, yet lead the trajectory-AUC leaderboard — read AUC
  alongside iterations-to-target. The stalled spline+adaptive-TR stacks also
  post deceptively good AUCs precisely because their monotone gating keeps the
  trajectory low even though they never converge.
- **Generalization was not the differentiator.** Test accuracy was similar
  across the strong optimizers; these results concern optimization speed and
  final training loss, not generalization (Adam in fact had the highest test
  accuracy at 0.8810).
- **Structured parameters change the oracle ranking.** The flat softmax
  parameter block here favors L-BFGS; on genuinely matrix-shaped models a
  structure-aware preconditioner (e.g. the Shampoo oracle) may compete
  differently, and its blocked inverse-root cost may amortize better.

## Overall Assessment

The empirical evidence supports QQN's central design claims: it is a competitive
quasi-Newton optimizer whose modular four-axis architecture (gradient, oracle,
search, region) behaves as specified, with the steepest-descent path anchor
delivering robust convergence and the L-BFGS oracle delivering the bulk of the
convergence speed (a **1.63× iteration speedup over L-BFGS** at deep memory).
The oracle axis is the dominant lever; the line search and region axes are best
understood as tunable trade-offs — efficiency and safety/acceleration levers,
respectively. Two cautionary findings temper the earlier optimism:
**strong-Wolfe and the bare adaptive trust-region both fail to converge** on
this well-conditioned smooth problem — and, under the current honest `pred`
model, the spline's monotone gating no longer rescues the adaptive region.
Meanwhile, the **Hessian-free Anderson and matrix-free secant oracles** emerge
as standout robust additions to the toolkit, leading the trajectory-AUC axis
despite their `O(n)`–`O(m²n)` memory and complete absence of a stored Hessian.