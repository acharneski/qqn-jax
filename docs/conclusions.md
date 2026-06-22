---
documents: results.md
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
optimizer on a smooth, deterministic, full-batch problem. The headline
results are:

- **QQN matches L-BFGS quality at a fraction of the cost.** On the softmax
  MNIST benchmark, QQN reached a final training loss of `1.034e-01` —
  statistically indistinguishable from Optax's L-BFGS (`1.039e-01`) — while
  running roughly **2.4× faster** in wall-clock time (0.908s vs. 2.166s).
- **QQN clearly beats first-order baselines on training loss.** It drove the
  loss roughly **3× lower than SGD** and well below Adam, confirming the
  benefit of quasi-Newton acceleration on smooth deterministic objectives.
- **The four-axis modular design behaves as specified.** Each swappable
  component (gradient, oracle, search, region) could be substituted
  independently, and the defaults (`oracle="lbfgs"`, `region=None`) reproduced
  the baseline behavior exactly.

## Validation of Core Algorithmic Claims

### The Combiner Model Holds

The central thesis of [`algorithm.md`](algorithm.md) — that QQN is a
**combiner** of orthogonal, independently swappable components — is borne out
by the controlled A/B sweeps. Swapping the oracle, line search, or region in
isolation produced predictable, decomposable effects on loss and wall-time,
with no cross-component coupling that would undermine the modularity claim.

### Global Convergence via the Steepest-Descent Anchor

Across every oracle (including the deliberately aggressive Shampoo and the
weak high-`beta` momentum oracle), the line search always returned a
decreasing step or rejected it. This is consistent with the theoretical
guarantee that the path property `d'(0) = -∇f` anchors global convergence
regardless of oracle quality, leaving the oracle free to be aggressive.

## Component-Level Conclusions

### Oracle Choice Is the Dominant Lever

- **L-BFGS history depth** is the single most important accuracy lever, with a
  monotone improvement `L5 < L10 < L20 < L50 < L100`, clear diminishing
  returns past size 50, and a hard plateau at 100 (`1.024e-01`). Very deep
  histories buy almost no accuracy for their extra cost.
- **Momentum** behaved as a first-order accelerator and trailed L-BFGS
  substantially; lighter damping (`beta = 0.1`) — which collapses toward
  steepest descent — outperformed heavier momentum on this smooth problem.
- **Shampoo** reached only a moderate loss and was orders of magnitude slower
  (174.4s), confirming it is suited to genuinely matrix-structured parameters
  rather than the flat softmax vector used here.

### Line Search Trades Time, Not Final Loss

On this smooth convex objective, the line search choice had negligible effect
on the converged loss but a large effect on wall-time. Armijo backtracking was
the clear efficiency winner; strong-Wolfe, Hager-Zhang, and the spline
refinement matched its quality at ~2–3× the cost. This confirms that the more
expensive searches do **not** degrade quality — they simply do not pay off on
a well-conditioned objective where curvature information is easy to exploit.

### The Spline Refinement Composes, As Designed

Consistent with [`spline_search.md`](spline_search.md), the spline behaves as
an orthogonal enhancement that *wraps* (rather than replaces) the inner search.
It did not change the converged loss on the smooth objective, but it
measurably **sharpened the deep-memory trajectory** — `QQN-L50Spln` reached the
`-0.99` log10 plateau distinctly earlier than the spline-less baseline.

### Regions Are Low-Overhead Safeguards

The adaptive trust-region barely perturbed the converged loss across radii,
confirming regions function as cheap safeguards rather than performance drivers
on a well-conditioned problem. The orthant region was the only configuration to
induce measurable weight sparsity (0.0008), exactly as its sign-preserving
projection predicts. Box and stacked constraints added negligible cost.

### Combinators Work Correctly

`Fallback([L-BFGS, Momentum])` reproduced the L-BFGS baseline exactly, because
the L-BFGS direction was always valid and the momentum fallback never
triggered — the intended behavior. Stacked oracle/region combinators ran
correctly and produced sensible results.

## Best-of-Breed Recommendation

Stacking the strongest pareto components — deep L-BFGS memory (size 50),
backtracking line search, and an adaptive trust-region — yielded the lowest
observed losses (`1.024e-01`) at competitive wall-time (~0.86s). For smooth,
deterministic, full-batch problems, this configuration represents a strong
default: most of the accuracy of the deepest histories with the cheapest robust
search and a low-overhead convergence safeguard.

## Limitations and Caveats

The conclusions above are drawn from a **single, smooth, deterministic,
full-batch convex benchmark** (softmax MNIST). They should be read with the
following caveats:

- **Smoothness flatters cheap searches.** On non-smooth or ill-conditioned
  objectives, the stronger Wolfe/Hager-Zhang searches and the spline
  refinement may pay off where they did not here.
- **Structured parameters change the oracle ranking.** Shampoo's poor showing
  is specific to the flat parameter block; on genuinely matrix-shaped models
  its structure-aware preconditioning may compete differently.
- **Generalization was not the differentiator.** Test accuracy was similar
  across the strong optimizers; these results concern optimization speed and
  final training loss, not generalization.

## Overall Assessment

The empirical evidence supports QQN's central design claims: it is a competitive
quasi-Newton optimizer whose modular four-axis architecture (gradient, oracle,
search, region) behaves as specified, with the steepest-descent path anchor
delivering robust convergence and the L-BFGS oracle delivering the bulk of the
accuracy. The line search and region axes are best understood as tunable
trade-offs — efficiency and safety levers, respectively — rather than primary
accuracy drivers on smooth problems.