# Analysis: 4-Layer MLP Benchmark (Fashion-MNIST) — QQN vs SGD/Adam/L-BFGS

**Run:** `fashion_mnist_mlp_comparison_20260622_144249.log`
**Date:** 2026-06-22

![fashion_mnist_mlp_comparison_vs_time_20260622-145248.png](fashion_mnist_mlp_comparison_vs_time_20260622-145248.png)

## 1. Experiment Configuration

| Setting             | Value                                                 |
|---------------------|-------------------------------------------------------|
| Dataset             | Fashion-MNIST (via `tensorflow.keras`)                |
| Architecture        | `x->256->256->256->10` (4-layer MLP, 3 hidden layers) |
| Activation          | `tanh,gelu,tanh` (mixed, cycled per hidden layer)     |
| Classes             | 10                                                    |
| Train / Test        | 20000 / 5000 (full-batch)                             |
| Parameters          | 335,114                                               |
| `maxiter`           | 100,000                                               |
| Stop: `f_target`    | 8.0e-2                                                |
| Stop: `gtol`        | 1.0e-8                                                |
| Stop: `time_budget` | 30.0 s                                                |
| Milestones          | 1.0, 0.5, 0.2, 0.1                                    |

This is a deliberately non-convex, full-batch objective (deep MLP with smooth
`tanh`/`gelu` activations). Per [`algorithm.md`](algorithm.md), this is the
regime where QQN's gradient+oracle blending along the quadratic path should pay
off: a richly anisotropic Hessian, smooth enough for cubic-spline modeling.

## 2. Headline Result

**The deep-memory L-BFGS oracle is the clear winner.** `QQN-L50` reaches the
target loss (8e-2) in **243 iterations vs. L-BFGS's 447** — a **1.84x
iteration speedup** — and is the *sole* point on the loss-vs-time Pareto
frontier (10.26 s, lowest final loss).

| Variant        | Final loss | Iters | →target | vs L-BFGS | Wall (s) | ms/it |
|----------------|------------|-------|---------|-----------|----------|-------|
| **QQN-L50**    | 7.956e-2   | 243   | 243     | **1.84x** | 10.26    | 42.2  |
| QQN-L50And     | 7.956e-2   | 243   | 243     | 1.84x     | 10.60    | 43.6  |
| QQN-L80        | 7.972e-2   | 246   | 246     | 1.82x     | 10.91    | 44.4  |
| QQN-Max        | 7.960e-2   | 253   | 253     | 1.77x     | 24.12    | 95.3  |
| QQN-L20        | 7.990e-2   | 325   | 325     | 1.38x     | 12.80    | 39.4  |
| L-BFGS         | 7.994e-2   | 447   | 447     | 1.00x     | 11.29    | 25.3  |
| QQN (baseline) | 7.995e-2   | 431   | 431     | 1.04x     | 17.21    | 39.9  |

The speedup is **monotone in the target tightness** (a robustness signal, not a
cherry-picked point):

| Target  | QQN-L50 iters | L-BFGS iters | Speedup   |
|---------|---------------|--------------|-----------|
| ≤2.0e-1 | 141           | 196          | 1.39x     |
| ≤1.5e-1 | 173           | 249          | 1.44x     |
| ≤1.0e-1 | 217           | 332          | 1.53x     |
| ≤8.0e-2 | 243           | 447          | **1.84x** |

The gap *widens* as the target tightens — QQN's curvature blend increasingly
dominates near the basin, consistent with the superlinear-convergence claim in
[`algorithm.md`](algorithm.md) (the selected `t` approaching 1 as the oracle
direction becomes trustworthy).

## 3. The Curvature-Memory Lever

Increasing L-BFGS history is the single most effective convergence lever, and
it is **monotone** up to a saturation point:

| History      | Variant | Iters→target |
|--------------|---------|--------------|
| 10 (default) | QQN     | 431          |
| 20           | QQN-L20 | 325          |
| 50           | QQN-L50 | 243          |
| 80           | QQN-L80 | 246          |

Returns saturate between 50 and 80 (243 → 246, essentially flat). On this
335k-parameter problem, ~50 curvature pairs already capture the dominant
anisotropy; the extra 30 pairs add memory/compute cost without benefit. **L50
is the sweet spot.**

## 4. The Probe-Feeding Trap (Critical Negative Result)

**Every `feed_probes_to_oracle=True` variant catastrophically failed** to reach
the target, despite the docstrings predicting a "free curvature boost":

| Variant    | Probe-fed | Final loss  | Iters | Status      |
|------------|-----------|-------------|-------|-------------|
| QQN-L50    | No        | **7.96e-2** | 243   | ✅ converged |
| QQN-L50P   | **Yes**   | 2.02e-1     | 230   | ❌ stalled   |
| QQN-L80P   | **Yes**   | 3.61e-1     | 163   | ❌ stalled   |
| QQN-MaxP   | **Yes**   | 4.13e-1     | 144   | ❌ stalled   |
| QQN-UltraP | **Yes**   | 4.59e-1     | 90    | ❌ stalled   |

The pattern is stark and *anti-correlated with aggression*: the more
aggressively a variant feeds probes (and the more warm-started its
backtracking), the **worse** it does. The loss trajectory confirms a hard
plateau — `QQN-UltraP` flattens at log10 ≈ -0.34 after iteration ~20 and never
recovers. These variants are also the slowest per-iteration (`QQN-UltraP` =
336 ms/it, ~8x the L50 baseline), so they burn the entire 30 s budget on a
handful of expensive, non-productive steps.

**Root cause hypothesis:** Feeding *rejected* line-search probes (or probes far
from the accepted step) injects spurious `(s, y)` curvature pairs that violate
the well-conditioning the L-BFGS two-loop recursion relies on (the `⟨y,s⟩ > ε`
safeguard in [`algorithm.md`](algorithm.md) admits them but cannot guarantee
they are *representative*). The Anderson fallback in the `*P` stacks does not
rescue this — it masks the degenerate direction but the polluted history still
distorts every subsequent step. The `QQN-Smooth` docstring claims
"descent-gated" probe-feeding makes this safe, but `QQN-Smooth` *also* stalled
(5.90e-1, worst converged variant), so the gating is either not active or
insufficient on this surface.

**Recommendation:** Treat `feed_probes_to_oracle=True` as **harmful by default**
on full-batch non-convex objectives. Do not ship it as a recommended lever
until the probe-filtering (descent-gating) is verified to actually prevent
history pollution.

## 5. The Spline Refinement Is a Wall-Clock Liability Here

The spline wrapper (`spline=True`) did **not** help and substantially hurt
throughput:

| Variant  | Spline | Iters→0.1 | ms/it | →target (8e-2) |
|----------|--------|-----------|-------|----------------|
| QQN      | No     | 340       | 39.9  | 431 ✅          |
| QQN-S    | Yes    | 308       | 82.7  | — ❌ (timeout)  |
| QQN-BT   | No     | 340       | 37.2  | 431 ✅          |
| QQN-BT-S | Yes    | 308       | 82.7  | — ❌ (timeout)  |

The spline modestly improves *per-iteration* progress (308 vs 340 iters to
reach 0.1) — its information-reuse claim from [`spline_search.md`] holds — but it
**doubles** the per-iteration cost (82.7 vs 39.9 ms/it). The net effect is that
both spline variants *time out* before reaching the 8e-2 target despite having
fewer iterations-to-0.1. The smooth `tanh`/`gelu` surface was chosen precisely
to favor the cubic model, yet the extra stationary-point probes do not pay for
themselves in wall-clock. The spline's iteration efficiency is real but its
constant-factor overhead is not amortized at this problem scale.

## 6. Oracle Comparison: L-BFGS Dominates

Among the alternative oracles, **only L-BFGS-based variants converged**:

| Oracle family            | Best variant | Result                         |
|--------------------------|--------------|--------------------------------|
| L-BFGS                   | QQN-L50      | ✅ 243 iters                    |
| Momentum                 | QQN-Mom      | ❌ stalled (4.49e-1)            |
| Secant                   | QQN-Sec      | ❌ stalled (3.17e-1)            |
| Anderson                 | QQN-And      | ❌ stalled (3.81e-1)            |
| L-BFGS+Anderson fallback | QQN-L50And   | ✅ 243 iters (identical to L50) |

`QQN-L50And` is **byte-identical** to `QQN-L50` (7.956406e-2, 243 iters): the
Anderson fallback never triggered because L-BFGS always produced a valid
direction. This confirms the fallback is a zero-cost safety net here (per the
`Fallback` semantics in [`algorithm.md`](algorithm.md)) — it neither helped nor
hurt. The first-order/matrix-free oracles (Momentum, Secant, Anderson) are not
competitive: they plateau early, lacking the rich curvature memory that the
anisotropic full-batch Hessian rewards.

## 7. Regions: Negligible Effect

| Variant | Region                | Iters→target | vs L-BFGS |
|---------|-----------------------|--------------|-----------|
| QQN     | None                  | 431          | 1.04x     |
| QQN-Box | Box[-2,2]             | 431          | 1.04x     |
| QQN-TR  | TrustRegion(adaptive) | 414          | 1.08x     |

`QQN-Box` is identical to bare `QQN` (the weights never reached the ±2 bounds,
so the projection was inert). `QQN-TR` gives a marginal 4% improvement. Neither
region is a meaningful lever on this unconstrained problem — consistent with the
"zero overhead when inactive" design goal, and confirming regions are a
feasibility/preference tool, not a convergence accelerator.

## 8. Cost-Aware View (Evaluations-to-Target)

The iteration-count win narrows under the eval-cost model because QQN issues
~5 evals/iteration (path eval + line-search probes) vs. L-BFGS's ~3:

| Variant | Iters | evals/it | evals→target | vs L-BFGS |
|---------|-------|----------|--------------|-----------|
| QQN-L50 | 243   | 5.0      | 1215         | **1.10x** |
| QQN-L80 | 246   | 5.0      | 1230         | 1.09x     |
| L-BFGS  | 447   | 3.0      | 1341         | 1.00x     |
| QQN-L20 | 325   | 5.0      | 1625         | 0.83x     |

Under this fairer metric the QQN-L50 advantage shrinks from 1.84x (iterations)
to **1.10x (evaluations)** — still a win, but a modest one. **Caveat:** these
eval counts are *analytic estimates* (`_estimate_evals_per_iter`), not measured;
the true L-BFGS zoom-probe count may differ. The iteration-count advantage is
the robust, directly-measured result; the eval-count advantage is suggestive
but should be validated with an instrumented `EvalCounter` (currently a no-op
stub in the harness).

## 9. Baselines (SGD / Adam)

Both first-order baselines exhausted the 30 s budget without reaching the
target (SGD 3.08e-1, Adam 2.74e-1) despite running 4500+ iterations at ~6.5
ms/it. On a smooth, full-batch, ill-conditioned objective this is expected:
they lack curvature information and crawl along the anisotropic basin. Adam's
trajectory is notably noisy (log10 bounces between -0.87 and +0.37), reflecting
its adaptive-moment instability at this learning rate on a full-batch loss.

## 10. Conclusions & Recommendations

1. **Ship `QQN-L50` (or `QQN-L50And`) as the recommended config** for
   full-batch non-convex problems of this scale: 1.84x fewer iterations than
   L-BFGS, sole Pareto-optimal point, and the speedup is monotone in target
   tightness.
2. **Deep L-BFGS memory is the dominant lever**; it saturates around
   history=50 on a 335k-param problem. Default `history_size=10` leaves
   substantial speed on the table.
3. **Disable `feed_probes_to_oracle` by default.** It is a *trap* on this
   surface — every probe-fed variant catastrophically stalled. The
   descent-gating safeguard described in the harness/docstrings does not work
   as advertised here and needs auditing.
4. **Reconsider the spline default cost.** It improves per-iteration progress
   but doubles wall-clock per step, causing timeouts. It is only worthwhile if
   evaluations are far more expensive than they are at this scale.
5. **Anderson fallback is free insurance** (zero effect when L-BFGS is healthy)
   and is the safe way to add robustness *without* the probe-feeding risk.
6. **Validate the eval-cost model.** The 1.84x iteration win drops to 1.10x
   under estimated evals; an instrumented evaluation counter would settle
   whether the cost-aware advantage is real.

### Suggested follow-up runs

- Sweep `history_size` ∈ {30, 40, 50, 60} to pin the saturation knee precisely.
- Re-run the `*P` (probe-fed) stacks with verified descent-gating and a strict
  `⟨y,s⟩` admission test to test the "free curvature" hypothesis fairly.
- A larger-batch / higher-dimensional variant to test whether the spline's
  per-iteration win finally amortizes when evaluations dominate cost.