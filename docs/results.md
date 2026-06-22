---
documents:
  - ../mnist_comparison.log
related:
  - algorithm.md
  - ../README.md
  - ../examples/mnist_comparison.py
  - ../qqn_jax/solver.py
  - ../qqn_jax/line_search.py
  - ../qqn_jax/oracles.py
  - ../qqn_jax/spline_search.py
  - ../qqn_jax/regions.py
---

# Experimental Results: MNIST Optimizer Comparison

This document summarizes the empirical validation of QQN against standard
baselines and across its own swappable components. The driving experiment is
[`examples/mnist_comparison.py`](../examples/mnist_comparison.py); the raw
output it produced is captured in
[`mnist_comparison.log`](../mnist_comparison.log).

## Experimental Setup

The benchmark frames MNIST classification as a **full-batch, deterministic**
optimization problem — a softmax (multinomial logistic regression) classifier
with L2 regularization (`l2 = 1e-4`). The full-batch framing is deliberate: it
keeps the comparison apples-to-apples for the second-order methods (QQN and
L-BFGS), which assume a smooth, deterministic objective.

| Setting        | Value                                                      |
|----------------|------------------------------------------------------------|
| Classes        | 10                                                         |
| Train samples  | 5000                                                       |
| Test samples   | 1000                                                       |
| Max iterations | 500                                                        |
| Model          | Softmax / multinomial logistic regression                  |
| Loss           | Cross-entropy + `0.5·l2·‖params‖²` (`l2 = 1e-4`)           |
| Init           | Shared `PRNGKey(42)` so every optimizer starts identically |

Data is loaded from real MNIST via `tensorflow.keras` or `torchvision` when
available, and falls back to a synthetic Gaussian-blob dataset otherwise, so
the experiment always runs.

All QQN variants are run **one update at a time** (via `solver.init_state` +
a JIT-compiled `solver.update`) to record the full loss trajectory; the Optax
baselines (`SGD`, `Adam`, `L-BFGS`) use their own JIT-compiled step loops.

The default QQN configuration uses the **L-BFGS oracle** (`history_size=10`),
the **Armijo backtracking line search**, and **no region**.

> **Robustness note (carried in the run banner):** under the current honest
> predicted-reduction model, the *adaptive* trust-region over-shrinks and
> stalls. The robust fast path on this problem class is a **fixed-radius
> trust-region (or no region) combined with deep L-BFGS memory and warm-started
> backtracking** — see the iteration-efficiency leaderboard below.

### Shared, Fair Termination Bounds

A key feature of the experiment is that **every optimizer races against the
same termination criteria**, making the comparison strictly apples-to-apples.
Rather than each method using its own private stopping rule, all share:

| Bound         |    Value | Meaning                                       |
|---------------|---------:|-----------------------------------------------|
| `f_target`    | `1.1e-1` | Stop once full-batch loss `≤` this value.     |
| `gtol`        | `1.0e-4` | Stop once `‖∇f‖ ≤` this value (stationarity). |
| `time_budget` | `15.0` s | Hard wall-clock cap per optimizer.            |

The summary table records, for every method, the iteration (`->target`) and
wall-clock time (`t->tgt`) at which the shared loss/gradient target was first
reached — or `—` when the method did not reach it within the iteration limit.
The `f_target` was deliberately tuned to `1.1e-1` so the `->target` / `t->tgt`
columns become *informative*. The `time_budget` is set to `15.0`s (and Shampoo
switched to a *blocked* preconditioner) so the comparison stays meaningful
while still capping runaways.

### Convergence-Rate Milestones

Beyond a single time-to-target, the experiment also records a full
**convergence-rate profile**: a tuple of descending loss thresholds
(`5.0e-1`, `2.0e-1`, `1.5e-1`, `1.2e-1`) and, per optimizer, the first
iteration at which each threshold is crossed. This separates *early-phase*
descent speed (large-loss milestones) from *late-phase* refinement
(small-loss milestones) far more sharply than the final target alone,
surfacing methods that descend fast early but stall late (e.g. momentum) vs.
those that accelerate near the optimum (e.g. deep-memory QQN).

### Reported Metrics

The summary table reports several derived efficiency metrics:

- **`ms/it`** — mean wall-clock cost per accepted iteration (total wall-time
  divided by the number of accepted iterations), a clean per-step cost metric.
- **`vs LBFGS`** — speedup factor in iterations-to-target relative to the
  classical L-BFGS baseline (`lbfgs_iters / variant_iters`); values above
  `1.00x` indicate a variant reaches the shared target in fewer iterations
  than L-BFGS.
- **`AUC`** — the **trajectory AUC**: `log10(loss)` integrated over the
  normalized iteration axis (trapezoid rule). A *lower* (more negative) AUC
  means the optimizer spent its whole trajectory at lower loss, rewarding
  fast early descent **and** deep late refinement simultaneously — a far more
  discriminating single-scalar summary than a single time-to-target.

The script also emits four leaderboards/reports derived from these metrics:
a **Pareto frontier** (loss vs. wall-time), an **iteration-efficiency
leaderboard** (converging variants ranked by fewest iterations-to-target),
a **trajectory-AUC leaderboard**, and an explicit **stall report** that
surfaces every variant which exhausted its budget without reaching the
shared target (with a classified cause).

## Baseline Comparison

With all defaults (L-BFGS oracle, Armijo line search, no region), QQN reaches
a substantially lower full-batch loss than the first-order baselines and is
competitive with — and faster than — Optax's L-BFGS, all converging to the
shared `f_target = 1.1e-1` within the iteration/time budget.

| Optimizer | Final loss | Iters | Train acc | Test acc | Time (s) | ->target | vs LBFGS |
|-----------|-----------:|------:|----------:|---------:|---------:|---------:|---------:|
| QQN       |  1.096e-01 |    65 |    0.9902 |   0.8700 |    1.546 |       65 |    1.08x |
| L-BFGS    |  1.098e-01 |    70 |    0.9910 |   0.8750 |    2.117 |       70 |    1.00x |
| Adam      |  1.100e-01 |   263 |    0.9898 |   0.8810 |    0.526 |      263 |    0.27x |
| SGD       |  2.266e-01 |   500 |    0.9422 |   0.8900 |    0.577 |        — |        — |

**Observations:**

- QQN reaches the shared loss target in **65 iterations**, fewer than Optax's
  L-BFGS (70, a **1.08×** iteration speedup) and running ~**1.4× faster** in
  wall-clock time, owing to its cheap Armijo backtracking search.
- **Adam** also reaches the target but needs **263 iterations** — far more
  than the quasi-Newton methods (a `0.27x` iteration speedup vs L-BFGS) —
  though its per-iteration cost is so low (~2.0 ms/it) that it is the fastest
  in wall-clock time (0.526s) on this problem.
- **SGD never reaches** the `f_target = 1.1e-1` target within 500 iterations,
  plateauing at `2.266e-01`.
- Test accuracy is similar across the strong optimizers; the differentiator
  here is optimization speed and iterations-to-target, not generalization
  (Adam actually has the highest test accuracy at 0.8810).

## QQN Component Sweeps (A/B Comparisons)

Because gradient, oracle, line search, and region are conceptually orthogonal
and independently swappable, the experiment runs controlled A/B sweeps where
each pair isolates a single variable against a named baseline. Each pair's
first entry is the baseline; later entries report deltas against it.

### Oracle: L-BFGS History Depth

Deeper L-BFGS history monotonically reduces the **iterations to target**, with
clear diminishing returns past size 50 and a hard plateau at size 100. The
converged final loss is essentially flat across depths (every variant hits the
shared target), so the lever here is *speed of convergence*, not final loss.

| Variant  | History | Final loss | Iters | ->target | Time (s) | AUC   |
|----------|--------:|-----------:|------:|---------:|---------:|------:|
| QQN-L5   |       5 |  1.093e-01 |    80 |       80 |    1.432 | -0.73 |
| QQN      |      10 |  1.096e-01 |    65 |       65 |    1.546 | -0.70 |
| QQN-L20  |      20 |  1.096e-01 |    59 |       59 |    1.218 | -0.68 |
| QQN-L50  |      50 |  1.097e-01 |    43 |       43 |    1.089 | -0.61 |
| QQN-L100 |     100 |  1.097e-01 |    43 |       43 |    1.108 | -0.61 |

The sweep `L5 > L10 > L20 > L50` in iterations-to-target (80 → 65 → 59 → 43)
confirms richer curvature memory accelerates convergence, but the count
plateaus exactly at 43 iterations from size 50 onward (`L50 == L100`) while
wall-time keeps growing — so very deep histories (L100) buy *no* extra speed
for extra cost on this problem.

### Oracle: Secant (Barzilai-Borwein)

The **secant** oracle is a matrix-free, `O(n)`-memory curvature estimate that
reuses the *realized* step's secant `(s, y)` to form a Barzilai-Borwein step
`α = ⟨s,s⟩/⟨s,y⟩`, then proposes `-α·∇f`. It carries no Hessian and no history
buffers — it probes how much curvature lives in a *single* realized step.

| Variant | Oracle             | Final loss | Iters | ->target | AUC   |
|---------|--------------------|-----------:|------:|---------:|------:|
| QQN-Sec | Secant (BB1, O(n)) |  1.097e-01 |   311 |      311 | -0.74 |

`QQN-Sec` does eventually reach the shared target, but needs **311 iterations**
(a `0.23x` iteration speedup vs L-BFGS) — far more than any L-BFGS depth, yet
far fewer than the momentum oracles (which never reach it). Notably its
**trajectory AUC of `-0.74` is among the best of all QQN variants**: the BB
step descends fast and deep on average even though it takes many iterations to
formally cross the tight target. This makes the single-step secant a strong,
zero-storage curvature signal.

### Oracle: Anderson Acceleration

The **Anderson** oracle is the variational ideal that L-BFGS approximates. It
forms the `t = 1` endpoint by solving a tiny constrained least-squares problem
over recent gradient *differences* — an `(m × m)` system over a sliding window
of residuals — with no Hessian ever formed (`AndersonOracle(window=5)`). It
captures the optimal multi-step residual combination that the fixed-window
two-loop recursion only approximates.

| Variant | Oracle               | Final loss | Iters | ->target | AUC   |
|---------|----------------------|-----------:|------:|---------:|------:|
| QQN-And | Anderson (window=5)  |  1.100e-01 |   269 |      269 | -0.84 |

`QQN-And` reaches the shared target in **269 iterations** (a `0.26x` speedup vs
L-BFGS) — like the secant, far more iterations than L-BFGS, but it posts the
**best trajectory AUC of any method in the study (`-0.84`)**, edging out even
Adam (`-0.82`). The deep, optimal residual combination keeps the whole
trajectory at very low loss even though many cheap steps are needed to cross
the tight final target.

### Combinator: L-BFGS with an Anderson / Secant Fallback

A `Fallback([LBFGSOracle, ...])` uses the L-BFGS direction while it is a valid
descent direction (finite, non-zero, and `⟨∇f, d⟩ < 0`), and only switches to
the secondary oracle when the L-BFGS curvature estimate degenerates. On this
smooth, well-conditioned problem the L-BFGS direction is always valid, so the
fallback never triggers and the combinator reproduces the deep-L-BFGS behavior
exactly.

| Variant    | Oracle                            | Final loss | Iters | ->target |
|------------|-----------------------------------|-----------:|------:|---------:|
| QQN-L50And | Fallback([L-BFGS(50), Anderson])  |  1.097e-01 |    43 |       43 |
| QQN-L50Sec | Fallback([L-BFGS(50), Secant])+TR |  6.272e-01 |   500 |        — |

`QQN-L50And` ties the bare `QQN-L50` exactly (43 iterations, `1.097e-01`),
confirming the fallback is inert when L-BFGS is healthy. `QQN-L50Sec` pairs the
same fallback idea with an *adaptive* `TrustRegion` and consequently **stalls**
at `6.272e-01` — the oracle fallback is sound, but it inherits the
adaptive-trust-region instability documented below.

> **Important update:** the `Fallback` validity test is now **descent-based**,
> not merely non-zero. A finite, non-zero quasi-Newton direction that points
> *uphill* (`⟨∇f, d⟩ ≥ 0`) is rejected and the fallback triggers — the gate is
> misalignment, not just collapse.

### Oracle: Momentum (heavy-ball) `beta`

The momentum oracle is a first-order accelerator and, as expected, **never
reaches the target** within 500 iterations. Notably, *lighter* damping
converges to a lower loss on this problem (the sweep is monotone in `beta`).

| Variant   | beta | Final loss | Iters | ->target | Time (s) |
|-----------|-----:|-----------:|------:|---------:|---------:|
| QQN-Mom01 | 0.01 |  1.892e-01 |   500 |        — |    5.427 |
| QQN-Mom10 | 0.10 |  1.940e-01 |   500 |        — |    5.491 |
| QQN-Mom50 | 0.50 |  2.265e-01 |   500 |        — |    5.534 |
| QQN-Mom   | 0.90 |  3.419e-01 |   500 |        — |    5.480 |

Near-zero momentum (`beta = 0.01`) effectively collapses toward steepest
descent, which on this smooth full-batch problem outperforms heavier momentum
(`Mom01 < Mom10 < Mom50 < Mom` in loss). All momentum variants exhaust the
full 500-iteration budget without reaching the target. The convergence-rate
profile is revealing: the lighter-damping variants (`Mom01`, `Mom10`) do
eventually cross the `2.0e-1` milestone (at iterations 410 and 449
respectively), while the heavier `Mom50`/`Mom` never do.

### Oracle: Accelerator Class (Momentum vs Shampoo)

The structure-aware **Shampoo** oracle recomputes inverse matrix roots on a
static cadence (`update_freq=25`) over a *blocked* preconditioner
(`block_size=64`), so the per-refresh eigendecomposition is `O(block³)`
instead of `O(n³)`. Even so, on this high-dimensional softmax problem the
refresh is expensive: `QQN-Sh` exhausts the shared **15-second wall-clock
budget** after only **9 iterations** and lands at a much higher loss than even
the momentum oracle.

| Variant   | Oracle              | Final loss | Iters | ->target | Time (s) |
|-----------|---------------------|-----------:|------:|---------:|---------:|
| QQN-Mom10 | Momentum (beta=0.1) |  1.940e-01 |   500 |        — |    5.491 |
| QQN-Sh    | Shampoo (block=64)  |  8.883e-01 |     9 |        — |   16.338 |

Shampoo's blocked inverse-root refresh still does not amortize well at this
scale (~1815 ms/it); it is the only oracle to exhaust the time budget before
reaching `maxiter`, and the only variant whose convergence-rate profile never
crosses even the loosest `5.0e-1` milestone.

### Region: Trust-Region Radius and Adaptivity — a Cautionary Result

> **Important:** with the current honest predicted-reduction model and the
> default Armijo (init_step=1.0) line search, the *adaptive* trust-region
> **destabilizes** convergence on this problem. Several trust-region variants
> that previously won the race now **stall and never reach the target**.

| Variant   | Radius | Adaptive | Final loss | ->target | Time (s) |
|-----------|-------:|:--------:|-----------:|---------:|---------:|
| QQN-TRfix |   1.00 |    no    |  1.099e-01 |       68 |    1.298 |
| QQN-TR025 |   0.25 |   yes    |  1.983e+00 |        — |    6.826 |
| QQN-TR    |   1.00 |   yes    |  6.272e-01 |        — |    6.576 |
| QQN-TR2   |   2.00 |   yes    |  6.256e-01 |        — |    6.370 |

The only trust-region variant to reach the target is the **fixed** radius
`QQN-TRfix` (68 iterations). Every *adaptive* radius (`TR025`, `TR`, `TR2`)
plateaus early: `QQN-TR` flatlines at `6.272e-01` and `QQN-TR2` at
`6.256e-01`. The tighter radius is the worst (`TR025` ends at `1.983e+00`,
i.e. *above* its start). This is a sharp reversal from earlier runs where the
adaptive trust-region marginally accelerated convergence: the current honest
`pred ≈ -⟨∇f, d(t)⟩` model (with a curvature correction and a floor at the
first-order model), combined with the radial step-clipping, drives the
adaptive radius to over-shrink and stall the search.

The **adaptivity** A/B makes this stark:

| Variant   | Adaptive | Final loss | ->target | Time (s) |
|-----------|:--------:|-----------:|---------:|---------:|
| QQN-TRfix |    no    |  1.099e-01 |       68 |    1.298 |
| QQN-TR    |   yes    |  6.272e-01 |        — |    6.576 |

Switching the radius from fixed to adaptive moves the result from *converged
at 68 iterations* to *never reaches the target*. **Use a fixed trust-region
radius (or no region) on this class of well-conditioned smooth problem.**

### Region: Combinator and Orthant Sparsity

The combinator `Sequential([Box, TrustRegion])` composes two projections in
order at negligible extra cost, and the orthant region is the only QQN region
to induce measurable weight sparsity.

| Variant  | Configuration                    | Final loss | Sparsity | ->target |
|----------|----------------------------------|-----------:|---------:|---------:|
| QQN-Box  | BoxRegion(-2, 2)                 |  1.098e-01 |   0.0000 |       65 |
| QQN-Orth | OrthantRegion (OWL-QN-style)     |  1.100e-01 |   0.0027 |       70 |
| QQN-Seq  | Sequential([Box(-2,2), TR(1.0)]) |  6.272e-01 |   0.0000 |        — |

The **box** region adds negligible cost while bounding weights (reaches target
at 65). The **orthant** region induces measurable sparsity (0.0027). The
`Sequential` combinator *inherits the adaptive trust-region's stall*
(`6.272e-01`, never reaching target): composition itself is correct, but its
nested adaptive `TrustRegion(1.0)` carries the same destabilizing behavior
documented above.

### Line Search (at fixed oracle depth, L-BFGS-10)

The line search choice has negligible effect on the *iterations-to-target* for
the **backtracking/Armijo** family but a large effect on **wall-time** — and a
*dramatic* effect for **strong-Wolfe**, which fails to converge here.

| Variant  | Line search   | Final loss | ->target | Time (s) | AUC   |
|----------|---------------|-----------:|---------:|---------:|------:|
| QQN      | armijo        |  1.096e-01 |       65 |    1.546 | -0.70 |
| QQN-BT   | backtracking  |  1.096e-01 |       65 |    1.294 | -0.70 |
| QQN-Spln | armijo+spline |  1.097e-01 |       64 |    2.638 | -0.71 |
| QQN-SW   | strong_wolfe  |  4.077e-01 |        — |    7.640 | -0.38 |

The default search is Armijo backtracking (`QQN`, 1.546s); the dedicated
`QQN-BT` backtracking variant is the cheapest robust search (1.294s) and
reaches the target in the same 65 iterations. The cubic Hermite spline
refinement (`QQN-Spln`) reaches the target slightly earlier (64) but at ~1.7×
the wall-time. **Strong-Wolfe (`QQN-SW`) fails to converge** on this problem,
plateauing at `4.077e-01` after exhausting all 500 iterations — its tight
curvature condition over-restricts the step along the quadratic path here.

### Spline Refinement (orthogonal enhancement)

The spline is **not** a line-search strategy but a boolean enhancement
(`spline=True`) that *wraps* any chosen line search (`spline_wrap(inner_search)`).
It reuses every probe along the consistent path as a cubic Hermite control
point, performs an **adaptive superlinear extension probe** when the downstream
tangent still descends (`m1 < 0`) — extending the bracket toward the estimated
stationary point `α* = a1·m0/(m0 − m1)` with a slope-ratio-relaxed cap — and
probes the spline's stationary points to improve on the inner search's accepted
step.

| Variant       | Configuration                           | Final loss | ->target |
|---------------|-----------------------------------------|-----------:|---------:|
| QQN-Spln      | armijo + spline                         |  1.097e-01 |       64 |
| QQN-BTSpln    | backtracking + spline                   |  1.097e-01 |       64 |
| QQN-L50Spln   | L50 oracle + spline                     |  1.091e-01 |       43 |
| QQN-L100Spln  | L100 oracle + spline                    |  1.091e-01 |       43 |
| QQN-SplnTR    | armijo + spline + adaptive trust-region |  2.039e-01 |        — |
| QQN-L50SplnTR | L50 + spline + adaptive trust-region    |  1.653e-01 |        — |

The spline refinement notably **sharpens the deep-memory trajectory**:
`QQN-L50Spln` reaches the target in **43 iterations** (tying the spline-less
L50 baseline) with the **lowest loss among the converging spline variants
(`1.091e-01`)**.

> **Reversal from earlier runs:** the spline-wrapped **adaptive** trust-region
> stacks (`QQN-SplnTR`, `QQN-L50SplnTR`) now **stall** under the current honest
> `pred` model — `QQN-SplnTR` plateaus at `2.039e-01` and `QQN-L50SplnTR` at
> `1.653e-01`, both exhausting the full 500 iterations. The spline's
> strict-improvement gating keeps these trajectories monotone (and they reach
> the best AUCs among the stalled set), but it is no longer sufficient to
> overcome the over-shrinking adaptive radius on this problem. **Prefer a fixed
> radius (or no region) when combining the spline with a trust-region.**

On the smooth convex objective the extra per-probe spline fitting costs
roughly ~2× wall-time for the shallow-memory variants.

## Best-of-Breed Combinations

The strongest **converging** stacks here are the **deep-memory + warm-started
backtracking + fixed-radius trust-region** combinations, not the deep-memory +
bare-adaptive-trust-region stacks (which stall, see below). The experiment
probes several *performance-tuned* stacks that warm-start the line search at a
larger initial step (`init_step > 1`) so deep-memory quasi-Newton steps can
stretch into the superlinear regime, paired with a *fixed* trust-region so the
aggressive initial step is a clean speed lever rather than feeding the
destabilizing adaptive radius.

| Variant       | Configuration                              | Final loss | ->target | Time (s) |
|---------------|--------------------------------------------|-----------:|---------:|---------:|
| QQN-L50       | L50 oracle (no region)                     |  1.097e-01 |       43 |    1.089 |
| QQN-L100      | L100 oracle (no region)                    |  1.097e-01 |       43 |    1.108 |
| QQN-L50And    | Fallback([L50, Anderson]) (no region)      |  1.097e-01 |       43 |    1.168 |
| QQN-L50Spln   | L50 + spline                               |  1.091e-01 |       43 |    2.352 |
| QQN-L100Spln  | L100 + spline                              |  1.091e-01 |       43 |    2.406 |
| QQN-L50TRfix  | L50 + BT + fixed trust-region              |  1.095e-01 |       46 |    1.100 |
| QQN-Fast      | L100 + warm-start BT + fixed trust-region  |  1.090e-01 |       57 |    1.281 |
| QQN-L50WS+    | L50 + warm-start(2.0) BT + fixed TR        |  1.097e-01 |       57 |    1.237 |
| QQN-L50Endpt  | L50 + warm-start(2.0) BT + fixed TR        |  1.097e-01 |       57 |    1.229 |
| QQN-Champion  | L50 + warm-start(3.0) BT + fixed TR(1.5)   |  1.097e-01 |       59 |    1.274 |
| QQN-L20HZ     | L20 + Hager-Zhang                          |  1.093e-01 |       62 |    1.248 |

The fewest iterations to target (**43**) are reached by the **bare deep-memory
oracles** `QQN-L50`, `QQN-L100`, the deep-memory + spline combos
`QQN-L50Spln` / `QQN-L100Spln`, and the deep-memory + Anderson-fallback
`QQN-L50And` — at the lowest wall-time (~1.04–1.17s for the region-free
variants), a strong pareto point on iterations vs. time. The **lowest loss
observed across the whole study (`1.090e-01`)** is reached by **`QQN-Fast`**
(L100 + warm-started backtracking + a *fixed* trust-region), at 57 iterations.

> **Critical caveat — the bare deep-memory + adaptive-trust-region stacks
> stall.** Combos such as `QQN-L50TR`, `QQN-L100TR`, and `QQN-L50BTTR` all
> **fail to converge**, plateauing at `6.272e-01` after the full 500
> iterations, and `QQN-L50TR2` at `6.256e-01`. The fix is to **drop the
> adaptive radius**: the warm-started backtracking stacks paired with a
> *fixed* trust-region (`QQN-Fast`, `QQN-L50WS`, `QQN-L50WS+`, `QQN-L50Endpt`,
> `QQN-Champion`, `QQN-L50TRfix`) **all converge cleanly** in 46–79
> iterations, recovering the intended speed lever.

The robust path to the best loss/iteration trade-off here is therefore:
**deep L-BFGS memory (L50/L100) + (warm-started) backtracking + a *fixed*
trust-region (or no region) + optionally the spline refinement**, and
*without* a bare adaptive trust-region.

### Pareto Frontier (loss vs. wall-time)

The experiment reports the **Pareto frontier** of non-dominated variants:
those for which no other variant is both faster *and* lower-loss.

| Variant      | Final loss | Time (s) |
|--------------|-----------:|---------:|
| Adam         | 1.0999e-01 |    0.526 |
| QQN-L50      | 1.0968e-01 |    1.089 |
| QQN-L50TRfix | 1.0949e-01 |    1.100 |
| QQN-L20HZ    | 1.0932e-01 |    1.248 |
| QQN-Fast     | 1.0904e-01 |    1.281 |

Adam anchors the cheap-but-higher-loss end of the frontier; the QQN variants
trade increasing wall-time for progressively lower loss, with `QQN-L50` the
standout efficiency/quality balance among the quasi-Newton methods (reaching
`1.097e-01` in close to a single second), and **`QQN-Fast` the lowest-loss
frontier point overall (`1.090e-01` at 1.281s)**.

### Iteration-Efficiency Leaderboard (converging variants)

Of the variants that actually reach the shared target, ranked by fewest
iterations (then wall-time) — the single most actionable view, since it
excludes every stalled variant:

| Variant      | Iters | Time (s) | vs LBFGS | Final loss |
|--------------|------:|---------:|---------:|-----------:|
| QQN-L50      |    43 |    1.079 |    1.63x | 1.0968e-01 |
| QQN-L100     |    43 |    1.099 |    1.63x | 1.0968e-01 |
| QQN-L50And   |    43 |    1.158 |    1.63x | 1.0968e-01 |
| QQN-L50Spln  |    43 |    2.344 |    1.63x | 1.0914e-01 |
| QQN-L100Spln |    43 |    2.396 |    1.63x | 1.0914e-01 |
| QQN-L50TRfix |    46 |    1.091 |    1.52x | 1.0949e-01 |
| QQN-L50Endpt |    57 |    1.220 |    1.23x | 1.0975e-01 |
| QQN-L50WS+   |    57 |    1.228 |    1.23x | 1.0975e-01 |
| QQN-Fast     |    57 |    1.273 |    1.23x | 1.0904e-01 |
| QQN-L20      |    59 |    1.209 |    1.19x | 1.0959e-01 |

The **1.63× iteration speedup vs L-BFGS** is the headline result: the
deep-memory oracles (with or without the spline / Anderson fallback) reach the
shared target in **43 iterations** versus L-BFGS's 70, at roughly half the
wall-time. The fixed-radius trust-region (`QQN-L50TRfix`, 46) and the
warm-started fixed-TR stacks (57) round out the robust fast tier.

### Trajectory-AUC Leaderboard

Ranking optimizers by the single-scalar **trajectory AUC** (lower = faster
*overall* descent — both early and late phase) gives a complementary view to
iterations-to-target:

| Variant      | AUC    | Final loss | Time (s) |
|--------------|-------:|-----------:|---------:|
| QQN-And      | -0.837 | 1.0998e-01 |    3.389 |
| Adam         | -0.821 | 1.0999e-01 |    0.526 |
| QQN-L50SplnTR| -0.757 | 1.6528e-01 |    7.965 |
| QQN-Best     | -0.757 | 1.6528e-01 |    8.312 |
| QQN-Sec      | -0.739 | 1.0974e-01 |    3.607 |
| QQN-L5       | -0.726 | 1.0931e-01 |    1.432 |
| QQN-Orth     | -0.715 | 1.1000e-01 |    1.361 |
| QQN-Spln     | -0.706 | 1.0969e-01 |    2.638 |
| QQN-BTSpln   | -0.706 | 1.0969e-01 |    2.564 |
| L-BFGS       | -0.705 | 1.0977e-01 |    2.117 |

**The Anderson oracle (`QQN-And`) leads the AUC leaderboard** (`-0.84`),
edging out Adam (`-0.82`) — a striking result for a Hessian-free, `(m × m)`
least-squares curvature estimate. The matrix-free **secant oracle (`QQN-Sec`)
also ranks highly** (`-0.74`). Note that the spline+adaptive-TR stalls
(`QQN-L50SplnTR`, `QQN-Best`) post deceptively good AUCs (`-0.76`) precisely
*because* the spline's monotone gating keeps their trajectories low even
though they never cross the final target — a reminder to read AUC alongside
iterations-to-target.

### Convergence-Rate Profile

The milestone profile crisply separates early- from late-phase descent. The
fastest variants to reach the tightest `1.2e-1` milestone are the
**deep-memory + spline** combos:

| Variant      | `≤5.0e-1` | `≤2.0e-1` | `≤1.5e-1` | `≤1.2e-1` |
|--------------|----------:|----------:|----------:|----------:|
| QQN-L50Spln  |         7 |        21 |        29 |        37 |
| QQN-L100Spln |         7 |        21 |        29 |        37 |
| QQN-L50And   |         7 |        22 |        30 |        38 |
| QQN-L50      |         7 |        22 |        30 |        38 |
| QQN-L100     |         7 |        22 |        30 |        38 |
| QQN-L50TRfix |         7 |        22 |        31 |        40 |
| QQN-L20      |         7 |        22 |        33 |        46 |
| QQN (L10)    |         7 |        24 |        35 |        52 |
| L-BFGS       |         7 |        25 |        37 |        54 |
| Adam         |         8 |        42 |        77 |       167 |
| QQN-Sec      |        15 |       104 |       185 |       240 |

The profile makes the deep-memory advantage stark: `QQN-L50Spln` crosses the
`1.2e-1` milestone at iteration **37**, while the L10 baseline takes 52,
L-BFGS takes 54, and Adam takes 167. The momentum oracles, SGD, and **all
bare adaptive-trust-region stacks** never reach even the `1.5e-1` milestone
(and most never cross `2.0e-1`); the spline+adaptive-TR stalls (`QQN-Best`,
`QQN-L50SplnTR`) cross `2.0e-1` (at iteration 22) but then plateau before
`1.5e-1`.

## Stall Report

The experiment explicitly surfaces every variant that exhausted its budget
**without** reaching the shared target, with a classified cause. This makes the
cautionary results a first-class, scannable finding:

| Variant       | Final loss | Iters | Time (s) | Cause                    |
|---------------|-----------:|------:|---------:|--------------------------|
| QQN-L50SplnTR | 1.6528e-01 |   500 |    7.965 | slow (no target)         |
| QQN-Best      | 1.6528e-01 |   500 |    8.312 | slow (no target)         |
| QQN-Mom01     | 1.8918e-01 |   500 |    5.427 | slow (no target)         |
| QQN-Mom10     | 1.9405e-01 |   500 |    5.491 | slow (no target)         |
| QQN-SplnTR    | 2.0387e-01 |   500 |    7.725 | slow (no target)         |
| QQN-Mom50     | 2.2652e-01 |   500 |    5.534 | slow (no target)         |
| SGD           | 2.2664e-01 |   500 |    0.577 | slow (no target)         |
| QQN-Mom       | 3.4187e-01 |   500 |    5.480 | slow (no target)         |
| QQN-SW        | 4.0768e-01 |   500 |    7.640 | stalled (plateau)        |
| QQN-L50TR2    | 6.2564e-01 |   500 |    6.879 | stalled (plateau)        |
| QQN-TR2       | 6.2564e-01 |   500 |    6.370 | stalled (plateau)        |
| QQN-L50Sec    | 6.2716e-01 |   500 |    7.162 | stalled (plateau)        |
| QQN-TR        | 6.2716e-01 |   500 |    6.576 | stalled (plateau)        |
| QQN-Seq       | 6.2716e-01 |   500 |    6.555 | stalled (plateau)        |
| QQN-L50TR     | 6.2716e-01 |   500 |    6.981 | stalled (plateau)        |
| QQN-L100TR    | 6.2716e-01 |   500 |    7.470 | stalled (plateau)        |
| QQN-L50BTTR   | 6.2716e-01 |   500 |    7.068 | stalled (plateau)        |
| QQN-SW+TR     | 8.7783e-01 |   500 |    7.751 | stalled (plateau)        |
| QQN-Sh        | 8.8827e-01 |     9 |   16.338 | time-budget exhausted    |
| QQN-TR025     | 1.9828e+00 |   500 |    6.826 | stalled (plateau)        |

The plateau cluster at `6.272e-01` is the signature of the **adaptive
trust-region stall**: every variant that wraps a bare adaptive `TrustRegion`
(`QQN-TR`, `QQN-Seq`, `QQN-L50TR`, `QQN-L100TR`, `QQN-L50BTTR`, `QQN-L50Sec`)
converges to the *same* plateau, confirming the cause is the region, not the
oracle or search. Shampoo is the sole **time-budget** casualty.

## Combinator and Constraint Variants

The experiment also exercises the combinator oracles and regions to confirm
they run correctly and produce sensible behavior:

| Variant    | Configuration                    | Final loss | Sparsity | ->target |
|------------|----------------------------------|-----------:|---------:|---------:|
| QQN-Fall   | Fallback([L-BFGS(10), Momentum]) |  1.096e-01 |   0.0000 |       65 |
| QQN-L50And | Fallback([L-BFGS(50), Anderson]) |  1.097e-01 |   0.0000 |       43 |
| QQN-Box    | BoxRegion(-2, 2)                 |  1.098e-01 |   0.0000 |       65 |
| QQN-Orth   | OrthantRegion (OWL-QN-style)     |  1.100e-01 |   0.0027 |       70 |
| QQN-L20Box | L-BFGS(20) + BoxRegion(-2, 2)    |  1.100e-01 |   0.0000 |       60 |
| QQN-L50Sec | Fallback([L-BFGS(50), Secant])+TR|  6.272e-01 |   0.0000 |        — |

- **Fallback** reproduces the L-BFGS baseline exactly here (`QQN-Fall`
  1.096e-01 at iteration 65; `QQN-L50And` 1.097e-01 at iteration 43), because
  the L-BFGS direction is always a valid descent direction, so the secondary
  oracle never triggers.
- **OrthantRegion** is the only configuration to induce measurable weight
  sparsity (0.0027), as expected from its sign-preserving projection.
- The **box** and **L20+box** constraints add negligible cost while keeping
  weights bounded; the deeper L20 oracle lets `QQN-L20Box` reach the target
  in 60 iterations, ahead of the shallow box variant (65).
- **`QQN-L50Sec`** (a `Fallback([L-BFGS(50), Secant])` paired with an adaptive
  `TrustRegion`) **stalls** at `6.272e-01` — the oracle fallback is sound, but
  it again inherits the adaptive-trust-region instability documented above.

## Loss Trajectories

The log records a compact log10-scale, sampled view of every trajectory. The
qualitative picture (over 10 sampled points across the run):

- **QQN (and most L-BFGS-10 variants)** drop from `0.36` to roughly `-0.96`
  in log10 loss by the end of the run.
- The **deep-memory + spline combos** (`QQN-L50Spln`, `QQN-L100Spln`) and the
  **warm-started fixed-TR combos** (`QQN-Fast`, `QQN-L50WS`, `QQN-L50Endpt`,
  `QQN-Champion`) match the deepest trajectory, reaching `-0.96` by the end.
- **Adam** descends fastest in log10 terms, reaching `-0.95` by the eighth
  sample and `-0.96` by the end (it just needs many cheap iterations).
- The **Anderson oracle** (`QQN-And`) reaches `-0.66` by its second sample and
  `-0.96` by the end — the deepest *early* descent of any variant — while the
  **secant oracle** (`QQN-Sec`) reaches `-0.96` by the end despite needing
  many iterations to formally cross the target.
- **SGD and the heavier momentum oracles** plateau between `-0.47` and
  `-0.64` (lighter momentum `QQN-Mom01`/`QQN-Mom10` reaching `-0.71`/`-0.72`).
- **The bare adaptive-trust-region stacks** (`QQN-TR`, `QQN-Seq`, `QQN-L50TR`,
  `QQN-L100TR`, `QQN-L50BTTR`, `QQN-L50Sec`) flatline early at `-0.20`
  (loss `6.272e-01`); the spline+adaptive-TR stalls (`QQN-SplnTR`,
  `QQN-L50SplnTR`, `QQN-Best`) plateau between `-0.69` and `-0.78`.
- **Shampoo** (`QQN-Sh`) barely moves before exhausting the time budget,
  reaching only `-0.05` after 9 iterations.

## Key Takeaways

1. **QQN converges to the shared target in fewer iterations than L-BFGS** on a
   smooth, deterministic full-batch problem, at a fraction of the wall-time,
   and clearly outperforms SGD (which never reaches the target). Adam reaches
   the target but needs ~4× the iterations of the quasi-Newton methods.
2. **L-BFGS history depth is the dominant convergence-speed lever**, cutting
   iterations-to-target from 80 (L5) to 43 (L50), with diminishing returns
   past size 50 and a hard plateau at 100 (`L50 == L100` at 43 iterations —
   a **1.63× iteration speedup over L-BFGS**).
3. **The line search choice trades wall-time, not convergence speed**, within
   the backtracking/Armijo family — backtracking/Armijo is the clear
   efficiency winner. The cubic Hermite spline matches their
   iterations-to-target while reaching a slightly *lower* loss, at ~2× the
   time. **Strong-Wolfe fails to converge** on this problem (its tight
   curvature condition over-restricts the path step).
4. **The spline refinement composes with any line search** (it wraps the inner
   search rather than replacing it) and achieves a low loss (`1.091e-01` for
   `QQN-L50Spln` at 43 iterations). However, under the current honest `pred`
   model its monotone gating is **no longer enough to rescue the adaptive
   trust-region**: `QQN-SplnTR` and `QQN-L50SplnTR` now stall.
5. **The adaptive trust-region is the single most fragile component here.**
   Under the current honest predicted-reduction model, *every* bare adaptive
   trust-region stack (`QQN-TR`, `QQN-Seq`, `QQN-L50TR`, `QQN-L100TR`,
   `QQN-L50BTTR`, `QQN-L50Sec`, and the spline variants) **stalls at
   `6.27e-01` (or `≤2e-1`) and never reaches the target**. A *fixed* radius
   (`QQN-TRfix` 68, `QQN-L50TRfix` 46) and the warm-started fixed-TR stacks
   (`QQN-Fast`, `QQN-L50WS+`, `QQN-Champion`) converge cleanly. On this class
   of well-conditioned smooth problem, prefer no region (or a fixed radius /
   box / orthant) over an adaptive trust-region.
6. **Hessian-free curvature oracles are remarkably strong on the AUC axis**:
   the **Anderson** oracle (`QQN-And`, `(m × m)` residual solve) posts the
   **best trajectory AUC of any method (`-0.84`)**, beating Adam, and the
   matrix-free **secant** oracle (`QQN-Sec`, `-0.74`) is close behind — both
   reach the target (269 and 311 iterations respectively) at `O(n)`–`O(m²n)`
   memory and no Hessian.
7. **Oracle choice matters more than search or region**: deep L-BFGS dominates
   iterations-to-target, the secant/Anderson oracles trail in iterations but
   lead in AUC, momentum never converges, and the blocked Shampoo oracle does
   not scale to this high-dimensional problem within the time budget.
8. **The convergence-rate milestone profile confirms the deep-memory edge**:
   the L50/L100 (+spline) combos cross the tight `1.2e-1` milestone at
   iteration 37–38 — far ahead of the L10 baseline (52), L-BFGS (54), and Adam
   (167).
9. **The lowest loss observed overall (`1.090e-01`) is `QQN-Fast`** — deep
   L100 memory + warm-started backtracking + a *fixed* trust-region — which
   also sits on the Pareto frontier, demonstrating that the robust fast path
   (deep memory + warm-start + fixed radius) wins on *both* loss and time.

## Reproducing

```bash
pip install -e ".[dev]"
python examples/mnist_comparison.py
```

The script prints the summary table (including `ms/it` per-iteration cost,
the `->target` iteration and `t->tgt` wall-clock time at which the shared
loss/gradient target was first hit, the `vs LBFGS` iteration speedup, and the
`AUC` trajectory integral), the Pareto frontier of non-dominated variants, the
iteration-efficiency leaderboard (converging variants ranked by fewest
iterations-to-target), the trajectory-AUC leaderboard, the convergence-rate
profile (first iteration crossing each loss milestone), an explicit stall
report (variants that never reached the target, with classified causes),
sampled log10 trajectories, and the controlled A/B comparison report. It also
saves both a `mnist_comparison.png` (loss vs. iteration) and a
`mnist_comparison_time.png` (loss vs. wall-clock time) convergence plot when
`matplotlib` is available.