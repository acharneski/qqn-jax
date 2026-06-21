# Oracle Abstraction

## Overview

In QQN, the **oracle** is the component that supplies the second search
direction blended along the quadratic path

```
d(t) = t(1 - t)(-∇f) + t²(-H∇f),   t ∈ [0, 1]
```

The default oracle is **L-BFGS**, which approximates `-H∇f` (the
quasi-Newton direction) from a limited history of gradient differences.
But L-BFGS is only *one* possible oracle. Conceptually, an oracle is any
black-box that, given the current gradient (and optionally some state),
returns a candidate direction that QQN can blend against steepest descent.

This document specifies the oracle abstraction and the concrete oracles to
be implemented as pure, functional JAX so they compose with `jit`, `vmap`,
`pmap`, and `grad`, consistent with the rest of `qqn-jax`.

---

## Conceptual Role

Recall QQN's three components (see [`algorithm.md`](algorithm.md)):

1. **Gradient** — steepest descent `-∇f(x)`.
2. **Oracle** — a curvature-aware direction `-H∇f(x)` (or any learned/
accelerated direction).
3. **Search** — the line search navigating `d(t)`.

The oracle provides the `t = 1` endpoint of the quadratic path. Because the
line search always retains access to the pure gradient direction at `t = 0`,
the oracle does **not** need to guarantee descent on its own: QQN's
convergence is anchored by the steepest-descent fallback, and the oracle is
free to be aggressive. This makes the oracle a natural extension point.

---

## Goals

* Add an optional, swappable `oracle` configuration to `qqn(...)` and
`QQN(...)`.
* Keep every oracle a pure function with no host-side control flow.
* Preserve QQN's convergence behavior: when `oracle="lbfgs"` (the default),
behavior is identical to the current implementation.
* Make oracles independent of the gradient/search/region components so they
can be combined and substituted freely.

## Non-Goals

* Oracles requiring per-step inner optimization (beyond cheap closed-form
updates), except where explicitly noted.
* Replacing the line search: the oracle only *proposes* a direction; the
search still selects `t` and `α`.

---

## Core Abstraction

An oracle is described by a small, pure interface. All functions are
JAX-traceable and operate on pytrees of parameters.

```python
class Oracle(NamedTuple):
  # Optional per-oracle state (e.g. L-BFGS history, momentum buffer).
  # Use an empty pytree () when no state is needed.
  init: Callable[[Params], OracleState]

  # Produce the oracle direction -H∇f at the current point.
  #   params:   current iterate x (pytree)
  #   grad:     current gradient ∇f(x) (pytree)
  #   state:    oracle state
  # returns (direction, new_state). `direction` has the same structure
  # as params and represents the t = 1 endpoint of d(t).
  direction: Callable[[Params, Params, OracleState],
                      Tuple[Params, OracleState]]

  # Optional update of oracle state after a step is accepted.
  #   Used by history-based oracles (e.g. L-BFGS curvature pairs).
  update: Callable[[OracleState, OracleInfo], OracleState]
```

`OracleState` and `OracleInfo` are oracle-specific pytrees. `OracleInfo`
carries quantities the outer loop already computes (e.g. the accepted step
`s = x_new − x`, the gradient difference `y = ∇f(x_new) − ∇f(x)`, `t`, `α`).

### Integration with the QQN path

At each iteration QQN queries the oracle for the quasi-Newton endpoint and
builds the quadratic path:

```python
def path_d(oracle, state, params, grad, t):
  neg_grad = tree_neg(grad)                       # -∇f
  oracle_dir, _ = oracle.direction(params, grad, state)  # -H∇f
  # d(t) = t(1 - t)(-∇f) + t²(-H∇f)
  return tree_add(
      tree_scale(neg_grad, t * (1.0 - t)),
      tree_scale(oracle_dir, t * t),
  )
```

The default L-BFGS oracle reproduces the current behavior exactly.

---

## Oracles to Implement

### 1. L-BFGS Oracle (default)

The limited-memory BFGS quasi-Newton oracle. Approximates `-H∇f` from the
most recent `m` curvature pairs `(sₖ, yₖ)` via the two-loop recursion.

* **State**: ring buffers of `s`/`y` pairs and the rolling scale `γ`.
* **Direction**: Optax `scale_by_lbfgs` applied to the gradient.
* **Update**: push the new `(s, y) = (x_new − x, ∇f_new − ∇f)` pair,
skipping updates that violate the curvature condition `⟨s, y⟩ > 0`.
* **Config**: `LBFGSOracle(history_size=10)`.
* **Notes**: This is the reference oracle; `oracle="lbfgs"` selects it and
is byte-for-byte equivalent to the current optimizer.

### 2. Momentum Oracle

A first-order accelerated direction. Instead of curvature, the oracle blends
in an exponentially-weighted history of past gradients.

* **State**: `velocity` (pytree, same structure as params).
* **Direction**:

```
v_new = β · v + (1 − β) · ∇f
direction = -v_new          # the t = 1 endpoint
```

* **Update**: store `v_new` (or update lazily inside `direction` and commit
on accept).
* **Config**: `MomentumOracle(beta=0.9)`.
* **Notes**: Pure pytree arithmetic; trivially `vmap`/`jit`-able. Gives QQN
a heavy-ball flavor at `t = 1` while retaining the gradient at `t = 0`.

### 3. Shampoo Oracle

A structure-aware preconditioner that maintains per-dimension second-moment
statistics (Kronecker-factored for matrix-shaped parameters) and applies an
inverse-root preconditioner to the gradient.

* **State**: accumulated statistics `L`, `R` per parameter tensor and a step
counter for inverse-root refresh.
* **Direction**:

```
L += G Gᵀ ;  R += Gᵀ G
direction = -(L^{-1/4}) G (R^{-1/4})
```

with a fixed, static refresh cadence for the inverse roots so the cost is
amortized and the computation stays `jit`-friendly.
* **Config**: `ShampooOracle(block_size=128, update_freq=20, epsilon=1e-6)`.
* **Notes**: More expensive than L-BFGS per step, but can capture richer
curvature structure on layered models.

### 4. Combinator Oracles

Compose or fall back between oracles.

* **`Fallback([O1, O2, ...])`**: Use `O1`'s direction when it is valid
(e.g. L-BFGS with a positive-curvature history), otherwise fall back to the
next oracle. Validity is expressed via `jnp.where`/`lax.select`, never
Python conditionals. State is a tuple of child states.
* **`Blend([(w1, O1), (w2, O2), ...])`** *(stretch)*: A fixed convex
combination of multiple oracle directions, e.g. mix momentum into L-BFGS.

Combinators must preserve the pure-function contract and a fixed (static)
structure so they remain `jit`-friendly.

---

## Public API

```python
qqn(
  history_size=10,
  line_search="strong_wolfe",
  t_grid=None,
  oracle="lbfgs",             # "lbfgs" | "momentum" | "shampoo" | Oracle
  region=None,
)

QQN(
  fun,
  maxiter=100,
  tol=1e-5,
  history_size=10,
  line_search="strong_wolfe",
  has_aux=False,
  t_grid=None,
  oracle="lbfgs",             # "lbfgs" | "momentum" | "shampoo" | Oracle
  region=None,
)
```

String shortcuts map to the default-configured concrete oracles; an explicit
`Oracle` instance overrides them for full control.

Convenience constructors:

```python
from qqn_jax.oracles import (
  LBFGSOracle, MomentumOracle, ShampooOracle, Fallback,
)

oracle = Fallback([
  LBFGSOracle(history_size=10),
  MomentumOracle(beta=0.9),
])

solver = QQN(fun, oracle=oracle)
```

When `oracle="lbfgs"` (the default), the optimizer is byte-for-byte
equivalent to the current behavior.

---

## Implementation Plan

1. **`oracles.py`**: Define the `Oracle` NamedTuple, the L-BFGS oracle
wrapping Optax's `scale_by_lbfgs`, and `direction`/`init`/`update`
helpers operating on pytrees (`jax.tree_util`).
2. **Wire into the solver** (`solver.py`): replace the direct
`scale_by_lbfgs` call with `oracle.direction(...)` when building `d(t)`.
Keep the L-BFGS path zero-overhead so the default is unchanged.
3. **Thread `OracleState`** through `QQNState`/`solver.py` so history-based
oracles update their state via `oracle.update` after each accepted step.
4. **Implement concrete oracles**: L-BFGS, Momentum, then Shampoo.
5. **Combinators**: `Fallback`, then (optional) `Blend`.

### State threading

`QQNState` carries the oracle state (today this is the L-BFGS history).
`opt.init` calls `oracle.init(params)`; `opt.update` calls `oracle.update(...)`
with an `OracleInfo` assembled from line-search results (accepted step `s`,
gradient difference `y`, `t`, `α`). The default L-BFGS oracle keeps the
existing state layout, so nothing changes when the default is selected.

---

## Testing Strategy

* **Direction correctness** (unit, per oracle):
* L-BFGS: matches Optax `scale_by_lbfgs` on identical histories.
* Momentum: `direction == -((β·v) + (1−β)·∇f)`; velocity accumulates.
* Shampoo: preconditioner shapes match parameter shapes; inverse roots
refresh on schedule.
* **Default equivalence**: `oracle="lbfgs"` reproduces baseline trajectories
bit-for-bit on Rosenbrock.
* **Descent preservation**: regardless of oracle, the line search still
returns a step with `f(x_new) ≤ f(x)` (or rejects), since `t = 0` recovers
steepest descent. Verified on convex quadratics.
* **Combinator**: `Fallback([LBFGS, Momentum])` uses momentum exactly when
the L-BFGS curvature history is invalid.
* **Transform compatibility**: every oracle passes through `jit`, `vmap`
(batched starting points), and `grad` (differentiate through `solver.run`).

---

## Open Questions

* Should `OracleInfo` expose the *projected* step (post-region) or the raw
step for curvature updates? (Initial: use the accepted, projected step so
history reflects the feasible path.)
* For Shampoo, what is the right default `update_freq` to balance cost and
accuracy inside a `jit`-compiled loop? (Initial: static `20`.)
* How should `Blend` weights interact with the line search, given the search
already adapts the gradient/oracle mix via `t`? (Deferred; `Blend` is a
stretch goal.)