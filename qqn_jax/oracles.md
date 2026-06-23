# `qqn_jax/oracles.py` — Oracle Abstraction and Implementations

## Overview

The **oracle** is the component that supplies the `t = 1` endpoint of the
QQN quadratic interpolation path:

```
d(t) = t(1 - t)·(-∇f) + t²·(-H∇f)
```

At `t = 1` the path reaches `-H∇f` — the curvature-aware (or otherwise
accelerated) direction. The oracle's job is to compute this endpoint given
the current iterate, gradient, and any accumulated state (e.g. curvature
history).

Every oracle is a **pure, functional JAX object** — no Python-side mutation,
no hidden state. All oracles compose with `jit`, `vmap`, `pmap`, and `grad`.
Oracles operate on **flat** parameter / gradient vectors, consistent with the
rest of `qqn-jax`.

---

## The `Oracle` Interface

```python
class Oracle(NamedTuple):
    init: Callable[[Any], Any]
    direction: Callable[[Any, Any, Any], Tuple[Any, Any]]
    update: Callable[[Any, Any], Any]
```

| Method      | Signature                                     | Description                                                                                                                             |
|-------------|-----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `init`      | `params -> oracle_state`                      | Initialise oracle state at the starting iterate. Use `()` for stateless oracles.                                                        |
| `direction` | `(params, grad, state) -> (direction, state)` | Compute the `t = 1` endpoint `-H∇f`. The returned state is **discarded** by the solver; only `update` persists state across iterations. |
| `update`    | `(state, info: OracleInfo) -> state`          | Commit a new curvature pair (or other information) after a step is accepted. No-op for stateless oracles.                               |

> **Important:** The solver calls `direction` to obtain the endpoint, then
> calls `update` after the step is accepted. The state returned by `direction`
> is **not** persisted — only the state returned by `update` carries forward.
> This separation keeps `direction` a pure read of the current state and
> concentrates all mutation in `update`.

---

## `OracleInfo`

```python
class OracleInfo(NamedTuple):
    params: Any = None  # iterate x before the step
    new_params: Any = None  # accepted iterate x_new
    grad: Any = None  # gradient ∇f(x) before the step
    new_grad: Any = None  # gradient ∇f(x_new) after the step
    t: Any = None  # chosen interpolation parameter
    step_size: Any = None  # accepted step size α (path parameter t)
    probe_params: Any = None  # optional (k, n) buffer of line-search probe points
    probe_grads: Any = None  # optional (k, n) buffer of probe gradients
    probe_valid: Any = None  # optional (k,) boolean mask of filled probe slots
    probe_alphas: Any = None  # optional (k,) step sizes for each probe
```

`OracleInfo` is passed to `Oracle.update` after each accepted step. The
`probe_*` fields are populated when `feed_probes_to_oracle=True` is set on
the solver, allowing the oracle to incorporate every gradient evaluated
during the line search — not just the accepted endpoint.

---

## Concrete Oracle Implementations

### `LBFGSOracle` (default)

```python
LBFGSOracle(history_size: int = 10) -> Oracle
```

**Limited-memory BFGS quasi-Newton oracle.** Wraps the `qqn_jax.lbfgs`
two-loop recursion. This is the default oracle and reproduces the original
QQN behavior byte-for-byte.

**Direction:** The standard L-BFGS two-loop recursion applied to the current
gradient. On the very first step (empty history) this reduces to `-∇f`.

**State:** An L-BFGS history buffer of `(s, y)` curvature pairs, where
`s = x_new - x` and `y = ∇f(x_new) - ∇f(x)`.

**Probe replay:** When `probe_params`, `probe_alphas`, and `probe_valid` are
all present in `OracleInfo`, the oracle replays line-search probes into the
curvature history in **increasing-α order** before committing the accepted
point as the newest pair. Probes are sorted by α because they are collinear
(all lie on the single ray `x + α·d`); monotone spacing ensures secant
differences are consistently oriented and carry meaningful 1-D curvature.
When any of the three probe fields is absent (e.g. from the spline-wrapped
line search), the oracle falls back to the single-pair update.

```python
from qqn_jax.oracles import LBFGSOracle

oracle = LBFGSOracle(history_size=20)
state = oracle.init(params)
d, _ = oracle.direction(params, grad, state)  # -H∇f
state = oracle.update(state, info)
```

| Parameter      | Default | Description                        |
|----------------|---------|------------------------------------|
| `history_size` | `10`    | Number of `(s, y)` pairs to store. |

---

### `MomentumOracle`

```python
MomentumOracle(beta: float = 0.9) -> Oracle
```

**Heavy-ball / exponentially-weighted momentum oracle.**

The `t = 1` endpoint blends the current steepest-descent move with a
decaying-weight average of the **actual per-iteration deltas**
`Δx = x_new − x` that the solver has already realized:

```
# committed in update, after each accepted step:
v_new     = β · v + (1 − β) · Δx        Δx = x_new − x

# returned by direction at the current iterate:
direction = -∇f + β · v
```

The velocity `v` tracks the direction the optimizer has actually been
travelling — true heavy-ball momentum — rather than an average of raw
gradients. On the very first step `v = 0` and the endpoint reduces to plain
steepest descent, preserving the `d'(0)` anchor.

**State:** `MomentumState(velocity)` — the running EWA of realized deltas.

```python
from qqn_jax.oracles import MomentumOracle

oracle = MomentumOracle(beta=0.9)
```

| Parameter | Default | Description                                  |
|-----------|---------|----------------------------------------------|
| `beta`    | `0.9`   | Exponential decay rate for the velocity EWA. |

---

### `SecantOracle`

```python
SecantOracle(alpha0: float = 1.0, alpha_max: float = 1e3) -> Oracle
```

**Barzilai-Borwein curvature oracle — matrix-free, O(n) memory.**

The `t = 1` endpoint is the gradient scaled by an inverse-curvature estimate
inferred from the *realized* secant of the previous step:

```
s = x      - x_prev
y = ∇f     - ∇f_prev
α = ⟨s, s⟩ / ⟨s, y⟩        (BB1 step; Rayleigh quotient's inverse)
direction = -α · ∇f
```

The very first step (no secant yet) falls back to `-alpha0 · ∇f`, i.e.
plain scaled steepest descent, preserving the `d'(0)` anchor.

**Curvature guard:** When `⟨s, y⟩ ≤ ε` (non-positive curvature), the prior
`α` is retained unchanged. This prevents degenerate or sign-flipping
curvature estimates from corrupting the scale.

**State:** `SecantState(prev_params, prev_grad, alpha, step_count)`.

```python
from qqn_jax.oracles import SecantOracle

oracle = SecantOracle(alpha0=1.0, alpha_max=1e3)
```

| Parameter   | Default | Description                                       |
|-------------|---------|---------------------------------------------------|
| `alpha0`    | `1.0`   | Initial inverse-curvature scale (used on step 0). |
| `alpha_max` | `1e3`   | Clamp on the BB step to prevent runaway scaling.  |

---

### `ShampooOracle`

```python
ShampooOracle(
    block_size: int = 128,
update_freq: int = 20,
epsilon:     float = 1e-6,
) -> Oracle
```

**Structure-aware preconditioned oracle (Shampoo).**

Operates on the flat parameter vector by treating the gradient `g` (shape
`(n,)`) as a column and preconditioning via accumulated second-moment
statistics. The inverse roots are recomputed on a fixed static cadence
(`update_freq`) so the per-step cost stays amortized and the whole
computation remains `jit`-friendly.

**Direction (on a refresh step):**

```
L_new  = L + g gᵀ
R_new  = R + ‖g‖²          (scalar, stored as (1,1))
d      = -(L_new^{-1/4} g) R_new^{-1/4}
```

**Direction (on a non-refresh step):** Falls back to `-grad` (scaled
steepest descent) to avoid paying the `O(n²)` eigh cost every step.

**State:** `ShampooState(L, R, step)` — the accumulated left/right
second-moment matrices and the step counter.

```python
from qqn_jax.oracles import ShampooOracle

oracle = ShampooOracle(update_freq=20, epsilon=1e-6)
```

| Parameter     | Default | Description                                                      |
|---------------|---------|------------------------------------------------------------------|
| `block_size`  | `128`   | Reserved for future block-diagonal extension (unused currently). |
| `update_freq` | `20`    | Cadence (in steps) at which inverse roots are recomputed.        |
| `epsilon`     | `1e-6`  | Tikhonov regularizer added to second-moment matrices.            |

> **Note:** The current implementation treats the entire flat parameter
> vector as a single block. The `block_size` parameter is reserved for a
> future block-diagonal extension.

---

### `AndersonOracle`

```python
AndersonOracle(
    window: int = 5,
reg:    float = 1e-8,
beta:   float = 1.0,
) -> Oracle
```

**Anderson-accelerated (Type-II) oracle** — the variational ideal that
L-BFGS approximates.

The `t = 1` endpoint is formed by solving a tiny constrained least-squares
problem over recent gradient *differences*:

```
min_θ  ‖ ∇f − ΔG θ ‖²  (+  reg · ‖θ‖²)
direction = −β·( ∇f − ΔG θ )  −  ΔX θ
```

where `ΔG`, `ΔX` are first-differences of the stored gradient/iterate
windows. With `window=1` this reduces to a secant step; with a deep window
it captures multi-step curvature the single-secant cannot. No Hessian is
ever formed; the only solve is an `(m × m)` system.

**Regularization:** The Tikhonov regularizer is scaled to the Gram trace
(`reg * trace(ΔGᵀΔG) / m`) so conditioning is invariant to the magnitude of
the residual window. An absolute diagonal ridge (`1e-12 · I`) is also added
to guarantee SPD-ness and prevent NaN backpropagation through degenerate
windows.

**Safeguard:** If the solve produces a non-finite direction, or if no history
has been accumulated yet (`step_count == 0`), the oracle falls back to
`-grad` (steepest descent).

**Coupling constant `β`:** Rescales the accelerated residual toward the
gradient's natural magnitude. `β = 1` recovers the pure Type-II update;
`β > 1` lets the deep-residual descent stretch.

**State:** `AndersonState(g_history, x_history, step_count)` — rolling
windows of recent gradients and iterates, plus a count of valid entries.

```python
from qqn_jax.oracles import AndersonOracle

oracle = AndersonOracle(window=8, beta=1.5)
```

| Parameter | Default | Description                                                  |
|-----------|---------|--------------------------------------------------------------|
| `window`  | `5`     | Number of recent (gradient, iterate) pairs to retain.        |
| `reg`     | `1e-8`  | Tikhonov regularization coefficient for the `(m × m)` solve. |
| `beta`    | `1.0`   | Mixing / coupling constant for the accelerated residual.     |

---

## The `Fallback` Combinator

```python
Fallback(oracles: Sequence[Oracle]) -> Oracle
```

**Use the first oracle's direction when valid; fall back to the next.**

Validity is defined as: the direction is **finite**, **non-zero**, and a
**descent direction** (`⟨∇f, d⟩ < 0`). All selection uses `jnp.where` /
`lax.select` — no Python conditionals — so the combinator is fully
`jit`-compatible.

**Priority:** The first oracle in the sequence has highest priority. If its
direction is valid, it is used. Otherwise the second oracle is tried, and so
on. If *every* oracle produces an invalid direction, the combinator falls
back to steepest descent (`-∇f`) as a terminal safety net.

**Why descent, not just finite?** A finite, non-zero quasi-Newton direction
that points uphill (`⟨∇f, d⟩ ≥ 0`) is worse than useless — it betrays a
degenerate curvature estimate. The fallback must trigger on misalignment, not
just on collapse.

**State:** A tuple of the child oracle states, in the same order as the
`oracles` argument. `update` fans out to all children.

```python
from qqn_jax.oracles import Fallback, LBFGSOracle, SecantOracle

# Deep curvature, with a featherweight backup for degenerate history.
oracle = Fallback([LBFGSOracle(history_size=10), SecantOracle()])
```

---

## `resolve_oracle`

```python
resolve_oracle(oracle, history_size: int = 10) -> Oracle
```

Maps a string shortcut or `Oracle` instance to a concrete oracle. Used
internally by the `QQN` solver to resolve the `oracle=` constructor argument.

| Input                | Resolved oracle                                           |
|----------------------|-----------------------------------------------------------|
| `None` or `"lbfgs"`  | `LBFGSOracle(history_size=history_size)`                  |
| `"momentum"`         | `MomentumOracle()`                                        |
| `"shampoo"`          | `ShampooOracle()`                                         |
| `"secant"`           | `SecantOracle()`                                          |
| `"anderson"`         | `AndersonOracle()`                                        |
| `"anderson+secant"`  | `Fallback([AndersonOracle(window=5), SecantOracle()])`    |
| `"lbfgs+secant"`     | `Fallback([LBFGSOracle(history_size=…), SecantOracle()])` |
| An `Oracle` instance | Returned as-is (passthrough).                             |
| Unknown string       | Raises `ValueError`.                                      |
| Any other type       | Raises `TypeError`.                                       |

```python
from qqn_jax.oracles import resolve_oracle

oracle = resolve_oracle("lbfgs+secant", history_size=20)
```

---

## Named Oracle Combinations

Two pre-built `Fallback` combinations are available by name:

### `"anderson+secant"`

```python
Fallback([AndersonOracle(window=5), SecantOracle()])
```

The variational ideal (Anderson), safeguarded by a featherweight secant.
A strictly-dominant pairing when the residual solve degenerates: Anderson
captures multi-step curvature when healthy; Secant provides a finite
curvature estimate the instant the window collapses.

### `"lbfgs+secant"`

```python
Fallback([LBFGSOracle(history_size=history_size), SecantOracle()])
```

Deep curvature while healthy, finite curvature the instant the history
collapses. The recommended safeguarded default for production use.

---

## NaN Safety

All oracles are designed to be NaN-safe under `jax.grad`:

- **L-BFGS:** Curvature reciprocals are guarded so masked-out branches never
  backpropagate NaNs.
- **Secant:** Non-positive curvature (`⟨s, y⟩ ≤ ε`) retains the prior `α`
  rather than dividing by near-zero.
- **Anderson:** The `(m × m)` solve is regularized with both a scaled
  Tikhonov term and an absolute diagonal ridge; non-finite results trigger
  the steepest-descent safeguard.
- **Shampoo:** `_matrix_inverse_pth_root` adds `ε·I` before `eigh` and
  clamps eigenvalues to `[ε, ∞)`.
- **Fallback:** The terminal safety net (`-∇f`) ensures the path's `t = 1`
  endpoint is always a valid descent direction, even when every child oracle
  degenerates simultaneously.

---

## Implementing a Custom Oracle

Any `Oracle` NamedTuple with the correct signatures can be passed directly
to `QQN`:

```python
from qqn_jax.oracles import Oracle, OracleInfo
import jax.numpy as jnp


def MyOracle(scale: float = 1.0) -> Oracle:
    """Scaled steepest descent — the simplest possible oracle."""

    def init(params):
        return ()  # stateless

    def direction(params, grad, state):
        return -scale * grad, state

    def update(state, info: OracleInfo):
        return state  # no-op

    return Oracle(init=init, direction=direction, update=update)


from qqn_jax import QQN

solver = QQN(fun, oracle=MyOracle(scale=0.5))
```

**Requirements for a well-behaved oracle:**

1. `direction` must return a **descent direction** (`⟨∇f, d⟩ < 0`) whenever
   possible. The `Fallback` combinator and the solver's line search both
   assume this.
2. `direction` must be **pure** — it must not mutate `state`. All state
   updates go through `update`.
3. All operations must be **JAX-traceable** (no Python-side data-dependent
   control flow on traced values).
4. The oracle must handle the **first step** gracefully (empty history,
   zero velocity, etc.) — typically by falling back to `-∇f`.

---

## State Types

| Oracle           | State type                                                   |
|------------------|--------------------------------------------------------------|
| `LBFGSOracle`    | `LBFGSState` (from `qqn_jax.lbfgs`)                          |
| `MomentumOracle` | `MomentumState(velocity)`                                    |
| `SecantOracle`   | `SecantState(prev_params, prev_grad, alpha, step_count)`     |
| `ShampooOracle`  | `ShampooState(L, R, step)`                                   |
| `AndersonOracle` | `AndersonState(g_history, x_history, step_count)`            |
| `Fallback`       | `tuple` of child oracle states (same order as `oracles` arg) |

---

## Public API

```python
from qqn_jax.oracles import (
    Oracle,
    OracleInfo,
    LBFGSOracle,
    MomentumOracle,
    ShampooOracle,
    SecantOracle,
    AndersonOracle,
    Fallback,
    resolve_oracle,
)
```

---

## See Also

- `qqn_jax/lbfgs.py` — L-BFGS two-loop recursion and curvature-history buffers.
- `qqn_jax/solver.py` — The `QQN` optimizer; shows how oracles are called.
- `qqn_jax/line_search.py` — Line-search strategies; probe feeding populates
  `OracleInfo.probe_*` fields.
- `tests/test_oracles.py` — Comprehensive unit and integration tests for all
  oracle implementations.