# `lbfgs.py` — L-BFGS Oracle

## Overview

`lbfgs.py` implements the **Limited-memory Broyden–Fletcher–Goldfarb–Shanno
(L-BFGS)** curvature oracle for QQN-JAX. It provides a self-contained,
circular-buffer curvature history and a direct implementation of the
**two-loop recursion** (Nocedal & Wright, Algorithm 7.4) that produces the
quasi-Newton direction `−H∇f`.

The module is designed to be:

- **JIT / vmap / pmap / grad compatible** — all operations are pure JAX,
  using `lax.scan` and `lax.while_loop`-safe primitives.
- **NaN-safe** — every reciprocal and matrix-free solve is guarded so that
  masked-out branches never backpropagate `NaN` under `jax.grad`.
- **Swappable** — the `LBFGSState` is a plain `NamedTuple`; the oracle is
  consumed by `QQN` through the generic `Oracle` interface in `oracles.py`.

---

## Data Structures

### `LBFGSState`

```python
class LBFGSState(NamedTuple):
    s_history: jnp.ndarray  # (history_size, n)  parameter differences
    y_history: jnp.ndarray  # (history_size, n)  gradient differences
    rho_history: jnp.ndarray  # (history_size,)    1 / (yᵀs)
    step_count: jnp.ndarray  # scalar int32        valid entries stored
    gamma: jnp.ndarray  # scalar float        H0 = gamma * I scale
    prev_params: jnp.ndarray  # (n,)               last accepted params
    prev_grad: jnp.ndarray  # (n,)               last accepted gradient
```

Buffers are stored **most-recent-first** (index `0` = newest pair). Unfilled
slots are zero-initialised; they contribute nothing to the two-loop recursion
because their `alpha` and correction terms vanish automatically.

| Field         | Shape               | Description                                         |
|---------------|---------------------|-----------------------------------------------------|
| `s_history`   | `(history_size, n)` | Circular buffer of `sₖ = xₖ₊₁ − xₖ`.                |
| `y_history`   | `(history_size, n)` | Circular buffer of `yₖ = ∇fₖ₊₁ − ∇fₖ`.              |
| `rho_history` | `(history_size,)`   | Circular buffer of `ρₖ = 1 / yₖᵀsₖ`.                |
| `step_count`  | scalar `int32`      | Number of valid pairs; saturates at `history_size`. |
| `gamma`       | scalar float        | Scaling for `H₀ = γI`; updated as `⟨y,s⟩/⟨y,y⟩`.    |
| `prev_params` | `(n,)`              | Anchor for the next `s` computation.                |
| `prev_grad`   | `(n,)`              | Anchor for the next `y` computation.                |

---

## Public API

### `init_lbfgs_state`

```python
def init_lbfgs_state(params, grad, history_size: int) -> LBFGSState
```

Allocates and returns a zero-filled `LBFGSState` for a parameter vector of
shape `(n,)`.

**Arguments**

| Name           | Type          | Description                               |
|----------------|---------------|-------------------------------------------|
| `params`       | `jnp.ndarray` | Initial parameter vector, shape `(n,)`.   |
| `grad`         | `jnp.ndarray` | Initial gradient, shape `(n,)`.           |
| `history_size` | `int`         | Maximum number of `(s, y)` pairs to keep. |

**Returns** — a fresh `LBFGSState` with all history buffers zeroed,
`step_count = 0`, `gamma = 1.0`, and `prev_params` / `prev_grad` set to the
supplied values.

---

### `update_lbfgs_history`

```python
def update_lbfgs_history(
        state: LBFGSState, params, grad, history_size: int
) -> LBFGSState
```

Pushes a new `(s, y)` curvature pair into the circular history buffer.

**Curvature guard.** The pair is admitted only when the *relative* curvature
condition

```
yᵀs  >  ε · ‖y‖₂ · ‖s‖₂        (ε = 1e-10)
```

is satisfied. This is stricter than the common absolute threshold `yᵀs > 1e-10`
and correctly rejects near-flat updates whose `yᵀs` is small only because the
step itself is tiny.

**NaN safety.** The reciprocal `ρ = 1 / yᵀs` is computed on a safe
denominator (`jnp.where(valid, ys, 1.0)`) before the guard is applied, so
the rejected branch never produces `inf` or `NaN` that could backpropagate
through `jax.grad`.

**Buffer management.** When the pair is admitted, the existing buffer is
shifted down by one row (oldest entry dropped) and the new pair is prepended
at index `0`. The shift uses `jnp.concatenate` rather than `jnp.roll` to
avoid an extra allocation, and the conditional update uses `jnp_select_buf`
(a scalar-flag whole-buffer select) rather than an element-wise `jnp.where`
over the full `(history_size, n)` array.

**Arguments**

| Name           | Type          | Description                                  |
|----------------|---------------|----------------------------------------------|
| `state`        | `LBFGSState`  | Current oracle state.                        |
| `params`       | `jnp.ndarray` | New accepted parameter vector, shape `(n,)`. |
| `grad`         | `jnp.ndarray` | Gradient at `params`, shape `(n,)`.          |
| `history_size` | `int`         | Buffer capacity (must match `state`).        |

**Returns** — updated `LBFGSState`. If the curvature guard rejects the pair,
the history buffers and `gamma` are unchanged; `prev_params` / `prev_grad`
are always advanced to the new values.

---

### `update_lbfgs_history_batch`

```python
def update_lbfgs_history_batch(
        state: LBFGSState,
        params_seq: jnp.ndarray,  # (k, n)
        grad_seq: jnp.ndarray,  # (k, n)
        valid_seq: jnp.ndarray,  # (k,)  bool
        history_size: int,
) -> LBFGSState
```

Replays a sequence of `k` probes into the history in a single `lax.scan`
pass. Probes are folded in **oldest-first** order so the most recent accepted
point ends up at index `0` (newest) in the buffer.

This is used by the **probe-feeding** feature (`feed_probes_to_oracle=True`
in `QQN`) to fold every gradient evaluated during the line search into the
oracle's curvature memory.

**Per-probe validity.** Each probe is gated by its entry in `valid_seq`
(e.g. unused scratch-buffer slots are marked `False`). The curvature guard
inside `update_lbfgs_history` provides a second layer of filtering, so
degenerate or zero-length pairs are automatically rejected even when
`valid_seq[i]` is `True`.

**`prev_params` / `prev_grad` anchoring.** Even when a probe is marked
invalid, `prev_params` and `prev_grad` are advanced to the latest probe so
that subsequent `(s, y)` differences anchor on the most recent real point.

**Arguments**

| Name           | Type          | Description                              |
|----------------|---------------|------------------------------------------|
| `state`        | `LBFGSState`  | Current oracle state.                    |
| `params_seq`   | `jnp.ndarray` | Probe parameter vectors, shape `(k, n)`. |
| `grad_seq`     | `jnp.ndarray` | Probe gradients, shape `(k, n)`.         |
| `valid_seq`    | `jnp.ndarray` | Boolean mask, shape `(k,)`.              |
| `history_size` | `int`         | Buffer capacity (must match `state`).    |

**Returns** — updated `LBFGSState` with all valid, positive-curvature probes
folded in.

---

### `lbfgs_direction`

```python
def lbfgs_direction(state: LBFGSState, grad) -> jnp.ndarray
```

Computes the L-BFGS quasi-Newton direction `−H∇f` via the **two-loop
recursion** (Nocedal & Wright, Algorithm 7.4).

**Algorithm.**

*First loop* (newest → oldest, `i = 0 … m−1`):

```
αᵢ  ←  ρᵢ · sᵢᵀ q
q   ←  q − αᵢ yᵢ
```

*Initial Hessian approximation:*

```
r  ←  γ · q          (H₀ = γI)
```

*Second loop* (oldest → newest, reverse of first loop):

```
βᵢ  ←  ρᵢ · yᵢᵀ r
r   ←  r + (αᵢ − βᵢ) sᵢ
```

Both loops are implemented with `jax.lax.scan` (the second with
`reverse=True`) so the recursion is fully JIT/vmap compatible and compiles
to a single fused kernel.

**Unfilled slots.** Zero-initialised buffer entries (`s = y = 0`, `ρ = 0`)
contribute `αᵢ = βᵢ = 0` and leave `q` / `r` unchanged, so no explicit
masking is required.

**Arguments**

| Name    | Type          | Description                                   |
|---------|---------------|-----------------------------------------------|
| `state` | `LBFGSState`  | Current oracle state (holds history buffers). |
| `grad`  | `jnp.ndarray` | Current gradient `∇f`, shape `(n,)`.          |

**Returns** — the quasi-Newton direction `−H∇f`, shape `(n,)`.

---

### `jnp_select_buf` *(internal helper)*

```python
def jnp_select_buf(flag, a, b) -> jnp.ndarray
```

Selects between two equally-shaped buffers using a scalar boolean `flag`.
Equivalent to `jnp.where(flag, a, b)` but documents the intent of a
whole-buffer select (as opposed to an element-wise blend). Used internally
to conditionally swap history buffers without materialising a per-element
mask over the full `(history_size, n)` array.

---

## Design Notes

### Buffer ordering

Pairs are stored **most-recent-first**. The first loop of the two-loop
recursion naturally iterates newest → oldest (index `0 … m−1`), and the
second loop uses `lax.scan(reverse=True)` to iterate oldest → newest without
an explicit buffer reversal.

### Relative curvature guard

The standard L-BFGS safeguard uses an absolute threshold `yᵀs > ε`. This
fails when the step `s` is very large (e.g. early in optimisation) because
`yᵀs` can be numerically large while the *relative* curvature `yᵀs / (‖y‖‖s‖)`
is near zero. The relative guard

```
yᵀs  >  ε · ‖y‖₂ · ‖s‖₂
```

anchors the threshold to the Cauchy-Schwarz scale and correctly rejects
near-flat updates regardless of the absolute magnitudes of `s` and `y`.

### NaN safety under `jax.grad`

`jnp.where` evaluates **both** branches before selecting. A raw `1.0 / ys`
in the rejected branch produces `inf` / `NaN` when `ys ≤ 0`, and under
`jax.grad` that `NaN` backpropagates through the non-selected branch and
poisons the gradient. The fix is to compute the reciprocal on a safe
denominator:

```python
safe_ys = jnp.where(valid, ys, jnp.ones_like(ys))
rho = jnp.where(valid, 1.0 / safe_ys, 0.0)
```

The same pattern is applied to the `γ = ⟨y,s⟩/⟨y,y⟩` update.

### `lax.scan` vs explicit loops

Both loops of the two-loop recursion use `jax.lax.scan`. This keeps the
recursion JIT/vmap compatible (no Python-level unrolling), compiles to a
single fused kernel, and avoids the quadratic trace-time cost of unrolling
`history_size` steps at trace time.

---

## Usage Examples

### Basic usage

```python
from qqn_jax.lbfgs import init_lbfgs_state, update_lbfgs_history, lbfgs_direction
import jax.numpy as jnp


def grad_fn(x):
    return 2.0 * x  # gradient of f(x) = ‖x‖²


history_size = 10
x0 = jnp.ones(4)
g0 = grad_fn(x0)

# Initialise state.
state = init_lbfgs_state(x0, g0, history_size)

# Take a step and update history.
x1 = x0 - 0.1 * g0
g1 = grad_fn(x1)
state = update_lbfgs_history(state, x1, g1, history_size)

# Compute quasi-Newton direction at x1.
d = lbfgs_direction(state, g1)  # ≈ -H∇f
```

### JIT compilation

```python
import jax


@jax.jit
def lbfgs_step(state, x_new, g_new):
    state = update_lbfgs_history(state, x_new, g_new, history_size=10)
    return lbfgs_direction(state, g_new)
```

### Batch probe feeding

```python
from qqn_jax.lbfgs import update_lbfgs_history_batch

# params_seq: (k, n) array of line-search probe positions
# grad_seq:   (k, n) array of corresponding gradients
# valid_seq:  (k,)   boolean mask (False = unused scratch slot)
state = update_lbfgs_history_batch(
    state, params_seq, grad_seq, valid_seq, history_size=10
)
```

### vmap over a batch of problems

```python
import jax

batch_x0 = jnp.ones((8, 4))  # 8 independent problems, dim=4
batch_g0 = 2.0 * batch_x0

init_batch = jax.vmap(
    lambda x, g: init_lbfgs_state(x, g, history_size=10)
)(batch_x0, batch_g0)
```

---

## Relationship to Other Modules

| Module           | Relationship                                                                                                         |
|------------------|----------------------------------------------------------------------------------------------------------------------|
| `oracles.py`     | Wraps `LBFGSState` + `lbfgs_direction` behind the `Oracle` interface consumed by `QQN`.                              |
| `solver.py`      | Calls `update_lbfgs_history` (or `update_lbfgs_history_batch` when `feed_probes_to_oracle=True`) once per iteration. |
| `line_search.py` | Probe positions / gradients collected here are optionally fed back via `update_lbfgs_history_batch`.                 |
| `types.py`       | `LBFGSState` satisfies the typed `OracleState` protocol defined there.                                               |

---

## Testing

The test suite in `tests/test_lbfgs.py` covers:

| Test                                                   | What it checks                                                        |
|--------------------------------------------------------|-----------------------------------------------------------------------|
| `test_initial_direction_is_negative_gradient`          | With no history, `H₀ = I` so `d = −∇f`.                               |
| `test_history_update_curvature`                        | Positive-curvature pair increments `step_count`; direction descends.  |
| `test_rejects_negative_curvature`                      | Pair with `yᵀs < 0` leaves `step_count = 0`.                          |
| `test_history_buffer_is_circular`                      | `step_count` saturates at `history_size` after overflow.              |
| `test_relative_curvature_guard_rejects_tiny_curvature` | Near-flat update rejected by relative guard.                          |
| `test_history_batch_replays_valid_probes`              | All three valid probes admitted by batch update.                      |
| `test_history_batch_skips_invalid_slots`               | `valid_seq=False` slot skipped; `step_count = 2`.                     |
| `test_direction_is_jittable`                           | `lbfgs_direction` compiles under `jax.jit`.                           |
| `test_update_is_jittable`                              | `update_lbfgs_history` compiles under `jax.jit`.                      |
| `test_gradient_does_not_poison_on_rejected_pair`       | `jax.grad` through a rejected pair stays finite.                      |
| `test_lbfgs_direction_solves_diagonal_quadratic`       | After sufficient history, direction descends on a diagonal quadratic. |

Run with:

```bash
pytest tests/test_lbfgs.py -v
```

---

## References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.),
  Algorithm 7.4 (L-BFGS two-loop recursion). Springer.
- Liu, D. C. & Nocedal, J. (1989). On the limited memory BFGS method for
  large scale optimization. *Mathematical Programming*, 45(1–3), 503–528.
- Barzilai, J. & Borwein, J. M. (1988). Two-point step size gradient methods.
  *IMA Journal of Numerical Analysis*, 8(1), 141–148.