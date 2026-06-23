# `line_search.py` — Line Search Strategies

Line searches are a **first-class component** of QQN. Each strategy operates
over the pre-constructed quadratic path direction `d` and selects a step size
`α` satisfying sufficient decrease (Armijo) and, optionally, the curvature
(strong Wolfe) condition.

---

## Overview

The line search traverses the path parameter `t` directly. Each evaluated
`x + d(t)` is a *state*, not a direction to be independently re-scaled.
Rescaling the gradient (or the oracle direction) does **not** change the
geometric path traced by `d(t)` — it only reparameterizes how `t` maps onto
arc length.

All strategies share a common return type (`LineSearchResult`) and a uniform
call signature so they are fully interchangeable inside the `QQN` solver.
Every strategy is implemented with `lax.while_loop` or pure JAX primitives
and is therefore compatible with `jit`, `vmap`, `pmap`, and `grad`.

---

## `LineSearchResult`

Every line search returns a `LineSearchResult` named tuple.

```python
class LineSearchResult(NamedTuple):
    step_size:    jnp.ndarray   # chosen step size α
    new_value:    jnp.ndarray   # f(x + α·d)
    new_grad:     jnp.ndarray   # ∇f(x + α·d)
    new_params:   jnp.ndarray   # x + α·d  (projected if a region is active)
    done:         jnp.ndarray   # bool — acceptance condition satisfied
    probe_params: Any           # (max_probes, n) buffer of evaluated points
    probe_grads:  Any           # (max_probes, n) buffer of probe gradients
    probe_valid:  Any           # (max_probes,)  boolean mask of filled slots
    probe_values: Any           # (max_probes,)  objective value at each probe
    probe_alphas: Any           # (max_probes,)  step size α at each probe
    num_evals:    Any           # int32 — number of value-and-grad calls made
```

### Probe buffers

Every strategy records the points it evaluates into fixed-size probe buffers
(`probe_params`, `probe_grads`, `probe_valid`, `probe_values`,
`probe_alphas`). These buffers are consumed by the solver when
`feed_probes_to_oracle=True` to enrich the oracle's curvature memory without
any extra forward passes.

- Buffer size is controlled by the `max_probes` argument (default `32`).
- When `record_probes=False` the buffer is shrunk to a single scratch slot,
  eliminating the allocation overhead for callers that do not consume probes.
- `probe_valid[i]` is `True` only for slots that were actually written.
- `probe_alphas` stores the step size at each probe so the oracle can replay
  probes in `α`-order rather than slot-order (important for secant
  differences).

### Eval counting

`num_evals` counts every combined value-and-grad oracle call made by the
search. Downstream code accumulates these into `QQNState.num_evals` so
benchmarks compare *work done*, not just iteration counts.

- For Optax-backed searches (`strong_wolfe`, `hager_zhang`) the internal
  probe count is not exposed; `num_evals` is reported as a conservative upper
  bound (`max_iter + 1`) so totals never silently undercount.

---

## Available Strategies

### `backtracking_search` / `armijo_search`

```python
backtracking_search(
    value_and_grad_fn,
    params, direction, value, grad,
    *args,
    init_step: float = 1.0,
    c1:        float = 1e-2,
    shrink:    float = 0.5,
    max_iter:  int   = 5,
    region=None,
    region_state=None,
    max_probes:    int  = 32,
    record_probes: bool = True,
) -> LineSearchResult
```

Self-contained Armijo backtracking search implemented with `lax.while_loop`.

**Algorithm:**
1. Evaluate `f` and `∇f` at `x + init_step · d`.
2. While the Armijo condition `f(x + α·d) ≤ f(x) + c1·α·gᵀd` is not
   satisfied and `i < max_iter`, shrink `α ← α · shrink` and re-evaluate.
3. Return the last accepted point.

**Parameters:**

| Parameter      | Default | Description                                              |
|----------------|---------|----------------------------------------------------------|
| `init_step`    | `1.0`   | Initial step size.                                       |
| `c1`           | `1e-2`  | Armijo sufficient-decrease constant.                     |
| `shrink`       | `0.5`   | Multiplicative shrink factor per backtracking step.      |
| `max_iter`     | `5`     | Maximum number of backtracking iterations.               |
| `region`       | `None`  | Optional `Region`; candidate is projected before eval.   |
| `region_state` | `None`  | State for the region (e.g. trust-region radius).         |
| `max_probes`   | `32`    | Probe buffer capacity.                                   |
| `record_probes`| `True`  | Set `False` to skip probe recording (saves allocation).  |

`armijo_search` is a direct alias for `backtracking_search` provided so
users can refer to the search by its classical name.

**Notes:**
- Every `eval_at` call (initial probe + each backtracking step) is recorded
  into the probe buffer in slot order.
- `done` is `True` when the Armijo condition holds at the returned `α`.
- This is the **recommended default** for smooth, full-batch objectives.

---

### `strong_wolfe_search`

```python
strong_wolfe_search(
    value_and_grad_fn,
    params, direction, value, grad,
    *args,
    c1:       float = 1e-3,
    c2:       float = 0.7,
    max_iter: int   = 10,
    region=None,
    region_state=None,
    max_probes:    int  = 32,
    record_probes: bool = True,
) -> LineSearchResult
```

Strong Wolfe line search delegated to Optax's
`scale_by_zoom_linesearch`.

**Algorithm:**
Enforces both the Armijo sufficient-decrease condition and the strong
curvature condition `|∇f(x + α·d)ᵀd| ≤ c2·|∇f(x)ᵀd|`, which keeps
L-BFGS curvature updates well-conditioned.

**Parameters:**

| Parameter  | Default | Description                                              |
|------------|---------|----------------------------------------------------------|
| `c1`       | `1e-3`  | Armijo sufficient-decrease constant.                     |
| `c2`       | `0.7`   | Strong Wolfe curvature constant.                         |
| `max_iter` | `10`    | Maximum zoom iterations (budget passed to Optax).        |

**Notes:**
- Optax's zoom search does not expose its internal probe count; `num_evals`
  is reported as `max_iter + 1` (conservative upper bound).
- Only the single accepted point is recorded as a probe (Optax hides
  intermediate evaluations).
- `done` is `True` when the accepted value is strictly less than `value`.
- **Warning:** `strong_wolfe` can over-restrict the quadratic-path step and
  fail to converge on some problems. The Armijo / backtracking family is the
  recommended default.

---

### `hager_zhang_search`

```python
hager_zhang_search(
    value_and_grad_fn,
    params, direction, value, grad,
    *args,
    c1:       float = 0.1,
    max_iter: int   = 30,
    region=None,
    region_state=None,
    max_probes:    int  = 32,
    record_probes: bool = True,
) -> LineSearchResult
```

Hager-Zhang approximate-Wolfe line search via Optax's
`scale_by_backtracking_linesearch`.

**Parameters:**

| Parameter  | Default | Description                                              |
|------------|---------|----------------------------------------------------------|
| `c1`       | `0.1`   | Slope sufficient-decrease tolerance.                     |
| `max_iter` | `30`    | Maximum backtracking steps.                              |

**Notes:**
- Like `strong_wolfe_search`, only the accepted point is recorded as a probe.
- `num_evals` is reported as `max_iter + 1` (conservative upper bound).
- `done` is `True` when the accepted value is ≤ `value` (within tolerance).

---

### `fixed_step_search`

```python
fixed_step_search(
    value_and_grad_fn,
    params, direction, value, grad,
    *args,
    step_size: float = 1.0,
    region=None,
    region_state=None,
    max_probes:    int  = 32,
    record_probes: bool = True,
) -> LineSearchResult
```

Trivial line search using a constant step size. Always reports `done=True`
(makes no acceptance test).

**Parameters:**

| Parameter   | Default | Description                                              |
|-------------|---------|----------------------------------------------------------|
| `step_size` | `1.0`   | Fixed step size `α` to use unconditionally.              |

**Notes:**
- Useful for debugging, ablation studies, or when the quadratic path scaling
  already provides a sensible step.
- Exactly one value-and-grad evaluation is performed; `num_evals = 1`.
- The accepted point is recorded in probe slot `0`.

---

## Projective Regions

All strategies accept optional `region` and `region_state` arguments. When
supplied, the candidate point `x + α·d` is **projected onto the region**
before evaluation, so the search navigates the feasible (projected) path.

```python
from qqn_jax.regions import BoxRegion

region = BoxRegion(lo=0.0, hi=1.0)
res = backtracking_search(
    value_and_grad_fn, x, direction, value, grad,
    region=region,
    region_state=(),
)
# res.new_params is guaranteed to lie in [0, 1]^n
```

Available regions (see `regions.py`):

| Region            | Description                                              |
|-------------------|----------------------------------------------------------|
| `IdentityRegion`  | No-op (default, zero overhead).                          |
| `BoxRegion`       | Elementwise bounds `lo ≤ x ≤ hi`.                        |
| `OrthantRegion`   | OWL-QN-style sparsity (sign preservation).               |
| `TrustRegion`     | Adaptive `‖x_new − x‖₂ ≤ Δ`.                            |
| `NoDecreaseRegion`| Protect a secondary objective.                           |
| `Sequential`      | Compose multiple regions in order.                       |

---

## Internal Helpers

These are implementation details used internally; they are not part of the
public API.

### `_empty_probes(params, max_probes)`

Allocates zeroed probe buffers shaped for a flat parameter vector `params`.
Returns `(probe_params, probe_grads, probe_valid, probe_values, probe_alphas)`.

### `_record_probe(..., slot, p, g, v, a, max_probes)`

Writes a single probe `(p, g, v, a)` into `slot` of the probe buffers.
JIT-safe: out-of-range slots are silently ignored via `jnp.clip` + masking.

### `_make_projected_point(region, region_state, params)`

Returns a closure `project_candidate(candidate)` that projects a tentative
point onto the region. When the region is `IdentityRegion`, this is a no-op
with zero overhead.

---

## Usage in `QQN`

Line searches are selected by name via the `line_search` argument to `QQN`:

```python
from qqn_jax import QQN

QQN(fun, line_search="armijo")         # default; robust efficiency winner
QQN(fun, line_search="backtracking")   # alias for armijo
QQN(fun, line_search="strong_wolfe")
QQN(fun, line_search="hager_zhang")
QQN(fun, line_search="fixed")

# Forward keyword arguments to the inner line search.
QQN(fun, line_search="backtracking",
    line_search_options={"c1": 1e-3, "shrink": 0.6, "max_iter": 10})
```

The `"spline"` shorthand enables cubic Hermite spline refinement on top of
the default backtracking search (see `spline_search.py`):

```python
QQN(fun, line_search="spline")
# Equivalent to:
QQN(fun, line_search="backtracking", spline=True)
```

---

## JIT / vmap Compatibility

All strategies are fully traceable:

```python
import jax

# JIT a single search.
jit_search = jax.jit(
    lambda x, d, v, g: backtracking_search(vg_fn, x, d, v, g).step_size
)

# vmap over a batch of starting points.
def run_one(x):
    v, g = vg_fn(x)
    return backtracking_search(vg_fn, x, -g, v, g).new_value

batched = jax.vmap(run_one)(xs)  # xs shape: (B, n)
```

`backtracking_search` and `armijo_search` are also vmappable because their
`lax.while_loop` carries only fixed-shape arrays. The Optax-backed searches
(`strong_wolfe`, `hager_zhang`) inherit whatever vmap support Optax provides.

---

## Design Notes

- **The path is the search space.** The line search traverses the path
  parameter `t` directly. Each evaluated `x + d(t)` is a *state*, not a
  direction to be independently re-scaled.
- **NaN-safety.** Curvature reciprocals and matrix solves are guarded so
  that masked-out branches never backpropagate NaNs under `jax.grad`.
- **Honest eval counting.** `num_evals` accumulates every value-and-grad
  evaluation (line-search probes, spline probes, aux recomputes, fallback
  recoveries) so benchmarks compare *work done*, not just iteration counts.
  Strong-Wolfe / Hager-Zhang counts are conservative upper bounds because
  Optax does not expose its internal probe count.
- **Probe feeding.** Setting `feed_probes_to_oracle=True` on `QQN` folds
  every gradient evaluated *during* the line search into the oracle's
  curvature memory, gated (by default) on genuine objective decrease so that
  non-representative probes never pollute the history.

---

## See Also

| Module             | Description                                              |
|--------------------|----------------------------------------------------------|
| `solver.py`        | `QQN` optimizer — consumes `LineSearchResult`.           |
| `spline_search.py` | Cubic Hermite spline refinement wrapper.                 |
| `regions.py`       | Projective regions applied inside each search.           |
| `oracles.py`       | Oracle abstraction (`-H∇f` endpoint).                    |
| `utils.py`         | `tree_add_scaled`, `tree_vdot` helpers used here.        |