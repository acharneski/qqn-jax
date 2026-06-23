# `solver.py` — QQN Optimizer and State

## Overview

`solver.py` implements the **Quasi-Quadratic-Newton (QQN)** optimizer: the
top-level `QQN` class and its immutable state container `QQNState`.

QQN constructs a quadratic interpolation path

```
d(t) = t(1 − t)·(−∇f) + t²·(−H∇f)
```

blending the steepest-descent direction `−∇f` (the path's tangent at `t = 0`)
with a curvature-aware quasi-Newton direction `−H∇f` (the `t = 1` endpoint
supplied by a pluggable *oracle*). A line search then traverses this path over
the scalar parameter `t ∈ [0, 1]`, selecting one point `x + d(t)` as the next
iterate.

The entire solver is written as pure, functional JAX and is fully compatible
with `jit`, `vmap`, `pmap`, and `grad`. The `run` method uses
`lax.while_loop` internally, so a complete optimization run is itself a single
traceable, differentiable, vmappable operation.

---

## Design Principles

### The path is the search space

The line search traverses the path parameter `t` directly. Each evaluated
point `x + d(t)` is a *state* on the quadratic curve, not a direction to be
independently re-scaled by a separate inner line search. Rescaling the gradient
(or the oracle direction) does **not** change the geometric path traced by
`d(t)` — it only reparameterizes how `t` maps onto arc length along the curve.
The curve itself, and therefore the set of candidate states, is invariant to
such rescaling.

### Honest predicted reduction

The along-path quadratic model has a closed-form directional derivative:

```
slope(τ) = ⟨∇f, d′(τ)⟩ = (1 − 2τ)·m_g + 2τ·m_q
```

whose integral gives the model's reduction in closed form:

```
pred(t) = −∫₀ᵗ slope(τ) dτ = −⟨∇f, d(t)⟩
        = −[t(1−t)·⟨∇f, −∇f⟩ + t²·⟨∇f, −H∇f⟩]
```

This is computed analytically as two O(n) dot products rather than by
materializing the full blended direction vector. There is no separate curvature
term to add: the path's curvature is already fully encoded in `d(t)`. Adding a
spurious second-order term double-counts curvature and drives the trust-region
acceptance ratio `ρ` negative near convergence.

### NaN-safety

Curvature reciprocals and matrix solves in the oracle are guarded so that
masked-out branches never backpropagate NaNs under `jax.grad`.

### Divergence termination

A run terminates early if an iterate becomes non-finite, so a single bad start
in a vmapped batch does not waste the rest of the batch's iterations on NaN
arithmetic.

### Honest eval counting

`QQNState.num_evals` accumulates every value-and-grad evaluation: line-search
probes, spline probes, aux recomputes, probe-value recovery passes, and the
initial `init_state` evaluation. Benchmarks should compare *work done*, not
just iteration counts — QQN performs several evaluations per iteration.

---

## `QQNState`

```python
class QQNState(NamedTuple):
    iter:          jnp.ndarray   # iteration counter
    value:         jnp.ndarray   # current objective value f(x)
    grad:          jnp.ndarray   # current gradient ∇f(x)
    oracle_state:  Any           # oracle state (e.g. L-BFGS curvature history)
    step_size:     jnp.ndarray   # last accepted path parameter t ∈ [0, 1]
    error:         jnp.ndarray   # gradient L2 norm ‖∇f‖ (convergence metric)
    done:          jnp.ndarray   # bool: convergence or divergence reached
    aux:           Any           # optional auxiliary output of the objective
    region_state:  Any           # optional projective-region state
    num_evals:     jnp.ndarray   # cumulative value-and-grad evaluations
    qn_slope:      jnp.ndarray   # ⟨∇f, −H∇f⟩ at t=1 (non-negative ⟹ non-descent oracle)
    ls_success:    jnp.ndarray   # bool: inner line search met its acceptance test
    last_reduction:jnp.ndarray   # actual objective decrease on the last accepted step
```

`QQNState` is a `NamedTuple` and therefore fully JIT/vmap compatible. All
fields are JAX arrays or nested pytrees; none are Python scalars.

### Field reference

| Field            | Type              | Description |
|------------------|-------------------|-------------|
| `iter`           | `int32 scalar`    | Number of completed iterations. |
| `value`          | `float scalar`    | Objective value `f(x)` at the current iterate. |
| `grad`           | pytree            | Gradient `∇f(x)` at the current iterate. |
| `oracle_state`   | pytree            | Internal state of the oracle (e.g. L-BFGS `s`/`y` history buffers). |
| `step_size`      | `float scalar`    | Accepted path parameter `t` from the last line search. |
| `error`          | `float scalar`    | `‖∇f‖₂`; the convergence metric compared against `tol`. |
| `done`           | `bool scalar`     | `True` when `error ≤ tol` or the iterate is non-finite. |
| `aux`            | pytree or `None`  | Auxiliary output of `fun` (only populated when `has_aux=True`). |
| `region_state`   | pytree or `()`    | State carried by the projective region (e.g. trust-region radius). |
| `num_evals`      | `int32 scalar`    | Cumulative value-and-grad evaluations since `init_state`. |
| `qn_slope`       | `float scalar`    | `⟨∇f, −H∇f⟩`; non-negative value flags a degenerate oracle direction. |
| `ls_success`     | `bool scalar`     | Whether the inner line search satisfied its acceptance criterion. |
| `last_reduction` | `float scalar`    | `f(x) − f(x_new)` on the last accepted step. |

---

## `QQN`

```python
class QQN:
    def __init__(
        self,
        fun: Callable,
        maxiter: int = 100,
        tol: float = 1e-5,
        history_size: int = 10,
        line_search: str = "armijo",
        line_search_options: Optional[Dict[str, Any]] = None,
        spline: bool = False,
        has_aux: bool = False,
        region=None,
        oracle="lbfgs",
        feed_probes_to_oracle: bool = False,
        probe_descent_gate: bool = True,
        max_probes: int = 32,
    ): ...
```

### Constructor arguments

| Argument                | Default      | Description |
|-------------------------|--------------|-------------|
| `fun`                   | —            | Objective `f(params, *args) → scalar` (or `(scalar, aux)` when `has_aux=True`). |
| `maxiter`               | `100`        | Maximum number of iterations before `run` returns. |
| `tol`                   | `1e-5`       | Convergence tolerance: stop when `‖∇f‖₂ ≤ tol`. |
| `history_size`          | `10`         | L-BFGS memory size `m` (number of `(s, y)` curvature pairs retained). |
| `line_search`           | `"armijo"`   | Line-search strategy. See [Line searches](#line-searches). |
| `line_search_options`   | `None`       | Dict of kwargs forwarded verbatim to the chosen line-search function. |
| `spline`                | `False`      | Enable cubic Hermite spline refinement (orthogonal to `line_search`). |
| `has_aux`               | `False`      | Whether `fun` returns `(scalar, aux)` instead of `scalar`. |
| `region`                | `None`       | A `Region` instance, or `None` for the identity (unconstrained) region. |
| `oracle`                | `"lbfgs"`    | Oracle name or `Oracle` instance. See [Oracles](#oracles). |
| `feed_probes_to_oracle` | `False`      | Forward line-search probe gradients into the oracle's curvature memory. |
| `probe_descent_gate`    | `True`       | When feeding probes, only admit those that strictly decrease the objective. |
| `max_probes`            | `32`         | Probe-buffer capacity (also sets the line-search probe-buffer size). |

### Methods

#### `init_state(params, *args) → QQNState`

Initializes the solver state at `params`. Performs exactly **one**
value-and-grad evaluation. Sets `num_evals = 1`.

```python
state = solver.init_state(x0)
# state.iter == 0
# state.value == fun(x0)
# state.done == (‖∇f(x0)‖ ≤ tol)
```

#### `update(params, state, *args) → (new_params, new_state)`

Performs a single QQN iteration:

1. **Oracle step** — calls `oracle.direction(params, grad, oracle_state)` to
   obtain the quasi-Newton direction `−H∇f` (the `t = 1` endpoint). Records
   `qn_slope = ⟨∇f, −H∇f⟩` as a diagnostic.
2. **Line search** — calls the configured line search (optionally wrapped by
   the spline refinement) to find an accepted step `t` along the quadratic
   path `d(t)`.
3. **Aux recompute** — if `has_aux=True`, calls `fun(new_params)` once more
   to recover the auxiliary output (one extra forward pass, counted in
   `num_evals`).
4. **Oracle update** — calls `oracle.update(oracle_state, oracle_info)` to
   incorporate the new `(s, y)` curvature pair. If `feed_probes_to_oracle` is
   enabled, all valid probe `(params, grad)` pairs from the line search are
   forwarded as well.
5. **Region update** — calls `region.update(region_state, region_info)` with
   the honest along-path predicted reduction and actual reduction.
6. **Convergence check** — sets `done = (‖∇f_new‖ ≤ tol) or not isfinite(x_new)`.

Returns `(new_params, new_state)`.

#### `run(init_params, *args) → (params, state)`

Runs QQN to convergence (or `maxiter`) using `lax.while_loop`. The entire
loop is JIT/vmap/pmap compatible.

```python
params, state = solver.run(x0)
# or, JIT-compiled:
params, state = jax.jit(solver.run)(x0)
# or, vmapped over a batch:
params, states = jax.vmap(solver.run)(x0_batch)
```

The loop terminates when either:
- `state.done` is `True` (`‖∇f‖ ≤ tol` or non-finite iterate), or
- `state.iter >= maxiter`.

---

## Line searches

The `line_search` argument selects the strategy used to traverse the
quadratic path. All strategies are registered in the module-level
`_LINE_SEARCHES` dict.

| Name            | Description |
|-----------------|-------------|
| `"armijo"`      | **(default)** Backtracking with Armijo sufficient-decrease condition. Robust efficiency winner on smooth full-batch problems. |
| `"backtracking"` | Backtracking line search (alias / variant of Armijo). |
| `"strong_wolfe"` | Strong Wolfe conditions (sufficient decrease + curvature). Can over-restrict the quadratic-path step and fail to converge on some problems. |
| `"hager_zhang"` | Hager-Zhang approximate Wolfe conditions. |
| `"fixed"`       | Fixed step size (no search). |
| `"spline"`      | Cubic Hermite spline search (equivalent to `line_search="armijo", spline=True`). |

### Forwarding options

```python
QQN(fun, line_search="backtracking",
    line_search_options={"c1": 1e-3, "shrink": 0.6, "max_iter": 10})
```

All keys in `line_search_options` are forwarded verbatim to the chosen
line-search function, overriding its defaults.

### Spline refinement

When `spline=True` (or `line_search="spline"`), the base line search is
*wrapped* by `spline_wrap`. Every probe evaluated along the consistent path
is reused as a control point of a cubic Hermite spline; the spline's
stationary points are then probed to improve on the inner search's accepted
step. This is orthogonal to the choice of base line search and composes with
any of the strategies above.

### Probe recording

When `feed_probes_to_oracle=True`, the line-search probe buffers are sized to
`max_probes` so they match the oracle's replay capacity. When
`feed_probes_to_oracle=False`, probe recording is disabled in the inner
`while_loop` to avoid allocating the `(max_probes, n)` scratch buffer.

---

## Oracles

The oracle supplies the `t = 1` endpoint `−H∇f` of the quadratic path. It is
resolved from the `oracle` constructor argument via `resolve_oracle` from
`qqn_jax.oracles`.

| Name               | Description |
|--------------------|-------------|
| `"lbfgs"`          | **(default)** Limited-memory BFGS two-loop recursion. |
| `"momentum"`       | Heavy-ball / exponentially-weighted gradient. |
| `"secant"`         | Barzilai-Borwein step (matrix-free, O(n) memory). |
| `"shampoo"`        | Structure-aware preconditioning. |
| `"anderson"`       | Anderson (Type-II) acceleration. |
| `"anderson+secant"` | Anderson with Barzilai-Borwein safeguarded fallback. |
| `"lbfgs+secant"`   | L-BFGS with Barzilai-Borwein safeguarded fallback. |
| custom instance    | Any object implementing the `Oracle` protocol. |

The `history_size` constructor argument is forwarded to `resolve_oracle` and
controls the L-BFGS memory size `m`.

---

## Probe feeding

When `feed_probes_to_oracle=True`, every gradient evaluated *during* the line
search is forwarded into the oracle's curvature memory via `OracleInfo`'s
`probe_params`, `probe_grads`, `probe_valid`, and `probe_alphas` fields.

The **descent gate** (`probe_descent_gate=True`, the default) admits only
probes whose objective value strictly improves on the current iterate:

```
probe_valid[i] = probe_valid[i] AND (probe_values[i] < state.value)
```

This prevents non-representative (rejected) line-search probes from polluting
the L-BFGS curvature history — the documented cause of catastrophic stalls
when feeding probes without gating.

**Value recovery:** If the line search recorded probe params and grads but not
their objective values (e.g. when wrapped by the spline), a single vmapped
forward pass recovers the values for the gate. The number of extra forward
passes is counted in `num_evals` via `extra_recovery_evals`.

---

## Regions

The `region` argument accepts a `Region` instance (or `None` for the identity
region). The region is called at two points per iteration:

- `region.init(params)` — in `init_state`, to initialize `region_state`.
- `region.update(region_state, region_info)` — in `update`, after the line
  search, to update the region state (e.g. adapt the trust-region radius).

`RegionInfo` carries:

| Field              | Description |
|--------------------|-------------|
| `params`           | Current iterate `x`. |
| `new_params`       | Proposed next iterate `x + d(t)`. |
| `pred_reduction`   | Along-path model reduction `−⟨∇f, d(t)⟩`. |
| `actual_reduction` | Actual reduction `f(x) − f(x + d(t))`. |
| `t`                | Accepted path parameter. |
| `step_size`        | Accepted step size (same as `t` for the quadratic path). |

The predicted reduction is computed analytically:

```python
m_g = -tree_vdot(grad, grad)          # ⟨∇f, −∇f⟩ = −‖∇f‖²
m_q =  tree_vdot(grad, qn_dir)        # ⟨∇f, −H∇f⟩
pred_reduction = -(t*(1-t)*m_g + t*t*m_q)
```

A small epsilon floor (`1e-16`) prevents a `0/0` acceptance ratio when the
step is degenerate.

---

## Eval accounting

`QQNState.num_evals` is the authoritative count of value-and-grad evaluations:

| Source                        | Count |
|-------------------------------|-------|
| `init_state`                  | 1 |
| Line-search probes (per iter) | `res.num_evals` |
| Aux recompute (per iter)      | 1 if `has_aux=True`, else 0 |
| Probe-value recovery (per iter) | `res.probe_params.shape[0]` if descent gate fires without stored values, else 0 |

The total after `k` iterations is:

```
num_evals = 1 + Σᵢ (ls_evals_i + aux_evals_i + recovery_evals_i)
```

---

## Internal helpers

### `_eval(params, *args) → (value, grad, aux)`

Calls `_value_and_grad` and splits off the auxiliary output when
`has_aux=True`. Used in `init_state`.

### `_plain_value_and_grad(params, *args) → (value, grad)`

Strips the auxiliary output before returning. This is the callable handed to
the line search, which expects a plain `(value, grad)` signature.

---

## Module-level registry

```python
_LINE_SEARCHES = {
    "strong_wolfe":  strong_wolfe_search,
    "backtracking":  backtracking_search,
    "armijo":        armijo_search,
    "hager_zhang":   hager_zhang_search,
    "fixed":         fixed_step_search,
    "spline":        spline_search,
}
```

An unknown `line_search` name raises `ValueError` at construction time (not at
`run` time), so misconfiguration is caught immediately.

---

## Usage examples

### Minimal

```python
from qqn_jax import QQN
import jax.numpy as jnp

def rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

solver = QQN(rosenbrock, maxiter=200, tol=1e-6)
x_opt, state = solver.run(jnp.zeros(10))
print(state.value, state.iter, state.error)
```

### JIT-compiled

```python
import jax
run_jit = jax.jit(solver.run)
x_opt, state = run_jit(x0)
```

### Vmapped over a batch of initializations

```python
x0_batch = jnp.stack([x0_a, x0_b, x0_c])   # (B, n)
xs, states = jax.vmap(solver.run)(x0_batch)
```

### Custom oracle and region

```python
from qqn_jax import QQN, BoxRegion

solver = QQN(
    fun,
    oracle="lbfgs+secant",
    region=BoxRegion(lo=-1.0, hi=1.0),
    maxiter=300,
)
```

### Objective with auxiliary output

```python
def fun_with_aux(x):
    loss = ...
    return loss, {"grad_norm": jnp.linalg.norm(x)}

solver = QQN(fun_with_aux, has_aux=True)
x_opt, state = solver.run(x0)
print(state.aux["grad_norm"])
```

### Probe feeding

```python
solver = QQN(
    fun,
    oracle="lbfgs",
    feed_probes_to_oracle=True,
    probe_descent_gate=True,
    max_probes=32,
)
```

### Step-by-step (JAXopt-style)

```python
state = solver.init_state(x0)
for _ in range(10):
    x0, state = solver.update(x0, state)
    print(state.iter, state.value, state.error)
```

---

## Dependencies

| Import                        | Purpose |
|-------------------------------|---------|
| `qqn_jax.line_search`         | Line-search implementations. |
| `qqn_jax.spline_search`       | Cubic Hermite spline refinement wrapper. |
| `qqn_jax.oracles`             | Oracle abstraction and `resolve_oracle`. |
| `qqn_jax.regions`             | Region abstraction and `resolve_region`. |
| `qqn_jax.utils`               | `make_value_and_grad`, `tree_l2_norm`, `tree_vdot`. |

---

## See also

- `oracles.py` — Oracle implementations (`LBFGSOracle`, `SecantOracle`, …).
- `lbfgs.py` — L-BFGS two-loop recursion and curvature-history buffers.
- `line_search.py` — Line-search strategies.
- `spline_search.py` — Cubic Hermite spline refinement.
- `regions.py` — Projective regions (`BoxRegion`, `TrustRegion`, …).
- `utils.py` — Pytree / tree-math helpers.
- `types.py` — Typed interfaces (`chex` / `jaxtyping`).