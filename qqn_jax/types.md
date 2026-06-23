# `types.py` — Typed Interfaces for QQN-JAX

This module defines the core type aliases and annotations used throughout
QQN-JAX. All array types are expressed with [`chex`](https://github.com/deepmind/chex)
and [`jaxtyping`](https://github.com/google/jaxtyping) so that shapes and
dtypes are both documented and (optionally) runtime-checkable.

---

## Overview

QQN-JAX is a fully functional, JAX-native optimizer. Every public API
surface — `init_state`, `update`, `run`, oracles, line searches, regions —
is typed using the aliases defined here. Importing from `types.py` (rather
than spelling out raw `jax.Array` annotations) keeps signatures consistent
and makes shape expectations explicit at a glance.

---

## Type Aliases

### `Scalar`

```python
Scalar = Float[Array, ""]
```

A zero-dimensional (rank-0) floating-point JAX array. Used wherever a
single real number is expected: objective values, step sizes, gradient
norms, and convergence tolerances.

> **Note:** `jaxtyping.Scalar` was removed in recent releases of the
> library. `Float[Array, ""]` is the canonical, forward-compatible
> replacement used throughout QQN-JAX.

---

### `Params`

```python
Params = Float[Array, " n"]
```

A flat, one-dimensional floating-point array representing the current
iterate (parameter vector). The dimension `n` is the total number of
scalar parameters after any pytree flattening performed by the solver.

Used as:
- The input to the objective function `f(params, *args)`.
- The `params` argument to `init_state`, `update`, and `run`.
- The first element of the `(params, state)` pair returned by `run`.

---

### `Grad`

```python
Grad = Float[Array, " n"]
```

A flat, one-dimensional floating-point array representing the gradient
`∇f(params)`. Always the same shape as `Params`.

Stored in `QQNState.grad` and passed internally to oracles and line
searches. The convergence metric `QQNState.error` is the L2 norm of this
vector.

---

### `Direction`

```python
Direction = Float[Array, " n"]
```

A flat, one-dimensional floating-point array representing a search
direction in parameter space. Shares the shape of `Params` and `Grad`.

In QQN the quadratic-path direction at parameter `t` is:

```
d(t) = t(1 - t)·(-∇f) + t²·(-H∇f)
```

Both the steepest-descent component `(-∇f)` and the quasi-Newton
component `(-H∇f)` are typed as `Direction`.

---

### `Value`

```python
Value = Float[Array, ""]
```

A zero-dimensional floating-point array holding the scalar objective
value `f(params)`. Identical in shape to `Scalar`; the separate alias
clarifies *semantic* intent (an objective evaluation result) versus a
generic scalar quantity (e.g. a step size).

---

### `ObjectiveFn`

```python
ObjectiveFn = Callable[..., Scalar]
```

The signature of a plain objective function:

```python
def fun(params: Params, *args: Any) -> Scalar: ...
```

Pass an `ObjectiveFn` to `QQN` when `has_aux=False` (the default).

---

### `ValueAndGradFn`

```python
ValueAndGradFn = Callable[..., Tuple[Value, Grad]]
```

The signature of a combined value-and-gradient function:

```python
def fun_and_grad(params: Params, *args: Any) -> Tuple[Value, Grad]: ...
```

QQN-JAX constructs this internally via `jax.value_and_grad`. It is
exposed as a type alias so that custom oracles and line searches can
declare their dependency on a value-and-grad callable unambiguously.

---

## Re-exports

The module re-exports the following names so that other QQN-JAX modules
can import everything they need from a single location:

| Name           | Source        | Purpose                                              |
|----------------|---------------|------------------------------------------------------|
| `Params`       | local         | Parameter-vector type alias.                         |
| `Grad`         | local         | Gradient-vector type alias.                          |
| `Direction`    | local         | Search-direction type alias.                         |
| `Value`        | local         | Scalar objective-value type alias.                   |
| `ObjectiveFn`  | local         | Plain objective callable type.                       |
| `ValueAndGradFn` | local       | Value-and-grad callable type.                        |
| `Any`          | `typing`      | Escape hatch for untyped auxiliary data.             |
| `chex`         | `chex`        | Runtime array-shape checking utilities.              |

---

## Usage Examples

### Annotating a custom oracle

```python
from qqn_jax.types import Params, Grad, Direction, Scalar

def my_direction(grad: Grad, state: MyOracleState) -> Direction:
    """Return -H∇f for a custom curvature estimate."""
    ...
```

### Annotating an objective with auxiliary output

```python
from qqn_jax.types import Params, Value, Any
from typing import Tuple, Dict

def fun_with_aux(params: Params) -> Tuple[Value, Dict[str, Any]]:
    loss = ...
    aux = {"accuracy": ...}
    return loss, aux

solver = QQN(fun_with_aux, has_aux=True)
```

### Runtime shape checking with `chex`

```python
import chex
from qqn_jax.types import Params, Grad

def checked_update(params: Params, grad: Grad) -> Params:
    chex.assert_equal_shape([params, grad])
    chex.assert_rank(params, 1)
    ...
```

---

## Design Notes

- **Flat arrays only.** All type aliases describe rank-1 (or rank-0)
  arrays. QQN-JAX flattens pytree parameter structures internally before
  any arithmetic; the typed interfaces operate on the flattened
  representation.

- **`Float` dtype family.** All aliases use `jaxtyping.Float`, which
  matches any floating-point dtype (`float16`, `bfloat16`, `float32`,
  `float64`). QQN-JAX does not hard-code a precision; the dtype follows
  the input.

- **`Scalar` vs `Value`.** Both are `Float[Array, ""]`. The distinction
  is purely semantic: `Value` signals "this came from evaluating the
  objective", while `Scalar` signals "this is some real number" (e.g. a
  step size `t`, a gradient norm, or a tolerance threshold).

- **`Any` for auxiliary data.** The `has_aux=True` path returns
  `(Value, Any)`. The auxiliary payload is intentionally untyped because
  it is user-defined and opaque to the solver internals.

- **`chex` re-export.** Re-exporting `chex` from `types.py` lets other
  modules write `from qqn_jax.types import chex` and use
  `chex.assert_*` without an additional import line.

---

## Related Modules

| Module             | How it uses `types.py`                                      |
|--------------------|-------------------------------------------------------------|
| `solver.py`        | `ObjectiveFn`, `ValueAndGradFn`, `Params`, `Grad`, `Value`. |
| `oracles.py`       | `Grad`, `Direction`, `Scalar`.                              |
| `lbfgs.py`         | `Grad`, `Direction`, `Scalar`.                              |
| `line_search.py`   | `Params`, `Grad`, `Direction`, `Value`, `Scalar`.           |
| `spline_search.py` | `Params`, `Value`, `Scalar`.                                |
| `regions.py`       | `Params`, `Direction`.                                      |
| `utils.py`         | `Params`, `Grad`.                                           |