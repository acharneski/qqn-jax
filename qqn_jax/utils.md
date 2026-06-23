# `qqn_jax/utils.py` — Shared Utilities

This module provides the core mathematical and functional building blocks
used throughout QQN-JAX. All functions operate on **JAX pytrees**, are
**JIT-compatible**, and compose freely with `jax.vmap`, `jax.pmap`, and
`jax.grad`.

---

## Overview

| Function                    | Category        | Description                                               |
|-----------------------------|-----------------|-----------------------------------------------------------|
| `tree_vdot`                 | Tree math       | Inner product over a pytree pair.                         |
| `tree_add_scaled`           | Tree math       | `tree + scale * other` over pytrees.                      |
| `tree_scale`                | Tree math       | `scale * tree` over a pytree.                             |
| `tree_negative`             | Tree math       | `-tree` over a pytree.                                    |
| `tree_l2_norm`              | Tree math       | L2 norm of a pytree.                                      |
| `make_value_and_grad`       | Differentiation | Build a value-and-grad function, with optional `has_aux`. |
| `quadratic_path`            | QQN geometry    | Evaluate the QQN blended direction `d(t)`.                |
| `quadratic_path_derivative` | QQN geometry    | Derivative `d'(t)` of the blended direction w.r.t. `t`.   |

---

## Pytree Math Helpers

These functions mirror standard linear-algebra operations but work on
arbitrary JAX pytrees (nested dicts, lists, named tuples, etc.), making
them suitable for use with any parameter structure.

---

### `tree_vdot(a, b)`

Computes the **inner product** (dot product) over two pytrees of the same
structure.

```python
def tree_vdot(a, b):
    ...
```

**Parameters**

| Name | Type   | Description                             |
|------|--------|-----------------------------------------|
| `a`  | pytree | First operand.                          |
| `b`  | pytree | Second operand (same structure as `a`). |

**Returns** — `jnp.ndarray` scalar: `Σ vdot(leaf_a, leaf_b)` over all
corresponding leaf pairs.

**Notes**

- For flat arrays this is equivalent to `jnp.vdot(a, b)`.
- The operation is **symmetric**: `tree_vdot(a, b) == tree_vdot(b, a)`.
- Used internally by `tree_l2_norm` and by the line searches to compute
  directional derivatives `⟨∇f, d(t)⟩`.

**Example**

```python
a = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}
b = {"w": jnp.array([1.0, 1.0]), "b": jnp.array([1.0])}
tree_vdot(a, b)  # 1*1 + 2*1 + 3*1 = 6.0
```

---

### `tree_add_scaled(tree, scale, other)`

Computes **`tree + scale * other`** leaf-wise over two pytrees.

```python
def tree_add_scaled(tree, scale, other):
    ...
```

**Parameters**

| Name    | Type   | Description                                         |
|---------|--------|-----------------------------------------------------|
| `tree`  | pytree | Base pytree.                                        |
| `scale` | scalar | Scalar multiplier applied to `other`.               |
| `other` | pytree | Pytree to scale and add (same structure as `tree`). |

**Returns** — pytree with the same structure as `tree`.

**Notes**

- The primary use is applying a step: `x_new = tree_add_scaled(x, alpha, direction)`.
- Equivalent to `jax.tree_util.tree_map(lambda t, o: t + scale * o, tree, other)`.

**Example**

```python
x = jnp.array([1.0, 2.0])
delta = jnp.array([3.0, 4.0])
tree_add_scaled(x, 2.0, delta)  # [7.0, 10.0]
```

---

### `tree_scale(scale, tree)`

Computes **`scale * tree`** leaf-wise.

```python
def tree_scale(scale, tree):
    ...
```

**Parameters**

| Name    | Type   | Description        |
|---------|--------|--------------------|
| `scale` | scalar | Scalar multiplier. |
| `tree`  | pytree | Pytree to scale.   |

**Returns** — pytree with the same structure as `tree`.

**Example**

```python
tree_scale(3.0, jnp.array([1.0, -2.0]))  # [3.0, -6.0]
```

---

### `tree_negative(tree)`

Computes **`-tree`** leaf-wise (unary negation).

```python
def tree_negative(tree):
    ...
```

**Parameters**

| Name   | Type   | Description       |
|--------|--------|-------------------|
| `tree` | pytree | Pytree to negate. |

**Returns** — pytree with the same structure as `tree`.

**Notes**

- This is an **involution**: `tree_negative(tree_negative(x)) == x`.
- Used to form the steepest-descent direction `grad_dir = tree_negative(grad)`.

**Example**

```python
tree_negative(jnp.array([1.0, -2.0]))  # [-1.0, 2.0]
```

---

### `tree_l2_norm(tree)`

Computes the **L2 norm** of a pytree, treating all leaves as a single
flattened vector.

```python
def tree_l2_norm(tree):
    ...
```

**Parameters**

| Name   | Type   | Description                   |
|--------|--------|-------------------------------|
| `tree` | pytree | Pytree whose norm to compute. |

**Returns** — `jnp.ndarray` scalar: `sqrt(tree_vdot(tree, tree))`.

**Notes**

- Used as the convergence metric: `QQNState.error = tree_l2_norm(grad)`.
- Convergence is declared when `error < tol`.

**Example**

```python
tree = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([12.0])}
tree_l2_norm(tree)  # sqrt(9 + 16 + 144) = 13.0
```

---

## Differentiation Helper

### `make_value_and_grad(fun, has_aux=False)`

Wraps a callable with `jax.value_and_grad`, transparently handling the
`has_aux` flag used throughout QQN-JAX.

```python
def make_value_and_grad(fun: Callable, has_aux: bool = False) -> Callable:
    ...
```

**Parameters**

| Name      | Type       | Default | Description                                                                    |
|-----------|------------|---------|--------------------------------------------------------------------------------|
| `fun`     | `Callable` | —       | Objective `f(params, *args) -> scalar` or `(scalar, aux)`.                     |
| `has_aux` | `bool`     | `False` | If `True`, `fun` returns `(value, aux)` and grad is taken w.r.t. `value` only. |

**Returns** — A callable with signature:

- `has_aux=False`: `(params, *args) -> (value, grad)`
- `has_aux=True`:  `(params, *args) -> ((value, aux), grad)`

**Notes**

- This is a thin wrapper around `jax.value_and_grad(fun, has_aux=has_aux)`.
- The returned function is JIT-compatible and differentiable.
- When `has_aux=True`, auxiliary outputs are accessible via `QQNState.aux`.

**Example — basic usage**

```python
def f(x):
    return jnp.sum(x ** 2)


vg = make_value_and_grad(f)
value, grad = vg(jnp.array([1.0, 2.0, 3.0]))
# value = 14.0,  grad = [2.0, 4.0, 6.0]
```

**Example — with auxiliary output**

```python
def f(x):
    return jnp.sum(x ** 2), {"norm": jnp.linalg.norm(x)}


vg = make_value_and_grad(f, has_aux=True)
(value, aux), grad = vg(jnp.array([3.0, 4.0]))
# value = 25.0,  aux["norm"] = 5.0,  grad = [6.0, 8.0]
```

---

## QQN Geometry

These two functions define the **quadratic interpolation path** that is the
geometric heart of the QQN algorithm.

### The path

The QQN search direction at path parameter `t ∈ [0, 1]` is:

```
d(t) = t(1 − t)·(−∇f)  +  t²·(−H∇f)
```

where:

- `−∇f` is the **steepest-descent direction** (`grad_dir`).
- `−H∇f` is the **quasi-Newton direction** (`qn_dir`), e.g. from L-BFGS.

Key properties:

| `t`   | `d(t)`                | Tangent `d'(t)`          |
|-------|-----------------------|--------------------------|
| `0`   | `0` (origin)          | `−∇f` (steepest descent) |
| `0.5` | `0.25·(−∇f + −H∇f)`   | `0` (turning point)      |
| `1`   | `−H∇f` (pure QN step) | `−∇f + 2·(−H∇f)`         |

The line search traverses `t` over `[0, 1]`, selecting the `t*` that
satisfies the chosen acceptance criterion (Armijo, Wolfe, etc.).

---

### `quadratic_path(t, grad_dir, qn_dir)`

Evaluates the QQN blended direction `d(t)`.

```python
def quadratic_path(t, grad_dir, qn_dir):
    ...
```

**Parameters**

| Name       | Type   | Description                                       |
|------------|--------|---------------------------------------------------|
| `t`        | scalar | Path parameter in `[0, 1]`.                       |
| `grad_dir` | pytree | Steepest-descent direction `−∇f`.                 |
| `qn_dir`   | pytree | Quasi-Newton direction `−H∇f` (e.g. from L-BFGS). |

**Returns** — pytree with the same structure as `grad_dir` / `qn_dir`:
`d(t) = t(1−t)·grad_dir + t²·qn_dir`.

**Notes**

- At `t = 0`: returns the zero vector (both coefficients vanish).
- At `t = 1`: returns `qn_dir` exactly (pure quasi-Newton step).
- The path is JIT-compatible and differentiable w.r.t. `t`, `grad_dir`,
  and `qn_dir`.
- The new iterate is `x_new = tree_add_scaled(x, 1.0, quadratic_path(t, ...))`.

**Example**

```python
grad_dir = jnp.array([1.0, 0.0])
qn_dir = jnp.array([0.0, 1.0])

quadratic_path(0.0, grad_dir, qn_dir)  # [0.0, 0.0]
quadratic_path(0.5, grad_dir, qn_dir)  # [0.25, 0.25]
quadratic_path(1.0, grad_dir, qn_dir)  # [0.0,  1.0]
```

---

### `quadratic_path_derivative(t, grad_dir, qn_dir)`

Evaluates the derivative of the QQN path w.r.t. `t`:

```
d'(t) = (1 − 2t)·(−∇f)  +  2t·(−H∇f)
```

```python
def quadratic_path_derivative(t, grad_dir, qn_dir):
    ...
```

**Parameters**

| Name       | Type   | Description                       |
|------------|--------|-----------------------------------|
| `t`        | scalar | Path parameter in `[0, 1]`.       |
| `grad_dir` | pytree | Steepest-descent direction `−∇f`. |
| `qn_dir`   | pytree | Quasi-Newton direction `−H∇f`.    |

**Returns** — pytree with the same structure as `grad_dir` / `qn_dir`:
`d'(t) = (1−2t)·grad_dir + 2t·qn_dir`.

**Notes**

- At `t = 0`: `d'(0) = grad_dir = −∇f`. The path's initial tangent is
  the steepest-descent direction, so the line search starts with a
  guaranteed descent direction.
- At `t = 1`: `d'(1) = −grad_dir + 2·qn_dir`.
- Used by the **Wolfe curvature condition** and **Hager-Zhang** line
  searches, which require the directional derivative
  `⟨∇f(x + d(t)), d'(t)⟩` at the accepted point.
- Also used by the **cubic Hermite spline refinement** (`spline=True`) as
  the slope control point at each probe.
- Numerically consistent with `quadratic_path` via finite differences:
  `d'(t) ≈ (d(t+ε) − d(t−ε)) / (2ε)` to within `O(ε²)`.

**Example**

```python
grad_dir = jnp.array([1.0, 2.0, 3.0])
qn_dir = jnp.array([-1.0, 0.0, 1.0])

quadratic_path_derivative(0.0, grad_dir, qn_dir)
# [1.0, 2.0, 3.0]  (== grad_dir)

quadratic_path_derivative(1.0, grad_dir, qn_dir)
# [-1.0*1 + 2*(-1.0), -1.0*2 + 2*0.0, -1.0*3 + 2*1.0]
# = [-3.0, -2.0, -1.0]
```

---

## JIT / vmap / grad Compatibility

All functions in this module are **pure** (no side effects, no Python-level
control flow on traced values) and therefore compose freely with JAX
transformations:

```python
import jax

# JIT-compile the path evaluation.
jit_path = jax.jit(quadratic_path)

# Differentiate the path w.r.t. t.
dt_path = jax.grad(lambda t: jnp.sum(quadratic_path(t, g, q)))

# vmap over a batch of (t, grad_dir, qn_dir) triples.
batch_path = jax.vmap(quadratic_path)
```

---

## Implementation Notes

- **`tree_vdot` uses `jnp.vdot`** on each leaf pair, which conjugates the
  first argument for complex arrays. For real-valued optimization (the
  primary use case) this is identical to the standard dot product.
- **`tree_l2_norm` delegates to `tree_vdot`** to avoid a second tree
  traversal: `norm = sqrt(vdot(x, x))`.
- **`quadratic_path` and `quadratic_path_derivative`** use
  `jax.tree_util.tree_map` directly, so they work on any registered pytree
  type without modification.
- **NaN safety**: these utilities do not introduce any division or
  reciprocal operations, so they cannot generate NaNs from well-formed
  inputs. NaN-guarding for curvature reciprocals lives in `lbfgs.py` and
  `oracles.py`.

---

## See Also

| Module             | Relationship                                                                                         |
|--------------------|------------------------------------------------------------------------------------------------------|
| `solver.py`        | Calls `quadratic_path`, `tree_add_scaled`, `tree_l2_norm`.                                           |
| `line_search.py`   | Uses `tree_vdot` for directional derivatives; `quadratic_path_derivative` for Wolfe / HZ conditions. |
| `spline_search.py` | Uses `quadratic_path_derivative` as Hermite slope control points.                                    |
| `oracles.py`       | Uses `tree_vdot`, `tree_scale`, `tree_add_scaled`, `tree_negative`.                                  |
| `lbfgs.py`         | Uses `tree_vdot` for curvature inner products.                                                       |
| `regions.py`       | Uses `tree_add_scaled`, `tree_l2_norm` for projection and trust-region radius checks.                |