# Projective Regions

`qqn_jax/regions.py` — constraint and remapping layer for the QQN optimizer.

---

## Overview

A **projective region** remaps a proposed parameter update onto a feasible
(or otherwise preferred) set before it is applied to the iterate. Because
QQN searches a single continuous quadratic path `d(t)`, regions integrate
cleanly: the line search navigates the *projected* path

```
d_R(t) = project_R(x, x + d(t)) − x
```

All regions are **pure, functional JAX** — they compose with `jit`, `vmap`,
`pmap`, and `grad`. When the region is the identity (`IdentityRegion` /
`region=None`), behavior is byte-for-byte equivalent to the un-regioned
optimizer.

---

## Core Interfaces

### `Region`

```python
class Region(NamedTuple):
    init: Callable[[Any], Any]
    project: Callable[[Any, Any, Any], Any]
    update: Callable[[Any, Any], Any]
```

A pure, composable projection interface. All three fields are plain
callables so a `Region` is itself a JAX-traceable pytree leaf.

| Field     | Signature                                 | Description                                                                                                                                         |
|-----------|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `init`    | `params -> region_state`                  | Build the initial region state from the starting parameters. Return `()` for stateless regions.                                                     |
| `project` | `(params, candidate, state) -> projected` | Map a proposed candidate back onto the feasible set. `params` is the *current* iterate (before the step); `candidate` is the proposed next iterate. |
| `update`  | `(state, info: RegionInfo) -> state`      | Update the region state after a step is accepted. No-op for stateless regions.                                                                      |

### `RegionInfo`

```python
class RegionInfo(NamedTuple):
    params: Any = None  # iterate x before the step
    new_params: Any = None  # accepted iterate x + α·d_R(t)
    pred_reduction: Any = None  # predicted reduction from the along-path model
    actual_reduction: Any = None  # actual reduction f(x) − f(x_new)
    t: Any = None  # chosen interpolation parameter
    step_size: Any = None  # accepted step size α
```

Passed to `Region.update` after each accepted step. Stateful regions
(e.g. `TrustRegion`) use `pred_reduction` and `actual_reduction` to
adapt their internal state.

---

## Built-in Regions

### `IdentityRegion`

```python
from qqn_jax.regions import IdentityRegion

region = IdentityRegion()
```

The trivial region: projection is the identity (no constraints). This is
the default when `region=None` is passed to `QQN`. It carries zero
overhead — the optimizer's behavior is byte-for-byte identical to the
un-regioned case.

**State:** `()` (stateless)

---

### `BoxRegion`

```python
from qqn_jax.regions import BoxRegion

region = BoxRegion(lo=-1.0, hi=1.0)
```

Enforce elementwise bounds `lo ≤ x_new ≤ hi` by clipping each coordinate.

| Argument | Default | Description                                                                                       |
|----------|---------|---------------------------------------------------------------------------------------------------|
| `lo`     | `None`  | Lower bound. Scalar, pytree broadcastable to the parameter structure, or `None` (mapped to `−∞`). |
| `hi`     | `None`  | Upper bound. Scalar, pytree broadcastable to the parameter structure, or `None` (mapped to `+∞`). |

**State:** `()` (stateless)

**Example — one-sided lower bound:**

```python
# Keep all weights non-negative.
region = BoxRegion(lo=0.0)
solver = QQN(fun, region=region)
```

**Example — per-coordinate bounds (pytree):**

```python
region = BoxRegion(lo=jnp.array([-1.0, 0.0]), hi=jnp.array([1.0, 2.0]))
```

---

### `OrthantRegion`

```python
from qqn_jax.regions import OrthantRegion

region = OrthantRegion(l1=0.0)
```

Constrain each step to remain within the **orthant** of the current
iterate's signs, zeroing coordinates that would cross zero. This is the
OWL-QN-style sparsity-inducing projection.

| Argument | Default | Description                                                                                                                                 |
|----------|---------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `l1`     | `0.0`   | L1 regularization strength. When `> 0`, the pseudo-gradient `∇f + l1·sign(x)` chooses the orthant for zero coordinates (OWL-QN convention). |

**Behavior:**

- For **nonzero** coordinates: the orthant is determined by `sign(x)`. A
  candidate coordinate that would flip sign is zeroed instead.
- For **zero** coordinates: the orthant is chosen from the step direction,
  soft-thresholded by `l1`. A coordinate only leaves zero when
  `|step| > l1`.

**State:** `()` (stateless)

**Example:**

```python
# Sparsity-inducing projection with L1 regularization.
region = OrthantRegion(l1=1e-3)
solver = QQN(fun, region=region)
```

---

### `QuantizationRegion`

```python
from qqn_jax.regions import QuantizationRegion

region = QuantizationRegion(bits=8, lo=-1.0, hi=1.0)
```

Confine weights to the quantization **interval** of their starting value.

Quantization with `bits` levels over `[lo, hi]` defines a uniform grid of
representable values spaced by `Δ = (hi − lo) / (2**bits − 1)`. Consecutive
grid points `[lo + k·Δ, lo + (k+1)·Δ]` form a *rounding interval*: every
real value in this interval rounds to one of the two surrounding grid points.

This region anchors the interval to the iterate's value **at the start of the
step** (`params`): a coordinate is free to explore the full width of *its own*
rounding interval, but is projected back at the **grid-point walls** instead
of being allowed to cross into a neighbouring interval.

The **cell center** — the midpoint between the two surrounding grid points —
acts as a natural attractor: it is the point of minimum rounding delta,
equidistant from both quantized neighbours. The optimizer is therefore drawn
toward low-rounding-error values rather than toward the grid points themselves.

| Argument | Default | Description                                                                                                                                                                                                   |
|----------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `bits`   | `None`  | Number of quantization bits; the grid has `2**bits` levels over `[lo, hi]`. Provide either `bits` or `step`.                                                                                                  |
| `step`   | `None`  | Explicit grid spacing `Δ` (overrides `bits` when given).                                                                                                                                                      |
| `lo`     | `-1.0`  | Lower bound of the quantization range.                                                                                                                                                                        |
| `hi`     | `1.0`   | Upper bound of the quantization range.                                                                                                                                                                        |
| `lock`   | `False` | If `True`, collapse each coordinate to the nearest grid point — hard-snap (true quantization, no exploration).                                                                                                |
| `window` | `1.0`   | Fraction of the half-interval the coordinate may explore when `lock=False`. `1.0` exposes the full interval `[grid_k, grid_{k+1}]`; smaller values tighten the box symmetrically about the interval midpoint. |

**State:** `()` (stateless)

**Example — 4-bit quantization-aware training:**

```python
# Weights may roam within their 4-bit quantization cell.
region = QuantizationRegion(bits=4, lo=-1.0, hi=1.0)
solver = QQN(fun, region=region)
```

**Example — hard quantization (snap to grid):**

```python
region = QuantizationRegion(bits=8, lo=0.0, hi=1.0, lock=True)
```

---

### `TrustRegion`

```python
from qqn_jax.regions import TrustRegion

region = TrustRegion(radius=1.0, adaptive=True)
```

Enforce `‖x_new − x‖₂ ≤ Δ` by radially clipping the step to the
L2-ball of radius `Δ` centered on the current iterate.

| Argument     | Default | Description                                                                                        |
|--------------|---------|----------------------------------------------------------------------------------------------------|
| `radius`     | `1.0`   | Initial trust-region radius `Δ`.                                                                   |
| `radius_max` | `1e3`   | Maximum allowed radius (caps expansion).                                                           |
| `adaptive`   | `True`  | If `True`, adapt the radius based on the ratio `ρ = ared / pred` of actual to predicted reduction. |
| `shrink`     | `0.5`   | Radius contraction factor applied when `ρ < eta_lo`.                                               |
| `expand`     | `2.0`   | Radius expansion factor applied when `ρ > eta_hi` and the step is at the boundary.                 |
| `eta_lo`     | `0.1`   | Lower threshold for `ρ`; below this the radius shrinks.                                            |
| `eta_hi`     | `0.75`  | Upper threshold for `ρ`; above this (at boundary) the radius expands.                              |

**State:** `TrustRegionState(radius: jnp.ndarray)`

**Adaptive update rule:**

```
ρ = ared / pred

if ρ < eta_lo:   radius ← shrink × radius
elif ρ > eta_hi and ‖step‖ ≈ radius:
                 radius ← min(expand × radius, radius_max)
else:            radius unchanged
```

A **floor safeguard** prevents the radius from falling below the realized
step length of any successful step (`ared > 0`): if a step of length `n`
was accepted, the geometry has demonstrably permitted it, so the region
must remain at least that large.

> **Design note:** On a curved QQN path the chord-length the radius
> constrains and the arc-length the predicted-reduction model integrates
> are different coordinates. The wide stable band `[eta_lo, eta_hi]` and
> the floor safeguard prevent the adaptive feedback from over-reacting to
> this chord/arc-length mismatch.

**Example:**

```python
region = TrustRegion(radius=1.0, adaptive=True)
solver = QQN(fun, maxiter=200, tol=1e-5, region=region)
```

---

### `NoDecreaseRegion`

```python
from qqn_jax.regions import NoDecreaseRegion

region = NoDecreaseRegion(secondary_grad_fn=lambda p: p)
```

Project each step onto the half-space `{s : ⟨∇g, s⟩ ≤ 0}`, ensuring the
step does **not increase** a secondary objective `g`.

Given a secondary objective `g` whose gradient is supplied by
`secondary_grad_fn(params) → ∇g`, this region removes only the component
of the proposed step that would *increase* `g` — preserving fitness on a
protected objective while optimizing the primary one.

The projection is the orthogonal removal of the offending component:

```
step   = candidate − x
c      = ⟨∇g, step⟩
s_proj = step − relu(c) / (‖∇g‖² + ε) · ∇g
```

Only the *positive* (g-increasing) component is removed; descent on `g`
passes through untouched.

| Argument            | Description                                                      |
|---------------------|------------------------------------------------------------------|
| `secondary_grad_fn` | `params -> ∇g` — gradient of the secondary objective to protect. |

**State:** `()` (stateless)

**Use cases:**

- **Continual learning:** protect previously learned tasks while fine-tuning
  on a new one.
- **Constrained fine-tuning:** prevent a regularizer or safety metric from
  increasing during optimization.

**Example:**

```python
# Protect a previously learned task (g = task_loss_A).
import jax

region = NoDecreaseRegion(secondary_grad_fn=jax.grad(task_loss_A))
solver = QQN(primary_loss, region=region)
```

---

### `Sequential`

```python
from qqn_jax.regions import Sequential

region = Sequential([BoxRegion(lo=-2.0, hi=2.0), TrustRegion(radius=1.0)])
```

Compose multiple regions by applying their projections **in order**:

```
project = R_k ∘ ... ∘ R_1
```

State is a tuple of child states; `update` fans out to each child
independently.

| Argument  | Description                                |
|-----------|--------------------------------------------|
| `regions` | Sequence of `Region` instances to compose. |

**State:** `tuple` of child region states (one per region, in order).

**Example — trust region inside a box:**

```python
from qqn_jax.regions import Sequential, BoxRegion, TrustRegion

region = Sequential([
    TrustRegion(radius=2.0, adaptive=True),
    BoxRegion(lo=0.0, hi=10.0),
])
solver = QQN(fun, region=region)
```

---

## Helper: `resolve_region`

```python
from qqn_jax.regions import resolve_region

region = resolve_region(None)  # → IdentityRegion()
region = resolve_region(BoxRegion())  # → BoxRegion() (passthrough)
```

Returns the region unchanged, or `IdentityRegion()` when `None` is passed.
Used internally by `QQN` to normalize the `region` constructor argument.

---

## Using Regions with `QQN`

Pass any `Region` instance (or `None`) to the `region` argument of `QQN`:

```python
from qqn_jax import QQN
from qqn_jax.regions import BoxRegion, TrustRegion, Sequential

# Elementwise bounds.
solver = QQN(fun, region=BoxRegion(lo=-1.0, hi=1.0))

# Adaptive trust region.
solver = QQN(fun, region=TrustRegion(radius=1.0, adaptive=True))

# Composed regions (applied in order).
solver = QQN(fun, region=Sequential([
    TrustRegion(radius=2.0),
    BoxRegion(lo=0.0),
]))

# No region (default).
solver = QQN(fun, region=None)
```

Because all regions are pure JAX, the solver remains fully `jit`-,
`vmap`-, and `pmap`-compatible regardless of which region is chosen:

```python
import jax

solver = QQN(fun, maxiter=100, region=BoxRegion(lo=-10.0, hi=10.0))
run_jit = jax.jit(solver.run)
params, state = run_jit(x0)
```

---

## Writing a Custom Region

A custom region is any `Region` NamedTuple with three callables. For a
stateless region, `init` returns `()` and `update` is a no-op:

```python
from qqn_jax.regions import Region
import jax
import jax.numpy as jnp


def L2BallRegion(radius: float = 1.0) -> Region:
    """Project the iterate onto the L2 ball ‖x‖₂ ≤ radius."""

    def init(params):
        return ()

    def project(params, candidate, state):
        norm = jnp.sqrt(sum(
            jnp.sum(l ** 2)
            for l in jax.tree_util.tree_leaves(candidate)
        ))
        scale = jnp.minimum(1.0, radius / (norm + 1e-12))
        return jax.tree_util.tree_map(lambda c: scale * c, candidate)

    def update(state, info):
        return state

    return Region(init=init, project=project, update=update)
```

For a **stateful** region, `init` returns a NamedTuple (or any pytree),
and `update` returns a new state of the same structure:

```python
from typing import NamedTuple
import jax.numpy as jnp
from qqn_jax.regions import Region, RegionInfo


class MyState(NamedTuple):
    step_count: jnp.ndarray


def CountingRegion() -> Region:
    """Stateful region that counts accepted steps (illustrative)."""

    def init(params):
        return MyState(step_count=jnp.zeros((), dtype=jnp.int32))

    def project(params, candidate, state):
        return candidate  # identity projection

    def update(state, info: RegionInfo):
        return MyState(step_count=state.step_count + 1)

    return Region(init=init, project=project, update=update)
```

---

## Design Notes

### The path is the search space

The line search traverses the path parameter `t` directly. Each evaluated
`x + d(t)` is a *state*, not a direction to be independently re-scaled.
The region projects the *candidate iterate*, not the direction vector, so
the projection is geometrically meaningful at every point on the path.

### Chord vs. arc length in `TrustRegion`

On a curved QQN path, the chord-length `‖x_new − x‖` that the trust
radius constrains and the arc-length that the predicted-reduction model
integrates are different coordinates. The adaptive update rule uses a wide
stable band `[eta_lo, eta_hi]` and a floor safeguard (the radius never
falls below the length of a successful step) to prevent the feedback from
over-reacting to this mismatch.

### NaN safety

All curvature reciprocals and norm divisions are guarded with a small
`eps` so that masked-out branches never backpropagate NaNs under
`jax.grad`.

### Stateless regions are zero-overhead

`IdentityRegion`, `BoxRegion`, `OrthantRegion`, `QuantizationRegion`,
`NoDecreaseRegion`, and `Sequential` (when all children are stateless) all
carry `()` as their state. JAX traces through the `()` pytree with no
dynamic allocation.

---

## API Reference

| Symbol                                                                      | Kind         | Description                                                   |
|-----------------------------------------------------------------------------|--------------|---------------------------------------------------------------|
| `Region`                                                                    | `NamedTuple` | Core projection interface (`init`, `project`, `update`).      |
| `RegionInfo`                                                                | `NamedTuple` | Step information passed to `Region.update`.                   |
| `RegionState`                                                               | type alias   | `Any` — backwards-compatibility alias for region state types. |
| `IdentityRegion()`                                                          | factory      | No-op projection (default).                                   |
| `BoxRegion(lo, hi)`                                                         | factory      | Elementwise clipping to `[lo, hi]`.                           |
| `OrthantRegion(l1)`                                                         | factory      | OWL-QN-style orthant constraint.                              |
| `QuantizationRegion(bits, step, lo, hi, lock, window)`                      | factory      | Quantization-cell confinement.                                |
| `TrustRegion(radius, radius_max, adaptive, shrink, expand, eta_lo, eta_hi)` | factory      | L2-ball step constraint with optional adaptive radius.        |
| `TrustRegionState`                                                          | `NamedTuple` | State for `TrustRegion` (`radius`).                           |
| `NoDecreaseRegion(secondary_grad_fn)`                                       | factory      | Half-space projection protecting a secondary objective.       |
| `Sequential(regions)`                                                       | factory      | Compose regions by sequential projection.                     |
| `resolve_region(r)`                                                         | function     | Return `r`, or `IdentityRegion()` if `r is None`.             |