# qqn-jax

**Quadratic Quasi-Newton (QQN)** — a JAX/Optax optimizer that blends
steepest descent with a quasi-Newton oracle (L-BFGS by default) along a
smooth quadratic path, navigated by a robust line search.

```
d(t) = t(1 - t)(-∇f) + t²(-H∇f),   t ∈ [0, 1]
```

At `t = 0` the path's tangent is pure gradient descent; at `t = 1` the
endpoint is the pure oracle (L-BFGS) direction. A single line search picks
the interpolation parameter `t` and the step size `α` **together**,
automatically discovering the right blend of first- and second-order
behavior at every iteration — with **no learning rate to tune**.

---

## Table of Contents

- [Why QQN?](#why-qqn)
- [When to Use QQN](#when-to-use-qqn)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [JAX Transforms](#jax-transforms-jit-vmap-pmap-grad)
- [Configuration](#configuration)
  - [Swappable Oracles](#swappable-oracles)
  - [Pluggable Line Searches](#pluggable-line-searches)
  - [Projective Regions](#projective-regions)
  - [Spline Refinement](#spline-refinement)
- [Theoretical Guarantees](#theoretical-guarantees)
- [Empirical Results](#empirical-results)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)

---

## Why QQN?

QQN is a **combiner** for three classic optimization ingredients:

1. **Gradient** — the reliable steepest-descent signal `-∇f(x)`.
2. **Oracle** — a curvature-aware direction `-H∇f(x)` (L-BFGS, Momentum,
   Shampoo, …).
3. **Search** — the line search that traverses the path and guarantees
   descent.

The quadratic path makes the search the *glue*: it blends the gradient and
the oracle coherently while retaining global-convergence guarantees from
the steepest-descent fallback at `t = 0`.

The key idea is that the gradient/oracle blend is **discovered
geometrically** rather than **tuned numerically**. There is no global
learning rate to sweep, no `β₁/β₂` schedule, no warmup — QQN introduces no
hyperparameters of its own beyond those of the components it composes.

A bonus of this design: many classical optimizers (L-BFGS, Newton,
momentum, Barzilai-Borwein, trust-region, OWL-QN, projected gradient) are
**special cases** of QQN under particular configurations of its four axes.
See [`equivalences.md`](docs/theory/equivalences.md).

See [`algorithm.md`](docs/theory/algorithm.md) for the full conceptual
treatment and [`genesis.md`](docs/genesis.md) for the algorithm's history.

---

## When to Use QQN

QQN is **not** a drop-in replacement for Adam on every problem. Its value
compounds on **ill-curved, anisotropic landscapes** where naïve direction
choices stall, oscillate, or diverge.

| Situation                                                      | Prefer           |
|----------------------------------------------------------------|------------------|
| Large-scale, noisy, stochastic minibatch training              | **Adam**         |
| Tight memory budget, very high dimension                       | **Adam / SGD**   |
| Smooth, full-batch, ill-conditioned objective                  | **QQN**          |
| Complex / anisotropic curvature where step tuning is brittle   | **QQN**          |
| Curvature that is locally useful but globally unreliable       | **QQN**          |
| You want a parameter-free, self-tuning blend of GD and L-BFGS  | **QQN**          |
| Bound / orthant / trust constraints alongside curvature        | **QQN + region** |

For everyday large-scale stochastic training, **Adam remains faster per
step and more memory efficient**. QQN earns its keep when curvature
structure matters and a robust line search is affordable. See
[`positioning.md`](docs/positioning.md) for the full discussion.

---

## Installation

Always work inside a virtual environment (see
[`python.md`](docs/project/python.md)):

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux

pip install qqn-jax
```

For local development (editable install with dev extras):

```bash
pip install -e ".[dev]"
```

QQN is built directly on **JAX** and **Optax** (and uses `chex` and
`jaxtyping`). The L-BFGS scaling and the zoom (Strong Wolfe) line search are
delegated to Optax; the rest of the solver is pure, functional JAX. If you
need GPU support, install the matching CUDA wheel of `jaxlib` (see
[`libraries.md`](docs/project/libraries.md)).

---

## Quick Start

```python
import jax.numpy as jnp
from qqn_jax import QQN

# Rosenbrock function
def fun(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

solver = QQN(fun, maxiter=100, tol=1e-6)
init = jnp.array([-1.2, 1.0])
params, state = solver.run(init)

print(params)        # ~ [1.0, 1.0]
print(state.value)   # ~ 0.0
print(state.iter)    # iterations taken
print(state.error)   # final gradient L2 norm
```

### The `QQN` interface

QQN follows a JAXopt-style `init_state` / `update` / `run` API:

| Method                          | Description                                          |
|---------------------------------|------------------------------------------------------|
| `init_state(params, *args)`     | Build the initial `QQNState` at `params`.            |
| `update(params, state, *args)`  | Perform one QQN iteration → `(new_params, new_state)`.|
| `run(init_params, *args)`       | Run to convergence (or `maxiter`) → `(params, state)`.|

---

## JAX Transforms (jit, vmap, pmap, grad)

Because the whole solver is written in JAX's functional model and uses
`lax.while_loop` internally, a full optimization run is itself a single
traceable, differentiable, vmappable operation. It composes with the
standard transforms out of the box:

```python
import jax

# JIT-compiled solve (XLA + GPU/TPU dispatch)
run_jit = jax.jit(QQN(fun).run)
params, state = run_jit(init)

# Batched over many starting points — solve a whole batch at once.
batched = jax.vmap(QQN(fun).run, in_axes=(0,))
params_batch, states = batched(init_batch)
```

A single bad start in a vmapped batch will not waste the rest of the
batch's iterations: a run terminates early if an iterate becomes
non-finite.

---

## Configuration

```python
QQN(
    fun,
    maxiter=100,
    tol=1e-5,
    history_size=10,            # L-BFGS memory size m
    line_search="armijo",       # "armijo" (default) | "backtracking" |
                                # "strong_wolfe" | "hager_zhang" |
                                # "fixed" | "spline"
    line_search_options=None,   # dict of kwargs for the line search
    spline=False,               # cubic-Hermite spline refinement
    has_aux=False,
    oracle="lbfgs",             # "lbfgs" | "momentum" | "secant" |
                                # "shampoo" | "anderson" | ... | Oracle
    region=None,                # Region | None
    feed_probes_to_oracle=False,
    probe_descent_gate=True,
    max_probes=32,
)
```

With all defaults, QQN behaves as a tightly-coupled gradient + L-BFGS
optimizer with an Armijo backtracking line search.

### Swappable Oracles

The `t = 1` endpoint `-H∇f` of the path is supplied by an **oracle**. Swap
it freely by name or with a custom `Oracle` instance:

| Name                | Endpoint                                              |
|---------------------|-------------------------------------------------------|
| `"lbfgs"` (default) | limited-memory BFGS two-loop recursion                |
| `"momentum"`        | heavy-ball / exponentially-weighted gradient          |
| `"secant"`          | Barzilai-Borwein step (matrix-free, `O(n)` memory)    |
| `"shampoo"`         | structure-aware preconditioning                       |
| `"anderson"`        | Anderson (Type-II) acceleration                       |
| `"lbfgs+secant"`    | safeguarded fallback (deep curvature + light backup)  |

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
equivalent to the reference behavior (numerically equivalent up to
floating-point reordering). See [`oracles.md`](docs/theory/oracles.md) for
details.

### Pluggable Line Searches

```python
QQN(fun, line_search="armijo")        # default; robust efficiency winner
QQN(fun, line_search="backtracking")
QQN(fun, line_search="strong_wolfe")
QQN(fun, line_search="hager_zhang")
QQN(fun, line_search="fixed")

# Forward extra keyword arguments to the inner line search.
QQN(fun, line_search="backtracking",
    line_search_options={"c1": 1e-3, "shrink": 0.6, "max_iter": 10})
```

> **Note:** `"strong_wolfe"` can over-restrict the quadratic-path step and
> fail to converge on some problems; the Armijo / backtracking family is
> the recommended default for smooth, full-batch objectives.

### Projective Regions

Constrain or remap the search onto a feasible set with a **region**. The
line search then navigates the *projected* path:

| Region              | Effect                                               |
|---------------------|-------------------------------------------------------|
| `IdentityRegion`    | default, zero overhead                                |
| `BoxRegion`         | elementwise bounds `[lo, hi]`                         |
| `OrthantRegion`     | OWL-QN-style L1 sparsity                              |
| `TrustRegion`       | adaptive `‖x_new − x‖₂ ≤ Δ`                           |
| `NoDecreaseRegion`  | protect a secondary objective                         |
| `Sequential`        | compose multiple regions (applied in order)          |

```python
from qqn_jax.regions import BoxRegion, TrustRegion, Sequential

region = Sequential([
    BoxRegion(lo=0.0, hi=1.0),
    TrustRegion(radius=0.5),
])

solver = QQN(fun, region=region)
```

When `region=None`, behavior is identical to the unconstrained optimizer.
See [`regions.md`](docs/theory/regions.md) for details.

### Spline Refinement

Orthogonal to the line search: each probe along the consistent path is
reused as a control point of a cubic Hermite spline, whose stationary
points are probed to improve on the inner search's accepted step.

```python
QQN(fun, line_search="backtracking", spline=True)
# Equivalent shorthand:
QQN(fun, line_search="spline")
```

See [`spline_search.md`](docs/theory/spline_search.md) for details.

---

## Theoretical Guarantees

Under standard smoothness assumptions, and contingent on a line search that
satisfies sufficient-decrease conditions:

- **Global convergence** — guaranteed by the steepest-descent fallback at
  `t = 0`, *regardless of oracle direction quality*, precisely because the
  path begins tangent to `-∇f`.
- **Superlinear convergence** — near the optimum, when the oracle direction
  dominates.
- **Descent property** — every accepted step decreases the objective
  (enforced by the line search).

Importantly, the *hybrid algorithm itself* needs only **`C⁰` continuity
along the path** to make monotone progress — the sufficient-decrease test
compares function *values*. Smoothness sharpens the rate proofs and
strengthens the oracle, but is not a precondition for descent. This makes
QQN well-suited to merely-piecewise-smooth objectives (ReLU networks,
max-pooling, hinge/L1 terms). See [`ideal_problem.md`](docs/ideal_problem.md)
for what QQN actually requires versus what merely helps.

---

## Empirical Results

---

## Documentation

| Document                                                | Description                                                         |
|---------------------------------------------------------|---------------------------------------------------------------------|
| [`algorithm.md`](docs/theory/algorithm.md)              | The QQN algorithm: quadratic path, line search, guarantees.         |
| [`oracles.md`](docs/theory/oracles.md)                  | The oracle abstraction (L-BFGS, Momentum, Shampoo, combinators).    |
| [`regions.md`](docs/theory/regions.md)                  | Projective regions (box, trust-region, orthant, combinators).       |
| [`spline_search.md`](docs/theory/spline_search.md)      | Cubic-Hermite spline line search that reuses gradient measurements. |
| [`equivalences.md`](docs/theory/equivalences.md)        | Classical optimizers as QQN special cases.                          |
| [`positioning.md`](docs/positioning.md)                 | Where QQN fits relative to Adam / L-BFGS.                           |
| [`ideal_problem.md`](docs/ideal_problem.md)             | What QQN actually requires vs. what merely helps.                   |
| [`genesis.md`](docs/genesis.md)                         | The origin and evolution of the QQN algorithm.                      |
| [`results.md`](docs/results.md)                         | Empirical MNIST benchmark: QQN vs. baselines and component sweeps.  |
| [`conclusions.md`](docs/conclusions.md)                 | Synthesis of the experimental findings and design-claim validation. |
| [`python.md`](docs/project/python.md)                   | venv, testing, linting, and publishing workflow.                    |
| [`libraries.md`](docs/project/libraries.md)             | Installing JAX/jaxlib and the MNIST dataset.                        |

---

## Development

```bash
pip install -e ".[dev]"

pytest                 # run the test suite
pytest --cov=qqn_jax   # with coverage
ruff format .          # auto-format
ruff check . --fix     # lint + autofix
```

See [`python.md`](docs/project/python.md) for the full developer and
publishing workflow.

---

## License

See the repository `LICENSE` file.