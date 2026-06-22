# qqn-jax

**Quadratic Quasi-Newton (QQN)** — a JAX/Optax optimizer that blends steepest
descent with a quasi-Newton oracle (L-BFGS by default) along a smooth quadratic
path, navigated by a robust line search.

```
d(t) = t(1 - t)(-∇f) + t²(-H∇f),   t ∈ [0, 1]
```

At `t = 0` the path is pure gradient descent; at `t = 1` it is the pure oracle
(L-BFGS) direction. The line search picks the interpolation parameter `t` and
the step size `α` together, automatically discovering the right blend of
gradient and oracle at every iteration.

---

## Why QQN?

QQN is a **combiner** for three classic optimization ingredients:

1. **Gradient** — the reliable steepest-descent signal `-∇f(x)`.
2. **Oracle** — a curvature-aware direction `-H∇f(x)` (L-BFGS, Momentum,
   Shampoo, …).
3. **Search** — the line search that traverses the path and guarantees descent.

The quadratic path makes the search the *glue*: it blends the gradient and the
oracle coherently while retaining global-convergence guarantees from the
steepest-descent fallback at `t = 0`.

See [`algorithm.md`](docs/algorithm.md) for the full conceptual treatment.

---

## Installation

Always work inside a virtual environment (see [`python.md`](docs/python.md)):

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux

pip install qqn-jax
```

For local development (editable install with dev extras):

```bash
pip install -e ".[dev]"
```

QQN is built directly on **JAX** and **Optax**. The L-BFGS scaling and the
zoom (Strong Wolfe) line search are delegated to Optax; the rest of the solver
is pure, functional JAX. If you need GPU support, install the matching CUDA
wheel of `jaxlib` (see [`libraries.md`](docs/libraries.md)).

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
```

### JIT, vmap, and grad

Because QQN is implemented in JAX's functional model, it composes with the
standard transforms out of the box:

```python
import jax

# JIT-compiled solve (XLA + GPU/TPU dispatch)
run_jit = jax.jit(QQN(fun).run)
params, state = run_jit(init)

# Batched over many starting points
batched = jax.vmap(QQN(fun).run, in_axes=(0,))
params_batch, states = batched(init_batch)
```

---

## Configuration

```python
QQN(
    fun,
    maxiter=100,
    tol=1e-5,
    history_size=10,
    line_search="armijo",  # "armijo" (default) | "backtracking" |
    # "strong_wolfe" | "hager_zhang" |
    # "fixed" | "spline"
    line_search_options=None,  # dict of kwargs for the line search
    has_aux=False,
    t_grid=None,
    oracle="lbfgs",  # "lbfgs" | "momentum" | "shampoo" | Oracle
    region=None,  # Region | None
)
```

With all defaults, QQN behaves as a tightly-coupled gradient + L-BFGS optimizer
with an Armijo backtracking line search.

### Swappable Oracles

The `t = 1` endpoint of the path is supplied by an **oracle**. Swap it freely:

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

When `oracle="lbfgs"` (the default), the optimizer is byte-for-byte equivalent
to the reference behavior. See [`oracles.md`](docs/oracles.md) for details.

### Projective Regions

Constrain or remap the search onto a feasible set with a **region**. The line
search then navigates the *projected* path:

```python
from qqn_jax.regions import BoxRegion, TrustRegion, Sequential

region = Sequential([
    BoxRegion(lo=0.0, hi=1.0),
    TrustRegion(radius=0.5),
])

solver = QQN(fun, region=region)
```

When `region=None`, behavior is identical to the unconstrained optimizer.
See [`regions.md`](docs/regions.md) for details.

---

## Documentation

| Document                                    | Description                                                         |
|---------------------------------------------|---------------------------------------------------------------------|
| [`algorithm.md`](docs/algorithm.md)         | The QQN algorithm: quadratic path, line search, guarantees.         |
| [`oracles.md`](docs/oracles.md)             | The oracle abstraction (L-BFGS, Momentum, Shampoo, combinators).    |
| [`regions.md`](docs/regions.md)             | Projective regions (box, trust-region, orthant, combinators).       |
| [`spline_search.md`](docs/spline_search.md) | Cubic-Hermite spline line search that reuses gradient measurements. |
| [`python.md`](docs/python.md)               | venv, testing, linting, and publishing workflow.                    |
| [`libraries.md`](docs/libraries.md)         | Installing JAX/jaxlib and the MNIST dataset.                        |
| [`results.md`](docs/results.md)             | Empirical MNIST benchmark: QQN vs. baselines and component sweeps.  |
| [`conclusions.md`](docs/conclusions.md)     | Synthesis of the experimental findings and design-claim validation. |
## Empirical Results
On a smooth, deterministic, full-batch softmax-MNIST benchmark, QQN reaches a
shared loss target in **fewer iterations than L-BFGS** (65 vs. 70) while
running ~1.3× faster in wall-clock time, and in ~4× fewer iterations than
Adam. Deep L-BFGS memory is the dominant convergence-speed lever, the line
search trades wall-time (not convergence speed), and regions act as
low-overhead safeguards. The best-of-breed stack (deep L-BFGS memory +
backtracking + adaptive trust-region, `QQN-L50BTTR`) reaches the target in
just **41 iterations**.
See [`results.md`](docs/results.md) for the full benchmark and
[`conclusions.md`](docs/conclusions.md) for the analysis. Reproduce with:
```bash
python examples/mnist_comparison.py
```
---

---


## Theoretical Guarantees

Under standard smoothness assumptions, and contingent on a line search that
satisfies sufficient-decrease conditions:

- **Global convergence** — guaranteed by the steepest-descent fallback at `t = 0`.
- **Superlinear convergence** — near the optimum, when the oracle direction
  dominates.
- **Descent property** — every accepted step decreases the objective (enforced
  by the line search).

---

## Development

```bash
pip install -e ".[dev]"

pytest                 # run the test suite
pytest --cov=qqn_jax   # with coverage
ruff format .          # auto-format
ruff check . --fix     # lint + autofix
```

See [`python.md`](docs/python.md) for the full developer and publishing workflow.

---

## License

See the repository `LICENSE` file.