"""Basic QQN usage examples.

Run with:  python examples/basic_usage.py
"""

import jax
import jax.numpy as jnp

from qqn_jax import QQN


def rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def main():
    x0 = jnp.array([-1.2, 1.0])

    # 1. Basic usage.
    solver = QQN(rosenbrock, maxiter=500, tol=1e-6)
    params, state = solver.run(x0)
    print("=== Basic ===")
    print(f"  solution: {params}")
    print(f"  value:    {float(state.value):.3e}")
    print(f"  error:    {float(state.error):.3e}")
    print(f"  iters:    {int(state.iter)}")

    # 2. JIT-compiled.
    run_jit = jax.jit(solver.run)
    params_j, state_j = run_jit(x0)
    print("\n=== JIT ===")
    print(f"  solution: {params_j}")

    # 3. Batched over multiple starting points.
    x0_batch = jnp.array([[-1.2, 1.0], [2.0, 2.0], [-0.5, 0.5]])
    batched = jax.vmap(solver.run)
    params_b, states_b = batched(x0_batch)
    print("\n=== Batched ===")
    for i in range(x0_batch.shape[0]):
        print(f"  start {x0_batch[i]} -> {params_b[i]}")


if __name__ == "__main__":
    main()
