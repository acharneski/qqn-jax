"""Convergence and interface tests for the QQN solver."""

import jax
import jax.numpy as jnp
import numpy as np

from qqn_jax import QQN


def rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def quadratic(x):
    A = jnp.diag(jnp.array([1.0, 5.0, 10.0]))
    return 0.5 * x @ A @ x


def test_init_state():
    solver = QQN(quadratic, maxiter=50)
    x0 = jnp.array([1.0, 1.0, 1.0])
    state = solver.init_state(x0)
    assert int(state.iter) == 0
    np.testing.assert_allclose(state.value, quadratic(x0))


def test_single_update_decreases_value():
    solver = QQN(quadratic, maxiter=50)
    x0 = jnp.array([1.0, 1.0, 1.0])
    state = solver.init_state(x0)
    new_x, new_state = solver.update(x0, state)
    assert float(new_state.value) < float(state.value)
    assert int(new_state.iter) == 1


def test_converges_on_quadratic():
    solver = QQN(quadratic, maxiter=100, tol=1e-6)
    x0 = jnp.array([5.0, -3.0, 2.0])
    params, state = solver.run(x0)
    assert float(state.error) < 1e-4
    np.testing.assert_allclose(params, jnp.zeros(3), atol=1e-3)


def test_converges_on_rosenbrock():
    solver = QQN(rosenbrock, maxiter=500, tol=1e-5, history_size=15)
    x0 = jnp.array([-1.2, 1.0])
    params, state = solver.run(x0)
    # Rosenbrock minimum is at (1, 1).
    np.testing.assert_allclose(params, jnp.ones(2), atol=1e-2)


def test_run_is_jittable():
    solver = QQN(quadratic, maxiter=100, tol=1e-6)
    x0 = jnp.array([5.0, -3.0, 2.0])
    run_jit = jax.jit(solver.run)
    params, state = run_jit(x0)
    assert float(state.error) < 1e-4


def test_run_is_vmappable():
    solver = QQN(quadratic, maxiter=100, tol=1e-6)
    x0_batch = jnp.array([[5.0, -3.0, 2.0], [1.0, 1.0, 1.0], [-2.0, 4.0, -1.0]])
    batched = jax.vmap(solver.run)
    params, states = batched(x0_batch)
    assert params.shape == (3, 3)
    np.testing.assert_allclose(params, jnp.zeros((3, 3)), atol=1e-2)


def test_backtracking_line_search_option():
    solver = QQN(quadratic, maxiter=200, tol=1e-5, line_search="backtracking")
    x0 = jnp.array([5.0, -3.0, 2.0])
    params, state = solver.run(x0)
    assert float(state.error) < 1e-3


def test_has_aux():
    def fun_with_aux(x):
        value = quadratic(x)
        aux = {"norm": jnp.linalg.norm(x)}
        return value, aux

    solver = QQN(fun_with_aux, maxiter=100, tol=1e-6, has_aux=True)
    x0 = jnp.array([5.0, -3.0, 2.0])
    params, state = solver.run(x0)
    assert "norm" in state.aux
    assert float(state.error) < 1e-3
