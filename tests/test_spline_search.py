"""Tests for the cubic Hermite spline line search."""

import jax
import jax.numpy as jnp

from qqn_jax import QQN, spline_search
from qqn_jax.utils import make_value_and_grad


def _quadratic(x):
    # Simple convex quadratic with minimum at the origin.
    return jnp.sum(x**2)


def _rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def test_spline_search_decreases_on_quadratic():
    vg = make_value_and_grad(_quadratic)
    params = jnp.array([2.0, -3.0])
    value, grad = vg(params)
    direction = -grad  # steepest descent

    res = spline_search(vg, params, direction, value, grad, init_step=1.0)
    # Spline search must not increase the objective.
    assert res.new_value <= value + 1e-6
    # On an exact quadratic, the minimizer along -grad is at alpha ~ 0.5.
    assert jnp.isfinite(res.step_size)
    assert bool(res.done)


def test_spline_search_jit_compatible():
    vg = make_value_and_grad(_quadratic)
    params = jnp.array([1.0, 1.0, 1.0])
    value, grad = vg(params)
    direction = -grad

    fn = jax.jit(lambda p, d, v, g: spline_search(vg, p, d, v, g))
    res = fn(params, direction, value, grad)
    assert res.new_value <= value + 1e-6


def test_spline_search_in_solver():
    solver = QQN(_rosenbrock, maxiter=500, tol=1e-5, line_search="spline")
    params, state = solver.run(jnp.array([-1.2, 1.0]))
    # Should make meaningful progress toward [1, 1].
    assert float(state.value) < 1.0


def test_spline_search_vmap_starting_points():
    solver = QQN(_quadratic, maxiter=50, tol=1e-6, line_search="spline")
    starts = jnp.array([[2.0, 2.0], [-1.0, 3.0], [0.5, -0.5]])
    params, states = jax.vmap(solver.run)(starts)
    # Each run should converge near the origin.
    assert jnp.all(states.value < 1e-3)
