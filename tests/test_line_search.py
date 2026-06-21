"""Tests for line search strategies."""

import jax
import jax.numpy as jnp

from qqn_jax.line_search import (
    backtracking_search,
    strong_wolfe_search,
)


def quad_value_and_grad(x):
    # f(x) = 0.5 * ||x||^2, grad = x.
    return 0.5 * jnp.vdot(x, x), x


def test_backtracking_decreases_value():
    x = jnp.array([2.0, 2.0])
    value, grad = quad_value_and_grad(x)
    direction = -grad  # steepest descent
    res = backtracking_search(quad_value_and_grad, x, direction, value, grad)
    assert float(res.new_value) < float(value)
    assert bool(res.done)


def test_strong_wolfe_decreases_value():
    x = jnp.array([2.0, 2.0])
    value, grad = quad_value_and_grad(x)
    direction = -grad
    res = strong_wolfe_search(quad_value_and_grad, x, direction, value, grad)
    assert float(res.new_value) < float(value)


def test_strong_wolfe_finds_exact_step_for_quadratic():
    # For f = 0.5||x||^2 along d = -x, the exact minimizer is alpha = 1.
    x = jnp.array([3.0, -1.0])
    value, grad = quad_value_and_grad(x)
    direction = -grad
    res = strong_wolfe_search(quad_value_and_grad, x, direction, value, grad)
    # New params should be near the origin.
    assert float(jnp.linalg.norm(res.new_params)) < 0.5


def test_line_search_jittable():
    x = jnp.array([2.0, 2.0])
    value, grad = quad_value_and_grad(x)
    direction = -grad
    fn = jax.jit(
        lambda x, d, v, g: (
            strong_wolfe_search(quad_value_and_grad, x, d, v, g).step_size
        )
    )
    step = fn(x, direction, value, grad)
    assert float(step) > 0.0
