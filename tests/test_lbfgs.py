"""Tests for the L-BFGS oracle."""

import jax.numpy as jnp
import numpy as np

from qqn_jax.lbfgs import (
    init_lbfgs_state,
    lbfgs_direction,
    update_lbfgs_history,
)


def test_initial_direction_is_negative_gradient():
    # With no history, H0 = gamma*I = I, so direction = -grad.
    params = jnp.array([1.0, 2.0])
    grad = jnp.array([0.5, -0.5])
    state = init_lbfgs_state(params, grad, history_size=5)
    d = lbfgs_direction(state, grad)
    np.testing.assert_allclose(d, -grad, atol=1e-7)


def test_history_update_curvature():
    # Quadratic f(x) = 0.5 xᵀA x with A = diag(1, 10).
    A = jnp.array([[1.0, 0.0], [0.0, 10.0]])

    def grad(x):
        return A @ x

    history_size = 5
    x0 = jnp.array([1.0, 1.0])
    g0 = grad(x0)
    state = init_lbfgs_state(x0, g0, history_size)

    x1 = jnp.array([0.5, 0.5])
    g1 = grad(x1)
    state = update_lbfgs_history(state, x1, g1, history_size)

    # Count should have incremented (positive curvature).
    assert int(state.count) == 1

    d = lbfgs_direction(state, g1)
    # The L-BFGS direction should be a descent direction.
    assert float(jnp.vdot(d, g1)) < 0.0


def test_rejects_negative_curvature():
    params = jnp.array([0.0, 0.0])
    grad0 = jnp.array([1.0, 1.0])
    state = init_lbfgs_state(params, grad0, history_size=5)

    # Construct an update with yᵀs < 0 (negative curvature).
    bad_params = jnp.array([1.0, 0.0])  # s = (1, 0)
    bad_grad = jnp.array([-2.0, 1.0])  # y = (-3, 0), yᵀs = -3 < 0
    new_state = update_lbfgs_history(state, bad_params, bad_grad, 5)
    assert int(new_state.count) == 0
