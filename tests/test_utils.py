"""Tests for qqn_jax.utils."""

import jax.numpy as jnp
import numpy as np

from qqn_jax.utils import (
    quadratic_path,
    quadratic_path_derivative,
    tree_l2_norm,
    tree_vdot,
)


def test_quadratic_path_endpoints():
    grad_dir = jnp.array([1.0, 2.0, 3.0])
    qn_dir = jnp.array([-1.0, 0.0, 1.0])

    # t = 0 -> zero (both terms vanish).
    d0 = quadratic_path(0.0, grad_dir, qn_dir)
    np.testing.assert_allclose(d0, jnp.zeros(3), atol=1e-7)

    # t = 1 -> pure L-BFGS direction.
    d1 = quadratic_path(1.0, grad_dir, qn_dir)
    np.testing.assert_allclose(d1, qn_dir, atol=1e-7)


def test_quadratic_path_derivative_at_zero():
    grad_dir = jnp.array([1.0, 2.0, 3.0])
    qn_dir = jnp.array([-1.0, 0.0, 1.0])
    # d'(0) = (-∇f), i.e. grad_dir.
    dprime = quadratic_path_derivative(0.0, grad_dir, qn_dir)
    np.testing.assert_allclose(dprime, grad_dir, atol=1e-7)


def test_tree_vdot_and_norm():
    a = jnp.array([3.0, 4.0])
    assert float(tree_vdot(a, a)) == 25.0
    assert float(tree_l2_norm(a)) == 5.0
