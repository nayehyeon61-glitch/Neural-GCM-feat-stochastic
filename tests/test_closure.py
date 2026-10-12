"""Numerical and causal contracts for the augmented memory SDE."""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ngcm_memory.closure import ClosureConfig, closure_step, init_params, initialize_memory


def _constant_diffusion_params(config, sigma_fraction=0.5):
    params = init_params(jax.random.PRNGKey(0), config)
    params["diffusion_w"] = jnp.zeros_like(params["diffusion_w"])
    params["diffusion_b"] = jnp.full_like(params["diffusion_b"], np.log(sigma_fraction / (1.0 - sigma_fraction)))
    return params


def test_constant_features_exact_memory_integral_is_step_partition_invariant():
    config = ClosureConfig(feature_dim=2, levels=2, hidden_dim=4, enable_stochastic=False)
    params = init_params(jax.random.PRNGKey(0), config)
    features = jnp.array([[[0.2], [1.2]], [[-0.5], [0.9]]])
    state = initialize_memory((2, 1), config, jax.random.PRNGKey(12))
    full_delta, full_state = closure_step(params, state, features, 12.0, config)
    first_delta, half_state = closure_step(params, state, features, 6.0, config)
    second_delta, split_state = closure_step(params, half_state, features, 6.0, config)
    np.testing.assert_allclose(full_delta, first_delta + second_delta, atol=2e-8, rtol=2e-5)
    np.testing.assert_allclose(full_state.memory, split_state.memory, atol=2e-7, rtol=2e-5)
    np.testing.assert_array_equal(full_state.key, state.key)


@pytest.mark.parametrize("dt_hours", [0.02, 0.5, 8.0])
def test_ou_joint_endpoint_integral_matches_analytic_sde_moments(dt_hours):
    # Many independent columns estimate a joint Gaussian law. This catches the
    # common error of using dt * endpoint instead of integrating colored noise.
    n = 50_000
    tau = 2.0
    config = ClosureConfig(feature_dim=1, levels=1, hidden_dim=1, stochastic_tau_hours=tau,
                           max_diffusion_k_sqrt_hour=0.8, enable_spatial=False, enable_memory=False)
    params = _constant_diffusion_params(config)
    state = initialize_memory((n, 1), config, jax.random.PRNGKey(17))
    delta, new_state = closure_step(params, state, jnp.zeros((1, n, 1)), dt_hours, config)
    samples = np.stack([np.asarray(new_state.ou).ravel(), tau * np.asarray(delta).ravel()])
    # Independent continuous-time analytic formula evaluated in float64.
    sigma = 0.8 * (1e-6 + (1.0 - 1e-6) * 0.5)
    a = np.exp(-dt_hours / tau)
    endpoint_var = sigma**2 * tau / 2.0 * (1.0 - a**2)
    integral_var = sigma**2 * tau**2 * (dt_hours - 2 * tau * (1 - a) + tau / 2 * (1 - a**2))
    covariance = sigma**2 * tau**2 / 2.0 * (1.0 - a)**2
    expected = np.array([[endpoint_var, covariance], [covariance, integral_var]])
    np.testing.assert_allclose(np.cov(samples, ddof=0), expected, rtol=0.035, atol=1e-10)
    assert abs(samples[0].mean()) < 0.02 * np.sqrt(endpoint_var)
    assert abs(samples[1].mean()) < 0.02 * np.sqrt(integral_var)


def test_colored_noise_small_time_integral_has_cubic_cold_start_variance():
    # A white-noise increment scales as dt, endpoint*dt has the wrong constant;
    # exact integrated OU from zero scales as sigma^2 dt^3/(3 tau^2).
    config = ClosureConfig(feature_dim=1, levels=1, hidden_dim=1, stochastic_tau_hours=10.0,
                           enable_spatial=False, enable_memory=False)
    params = _constant_diffusion_params(config)
    state = initialize_memory((40_000, 1), config, jax.random.PRNGKey(42))
    features = jnp.zeros((1, 40_000, 1))
    small, _ = closure_step(params, state, features, 0.001, config)
    doubled, _ = closure_step(params, state, features, 0.002, config)
    assert np.isfinite(np.asarray(small)).all()
    np.testing.assert_allclose(np.var(doubled) / np.var(small), 8.0, rtol=0.002)


def test_reproducible_member_streams_and_distinct_ensemble_members():
    config = ClosureConfig(feature_dim=1, levels=1, hidden_dim=2)
    params = init_params(jax.random.PRNGKey(0), config)
    features = jnp.ones((1, 3, 2))
    keys = jax.random.split(jax.random.PRNGKey(100), 4)

    def member_step(key):
        return closure_step(params, initialize_memory((3, 2), config, key), features, 1.0, config)

    first, states = jax.vmap(member_step)(keys)
    repeated, _ = jax.vmap(member_step)(keys)
    np.testing.assert_array_equal(first, repeated)
    assert all(not np.array_equal(first[0], first[i]) for i in range(1, 4))
    assert len(np.unique(np.asarray(states.key), axis=0)) == 4


def test_stochastic_toggle_removes_randomness_and_clears_noise_state():
    config = ClosureConfig(feature_dim=1, levels=1, hidden_dim=2, enable_stochastic=False)
    params = init_params(jax.random.PRNGKey(0), config)
    features = jnp.ones((1, 2, 2))
    first_state = initialize_memory((2, 2), config, jax.random.PRNGKey(10))
    second_state = initialize_memory((2, 2), config, jax.random.PRNGKey(11))
    first_state = first_state._replace(ou=jnp.ones_like(first_state.ou) * 100)
    a, new_state = closure_step(params, first_state, features, 1.0, config)
    b, _ = closure_step(params, second_state, features, 1.0, config)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(new_state.ou, 0)
    np.testing.assert_array_equal(new_state.key, first_state.key)


def test_memory_is_causal_history_dependent_and_can_be_ablated():
    config = ClosureConfig(feature_dim=1, levels=1, hidden_dim=2, enable_spatial=False, enable_stochastic=False)
    params = init_params(jax.random.PRNGKey(0), config)
    # Explicit nonzero readout makes the effect of different histories robust.
    params["encoder_w"] = jnp.ones_like(params["encoder_w"])
    params["memory_w"] = jnp.ones_like(params["memory_w"])
    cold = initialize_memory((1, 1), config, jax.random.PRNGKey(10))
    _, warm = closure_step(params, cold, jnp.ones((1, 1, 1)), 6.0, config)
    current = jnp.zeros((1, 1, 1))
    warm_delta, _ = closure_step(params, warm, current, 1.0, config)
    cold_delta, _ = closure_step(params, cold, current, 1.0, config)
    assert float(warm_delta[0, 0, 0]) > float(cold_delta[0, 0, 0])
    disabled_delta, disabled_state = closure_step(params, warm, current, 1.0, replace(config, enable_memory=False))
    np.testing.assert_array_equal(disabled_delta, 0)
    np.testing.assert_array_equal(disabled_state.memory, 0)


def test_jit_and_training_gradients_are_finite_and_nonzero():
    config = ClosureConfig(feature_dim=2, levels=2, hidden_dim=3)
    params = init_params(jax.random.PRNGKey(0), config)
    features = jnp.ones((2, 3, 2)) * 0.3
    state = initialize_memory((3, 2), config, jax.random.PRNGKey(10))
    step = jax.jit(partial(closure_step, config=config))
    compiled, _ = step(params, state, features, jnp.array(0.1))
    eager, _ = closure_step(params, state, features, 0.1, config)
    np.testing.assert_allclose(compiled, eager, atol=2e-8, rtol=2e-5)
    grads = jax.grad(lambda p: jnp.sum(step(p, state, features, jnp.array(0.1))[0] ** 2))(params)
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree_util.tree_leaves(grads))
    assert all(np.linalg.norm(np.asarray(grads[name])) > 0 for name in ["spatial_w", "memory_w", "diffusion_b"])


def test_integer_features_preserve_fractional_time_and_finite_diffusion():
    config = ClosureConfig(feature_dim=1, levels=1, hidden_dim=2)
    params = init_params(jax.random.PRNGKey(0), config)
    state = initialize_memory((2, 2), config, jax.random.PRNGKey(1))
    integer_delta, _ = closure_step(params, state, jnp.ones((1, 2, 2), dtype=jnp.int32), 0.1, config)
    floating_delta, _ = closure_step(params, state, jnp.ones((1, 2, 2)), 0.1, config)
    assert np.isfinite(np.asarray(integer_delta)).all()
    np.testing.assert_array_equal(integer_delta, floating_delta)


def test_cold_memory_integral_resolves_tiny_timestep_quadratic_scaling():
    config = ClosureConfig(feature_dim=1, levels=1, hidden_dim=1, tau_hours=(120.0,),
                           enable_spatial=False, enable_stochastic=False)
    params = init_params(jax.random.PRNGKey(0), config)
    params["encoder_w"] = jnp.ones_like(params["encoder_w"])
    params["memory_w"] = jnp.ones_like(params["memory_w"])
    state = initialize_memory((1, 1), config, jax.random.PRNGKey(1))
    small, _ = closure_step(params, state, jnp.ones((1, 1, 1)), 1e-5, config)
    doubled, _ = closure_step(params, state, jnp.ones((1, 1, 1)), 2e-5, config)
    assert float(small[0, 0, 0]) > 0.0
    np.testing.assert_allclose(doubled / small, 4.0, rtol=1e-5)


@pytest.mark.parametrize("dt", [0.0, -1.0, np.inf, np.nan])
def test_invalid_timestep_rejected(dt):
    config = ClosureConfig(feature_dim=1, levels=1)
    with pytest.raises(ValueError, match="dt_hours"):
        closure_step(init_params(jax.random.PRNGKey(0), config),
                     initialize_memory((1, 1), config, jax.random.PRNGKey(1)),
                     jnp.ones((1, 1, 1)), dt, config)


@pytest.mark.parametrize("kwargs", [{"tau_hours": (0,)}, {"stochastic_tau_hours": np.nan},
                                    {"max_diffusion_k_sqrt_hour": -1}, {"hidden_dim": 0}])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        ClosureConfig(feature_dim=1, levels=1, **kwargs)
