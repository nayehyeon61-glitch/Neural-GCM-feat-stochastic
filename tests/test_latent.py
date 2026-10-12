"""Recurrent latent dynamics: conditional law, accumulated history, feedback."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ngcm_memory.closure import ClosureConfig, init_params
from ngcm_memory.latent import initialize_latent, latent_step, transition_moments
from ngcm_memory.neuralgcm_adapter import AugmentedState, NeuralGCMMemoryAdapter
from test_neuralgcm_adapter import BaseModel, ModelState, Prognostics


def _config(**kwargs):
    return ClosureConfig(feature_dim=3, levels=2, latent_dim=1,
                         latent_hidden_dim=3, hidden_dim=2, **kwargs)


def _state(adapter):
    model = ModelState(
        Prognostics(jnp.ones((2, 4, 3)), jnp.asarray(0.), jnp.ones((2, 4, 3))),
        {"existing_memory": jnp.asarray(9.)}, jnp.asarray(7.), {"diagnostic": jnp.asarray(2.)})
    return adapter.initialize(model, jax.random.key(2))


def _constant_transition(config):
    params = init_params(jax.random.key(0), config)["latent_transition"]
    params["target_w"] = jnp.zeros_like(params["target_w"])
    params["target_b"] = jnp.full_like(params["target_b"], np.arctanh(0.4 / config.latent_target_bound))
    params["diffusion_w"] = jnp.zeros_like(params["diffusion_w"])
    params["diffusion_b"] = jnp.zeros_like(params["diffusion_b"])
    return params


@pytest.mark.parametrize("dt", [1e-7, 0.5, 48.])
def test_exact_conditional_gaussian_endpoint_including_tiny_steps(dt):
    config = _config(latent_tau_hours=2., latent_max_diffusion_per_sqrt_hour=0.8)
    params = _constant_transition(config)
    state = initialize_latent((40_000, 1), config, jax.random.key(7))
    # Zero initial state permits tiny-step Monte Carlo without subtracting a
    # float32 O(1) mean from O(sqrt(dt)) innovations.
    thermal = jnp.ones((2, 40_000, 1))
    mean, variance = transition_moments(params, state, thermal, dt, config)
    sigma = 0.8 * (1e-6 + (1 - 1e-6) * 0.5)
    expected_mean = -np.expm1(-dt / 2.) * 0.4
    expected_variance = sigma**2 * -np.expm1(-dt)
    np.testing.assert_allclose(mean, expected_mean, rtol=3e-6)
    np.testing.assert_allclose(variance, expected_variance, rtol=3e-6)
    samples = np.asarray(latent_step(params, state, thermal, dt, config).latent)
    np.testing.assert_allclose(np.var(samples), expected_variance, rtol=0.035)
    assert abs(samples.mean() - expected_mean) < 0.02 * np.sqrt(expected_variance)


def test_memory_accumulates_each_latent_and_retains_history_in_conditional_law():
    config = _config(enable_stochastic=False, latent_memory_tau_hours=(2., 10.))
    params = init_params(jax.random.key(0), config)["latent_transition"]
    # Explicit dependency on the first memory, with all current inputs fixed.
    params["encoder_w"] = jnp.zeros_like(params["encoder_w"]).at[0, 3].set(1.)
    params["target_w"] = jnp.zeros_like(params["target_w"]).at[0, 0].set(1.)
    thermal = jnp.zeros((2, 1, 1))
    state = initialize_latent((1, 1), config, jax.random.key(1))
    history = [2., -1., 3.]
    expected = np.zeros((2, 1, 1, 1))
    decay = np.exp(-1. / np.array([2., 10.]))[:, None, None, None]
    for z in history:
        state = state._replace(latent=jnp.full_like(state.latent, z))
        state = latent_step(params, state, thermal, 1., config)
        expected = decay * expected + (1 - decay) * z
    np.testing.assert_allclose(state.memory, expected, rtol=2e-6)
    current = state._replace(latent=jnp.zeros_like(state.latent))
    no_history = current._replace(memory=jnp.zeros_like(current.memory))
    mean, _ = transition_moments(params, current, thermal, 1., config)
    reference, _ = transition_moments(params, no_history, thermal, 1., config)
    assert float(mean[0, 0, 0]) > float(reference[0, 0, 0])
    ablated = latent_step(params, current, thermal, 1., replace(config, enable_memory=False))
    np.testing.assert_array_equal(ablated.memory, 0.)
    np.testing.assert_allclose(ablated.latent, reference)


def test_new_innovations_are_carried_through_multiple_latent_steps():
    config = _config(latent_tau_hours=2., latent_max_diffusion_per_sqrt_hour=.8)
    params = _constant_transition(config)
    state = initialize_latent((40_000, 1), config, jax.random.key(44))
    thermal = jnp.zeros((2, 40_000, 1))
    first = latent_step(params, state, thermal, 0.5, config)
    second = latent_step(params, first, thermal, 0.5, config)
    # For constant coefficients two successive exact Gaussian steps have the
    # one-hour endpoint law. Resetting z each step would fail this variance.
    _, one_hour_variance = transition_moments(params, state, thermal, 1., config)
    np.testing.assert_allclose(np.var(second.latent), one_hour_variance[0, 0, 0], rtol=.035)
    empirical_covariance = np.cov(np.asarray(first.latent).ravel(), np.asarray(second.latent).ravel(), ddof=0)[0, 1]
    np.testing.assert_allclose(empirical_covariance, np.exp(-.25) * np.var(first.latent), rtol=.035)
    assert np.linalg.norm(second.memory) > 0


def test_sampling_reproducibility_and_separate_thermal_noise_stream():
    config = _config()
    adapter = NeuralGCMMemoryAdapter(BaseModel(), config)
    state = _state(adapter)
    params = init_params(jax.random.key(0), config)
    assert not np.array_equal(jax.random.key_data(state.closure_state.key), jax.random.key_data(state.latent_state.key))
    actual = adapter.advance(params, state, {})
    repeated = adapter.advance(params, state, {})
    np.testing.assert_array_equal(actual.latent_state.latent, repeated.latent_state.latent)
    # Changing only thermal OU randomness cannot change this step's z draw.
    changed_ou = state._replace(closure_state=state.closure_state._replace(key=jax.random.key(100)))
    other = adapter.advance(params, changed_ou, {})
    np.testing.assert_array_equal(actual.latent_state.latent, other.latent_state.latent)
    assert not np.array_equal(actual.closure_state.ou, other.closure_state.ou)
    changed_latent = state._replace(latent_state=state.latent_state._replace(key=jax.random.key(101)))
    other_latent = adapter.advance(params, changed_latent, {})
    assert not np.array_equal(actual.latent_state.latent, other_latent.latent_state.latent)
    np.testing.assert_array_equal(actual.closure_state.ou, other_latent.closure_state.ou)


def test_deterministic_mode_and_zero_diffusion_have_finite_gradients():
    for config in (_config(enable_stochastic=False), _config(latent_max_diffusion_per_sqrt_hour=0.)):
        params = init_params(jax.random.key(0), config)["latent_transition"]
        state = initialize_latent((2, 2), config, jax.random.key(1))
        thermal = jnp.ones((2, 2, 2))
        first = latent_step(params, state, thermal, 1., config)
        second = latent_step(params, state._replace(key=jax.random.key(2)), thermal, 1., config)
        np.testing.assert_array_equal(first.latent, second.latent)
        if not config.enable_stochastic:
            np.testing.assert_array_equal(jax.random.key_data(first.key), jax.random.key_data(state.key))
        gradients = jax.grad(lambda p: jnp.mean(latent_step(p, state, thermal, 1., config).latent**2))(params)
        assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(gradients))


def test_multistep_corrected_forecast_depends_on_carried_latent_and_transition_parameters():
    config = _config(enable_stochastic=False)
    adapter = NeuralGCMMemoryAdapter(BaseModel(), config)
    state = _state(adapter)
    params = init_params(jax.random.key(0), config)
    # Route closure corrections through z explicitly for a robust dependency.
    params["encoder_w"] = jnp.array([[0., 0., 1.], [0., 0., 1.]])
    params["spatial_w"] = jnp.ones_like(params["spatial_w"])
    def forecast(p, carry):
        for _ in range(3):
            carry = adapter.advance(p, carry, {})
        return jnp.mean(adapter.decode(carry, {})["temperature"])
    value, gradient = jax.jit(jax.value_and_grad(forecast))(params, state)
    assert np.isfinite(float(value))
    assert np.linalg.norm(gradient["latent_transition"]["target_w"]) > 0
    changed = state._replace(latent_state=state.latent_state._replace(latent=jnp.ones_like(state.latent_state.latent)))
    assert float(forecast(params, changed)) > float(value)
    # The online feature channel is the carried forecast latent, never zeroed.
    np.testing.assert_array_equal(adapter.features(changed.model_state, {}, changed.latent_state)[-1], 1.)


def test_legacy_parameter_tree_and_two_argument_state_remain_compatible():
    latent_config = _config()
    legacy_config = replace(latent_config, latent_dim=0)
    legacy = init_params(jax.random.key(0), legacy_config)
    recurrent = init_params(jax.random.key(0), latent_config)
    assert set(recurrent) == set(legacy) | {"latent_transition"}
    for name, array in legacy.items():
        np.testing.assert_array_equal(array, recurrent[name])
    assert AugmentedState("base", "closure").latent_state is None


def test_disabled_adapter_preserves_both_recurrent_states_exactly():
    adapter = NeuralGCMMemoryAdapter(BaseModel(), _config(), enabled=False)
    state = _state(adapter)
    result = adapter.advance(None, state, {})
    assert result.latent_state is state.latent_state
    assert result.closure_state is state.closure_state
    expected = adapter.base_model.advance(state.model_state, {})
    for actual, wanted in zip(jax.tree.leaves(result.model_state), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(actual, wanted)


def test_observed_warmup_keeps_latent_state_unchanged_and_zeroes_source_features():
    config = _config(enable_stochastic=False)
    adapter = NeuralGCMMemoryAdapter(BaseModel(), config)
    state = _state(adapter)
    params = init_params(jax.random.key(0), config)
    changed = state._replace(latent_state=state.latent_state._replace(
        latent=jnp.ones_like(state.latent_state.latent) * 100.,
        memory=jnp.ones_like(state.latent_state.memory)))
    original_result = adapter.advance_warmup(params, state, {})
    changed_result = adapter.advance_warmup(params, changed, {})
    assert changed_result.latent_state is changed.latent_state
    for actual, expected in zip(jax.tree.leaves(original_result.model_state), jax.tree.leaves(changed_result.model_state)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(original_result.closure_state.memory, changed_result.closure_state.memory)
    assert np.linalg.norm(changed_result.closure_state.memory) > 0


def test_recurrent_features_reject_external_callbacks_and_missing_state():
    with pytest.raises(ValueError, match="feature_fn must be None"):
        NeuralGCMMemoryAdapter(BaseModel(), _config(), feature_fn=lambda s, f: s)
    adapter = NeuralGCMMemoryAdapter(BaseModel(), _config())
    with pytest.raises(ValueError, match="requires carried latent_state"):
        adapter.features(_state(adapter).model_state, {})


@pytest.mark.parametrize("kwargs", [{"latent_dim": -1}, {"latent_dim": True},
                                   {"latent_hidden_dim": 0}, {"latent_tau_hours": 0.},
                                   {"latent_memory_tau_hours": (1., np.inf)},
                                   {"latent_max_diffusion_per_sqrt_hour": -1.},
                                   {"latent_target_bound": 0.}])
def test_invalid_latent_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        ClosureConfig(feature_dim=3, levels=2, **kwargs)
