"""Causal spatial-history transfer through the actual adapter interface."""
from dataclasses import replace
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ngcm_memory.closure import ClosureConfig, init_params
from ngcm_memory.features import SpatialHistory, regrid_lonlat, thermal_with_empty_spatial
from ngcm_memory.neuralgcm_adapter import NeuralGCMMemoryAdapter
from test_neuralgcm_adapter import BaseModel, ModelState, Prognostics, Transform


def _base():
    base = BaseModel()
    horizontal = Transform()
    horizontal.longitudes = np.deg2rad([0., 90., 180., 270.])
    horizontal.latitudes = np.deg2rad([-60., 0., 60.])
    base.model_coords = SimpleNamespace(vertical=SimpleNamespace(layers=2), horizontal=horizontal)
    return base


def _history(tmp_path, name="history", values=None, *, hours=None, metadata=None):
    hours = np.arange(6) if hours is None else np.asarray(hours)
    times = np.datetime64("2001-01-01T00", "h") + hours.astype("timedelta64[h]")
    if values is None:
        values = np.broadcast_to(np.arange(len(times), dtype=float)[:, None, None, None], (len(times), 1, 4, 3)).copy()
    path = tmp_path / f"{name}.npz"
    np.savez(path, times=times, latents=np.asarray(values, dtype=np.float32),
             longitude=[0., 90., 180., 270.], latitude=[-60., 0., 60.],
             metadata_json=np.asarray(json.dumps({"format": "ngcm_memory.spatial_history.v1",
                                                   "periodic_lon": True, **(metadata or {})})))
    return SpatialHistory.from_npz(path, _base())


def test_future_and_origin_observations_cannot_change_context(tmp_path):
    history = _history(tmp_path)
    origin = history.times[3]
    expected = history.context(origin, history_steps=2)
    changed = history.latents.copy()
    changed[3:] += 1e6  # Includes origin itself, not just later verification.
    perturbed = _history(tmp_path, "future_changed", changed).context(origin, history_steps=2)
    np.testing.assert_array_equal(expected["features"], perturbed["features"])
    np.testing.assert_array_equal(expected["dt_hours"], [1., 1.])
    np.testing.assert_array_equal(expected["features"][:, 0, 0, 0], [1., 2.])
    changed[2] += 2
    past_changed = _history(tmp_path, "past_changed", changed).context(origin, history_steps=2)
    assert not np.array_equal(expected["features"], past_changed["features"])
    with pytest.raises(ValueError, match="pre-origin"):
        history.context(history.times[0], history_steps=1)


def test_stale_history_cannot_be_held_until_a_distant_origin(tmp_path):
    history = _history(tmp_path, hours=[0, 24])
    assert history.max_history_gap_hours == 24.
    assert history.metadata["history_gap_policy"] == "inferred_median_interval"
    with pytest.raises(ValueError, match="stale last observation.*8736 h"):
        history.context("2002-01-01", history_steps=2)
    # Ending exactly one cadence after the last sample remains valid.
    np.testing.assert_array_equal(history.context("2001-01-03", 2)["dt_hours"], [24., 24.])


def test_internal_missing_interval_is_rejected_but_complete_later_windows_work(tmp_path):
    history = _history(tmp_path, hours=[0, 1, 2, 10, 11, 12])
    assert history.max_history_gap_hours == 1.
    with pytest.raises(ValueError, match="internal gap.*8 h"):
        history.context("2001-01-01T11", 2)
    np.testing.assert_array_equal(history.context("2001-01-01T13", 2)["dt_hours"], [1., 1.])


def test_explicit_maximum_preserves_irregular_intervals_without_resampling(tmp_path):
    history = _history(tmp_path, hours=[0, 1, 2, 10], metadata={"max_history_gap_hours": 8.})
    np.testing.assert_array_equal(history.context("2001-01-01T11", 3)["dt_hours"], [1., 8., 1.])
    with pytest.raises(ValueError, match="stale last observation.*9 h"):
        history.context("2001-01-01T19", 3)


def test_single_sample_requires_an_explicit_gap_contract(tmp_path):
    with pytest.raises(ValueError, match="Cannot infer.*one sample"):
        _history(tmp_path, hours=[0])
    history = _history(tmp_path, "explicit", hours=[0], metadata={"max_history_gap_hours": 2.})
    np.testing.assert_array_equal(history.context("2001-01-01T02", 1)["dt_hours"], [2.])
    with pytest.raises(ValueError, match="stale last observation"):
        history.context("2001-01-01T03", 1)


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), True, "24"])
def test_invalid_declared_gap_contract_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="max_history_gap_hours must be finite and positive"):
        _history(tmp_path, metadata={"max_history_gap_hours": value})


def test_declared_cadence_takes_precedence_over_legacy_inference(tmp_path):
    history = _history(tmp_path, hours=[0, 24], metadata={"cadence_hours": 6.})
    assert history.max_history_gap_hours == 6.
    with pytest.raises(ValueError, match="internal gap"):
        history.context("2001-01-02T06", 2)


def test_origin_filter_reuses_context_validation_without_regridding(tmp_path, monkeypatch):
    history = _history(tmp_path, hours=[0, 1, 2, 10, 11, 12])
    times = np.array(["2001-01-01T03", "2001-01-01T11", "2001-01-01T13", "2001-01-01T14"],
                     dtype="datetime64[ns]")
    def no_regrid(*args):
        raise AssertionError("Eligibility checks must not materialize accelerator feature grids")
    monkeypatch.setattr("ngcm_memory.features.regrid_lonlat", no_regrid)
    valid, report = history.filter_origins(times, np.arange(len(times)), 2)
    np.testing.assert_array_equal(valid, [0, 2])
    assert report["rejected_windows"] == 2
    assert "internal gap" in report["first_error"]


def test_periodic_regrid_preserves_axes_seam_and_polar_extension():
    # Descending source axes exercise ordering; periodic seam at 315 degrees
    # must interpolate 270 -> 360/0 rather than use a nonperiodic edge value.
    lon = np.array([270., 180., 90., 0.])
    lat = np.array([60., 0., -60.])
    field = np.sin(np.deg2rad(lon))[:, None] + lat[None, :] / 60
    source = np.stack([field, field + 10.])[None]
    actual = regrid_lonlat(source, lon, lat, [315., 45.], [-90., -30., 30., 90.])
    expected = np.array([-.5, .5])[:, None] + np.array([-1., -.5, .5, 1.])[None, :]
    assert actual.shape == (1, 2, 2, 4)
    np.testing.assert_allclose(actual[0, 0], expected, atol=1e-6)
    np.testing.assert_allclose(actual[0, 1], expected + 10, atol=1e-6)


def test_past_spatial_features_change_differentiable_corrected_forecast(tmp_path):
    history = _history(tmp_path)
    base = _base()
    config = ClosureConfig(feature_dim=3, levels=2, hidden_dim=2,
                           enable_spatial=False, enable_stochastic=False)
    adapter = NeuralGCMMemoryAdapter(base, config,
                                    feature_fn=thermal_with_empty_spatial(base, 1))
    model_state = ModelState(
        Prognostics(jnp.ones((2, 4, 3)), jnp.asarray(0.), jnp.ones((2, 4, 3))),
        {"existing_memory": jnp.asarray(9.)}, jnp.asarray(7.), {"diagnostic": jnp.asarray(2.)})
    state = adapter.initialize(model_state, jax.random.key(2))
    # Nonzero OU verifies history initialization changes deterministic memory
    # only, even when stochastic rollout has been disabled for this experiment.
    state = state._replace(closure_state=state.closure_state._replace(ou=jnp.ones((2, 4, 3))))
    params = init_params(jax.random.key(0), config)
    params["encoder_w"] = jnp.asarray([[0., 0., 1.], [0., 0., 1.]])
    params["memory_w"] = jnp.ones_like(params["memory_w"])
    context = history.context(history.times[3], history_steps=2)
    seed = history.initializer(config)
    warm = seed(params, state, context)
    assert warm.model_state is state.model_state
    np.testing.assert_array_equal(warm.closure_state.ou, state.closure_state.ou)
    np.testing.assert_array_equal(jax.random.key_data(warm.closure_state.key),
                                  jax.random.key_data(state.closure_state.key))
    assert np.linalg.norm(warm.closure_state.memory) > 0
    # The online source channels are zero: forecast sensitivity can only arise
    # from the explicitly carried, observed historical memory.
    online = adapter.features(warm.model_state, {})
    np.testing.assert_array_equal(online[-1], 0.)

    def forecast(parameters, selected_context):
        carried = seed(parameters, state, selected_context)
        for _ in range(2):
            carried = adapter.advance(parameters, carried, {})
        return jnp.mean(adapter.decode(carried, {})["temperature"])

    value, gradient = jax.jit(jax.value_and_grad(forecast))(params, context)
    zero_context = {**context, "features": jnp.zeros_like(context["features"])}
    assert float(value) > float(forecast(params, zero_context))
    assert np.isfinite(float(value))
    assert np.linalg.norm(gradient["encoder_w"][:, -1]) > 0
    assert np.linalg.norm(gradient["memory_w"]) > 0
    no_memory_config = replace(config, enable_memory=False)
    ablated = history.initializer(no_memory_config)(params, state, context)
    np.testing.assert_array_equal(ablated.closure_state.memory, 0.)


def test_recurrent_context_uses_origin_but_never_future_and_requires_exact_times(tmp_path):
    history = _history(tmp_path)
    origin = history.times[3]
    context = history.transition_context(origin, 2)
    np.testing.assert_array_equal(context["features"][:, 0, 0, 0], [1., 2.])
    np.testing.assert_array_equal(context["initial_latent"], 3.)
    changed = history.latents.copy()
    changed[4:] += 1e6
    other = _history(tmp_path, "changed_after_origin", changed)
    for name, value in other.transition_context(origin, 2).items():
        np.testing.assert_array_equal(value, context[name])
    np.testing.assert_array_equal(history.targets_at(history.times[[4, 5]])[:, 0, 0, 0], [4., 5.])
    with pytest.raises(ValueError, match="exact timestamp"):
        history.transition_context("2001-01-01T03:30", 2)
    with pytest.raises(ValueError, match="exact timestamp"):
        history.targets_at(np.asarray(["2001-01-01T06"], dtype="datetime64[ns]"))


def test_spatial_initialization_preserves_thermal_memory_and_seeds_all_past_latents(tmp_path):
    history = _history(tmp_path)
    config = ClosureConfig(feature_dim=3, levels=2, hidden_dim=2, latent_dim=1,
                           latent_memory_tau_hours=(1., 2.), enable_stochastic=False)
    adapter = NeuralGCMMemoryAdapter(_base(), config)
    model_state = ModelState(
        Prognostics(jnp.ones((2, 4, 3)), jnp.asarray(0.), jnp.ones((2, 4, 3))),
        {"existing_memory": jnp.asarray(9.)}, jnp.asarray(7.), {"diagnostic": jnp.asarray(2.)})
    state = adapter.initialize(model_state, jax.random.key(2))
    state = state._replace(closure_state=state.closure_state._replace(
        memory=jnp.ones_like(state.closure_state.memory), ou=jnp.ones_like(state.closure_state.ou)))
    context = history.transition_context(history.times[3], 2)
    warm = jax.jit(history.transition_initializer(config))({}, state, context)
    for actual, expected in zip(jax.tree.leaves(warm.closure_state), jax.tree.leaves(state.closure_state)):
        # Typed PRNG keys require key_data rather than numpy conversion.
        if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
            actual, expected = jax.random.key_data(actual), jax.random.key_data(expected)
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(warm.latent_state.latent, 3.)
    a = np.exp(-1. / np.array([1., 2.]))
    expected_memory = a * (1 - a) * 1. + (1 - a) * 2.
    np.testing.assert_allclose(warm.latent_state.memory[:, 0, 0, 0], expected_memory, rtol=1e-6)
    np.testing.assert_array_equal(jax.random.key_data(warm.latent_state.key), jax.random.key_data(state.latent_state.key))
    ablated = history.transition_initializer(replace(config, enable_memory=False))({}, state, context)
    np.testing.assert_array_equal(ablated.latent_state.memory, 0.)
    np.testing.assert_array_equal(ablated.closure_state.memory, state.closure_state.memory)
