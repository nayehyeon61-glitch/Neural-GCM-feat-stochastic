from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ngcm_memory.closure import ClosureConfig, init_params
from ngcm_memory.training import (
    evaluation_origins, forecast_loss, independent_test_start_index, load_closure_checkpoint, make_train_step,
    rollout_ensemble, save_closure_checkpoint, temporal_window_split,
    validate_time_axis,
)


class ToyState(NamedTuple):
    model_state: object
    closure_state: object


class ToyBase:
    def encode(self, inputs, forcings, rng_key):
        del forcings, rng_key
        return inputs["temperature"]


class ToyAdapter:
    """Minimal scalar recurrence to test causality/gradients, not model skill."""
    base_model = ToyBase()

    def encode(self, inputs, forcings, key):
        return ToyState(self.base_model.encode(inputs, forcings, key), jnp.asarray(0.))

    def advance(self, params, state, forcings):
        return ToyState(
            state.model_state * params["gain"] + forcings["boundary"],
            state.closure_state + 1,
        )

    def decode(self, state, forcings):
        del forcings
        return {"temperature": state.model_state}


def window(values):
    x = {"temperature": jnp.asarray(values, dtype=jnp.float32)[:, None, None, None],
         "sim_time": jnp.arange(len(values), dtype=jnp.float32)}
    forcing = {"boundary": jnp.zeros((len(values), 1, 1, 1)), "sim_time": x["sim_time"]}
    return x, forcing


def test_temporal_split_never_crosses_boundary():
    train, validation, cut = temporal_window_split(30, 2, 3, .6)
    assert max(train) + 3 < cut
    assert min(validation) - 2 >= cut
    with pytest.raises(ValueError, match="complete"):
        temporal_window_split(10, 3, 3, .8)
    np.testing.assert_array_equal(evaluation_origins(12, 1, 2, 7), [8, 9])


def test_time_requires_regular_integer_internal_steps():
    times = np.datetime64("2020-01-01T00") + np.arange(6) * np.timedelta64(6, "h")
    assert validate_time_axis(times, np.timedelta64(30, "m")) == (6., 12)
    with pytest.raises(ValueError, match="integer multiple"):
        validate_time_axis(times, np.timedelta64(5, "h"))
    with pytest.raises(ValueError, match="regularly"):
        validate_time_axis(times[[0, 1, 3]], np.timedelta64(1, "h"))


def test_test_period_excludes_model_selection_validation_dates():
    times = np.datetime64("2020-01-01") + np.arange(12) * np.timedelta64(1, "D")
    metadata = {"training_last_observation": "2020-01-03", "validation_last_observation": "2020-01-07"}
    assert independent_test_start_index(times, metadata) == 7
    assert independent_test_start_index(times, metadata, "2020-01-10") == 9
    with pytest.raises(ValueError, match="independent test"):
        independent_test_start_index(times, {"training_last_observation": "2020-01-03"})


def test_forecast_targets_never_enter_recurrent_state():
    x, forcing = window([7., 2., 99., -99.])
    kwargs = dict(warmup_steps=1, horizon_steps=2, internal_steps=1)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    first = rollout_ensemble({"gain": jnp.asarray(2.)}, ToyAdapter(), x, forcing, keys, **kwargs)
    altered = {**x, "temperature": x["temperature"].at[2:].set(1e6)}
    second = rollout_ensemble({"gain": jnp.asarray(2.)}, ToyAdapter(), altered, forcing, keys, **kwargs)
    np.testing.assert_array_equal(first["temperature"], second["temperature"])
    np.testing.assert_allclose(first["temperature"][0, :, 0, 0, 0], [4., 8.])


def test_default_boundary_forcing_is_persistent():
    x, forcing = window([1., 0., 0.])
    forcing["boundary"] = jnp.asarray([1., 100., 1000.])[:, None, None, None]
    kwargs = dict(warmup_steps=0, horizon_steps=2, internal_steps=1)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    persistent = rollout_ensemble({"gain": 1.}, ToyAdapter(), x, forcing, keys, **kwargs)
    observed = rollout_ensemble({"gain": 1.}, ToyAdapter(), x, forcing, keys, forcing_mode="observed", **kwargs)
    np.testing.assert_allclose(persistent["temperature"][0, :, 0, 0, 0], [2., 3.])
    np.testing.assert_allclose(observed["temperature"][0, :, 0, 0, 0], [2., 102.])


def test_gradient_follows_corrected_multistep_state():
    x, forcing = window([1., 0., 0., 0.])
    def terminal(gain):
        prediction = rollout_ensemble(
            {"gain": gain}, ToyAdapter(), x, forcing,
            jax.random.split(jax.random.PRNGKey(0), 2),
            warmup_steps=0, horizon_steps=3, internal_steps=1,
        )
        return prediction["temperature"][0, -1, 0, 0, 0]
    # x3 = gain^3; a residual-only fit would not have this derivative.
    assert float(jax.grad(terminal)(2.)) == pytest.approx(12.)


def test_one_optimizer_step_changes_rollout_parameter():
    x, forcing = window([1., 1., 1.])
    optimizer = optax.sgd(.01)
    params = {"gain": jnp.asarray(2.)}
    step = make_train_step(
        ToyAdapter(), optimizer, warmup_steps=0, horizon_steps=2,
        internal_steps=1, variables=("temperature",), scales={"temperature": 1.},
        area_weights=jnp.ones((1, 1)),
    )
    updated, _, loss, norm = step(
        params, optimizer.init(params), x, forcing,
        jax.random.split(jax.random.PRNGKey(0), 2),
    )
    assert np.isfinite(loss) and float(norm) > 0
    assert float(updated["gain"]) < float(params["gain"])


def test_auxiliary_mean_score_removes_member_sampling_variance():
    members = jnp.asarray([0., 2.])[:, None, None, None, None]
    target = jnp.ones((1, 1, 1, 1))
    value = forecast_loss({"temperature": members}, {"temperature": target},
                          jnp.ones((1, 1)), {"temperature": 1.}, mean_mse_weight=.1)
    # Fair CRPS=0, sample-mean squared error=0, sample variance/n=1.
    # Unbiased estimators can be negative on a particular finite sample.
    assert float(value) == pytest.approx(-.1)


def test_memory_initializer_receives_past_warmup_and_reencoded_origin():
    x, forcing = window([7., 3., 2., 0.])
    def seed(params, state, context):
        del params
        # Warmup has 2 intervals * 3 internal steps; origin physical state is 2.
        return state._replace(model_state=state.model_state + state.closure_state + context)
    prediction = rollout_ensemble(
        {"gain": 1.}, ToyAdapter(), x, forcing,
        jax.random.split(jax.random.PRNGKey(0), 2),
        warmup_steps=2, horizon_steps=1, internal_steps=3,
        memory_initializer=seed, memory_context=jnp.asarray(10.),
    )
    np.testing.assert_allclose(prediction["temperature"], 18.)


def test_safe_checkpoint_roundtrip_and_shape_rejection(tmp_path):
    config = ClosureConfig(feature_dim=2, levels=2)
    params = init_params(jax.random.PRNGKey(4), config)
    path = tmp_path / "closure.npz"
    save_closure_checkpoint(path, params, config, {"source": "synthetic_test"})
    restored, restored_config, metadata = load_closure_checkpoint(path)
    assert restored_config == config and metadata["source"] == "synthetic_test"
    for expected, actual in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(expected, actual)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["param_00000"] = np.zeros((999,))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        load_closure_checkpoint(path)
