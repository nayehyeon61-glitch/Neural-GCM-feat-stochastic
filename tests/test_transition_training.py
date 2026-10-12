"""Causal rollout, multi-step gradients and source-label training contracts."""
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ngcm_memory.closure import ClosureConfig, init_params
from ngcm_memory.features import SpatialHistory
from ngcm_memory.training import (
    filter_spatial_origins, joint_forecast_loss, load_closure_checkpoint,
    make_train_step, rollout_ensemble, save_closure_checkpoint,
)


class ToyLatent(NamedTuple):
    latent: object
    memory: object
    key: object


class ToyState(NamedTuple):
    model_state: object
    closure_state: object
    latent_state: object


class ToyBase:
    def encode(self, inputs, forcings, rng_key):
        del forcings, rng_key
        return inputs["temperature"]


class ToyRecurrentAdapter:
    base_model = ToyBase()

    def encode(self, inputs, forcings, key):
        x = self.base_model.encode(inputs, forcings, key)
        return ToyState(x, jnp.asarray(0.), ToyLatent(jnp.ones_like(x), jnp.zeros_like(x), key))

    def advance(self, params, state, forcings):
        del forcings
        previous = state.latent_state
        z = params["gain"] * previous.latent
        latent = previous._replace(latent=z, memory=.5 * previous.memory + .5 * previous.latent)
        return ToyState(state.model_state + z, state.closure_state + 1, latent)

    def decode(self, state, forcings):
        del forcings
        return {"temperature": state.model_state}


def window(values):
    x = {"temperature": jnp.asarray(values, dtype=jnp.float32)[:, None, None, None],
         "sim_time": jnp.arange(len(values), dtype=jnp.float32)}
    return x, {"sim_time": x["sim_time"]}


def seed_latent(params, state, context):
    del params
    # The generic rollout itself must keep future labels outside initialization.
    assert "latent_targets" not in context
    return state._replace(latent_state=state.latent_state._replace(latent=context["initial_latent"]))


def test_future_latent_and_physical_labels_change_loss_but_never_rollout():
    x, forcings = window([0., 10., 20., 30.])
    context = {"initial_latent": jnp.ones((1, 1, 1)),
               "latent_targets": jnp.zeros((3, 1, 1, 1))}
    kwargs = dict(warmup_steps=0, horizon_steps=3, internal_steps=1,
                  memory_initializer=seed_latent, return_latents=True)
    keys = jax.random.split(jax.random.PRNGKey(4), 2)
    params = {"gain": jnp.asarray(2.)}
    fields, latents = rollout_ensemble(params, ToyRecurrentAdapter(), x, forcings, keys,
                                      memory_context=context, **kwargs)
    changed_x = {**x, "temperature": x["temperature"].at[1:].set(999.)}
    changed_context = {**context, "latent_targets": jnp.full_like(context["latent_targets"], -999.)}
    changed_fields, changed_latents = rollout_ensemble(
        params, ToyRecurrentAdapter(), changed_x, forcings, keys,
        memory_context=changed_context, **kwargs)
    np.testing.assert_array_equal(fields["temperature"], changed_fields["temperature"])
    np.testing.assert_array_equal(latents, changed_latents)
    np.testing.assert_allclose(latents[0, :, 0, 0, 0], [2., 4., 8.])
    np.testing.assert_allclose(fields["temperature"][0, :, 0, 0, 0], [2., 6., 14.])
    def losses(data, labels):
        return joint_forecast_loss(
            fields, {"temperature": data["temperature"][1:]}, jnp.ones((1, 1)), {"temperature": 1.},
            latent_predictions=latents, latent_targets=labels, latent_loss_weight=.3,
            latent_area_weights=jnp.ones((1, 1)),
        )
    first, physical, latent = losses(x, context["latent_targets"])
    second, changed_physical, changed_latent = losses(changed_x, changed_context["latent_targets"])
    assert float(second) != float(first)
    assert float(changed_physical) != float(physical)
    assert float(changed_latent) != float(latent)
    assert float(first) == pytest.approx(float(physical + .3 * latent))


def test_latent_chain_drives_terminal_physical_gradient():
    x, forcing = window([0., 0., 0., 0.])
    def terminal(gain):
        prediction = rollout_ensemble(
            {"gain": gain}, ToyRecurrentAdapter(), x, forcing,
            jax.random.split(jax.random.PRNGKey(0), 2),
            warmup_steps=0, horizon_steps=3, internal_steps=1,
        )
        return prediction["temperature"][0, -1, 0, 0, 0]
    # x3 = gain + gain^2 + gain^3. A one-off z0 read cannot yield this path.
    assert float(jax.grad(terminal)(2.)) == pytest.approx(17.)


def test_observed_warmup_uses_thermal_path_then_starts_latent_chain():
    class WarmupAdapter(ToyRecurrentAdapter):
        def advance_warmup(self, params, state, forcings):
            del params, forcings
            return state._replace(closure_state=state.closure_state + 5.)

    def initialize(params, state, context):
        del params, context
        # Expose accumulated warmup in the output while preserving the latent
        # that advance_warmup must leave alone. Native state was reencoded at t0.
        return state._replace(model_state=state.model_state + state.closure_state)

    x, forcing = window([8., 3., 2., 0.])
    fields, z = rollout_ensemble(
        {"gain": 2.}, WarmupAdapter(), x, forcing,
        jax.random.split(jax.random.PRNGKey(0), 2),
        warmup_steps=2, horizon_steps=1, internal_steps=3,
        memory_initializer=initialize, return_latents=True,
    )
    # 6 warmup internal steps -> memory=30, origin x=2, then z=2,4,8 adds 14.
    np.testing.assert_allclose(fields["temperature"], 46.)
    np.testing.assert_allclose(z, 8.)


def test_latent_supervision_updates_params_and_requires_labels():
    x, forcing = window([0., 2., 6.])
    params = {"gain": jnp.asarray(2.)}
    optimizer = optax.sgd(.01)
    kwargs = dict(warmup_steps=0, horizon_steps=2, internal_steps=1,
                  variables=("temperature",), scales={"temperature": 1.},
                  area_weights=jnp.ones((1, 1)), memory_initializer=seed_latent,
                  latent_loss_weight=1., latent_area_weights=jnp.ones((1, 1)))
    train_step = make_train_step(ToyRecurrentAdapter(), optimizer, **kwargs)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    context = {"initial_latent": jnp.ones((1, 1, 1)), "latent_targets": jnp.zeros((2, 1, 1, 1))}
    updated, _, loss, norm = train_step(params, optimizer.init(params), x, forcing, keys, context)
    assert np.isfinite(loss) and float(norm) > 0
    assert float(updated["gain"]) < 2.
    with pytest.raises(ValueError, match="latent_targets"):
        train_step(params, optimizer.init(params), x, forcing, keys, {"initial_latent": context["initial_latent"]})
    with pytest.raises(ValueError, match="nonnegative"):
        make_train_step(ToyRecurrentAdapter(), optimizer, **{**kwargs, "latent_loss_weight": -1.})


def test_latent_score_uses_its_own_grid_and_rejects_broadcast_labels():
    fields = {"temperature": jnp.ones((2, 2, 1, 1, 1))}
    targets = {"temperature": jnp.zeros((2, 1, 1, 1))}
    z = jnp.ones((2, 2, 3, 4, 2))
    labels = jnp.zeros((2, 3, 4, 2))
    total, physical, latent = joint_forecast_loss(
        fields, targets, jnp.ones((1, 1)), {"temperature": 1.},
        latent_predictions=z, latent_targets=labels, latent_loss_weight=.2,
        latent_area_weights=jnp.ones((4, 2)),
    )
    assert float(total) == pytest.approx(float(physical + .2 * latent))
    with pytest.raises(ValueError, match="dimensions exactly"):
        joint_forecast_loss(fields, targets, jnp.ones((1, 1)), {"temperature": 1.},
                            latent_predictions=z, latent_targets=labels[:1], latent_loss_weight=.2,
                            latent_area_weights=jnp.ones((4, 2)))


def test_source_labels_required_for_training_but_not_forecasting():
    times = np.datetime64("2020-01-01") + np.arange(8) * np.timedelta64(1, "D")
    coords = np.asarray([0., 180.])
    history = SpatialHistory(times[:5], np.zeros((5, 1, 2, 2)), coords, np.asarray([-45., 45.]),
                             coords, np.asarray([-45., 45.]), {"cadence_hours": 24.}, "synthetic")
    forecast, details = filter_spatial_origins(history, times, [3, 4, 5], 2, recurrent=True)
    np.testing.assert_array_equal(forecast, [3, 4])
    assert details["rejected_windows"] == 1
    train, details = filter_spatial_origins(history, times, [2, 3, 4], 2, recurrent=True,
                                           horizon_steps=2, require_future_targets=True)
    np.testing.assert_array_equal(train, [2])
    assert details["rejected_windows"] == 2
    assert "exact spatial latents" in details["first_error"]


def test_recurrent_checkpoint_and_pre_transition_schema_restore(tmp_path):
    recurrent = ClosureConfig(feature_dim=3, levels=2, latent_dim=1,
                              latent_memory_tau_hours=(12., 240.))
    params = init_params(jax.random.PRNGKey(0), recurrent)
    path = tmp_path / "new.npz"
    save_closure_checkpoint(path, params, recurrent, {"spatial_mode": "recurrent"})
    restored, config, metadata = load_closure_checkpoint(path)
    assert config == recurrent and metadata["spatial_mode"] == "recurrent"
    for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(a, b)

    legacy = ClosureConfig(feature_dim=2, levels=2)
    legacy_params = init_params(jax.random.PRNGKey(0), legacy)
    save_closure_checkpoint(path, legacy_params, legacy, {"legacy": True})
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    # Simulate a genuine v1 checkpoint, whose config knew no latent fields.
    old_config = json.loads(arrays["config"].item())
    old_config = {name: value for name, value in old_config.items() if not name.startswith("latent_")}
    arrays["config"] = np.asarray(json.dumps(old_config))
    np.savez(path, **arrays)
    restored, config, metadata = load_closure_checkpoint(path)
    assert config.latent_dim == 0 and metadata["legacy"]
    for a, b in zip(jax.tree_util.tree_leaves(legacy_params), jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(a, b)
