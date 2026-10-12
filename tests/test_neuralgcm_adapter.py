"""Boundary tests; scripts/smoke_neuralgcm.py tests the actual upstream model."""

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ngcm_memory.closure import ClosureConfig
from ngcm_memory.neuralgcm_adapter import NeuralGCMMemoryAdapter
import ngcm_memory.neuralgcm_adapter as adapter_module


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Prognostics:
    temperature_variation: object
    sim_time: object
    humidity: object


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ModelState:
    state: object
    memory: object
    randomness: object
    diagnostics: object


class Transform:
    nodal_shape = (4, 3)
    quadrature_weights = np.broadcast_to(np.array([1.0, 2.0, 1.0]), (4, 3))

    def to_nodal(self, x):
        return 2 * x

    def to_modal(self, x):
        return x / 2


class BaseModel:
    """Small deterministic boundary harness, not a NeuralGCM substitute."""

    timestep = np.timedelta64(30, "m")
    model_coords = SimpleNamespace(
        vertical=SimpleNamespace(layers=2), horizontal=Transform()
    )

    def from_nondim_units(self, x, units):
        assert units == "K"
        return 3 * x

    def to_nondim_units(self, x, units):
        assert units == "K"
        return x / 3

    def advance(self, state, forcings):
        del forcings
        return dataclasses.replace(
            state,
            state=dataclasses.replace(
                state.state,
                temperature_variation=state.state.temperature_variation + 0.25,
                sim_time=state.state.sim_time + 0.5,
            ),
        )

    def decode(self, state, forcings):
        del forcings
        return {"temperature": self.from_nondim_units(
            self.model_coords.horizontal.to_nodal(state.state.temperature_variation),
            "K",
        )}


def setup_adapter(**kwargs):
    config = ClosureConfig(feature_dim=2, levels=2, hidden_dim=3)
    base = BaseModel()
    state = ModelState(
        Prognostics(jnp.ones((2, 4, 3)), jnp.asarray(0.0), jnp.ones((2, 4, 3))),
        {"existing_memory": jnp.asarray(123.0)},
        jnp.asarray(7.0),
        {"rainfall": jnp.asarray(4.0)},
    )
    adapter = NeuralGCMMemoryAdapter(base, config, **kwargs)
    return adapter, adapter.initialize(state, jax.random.key(2))


def test_disabled_closure_exact_passthrough():
    def forbidden_features(*args):
        raise AssertionError("Disabled closure must not call feature function")

    adapter, state = setup_adapter(enabled=False, feature_fn=forbidden_features)
    actual = adapter.advance(None, state, {})
    expected = adapter.base_model.advance(state.model_state, {})
    for x, y in zip(jax.tree.leaves(actual.model_state), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(x, y)
    assert actual.closure_state is state.closure_state


def test_kelvin_conversion_internal_dt_and_other_state_preservation(monkeypatch):
    adapter, state = setup_adapter(feature_scale_kelvin=2.0)

    def constant_correction(params, memory, features, dt, config):
        assert dt == 0.5
        np.testing.assert_allclose(features, 3.0)
        return jnp.ones((2, 4, 3)), memory

    monkeypatch.setattr(adapter_module, "closure_step", constant_correction)
    actual = adapter.advance({}, state, {})
    np.testing.assert_allclose(actual.model_state.state.temperature_variation, 1.25 + 1 / 6)
    np.testing.assert_array_equal(actual.model_state.state.sim_time, 0.5)
    assert actual.model_state.memory is state.model_state.memory
    assert actual.model_state.randomness is state.model_state.randomness
    assert actual.model_state.diagnostics is state.model_state.diagnostics
    assert actual.model_state.state.humidity is state.model_state.state.humidity


def test_correction_removes_gaussian_area_mean_only_when_requested(monkeypatch):
    adapter, state = setup_adapter(remove_global_mean=True)
    field = jnp.broadcast_to(jnp.array([0.0, 3.0, 0.0]), (2, 4, 3))
    monkeypatch.setattr(
        adapter_module, "closure_step", lambda p, s, f, dt, c: (field, s)
    )
    actual = adapter.advance({}, state, {})
    delta = actual.model_state.state.temperature_variation - 1.25
    np.testing.assert_allclose(jnp.sum(delta * adapter.area_weights, (-2, -1)), 0, atol=1e-6)
    np.testing.assert_allclose(delta[0, 0], [-0.25, 0.25, -0.25])


def test_feature_axes_are_explicit():
    adapter, state = setup_adapter(
        feature_fn=lambda s, f: jnp.zeros((2, 3, 4))
    )
    with pytest.raises(ValueError, match="feature,lon,lat"):
        adapter.features(state.model_state, {})


def test_validates_sigma_levels_and_feature_dimension():
    with pytest.raises(ValueError, match="sigma levels"):
        NeuralGCMMemoryAdapter(BaseModel(), ClosureConfig(feature_dim=3, levels=3))
    with pytest.raises(ValueError, match="feature_dim == levels"):
        NeuralGCMMemoryAdapter(BaseModel(), ClosureConfig(feature_dim=4, levels=2))


def test_multistep_jit_gradient_uses_corrected_recurrent_state(monkeypatch):
    adapter, state = setup_adapter(feature_scale_kelvin=6.0)
    monkeypatch.setattr(
        adapter_module,
        "closure_step",
        lambda p, s, f, dt, c: (p * f * dt, s),
    )

    def loss(parameter):
        carry = state
        for _ in range(3):
            carry = adapter.advance(parameter, carry, {})
        return jnp.mean(adapter.decode(carry, {})["temperature"] ** 2)

    value, gradient = jax.jit(jax.value_and_grad(loss))(jnp.asarray(0.2))
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient) and abs(float(gradient)) > 0
