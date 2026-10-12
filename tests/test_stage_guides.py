"""Causal forecast/time/grid boundary tests for frozen NeuralGCM stage guides."""

from dataclasses import dataclass, replace
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from ngcm_memory.stage_guides import GuideFrame, NeuralGCMStageGuide


@dataclass(frozen=True)
class ToyState:
    model_state: object
    closure_state: int
    latent_state: object


class ToyAdapter:
    """Immutable boundary harness, explicitly not a physical-model substitute."""

    enabled = True
    dt_hours = 0.5
    config = SimpleNamespace(latent_dim=2)

    def __init__(self):
        self.base_model = SimpleNamespace(
            data_coords=SimpleNamespace(
                vertical=SimpleNamespace(centers=np.array([250., 500., 850.])),
                horizontal=SimpleNamespace(longitudes=np.deg2rad([0., 90., 180., 270.]),
                                           latitudes=np.deg2rad([-60., 0., 60.])),
            ),
            model_coords=SimpleNamespace(horizontal=SimpleNamespace(
                longitudes=np.deg2rad([0., 120., 240.]), latitudes=np.deg2rad([-45., 45.]),
            )),
        )
        self.advance_calls = []
        self.decode_times = []

    def advance(self, params, state, forcings):
        self.advance_calls.append(forcings.copy())
        count = state.closure_state + 1
        model = SimpleNamespace(
            sim_time=state.model_state.sim_time + 0.125,
            temperature=state.model_state.temperature + params["increment"],
        )
        latent = SimpleNamespace(latent=state.latent_state.latent + count)
        return ToyState(model, count, latent)

    def decode(self, state, forcings):
        self.decode_times.append(float(forcings["sim_time"]))
        return {"temperature": state.model_state.temperature}


def setup(**kwargs):
    adapter = ToyAdapter()
    temperature = (np.array([230., 250., 270.])[:, None, None]
                   + np.arange(4)[None, :, None] + np.array([-1., 0., 1.])[None, None, :])
    latent = (np.array([0., 5.])[:, None, None]
              + np.arange(3)[None, :, None] + np.array([-1., 1.])[None, None, :])
    state = ToyState(SimpleNamespace(sim_time=np.float32(2.), temperature=temperature), 0,
                     SimpleNamespace(latent=latent))
    args = dict(target_longitude=[-90., 0., 90.], target_latitude=[60., 0., -60.],
                pressure_levels=[850., 500.], temperature_means=[270., 250.],
                temperature_stds=[2., 1.])
    args.update(kwargs)
    guide = NeuralGCMStageGuide(adapter, {"increment": 0.5}, **args)
    return adapter, state, guide


@pytest.fixture
def jax_available():
    pytest.importorskip("jax")


def test_explicit_levels_normalization_coordinates_and_axes(jax_available):
    adapter, state, guide = setup()
    frame = guide.current_frame(state, {"sim_time": 99.}, lead_hours=0)
    # Decoded lon-major data become north-to-south lat-major channels. -90
    # uses the periodic 270-degree source and the requested levels are reordered.
    expected = np.array([[4., 1., 2.], [3., 0., 1.], [2., -1., 0.]])
    np.testing.assert_allclose(frame.physical, np.stack([expected / 2, expected]), atol=1e-6)
    expected_latent = np.array([[2.5, 1., 1.75], [1.5, 0., .75], [.5, -1., -.25]])
    np.testing.assert_allclose(frame.latent, np.stack([expected_latent, expected_latent + 5]), atol=1e-6)
    assert frame.physical.dtype == np.float32 and frame.latent.dtype == np.float32
    assert frame.physical.shape == (2, 3, 3)
    assert adapter.decode_times == [2.]
    assert adapter.advance_calls == []


def test_interval_advances_once_then_four_stage_reads_do_not_advance(jax_available):
    adapter, state, guide = setup()
    boundary = np.array([[[280.]]])
    forcings = {"sim_time": np.float32(2), "sea_surface_temperature": boundary}
    original_temperature = state.model_state.temperature.copy()
    original_latent = state.latent_state.latent.copy()
    next_state, frame = guide.advance(state, forcings, interval_hours=1., lead_hours=1.)
    assert len(adapter.advance_calls) == 2  # Two native half-hour steps, not four stages.
    assert [float(f["sim_time"]) for f in adapter.advance_calls] == [2., 2.125]
    assert all(f["sea_surface_temperature"] is boundary for f in adapter.advance_calls)
    assert adapter.decode_times[-1] == 2.25
    assert frame.lead_hours == 1.
    assert next_state.closure_state == 2
    np.testing.assert_array_equal(next_state.latent_state.latent, original_latent + 3)
    for _ in range(4):
        stage_frame = guide.current_frame(next_state, forcings, lead_hours=1.)
        np.testing.assert_array_equal(stage_frame.physical, frame.physical)
        np.testing.assert_array_equal(stage_frame.latent, frame.latent)
    assert len(adapter.advance_calls) == 2
    # The caller must carry the whole augmented state; neither memory nor z reset.
    final_state, final_frame = guide.advance(next_state, forcings, interval_hours=1., lead_hours=2.)
    assert final_frame.lead_hours == 2.
    assert final_state.closure_state == 4
    np.testing.assert_array_equal(final_state.latent_state.latent, original_latent + 10)
    np.testing.assert_array_equal(state.model_state.temperature, original_temperature)
    np.testing.assert_array_equal(state.latent_state.latent, original_latent)
    assert state.closure_state == 0 and float(state.model_state.sim_time) == 2.
    assert float(forcings["sim_time"]) == 2.
    assert forcings["sea_surface_temperature"] is boundary


def test_default_lead_is_interval_and_outputs_do_not_alias_state(jax_available):
    _, state, guide = setup()
    _, frame = guide.advance(state, {}, interval_hours=0.5)
    assert frame.lead_hours == .5
    assert not np.shares_memory(frame.latent, state.latent_state.latent)
    assert not np.shares_memory(frame.physical, state.model_state.temperature)


@pytest.mark.parametrize("interval", [0, -1, 0.75, 0.1, float("nan"), True, [1], "1"])
def test_invalid_interval_cannot_consume_state(interval):
    adapter, state, guide = setup()
    with pytest.raises(ValueError):
        guide.advance(state, {}, interval_hours=interval)
    assert adapter.advance_calls == []


@pytest.mark.parametrize("args,pattern", [
    ({"pressure_levels": [50000., 85000.]}, "hPa"),
    ({"pressure_levels": [500., 500.]}, "unique"),
    ({"pressure_levels": [501., 850.]}, "hPa"),
    ({"pressure_levels": []}, "pressure_levels"),
    ({"temperature_means": [250.]}, "temperature_means"),
    ({"temperature_stds": [2., 0.]}, "positive"),
    ({"temperature_stds": [float("inf"), 1.]}, "finite"),
    ({"target_longitude": [0., 360.]}, "unique"),
    ({"target_longitude": [0., float("nan")]}, "finite"),
    ({"target_latitude": [-91., 90.]}, "degrees"),
    ({"target_latitude": [0.]}, "two finite"),
])
def test_explicit_contract_validation(args, pattern):
    with pytest.raises(ValueError, match=pattern):
        setup(**args)


def test_requires_enabled_recurrent_generator():
    adapter, _, guide = setup()
    arguments = dict(target_longitude=guide.target_longitude, target_latitude=guide.target_latitude,
                     pressure_levels=guide.pressure_levels, temperature_means=guide.temperature_means,
                     temperature_stds=guide.temperature_stds)
    adapter.enabled = False
    with pytest.raises(ValueError, match="enabled"):
        NeuralGCMStageGuide(adapter, {}, **arguments)
    adapter.enabled = True
    adapter.config = SimpleNamespace(latent_dim=0)
    with pytest.raises(ValueError, match="latent_dim"):
        NeuralGCMStageGuide(adapter, {}, **arguments)


@pytest.mark.parametrize("clock", [np.datetime64("2020-01-01"), np.datetime64("NaT", "ns"),
                                   np.nan, np.array([1., 2.])])
def test_invalid_model_clock_or_origin_series_rejected(clock):
    adapter, state, guide = setup()
    broken = replace(state, model_state=SimpleNamespace(sim_time=clock))
    with pytest.raises(ValueError, match="sim_time"):
        guide.advance(broken, {}, interval_hours=1.)
    with pytest.raises(ValueError, match="sim_time"):
        guide.advance(state, {"sim_time": clock}, interval_hours=1.)
    assert adapter.advance_calls == []


def test_missing_latent_and_mislabelled_lead_fail_before_advancing():
    adapter, state, guide = setup()
    with pytest.raises(ValueError, match="latent"):
        guide.advance(replace(state, latent_state=None), {}, interval_hours=1.)
    with pytest.raises(ValueError, match="Cumulative"):
        guide.advance(state, {}, interval_hours=1., lead_hours=.5)
    assert adapter.advance_calls == []


def test_bad_decoded_layout_and_latent_layout_rejected(jax_available):
    _, state, guide = setup()
    bad_model = SimpleNamespace(sim_time=2., temperature=np.zeros((3, 3, 4)))
    with pytest.raises(ValueError, match="pressure,lon,lat"):
        guide.current_frame(replace(state, model_state=bad_model), {}, lead_hours=0)
    with pytest.raises(ValueError, match="D,lon,lat"):
        guide.current_frame(replace(state, latent_state=SimpleNamespace(latent=np.zeros((2, 2, 3)))),
                            {}, lead_hours=0)


def test_no_future_labels_or_observation_reset_api():
    _, state, guide = setup()
    assert set(inspect.signature(guide.advance).parameters) == {
        "state", "origin_forcings", "interval_hours", "lead_hours",
    }
    with pytest.raises(TypeError, match="future_targets"):
        guide.advance(state, {}, interval_hours=1., future_targets=np.ones((2, 3, 4)))


def test_frame_contract_and_array_copy():
    values = np.ones((2, 3, 4))
    frame = GuideFrame(values, values, 0)
    values[:] = 9
    np.testing.assert_array_equal(frame.physical, 1.)
    for physical, latent, lead in [(np.ones((2, 3)), values, 0),
                                   (values, np.ones((2, 4, 3)), 0),
                                   (values * np.nan, values, 0), (values, values, -1)]:
        with pytest.raises(ValueError):
            GuideFrame(physical, latent, lead)


def test_real_memory_adapter_steps_match_direct_recurrence_and_preserve_rng(jax_available):
    """Exercise the real JAX thermal/latent closure around the small base harness."""
    import jax
    import jax.numpy as jnp

    from ngcm_memory.closure import ClosureConfig, init_params
    from ngcm_memory.neuralgcm_adapter import NeuralGCMMemoryAdapter
    from test_neuralgcm_adapter import BaseModel, ModelState, Prognostics, Transform

    base = BaseModel()
    grid = Transform()
    grid.longitudes = np.deg2rad([0., 90., 180., 270.])
    grid.latitudes = np.deg2rad([-60., 0., 60.])
    base.model_coords = SimpleNamespace(vertical=SimpleNamespace(layers=2), horizontal=grid)
    base.data_coords = SimpleNamespace(vertical=SimpleNamespace(centers=np.array([500., 850.])),
                                       horizontal=grid)
    config = ClosureConfig(levels=2, feature_dim=3, latent_dim=1, hidden_dim=2, latent_hidden_dim=2)
    adapter = NeuralGCMMemoryAdapter(base, config)
    params = init_params(jax.random.key(5), config)
    native = ModelState(Prognostics(jnp.ones((2, 4, 3)), jnp.asarray(0.), jnp.ones((2, 4, 3))),
                        {}, jnp.asarray(0.), {})
    initial = adapter.initialize(native, jax.random.key(7))
    guide = NeuralGCMStageGuide(adapter, params, target_longitude=[0., 90., 180., 270.],
                                target_latitude=[-60., 0., 60.], pressure_levels=[850.],
                                temperature_means=[0.], temperature_stds=[1.])
    actual, frame = guide.advance(initial, {"sim_time": jnp.asarray(0.)}, interval_hours=1.)
    expected = initial
    for _ in range(2):
        expected = adapter.advance(params, expected, {"sim_time": expected.model_state.state.sim_time})
    for lhs, rhs in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        if jax.dtypes.issubdtype(lhs.dtype, jax.dtypes.prng_key):
            lhs, rhs = jax.random.key_data(lhs), jax.random.key_data(rhs)
        np.testing.assert_array_equal(lhs, rhs)
    assert np.linalg.norm(actual.latent_state.latent) > 0
    assert np.linalg.norm(actual.latent_state.memory) > 0
    np.testing.assert_allclose(frame.latent, np.asarray(actual.latent_state.latent).transpose(0, 2, 1),
                               rtol=1e-6, atol=1e-8)
    keys_before = (np.array(jax.random.key_data(actual.closure_state.key)),
                   np.array(jax.random.key_data(actual.latent_state.key)))
    for _ in range(4):
        guide.current_frame(actual, {}, lead_hours=1.)
    np.testing.assert_array_equal(jax.random.key_data(actual.closure_state.key), keys_before[0])
    np.testing.assert_array_equal(jax.random.key_data(actual.latent_state.key), keys_before[1])
    np.testing.assert_array_equal(initial.latent_state.latent, 0.)
