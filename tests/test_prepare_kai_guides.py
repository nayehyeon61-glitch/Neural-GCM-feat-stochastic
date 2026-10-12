"""Export CLI integration with small deterministic boundary doubles.

The physical guide bridge, regridding, NPZ serialization and archive reader are
real. Only heavyweight checkpoint loading, NeuralGCM advance and encoded
history are replaced. These checks establish wiring/causality, not model skill.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import numpy as np
import pytest
import xarray as xr

pytest.importorskip("torch")

from ngcm_memory.kai_training import GuideArchive


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_kai_guides.py"
_SPEC = importlib.util.spec_from_file_location("prepare_kai_guides_under_test", _SCRIPT)
exporter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(exporter)


class PhysicalState(NamedTuple):
    temperature: object
    sim_time: object


class LatentState(NamedTuple):
    latent: object


class State(NamedTuple):
    model_state: PhysicalState
    closure_state: object
    latent_state: LatentState


class Base:
    timestep = np.timedelta64(1, "h")

    def __init__(self):
        self.encodes = []
        grid = SimpleNamespace(
            longitudes=np.deg2rad(np.arange(12) * 30),
            latitudes=np.deg2rad(np.linspace(-75, 75, 6)),
        )
        self.data_coords = SimpleNamespace(horizontal=grid, vertical=SimpleNamespace(centers=np.array([500., 850.])))
        self.model_coords = SimpleNamespace(horizontal=grid)

    def encode(self, inputs, forcings, rng_key):
        del forcings, rng_key
        self.encodes.append(float(inputs["sim_time"]))
        return PhysicalState(np.array(inputs["temperature"]), inputs["sim_time"])


class Adapter:
    enabled = True
    dt_hours = 1.

    def __init__(self, base, config, **_kwargs):
        self.base_model, self.config = base, config
        self.warmup_calls, self.forecast_calls = [], []

    def encode(self, inputs, forcings, key):
        return State(self.base_model.encode(inputs, forcings, key), 0,
                     LatentState(np.zeros((2, 12, 6), dtype=np.float32)))

    def advance_warmup(self, params, state, forcings):
        del params
        self.warmup_calls.append((float(forcings["sim_time"]), float(forcings["boundary"])))
        physical = state.model_state
        return state._replace(
            model_state=PhysicalState(physical.temperature + forcings["boundary"], physical.sim_time + 1),
            closure_state=state.closure_state + 1,
        )

    def advance(self, params, state, forcings):
        del params
        self.forecast_calls.append((float(forcings["sim_time"]), float(forcings["boundary"])))
        physical = state.model_state
        return State(
            PhysicalState(physical.temperature + forcings["boundary"], physical.sim_time + 1),
            state.closure_state + 1, LatentState(state.latent_state.latent + 1),
        )

    def decode(self, state, forcings):
        assert float(forcings["sim_time"]) == float(state.model_state.sim_time)
        return {"temperature": state.model_state.temperature}


class History:
    feature_dim = 2
    max_history_gap_hours = 6.
    metadata = {"encoder_sha256": "test-encoder"}
    artifact_sha256 = "test-spatial-history"

    def __init__(self):
        self.requested_origins, self.initialized = [], []

    def transition_context(self, origin, history_steps):
        assert history_steps == 1
        self.requested_origins.append(origin)
        latent = np.broadcast_to(np.array([7., 9.])[:, None, None], (2, 12, 6)).copy()
        return {"initial_latent": latent}

    def transition_initializer(self, _config):
        def initialize(_params, state, context):
            self.initialized.append(state)
            return state._replace(latent_state=LatentState(context["initial_latent"]))
        return initialize


@pytest.fixture
def problem(tmp_path, monkeypatch):
    times = np.datetime64("2020-01-01T00", "ns") + np.arange(5) * np.timedelta64(6, "h")
    temperature = np.broadcast_to(
        np.array([282., 294.])[None, :, None, None] + np.arange(5)[:, None, None, None],
        (5, 2, 12, 6),
    ).copy()
    inputs = {"temperature": temperature, "sim_time": np.arange(5, dtype=float) * 6}
    forcings = {"boundary": np.arange(5, dtype=float) + 1, "sim_time": inputs["sim_time"].copy()}
    initial_fields = np.broadcast_to(
        np.array([280., 290., 10.])[None, :, None, None] + np.arange(5)[:, None, None, None],
        (5, 3, 6, 12),
    ).copy()
    kai = xr.Dataset({"fields": (("time", "channel", "latitude", "longitude"), initial_fields)}, coords={
        "time": times, "channel": ["t500", "t850", "other"],
        "latitude": np.linspace(-75, 75, 6), "longitude": np.arange(12) * 30.,
    })
    base, history = Base(), History()
    config = SimpleNamespace(latent_dim=2)
    adapters = []
    def make_adapter(base, config, **kwargs):
        adapter = Adapter(base, config, **kwargs)
        adapters.append(adapter)
        return adapter
    data = SimpleNamespace(
        times=times, inputs=inputs, forcings=forcings, kai=kai,
        native=xr.Dataset(coords={"time": times}), base=base, history=history,
        adapters=adapters, config=config, root=tmp_path,
    )
    trained = {
        "spatial_mode": "recurrent", "base_checkpoint_sha256": "test-base",
        "training_last_observation": "2018-12-31", "validation_last_observation": "2019-06-30",
        "observation_interval_hours": 6., "feature_scale_kelvin": 10.,
        "remove_global_mean": False, "warmup_steps": 1, "spatial_history_steps": 1,
        "spatial_history": {"encoder_sha256": "test-encoder", "max_history_gap_hours": 6.},
    }
    monkeypatch.setattr(exporter, "load_closure_checkpoint", lambda _path: ({}, config, trained))
    monkeypatch.setattr(exporter, "sha256_file", lambda path: "test-base" if path.name == "base.pkl" else "test-closure")
    monkeypatch.setattr(exporter, "load_local_problem", lambda *_args, **_kw: (base, data.native, data.inputs, data.forcings))
    monkeypatch.setattr(exporter, "SpatialHistory", SimpleNamespace(from_npz=lambda *_args: history))
    monkeypatch.setattr(exporter, "NeuralGCMMemoryAdapter", make_adapter)
    np.save(tmp_path / "means.npy", np.array([280., 290., 10.]))
    np.save(tmp_path / "stds.npy", np.array([2., 4., 5.]))
    return data


def export_once(problem, name, *, without_targets=False, extra_args=()):
    source = problem.root / f"{name}.nc"
    problem.kai.to_netcdf(source, engine="scipy")
    output = problem.root / name
    args = [
        "--base-checkpoint", str(problem.root / "base.pkl"), "--closure", str(problem.root / "closure.npz"),
        "--data", str(problem.root / "native.nc"), "--spatial-history", str(problem.root / "history.npz"),
        "--kai-data", str(source), "--global-means", str(problem.root / "means.npy"),
        "--global-stds", str(problem.root / "stds.npy"), "--temperature-channels", "0,1",
        "--pressure-levels", "500,850", "--output-dir", str(output), "--interval-hours", "6",
        "--horizon-steps", "2", "--ensemble-size", "2", "--start-time", str(problem.times[2]),
        "--max-origins", "1", *extra_args,
    ]
    if without_targets:
        args.append("--without-targets")
    exporter.main(args)
    paths = list(output.glob("*.npz"))
    assert len(paths) == 1
    return GuideArchive.from_file(paths[0], require_targets=not without_targets)


def test_origin_warmup_retains_memory_then_refreshes_origin_physics(problem):
    adapter = Adapter(problem.base, problem.config)
    context = problem.history.transition_context(problem.times[2], 1)
    state = exporter._origin_state(
        adapter, {}, problem.inputs, problem.forcings, 2, 1, 6, jax.random.key(7),
        problem.history.transition_initializer(problem.config), context,
    )
    assert problem.base.encodes == [6., 6., 12.]
    assert adapter.warmup_calls == [(float(hour), 2.) for hour in range(6, 12)]
    assert state.closure_state == 6
    assert state.model_state.sim_time == 12
    np.testing.assert_array_equal(state.model_state.temperature, problem.inputs["temperature"][2])
    np.testing.assert_array_equal(state.latent_state.latent, context["initial_latent"])
    assert adapter.forecast_calls == []


def test_exported_archive_matches_training_schema_and_first_forecast_lead(problem):
    archive = export_once(problem, "ordinary")
    assert archive.initial.shape == (3, 6, 12)
    assert archive.targets.shape == (2, 3, 6, 12)
    assert archive.physical_guides.shape == archive.latent_guides.shape == (2, 2, 2, 6, 12)
    np.testing.assert_array_equal(archive.lead_hours, [6., 12.])
    assert archive.schema()["step_hours"] == 6
    np.testing.assert_allclose(archive.physical_guides[0, :, :, 0, 0], [[11., 6.], [20., 10.5]])
    np.testing.assert_allclose(archive.latent_guides[0, :, :, 0, 0], [[13., 15.], [19., 21.]])
    np.testing.assert_allclose(archive.initial[:, 0, 0], [1., .5, .4])
    np.testing.assert_allclose(archive.targets[:, :, 0, 0], [[1.5, .75, .6], [2., 1., .8]])
    assert archive.metadata["future_observations_used"] is False
    assert archive.metadata["forcing_mode"] == "persistence"
    assert problem.history.requested_origins == [problem.times[2]]
    adapter = problem.adapters[-1]
    assert adapter.forecast_calls == [(float(hour), 3.) for hour in range(12, 24)] * 2
    assert all(time <= 12 for time in problem.base.encodes)


def test_future_observations_and_boundaries_do_not_change_generated_guides(problem):
    first = export_once(problem, "first")
    problem.inputs["temperature"][3:] += 1e4
    problem.forcings["boundary"][3:] += 1e6
    problem.kai["fields"].values[3:] += 1e5
    altered = export_once(problem, "altered")
    np.testing.assert_array_equal(first.initial, altered.initial)
    np.testing.assert_array_equal(first.physical_guides, altered.physical_guides)
    np.testing.assert_array_equal(first.latent_guides, altered.latent_guides)
    assert not np.array_equal(first.targets, altered.targets)


def test_without_targets_exports_beyond_available_observations(problem):
    problem.kai = problem.kai.isel(time=[2])
    problem.native = problem.native.isel(time=slice(0, 3))
    problem.inputs = {key: value[:3] for key, value in problem.inputs.items()}
    problem.forcings = {key: value[:3] for key, value in problem.forcings.items()}
    archive = export_once(problem, "forecast_only", without_targets=True)
    assert archive.targets is None
    assert archive.last_valid > problem.native.time.values[-1]
    np.testing.assert_allclose(archive.physical_guides[0, :, :, 0, 0], [[11., 6.], [20., 10.5]])
    with pytest.raises(ValueError, match="require targets"):
        GuideArchive.from_file(archive.path, require_targets=True)


def test_interval_must_cover_integer_native_steps_before_archive_is_written(problem):
    with pytest.raises(ValueError, match="integer multiple"):
        export_once(problem, "fractional_native_step", without_targets=True,
                    extra_args=("--interval-hours", "1.5"))
    assert not list((problem.root / "fractional_native_step").glob("*.npz"))


@pytest.mark.parametrize("interval", ["0", "nan", "inf", "-1"])
def test_bad_interval_rejected_before_checkpoint_loading(monkeypatch, interval):
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid interval reached checkpoint loading")
    monkeypatch.setattr(exporter, "load_closure_checkpoint", forbidden)
    args = [
        "--base-checkpoint", "base", "--closure", "closure", "--data", "native",
        "--spatial-history", "history", "--kai-data", "kai", "--global-means", "means",
        "--global-stds", "stds", "--temperature-channels", "0", "--pressure-levels", "500",
        "--output-dir", "unused", "--interval-hours", interval,
    ]
    with pytest.raises(SystemExit):
        exporter.main(args)


@pytest.mark.parametrize("bounds", [
    ("--start-time", "NaT"), ("--end-time", "NaT"),
    ("--start-time", "2020-02-02", "--end-time", "2020-02-01"),
    ("--start-time", "2020-02-02", "--end-time", "2020-02-02"),
])
def test_bad_time_bounds_rejected_before_checkpoint_loading(problem, monkeypatch, bounds):
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid bounds reached checkpoint loading")
    monkeypatch.setattr(exporter, "load_closure_checkpoint", forbidden)
    with pytest.raises(SystemExit):
        export_once(problem, "bad_bounds", extra_args=bounds)


@pytest.mark.parametrize("name,value", [
    ("warmup_steps", -1), ("warmup_steps", 1.5), ("warmup_steps", True),
    ("spatial_history_steps", 0), ("spatial_history_steps", 1.5),
    ("spatial_history_steps", False),
])
def test_invalid_checkpoint_history_lengths_rejected_before_native_loading(problem, monkeypatch, name, value):
    _, _, trained = exporter.load_closure_checkpoint(None)
    trained[name] = value
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid history lengths reached native model loading")
    monkeypatch.setattr(exporter, "load_local_problem", forbidden)
    with pytest.raises(SystemExit):
        export_once(problem, "bad_history_length")
    assert not problem.adapters


@pytest.mark.parametrize("warmup,internal_steps", [(-1, 6), (3, 6), (1, 0), (1, 1.5)])
def test_direct_origin_helper_rejects_invalid_counts_before_encoding(problem, warmup, internal_steps):
    adapter = Adapter(problem.base, problem.config)
    with pytest.raises(ValueError):
        exporter._origin_state(
            adapter, {}, problem.inputs, problem.forcings, 2, warmup, internal_steps,
            jax.random.key(7), lambda _params, state, _context: state, {},
        )
    assert problem.base.encodes == []
    assert adapter.warmup_calls == []


@pytest.mark.parametrize("kind", ["numeric_time", "repeated_time", "repeated_channel"])
def test_canonical_input_rejects_ambiguous_coordinates(problem, kind):
    ds = problem.kai.copy()
    if kind == "numeric_time":
        ds = ds.assign_coords(time=np.arange(5))
    elif kind == "repeated_time":
        ds = ds.assign_coords(time=[problem.times[0]] * 5)
    else:
        ds = ds.assign_coords(channel=["temperature"] * 3)
    with pytest.raises(ValueError):
        exporter._canonical_kai(ds)
