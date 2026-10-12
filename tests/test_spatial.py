"""Checkpoint contract and equivalence against the actual pinned Torch encoder."""
import ast
import copy
import json
import importlib.util
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ngcm_memory.spatial import SOURCE_COMMIT, SpatialEncoder, export_payload


def _payload(*, variable=False, information=False, periodic=True, seed=1):
    rng = np.random.default_rng(seed)
    grid, latent, hidden, factor = (4, 5, 7), (3, 3, 4), 5, 2
    info_grid = (2, *grid[1:])
    lat = np.linspace(-60, 60, grid[1]).tolist()
    lon = (np.arange(grid[2]) * (360 / grid[2] if periodic else 10)).tolist()
    coords = {"lat": lat, "lon": lon}
    variables = [{"name": name, "shape": list(grid[1:]), "coords": coords,
                  "attrs": {"units": unit}}
                 for name, unit in zip(("msl", "t2m", "u10", "v10"), ("Pa", "K", "m/s", "m/s"))]
    state = {}

    def encoder(prefix, inputs):
        prefixes = [f"{prefix}.variable_encoders.{i}" for i in range(inputs)] if variable else [prefix]
        for p in prefixes:
            for name, shape in (("features.0.conv", (hidden, 1 if variable else inputs, 3, 3)),
                                ("features.2.conv", (hidden, hidden, 3, 3)),
                                ("channels", (latent[0], hidden, 1, 1))):
                state[p + "." + name + ".weight"] = rng.normal(0, .1, shape).astype("float32")
                state[p + "." + name + ".bias"] = rng.normal(0, .1, shape[0]).astype("float32")
        if variable:
            state[prefix + ".fusion.weight"] = rng.normal(0, .1, (latent[0], inputs * latent[0], 1, 1)).astype("float32")
            state[prefix + ".fusion.bias"] = rng.normal(0, .1, latent[0]).astype("float32")

    prefix = "bridge.manifold."
    encoder(prefix + "core.manifold.encoder", grid[0])
    if information:
        encoder(prefix + "information", info_grid[0])
        if variable:
            state[prefix + "latent_fusion.weight"] = rng.normal(0, .1, (latent[0], 2 * latent[0], 1, 1)).astype("float32")
            state[prefix + "latent_fusion.bias"] = rng.normal(0, .1, latent[0]).astype("float32")
    state[prefix + "core.latent_mean"] = rng.normal(0, .1, math.prod(latent)).astype("float32")
    state[prefix + "core.latent_scale"] = rng.uniform(.4, 2, math.prod(latent)).astype("float32")
    metadata = {"config": {"representation_kind": "spatial", "grid": list(grid),
                            "spatial_hidden_dim": hidden, "spatial_downsample": factor,
                            "latent_channels": latent[0], "spatial_variable_conditioning": variable},
                "schema": {"variables": variables},
                "mean": rng.normal(0, 1, math.prod(grid)).astype("float32"),
                "scale": rng.uniform(.5, 2, math.prod(grid)).astype("float32"),
                "information_metadata": None, "split": {"train": [0, 10]}}
    if information:
        metadata.update(information_metadata={"shape": list(info_grid), "grid": coords,
                                              "variables": [{"name": "z500", "unit": "m"},
                                                            {"name": "terrain_height", "unit": "m"}]},
                        information_mean=rng.normal(0, .1, math.prod(info_grid)).astype("float32"),
                        information_scale=rng.uniform(.5, 2, math.prod(info_grid)).astype("float32"))
    return {"format": "climate_manifold.downstream.v3", "model": state, "a_metadata": metadata,
            "config": {"bridge": "guided"}, "source_commit": SOURCE_COMMIT,
            "statistical_loss": "signed_measure"}


def test_export_roundtrip_jit_gradient_and_no_overwrite(tmp_path):
    payload = _payload(information=True, variable=True)
    path = tmp_path / "encoder.npz"
    contract = export_payload(payload, path, checkpoint_sha256="a" * 64)
    encoder = SpatialEncoder.from_npz(path)
    x = jnp.arange(4 * 5 * 7, dtype=jnp.float32).reshape(4, 5, 7) / 100
    information = jnp.ones((2, 5, 7))
    latent = jax.jit(encoder.encode)(x, information)
    assert latent.shape == (3, 3, 4)
    assert encoder.encode_lonlat(x, information).shape == (3, 4, 3)
    assert np.isfinite(jax.grad(lambda s: encoder.encode(s, information).sum())(x)).all()
    assert contract["checkpoint_sha256"] == "a" * 64
    assert encoder.surface_fields == ("msl", "t2m", "u10", "v10")
    np.testing.assert_allclose(encoder.weights["surface_mean"].ravel(), payload["a_metadata"]["mean"])
    with pytest.raises(FileExistsError):
        export_payload(payload, path)


def test_information_mandatory_shape_and_raw_global_rejected(tmp_path):
    payload = _payload(information=True)
    path = tmp_path / "encoder.npz"
    export_payload(payload, path)
    encoder = SpatialEncoder.from_npz(path)
    with pytest.raises(ValueError, match="requires observed information"):
        encoder.encode(jnp.zeros((4, 5, 7)))
    with pytest.raises(ValueError, match="wrong shape"):
        encoder.encode(jnp.zeros((4, 5, 7)), jnp.zeros((2, 7, 5)))
    changed = copy.deepcopy(payload)
    changed["config"]["bridge"] = "raw"
    with pytest.raises(ValueError, match="Raw checkpoints"):
        export_payload(changed, tmp_path / "raw.npz")
    changed["config"]["bridge"] = "guided"
    changed["a_metadata"]["config"]["representation_kind"] = "global"
    with pytest.raises(ValueError, match="global DCT"):
        export_payload(changed, tmp_path / "global.npz")


def test_corrupt_normalization_and_unknown_encoder_parameter_rejected(tmp_path):
    payload = _payload()
    payload["a_metadata"]["scale"][0] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        export_payload(payload, tmp_path / "zero.npz")
    payload = _payload()
    payload["model"]["bridge.manifold.core.manifold.encoder.unknown"] = np.ones(1, dtype="float32")
    with pytest.raises(ValueError, match="keys differ"):
        export_payload(payload, tmp_path / "unknown.npz")


def test_observed_history_cutoff_static_information_and_unit_contract(tmp_path):
    xr = pytest.importorskip("xarray")
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_spatial_history.py"
    spec = importlib.util.spec_from_file_location("prepare_spatial_history", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Upstream pressure-information checkpoints also support humidity and
    # pressure velocity. Equal units normalize; height/geopotential do not.
    assert module._canonical_unit("kg kg**-1") == module._canonical_unit("kg/kg")
    assert module._canonical_unit("Pa s**-1") == module._canonical_unit("Pa/s")
    assert module._canonical_unit("m**2 s**-2") != module._canonical_unit("m")
    payload = _payload(information=True)
    payload["a_metadata"]["information_metadata"]["variables"][1]["kind"] = "static"
    path = tmp_path / "encoder.npz"
    contract = export_payload(payload, path)
    times = np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]")
    variables = {}
    for name, unit in zip(contract["surface_fields"] + contract["information_fields"],
                          contract["surface_units"] + contract["information_units"]):
        values = np.ones((3, 5, 7), dtype="float32")
        # Future missing values must not be read at earlier forecast origins.
        values[2] = np.nan
        if name == "terrain_height":
            variables[name] = (("lat", "lon"), values[0], {"units": unit})
        else:
            variables[name] = (("time", "lat", "lon"), values, {"units": unit})
    dataset = xr.Dataset(variables, coords={"time": times, **contract["coordinates"]})
    data = tmp_path / "observations.nc"
    dataset.to_netcdf(data)
    history = tmp_path / "history.npz"
    module.prepare(path, data, history, through="2020-01-02", batch_size=1)
    with np.load(history, allow_pickle=False) as saved:
        assert saved["latents"].shape == (2, 3, 4, 3)
        np.testing.assert_array_equal(saved["times"], times[:2].astype("datetime64[ns]"))
        metadata = json.loads(str(saved["metadata_json"]))
        assert metadata["feature_count"] == 3
        assert metadata["max_history_gap_hours"] == 24.
        assert metadata["history_gap_policy"] == "inferred_median_interval"
    from ngcm_memory.features import SpatialHistory
    horizontal = SimpleNamespace(longitudes=np.deg2rad(contract["latent_coordinates"]["lon"]),
                                 latitudes=np.deg2rad(contract["latent_coordinates"]["lat"]))
    base = SimpleNamespace(model_coords=SimpleNamespace(horizontal=horizontal))
    loaded = SpatialHistory.from_npz(history, base)
    context = loaded.context("2020-01-03", history_steps=2)
    assert context["features"].shape == (2, 3, 4, 3)
    np.testing.assert_allclose(context["dt_hours"], [24, 24])
    with pytest.raises(ValueError, match="stale last observation"):
        loaded.context("2020-01-04", history_steps=2)
    with pytest.raises(ValueError, match="Cannot infer.*one sample"):
        module.prepare(path, data, tmp_path / "one_implicit.npz", through="2020-01-01")
    one_path = tmp_path / "one_explicit.npz"
    module.prepare(path, data, one_path, through="2020-01-01", max_history_gap_hours=6.)
    single = SpatialHistory.from_npz(one_path, base)
    assert single.metadata["history_gap_policy"] == "declared_max_history_gap_hours"
    np.testing.assert_array_equal(single.context("2020-01-01T06", 1)["dt_hours"], [6.])
    dataset["msl"].attrs["units"] = "hPa"
    bad = tmp_path / "wrong_units.nc"
    dataset.to_netcdf(bad)
    with pytest.raises(ValueError, match="Unsupported physical unit"):
        module.prepare(path, bad, tmp_path / "bad.npz", through="2020-01-02")


@pytest.fixture(scope="module")
def actual_source_classes():
    """Compile original class ASTs; do not recreate a second reference encoder.

    This avoids loading unrelated upstream forecasting dependencies. The source
    clone is optional for portable tests; full equivalence is run when present.
    """
    torch = pytest.importorskip("torch")
    default = Path(__file__).resolve().parents[1] / "upstream/climate_manifold"
    if not (default / "src/climate_manifold/spatial.py").is_file():
        default = Path(__file__).resolve().parents[2] / "reference-climate-manifold"
    source = Path(os.environ.get("CLIMATE_MANIFOLD_SOURCE", default))
    path = source / "src/climate_manifold/spatial.py"
    if not path.is_file():
        pytest.skip("Set CLIMATE_MANIFOLD_SOURCE to the pinned upstream checkout")
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert sha == SOURCE_COMMIT, "Equivalence reference must use the pinned architecture commit"
    names = {"GeographicConv2d", "SpatialEncoder", "VariableSpatialEncoder"}
    parsed = ast.parse(path.read_text())
    nodes = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {"math": math, "torch": torch, "nn": torch.nn, "F": torch.nn.functional}
    exec(compile(module, str(path), "exec"), namespace)
    return torch, namespace


@pytest.mark.parametrize("variable,information,periodic", [
    (False, False, False), (False, True, True), (True, False, True), (True, True, False),
])
def test_exact_pinned_source_equivalence(tmp_path, actual_source_classes, variable, information, periodic):
    torch, classes = actual_source_classes
    payload = _payload(variable=variable, information=information, periodic=periodic)
    metadata, state = payload["a_metadata"], payload["model"]
    config = metadata["config"]
    grid = tuple(config["grid"])
    latent = (3, 3, 4)
    cls = classes["VariableSpatialEncoder" if variable else "SpatialEncoder"]
    surface_encoder = cls(grid, latent, 5, 2, periodic)
    p = "bridge.manifold.core.manifold.encoder."
    surface_encoder.load_state_dict({k.removeprefix(p): torch.tensor(v) for k, v in state.items() if k.startswith(p)})
    rng = np.random.default_rng(21)
    # Multiple leading dimensions exercise history/batch flattening.
    surface = rng.normal(size=(2, 3, *grid)).astype("float32")
    x = (surface - metadata["mean"].reshape(grid)) / metadata["scale"].reshape(grid)
    z = surface_encoder(torch.tensor(x.reshape(2, 3, -1))).reshape(-1, *latent)
    info = None
    if information:
        info_grid = tuple(metadata["information_metadata"]["shape"])
        info_encoder = cls(info_grid, latent, 5, 2, periodic)
        p = "bridge.manifold.information."
        info_encoder.load_state_dict({k.removeprefix(p): torch.tensor(v) for k, v in state.items() if k.startswith(p)})
        info = rng.normal(size=(2, 3, *info_grid)).astype("float32")
        i = (info - metadata["information_mean"].reshape(info_grid)) / metadata["information_scale"].reshape(info_grid)
        zi = info_encoder(torch.tensor(i.reshape(2, 3, -1))).reshape(-1, *latent)
        if variable:
            fusion = torch.nn.Conv2d(6, 3, 1)
            p = "bridge.manifold.latent_fusion."
            fusion.load_state_dict({k.removeprefix(p): torch.tensor(v) for k, v in state.items() if k.startswith(p)})
            z = fusion(torch.cat((z, zi), dim=1))
        else:
            z = z + zi
    z = z.reshape(2, 3, *latent)
    z = (z - torch.tensor(state["bridge.manifold.core.latent_mean"].reshape(latent))) / torch.tensor(
        state["bridge.manifold.core.latent_scale"].reshape(latent))
    path = tmp_path / "exact.npz"
    export_payload(payload, path)
    actual = jax.jit(SpatialEncoder.from_npz(path).encode)(surface, info)
    np.testing.assert_allclose(np.asarray(actual), z.detach().numpy(), rtol=2e-5, atol=2e-6)
