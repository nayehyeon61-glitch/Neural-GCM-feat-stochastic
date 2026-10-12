"""Exact JAX inference for the Climate Manifold spatial statistical encoder.

Source architecture: climate_manifold/spatial.py at ``SOURCE_COMMIT``. Exported
artifacts contain weights and the *frozen training* normalization, never training
observations. This module imports no Torch and can run inside ``jax.jit``.

Inputs must already match the checkpoint's variable order, physical units and
grid. No sea-level/surface-pressure or 10 m/lowest-level wind substitution is
made. Spatial regridding is an explicit caller responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

SOURCE_COMMIT = "bd1592fe9283f650b5b1f62deb7e196743b9db9a"
SOURCE_REPOSITORY = "https://github.com/nayehyeon61-glitch/climate_manifold"
ARTIFACT_FORMAT = "ngcm_memory.spatial_encoder.v1"
SUPPORTED_CHECKPOINT_FORMATS = tuple(f"climate_manifold.downstream.v{i}" for i in (1, 2, 3))


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.kind != "f" or not np.isfinite(result).all():
        raise ValueError("Encoder weights and normalization must be finite floating arrays")
    return result.astype(np.float32)


def _expected_encoder(prefix, input_channels, hidden, channels, variable):
    shapes = {}
    prefixes = [f"{prefix}.variable_encoders.{i}" for i in range(input_channels)] if variable else [prefix]
    for p in prefixes:
        inputs = 1 if variable else input_channels
        shapes.update({f"{p}.features.0.conv.weight": (hidden, inputs, 3, 3),
                       f"{p}.features.0.conv.bias": (hidden,),
                       f"{p}.features.2.conv.weight": (hidden, hidden, 3, 3),
                       f"{p}.features.2.conv.bias": (hidden,),
                       f"{p}.channels.weight": (channels, hidden, 1, 1),
                       f"{p}.channels.bias": (channels,)})
    if variable:
        shapes.update({f"{prefix}.fusion.weight": (channels, input_channels * channels, 1, 1),
                       f"{prefix}.fusion.bias": (channels,)})
    return shapes


def _expected_shapes(contract):
    grid = tuple(contract["input_grid"])
    latent = tuple(contract["latent_grid"])
    hidden, variable = contract["hidden_dim"], contract["variable_conditioning"]
    shapes = _expected_encoder("surface", grid[0], hidden, latent[0], variable)
    if contract["information_grid"] is not None:
        info = tuple(contract["information_grid"])
        shapes.update(_expected_encoder("information", info[0], hidden, latent[0], variable))
        shapes.update({"information_mean": info, "information_scale": info})
        if variable:
            shapes.update({"latent_fusion.weight": (latent[0], 2 * latent[0], 1, 1),
                           "latent_fusion.bias": (latent[0],)})
    shapes.update({"surface_mean": grid, "surface_scale": grid,
                   "latent_mean": latent, "latent_scale": latent})
    return shapes


def _validate(contract, arrays):
    if contract.get("format") != ARTIFACT_FORMAT:
        raise ValueError("Unsupported spatial artifact format")
    for key in ("input_grid", "latent_grid"):
        grid = contract[key]
        if len(grid) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in grid):
            raise ValueError(f"Invalid {key}")
    factor = contract["downsample"]
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("Invalid spatial pooling factor")
    if contract["latent_grid"][1:] != [(n + factor - 1) // factor for n in contract["input_grid"][1:]]:
        raise ValueError("Latent grid does not match ceil pooling")
    if len(contract["surface_fields"]) != contract["input_grid"][0]:
        raise ValueError("Surface variable metadata does not match channels")
    if contract["information_grid"] is not None:
        if (contract["information_grid"][1:] != contract["input_grid"][1:]
                or len(contract["information_fields"]) != contract["information_grid"][0]):
            raise ValueError("Information variable/grid metadata does not match")
    for axis, n in zip(("lat", "lon"), contract["input_grid"][1:]):
        coords = np.asarray(contract["coordinates"][axis])
        if (coords.shape != (n,) or not np.isfinite(coords).all()
                or not (np.all(np.diff(coords) > 0) or np.all(np.diff(coords) < 0))):
            raise ValueError(f"Invalid source {axis} coordinates")
    expected = _expected_shapes(contract)
    if set(expected) != set(arrays):
        raise ValueError(f"Spatial artifact keys differ: {set(expected) ^ set(arrays)}")
    for key, shape in expected.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid shape/values for {key}: expected {shape}")
        if key.endswith("scale") and not (arrays[key] > 0).all():
            raise ValueError(f"Normalization {key} must be strictly positive")


def export_payload(payload: Mapping[str, Any], output, *, checkpoint_sha256=None):
    """Export an already-loaded, verified source downstream checkpoint.

    Use ``scripts/export_spatial.py`` for manifest verification and safe loading.
    Legacy global DCT A checkpoints and raw-only forecasters are rejected. The
    checkpoint's training SHA is preserved separately from architecture provenance.
    """
    if payload.get("format") not in SUPPORTED_CHECKPOINT_FORMATS:
        raise ValueError("Expected a climate_manifold.downstream.v1/v2/v3 checkpoint")
    if payload.get("config", {}).get("bridge") == "raw":
        raise ValueError("Raw checkpoints contain no trained spatial encoder")
    metadata = payload["a_metadata"]
    config = metadata["config"]
    if config.get("representation_kind", "global") != "spatial":
        raise ValueError("Only actual spatial encoders are supported; a global DCT latent is not a spatial grid")
    grid = tuple(config["grid"])
    factor = config["spatial_downsample"]
    channels = config["latent_channels"]
    latent = (channels, *((n + factor - 1) // factor for n in grid[1:]))
    info_meta = metadata.get("information_metadata")
    information_grid = None if info_meta is None else list(info_meta["shape"])
    variables = metadata["schema"]["variables"]
    coords = {k: list(variables[0]["coords"][k]) for k in ("lat", "lon")}
    for variable in variables:
        if tuple(variable["shape"]) != grid[1:]:
            raise ValueError("Source surface variables must share the configured spatial grid")
        if any(not np.array_equal(variable["coords"][axis], coords[axis]) for axis in ("lat", "lon")):
            raise ValueError("Source surface variables have different coordinates")
    if info_meta is not None and info_meta.get("grid") is not None:
        if any(not np.array_equal(info_meta["grid"][axis], coords[axis]) for axis in ("lat", "lon")):
            raise ValueError("Source information coordinates differ from surface coordinates")
    lon = np.deg2rad(coords["lon"])
    periodic = bool(len(lon) >= 3 and np.allclose(np.diff(lon), np.diff(lon)[0])
                    and np.isclose(abs(lon[1] - lon[0]) * len(lon), 2 * math.pi))
    contract = {
        "format": ARTIFACT_FORMAT, "source_repository": SOURCE_REPOSITORY,
        "architecture_source_commit": SOURCE_COMMIT,
        "checkpoint_source_commit": payload.get("source_commit"),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_format": payload["format"],
        "input_grid": list(grid), "latent_grid": list(latent),
        "information_grid": information_grid,
        "hidden_dim": config["spatial_hidden_dim"], "downsample": factor,
        "periodic_lon": periodic, "variable_conditioning": config.get("spatial_variable_conditioning", False),
        "surface_fields": [v["name"] for v in variables],
        "surface_units": [v.get("attrs", {}).get("units") for v in variables],
        "information_fields": [] if info_meta is None else [v["name"] for v in info_meta["variables"]],
        "information_units": [] if info_meta is None else [v.get("unit") for v in info_meta["variables"]],
        "information_kinds": [] if info_meta is None else [v.get("kind") for v in info_meta["variables"]],
        "coordinates": coords,
        "latent_coordinates": {k: [float(np.mean(v[i:i + factor])) for i in range(0, len(v), factor)]
                               for k, v in coords.items()},
        "normalization": "frozen checkpoint surface per-cell and information per-channel statistics",
        "input_layout": "[..., variable, latitude, longitude] in declared physical units",
        "history_contract": "observed surface history with origin information broadcast by caller; no future observations",
        "statistical_loss": payload.get("statistical_loss"),
        "statistical_loss_config": payload.get("statistical_loss_config"),
        "training_split": metadata.get("split"),
    }
    source = payload["model"]
    prefix = "bridge.manifold."
    arrays = {}
    for key, value in source.items():
        if key.startswith(prefix + "core.manifold.encoder."):
            arrays["surface." + key.removeprefix(prefix + "core.manifold.encoder.")] = _array(value)
        elif key.startswith(prefix + "information."):
            arrays["information." + key.removeprefix(prefix + "information.")] = _array(value)
        elif key.startswith(prefix + "latent_fusion."):
            arrays[key.removeprefix(prefix)] = _array(value)
    for key, source_key, shape in (("surface_mean", "mean", grid), ("surface_scale", "scale", grid)):
        arrays[key] = _array(metadata[source_key]).reshape(shape)
    for key in ("latent_mean", "latent_scale"):
        arrays[key] = _array(source[prefix + "core." + key]).reshape(latent)
    if information_grid is not None:
        for key in ("information_mean", "information_scale"):
            arrays[key] = _array(metadata[key]).reshape(information_grid)
    _validate(contract, arrays)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: an exported checkpoint never silently overwrites another.
    with output.open("xb") as stream:
        np.savez_compressed(stream, contract_json=np.asarray(json.dumps(contract, allow_nan=False)), **arrays)
    return contract


@dataclass(frozen=True)
class SpatialEncoder:
    """Frozen exact source encoder; returns actual spatial latents, without noise."""

    contract: Mapping[str, Any]
    weights: Mapping[str, Any]

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as data:
            contract = json.loads(str(data["contract_json"]))
            arrays = {key: np.asarray(data[key]) for key in data.files if key != "contract_json"}
        _validate(contract, arrays)
        return cls(contract, {key: jnp.asarray(value) for key, value in arrays.items()})

    @property
    def surface_fields(self):
        return tuple(self.contract["surface_fields"])

    @property
    def information_fields(self):
        return tuple(self.contract["information_fields"])

    def _conv(self, x, key, geographic=False):
        if geographic:
            x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (1, 1)),
                        mode="wrap" if self.contract["periodic_lon"] else "edge")
            x = jnp.pad(x, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="edge")
        y = jax.lax.conv_general_dilated(x, self.weights[key + ".weight"], (1, 1), "VALID",
                                       dimension_numbers=("NCHW", "OIHW", "NCHW"))
        return y + self.weights[key + ".bias"][None, :, None, None]

    def _single(self, x, prefix):
        x = jax.nn.silu(self._conv(x, prefix + ".features.0.conv", True))
        x = jax.nn.silu(self._conv(x, prefix + ".features.2.conv", True))
        factor = self.contract["downsample"]
        if factor != 1:
            # Exact ceil_mode=True, count_include_pad=False, including odd bins.
            padding = ((0, 0), (0, 0), (0, (-x.shape[-2]) % factor), (0, (-x.shape[-1]) % factor))
            dims = (1, 1, factor, factor)
            sums = jax.lax.reduce_window(x, 0., jax.lax.add, dims, dims, padding)
            counts = jax.lax.reduce_window(jnp.ones((1, 1, *x.shape[-2:]), x.dtype), 0., jax.lax.add,
                                           dims, dims, padding)
            x = sums / counts
        return self._conv(x, prefix + ".channels")

    def _encoder(self, x, prefix):
        if not self.contract["variable_conditioning"]:
            return self._single(x, prefix)
        parts = [self._single(x[:, i:i + 1], f"{prefix}.variable_encoders.{i}") for i in range(x.shape[1])]
        return self._conv(jnp.concatenate(parts, axis=1), prefix + ".fusion")

    def encode_normalized(self, surface, information=None):
        """Encode fields normalized with *this checkpoint's* frozen statistics."""
        surface = jnp.asarray(surface, dtype=jnp.float32)
        grid, latent_grid = tuple(self.contract["input_grid"]), tuple(self.contract["latent_grid"])
        if surface.ndim < 3 or surface.shape[-3:] != grid:
            raise ValueError(f"Surface input must end in {grid} [variable, lat, lon]")
        leading = surface.shape[:-3]
        latent = self._encoder(surface.reshape((-1, *grid)), "surface")
        info_grid = self.contract["information_grid"]
        if info_grid is None:
            if information is not None:
                raise ValueError("Surface-only checkpoint does not accept information")
        else:
            if information is None:
                raise ValueError("Checkpoint requires observed information: " + ", ".join(self.information_fields))
            information = jnp.asarray(information, dtype=jnp.float32)
            if information.shape != (*leading, *info_grid):
                raise ValueError("Information must match surface leading dimensions and checkpoint information grid")
            extra = self._encoder(information.reshape((-1, *info_grid)), "information")
            latent = (self._conv(jnp.concatenate((latent, extra), axis=1), "latent_fusion")
                      if self.contract["variable_conditioning"] else latent + extra)
        latent = latent.reshape((*leading, *latent_grid))
        return (latent - self.weights["latent_mean"]) / self.weights["latent_scale"]

    def encode(self, surface, information=None):
        """Encode physical fields; normalization uses no present/future data fit."""
        surface = jnp.asarray(surface, dtype=jnp.float32)
        expected = tuple(self.contract["input_grid"])
        if surface.ndim < 3 or surface.shape[-3:] != expected:
            raise ValueError(f"Surface physical input must end in {expected}")
        surface = (surface - self.weights["surface_mean"]) / self.weights["surface_scale"]
        if information is not None:
            if self.contract["information_grid"] is None:
                raise ValueError("Surface-only checkpoint does not accept information")
            information = jnp.asarray(information, dtype=jnp.float32)
            if information.shape != (*surface.shape[:-3], *self.contract["information_grid"]):
                raise ValueError("Information physical input has the wrong shape")
            information = (information - self.weights["information_mean"]) / self.weights["information_scale"]
        return self.encode_normalized(surface, information)

    def encode_lonlat(self, surface, information=None):
        """Return [..., channel, lon, lat], still on the checkpoint latent grid."""
        return jnp.swapaxes(self.encode(surface, information), -1, -2)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
