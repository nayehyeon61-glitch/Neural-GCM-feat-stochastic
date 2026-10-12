"""Causal transfer of observed Climate Manifold statistics into forecast memory.

Source surface fields are used only before the forecast origin. During the
forecast the native sigma-temperature channels drive the memory; the source
latent channels are zero. Thus no unavailable surface diagnostic is invented,
and the statistical information persists through the recurrent memory state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np

from ngcm_memory.closure import closure_step, initialize_memory


def history_gap_contract(times, metadata):
    """Resolve the maximum supported hold interval without relaxing old files.

    Explicit ``max_history_gap_hours`` permits irregular observations. A declared
    ``cadence_hours`` is otherwise the maximum; legacy files infer the median
    consecutive interval. Inference cannot distinguish missing samples from a
    genuinely slower archive, so production artifacts should declare the bound.
    """
    for name in ("max_history_gap_hours", "cadence_hours"):
        if metadata.get(name) is not None:
            value = metadata[name]
            if isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Spatial history {name} must be finite and positive")
            return float(value), f"declared_{name}"
    times = np.asarray(times).astype("datetime64[ns]")
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("Cannot infer a spatial-history cadence from one sample; declare max_history_gap_hours")
    intervals = np.diff(times) / np.timedelta64(1, "h")
    if not np.isfinite(intervals).all() or np.any(intervals <= 0):
        raise ValueError("Cannot infer cadence from non-increasing spatial-history times")
    return float(np.median(intervals)), "inferred_median_interval"


def regrid_lonlat(values, source_lon, source_lat, target_lon, target_lat):
    """Periodic longitude / linear latitude interpolation of [...,lon,lat].

    This interpolates dimensionless latent features, not conserved fields.
    Longitude must cover a global periodic grid; latitude is edge-extended at
    the polar caps. No pressure or physical unit conversion is performed.
    """
    lon = np.asarray(source_lon, dtype=float) % 360
    lat = np.asarray(source_lat, dtype=float)
    if lon.ndim != 1 or lat.ndim != 1 or len(lon) < 2 or len(lat) < 2:
        raise ValueError("Latent coordinates need at least two points per axis")
    xi, yi = np.argsort(lon), np.argsort(lat)
    lon, lat = lon[xi], lat[yi]
    if np.any(np.diff(lon) <= 0) or np.any(np.diff(lat) <= 0):
        raise ValueError("Latent coordinates must be unique")
    array = jnp.take(jnp.take(jnp.asarray(values), xi, axis=-2), yi, axis=-1)
    if array.shape[-2:] != (len(lon), len(lat)):
        raise ValueError("Feature grid and coordinates disagree")
    flat = array.reshape((-1, len(lon), len(lat)))
    tx, ty = jnp.asarray(target_lon) % 360, jnp.asarray(target_lat)
    along_lat = jax.vmap(jax.vmap(lambda a: jnp.interp(ty, jnp.asarray(lat), a)))(flat)
    along_lon = jax.vmap(jax.vmap(
        lambda a: jnp.interp(tx, jnp.asarray(lon), a, period=360), in_axes=1, out_axes=1
    ))(along_lat)
    return along_lon.reshape((*array.shape[:-2], len(tx), len(ty)))


def thermal_with_empty_spatial(base_model, spatial_dim, scale_kelvin=10.):
    """Build online features without using any future observed source fields."""
    if spatial_dim < 1 or not np.isfinite(scale_kelvin) or scale_kelvin <= 0:
        raise ValueError("Invalid feature dimension/temperature scale")
    def features(state, forcings):
        del forcings
        prognostics = state.state if hasattr(state, "state") else state
        nodal = base_model.model_coords.horizontal.to_nodal(prognostics.temperature_variation)
        thermal = base_model.from_nondim_units(nodal, "K") / scale_kelvin
        return jnp.concatenate([thermal, jnp.zeros((spatial_dim, *thermal.shape[-2:]), thermal.dtype)])
    return features


@dataclass
class SpatialHistory:
    times: np.ndarray
    latents: np.ndarray
    source_lon: np.ndarray
    source_lat: np.ndarray
    target_lon: np.ndarray
    target_lat: np.ndarray
    metadata: dict
    artifact_sha256: str

    def __post_init__(self):
        maximum, source = history_gap_contract(self.times, self.metadata)
        # Preserve the resolved contract in training/evaluation metadata, even
        # when loading a legacy artifact that did not declare its cadence.
        self.metadata = {**self.metadata, "max_history_gap_hours": maximum,
                         "history_gap_policy": self.metadata.get("history_gap_policy", source)}

    @property
    def max_history_gap_hours(self):
        return self.metadata["max_history_gap_hours"]

    @classmethod
    def from_npz(cls, path, base_model):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if metadata.get("format") != "ngcm_memory.spatial_history.v1":
                raise ValueError("Unsupported observed spatial history artifact")
            times = np.asarray(data["times"]).astype("datetime64[ns]")
            latents = np.asarray(data["latents"], dtype=np.float32)
            lon, lat = np.asarray(data["longitude"]), np.asarray(data["latitude"])
        if (times.ndim != 1 or len(times) < 1 or np.isnat(times).any()
                or np.any(np.diff(times) <= np.timedelta64(0, "ns"))):
            raise ValueError("Spatial history needs strictly increasing finite times")
        if (latents.ndim != 4 or latents.shape[0] != len(times)
                or latents.shape[-2:] != (len(lon), len(lat)) or not np.isfinite(latents).all()):
            raise ValueError("Expected finite latents[time,channel,longitude,latitude]")
        if metadata.get("periodic_lon") is not True:
            raise ValueError("Global NeuralGCM requires a global periodic source spatial grid")
        grid = base_model.model_coords.horizontal
        with open(path, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return cls(times, latents, lon, lat, np.rad2deg(grid.longitudes),
                   np.rad2deg(grid.latitudes), metadata, digest)

    @property
    def feature_dim(self):
        return self.latents.shape[1]

    def validate_context(self, origin_time, history_steps=8):
        """Validate/select temporal context without materializing latent grids.

        Each selected feature drives its following historical interval. The
        final interval ends exactly at origin; no future or origin observation
        is read. Internal gaps and the last-observation age must both respect
        the artifact's maximum hold interval. Shorter/irregular intervals are
        allowed; a missing history never silently becomes a long extrapolation.
        """
        if isinstance(history_steps, bool) or not isinstance(history_steps, int) or history_steps < 1:
            raise ValueError("history_steps must be a positive integer")
        origin = np.datetime64(origin_time, "ns")
        if np.isnat(origin):
            raise ValueError("Invalid forecast origin")
        stop = int(np.searchsorted(self.times, origin, side="left"))
        if stop < history_steps:
            raise ValueError(f"Need {history_steps} pre-origin spatial samples at {origin}")
        indices = slice(stop - history_steps, stop)
        selected_times = self.times[indices]
        ends = np.concatenate([selected_times[1:], np.asarray([origin])])
        durations = (ends - selected_times) / np.timedelta64(1, "h")
        # One nanosecond accommodates conversion roundoff, not missing samples.
        too_long = np.flatnonzero(durations > self.max_history_gap_hours + 1. / 3_600_000_000_000)
        if too_long.size:
            i = int(too_long[0])
            kind = "stale last observation" if i == len(durations) - 1 else "internal gap"
            raise ValueError(
                f"Spatial history {kind} at origin {origin}: {selected_times[i]} -> {ends[i]} "
                f"is {durations[i]:g} h, exceeds max_history_gap_hours={self.max_history_gap_hours:g}; "
                "provide complete history or declare an intentional maximum during preparation"
            )
        return indices, durations

    def filter_origins(self, times, origins, history_steps=8):
        """Use the same temporal validation as context(); report skipped windows."""
        accepted, rejected, first_error = [], 0, None
        for origin in origins:
            try:
                self.validate_context(times[origin], history_steps)
            except ValueError as error:
                rejected += 1
                if first_error is None:
                    first_error = str(error)
            else:
                accepted.append(origin)
        return np.asarray(accepted, dtype=int), {"rejected_windows": rejected, "first_error": first_error}

    def context(self, origin_time, history_steps=8):
        """Regrid a validated, fixed-size, strictly pre-origin history."""
        indices, durations = self.validate_context(origin_time, history_steps)
        features = regrid_lonlat(self.latents[indices], self.source_lon, self.source_lat,
                                self.target_lon, self.target_lat)
        return {"features": features, "dt_hours": jnp.asarray(durations, dtype=jnp.float32)}

    def initializer(self, config):
        """Return ``(params, augmented_state, context) -> augmented_state``.

        Historical spatial statistics initialize deterministic memory. Existing
        base state, OU state, and key remain intact. This explicitly replaces
        thermal-only memory warmup, and the same recipe is used at evaluation.
        """
        if config.feature_dim != config.levels + self.feature_dim:
            raise ValueError("Closure channels must equal sigma levels + spatial latent channels")
        deterministic = replace(config, enable_stochastic=False)
        def seed(params, state, context):
            memory_state = initialize_memory(context["features"].shape[-2:], deterministic,
                                             state.closure_state.key)
            def step(previous, item):
                latent, dt = item
                inputs = jnp.concatenate([jnp.zeros((config.levels, *latent.shape[-2:]), latent.dtype), latent])
                _, current = closure_step(params, previous, inputs, dt, deterministic)
                return current, None
            memory_state, _ = jax.lax.scan(step, memory_state,
                                           (context["features"], context["dt_hours"]))
            return state._replace(closure_state=state.closure_state._replace(memory=memory_state.memory))
        return seed
