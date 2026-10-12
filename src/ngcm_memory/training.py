"""Causal, frozen-base, multi-step training utilities.

Only the closure parameter pytree is differentiated. The decoder loss follows
the corrected physical state through every forecast step, so this is not
offline residual regression. Warm-up refreshes the physical state from past
observations while retaining closure memory; forecast targets are never
encoded. ERA5 forcings persist from initialization by default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ngcm_memory.metrics import area_mean, ensemble_crps


def validate_time_axis(times: Any, model_timestep: Any) -> tuple[float, int]:
    """Return observation spacing in hours and integral internal step count."""
    times = np.asarray(times)
    if times.ndim != 1 or times.size < 2 or times.dtype.kind != "M":
        raise ValueError("time must contain at least two datetime64 observations")
    times = times.astype("datetime64[ns]")
    if np.isnat(times).any():
        raise ValueError("time contains NaT")
    differences = np.diff(times).astype("timedelta64[ns]").astype(np.int64)
    if np.any(differences <= 0) or not np.all(differences == differences[0]):
        raise ValueError("time must be strictly increasing and regularly spaced")
    timestep = np.asarray(model_timestep).astype("timedelta64[ns]").astype(np.int64).item()
    if timestep <= 0 or differences[0] % timestep:
        raise ValueError("Observation interval must be an integer multiple of model timestep")
    return float(differences[0] / 3_600_000_000_000), int(differences[0] // timestep)


def temporal_window_split(
    length: int, warmup_steps: int, horizon_steps: int,
    train_fraction: float = 0.8, stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split observations first, then construct windows inside each partition.

    Returned indices are forecast origins. Validation history, initial states,
    and targets all lie after the split; no window crosses the boundary.
    """
    if warmup_steps < 0 or horizon_steps < 1 or stride < 1:
        raise ValueError("warmup >= 0, horizon >= 1, and stride >= 1 required")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie strictly between 0 and 1")
    cut = int(np.floor(length * train_fraction))
    train = np.arange(warmup_steps, cut - horizon_steps, stride, dtype=int)
    validation = np.arange(cut + warmup_steps, length - horizon_steps, stride, dtype=int)
    if not len(train) or not len(validation):
        raise ValueError("Both temporal partitions need a complete warmup + origin + forecast window")
    return train, validation, cut


def evaluation_origins(
    length: int, warmup_steps: int, horizon_steps: int,
    start_index: int = 0, stride: int = 1,
) -> np.ndarray:
    if warmup_steps < 0 or horizon_steps < 1 or stride < 1 or start_index < 0:
        raise ValueError("Invalid evaluation window bounds")
    origins = np.arange(start_index + warmup_steps, length - horizon_steps, stride, dtype=int)
    if not origins.size:
        raise ValueError("No complete evaluation window fits the requested interval")
    return origins


def filter_spatial_origins(
    history: Any, times: Any, origins: Any, history_steps: int, *,
    recurrent: bool = False, horizon_steps: int = 0, require_future_targets: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate source coverage without regridding every candidate window.

    Recurrent forecasts need an observed latent exactly at the origin. Future
    latents are needed only for supervised training/validation, never for a
    forecast or evaluation. Reject rather than interpolate absent labels.
    """
    if require_future_targets and (not recurrent or horizon_steps < 1):
        raise ValueError("Future latent labels require recurrent mode and a positive horizon")
    accepted, rejected, first_error = [], 0, None
    times = np.asarray(times).astype("datetime64[ns]")
    source_times = np.asarray(history.times).astype("datetime64[ns]")
    for origin in origins:
        try:
            history.validate_context(times[origin], history_steps)
            if recurrent:
                count = horizon_steps + 1 if require_future_targets else 1
                required = times[origin:origin + count]
                if len(required) != count:
                    raise ValueError("Source target window extends beyond available observations")
                indices = np.searchsorted(source_times, required)
                if np.any(indices >= len(source_times)) or not np.array_equal(source_times[indices], required):
                    purpose = "origin and future labels" if require_future_targets else "forecast origin"
                    raise ValueError(f"Need exact spatial latents at {purpose} timestamps for origin {times[origin]}")
        except ValueError as error:
            rejected += 1
            if first_error is None:
                first_error = str(error)
        else:
            accepted.append(origin)
    return np.asarray(accepted, dtype=int), {"rejected_windows": rejected, "first_error": first_error}


def independent_test_start_index(times: Any, metadata: dict[str, Any], start_time: str | None = None) -> int:
    """Keep reported test targets and their warmup after model selection data."""
    if "validation_last_observation" not in metadata:
        raise ValueError("Checkpoint lacks validation_last_observation; cannot establish an independent test cutoff")
    last_used = max(
        np.datetime64(metadata["training_last_observation"], "ns"),
        np.datetime64(metadata["validation_last_observation"], "ns"),
    )
    index = int(np.searchsorted(times, last_used, side="right"))
    if start_time:
        index = max(index, int(np.searchsorted(times, np.datetime64(start_time, "ns"))))
    return index


def slice_window(tree: Any, origin: int, warmup_steps: int, horizon_steps: int) -> Any:
    return jax.tree_util.tree_map(
        lambda value: jnp.asarray(value[origin - warmup_steps:origin + horizon_steps + 1]), tree
    )


def _at(tree: Any, index: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: value[index], tree)


def _timed_forcings(forcings: dict[str, Any], sim_time: Any) -> dict[str, Any]:
    return {**forcings, "sim_time": sim_time}


def _advance_interval(
    adapter: Any, params: Any, state: Any, forcings: Any,
    sim_start: Any, sim_delta: Any, internal_steps: int,
    *, warmup: bool = False,
) -> Any:
    advance_state = getattr(adapter, "advance_warmup", adapter.advance) if warmup else adapter.advance
    def advance(carry: Any, step: Any) -> tuple[Any, None]:
        timed = _timed_forcings(forcings, sim_start + sim_delta * step / internal_steps)
        return advance_state(params, carry, timed), None
    return jax.lax.scan(advance, state, jnp.arange(internal_steps))[0]


def rollout_ensemble(
    params: Any, adapter: Any, inputs: dict[str, Any], forcings: dict[str, Any],
    keys: Any, *, warmup_steps: int, horizon_steps: int, internal_steps: int,
    variables: tuple[str, ...] = ("temperature",), forcing_mode: str = "persistence",
    memory_initializer: Callable[[Any, Any, Any], Any] | None = None,
    memory_context: Any = None,
    return_latents: bool = False,
) -> Any:
    """Roll out [member,lead,...] fields with causal observed-history warmup.

    inputs/forcings include warmup + origin + horizon observations. Only input
    frames up to the origin are encoded. `observed` forcing is an explicitly
    conditioned hindcast using future supplied boundary fields; persistence
    uses only the origin's boundary values. The scalar sim_time always advances.

    memory_initializer optionally replaces/seeds the warmed memory from a
    caller-prepared, causal context. It must be a JAX-transformable function.
    When return_latents=True, return (fields, latent_trajectories), with latent
    trajectories sampled only at observation leads. Labels are stripped before
    passing context to an initializer and never enter the autoregressive state.
    """
    if forcing_mode not in {"persistence", "observed"}:
        raise ValueError("forcing_mode must be persistence or observed")
    if warmup_steps < 0 or horizon_steps < 1 or internal_steps < 1:
        raise ValueError("Invalid rollout lengths")
    required = warmup_steps + horizon_steps + 1
    if inputs["sim_time"].shape[0] != required:
        raise ValueError("Window length does not match warmup + origin + horizon")

    causal_context = ({name: value for name, value in memory_context.items() if name != "latent_targets"}
                      if isinstance(memory_context, dict) else memory_context)

    def member(key: Any) -> Any:
        initial_key, refresh_key = jax.random.split(key)
        state = adapter.encode(_at(inputs, 0), _at(forcings, 0), initial_key)

        def warm(carry: Any, index: Any) -> tuple[Any, None]:
            observed = adapter.base_model.encode(
                _at(inputs, index), _at(forcings, index),
                rng_key=jax.random.fold_in(refresh_key, index),
            )
            carry = carry._replace(model_state=observed)
            sim_start = inputs["sim_time"][index]
            sim_delta = inputs["sim_time"][index + 1] - sim_start
            carry = _advance_interval(
                adapter, params, carry, _at(forcings, index),
                sim_start, sim_delta, internal_steps, warmup=True,
            )
            return carry, None

        if warmup_steps:
            state = jax.lax.scan(warm, state, jnp.arange(warmup_steps))[0]
            observed_origin = adapter.base_model.encode(
                _at(inputs, warmup_steps), _at(forcings, warmup_steps),
                rng_key=jax.random.fold_in(refresh_key, warmup_steps),
            )
            state = state._replace(model_state=observed_origin)
        if memory_initializer is not None:
            state = memory_initializer(params, state, causal_context)
        if return_latents and getattr(state, "latent_state", None) is None:
            raise ValueError("return_latents requires an adapter with a recurrent latent state")

        def forecast(carry: Any, lead: Any) -> tuple[Any, dict[str, Any]]:
            index = warmup_steps + lead
            forcing_index = index if forcing_mode == "observed" else warmup_steps
            boundary = _at(forcings, forcing_index)
            sim_start = inputs["sim_time"][index]
            sim_end = inputs["sim_time"][index + 1]
            carry = _advance_interval(
                adapter, params, carry, boundary, sim_start,
                sim_end - sim_start, internal_steps,
            )
            decode_boundary = _at(forcings, index + 1) if forcing_mode == "observed" else boundary
            decoded = adapter.decode(carry, _timed_forcings(decode_boundary, sim_end))
            fields = {variable: decoded[variable] for variable in variables}
            if return_latents:
                return carry, (fields, carry.latent_state.latent)
            return carry, fields

        return jax.lax.scan(forecast, state, jnp.arange(horizon_steps))[1]

    return jax.vmap(member)(keys)


def forecast_loss(
    predictions: dict[str, Any], targets: dict[str, Any], area_weights: Any,
    scales: dict[str, float], mean_mse_weight: float = 0.1,
) -> Any:
    """Normalized fair CRPS plus an unbiased estimate of predictive-mean MSE.

    Scales are fixed user-supplied physical-unit scales, never fit on validation
    data. Multi-step targets are scored equally. At least two ensemble members
    are required. The finite-ensemble variance/n contribution is subtracted
    from mean MSE, so that auxiliary term does not favor underdispersion either.
    """
    losses = []
    for name, prediction in predictions.items():
        scale = scales[name]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Scale for {name} must be positive")
        normalized, truth = prediction / scale, targets[name] / scale
        crps = jnp.mean(ensemble_crps(normalized, truth, area_weights, fair=True))
        mean_squared_error = (normalized.mean(axis=0) - truth)**2
        sampling_variance = jnp.var(normalized, axis=0, ddof=1) / normalized.shape[0]
        mse = jnp.mean(area_mean(mean_squared_error - sampling_variance, area_weights))
        losses.append(crps + mean_mse_weight * mse)
    if not losses:
        raise ValueError("At least one forecast variable is required")
    return jnp.mean(jnp.stack(losses))


def make_train_step(
    adapter: Any, optimizer: optax.GradientTransformation, *, warmup_steps: int,
    horizon_steps: int, internal_steps: int, variables: tuple[str, ...],
    scales: dict[str, float], area_weights: Any, forcing_mode: str = "persistence",
    mean_mse_weight: float = 0.1, memory_initializer: Callable | None = None,
    latent_loss_weight: float = 0., latent_area_weights: Any = None,
) -> Callable:
    """JIT one window update; gradients are only with respect to params."""
    if not np.isfinite(latent_loss_weight) or latent_loss_weight < 0:
        raise ValueError("latent_loss_weight must be finite and nonnegative")
    supervise_latents = latent_loss_weight > 0

    def objective(params: Any, inputs: Any, forcings: Any, keys: Any, context: Any) -> Any:
        output = rollout_ensemble(
            params, adapter, inputs, forcings, keys, warmup_steps=warmup_steps,
            horizon_steps=horizon_steps, internal_steps=internal_steps,
            variables=variables, forcing_mode=forcing_mode,
            memory_initializer=memory_initializer, memory_context=context,
            return_latents=supervise_latents,
        )
        predictions, latents = output if supervise_latents else (output, None)
        targets = {name: inputs[name][warmup_steps + 1:] for name in variables}
        if supervise_latents and (not isinstance(context, dict) or "latent_targets" not in context):
            raise ValueError("Positive latent_loss_weight requires context['latent_targets']")
        total, _, _ = joint_forecast_loss(
            predictions, targets, area_weights, scales, mean_mse_weight,
            latent_predictions=latents,
            latent_targets=context["latent_targets"] if supervise_latents else None,
            latent_loss_weight=latent_loss_weight, latent_area_weights=latent_area_weights,
        )
        return total

    @jax.jit
    def step(params: Any, opt_state: Any, inputs: Any, forcings: Any, keys: Any, context: Any = None) -> Any:
        loss, gradients = jax.value_and_grad(objective)(params, inputs, forcings, keys, context)
        grad_norm = optax.global_norm(gradients)
        updates, next_opt_state = optimizer.update(gradients, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, loss, grad_norm
    return step


def joint_forecast_loss(
    predictions: dict[str, Any], targets: dict[str, Any], area_weights: Any,
    scales: dict[str, float], mean_mse_weight: float = .1, *,
    latent_predictions: Any = None, latent_targets: Any = None,
    latent_loss_weight: float = 0., latent_area_weights: Any = None,
) -> tuple[Any, Any, Any]:
    """Return total, physical, latent fair-score objectives.

    The source encoder's frozen normalized latents are supervised at actual
    observation leads, not interpolated hourly pseudo-targets. Latent and
    decoded physical grids may differ, so each has its own area quadrature.
    """
    if not np.isfinite(latent_loss_weight) or latent_loss_weight < 0:
        raise ValueError("latent_loss_weight must be finite and nonnegative")
    physical = forecast_loss(predictions, targets, area_weights, scales, mean_mse_weight)
    latent = jnp.zeros_like(physical)
    if latent_loss_weight:
        if latent_predictions is None or latent_targets is None or latent_area_weights is None:
            raise ValueError("Latent supervision requires predictions, targets and latent area weights")
        if latent_predictions.shape[1:] != latent_targets.shape:
            raise ValueError("Latent labels must match forecast lead, channel and grid dimensions exactly")
        latent = forecast_loss(
            {"latent": latent_predictions}, {"latent": latent_targets},
            latent_area_weights, {"latent": 1.}, mean_mse_weight,
        )
    return physical + latent_loss_weight * latent, physical, latent


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_closure_checkpoint(
    path: str | Path, params: Any, config: Any, metadata: dict[str, Any],
) -> None:
    """Safe array-only NPZ checkpoint; no pickle or executable object arrays."""
    leaves = jax.tree_util.tree_leaves(params)
    arrays = {f"param_{i:05d}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    if any(value.dtype.hasobject or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Closure parameters must be finite numeric arrays")
    arrays.update(
        format_version=np.asarray(1, dtype=np.int32),
        config=np.asarray(json.dumps(dataclasses.asdict(config), sort_keys=True)),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A file handle avoids numpy silently appending .npz to another suffix.
    with destination.open("wb") as output:
        np.savez_compressed(output, **arrays)


def load_closure_checkpoint(path: str | Path) -> tuple[Any, Any, dict[str, Any]]:
    """Restore against the current parameter schema; reject shape mismatches."""
    from ngcm_memory.closure import ClosureConfig, init_params

    with np.load(path, allow_pickle=False) as checkpoint:
        if checkpoint["format_version"].item() != 1:
            raise ValueError("Unsupported closure checkpoint format")
        config_dict = json.loads(checkpoint["config"].item())
        for name in ("tau_hours", "latent_memory_tau_hours"):
            if name in config_dict:
                config_dict[name] = tuple(config_dict[name])
        config = ClosureConfig(**config_dict)
        metadata = json.loads(checkpoint["metadata"].item())
        template = init_params(jax.random.PRNGKey(0), config)
        expected, structure = jax.tree_util.tree_flatten(template)
        expected_keys = {"format_version", "config", "metadata"} | {f"param_{i:05d}" for i in range(len(expected))}
        if set(checkpoint.files) != expected_keys:
            raise ValueError("Closure checkpoint parameter schema differs from this version")
        restored = []
        for index, reference in enumerate(expected):
            value = checkpoint[f"param_{index:05d}"]
            if value.shape != reference.shape or value.dtype.hasobject or not np.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint parameter {index}")
            restored.append(jnp.asarray(value, dtype=reference.dtype))
    return jax.tree_util.tree_unflatten(structure, restored), config, metadata


def load_local_problem(base_checkpoint: str | Path, data_path: str | Path, *, trust_checkpoint: bool) -> tuple[Any, Any, Any, Any]:
    """Load a user-trusted upstream pickle and already regridded ERA5 NetCDF.

    NeuralGCM's checkpoint format uses pickle. This function deliberately
    requires explicit trust and never downloads or deserializes remote input.
    Its strict xarray conversion verifies checkpoint variables/coordinates.
    """
    if not trust_checkpoint:
        raise ValueError("Upstream pickle requires explicit --trust-base-checkpoint")
    import pickle
    import neuralgcm
    import xarray as xr

    with Path(base_checkpoint).open("rb") as source:
        base = neuralgcm.PressureLevelModel.from_checkpoint(pickle.load(source))
    with xr.open_dataset(data_path) as source:
        dataset = source.load()
    if "time" not in dataset.coords:
        raise ValueError("Dataset requires a time coordinate")
    validate_time_axis(dataset.time.values, base.timestep)
    # Coordinate matching alone cannot distinguish geopotential (m²/s²)
    # from geopotential height (m), or Celsius from absolute Kelvin.
    accepted_units = {
        "temperature": {"k", "kelvin"},
        "sea_surface_temperature": {"k", "kelvin"},
        "u_component_of_wind": {"m/s", "ms^-1", "ms-1"},
        "v_component_of_wind": {"m/s", "ms^-1", "ms-1"},
        "geopotential": {"m^2s^-2", "m2s-2", "m^2/s^2", "m2/s2"},
        "specific_humidity": {"kg/kg", "kgkg^-1", "kgkg-1", "1"},
        "specific_cloud_ice_water_content": {"kg/kg", "kgkg^-1", "kgkg-1", "1"},
        "specific_cloud_liquid_water_content": {"kg/kg", "kgkg^-1", "kgkg-1", "1"},
        "sea_ice_cover": {"1", "(0-1)", "0-1", "dimensionless"},
    }
    for name, allowed in accepted_units.items():
        if name in dataset:
            unit = str(dataset[name].attrs.get("units", "")).lower().replace(" ", "").replace("**", "^")
            if unit not in allowed:
                raise ValueError(f"{name}: units must be explicitly declared in NeuralGCM physical units; got {unit!r}, accepted {sorted(allowed)}")
    inputs = base.inputs_from_xarray(dataset)
    forcings = base.forcings_from_xarray(dataset)
    for name, value in {**inputs, **forcings}.items():
        if np.asarray(value).shape[0] != dataset.sizes["time"]:
            raise ValueError(f"{name} must include the leading time axis")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values; prepare/fill ERA5 first")
    return base, dataset, inputs, forcings
