#!/usr/bin/env python3
"""Generate frozen, causal NeuralGCM + recurrent-manifold guides for KAI.

One NPZ per origin. Only origin/past observations initialize the generator;
targets never enter its recurrence. Each physical interval is advanced once
per member, then that same guide is available to every CNN stage. Both source
checkpoints must already be trained. No synthetic weights are substituted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import numpy as np
import xarray as xr

from ngcm_memory.features import SpatialHistory
from ngcm_memory.neuralgcm_adapter import NeuralGCMMemoryAdapter
from ngcm_memory.stage_guides import NeuralGCMStageGuide
from ngcm_memory.training import load_closure_checkpoint, load_local_problem, sha256_file, validate_time_axis


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--trust-base-checkpoint", action="store_true")
    p.add_argument("--closure", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="Prepared NeuralGCM NetCDF on checkpoint grid")
    p.add_argument("--spatial-history", type=Path, required=True)
    p.add_argument("--kai-data", type=Path, required=True, help="KAI fields(time,channel,lat,lon), in physical units")
    p.add_argument("--global-means", type=Path, required=True)
    p.add_argument("--global-stds", type=Path, required=True)
    p.add_argument("--temperature-channels", required=True, help="Explicit KAI channel indices, e.g. 10,11; never inferred")
    p.add_argument("--pressure-levels", required=True, help="Matching temperature levels in hPa, e.g. 500,850")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--horizon-steps", type=int, default=4)
    p.add_argument("--interval-hours", type=float, default=24.)
    p.add_argument("--ensemble-size", type=int, default=4)
    p.add_argument("--start-time", help="Earliest origin; actual timestamp must exist in both inputs")
    p.add_argument("--end-time", help="Exclusive origin bound")
    p.add_argument("--stride", type=int, default=1, help="Stride through available KAI origin timestamps")
    p.add_argument("--max-origins", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--without-targets", action="store_true", help="Forecast-only archives need KAI observations only at origin")
    return p


def _canonical_kai(ds):
    rename = {old: new for old, new in (("latitude", "lat"), ("longitude", "lon")) if old in ds.dims}
    ds = ds.rename(rename)
    if "fields" not in ds or set(ds.fields.dims) != {"time", "channel", "lat", "lon"}:
        raise ValueError("KAI data requires fields(time,channel,lat,lon) with explicit coordinates")
    if any(name not in ds.coords for name in ("time", "channel", "lat", "lon")):
        raise ValueError("KAI data requires time, channel, latitude and longitude coordinates")
    ds["fields"] = ds.fields.transpose("time", "channel", "lat", "lon")
    if ds.time.dtype.kind != "M":
        raise ValueError("KAI time must be decoded datetime64 coordinates")
    times = ds.time.values.astype("datetime64[ns]")
    if np.isnat(times).any() or len(times) < 1 or np.any(np.diff(times) <= np.timedelta64(0, "ns")):
        raise ValueError("KAI time must be finite and strictly increasing")
    if len(set(map(str, ds.channel.values))) != ds.sizes["channel"]:
        raise ValueError("KAI channel names must be unique")
    return ds


def _at(tree, index):
    return jax.tree.map(lambda value: value[index], tree)


def _sim_time(state):
    native = state.model_state
    return (native.state if hasattr(native, "state") else native).sim_time


def _origin_state(adapter, params, inputs, forcings, origin, warmup, internal_steps, key, initializer, context):
    """Same observed thermal warmup and separate origin-z seed as training."""
    if type(warmup) is not int or warmup < 0 or origin < warmup:
        raise ValueError("warmup must be an integer >= 0 with sufficient pre-origin observations")
    if type(internal_steps) is not int or internal_steps < 1:
        raise ValueError("internal_steps must be a positive integer")
    initial_key, refresh_key = jax.random.split(key)
    start = origin - warmup
    state = adapter.encode(_at(inputs, start), _at(forcings, start), initial_key)
    for index in range(start, origin):
        observed = adapter.base_model.encode(_at(inputs, index), _at(forcings, index),
                                            rng_key=jax.random.fold_in(refresh_key, index))
        state = state._replace(model_state=observed)
        boundary = _at(forcings, index)
        for _ in range(internal_steps):
            state = adapter.advance_warmup(params, state, {**boundary, "sim_time": _sim_time(state)})
    if warmup:
        observed = adapter.base_model.encode(_at(inputs, origin), _at(forcings, origin),
                                            rng_key=jax.random.fold_in(refresh_key, origin))
        state = state._replace(model_state=observed)
    return initializer(params, state, context)


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.horizon_steps < 1 or a.ensemble_size < 1 or a.stride < 1 or (a.max_origins is not None and a.max_origins < 1):
        p.error("horizon, ensemble size, stride, max-origins must be positive")
    if not np.isfinite(a.interval_hours) or a.interval_hours <= 0:
        p.error("interval-hours must be finite and positive")
    bounds = {}
    for name in ("start_time", "end_time"):
        if getattr(a, name) is not None:
            try:
                bounds[name] = np.datetime64(getattr(a, name), "ns")
            except ValueError:
                p.error(f"{name} must be a valid datetime")
            if np.isnat(bounds[name]):
                p.error(f"{name} must not be NaT")
    if len(bounds) == 2 and bounds["end_time"] <= bounds["start_time"]:
        p.error("end-time must be later than start-time")
    params, config, trained = load_closure_checkpoint(a.closure)
    if config.latent_dim < 1 or trained.get("spatial_mode") != "recurrent":
        p.error("A trained recurrent-latent closure checkpoint is required")
    for name, minimum in (("warmup_steps", 0), ("spatial_history_steps", 1)):
        value = trained.get(name)
        if type(value) is not int or value < minimum:
            p.error(f"Closure metadata {name} must be an integer >= {minimum}")
    if sha256_file(a.base_checkpoint) != trained.get("base_checkpoint_sha256"):
        p.error("Base checkpoint must match closure training")
    fit_dates = [np.datetime64(trained[name], "ns") for name in
                 ("training_last_observation", "validation_last_observation")]
    if np.isnat(fit_dates).any():
        p.error("Closure checkpoint requires finite training/validation cutoff dates")
    base, native_ds, inputs, forcings = load_local_problem(a.base_checkpoint, a.data,
                                                         trust_checkpoint=a.trust_base_checkpoint)
    hours, internal_steps = validate_time_axis(native_ds.time.values, base.timestep)
    if not np.isclose(hours, trained["observation_interval_hours"]):
        p.error("Native data interval must match the trained thermal warmup recipe")
    history = SpatialHistory.from_npz(a.spatial_history, base)
    source_metadata = trained.get("spatial_history") or {}
    if not history.metadata.get("encoder_sha256") or history.metadata["encoder_sha256"] != source_metadata.get("encoder_sha256"):
        p.error("Spatial history must come from the same encoder as closure training")
    if history.feature_dim != config.latent_dim:
        p.error("Spatial history channels differ from recurrent checkpoint")
    if source_metadata.get("max_history_gap_hours") is not None and not np.isclose(
            history.max_history_gap_hours, source_metadata["max_history_gap_hours"], rtol=0, atol=1e-9):
        p.error("Spatial history gap contract differs from closure training")
    adapter = NeuralGCMMemoryAdapter(base, config,
                                    feature_scale_kelvin=trained["feature_scale_kelvin"],
                                    remove_global_mean=trained["remove_global_mean"])
    initializer = history.transition_initializer(config)
    warmup = int(trained["warmup_steps"])
    history_steps = int(trained["spatial_history_steps"])
    temp_channels = np.array([int(x) for x in a.temperature_channels.split(",")])
    pressure_levels = np.array([float(x) for x in a.pressure_levels.split(",")])
    means = np.load(a.global_means, allow_pickle=False).squeeze().reshape(-1)
    stds = np.load(a.global_stds, allow_pickle=False).squeeze().reshape(-1)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with xr.open_dataset(a.kai_data) as source:
        kai = _canonical_kai(source)
        c = kai.sizes["channel"]
        if means.shape != (c,) or stds.shape != (c,) or not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
            p.error("KAI means/stds require one finite entry per channel, stds > 0")
        if temp_channels.size != pressure_levels.size or temp_channels.size == 0 or len(set(temp_channels)) != len(temp_channels) or np.any((temp_channels < 0) | (temp_channels >= c)):
            p.error("Explicit temperature-channel/pressure-level mapping is invalid")
        guide = NeuralGCMStageGuide(adapter, params, target_longitude=kai.lon.values,
                                    target_latitude=kai.lat.values, pressure_levels=pressure_levels,
                                    temperature_means=means[temp_channels], temperature_stds=stds[temp_channels])
        native_times = native_ds.time.values.astype("datetime64[ns]")
        kai_times = kai.time.values.astype("datetime64[ns]")
        lead_hours = np.arange(1, a.horizon_steps + 1) * a.interval_hours
        lead_ns = np.rint(lead_hours * 3_600_000_000_000).astype("timedelta64[ns]")
        for k in range(0, len(kai_times), a.stride):
            origin_time = kai_times[k]
            if "start_time" in bounds and origin_time < bounds["start_time"]:
                continue
            if "end_time" in bounds and origin_time >= bounds["end_time"]:
                continue
            native_index = int(np.searchsorted(native_times, origin_time))
            if native_index >= len(native_times) or native_times[native_index] != origin_time or native_index < warmup:
                continue
            valid_times = origin_time + lead_ns
            target_indices = np.searchsorted(kai_times, valid_times)
            if not a.without_targets and (np.any(target_indices >= len(kai_times)) or not np.array_equal(kai_times[target_indices], valid_times)):
                continue
            try:
                context = history.transition_context(origin_time, history_steps)
            except ValueError as error:
                print(f"Skipping origin {origin_time}: {error}", file=sys.stderr, flush=True)
                continue
            origin_stamp = int(origin_time.astype("datetime64[s]").astype(np.int64))
            keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(a.seed), origin_stamp & 0xffffffff), a.ensemble_size)
            physical, latent = [], []
            for key in keys:
                state = _origin_state(adapter, params, inputs, forcings, native_index, warmup,
                                      internal_steps, key, initializer, context)
                boundary = _at(forcings, native_index)
                member_p, member_z = [], []
                for lead in lead_hours:
                    state, frame = guide.advance(state, boundary, interval_hours=a.interval_hours, lead_hours=float(lead))
                    member_p.append(frame.physical)
                    member_z.append(frame.latent)
                physical.append(member_p)
                latent.append(member_z)
            initial = (kai.fields.isel(time=k).values - means[:, None, None]) / stds[:, None, None]
            metadata = dict(format="ngcm_memory.kai_guides.v1", normalized=True,
                            origin_time=str(origin_time), valid_times=list(map(str, valid_times)),
                            channels=list(map(str, kai.channel.values)), pressure_levels=pressure_levels.tolist(),
                            temperature_channels=temp_channels.tolist(), means=means.tolist(), stds=stds.tolist(),
                            base_checkpoint_sha256=trained["base_checkpoint_sha256"],
                            closure_checkpoint_sha256=sha256_file(a.closure), guide_fit_end=str(max(fit_dates)),
                            source_encoder_sha256=history.metadata["encoder_sha256"],
                            forcing_mode="persistence", seed=a.seed, physical_guide_units="normalized_kelvin",
                            latent_guide_units="dimensionless", future_observations_used=False,
                            regridding="periodic_linear_longitude_linear_latitude_not_conservative",
                            spatial_history_sha256=history.artifact_sha256,
                            warmup_steps=warmup, spatial_history_steps=history_steps)
            arrays = dict(initial=initial.astype(np.float32), physical_guides=np.asarray(physical, dtype=np.float32),
                          latent_guides=np.asarray(latent, dtype=np.float32), latitude=kai.lat.values,
                          longitude=kai.lon.values, lead_hours=lead_hours,
                          metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))
            if not a.without_targets:
                targets = kai.fields.isel(time=target_indices).values
                arrays["targets"] = ((targets - means[None, :, None, None]) / stds[None, :, None, None]).astype(np.float32)
            for name in ("initial", "targets", "physical_guides", "latent_guides"):
                if name in arrays and not np.isfinite(arrays[name]).all():
                    raise ValueError(f"Nonfinite {name} at {origin_time}")
            filename = str(origin_time.astype("datetime64[s]")).replace(":", "-") + ".npz"
            destination = a.output_dir / filename
            # Refuse accidental overwrite, including a matching input filename.
            with destination.open("xb") as output:
                np.savez_compressed(output, **arrays)
            count += 1
            print(json.dumps({"origin": str(origin_time), "saved": str(destination), "members": a.ensemble_size}), flush=True)
            if a.max_origins is not None and count >= a.max_origins:
                break
    if count == 0:
        raise ValueError("No complete aligned origins with causal history; check timestamps and coverage")


if __name__ == "__main__":
    main()
