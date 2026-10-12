#!/usr/bin/env python3
"""Encode observed source-grid fields for causal NeuralGCM memory initialization.

The NetCDF must already match the Climate Manifold checkpoint grid, field names
and physical units. This script never substitutes NeuralGCM model-level fields
for surface observations. Output features retain the source latent grid. The
rollout initializer must select only observations at or before its forecast
origin; ``--through`` can additionally enforce a preparation cutoff.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import jax
import numpy as np
import xarray as xr

from ngcm_memory.spatial import SpatialEncoder, file_sha256
from ngcm_memory.features import history_gap_contract


def _canonical_unit(unit):
    if unit is None:
        raise ValueError("Every source and checkpoint field must declare physical units")
    compact = str(unit).lower().replace(" ", "").replace("²", "2").replace("⁻", "-").replace("¹", "1")
    aliases = {"pa": "Pa", "pascal": "Pa", "pascals": "Pa", "k": "K", "kelvin": "K",
               "m": "m", "meter": "m", "metre": "m", "meters": "m", "metres": "m",
               "m/s": "m/s", "ms-1": "m/s", "ms^-1": "m/s", "ms**-1": "m/s",
               "pa/s": "Pa/s", "pas-1": "Pa/s", "pas^-1": "Pa/s", "pas**-1": "Pa/s",
               "kg/kg": "1", "kgkg-1": "1", "kgkg^-1": "1", "kgkg**-1": "1",
               "m2s-2": "m2/s2", "m**2s**-2": "m2/s2", "m^2/s^2": "m2/s2", "m2/s2": "m2/s2",
               "1": "1", "dimensionless": "1", "m/m": "1"}
    if compact not in aliases:
        raise ValueError(f"Unsupported physical unit {unit!r}; convert fields explicitly to checkpoint units first")
    return aliases[compact]


def prepare(encoder_path, data_path, output, *, batch_size=16, latitude_name="lat", longitude_name="lon",
            time_name="time", through=None, max_history_gap_hours=None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    encoder_path, data_path, output = Path(encoder_path), Path(data_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    encoder = SpatialEncoder.from_npz(encoder_path)
    contract = encoder.contract
    output.parent.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(data_path) as ds:
        for axis, name in (("lat", latitude_name), ("lon", longitude_name)):
            if name not in ds.coords or ds[name].dims != (name,):
                raise ValueError(f"Expected 1-D coordinate {name}")
            coords = np.asarray(ds[name].values)
            expected = np.asarray(contract["coordinates"][axis])
            if coords.shape != expected.shape or not np.allclose(coords, expected, rtol=0, atol=1e-6):
                raise ValueError(f"Source {axis} grid/order differs from checkpoint; regrid explicitly first")
        if time_name not in ds.coords or ds[time_name].dims != (time_name,):
            raise ValueError(f"Expected 1-D coordinate {time_name}")
        raw_times = np.asarray(ds[time_name].values)
        if not np.issubdtype(raw_times.dtype, np.datetime64):
            raise ValueError("Observed times must be decoded Gregorian datetime64 UTC values")
        times = raw_times.astype("datetime64[ns]")
        if not len(times) or np.isnat(times).any() or not np.all(np.diff(times) > np.timedelta64(0, "ns")):
            raise ValueError("Observed times must be nonempty, finite, unique and increasing")
        count = len(times)
        if through is not None:
            cutoff = np.datetime64(through, "ns")
            if np.isnat(cutoff):
                raise ValueError("Invalid observation cutoff")
            count = int(np.searchsorted(times, cutoff, side="right"))
            if count == 0:
                raise ValueError("No observations exist at or before the cutoff")
            times = times[:count]
        maximum_gap, gap_policy = history_gap_contract(times, {"max_history_gap_hours": max_history_gap_hours})
        fields = []
        for i, name in enumerate(contract["surface_fields"]):
            fields.append((name, contract["surface_units"][i], False))
        kinds = contract.get("information_kinds", [None] * len(contract["information_fields"]))
        for i, name in enumerate(contract["information_fields"]):
            fields.append((name, contract["information_units"][i], kinds[i] == "static"))
        for name, unit, static in fields:
            if name not in ds:
                raise ValueError(f"Missing required observed field {name!r}; no physical-variable substitutions are permitted")
            if _canonical_unit(ds[name].attrs.get("units")) != _canonical_unit(unit):
                raise ValueError(f"Field {name!r} must be in checkpoint units {unit!r}; convert explicitly first")
            wanted = {time_name, latitude_name, longitude_name}
            if static and set(ds[name].dims) == {latitude_name, longitude_name}:
                continue
            if set(ds[name].dims) != wanted:
                raise ValueError(f"Field {name!r} must have time/lat/lon axes (static information may omit time)")

        def stack(names, start, stop):
            arrays = []
            for name in names:
                field = ds[name]
                if time_name in field.dims:
                    array = field.isel({time_name: slice(start, stop)}).transpose(time_name, latitude_name, longitude_name).values
                else:
                    array = np.broadcast_to(field.transpose(latitude_name, longitude_name).values,
                                            (stop - start, *contract["input_grid"][1:]))
                arrays.append(np.asarray(array, dtype=np.float32))
            result = np.stack(arrays, axis=1)
            if not np.isfinite(result).all():
                raise ValueError(f"Missing/nonfinite observed fields in rows {start}:{stop}; no imputation")
            return result

        latent_c, latent_lat, latent_lon = contract["latent_grid"]
        # A disk-backed buffer avoids accumulating full high-resolution history.
        with tempfile.TemporaryDirectory(prefix="spatial-history-", dir=output.parent) as temporary:
            features = np.lib.format.open_memmap(Path(temporary) / "features.npy", mode="w+", dtype="float32",
                                                shape=(count, latent_c, latent_lon, latent_lat))
            encode = jax.jit(encoder.encode_lonlat)
            for start in range(0, count, batch_size):
                stop = min(count, start + batch_size)
                surface = stack(contract["surface_fields"], start, stop)
                info = stack(contract["information_fields"], start, stop) if contract["information_fields"] else None
                features[start:stop] = np.asarray(encode(surface, info))
            features.flush()
            metadata = {"format": "ngcm_memory.spatial_history.v1", "encoder_sha256": file_sha256(encoder_path),
                        "encoder_checkpoint_sha256": contract["checkpoint_sha256"],
                        "architecture_source_commit": contract["architecture_source_commit"],
                        "checkpoint_source_commit": contract["checkpoint_source_commit"],
                        "training_split": contract.get("training_split"),
                        "statistical_loss": contract.get("statistical_loss"),
                        "statistical_loss_config": contract.get("statistical_loss_config"),
                        "normalization": contract["normalization"],
                        "data_sha256": file_sha256(data_path), "surface_fields": contract["surface_fields"],
                        "information_fields": contract["information_fields"], "layout": "time,feature,longitude,latitude",
                        "through": None if through is None else str(times[-1]),
                        "max_history_gap_hours": maximum_gap, "history_gap_policy": gap_policy,
                        "periodic_lon": contract["periodic_lon"], "feature_count": latent_c,
                        "usage": "observed history only; initializer must exclude all times after forecast origin"}
            with output.open("xb") as stream:
                np.savez_compressed(stream, latents=features, times=times,
                                    latitude=np.asarray(contract["latent_coordinates"]["lat"]),
                                    longitude=np.asarray(contract["latent_coordinates"]["lon"]),
                                    metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latitude-name", default="lat")
    parser.add_argument("--longitude-name", default="lon")
    parser.add_argument("--time-name", default="time")
    parser.add_argument("--through", help="Optional UTC ISO observation cutoff, inclusive")
    parser.add_argument("--max-history-gap-hours", type=float,
                        help="Maximum historical hold interval, including last sample to forecast origin. "
                             "Default: median observed interval. Required for a single sample; "
                             "declare explicitly for irregular observation cadence.")
    args = parser.parse_args(argv)
    metadata = prepare(args.encoder, args.data, args.output, batch_size=args.batch_size,
                       latitude_name=args.latitude_name, longitude_name=args.longitude_name,
                       time_name=args.time_name, through=args.through,
                       max_history_gap_hours=args.max_history_gap_hours)
    print(json.dumps({"output": str(args.output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
