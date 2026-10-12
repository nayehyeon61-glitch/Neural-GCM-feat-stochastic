#!/usr/bin/env python3
"""Prepare a bounded ERA5 time chunk on a checkpoint's pressure-level grid.

Inputs are read-only. Repeated --input arguments may be quoted local globs.
All selected data are loaded into memory: prepare manageable time chunks, not
an entire high-resolution archive at once. Atmospheric pressure levels must
exist in the source; no vertical interpolation or missing-field synthesis is
performed. Explicit --fill-forcing-nans enables nearest-horizontal filling of
existing masked SST/ice cells, as in the official NeuralGCM inference example.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import pickle

import numpy as np
import xarray as xr


ALIASES = {
    "lon": "longitude", "lat": "latitude", "pressure_level": "level",
    "valid_time": "time", "z": "geopotential", "q": "specific_humidity",
    "t": "temperature", "u": "u_component_of_wind", "v": "v_component_of_wind",
    "sst": "sea_surface_temperature", "siconc": "sea_ice_cover", "ci": "sea_ice_cover",
    "clwc": "specific_cloud_liquid_water_content",
    "ciwc": "specific_cloud_ice_water_content",
}

_UNITS = {
    "temperature": ({"k", "kelvin"}, "K"),
    "sea_surface_temperature": ({"k", "kelvin"}, "K"),
    "u_component_of_wind": ({"ms-1", "m/s"}, "m s-1"),
    "v_component_of_wind": ({"ms-1", "m/s"}, "m s-1"),
    "geopotential": ({"m2s-2", "m2/s2"}, "m2 s-2"),
    "specific_humidity": ({"kgkg-1", "kg/kg", "1"}, "kg kg-1"),
    "specific_cloud_liquid_water_content": ({"kgkg-1", "kg/kg", "1"}, "kg kg-1"),
    "specific_cloud_ice_water_content": ({"kgkg-1", "kg/kg", "1"}, "kg kg-1"),
    "sea_ice_cover": ({"1", "0-1", "(0-1)", "fraction"}, "1"),
}


def _unit_token(units: object) -> str:
    return str(units).strip().lower().replace(" ", "").replace("**", "").replace("^", "")


def canonicalize_era5(data: xr.Dataset) -> xr.Dataset:
    """Rename unambiguous ERA5 aliases and require known, physical units."""
    renames = {}
    for old, new in ALIASES.items():
        if old in data.variables or old in data.dims:
            if new in data.variables or new in data.dims or new in renames.values():
                raise ValueError(f"Ambiguous aliases: both {old!r} and {new!r} are present")
            renames[old] = new
    data = data.rename(renames).copy()
    for name, (accepted, canonical) in _UNITS.items():
        if name not in data:
            continue
        units = data[name].attrs.get("units")
        if units is None or _unit_token(units) not in accepted:
            raise ValueError(
                f"Unsupported or missing units for {name}: {units!r}; expected {canonical}. "
                "Provide physical ERA5 values, not standardized fields."
            )
        data[name].attrs = {**data[name].attrs, "units": canonical}
    if "level" in data.coords:
        units = _unit_token(data.level.attrs.get("units", ""))
        if units in {"pa", "pascal", "pascals"}:
            level = data.level.values / 100.0
        elif units in {"hpa", "hectopascal", "hectopascals", "millibar", "millibars", "mbar"}:
            level = data.level.values
        else:
            raise ValueError(f"Pressure level units must explicitly be Pa or hPa; got {units!r}")
        data = data.assign_coords(level=("level", level, {"units": "hPa"})).sortby("level")
        if not np.isfinite(data.level.values).all() or np.any(np.diff(data.level.values) <= 0):
            raise ValueError("Pressure levels must be finite, positive and unique")
        if np.any(data.level.values <= 0):
            raise ValueError("Pressure levels must be positive")
    for coordinate in ("latitude", "longitude", "time"):
        if coordinate not in data.coords or data[coordinate].dims != (coordinate,):
            raise ValueError(f"Expected a one-dimensional {coordinate!r} coordinate")
    if not np.issubdtype(data.time.dtype, np.datetime64):
        raise ValueError("time must contain decoded datetime64 timestamps")
    if np.isnat(data.time.values).any():
        raise ValueError("time contains NaT")
    if data.sizes["time"] == 0:
        raise ValueError("No input timestamps selected")
    data = data.sortby("time")
    if np.any(np.diff(data.time.values) <= np.timedelta64(0, "ns")):
        raise ValueError("time must contain unique timestamps")
    for coordinate, accepted in (
        ("latitude", {"degrees_north", "degree_north", "degrees_n", "degree_n", "degrees"}),
        ("longitude", {"degrees_east", "degree_east", "degrees_e", "degree_e", "degrees"}),
    ):
        units = str(data[coordinate].attrs.get("units", "")).lower()
        if units not in accepted:
            raise ValueError(f"{coordinate} units must explicitly be degrees, got {units!r}")
        if not np.isfinite(data[coordinate].values).all():
            raise ValueError(f"{coordinate} contains nonfinite coordinates")
    if np.any(np.abs(data.latitude.values) > 90):
        raise ValueError("Latitude outside [-90,90] degrees")
    longitude = np.mod(data.longitude.values, 360.0)
    data = data.assign_coords(longitude=("longitude", longitude, {"units": "degrees_east"}))
    data = data.sortby("latitude").sortby("longitude")
    for coordinate in ("latitude", "longitude"):
        if np.any(np.diff(data[coordinate].values) <= 0):
            raise ValueError(f"Duplicate {coordinate} coordinates; remove cyclic endpoints explicitly")
    return data


def merge_exact_forcings(data: xr.Dataset, forcings: xr.Dataset, names: list[str]) -> xr.Dataset:
    """Merge forcings only when their timestamps and horizontal grid match."""
    for coordinate in ("time", "latitude", "longitude"):
        if not np.array_equal(data[coordinate].values, forcings[coordinate].values):
            raise ValueError(f"Forcing {coordinate} must exactly match atmospheric inputs")
    available = [name for name in names if name in forcings]
    overlap = set(available) & set(data.data_vars)
    if overlap:
        raise ValueError(f"Forcings occur in both input sources: {sorted(overlap)}")
    return xr.merge([data, forcings[available]], join="exact", compat="equals")


def prepare_dataset(
    data: xr.Dataset,
    model,
    *,
    forcings: xr.Dataset | None = None,
    fill_forcing_nans: bool = False,
) -> xr.Dataset:
    """Validate and conservatively regrid using official Dinosaur operations."""
    from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

    data = canonicalize_era5(data)
    if forcings is not None:
        forcings = canonicalize_era5(forcings)
        data = merge_exact_forcings(data, forcings, model.forcing_variables)
    names = model.input_variables + model.forcing_variables
    missing = sorted(set(names) - set(data.data_vars))
    if missing:
        raise ValueError(f"Missing checkpoint-required fields: {missing}; no fields will be fabricated")
    if "level" not in data.coords:
        raise ValueError("Atmospheric input requires pressure levels")
    target_levels = np.asarray(model.data_coords.vertical.centers)
    source_levels = np.asarray(data.level.values)
    indices = []
    for target in target_levels:
        matches = np.flatnonzero(np.isclose(source_levels, target, rtol=0.0, atol=1e-5))
        if len(matches) != 1:
            raise ValueError(f"Required pressure level {target:g} hPa is missing; no vertical interpolation")
        indices.append(int(matches[0]))
    data = data[names].isel(level=indices).assign_coords(
        level=("level", target_levels, {"units": "hPa"})
    )
    for name in names:
        expected = {"time", "longitude", "latitude"}
        if name in model.input_variables:
            expected.add("level")
        if set(data[name].dims) != expected:
            raise ValueError(f"Unexpected dimensions for {name}: {data[name].dims}; expected {sorted(expected)}")
        values = data[name].values
        if np.isinf(values).any():
            raise ValueError(f"{name} contains infinite values")
        if name in model.input_variables and not np.isfinite(values).all():
            raise ValueError(f"Atmospheric field {name} has missing data; it will not be filled")
        if name in model.forcing_variables and np.isnan(values).any() and not fill_forcing_nans:
            raise ValueError(f"{name} has masked cells; use --fill-forcing-nans for nearest-horizontal filling")
        if name == "sea_ice_cover" and np.any((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)):
            raise ValueError("sea_ice_cover must be a fraction in [0,1]")
    source_grid = spherical_harmonic.Grid(
        latitude_nodes=data.sizes["latitude"],
        longitude_nodes=data.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(data.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(data.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        source_grid, model.data_coords.horizontal, skipna=True
    )
    # Keep explicit units; xarray apply_ufunc defaults may drop them.
    attrs = {name: dict(data[name].attrs) for name in names}
    output = xarray_utils.regrid_horizontal(data, regridder).load()
    if fill_forcing_nans:
        for name in model.forcing_variables:
            # Per-time filling permits time-varying ocean masks. The official
            # helper rejects a completely missing horizontal field.
            output[name] = xr.concat(
                [xarray_utils.fill_nan_with_nearest(output[name].isel(time=i))
                 for i in range(output.sizes["time"])],
                dim=output.time,
            )
    for name in names:
        output[name].attrs = attrs[name]
        if not np.isfinite(output[name].values).all():
            raise ValueError(f"Prepared field {name} still contains nonfinite values")
    output.latitude.attrs["units"] = "degrees_north"
    output.longitude.attrs["units"] = "degrees_east"
    output.level.attrs["units"] = "hPa"
    output.attrs.update({
        "preparation": "NeuralGCM conservative horizontal regrid; exact existing pressure levels",
        "forcing_nan_fill": "nearest horizontal" if fill_forcing_nans else "none",
    })
    model.inputs_from_xarray(output.isel(time=0))
    model.forcings_from_xarray(output.isel(time=0))
    return output


def _paths(patterns: list[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        matched = [Path(path).resolve() for path in sorted(glob.glob(pattern))]
        if not matched:
            raise ValueError(f"Input pattern matched no files: {pattern}")
        paths.extend(matched)
    if len(set(paths)) != len(paths):
        raise ValueError("Repeated/globbed inputs contain duplicate files")
    return paths


def _load(paths: list[Path], start: str | None, end: str | None) -> xr.Dataset:
    datasets = []
    for path in paths:
        with xr.open_dataset(path) as source:
            data = canonicalize_era5(source)
            if start or end:
                data = data.sel(time=slice(start, end))
            if data.sizes["time"]:
                datasets.append(data.load())
    if not datasets:
        raise ValueError("No timestamps in selected files/time interval")
    if len(datasets) == 1:
        return datasets[0]
    return xr.combine_by_coords(datasets, join="exact", combine_attrs="drop_conflicts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Local NetCDF path or quoted glob; repeatable")
    parser.add_argument("--forcings-file", type=Path, help="Optional SST/ice NetCDF on exactly the same timestamps/grid")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--trust-base-checkpoint", action="store_true", help="Required acknowledgement that this pickle is trusted")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-start")
    parser.add_argument("--time-end")
    parser.add_argument("--fill-forcing-nans", action="store_true")
    args = parser.parse_args()
    if not args.trust_base_checkpoint:
        parser.error("--trust-base-checkpoint is required: Python pickle checkpoints can execute code")
    paths = _paths(args.input)
    output_path = args.output.resolve()
    input_paths = paths + [args.base_checkpoint.resolve()]
    if args.forcings_file:
        input_paths.append(args.forcings_file.resolve())
    if output_path in input_paths:
        parser.error("Output must not overwrite any input or checkpoint")
    if output_path.exists():
        parser.error("Output already exists; choose a new file")
    import neuralgcm
    with args.base_checkpoint.open("rb") as file:
        model = neuralgcm.PressureLevelModel.from_checkpoint(pickle.load(file))
    data = _load(paths, args.time_start, args.time_end)
    forcings = _load([args.forcings_file], args.time_start, args.time_end) if args.forcings_file else None
    output = prepare_dataset(data, model, forcings=forcings, fill_forcing_nans=args.fill_forcing_nans)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(output_path)
    # Verify the exact saved product with the checkpoint's public input API.
    with xr.open_dataset(output_path) as saved:
        model.inputs_from_xarray(saved.isel(time=0))
        model.forcings_from_xarray(saved.isel(time=0))
    print(f"Prepared {output.sizes['time']} timestamps -> {output_path}")
    print(f"Pressure levels: {output.sizes['level']}; grid: {output.sizes['longitude']} x {output.sizes['latitude']}")


if __name__ == "__main__":
    main()
