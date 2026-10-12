"""Physical-unit and coordinate safeguards around official conservative regrid."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

_spec = importlib.util.spec_from_file_location(
    "prepare_neuralgcm_data",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_neuralgcm_data.py",
)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def era5():
    atmospheric_dims = ("time", "pressure_level", "lon", "lat")
    surface_dims = ("time", "lon", "lat")
    fields = {}
    for name, value, units in (
        ("z", 50000.0, "m**2 s**-2"),
        ("q", 0.005, "kg kg**-1"),
        ("t", 280.0, "K"),
        ("u", 4.0, "m s**-1"),
        ("v", 2.0, "m s**-1"),
    ):
        fields[name] = (atmospheric_dims, np.full((2, 2, 8, 5), value, dtype=np.float32), {"units": units})
    fields["sst"] = (surface_dims, np.full((2, 8, 5), 285.0, dtype=np.float32), {"units": "K"})
    fields["siconc"] = (surface_dims, np.zeros((2, 8, 5), dtype=np.float32), {"units": "(0 - 1)"})
    return xr.Dataset(fields, coords={
        "time": np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]"),
        "pressure_level": ("pressure_level", [85000.0, 50000.0], {"units": "Pa"}),
        "lon": ("lon", np.arange(-180, 180, 45), {"units": "degrees_east"}),
        "lat": ("lat", np.linspace(90, -90, 5), {"units": "degrees_north"}),
    })


class ModelBoundary:
    input_variables = ["geopotential", "specific_humidity", "temperature", "u_component_of_wind", "v_component_of_wind"]
    forcing_variables = ["sea_surface_temperature", "sea_ice_cover"]

    def __init__(self):
        from dinosaur import spherical_harmonic
        self.data_coords = SimpleNamespace(
            horizontal=spherical_harmonic.Grid.TL31(),
            vertical=SimpleNamespace(centers=np.array([500, 850])),
        )

    def inputs_from_xarray(self, data):
        np.testing.assert_array_equal(data.level, [500, 850])
        np.testing.assert_allclose(data.longitude, np.rad2deg(self.data_coords.horizontal.longitudes))
        np.testing.assert_allclose(data.latitude, np.rad2deg(self.data_coords.horizontal.latitudes))
        for name in self.input_variables:
            assert set(data[name].dims) == {"level", "longitude", "latitude"}

    def forcings_from_xarray(self, data):
        for name in self.forcing_variables:
            assert set(data[name].dims) == {"longitude", "latitude"}


def test_alias_units_and_coordinate_order_preserve_geopotential():
    source = era5()
    original = source.copy(deep=True)
    data = prep.canonicalize_era5(source)
    np.testing.assert_array_equal(data.level, [500, 850])
    assert data.level.attrs["units"] == "hPa"
    assert np.all(np.diff(data.latitude) > 0)
    np.testing.assert_array_equal(data.longitude, np.arange(0, 360, 45))
    np.testing.assert_allclose(data.geopotential, 50000)
    xr.testing.assert_identical(source, original)


@pytest.mark.parametrize("variable,units", [("z", "m"), ("t", "C"), ("q", "g/kg"), ("u", None), ("siconc", "%")])
def test_rejects_unsupported_units(variable, units):
    data = era5()
    data[variable].attrs = {} if units is None else {"units": units}
    with pytest.raises(ValueError, match="units"):
        prep.canonicalize_era5(data)


def test_actual_conservative_regrid_constant_fields():
    output = prep.prepare_dataset(era5(), ModelBoundary())
    assert output.temperature.shape == (2, 2, 64, 32)
    np.testing.assert_allclose(output.geopotential, 50000, rtol=1e-6)
    np.testing.assert_allclose(output.temperature, 280, rtol=1e-6)
    assert output.geopotential.attrs["units"] == "m2 s-2"
    assert np.isfinite(output.to_array()).all()


def test_rejects_missing_pressure_level():
    with pytest.raises(ValueError, match="pressure level 850"):
        prep.prepare_dataset(era5().isel(pressure_level=[1]), ModelBoundary())


def test_rejects_missing_forcing_field():
    with pytest.raises(ValueError, match="sea_surface_temperature"):
        prep.prepare_dataset(era5().drop_vars("sst"), ModelBoundary())


def test_forcing_file_requires_exact_timestamps():
    data = era5()
    atmos = data.drop_vars(["sst", "siconc"])
    forcing = data[["sst", "siconc"]].assign_coords(time=data.time + np.timedelta64(1, "h"))
    with pytest.raises(ValueError, match="Forcing time must exactly match"):
        prep.prepare_dataset(atmos, ModelBoundary(), forcings=forcing)


def test_atmospheric_nans_are_never_filled():
    data = era5()
    data.t.values[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Atmospheric field temperature has missing"):
        prep.prepare_dataset(data, ModelBoundary(), fill_forcing_nans=True)


def test_masked_forcing_fill_requires_explicit_flag():
    data = era5()
    data.sst.values[:, :, :2] = np.nan
    with pytest.raises(ValueError, match="fill-forcing-nans"):
        prep.prepare_dataset(data, ModelBoundary())
    output = prep.prepare_dataset(data, ModelBoundary(), fill_forcing_nans=True)
    assert np.isfinite(output.sea_surface_temperature).all()
    np.testing.assert_allclose(output.sea_surface_temperature, 285, rtol=1e-6)


def test_entirely_absent_forcing_frame_is_rejected():
    data = era5()
    data.sst.values[1] = np.nan
    with pytest.raises(ValueError, match="all values are NaN"):
        prep.prepare_dataset(data, ModelBoundary(), fill_forcing_nans=True)
