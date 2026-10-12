"""Frozen NeuralGCM guides for stages of an external forecast backbone.

One shared, recurrent NeuralGCM trajectory advances once per *physical forecast
interval*. Its decoded pressure-level temperature and carried latent state are
then reused by every CNN stage at that lead. CNN depth is not physical time.

This module deliberately exports NumPy arrays: gradients do not cross the
JAX/NumPy/PyTorch boundary. The guide generator must be trained separately;
stage adapters in the consuming model remain trainable. Only temperature is
exported as a physical guide, not winds, pressure, or a conservation guarantee.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


def _scalar(value: Any, name: str, *, positive: bool = False) -> float:
    array = np.asarray(value)
    if (array.ndim != 0 or array.dtype.kind not in "iuf"
            or not np.isfinite(array).item()):
        raise ValueError(f"{name} must be a finite numeric scalar")
    result = float(array)
    if result < 0 or (positive and result == 0):
        adjective = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {adjective}")
    return result


def _axis(values: Any, name: str, *, longitude: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must have at least two finite coordinates")
    comparable = values % 360 if longitude else values
    if np.unique(comparable).size != values.size:
        raise ValueError(f"{name} coordinates must be unique (longitude modulo 360)")
    if not longitude and np.any(np.abs(values) > 90 + 1e-8):
        raise ValueError(f"{name} must be in degrees within [-90, 90]")
    return values.copy()


def _sim_time(state: Any) -> Any:
    try:
        model_state = state.model_state
        prognostics = model_state.state if hasattr(model_state, "state") else model_state
        value = prognostics.sim_time
    except AttributeError as error:
        raise ValueError("Guide state requires model prognostics.sim_time") from error
    # NeuralGCM sim_time is a nondimensional model clock, not datetime64 or hours.
    array = np.asarray(value)
    if (array.ndim != 0 or array.dtype.kind not in "iuf"
            or not np.isfinite(array).item()):
        raise ValueError("State sim_time must be a finite scalar model clock")
    return value


def _boundary(origin_forcings: Any, sim_time: Any) -> dict[str, Any]:
    if not isinstance(origin_forcings, Mapping):
        raise TypeError("origin_forcings must be a mapping of one origin's boundary fields")
    if "sim_time" in origin_forcings:
        value = np.asarray(origin_forcings["sim_time"])
        if (value.ndim != 0 or value.dtype.kind not in "iuf"
                or not np.isfinite(value).item()):
            raise ValueError("Origin forcing sim_time must be a finite scalar, not a time series")
    # The physical boundary values persist; only the clock comes from the
    # accepted recurrent state. Never edit the caller's forcing dictionary.
    return {**origin_forcings, "sim_time": sim_time}


@dataclass(frozen=True)
class GuideFrame:
    """Single lead's reusable guide, with channels first and latitude before longitude.

    ``physical[P,H,W]`` is normalized pressure-level absolute temperature;
    ``latent[D,H,W]`` is the generator's actual recurrent latent state. Leads
    are hours from the caller's declared forecast origin. Arrays are copied
    when constructed so they do not alias generator state.
    """

    physical: np.ndarray
    latent: np.ndarray
    lead_hours: float

    def __post_init__(self) -> None:
        physical = np.array(self.physical, dtype=np.float32, copy=True)
        latent = np.array(self.latent, dtype=np.float32, copy=True)
        if (physical.ndim != 3 or latent.ndim != 3
                or min(physical.shape) < 1 or min(latent.shape) < 1
                or physical.shape[-2:] != latent.shape[-2:]):
            raise ValueError("Guides require physical[P,H,W] and latent[D,H,W] on the same grid")
        if not np.isfinite(physical).all() or not np.isfinite(latent).all():
            raise ValueError("Guide arrays must be finite")
        object.__setattr__(self, "physical", physical)
        object.__setattr__(self, "latent", latent)
        object.__setattr__(self, "lead_hours", _scalar(self.lead_hours, "lead_hours"))


class NeuralGCMStageGuide:
    """Extract real physical and recurrent latent guidance from an enabled adapter.

    Target coordinates are explicit degrees, ordered exactly like the external
    backbone's axes. ``pressure_levels`` is an explicit subset/permutation of
    ``adapter.base_model.data_coords.vertical.centers`` in hPa. No pressure
    interpolation or inference from channel count is performed. Means/stds
    are one value per requested temperature level, in kelvin, preferably
    fitted on training data only.

    State is owned by the caller. Retain the returned augmented state across
    intervals to preserve thermal memory, latent memory, innovations and RNG.
    Initialize/warm this state with the existing NeuralGCM workflow; this class
    never re-encodes observations or resets memory. JAX is imported lazily only
    when extracting a frame through the existing periodic regridding function.
    """

    def __init__(
        self, adapter: Any, params: Any, *, target_longitude: Any,
        target_latitude: Any, pressure_levels: Any,
        temperature_means: Any, temperature_stds: Any,
    ) -> None:
        if not getattr(adapter, "enabled", False):
            raise ValueError("Stage guidance requires an enabled NeuralGCM memory adapter")
        self.latent_dim = getattr(getattr(adapter, "config", None), "latent_dim", 0)
        if (isinstance(self.latent_dim, (bool, np.bool_))
                or not isinstance(self.latent_dim, (int, np.integer))
                or self.latent_dim < 1):
            raise ValueError("Stage guidance requires recurrent config.latent_dim > 0")
        self.dt_hours = _scalar(adapter.dt_hours, "adapter.dt_hours", positive=True)
        self.adapter, self.params = adapter, params
        self.target_longitude = _axis(target_longitude, "target_longitude", longitude=True)
        self.target_latitude = _axis(target_latitude, "target_latitude", longitude=False)
        base = adapter.base_model
        self._data_lon, self._data_lat = self._grid_axes(base.data_coords.horizontal, "data")
        self._model_lon, self._model_lat = self._grid_axes(base.model_coords.horizontal, "model")
        centers = np.asarray(base.data_coords.vertical.centers, dtype=float)
        requested = np.asarray(pressure_levels, dtype=float)
        if (centers.ndim != 1 or requested.ndim != 1 or not requested.size
                or not np.isfinite(centers).all() or not np.isfinite(requested).all()
                or np.any(centers <= 0) or np.any(requested <= 0)
                or np.unique(requested).size != requested.size):
            raise ValueError("pressure_levels must be unique positive finite hPa values")
        indices = []
        for level in requested:
            matches = np.flatnonzero(np.isclose(centers, level, rtol=0, atol=1e-5))
            if matches.size != 1:
                raise ValueError(
                    f"Requested pressure level {level:g} hPa must match exactly one "
                    "data_coords.vertical.centers entry; no vertical interpolation"
                )
            indices.append(int(matches[0]))
        self.pressure_levels = requested.copy()
        self._pressure_indices = np.asarray(indices)
        self._source_pressure_count = centers.size
        self.temperature_means = np.asarray(temperature_means, dtype=float).copy()
        self.temperature_stds = np.asarray(temperature_stds, dtype=float).copy()
        for name, values in (("temperature_means", self.temperature_means),
                             ("temperature_stds", self.temperature_stds)):
            if values.shape != requested.shape or not np.isfinite(values).all():
                raise ValueError(f"{name} must provide one finite kelvin value per pressure level")
        if np.any(self.temperature_stds <= 0):
            raise ValueError("temperature_stds must be strictly positive")

    @staticmethod
    def _grid_axes(grid: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
        return (_axis(np.rad2deg(grid.longitudes), f"{name} longitude", longitude=True),
                _axis(np.rad2deg(grid.latitudes), f"{name} latitude", longitude=False))

    def current_frame(
        self, state: Any, forcings: Mapping[str, Any], *, lead_hours: float,
    ) -> GuideFrame:
        """Decode and regrid current state without advancing time or consuming RNG.

        Reuse this returned frame at all CNN stages of one forecast lead.
        Latitude interpolation edge-extends the polar caps; longitude is
        periodic. This is interpolation for guidance, not conservative remapping.
        """
        lead = _scalar(lead_hours, "lead_hours")
        sim_time = _sim_time(state)
        latent_state = getattr(state, "latent_state", None)
        if latent_state is None or not hasattr(latent_state, "latent"):
            raise ValueError("Guide state must carry a recurrent latent_state.latent")
        latent = np.asarray(latent_state.latent)
        expected_latent = (self.latent_dim, self._model_lon.size, self._model_lat.size)
        if latent.shape != expected_latent or not np.isfinite(latent).all():
            raise ValueError(f"Recurrent latent must be finite with shape {expected_latent} [D,lon,lat]")
        decoded = self.adapter.decode(state, _boundary(forcings, sim_time))
        if not isinstance(decoded, Mapping) or "temperature" not in decoded:
            raise ValueError("Adapter decoder must return absolute temperature in kelvin")
        temperature = np.asarray(decoded["temperature"])
        expected = (self._source_pressure_count, self._data_lon.size, self._data_lat.size)
        if temperature.shape != expected or not np.isfinite(temperature).all():
            raise ValueError(f"Decoded temperature must be finite with shape {expected} [pressure,lon,lat]")
        from ngcm_memory.features import regrid_lonlat

        physical = np.asarray(regrid_lonlat(
            temperature[self._pressure_indices], self._data_lon, self._data_lat,
            self.target_longitude, self.target_latitude,
        ))
        physical = ((physical - self.temperature_means[:, None, None])
                    / self.temperature_stds[:, None, None])
        latent = np.asarray(regrid_lonlat(
            latent, self._model_lon, self._model_lat,
            self.target_longitude, self.target_latitude,
        ))
        return GuideFrame(physical.transpose(0, 2, 1), latent.transpose(0, 2, 1), lead)

    def advance(
        self, state: Any, origin_forcings: Mapping[str, Any], *,
        interval_hours: float, lead_hours: float | None = None,
    ) -> tuple[Any, GuideFrame]:
        """Advance ONE physical interval and return its reusable guidance frame.

        ``interval_hours`` must be an integer multiple of the native timestep.
        Boundary values persist from ``origin_forcings`` and each native step
        reads sim_time from current prognostics. No future observations or
        labels enter this API. ``lead_hours`` explicitly labels cumulative lead;
        if omitted it defaults to this interval's length (single-interval use).
        For a trajectory, supply cumulative leads, e.g. 24, 48, 72 hours.
        """
        interval = _scalar(interval_hours, "interval_hours", positive=True)
        ratio = interval / self.dt_hours
        if not np.isfinite(ratio) or ratio < 1 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-8):
            raise ValueError("interval_hours must be an integer multiple of adapter.dt_hours")
        lead = interval if lead_hours is None else _scalar(lead_hours, "lead_hours")
        if lead < interval:
            raise ValueError("Cumulative lead_hours must be at least interval_hours")
        if getattr(state, "latent_state", None) is None:
            raise ValueError("Guide state must carry a recurrent latent_state")
        current = state
        for _ in range(round(ratio)):
            before = _sim_time(current)
            current = self.adapter.advance(self.params, current, _boundary(origin_forcings, before))
            if float(np.asarray(_sim_time(current))) <= float(np.asarray(before)):
                raise ValueError("Adapter must advance prognostic sim_time at each internal step")
        return current, self.current_frame(current, origin_forcings, lead_hours=lead)
