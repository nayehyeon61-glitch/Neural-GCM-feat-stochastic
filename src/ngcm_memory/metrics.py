"""Area-weighted forecast scores on [member, lead, level, lon, lat] fields.

Both empirical-distribution CRPS and the fair CRPS estimator are provided.
The latter removes the finite-member underdispersion incentive during training
when members are conditionally independent draws. Levels are averaged equally;
this is not a mass-weighted column energy metric.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def area_mean(values: Any, area_weights: Any) -> Any:
    """Reduce only the two final (longitude, latitude) dimensions."""
    values, weights = jnp.asarray(values), jnp.asarray(area_weights)
    if values.ndim < 2 or weights.shape != values.shape[-2:]:
        raise ValueError("area_weights must match the final [lon, lat] axes")
    return jnp.sum(values * weights, axis=(-2, -1)) / jnp.sum(weights)


def ensemble_crps(ensemble: Any, target: Any, area_weights: Any, *, fair: bool = False) -> Any:
    """CRPS reduced over space, retaining target's other axes.

    ensemble has one extra leading member axis. Pairwise differences are
    accumulated in a loop to avoid allocating [member, member, ...] fields.
    fair=True uses the unbiased n(n-1) estimator for conditionally iid members;
    its score on an individual realization need not be nonnegative.
    """
    ensemble, target = jnp.asarray(ensemble), jnp.asarray(target)
    if ensemble.shape[1:] != target.shape or ensemble.shape[0] < 1:
        raise ValueError("ensemble must have shape [members, *target.shape]")
    n = ensemble.shape[0]
    if fair and n < 2:
        raise ValueError("Fair CRPS requires at least two members")
    score = jnp.mean(jnp.abs(ensemble - target), axis=0)
    pairwise = jnp.zeros_like(target)
    for member in range(n):
        pairwise = pairwise + jnp.mean(
            jnp.abs(ensemble[member] - ensemble), axis=0
        )
    denominator = n - 1 if fair else n
    return area_mean(score - 0.5 * pairwise / denominator, area_weights)


def forecast_metrics(
    ensemble: Any, target: Any, initial: Any, area_weights: Any
) -> dict[str, np.ndarray]:
    """Per-lead diagnostics for [member,lead,level,lon,lat] predictions.

    Forecast global-mean change is measured against the observed initial
    state. Its error subtracts the observed change, distinguishing physical
    evolution from excess drift. Spread uses population ensemble variance.
    """
    ensemble, target, initial = map(jnp.asarray, (ensemble, target, initial))
    if ensemble.ndim != 5 or ensemble.shape[1:] != target.shape:
        raise ValueError("Expected [member,lead,level,lon,lat] and matching target")
    if initial.shape != target.shape[1:]:
        raise ValueError("initial must have shape [level,lon,lat]")
    mean = jnp.mean(ensemble, axis=0)
    error = mean - target
    spatial_mse = area_mean(error**2, area_weights).mean(axis=-1)
    spread_squared = area_mean(jnp.var(ensemble, axis=0), area_weights).mean(axis=-1)
    forecast_change = area_mean(mean - initial, area_weights).mean(axis=-1)
    observed_change = area_mean(target - initial, area_weights).mean(axis=-1)
    result = {
        "rmse": jnp.sqrt(spatial_mse),
        "bias": area_mean(error, area_weights).mean(axis=-1),
        "crps": ensemble_crps(ensemble, target, area_weights).mean(axis=-1),
        "fair_crps": ensemble_crps(ensemble, target, area_weights, fair=True).mean(axis=-1),
        "spread": jnp.sqrt(spread_squared),
        "forecast_global_mean_change": forecast_change,
        "observed_global_mean_change": observed_change,
        "global_mean_drift_error": forecast_change - observed_change,
    }
    return {name: np.asarray(value) for name, value in result.items()}


def aggregate_window_metrics(
    windows: list[dict[str, np.ndarray]],
) -> dict[str, list[float]]:
    """Aggregate equal-size windows; pool MSE/variance before square roots."""
    if not windows:
        raise ValueError("No evaluation windows")
    result = {}
    for name in windows[0]:
        values = np.stack([window[name] for window in windows])
        mean = np.sqrt(np.mean(values**2, axis=0)) if name in {"rmse", "spread"} else np.mean(values, axis=0)
        result[name] = mean.tolist()
    return result
