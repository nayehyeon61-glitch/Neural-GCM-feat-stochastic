import jax.numpy as jnp
import numpy as np
import pytest

from ngcm_memory.metrics import (
    aggregate_window_metrics, area_mean, ensemble_crps, forecast_metrics,
)


def test_area_mean_uses_nonuniform_grid_weights():
    field = jnp.asarray([[0., 10.], [0., 10.]])
    weights = jnp.asarray([[1., 3.], [1., 3.]])
    assert float(area_mean(field, weights)) == pytest.approx(7.5)


def test_crps_has_correct_empirical_pairwise_factor():
    members = jnp.asarray([0., 2.])[:, None, None]
    target = jnp.ones((1, 1))
    assert float(ensemble_crps(members, target, jnp.ones((1, 1)))) == pytest.approx(.5)
    assert float(ensemble_crps(members, target, jnp.ones((1, 1)), fair=True)) == pytest.approx(0.)
    assert float(ensemble_crps(members[:1], target, jnp.ones((1, 1)))) == pytest.approx(1.)
    with pytest.raises(ValueError, match="two members"):
        ensemble_crps(members[:1], target, jnp.ones((1, 1)), fair=True)


def test_forecast_metrics_distinguishes_evolution_from_drift():
    # A perfect ensemble with real global warming should have zero drift error.
    truth = jnp.asarray([1., 2.])[:, None, None, None]
    predictions = jnp.stack([truth - 1, truth + 1])
    result = forecast_metrics(predictions, truth, jnp.zeros((1, 1, 1)), jnp.ones((1, 1)))
    np.testing.assert_allclose(result["rmse"], [0., 0.])
    np.testing.assert_allclose(result["spread"], [1., 1.])
    np.testing.assert_allclose(result["forecast_global_mean_change"], [1., 2.])
    np.testing.assert_allclose(result["global_mean_drift_error"], [0., 0.])


def test_rmse_aggregation_pools_squared_errors():
    pooled = aggregate_window_metrics([
        {"rmse": np.asarray([0.]), "spread": np.asarray([1.]), "bias": np.asarray([-1.])},
        {"rmse": np.asarray([2.]), "spread": np.asarray([3.]), "bias": np.asarray([1.])},
    ])
    assert pooled["rmse"] == pytest.approx([np.sqrt(2)])
    assert pooled["spread"] == pytest.approx([np.sqrt(5)])
    assert pooled["bias"] == pytest.approx([0.])


def test_invalid_metric_shapes_rejected():
    with pytest.raises(ValueError):
        ensemble_crps(jnp.zeros((2, 3, 4)), jnp.zeros((2, 4)), jnp.ones((2, 4)))
