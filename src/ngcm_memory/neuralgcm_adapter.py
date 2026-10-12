"""Checkpoint-preserving thermal memory closure for NeuralGCM.

This is a Lie/operator-split residual closure, not a replacement of the
semi-implicit primitive-equation integrator.  At each *internal* NeuralGCM
timestep it freezes closure features at the current state, advances the base
model once, and adds the integrated closure temperature increment spectrally.
No pressure-level decode/re-encode is performed inside the advance step.

Only ``temperature_variation`` is corrected.  Model time, winds, humidity,
surface pressure, existing randomness and existing memory retain the values
returned by the base model.  Existing diagnostic fields are not recomputed;
in particular, they must not be treated as closure-consistent precipitation
or energy-budget diagnostics.  Temperature corrections are not, by themselves,
a conservative moist-energy closure.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ngcm_memory.closure import (
    ClosureConfig,
    ClosureState,
    closure_step,
    initialize_memory,
)
from ngcm_memory.latent import LatentState, initialize_latent, latent_step


class AugmentedState(NamedTuple):
    """Explicit recurrent state; safe to carry through ``jax.lax.scan``."""

    model_state: Any
    closure_state: ClosureState
    latent_state: LatentState | None = None


def _prognostics(state: Any) -> Any:
    # Stochastic checkpoints return dinosaur.typing.ModelState, whereas some
    # deterministic checkpoints expose the primitive-equation state directly.
    return state.state if hasattr(state, "state") else state


def _replace_temperature(state: Any, temperature_variation: jax.Array) -> Any:
    prognostics = _prognostics(state)
    if not dataclasses.is_dataclass(prognostics):
        raise TypeError("Expected a dinosaur primitive-equation dataclass state")
    corrected = dataclasses.replace(
        prognostics, temperature_variation=temperature_variation
    )
    if hasattr(state, "state"):
        return dataclasses.replace(state, state=corrected)
    return corrected


class NeuralGCMMemoryAdapter:
    """Add a trainable, spatially conditioned closure to an existing model.

    Args:
        base_model: A ``neuralgcm.PressureLevelModel`` loaded from a checkpoint.
        config: Configuration shared with ``init_params`` and ``closure_step``.
            ``levels`` must match the checkpoint's *sigma* levels.
        feature_fn: Optional differentiable function ``(model_state, forcings)
            -> features[feature_dim, longitude, latitude]`` on the model's
            internal nodal grid.  Inputs are the current, pre-advance state.
            It must not inspect future observations.  If omitted, normalized
            temperature variations alone are used and feature_dim == levels.
        feature_scale_kelvin: Scale of temperature variation features used by
            the default feature function; this is not an output scale.
        enabled: Static ablation switch.  False skips all closure operations
            and returns exactly the state produced by ``base_model.advance``.
        remove_global_mean: Subtract the Gaussian-grid area mean of each sigma
            layer's correction.  This is optional drift control, not moist
            energy conservation.  False permits learning global thermal bias.

    Parameters are passed explicitly to ``advance`` so JAX can differentiate
    the actual corrected forecast recurrence.  Base parameters are fixed when
    differentiating only the closure parameter argument.
    """

    def __init__(
        self,
        base_model: Any,
        config: ClosureConfig,
        feature_fn: Callable[[Any, Any], jax.Array] | None = None,
        *,
        feature_scale_kelvin: float = 10.0,
        enabled: bool = True,
        remove_global_mean: bool = False,
    ) -> None:
        if not np.isfinite(feature_scale_kelvin) or feature_scale_kelvin <= 0:
            raise ValueError("feature_scale_kelvin must be finite and positive")
        self.base_model = base_model
        self.config = config
        self.enabled = bool(enabled)
        self.remove_global_mean = bool(remove_global_mean)
        self.feature_scale_kelvin = float(feature_scale_kelvin)
        self.feature_fn = feature_fn
        coords = base_model.model_coords
        if config.levels != coords.vertical.layers:
            raise ValueError(
                f"Closure levels={config.levels} does not match NeuralGCM "
                f"sigma levels={coords.vertical.layers}"
            )
        if config.latent_dim and feature_fn is not None:
            raise ValueError("Latent mode owns thermal + recurrent latent features; feature_fn must be None")
        if feature_fn is None and not config.latent_dim and config.feature_dim != config.levels:
            raise ValueError(
                "Default thermal features require feature_dim == levels; "
                "provide feature_fn for spatial/statistical conditioning"
            )
        self.grid_shape = tuple(coords.horizontal.nodal_shape)
        self.dt_hours = float(base_model.timestep / np.timedelta64(1, "h"))
        if not np.isfinite(self.dt_hours) or self.dt_hours <= 0:
            raise ValueError("NeuralGCM timestep must be finite and positive")
        self.area_weights = jnp.asarray(coords.horizontal.quadrature_weights)
        if self.area_weights.shape != self.grid_shape:
            raise ValueError("NeuralGCM quadrature weights have unexpected shape")

    def temperature_variation_kelvin(self, model_state: Any) -> jax.Array:
        """Return temperature *variation* (not absolute T), [sigma,lon,lat]."""
        field = _prognostics(model_state).temperature_variation
        nodal = self.base_model.model_coords.horizontal.to_nodal(field)
        return self.base_model.from_nondim_units(nodal, "K")

    def features(self, model_state: Any, forcings: Any,
                 latent_state: LatentState | None = None) -> jax.Array:
        if self.feature_fn is None:
            values = (
                self.temperature_variation_kelvin(model_state)
                / self.feature_scale_kelvin
            )
            if self.config.latent_dim:
                if latent_state is None:
                    raise ValueError("Latent mode requires carried latent_state")
                values = jnp.concatenate([values, latent_state.latent], axis=0)
        else:
            values = self.feature_fn(model_state, forcings)
        values = jnp.asarray(values)
        expected_shape = (self.config.feature_dim, *self.grid_shape)
        if values.shape != expected_shape:
            raise ValueError(
                f"Expected features {expected_shape} [feature,lon,lat], "
                f"got {values.shape}"
            )
        return values

    def initialize(self, model_state: Any, key: jax.Array) -> AugmentedState:
        """Attach cold-start closure state to an already encoded base state.

        For evaluation after a warm-up period, retain the returned recurrent
        state through the warm-up; do not reinitialize at each forecast step.
        """
        if not hasattr(_prognostics(model_state), "temperature_variation"):
            raise TypeError("Checkpoint has no temperature_variation prognostic")
        return AugmentedState(
            model_state,
            initialize_memory(self.grid_shape, self.config, key),
            initialize_latent(self.grid_shape, self.config, jax.random.fold_in(key, 7320))
            if self.config.latent_dim else None,
        )

    def encode(
        self, inputs: Any, forcings: Any, key: jax.Array
    ) -> AugmentedState:
        """Encode once with independent base-model and closure random keys."""
        base_key, closure_key = jax.random.split(key)
        model_state = self.base_model.encode(inputs, forcings, rng_key=base_key)
        return self.initialize(model_state, closure_key)

    def advance(
        self, params: Any, state: AugmentedState, forcings: Any
    ) -> AugmentedState:
        """Advance one internal model timestep with the thermal closure."""
        return self._advance(params, state, forcings, evolve_latent=True)

    def advance_warmup(
        self, params: Any, state: AugmentedState, forcings: Any
    ) -> AugmentedState:
        """Warm thermal memory on observations without inventing historical z.

        Recurrent mode uses zero source channels during native thermal warmup
        and preserves the entire latent state. Observed source history then
        initializes separate spatial memory and z once at the forecast origin.
        Legacy mode follows its ordinary advance behavior.
        """
        return self._advance(params, state, forcings, evolve_latent=False)

    def _advance(self, params, state, forcings, *, evolve_latent):
        next_model_state = self.base_model.advance(state.model_state, forcings)
        if not self.enabled:
            # No spectral round-trip and no RNG or memory consumption.
            return AugmentedState(next_model_state, state.closure_state, state.latent_state)

        feature_latent_state = state.latent_state
        if self.config.latent_dim and not evolve_latent:
            if feature_latent_state is None:
                raise ValueError("Latent mode requires carried latent_state")
            feature_latent_state = feature_latent_state._replace(
                latent=jnp.zeros_like(feature_latent_state.latent)
            )
        features = self.features(state.model_state, forcings, feature_latent_state)
        increment_kelvin, next_closure_state = closure_step(
            params, state.closure_state, features, self.dt_hours, self.config
        )
        if self.remove_global_mean:
            mean = jnp.sum(
                increment_kelvin * self.area_weights, axis=(-2, -1), keepdims=True
            ) / jnp.sum(self.area_weights)
            increment_kelvin = increment_kelvin - mean
        increment_nondim = self.base_model.to_nondim_units(increment_kelvin, "K")
        # Transform only the increment.  Re-transforming the entire base state
        # would alter it even for an identically zero closure.
        increment_modal = self.base_model.model_coords.horizontal.to_modal(
            increment_nondim
        )
        original_temperature = _prognostics(next_model_state).temperature_variation
        corrected_state = _replace_temperature(
            next_model_state, original_temperature + increment_modal
        )
        next_latent_state = state.latent_state
        if self.config.latent_dim and evolve_latent:
            # Both closures see the current accepted state. The new z becomes
            # input on the NEXT step; no future observed latent enters rollout.
            next_latent_state = latent_step(
                params["latent_transition"], state.latent_state,
                features[:self.config.levels], self.dt_hours, self.config,
            )
        return AugmentedState(corrected_state, next_closure_state, next_latent_state)

    def decode(self, state: AugmentedState, forcings: Any) -> Any:
        """Decode corrected prognostics using the unchanged pretrained decoder."""
        return self.base_model.decode(state.model_state, forcings)
