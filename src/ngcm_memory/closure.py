"""Causal, spatially conditioned memory closure for a NeuralGCM split step.

Time is measured in hours. The output is an *integrated* temperature correction
in kelvin, not a tendency to add at every dynamical-core RHS evaluation.

For frozen input features during one accepted model step, the augmented system
is

    dm_r = (u(features) - m_r) dt / tau_r
    dZ   = -Z dt / tau_noise + sigma(features) dW
    dT   = [A_spatial u + sum_r A_r m_r + Z/tau_noise] dt.

The memory, OU endpoint, and joint OU time integral are advanced exactly. This
is an augmented SDE with a *colored* thermal forcing, not independent white
noise added directly to temperature. Freezing the features and applying this
as an operator split does not make the full coupled NeuralGCM scheme exact.
The timescales are fixed positive configuration values in this first version.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ClosureConfig:
    """Static configuration (pass as a static argument when directly JITing).

    ``max_diffusion_k_sqrt_hour`` bounds sigma in K / sqrt(hour), despite the
    abbreviated name. The OU state Z is in K and its thermal tendency Z/tau is
    in K/hour. ``max_tendency_k_per_hour`` bounds the deterministic tendency,
    assuming the memory was initialized with :func:`initialize_memory`.

    ``enable_spatial`` controls the instantaneous learned spatial term only;
    feature conditioning of memory/diffusion remains active in that ablation.
    ``enable_memory=False`` disables and clears deterministic memory, while
    ``enable_stochastic=False`` disables and clears the OU state.
    """

    feature_dim: int
    levels: int
    hidden_dim: int = 16
    tau_hours: tuple[float, ...] = (6.0, 24.0, 120.0)
    stochastic_tau_hours: float = 24.0
    max_diffusion_k_sqrt_hour: float = 0.1
    max_tendency_k_per_hour: float = 0.2
    enable_memory: bool = True
    enable_stochastic: bool = True
    enable_spatial: bool = True
    # Zero keeps the original checkpoint/feature contract exactly unchanged.
    latent_dim: int = 0
    latent_hidden_dim: int = 16
    latent_tau_hours: float = 24.0
    latent_memory_tau_hours: tuple[float, ...] = (24.0, 120.0, 720.0)
    latent_max_diffusion_per_sqrt_hour: float = 0.1
    latent_target_bound: float = 5.0

    def __post_init__(self) -> None:
        for name in ("feature_dim", "levels", "hidden_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        # Tuples keep this frozen dataclass hashable for jax.jit(static_argnames).
        object.__setattr__(self, "tau_hours", tuple(float(t) for t in self.tau_hours))
        if not self.tau_hours or any(not math.isfinite(t) or t <= 0 for t in self.tau_hours):
            raise ValueError("tau_hours must contain finite positive timescales")
        if not math.isfinite(self.stochastic_tau_hours) or self.stochastic_tau_hours <= 0:
            raise ValueError("stochastic_tau_hours must be finite and positive")
        for name in ("max_diffusion_k_sqrt_hour", "max_tendency_k_per_hour"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if isinstance(self.latent_dim, bool) or not isinstance(self.latent_dim, (int, np.integer)) or self.latent_dim < 0:
            raise ValueError("latent_dim must be a nonnegative integer")
        if isinstance(self.latent_hidden_dim, bool) or not isinstance(self.latent_hidden_dim, (int, np.integer)) or self.latent_hidden_dim <= 0:
            raise ValueError("latent_hidden_dim must be a positive integer")
        if self.latent_dim and self.feature_dim != self.levels + self.latent_dim:
            raise ValueError("Latent mode requires feature_dim == levels + latent_dim")
        object.__setattr__(self, "latent_memory_tau_hours", tuple(float(t) for t in self.latent_memory_tau_hours))
        if not self.latent_memory_tau_hours or any(not math.isfinite(t) or t <= 0 for t in self.latent_memory_tau_hours):
            raise ValueError("latent_memory_tau_hours must contain finite positive timescales")
        for name in ("latent_tau_hours", "latent_target_bound"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.latent_max_diffusion_per_sqrt_hour) or self.latent_max_diffusion_per_sqrt_hour < 0:
            raise ValueError("latent_max_diffusion_per_sqrt_hour must be finite and nonnegative")


class ClosureState(NamedTuple):
    """A JAX pytree; keep a separate state and PRNG key for each ensemble member.

    memory: dimensionless [timescales, hidden_dim, lon, lat].
    ou: auxiliary temperature [levels, lon, lat], in kelvin.
    key: JAX PRNG key, consumed once per accepted stochastic closure step.
    """

    memory: jax.Array
    ou: jax.Array
    key: jax.Array


def init_params(key: jax.Array, config: ClosureConfig) -> dict[str, jax.Array]:
    """Initialize small trainable correction heads and a bounded feature encoder.

    There is intentionally no hidden global RNG or parameter/state mutation.
    Use a common parameter tree but distinct state keys across ensemble members.
    """

    encoder_key, spatial_key, memory_key, diffusion_key = jax.random.split(key, 4)
    params = {
        "encoder_w": jax.random.normal(encoder_key, (config.hidden_dim, config.feature_dim))
        / math.sqrt(config.feature_dim),
        "encoder_b": jnp.zeros((config.hidden_dim,)),
        "spatial_w": 0.05 * jax.random.normal(spatial_key, (config.levels, config.hidden_dim)),
        "memory_w": 0.05
        * jax.random.normal(memory_key, (config.levels, len(config.tau_hours), config.hidden_dim)),
        "diffusion_w": 0.05
        * jax.random.normal(diffusion_key, (config.levels, config.hidden_dim)),
        "diffusion_b": jnp.full((config.levels,), -2.0),
    }
    if config.latent_dim:
        # Fold rather than resplit so all original closure heads retain their
        # exact initial values. Latent parameters have an independent stream.
        from ngcm_memory.latent import init_latent_params
        params["latent_transition"] = init_latent_params(jax.random.fold_in(key, 7319), config)
    return params


def initialize_memory(
    grid_shape: tuple[int, int], config: ClosureConfig, key: jax.Array
) -> ClosureState:
    """Initialize a cold-start memory and OU state with the supplied member key.

    A cold start is deliberately not a stationary OU sample. For hindcasts,
    warm up the state causally on the available historical sequence; never use
    verifying future fields to initialize it.
    """

    if len(grid_shape) != 2 or any(
        isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0 for n in grid_shape
    ):
        raise ValueError("grid_shape must be two positive integers (lon, lat)")
    return ClosureState(
        memory=jnp.zeros((len(config.tau_hours), config.hidden_dim, *grid_shape)),
        ou=jnp.zeros((config.levels, *grid_shape)),
        key=key,
    )


def _ou_joint_unit_moments(dt_hours: jax.Array, tau_hours: float) -> tuple[jax.Array, ...]:
    """Exact unit-diffusion variance/covariance of (OU endpoint, OU integral).

    The series prevents cancellation of the O(dt**3) integral variance when
    dt/tau is small, which otherwise becomes negative in float32 arithmetic.
    """

    x = dt_hours / tau_hours
    one_minus_a = -jnp.expm1(-x)
    one_minus_a2 = -jnp.expm1(-2.0 * x)
    series = x**3 * (
        1.0 / 3.0
        + x
        * (
            -1.0 / 4.0
            + x
            * (7.0 / 60.0 + x * (-1.0 / 24.0 + x * (31.0 / 2520.0 + x * (-1.0 / 320.0))))
        )
    )
    integral_factor = jnp.where(x < 0.1, series, x - 2.0 * one_minus_a + 0.5 * one_minus_a2)
    var_endpoint = 0.5 * tau_hours * one_minus_a2
    cov_endpoint_integral = 0.5 * tau_hours**2 * one_minus_a**2
    var_integral = tau_hours**3 * integral_factor
    return jnp.exp(-x), tau_hours * one_minus_a, var_endpoint, cov_endpoint_integral, var_integral


def closure_step(
    params: dict[str, jax.Array],
    state: ClosureState,
    features: jax.Array,
    dt_hours: float | jax.Array,
    config: ClosureConfig,
) -> tuple[jax.Array, ClosureState]:
    """Advance one accepted step, returning ``(delta_temperature_K, state)``.

    ``features`` is [feature_dim, lon, lat] from the current/past resolved state
    and spatial statistics. The returned correction is [levels, lon, lat].
    Features must already be normalized using training-period statistics.

    Eager calls reject a nonpositive/nonfinite timestep. Under JIT, callers
    passing a traced timestep must enforce the same positive/finite contract
    outside the compiled loop. Configuration and array shapes are static.
    """

    features = jnp.asarray(features)
    if jnp.issubdtype(features.dtype, jnp.complexfloating):
        raise ValueError("features must be real-valued")
    # Integer-valued supplied features must not truncate a fractional timestep.
    features = features.astype(jnp.result_type(features.dtype, jnp.float32))
    if features.ndim != 3 or features.shape[0] != config.feature_dim:
        raise ValueError(f"features must have shape ({config.feature_dim}, lon, lat)")
    grid_shape = features.shape[1:]
    if state.memory.shape != (len(config.tau_hours), config.hidden_dim, *grid_shape):
        raise ValueError("memory state shape does not match config and feature grid")
    if state.ou.shape != (config.levels, *grid_shape):
        raise ValueError("OU state shape does not match config and feature grid")
    if not isinstance(dt_hours, jax.core.Tracer):
        dt_array = np.asarray(dt_hours)
        if dt_array.ndim != 0 or not np.isfinite(dt_array) or float(dt_array) <= 0:
            raise ValueError("dt_hours must be a finite positive scalar")
    dt = jnp.asarray(dt_hours, dtype=features.dtype)
    if dt.ndim != 0:
        raise ValueError("dt_hours must be scalar")
    encoded = jnp.tanh(jnp.einsum("hd,dxy->hxy", params["encoder_w"], features) + params["encoder_b"][:, None, None])
    delta = jnp.zeros((config.levels, *grid_shape), dtype=encoded.dtype)
    # tanh weights and normalized sums bound the deterministic tendency while
    # preserving exact integration of the linear readout of memory states.
    half_bound = 0.5 * config.max_tendency_k_per_hour
    if config.enable_spatial:
        spatial_weights = jnp.tanh(params["spatial_w"]) / config.hidden_dim
        delta = delta + half_bound * dt * jnp.einsum("lh,hxy->lxy", spatial_weights, encoded)

    if config.enable_memory:
        taus = jnp.asarray(config.tau_hours, dtype=encoded.dtype)[:, None, None, None]
        scaled_dt = dt / taus
        one_minus_decay = -jnp.expm1(-scaled_dt)
        integral_weight = taus * one_minus_decay
        # The initially cold memory contributes O(dt**2), so evaluating
        # dt - tau*(1-exp(-dt/tau)) directly loses precision for tiny steps.
        target_integral_factor = jnp.where(
            scaled_dt < 0.01,
            scaled_dt**2 * (0.5 + scaled_dt * (-1.0 / 6.0 + scaled_dt * (1.0 / 24.0 - scaled_dt / 120.0))),
            scaled_dt - one_minus_decay,
        )
        target = encoded[None, ...]
        memory = state.memory + one_minus_decay * (target - state.memory)
        memory_integral = integral_weight * state.memory + taus * target_integral_factor * target
        memory_weights = jnp.tanh(params["memory_w"]) / (len(config.tau_hours) * config.hidden_dim)
        delta = delta + half_bound * jnp.einsum("lmh,mhxy->lxy", memory_weights, memory_integral)
    else:
        memory = jnp.zeros_like(state.memory)

    next_key = state.key
    if config.enable_stochastic:
        logits = jnp.einsum("lh,hxy->lxy", params["diffusion_w"], encoded) + params["diffusion_b"][:, None, None]
        # A small floor keeps the conditioned amplitude strictly positive when
        # enabled with a positive configured bound, including saturated logits.
        sigma = config.max_diffusion_k_sqrt_hour * (1e-6 + (1.0 - 1e-6) * jax.nn.sigmoid(logits))
        decay, integral_weight, v_endpoint, covariance, v_integral = _ou_joint_unit_moments(dt, config.stochastic_tau_hours)
        next_key, endpoint_key, integral_key = jax.random.split(state.key, 3)
        eps_endpoint = jax.random.normal(endpoint_key, state.ou.shape, dtype=encoded.dtype)
        eps_integral = jax.random.normal(integral_key, state.ou.shape, dtype=encoded.dtype)
        std_endpoint = jnp.sqrt(v_endpoint)
        correlated_integral = covariance / std_endpoint
        std_conditional_integral = jnp.sqrt(jnp.maximum(v_integral - correlated_integral**2, 0.0))
        ou = decay * state.ou + sigma * std_endpoint * eps_endpoint
        ou_integral = integral_weight * state.ou + sigma * (
            correlated_integral * eps_endpoint + std_conditional_integral * eps_integral
        )
        delta = delta + ou_integral / config.stochastic_tau_hours
    else:
        ou = jnp.zeros_like(state.ou)

    return delta, ClosureState(memory=memory, ou=ou, key=next_key)
