"""Autonomous conditional spatial-latent recurrence with separate memory.

The forecast carries z and spatial memory rather than replaying a fixed past
latent. At each accepted physical timestep, coefficients are conditioned on
``[z, thermal_features, flatten(memory)]`` on the internal nodal grid:

    dz = (g_theta(z, T, m) - z) dt / tau_z + sigma_theta(z, T, m) dW
    dm_r = (z - m_r) dt / tau_r.

The scalar-linear relaxation endpoint has an exact Gaussian transition when
g and sigma are frozen over the step. The nonlinear coupled process is a
frozen-coefficient approximation. Memory uses the CURRENT z held over this
step, not the next observation or next sampled endpoint. The conditional law
is p(z_next | z, T, m), not p(z_next | z) alone. This is supervised stochastic
state-space modeling, with no reward, policy, or reinforcement learning.
"""
from __future__ import annotations

import math
from typing import NamedTuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from ngcm_memory.closure import ClosureConfig


class LatentState(NamedTuple):
    """Member-specific recurrent state; dimensionless z[D,x,y], m[R,D,x,y]."""

    latent: jax.Array
    memory: jax.Array
    key: jax.Array


def init_latent_params(key: jax.Array, config: ClosureConfig) -> dict[str, jax.Array]:
    """Initialize a pointwise conditional network (weights shared over space).

    encoder_w/b map concatenated current z, thermal channels, and memory to a
    tanh hidden state. target_w/b predict a bounded relaxation target;
    diffusion_w/b predict positive bounded diagonal diffusion. Neither these
    heads nor their samples impose cross-column spatial covariance.
    """
    if config.latent_dim < 1:
        raise ValueError("Latent transition requires latent_dim > 0")
    dim, hidden = config.latent_dim, config.latent_hidden_dim
    inputs = dim + config.levels + len(config.latent_memory_tau_hours) * dim
    encoder_key, target_key, diffusion_key = jax.random.split(key, 3)
    return {
        "encoder_w": jax.random.normal(encoder_key, (hidden, inputs)) / math.sqrt(inputs),
        "encoder_b": jnp.zeros((hidden,)),
        "target_w": 0.05 * jax.random.normal(target_key, (dim, hidden)),
        "target_b": jnp.zeros((dim,)),
        "diffusion_w": 0.05 * jax.random.normal(diffusion_key, (dim, hidden)),
        "diffusion_b": jnp.full((dim,), -2.0),
    }


def initialize_latent(grid_shape, config: ClosureConfig, key: jax.Array) -> LatentState:
    """Cold start; observed pre-origin history and origin z can seed it later."""
    if config.latent_dim < 1:
        raise ValueError("Latent state requires latent_dim > 0")
    if len(grid_shape) != 2 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0 for n in grid_shape):
        raise ValueError("grid_shape must be two positive integers (lon, lat)")
    return LatentState(
        jnp.zeros((config.latent_dim, *grid_shape)),
        jnp.zeros((len(config.latent_memory_tau_hours), config.latent_dim, *grid_shape)),
        key,
    )


def _validate(state, thermal_features, dt_hours, config):
    features = jnp.asarray(thermal_features)
    if jnp.issubdtype(features.dtype, jnp.complexfloating):
        raise ValueError("thermal_features must be real-valued")
    features = features.astype(jnp.result_type(features.dtype, jnp.float32))
    if features.ndim != 3 or features.shape[0] != config.levels:
        raise ValueError(f"thermal_features must have shape ({config.levels}, lon, lat)")
    grid = features.shape[-2:]
    if state.latent.shape != (config.latent_dim, *grid):
        raise ValueError("Latent state shape does not match config and thermal grid")
    if state.memory.shape != (len(config.latent_memory_tau_hours), config.latent_dim, *grid):
        raise ValueError("Spatial memory shape does not match config and thermal grid")
    if not isinstance(dt_hours, jax.core.Tracer):
        dt_array = np.asarray(dt_hours)
        if dt_array.ndim != 0 or not np.isfinite(dt_array) or float(dt_array) <= 0:
            raise ValueError("dt_hours must be a finite positive scalar")
    dt = jnp.asarray(dt_hours, features.dtype)
    if dt.ndim != 0:
        raise ValueError("dt_hours must be scalar")
    return features, dt


def transition_moments(params, state: LatentState, thermal_features, dt_hours,
                       config: ClosureConfig) -> tuple[jax.Array, jax.Array]:
    """Conditional endpoint mean and diagonal variance, both [D,lon,lat].

    ``params`` is the ``latent_transition`` subtree of closure parameters.
    expm1 preserves O(dt) variance for tiny dt. Disabled stochastic mode returns
    zero variance, consistent with latent_step's deterministic ablation.
    """
    thermal, dt = _validate(state, thermal_features, dt_hours, config)
    memory = state.memory if config.enable_memory else jnp.zeros_like(state.memory)
    conditioning = jnp.concatenate([
        state.latent, thermal, memory.reshape((-1, *thermal.shape[-2:]))
    ], axis=0)
    hidden = jnp.tanh(jnp.einsum("hd,dxy->hxy", params["encoder_w"], conditioning)
                      + params["encoder_b"][:, None, None])
    target = config.latent_target_bound * jnp.tanh(
        jnp.einsum("dh,hxy->dxy", params["target_w"], hidden) + params["target_b"][:, None, None]
    )
    one_minus_decay = -jnp.expm1(-dt / config.latent_tau_hours)
    mean = state.latent + one_minus_decay * (target - state.latent)
    if config.enable_stochastic:
        logits = jnp.einsum("dh,hxy->dxy", params["diffusion_w"], hidden) + params["diffusion_b"][:, None, None]
        sigma = config.latent_max_diffusion_per_sqrt_hour * (
            1e-6 + (1.0 - 1e-6) * jax.nn.sigmoid(logits)
        )
        variance = sigma**2 * (0.5 * config.latent_tau_hours) * (-jnp.expm1(-2.0 * dt / config.latent_tau_hours))
    else:
        variance = jnp.zeros_like(mean)
    return mean, variance


def latent_step(params, state: LatentState, thermal_features, dt_hours,
                config: ClosureConfig) -> LatentState:
    """Sample the next latent and carry accumulated spatial history onward.

    This function never accepts labels or source observations. Independent
    innovations enter the recurrence once per accepted internal model step.
    """
    mean, variance = transition_moments(params, state, thermal_features, dt_hours, config)
    next_key = state.key
    if config.enable_stochastic:
        next_key, sample_key = jax.random.split(state.key)
        # A zero configured bound is a supported deterministic limiting case.
        # Avoid sqrt(0)'s undefined parameter derivative in this static branch.
        if config.latent_max_diffusion_per_sqrt_hour > 0:
            latent = mean + jnp.sqrt(variance) * jax.random.normal(sample_key, mean.shape, dtype=mean.dtype)
        else:
            latent = mean
    else:
        latent = mean
    if config.enable_memory:
        taus = jnp.asarray(config.latent_memory_tau_hours, mean.dtype)[:, None, None, None]
        weight = -jnp.expm1(-jnp.asarray(dt_hours, mean.dtype) / taus)
        memory = state.memory + weight * (state.latent[None] - state.memory)
    else:
        memory = jnp.zeros_like(state.memory)
    return LatentState(latent, memory, next_key)
