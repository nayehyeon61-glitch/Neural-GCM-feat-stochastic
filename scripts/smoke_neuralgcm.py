#!/usr/bin/env python3
"""Real upstream checkpoint integration test, not a skill benchmark.

The default bundled TL63 checkpoint is a deliberately tiny upstream test
model. Passing this smoke test does not establish forecast skill or stability.
Set JAX_PLATFORMS=cpu to explicitly select a CPU installation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

import jax
import jax.numpy as jnp
import neuralgcm
import numpy as np

from ngcm_memory.closure import ClosureConfig, init_params
from ngcm_memory.neuralgcm_adapter import NeuralGCMMemoryAdapter


def to_numpy(value):
    if hasattr(value, "dtype") and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        value = jax.random.key_data(value)
    return np.asarray(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Optional TRUSTED pickle checkpoint; otherwise use upstream toy fixture",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--recurrent-latent", action="store_true",
                        help="Exercise a two-step recurrent latent forecast with synthetic latent initialization")
    args = parser.parse_args()
    started = time.monotonic()
    if args.checkpoint is None:
        checkpoint = neuralgcm.demo.load_checkpoint_tl63_stochastic()
        source = "upstream toy TL63 fixture (not pretrained skill validation)"
    else:
        # NeuralGCM public checkpoints use pickle; loading untrusted files can
        # execute code. This is an explicit local user-supplied path.
        with args.checkpoint.open("rb") as file:
            checkpoint = pickle.load(file)
        source = str(args.checkpoint)
    base = neuralgcm.PressureLevelModel.from_checkpoint(checkpoint)
    data = neuralgcm.demo.load_data(base.data_coords).isel(time=0)
    inputs = base.inputs_from_xarray(data)
    forcing = base.forcings_from_xarray(data)
    key = jax.random.key(args.seed)
    encode_key, closure_key, params_key = jax.random.split(key, 3)
    print("Encoding actual NeuralGCM fixture...", flush=True)
    base_state = base.encode(inputs, forcing, rng_key=encode_key)
    latent_dim = 2 if args.recurrent_latent else 0
    config = ClosureConfig(
        feature_dim=base.model_coords.vertical.layers + latent_dim,
        levels=base.model_coords.vertical.layers,
        hidden_dim=4,
        tau_hours=(6.0, 24.0),
        enable_stochastic=True,
        latent_dim=latent_dim,
        latent_hidden_dim=4,
    )
    adapter = NeuralGCMMemoryAdapter(base, config)
    disabled = NeuralGCMMemoryAdapter(base, config, enabled=False)
    state = adapter.initialize(base_state, closure_key)
    if latent_dim:
        # Synthetic values establish connectivity, not a trained source latent.
        state = state._replace(latent_state=state.latent_state._replace(
            latent=jnp.full_like(state.latent_state.latent, 0.2),
            memory=jnp.full_like(state.latent_state.memory, 0.1),
        ))
    params = init_params(params_key, config)
    steps = 2 if latent_dim else 1

    def boundary(model_state):
        prognostics = model_state.state if hasattr(model_state, "state") else model_state
        return {**forcing, "sim_time": prognostics.sim_time}

    print("Advancing unchanged base and disabled closure...", flush=True)
    expected, bypass_state = base_state, state
    for _ in range(steps):
        expected = base.advance(expected, boundary(expected))
        bypass_state = disabled.advance(params, bypass_state, boundary(bypass_state.model_state))
    bypass = bypass_state.model_state
    equal = all(
        np.array_equal(to_numpy(a), to_numpy(b))
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(bypass))
    )
    if not equal:
        raise AssertionError("Disabled adapter changed the NeuralGCM forecast")

    print("Advancing active closure and differentiating actual thermal state...", flush=True)

    def loss(parameters):
        next_state = state
        for _ in range(steps):
            next_state = adapter.advance(parameters, next_state, boundary(next_state.model_state))
        temperature = adapter.temperature_variation_kelvin(next_state.model_state)
        # An artificial local target establishes parameter connectivity only.
        target = jax.lax.stop_gradient(
            adapter.temperature_variation_kelvin(expected) + 0.1
        )
        return jnp.mean((temperature - target) ** 2)

    value, grads = jax.jit(jax.value_and_grad(loss))(params)
    gradient_norm = float(jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads))))
    if not np.isfinite(float(value)) or not np.isfinite(gradient_norm) or gradient_norm <= 0:
        raise AssertionError(f"Invalid gradient: loss={value}, norm={gradient_norm}")
    latent_gradient_norm = None
    if latent_dim:
        latent_gradient_norm = float(jnp.sqrt(sum(
            jnp.sum(g * g) for g in jax.tree.leaves(grads["latent_transition"]))))
        if not np.isfinite(latent_gradient_norm) or latent_gradient_norm <= 0:
            raise AssertionError("Future physical-state loss does not reach latent transition")
    active = state
    for _ in range(steps):
        active = adapter.advance(params, active, boundary(active.model_state))
    print("Decoding corrected state...", flush=True)
    decoded = adapter.decode(active, boundary(active.model_state))
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(decoded)):
        raise AssertionError("Nonfinite values in decoded corrected forecast")
    print(json.dumps({
        "checkpoint": source,
        "neuralgcm_version": neuralgcm.__version__,
        "jax_version": jax.__version__,
        "sigma_levels": config.levels,
        "nodal_shape": adapter.grid_shape,
        "timestep_hours": adapter.dt_hours,
        "disabled_bitwise_equal": equal,
        "loss": float(value),
        "gradient_norm": gradient_norm,
        "recurrent_latent": bool(latent_dim),
        "internal_steps": steps,
        "latent_transition_gradient_norm_from_field_loss": latent_gradient_norm,
        "decoded_finite": True,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "skill_validated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
